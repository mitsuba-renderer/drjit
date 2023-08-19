/** Dr.Jit automatic differentiation library
 *
 * This file implements the data structures and traversal routines underlying
 * the Dr.Jit ``DiffArray<T>`` type. THis file is part of a separately compiled
 * library to reduce compilation overheads in large projects that make
 * extensive use of autodiff. Consequently, most methods of ``DiffArray<T>``
 * merely wrap functions implemented here.
 *
 * Forward and reverse-mode traversal build on three main data structures:
 *
 * - ``state.variable``: A hash table to map from variable IDs (uint32_t) to
 *   ``Variable`` instances. They store the gradient associated with each
 *   variable and link into the ``state.edges`` list to specify the variable's
 *   connectivity.
 *
 * - ``state.edges``: An interlinked array storing edges, which encode the
 *   connectivity between variables. Each edge can be simple or special. A
 *   simple edge records an edge weight that scales gradients flowing along it.
 *   Special edges implement more complex gradient transformation via
 *   callbacks. Operations that exchange across array entries (e.g.,
 *   scatter/gather) require such special edges.
 *
 * - ``local_state.todo``: A list of edges that should be traversed by the next
 *   call to ``ad_traverse()``. This list is thread-local in contrast to the
 *   previous two data structures that are shared by all threads.
 *
 * To understand how everything fits together, start by looking at an arithmetic
 * operation like ``ad_var_add()``, which triggers ``ad_var_new()`` to allocate
 * a new variable. Next, look at and ``ad_traverse()``, which traverses the AD
 * graph in either the forward or reverse direction to propagate gradients.
 *
 * Variables are reference-counted and freed automatically when they go out of
 * scope. The whole system is built to work with essentially no dynamic memory
 * allocation after a warm-up phase.
 *
 * The implementation on top of drjit-core (a JIT compiler) is noteworthy:
 * derivative propagation performs arithmetic operations that appear
 * instantanous but are actually postponed for evaluation at a later point.
 * While Dr.Jit's AD backend tape-based, the combination with deferred
 * evaluation resembles AD via code transformation. The generated code
 * leverages optimizations such as constant propagation and local value
 * numbering and can handle function calls without unrolling.
 */

#include "common.h"
#include <drjit/jit.h>
#include <drjit/math.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <mutex>

namespace dr = drjit;
using dr::ADScope;

// ==========================================================================
// Aliases for various indices to clarify their use in the code below
// ==========================================================================

/// Index of an edge between AD variables in the 'state.edges' list
using EdgeIndex     = uint32_t;

/// Index of an AD variable in the 'state.variables' hash table
using ADIndex       = uint32_t;

/// Index of a JIT compiler variable in the drjit-core library
using JitIndex      = uint32_t;

/// Combined index that simultaneously stores a JIT variable index (low part)
/// and an AD variable index (high part)
using Index         = uint64_t;

/// Return the low (JIT) part of a combined variable index
inline JitIndex jit_index(Index i) { return (JitIndex) i; }

/// Return the high (AD) part of a combined variable index
inline ADIndex ad_index(Index i) { return (ADIndex) (i >> 32); }

/// Combine an AD and JIT index into
inline Index combine(ADIndex ad_index, JitIndex jit_index) {
    return (((Index) ad_index) << 32) | ((Index) jit_index);
}

template <typename... Args> bool is_detached(Args... args) {
    return ad_index((args | ...)) == 0;
}

// ==========================================================================
// Central data structures: edges, variables, global state
// ==========================================================================

struct Special;

/**
 * Represents an edge in the AD graph that futhermore stores either
 *
 * 1. An edge weight that scales gradients passing along this edge
 *
 * 2. An unspecified instance of the 'Special' interface that implements
 *    some more advanced way of transforming gradients between source/target
 *    variable. Masking and scatter/gather operations, e.g., require this.
 *
 * Instead of storing an explicit adjacency list of the AD graph structure, the
 * adjacency information is directly encoded in the edges. In particular, the
 * 'next_fwd' and 'next_bwd' indices each implement a singly linked list that
 * can be used to iterate through the forward edges (of the 'source' variable)
 * and backward edges (of the 'target' variable).
 */
struct Edge {
    /// Variable index of source operand
    ADIndex source;

    /// Source variable index
    ADIndex target;

    /// Link to the next forward edge
    EdgeIndex next_fwd;

    /// Link to the next backward edge
    EdgeIndex next_bwd;

    /// Payload area storing:
    ///
    /// Bit 0: Is this a special or an ordinary edge?
    /// Bit 1: Was this edge previously visited as part of an AD traversal?
    ///
    /// If Bit 0 is set:
    ///    - bit 2..63   Pointer to an instance of the 'Special' interface
    ///    - bit 32..63  JIT variable index referencing the edge weight
    uint64_t payload;

    bool is_special() const { return (payload & 1ull) == 0ull; }
    bool is_normal() const  { return (payload & 1ull) == 1ull; }
    bool is_visited() const { return (payload & 2ull) == 2ull; }

    // Override the edge's 'visited' bit
    void set_visited(bool value) {
        payload = (payload & ~2ull) | (uint64_t(value) << 1);
    }

    /// Inititialize an edge with a special callback handler
    void set_special(Special *special) {
        ad_assert(payload == 0);
        payload = (uint64_t) special;
    }

    /// Initialize a normal edge with a given weight value
    void set_weight(JitIndex index) {
        ad_assert(payload == 0);
        payload = (((uint64_t) index) << 32) | 1ull;
    }

    /// Return the JIT variable index referencing the weight of an ordinary edge
    JitIndex weight() const {
        ad_assert(is_normal());
        return (JitIndex) (payload >> 32);
    }

    /// Return the pointer to a special edge's callback handler
    Special *special() const {
        ad_assert(is_special());
        return (Special *) (payload & ~2ull);
    }
};

static_assert(sizeof(Edge) == 24);

/// Flags characterizing the 'Variable.flags' bit field
enum VariableFlags : uint8_t {
    /// Was this AD node created while capturing symbolic computation in the
    /// JIT compiler? (e.g. a symbolic loop, virtual function call, etc.)
    Symbolic = 1 << 0,

    /// Was the label manually overwritten via drjit.set_label()?
    CustomLabel = 1 << 1,

    /// Should the label be freed when the variable is deallocated?
    FreeLabel = 1 << 2
};

/**
 * Reference-counted data structure representing an AD variable
 *
 * The data structure associates a gradient with each variable, which may or
 * may not be set. It's the job of the AD traversal to propagate gradients to
 * variables lacking them. No "primal" information (i.e. details about a
 * variable's content during the original program execution) is stored except
 * for each variable's size.
 *
 * Adjacency, i.e., how the variable is connected to other variables in either
 * the forward or backward direction, is represented using linked lists. The
 * 'next_fwd' and 'next_bwd' fields each provide an entry point into such a
 * linked list of edges (see also \ref Edge).
 */
struct Variable {
    /// Number of references to this AD variable
    uint32_t ref_count = 0;

    /// Link to the first forward edge at this node
    EdgeIndex next_fwd = 0;

    /// Link to the first backward edge at this node
    EdgeIndex next_bwd = 0;

    /// JIT variable index referencing the gradient
    JitIndex grad = 0;

    /// Size of the associated primal variable
    size_t size = 0;

    /// Descriptive label
    char *label = nullptr;

    /// High bits of variable index
    uint32_t index_hi = 0;

    /// Gradient reference count for custom operations
    uint16_t ref_count_grad = 0;

    /// Custom flags (see the 'VariableFlag' enum above)
    uint8_t flags = 0;

    /// Floating point type (half/single/double)
    uint8_t type = 0;
};

static_assert(sizeof(Variable) ==
              6 * sizeof(uint32_t) + sizeof(size_t) + sizeof(char *));

/// Represents the global state of the AD system
struct State {
    using VariableMap = tsl::robin_map<ADIndex, Variable, UInt32Hasher,
                                       std::equal_to<ADIndex>>;
    using EdgeVector  = std::vector<Edge>;

    /// std::mutex protecting the state data structure
    std::mutex mutex;

    /// Hash table mapping variable IDs to variable instances
    VariableMap variables;

    /// List of all edges (used and unused ones)
    EdgeVector edges;

    /// List of currently unused edges
    std::vector<EdgeIndex> unused_edges;

    /// Counter for variable indices
    uint64_t variable_index = 1;

    State() : edges(1) { }

    ~State() {
        if (!variables.empty()) {
            ad_warn("Variable leak detected (%zu variables remain in use)!",
                    variables.size());
            size_t counter = 0;
            for (auto kv : variables) {
                ad_warn(" - variable a%u (%u references)", kv.first, kv.second.ref_count);
                if (++counter == 10) {
                    ad_warn(" - (skipping the rest)");
                    break;
                }
            }
        }

        size_t edges_used = edges.size() - unused_edges.size() - 1;
        if (edges_used != 0)
            ad_warn("Edge leak detected (%zu edges remain in use)!",
                    edges_used);
    }

    Variable *operator[](ADIndex index) {
        VariableMap::iterator it = variables.find(index);
        if (unlikely(index == 0 || it == variables.end()))
            ad_fail("Referenced an unknown variable a%u!", index);
        return &it.value();
    }
};

// Special edge (scatter, gather, scatter_reduce, block_sum, etc.)
struct Special {
    virtual void backward(Variable * /* source */,
                          const Variable * /* target */,
                          uint32_t /* flags */) const {
        ad_fail("Special::backward(): not implemented!");
    }

    virtual void forward(const Variable * /* source */,
                         Variable * /* target */,
                         uint32_t /* flags */) const {
        ad_fail("Special::forward(): not implemented!");
    }

    virtual ~Special() = default;
};

/// Encodes details about an edge to be traversed
struct EdgeRef {
    EdgeIndex id;
    ADIndex source;
    ADIndex target;

    EdgeRef() : id(0), source(0), target(0) { }
    EdgeRef(EdgeIndex id, ADIndex source, ADIndex target)
    : id(id), source(source), target(target) { }
};

/**
 * This data structure encodes an AD scope that can be used to selectively
 * enable/disable derivative propagation for certain variables
 */
struct Scope {
    /// How should this scope be interpreted?
    ADScope type = ADScope::Invalid;

    /**
     * \brief Controls the interpretation of the 'indices' field
     *
     * If <tt>complement=false</tt> gradients are enabled for variables that
     * members of the \c indices set.
     *
     * If <tt>complement=true</tt> gradients are enabled for variables that
     * are <b>not</b> members of the 'indices' set.
     */
    bool complement = true;

    /**
     * \brief When the AD traversal encounters operations leaving this scope,
     * should their traversal be postponed? In that case, edges will be added
     * to the 'postponed' list.
     */
    bool isolate = false;

    // Current variable index when entering this scope
    uint64_t variable_index = 0;

    /**
     * \brief Depending on the value of 'complement', this set specifies
     * variables for which AD is enabled or disabled.
     */
    tsl::robin_set<ADIndex, UInt32Hasher> indices;

    /// List of AD postponed edges that will be traversed when leaving the scope
    std::vector<EdgeRef> postponed;

    Scope() = default;
    Scope(Scope&&) = default;
    Scope(const Scope&) = default;
    Scope& operator=(Scope&&) = default;
    Scope& operator=(const Scope&) = default;

    /// Check if a variable has gradients enabled
    bool enabled(ADIndex index) const {
        return (indices.find(index) != indices.end()) != complement;
    }

    /// Potentially zero out 'index' if the variable has gradients disabled
    bool maybe_disable(ADIndex &index) const {
        if (index && !enabled(index))
            index = 0;
        return index != 0;
    }

    /// Track gradients for the given variable
    void enable(ADIndex index) {
        if (!index)
            return;

        if (complement)
            indices.erase(index);
        else
            indices.insert(index);
    }

    /// Disable gradients for the given variable
    void disable(ADIndex index) {
        if (!index)
            return;

        if (complement)
            indices.insert(index);
        else
            indices.erase(index);
    }
};

// Stores per-thread state
struct LocalState {
    /// Thread-local edge list used by ad_enqueue_*() and ad_traverse()
    std::vector<EdgeRef> todo;

    /// Keeps track of implicit input dependencies of symbolic computation
    std::vector<EdgeRef> implicit;

    /// Nested scopes that restrict AD to specific variables
    std::vector<Scope> scopes;

    /// List of edges that should be cleaned up
    std::vector<Special *> cleanup;

    ~LocalState() {
        for (Special *s : cleanup)
            delete s;

        if (!scopes.empty())
            ad_warn("Scope leak detected (%zu scopes remain in use)!",
                    scopes.size());
    }
};

static State state;
static thread_local LocalState local_state;

// ==========================================================================
// Variable prefix (mainly useful for GraphViz visualizations)
// ==========================================================================

struct PrefixEntry {
    PrefixEntry *prev;
    char *value;
};

static DRJIT_THREAD PrefixEntry *prefix = nullptr;

/// Concatenate two strings with a '/' separator. 's1' may be null.
char *concat_str(const char *s1, const char *s2) {
    size_t l1 = s1 ? strlen(s1) : 0,
           l2 = strlen(s2);

    char *buf = (char *) malloc(l1 + l2 + 2);
    if (!buf)
        ad_fail("concat(): memory allocation failed!");

    char *s = buf;
    if (s1) {
        memcpy(s, s1, l1);
        s += l1;
        *s++ = '/';
    }

    memcpy(s, s2, l2);
    s += l2;
    *s = '\0';

    return buf;
}

DRJIT_EXPORT void ad_prefix_push(const char *value) {
    if (strchr(value, '/'))
        ad_raise("ad_prefix_push(): may not contain a '/' character.");

    PrefixEntry *&p = prefix;
    p = new PrefixEntry{ p, concat_str(p ? p->value : nullptr, value ) };
}

DRJIT_EXPORT void ad_prefix_pop() {
    PrefixEntry *p = prefix;
    if (p) {
        prefix = p->prev;
        free(p->value);
        delete p;
    }
}

// ==========================================================================
// Reference counting and variable cleanup
// ==========================================================================
//
static void ad_free(ADIndex, Variable *);

static void ad_inc_ref_int(ADIndex index, Variable *v) noexcept {
    DRJIT_MARK_USED(index);
    ad_log("ad_inc_ref(a%u): %u", index, v->ref_count + 1);
    v->ref_count++;
}

static bool ad_dec_ref_int(ADIndex index, Variable *v) noexcept {
    DRJIT_MARK_USED(index);
    ad_log("ad_dec_ref(a%u): %u", index, v->ref_count - 1);
    ad_assert(v->ref_count > 0);

    if (--v->ref_count > 0) {
        return false;
    } else {
        ad_free(index, v);
        return true;
    }
}

static void ad_free(ADIndex index, Variable *v) {
    ad_log("ad_free(a%u)", index);

    if (v->flags & (uint8_t) VariableFlags::FreeLabel) {
        free(v->label);
        v->label = nullptr;
    }

    jit_var_dec_ref(v->grad);
    v->grad = 0;

    EdgeIndex edge_id = v->next_bwd;
    v->next_bwd = 0;

    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        ad_log("ad_free(): freeing edge a%u -> a%u", edge.source,
               edge.target);

        ad_assert(edge.target == index);

        ADIndex source = edge.source;
        EdgeIndex next_bwd = edge.next_bwd,
                  next_fwd = edge.next_fwd;

        // Postpone deallocation of the edge callback, if there is one
        if (unlikely(edge.is_special()))
            local_state.cleanup.push_back(edge.special());

        edge = Edge { };

        Variable *v2 = state[source];
        ad_assert(v2->ref_count > 0);

        if (!ad_dec_ref_int(source, v2)) {
            EdgeIndex fwd = v2->next_fwd;
            if (fwd == edge_id) {
                v2->next_fwd = next_fwd;
            } else {
                while (true) {
                    Edge &edge2 = state.edges[fwd];
                    ad_assert(edge2.source == source);
                    if (edge2.next_fwd != edge_id) {
                        fwd = edge2.next_fwd;
                    } else {
                        edge2.next_fwd = next_fwd;
                        break;
                    }
                }
            }
        }

        state.unused_edges.push_back(edge_id);

        edge_id = next_bwd;
    }

    state.variables.erase(index);
}

template <typename T> Index ad_inc_ref_impl(Index index) JIT_NOEXCEPT {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_var_inc_ref(jit_index);

    if (unlikely(ad_index)) {
        const std::vector<Scope> &scopes = local_state.scopes;
        if (!scopes.empty())
            scopes.back().maybe_disable(ad_index);

        if (ad_index) {
            std::lock_guard<std::mutex> guard(state.mutex);
            ad_inc_ref_int(ad_index, state[ad_index]);
        }
    }

    return combine(ad_index, jit_index);
}

template <typename T> void ad_dec_ref_impl(Index index) JIT_NOEXCEPT {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_var_dec_ref(jit_index);

    if (unlikely(ad_index)) {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (unlikely(ad_dec_ref_int(ad_index, state[ad_index]))) {
            /* Extra-careful here: deallocate cleanup queue of
               custom AD edge callbacks (reentrant!) */
            std::vector<Special *> temp, &cleanup = local_state.cleanup;

            if (!cleanup.empty()) {
                temp.swap(cleanup);
                for (Special *special : temp)
                    delete special;
            }
        }
    }
}

// ==========================================================================
// Variable and edge creation
// ==========================================================================

/// Allocate a new variable and initialize basic fields
static std::pair<ADIndex, Variable *> ad_var_new(const char *label, size_t size,
                                                 VarType type, bool symbolic) {
    while (true) {
        Index index = state.variable_index++;
        ADIndex index_lo = (uint32_t) index;

        if (unlikely(index_lo == 0)) // overflow of the low part
            continue;

        Variable v;
        v.ref_count = 1;
        v.size = size;
        v.index_hi = (uint32_t) (index >> 32);
        v.flags = symbolic ? (uint8_t) VariableFlags::Symbolic : (uint8_t) 0;
        v.type = (uint8_t) type;

        PrefixEntry *p = prefix;
        if (p) {
            label = concat_str(p->value, label);
            v.flags |= (uint8_t) VariableFlags::FreeLabel;
        }

        v.label = (char *) label;

        auto result = state.variables.emplace(index_lo, v);
        if (likely(result.second))
            return { index, &result.first.value() };
    }
}

/// Allocate a new edge from the pool
static EdgeIndex ad_edge_new() {
    EdgeIndex index;
    if (likely(!state.unused_edges.empty())) {
        index = state.unused_edges.back();
        state.unused_edges.pop_back();
    } else {
        index = (EdgeIndex) state.edges.size();
        state.edges.emplace_back();
    }
    return index;
}

/// Ensure consistent size of symbolic variables to avoid horiz. reductions
static void ad_propagate_size(Variable *v) {
    EdgeIndex edge = v->next_bwd;
    while (edge) {
        Edge &e = state.edges[edge];
        Variable *v2 = state[e.source];
        if ((v2->flags & (uint8_t) VariableFlags::Symbolic) &&
            v2->size != v->size && v2->size == 1) {
            v2->size = v->size;
            ad_propagate_size(v2);
        }
        edge = e.next_bwd;
    }
}

/**
 * This step declares a generic Jit type with an unknown backend and type.
 * Despite the lack of information, Dr.Jit arithmetic below is functional
 * since the underlying drjit-core implementation infers the backend+type
 * from the existing operands.
 */
using JitVar = GenericArray<void>;
using JitMask = GenericArray<bool>;

DRJIT_NOINLINE JitVar scalar(Index index, double value) {
    VarInfo info = jit_set_backend(jit_index(index));

    return JitVar::steal(info.type == VarType::Float32
                             ? jit_var_f32(info.backend, (float) value)
                             : jit_var_f64(info.backend, value));
}

// This data structure encodes an ordianry dependence on a function argument
struct Arg {
    Arg(Index index, JitVar &&weight)
        : ad_index(::ad_index(index)), weight(std::move(weight)) { }

    Arg(Index index, double value)
        : ad_index(::ad_index(index)), weight(scalar(index, value)) { }

    Arg(Arg &&a) : ad_index(a.ad_index), weight(std::move(a.weight)) { }

    Arg(const Arg &a) = delete;
    Arg &operator=(const Arg &a) = delete;
    Arg &operator=(Arg &&a) = delete;
    void release() { weight.release(); }

    ADIndex ad_index;
    JitVar weight;
};

// This data structure encodes a special dependence on a function argument
struct SpecialArg {
    SpecialArg(Index index, Special *special)
        : ad_index(::ad_index(index)), special(special) { }

    SpecialArg(SpecialArg &&a) : ad_index(a.ad_index), special(a.special) {
        a.special = nullptr;
    }

    ~SpecialArg() {
        delete special;
    }

    void release() { special = nullptr; }

    SpecialArg(const SpecialArg &a) = delete;
    SpecialArg &operator=(const SpecialArg &a) = delete;
    SpecialArg &operator=(SpecialArg &&a) = delete;

    ADIndex ad_index;
    Special *special;
};

template <typename T, typename ...> struct first {
    using type = T;
};

template <typename... Ts> using first_t = typename first<Ts...>::type;

template <typename... Args>
DRJIT_NOINLINE Index ad_var_new_impl(const char *label, JitVar &&result,
                                     Args &&...args_) {
    using ArgType = first_t<Args...>;
    ArgType args[] { std::move(args_)... };

    std::lock_guard<std::mutex> guard(state.mutex);

    /* Potentially turn off derivative tracking for some of the operands if
       we're within a scope that enables/disables gradient propagation
       (globally, or only for specific variables) */
    std::vector<Scope> &scopes = local_state.scopes;
    if (unlikely(!scopes.empty())) {
        const Scope &scope = scopes.back();

        bool active = false;
        if constexpr (sizeof...(Args) == 0) {
            // If AD is completely disabled (i.e. this is an dr.suspend_grad()
            // region), don't allow creating new AD variables
            active = scope.complement || !scope.indices.empty();
        } else {
            for (size_t i = 0; i < sizeof...(Args); ++i)
                active |= scope.maybe_disable(args[i].ad_index);
        }

        if (!active)
            return (Index) result.release();
    }

    bool symbolic = jit_flag(JitFlag::Recording);

    // ReleaseOperandHelper helper;
    if (unlikely(symbolic)) {
        for (size_t i = 0; i < sizeof...(Args); ++i) {
            if (args[i].ad_index == 0)
                continue;

            ADIndex arg_index = args[i].ad_index;
            const Variable *arg = state[arg_index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(arg->flags & (uint8_t) VariableFlags::Symbolic)) {
                if (arg->size != 1)
                    ad_raise(
                        "ad_var_new(): symbolic computation performs an implicit "
                        "read of variable (a%u), which has size %zu! However, "
                        "only scalar (size == 1) accesses are permitted in "
                        "this manner. You will likely want to convert the "
                        "read into an drjit.gather() operation.",
                        arg_index, arg->size);

                ad_log("ad_var_new(): implicit read of variable a%u, inserting a "
                       "gather operation..", arg_index);

                // index = ad_new_gather_impl<Value>("gather", size, arg_index, Index(0),
                //                                   Mask(true), false);
                //
                // op[2 * i] = index;
                // helper.put(index);
            }
        }
    }

    auto [ad_index, var] = ad_var_new(label, jit_var_size(result.index()),
                                      jit_var_type(result.index()), symbolic);

    switch (sizeof...(Args)) {
        case 0:
            ad_log("ad_var_new(a%u, label=\"%s\")", ad_index, label);
            break;
        case 1:
            ad_log("ad_var_new(a%u <- a%u, label=\"%s\")", ad_index, args[0].ad_index,
                   label);
            break;
        case 2:
            ad_log("ad_var_new(a%u <- a%u, a%u, label=\"%s\")", ad_index, args[0].ad_index,
                   args[1].ad_index, label);
            break;
        case 3:
            ad_log("ad_var_new(a%u <- a%u, a%u, a%u, label=\"%s\")", ad_index,
                   args[0].ad_index, args[1].ad_index, args[2].ad_index, label);
            break;
        default:
            break;
    }

    EdgeIndex edge_index = 0;

    for (size_t i = 0; i < sizeof...(Args); ++i) {
        ADIndex source = args[i].ad_index;

        if (!source)
            continue;

        if constexpr (!std::is_same_v<ArgType, SpecialArg>) {
            if (jit_var_is_literal_zero(args[i].weight.index())) {
                ad_log("ad_var_new(a%u <- a%u): weight of edge %zu is zero, skipping!",
                       ad_index, source, i);
                continue;
            }
        }

        Variable *v_source = state[source];

        EdgeIndex edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = source;
        edge.target = ad_index;

        if constexpr (std::is_same_v<ArgType, SpecialArg>) {
            edge.set_special(args[i].special);
            args[i].special = nullptr;
        } else {
            edge.set_weight(args[i].weight.release());
        }

        edge.next_fwd = v_source->next_fwd;
        edge.next_bwd = edge_index;
        edge_index = edge_index_new;

        ad_inc_ref_int(source, v_source);
        v_source->next_fwd = edge_index_new;
    }

    if (sizeof...(Args) > 0 && !edge_index) {
        // All edges were pruned, don't create the node after all
        ad_log("ad_var_new(a%u): all edges pruned, removing variable.", ad_index);
        ad_free(ad_index, var);
        return 0;
    }

    var->next_bwd = edge_index;

    if (var->flags & (uint8_t) VariableFlags::Symbolic)
        ad_propagate_size(var);

    /* If we're selectively tracking gradients and this operation generates a
       new AD variable, then its index must be added to the index set */
    if (unlikely(!scopes.empty()))
        scopes.back().enable(ad_index);

    return combine(ad_index, result.release());
}

template <typename... Args>
DRJIT_INLINE Index ad_var_new(const char *label, JitVar &&result,
                              Args &&...args) {
    Index rv = ad_var_new_impl(label, (JitVar &&) result, (Args &&) args...);
    // When ad_var_new_impl returns, the inputs have been move-constructed
    // away. Make this explicit to avoid superfluous reference counting at the
    // call site.
    result.release();
    (args.release(), ...);
    return rv;
}

// ==========================================================================
// AD traversal callbacks for special operations: masks, gathers, scatters
// ==========================================================================

struct MaskEdge : Special {
    MaskEdge(const JitMask &mask, bool negate) : mask(mask), negate(negate) { }

    void backward(Variable *source, const Variable *target, uint32_t) const override {
        (void) source;
        (void) target;
        // if (!negate)
        //     JitVar::borrow(target->grad) & mask;
        // source->accum(!negate ? detail::and_(target->grad, mask)
        //                       : detail::andnot_(target->grad, mask),
        //               target->size);
    }

    void forward(const Variable *source, Variable *target, uint32_t) const override {
        (void) source;
        (void) target;
        // target->accum(!negate ? detail::and_(source->grad, mask)
        //                       : detail::andnot_(source->grad, mask),
        //               source->size);
    }

    JitMask mask;
    bool negate;
};

// ==========================================================================
// Implementation of the JIT backend arithmetic operations
// ==========================================================================

Index ad_var_neg(Index i0) {
    JitVar result = JitVar::steal(jit_var_neg(jit_index(i0)));

    if (is_detached(i0))
        return result.release();
    else
        return ad_var_new("neg", std::move(result), Arg(i0, -1.0));
}

Index ad_var_abs(Index i0) {
    JitVar result = JitVar::steal(jit_var_abs(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               vz = scalar(i0, 0.0),
               vp = scalar(i0, 1.0),
               vn = scalar(i0, -1.0);

        return ad_var_new("abs", std::move(result),
                          Arg(i0, dr::select(v0 >= vz, vp, vn)));
    }
}

Index ad_var_sqrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_sqrt(jit_index(i0)));

    if (is_detached(i0))
        return result.release();
    else
        return ad_var_new("sqrt", std::move(result),
                          Arg(i0, dr::rcp(result) * scalar(i0, .5f)));
}

Index ad_var_rcp(Index i0) {
    JitVar result = JitVar::steal(jit_var_rcp(jit_index(i0)));

    if (is_detached(i0))
        return result.release();
    else
        return ad_var_new("rcp", std::move(result),
                          Arg(i0, -dr::sqr(result)));
}

Index ad_var_rsqrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_rsqrt(jit_index(i0)));

    if (is_detached(i0))
        return result.release();
    else
        return ad_var_new("rsqrt", std::move(result),
                          Arg(i0, -dr::sqr(result) * result * scalar(i0, -.5)));
}

Index ad_var_cbrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_cbrt(jit_index(i0)));

    if (is_detached(i0))
        return result.release();
    else
        return ad_var_new(
            "cbrt", std::move(result),
            Arg(i0, dr::sqr(dr::rcp(result)) * scalar(i0, 1.0 / 3.f)));
}

Index ad_var_erf(Index i0) {
    JitVar result = JitVar::steal(jit_var_erf(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        return ad_var_new("erf", std::move(result),
                          Arg(i0, scalar(i0, 2.0 * dr::InvSqrtPi<double>) *
                                      dr::exp(-dr::sqr(v0))));
    }
}

Index ad_var_add(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_add(jit_index(i0), jit_index(i1)));

    if (likely(is_detached(i0, i1)))
        return result.release();
    else
        return ad_var_new("add", std::move(result), Arg(i0, 1.0), Arg(i1, 1.0));
}

Index ad_var_sub(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_sub(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1))
        return result.release();
    else
        return ad_var_new("sub", std::move(result), Arg(i0, 1.0), Arg(i1, -1.0));
}

Index ad_var_mul(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_mul(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1))
        return result.release();
    else
        return ad_var_new("mul", std::move(result),
                          Arg(i0, JitVar::borrow(jit_index(i1))),
                          Arg(i1, JitVar::borrow(jit_index(i0))));
}

Index ad_var_div(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_div(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               v1 = JitVar::borrow(jit_index(i1)),
               w0 = dr::rcp(v1),
               w1 = -v0 * dr::sqr(w0);

        return ad_var_new("div", std::move(result),
                          Arg(i0, std::move(w0)),
                          Arg(i1, std::move(w1)));
    }
}

Index ad_var_min(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_min(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               v1 = JitVar::borrow(jit_index(i1)),
               zero = scalar(i0, 0.0),
               one  = scalar(i0, 1.0);

        JitMask mask = v0 <= v1;

        return ad_var_new("min", std::move(result),
                          Arg(i0, dr::select(mask, one, zero)),
                          Arg(i1, dr::select(mask, zero, one)));
    }
}

Index ad_var_max(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_max(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               v1 = JitVar::borrow(jit_index(i1)),
               zero = scalar(i0, 0.0),
               one  = scalar(i0, 1.0);

        JitMask mask = v0 > v1;

        return ad_var_new("max", std::move(result),
                          Arg(i0, dr::select(mask, one, zero)),
                          Arg(i1, dr::select(mask, zero, one)));
    }
}

Index ad_var_select(Index i0, Index i1, Index i2) {
    JitVar result = JitVar::steal(
        jit_var_select(jit_index(i0), jit_index(i1), jit_index(i2)));

    if (is_detached(i1, i2)) {
        return result.release();
    } else if (jit_var_is_literal(i0) || i1 == i2) {
        ad_log("ad_new_select(a%u <- a%u, a%u): simplified",
               ad_index(i0), ad_index(i1), ad_index(i2));

        Index ad_index = jit_var_is_literal_zero(i0) ? i2 : i1;
        std::lock_guard<std::mutex> guard(state.mutex);
        ad_inc_ref_int(ad_index, state[ad_index]);
        return ad_index;
    } else {
        JitMask m = JitMask::borrow(i0);
        return ad_var_new("select", std::move(result),
                          SpecialArg(i0, new MaskEdge(m, false)),
                          SpecialArg(i1, new MaskEdge(m, true)));
    }
}

Index ad_var_fma(Index i0, Index i1, Index i2) {
    JitVar result = JitVar::steal(
        jit_var_fma(jit_index(i0), jit_index(i1), jit_index(i2)));

    if (is_detached(i0, i1, i2))
        return result.release();
    else
        return ad_var_new("fma", std::move(result),
                          Arg(i0, JitVar::borrow(jit_index(i1))),
                          Arg(i1, JitVar::borrow(jit_index(i0))),
                          Arg(i2, 1.0));
}

