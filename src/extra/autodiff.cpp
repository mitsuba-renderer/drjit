/** Dr.Jit automatic differentiation library
 *
 * This file implements the AD data structures and traversal routines
 * underlying the Dr.Jit 'DiffArray<T>' type. The compilation process
 * explicitly instantiates these templates for scalar/LLVM/CUDA arrays in both
 * single and double precision and merges them into a shared library
 * "drjit-extra.so/dll". In this way, the machinery below only needs to be
 * compiled once instead of adding a heavy compilation burden to any code using
 * the AD type.
 *
 * Forward and reverse-mode traversal build on three main data structures:
 *
 * - 'state.variable': A hash table mapping from variable IDs (uint32_t) to
 *   'Variable' instances, which mainly stores the gradient associated with
 *   each variable, as well as links into the 'state.edges' list for
 *   connectivity.
 *
 * - 'state.edges': An interlinked array storing edges that provide
 *   connectivity between the variables. Each edge can be simple or special---a
 *   simple edge records an edge weight that is used to scale gradients passing
 *   along it. A special edge implements some more complex gradient
 *   transformation via callbacks. Computation that operates across array
 *   entries (e.g. scatter/gather) requires such special edges.
 *
 * - 'local_state.todo': List of edges that should be traversed by the next
 *   call to 'ad_traverse()'. This list is thread-local in contrast to the
 *   previous two data structures that are shared by all threads.
 *
 * To understand how everything fits together, start by looking at 'ad_new()'
 * and 'ad_traverse()': Arithmetic involving differentiable Dr.Jit arrays
 * triggers various calls to 'ad_new()', which creates the necessary variables
 * and edge connectivity in the above graph data structures. Forward or
 * reverse-mode differentiation with 'ad_traverse()' moves gradients through
 * the desired sub-part of the graph, while executing the derivative
 * transformation encoded along edges.
 *
 * Variables are reference-counted and freed automatically when they go out of
 * scope. The whole system is built to work with essentially no dynamic memory
 * allocation after a warm-up phase.
 *
 * The combination with JITted (CUDA/LLVM) array types is interesting: in this
 * case, AD traversal generates code that can be executed at some later point.
 * While Dr.Jit's AD backend is principally tape-based, this combination then
 * begins to resemble classical AD via code transformation. The JITted modes
 * also exploit their ability to peek into literal constant arrays to optimize
 * generated derivative code.
 */

#include "common.h"
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <mutex>

using drjit::ADScope;

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
inline JitIndex lo(Index i) { return (JitIndex) i; }

/// Return the high (AD) part of a combined variable index
inline ADIndex hi(Index i) { return (ADIndex) (i >> 32); }

/// Combine an AD and JIT index into
inline Index combine(ADIndex ad_index, JitIndex jit_index) {
    return (((Index) ad_index) << 32) | ((Index) jit_index);
}

template <typename... Args> bool is_detached(uint64_t i0, Args... args) {
    if (hi(i0 | (args | ...)) == 0) {
        return true;
    } else {
        return false;
    }
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

    /// Return the pointer to a special edge's callback handler
    Special *special() const {
        ad_assert(is_special());
        return (Special *) (payload & ~2ull);
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
 * may not be set (it's the job of the AD traversal to propagate gradients to
 * variables lacking them). No "primal" information is stored except for the
 * size of the original program variable.
 *
 * Adjacency, i.e. how the variable is connected to other variables in either
 * direction, is represented using linked lists. The 'next_fwd' and 'next_bwd'
 * fields each provide an entry point into such a linked list of edges (see
 * also \ref Edge).
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
     * \brief Should AD operations leaving this scope be postponed (edges
     * will be added to the 'postponed' list)
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
static DRJIT_THREAD LocalState local_state;

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
        memcpy(s, s1, l1); s += l1;
        *s++ = '/';
    }

    memcpy(s, s2, l2); s += l2;
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

static void ad_inc_ref(ADIndex index, Variable *v) noexcept (true) {
    DRJIT_MARK_USED(index);
    ad_log("ad_inc_ref(a%u): %u", index, v->ref_count + 1);
    v->ref_count++;
}

static bool ad_dec_ref(ADIndex index, Variable *v) noexcept (true) {
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

        edge = Edge();

        Variable *v2 = state[source];
        ad_assert(v2->ref_count > 0);

        if (!ad_dec_ref(source, v2)) {
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


template <typename T> void ad_inc_ref_impl(Index index) noexcept(true) {
    jit_var_inc_ref(lo(index));

    ADIndex ad_index = hi(index);
    if (unlikely(ad_index)) {
        std::lock_guard<std::mutex> guard(state.mutex);
        ad_inc_ref(ad_index, state[ad_index]);
    }
}

template <typename T> Index ad_inc_ref_cond_impl(Index index) noexcept(true) {
    JitIndex jit_index = lo(index);
    ADIndex ad_index = hi(index);

    jit_var_inc_ref(jit_index);

    if (unlikely(ad_index)) {
        const std::vector<Scope> &scopes = local_state.scopes;
        if (!scopes.empty())
            scopes.back().maybe_disable(ad_index);

        if (ad_index) {
            std::lock_guard<std::mutex> guard(state.mutex);
            ad_inc_ref(ad_index, state[ad_index]);
        }
    }


    return combine(ad_index, jit_index);
}

template <typename T> void ad_dec_ref_impl(Index index) noexcept(true) {
    jit_var_dec_ref(lo(index));

    ADIndex ad_index = hi(index);
    if (unlikely(ad_index)) {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (unlikely(ad_dec_ref(ad_index, state[ad_index]))) {
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
        ADIndex index_lo = lo(index);

        if (unlikely(index_lo == 0)) // overflow of the low part
            continue;

        Variable v;
        v.ref_count = 1;
        v.size = size;
        v.index_hi = hi(index);
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

template <typename Result, typename... Args>
Index ad_new(const char *label, Result &&result, Args... args) {
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
            active |= scope.maybe_disable(args...);
        }

        if (!active)
            return result.release();
    }

    bool symbolic = jit_flag(JitFlag::Recording);

    // ReleaseOperandHelper helper;
    if (unlikely(rec)) {
        for (size_t i = 0; i < N; ++i) {
            if (args[2 * i] == 0)
                continue;

            ADIndex arg_index = op[2 * i];
            const Variable *arg = state[arg_index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(arg->flags & (uint8_t) VariableFlags::Symbolic)) {
                if (arg->size != 1)
                    ad_raise(
                        "ad_new(): symbolic computation performs an implicit "
                        "read of variable (a%u), which has size %u! However, "
                        "only scalar (size == 1) accesses are permitted in "
                        "this manner. You will likely want to convert the "
                        "read into an drjit.gather() operation.",
                        arg_index, arg->size);

                ad_log("ad_new(): implicit read of variable a%u, inserting a "
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

    switch (N) {
        case 0:
            ad_log("ad_new(a%u, label=\"%s\")", ad_index, label);
            break;
        case 1:
            ad_log("ad_new(a%u <- a%u, label=\"%s\")", ad_index, args[0],
                   label);
            break;
        case 2:
            ad_log("ad_new(a%u <- a%u, a%u, label=\"%s\")", ad_index, args[0],
                   args[2], label);
            break;
        case 3:
            ad_log("ad_new(a%u <- a%u, a%u, a%u, label=\"%s\")", ad_index,
                   args[0], args[2], args[4], label);
            break;
        default:
            break;
    }

    EdgeIndex edge_index = 0;

    for (size_t i = 0; i < N; ++i) {
        ADIndex  source = args[2*i];
        JitIndex weight = args[2*i + 1];

        if (!source) {
            jit_var_dec_ref(weight);
            continue;
        }

        if (jit_var_is_literal_zero(weight)) {
            ad_log("ad_new(a%u <- a%u): weight of edge %i is zero, skipping!",
                   ad_index, source, i);
            jit_var_dec_ref(weight);
            continue;
        }

        Variable *v_source = state[source];

        EdgeIndex edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = source;
        edge.target = ad_index;
        edge.set_weight(weight);
        edge.next_fwd = v_source->next_fwd;
        edge.next_bwd = edge_index;
        edge_index = edge_index_new;

        ad_inc_ref(source, v_source);
        v_source->next_fwd = edge_index_new;
    }

    if (N && !edge_index) {
        // All edges were pruned, don't create the node after all
        ad_log("ad_new(a%u): all edges pruned, removing variable.", ad_index);
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

    return combine(ad_index, jit_index);
}

#define OP_2(name, op)                                                         \
    template <typename S> Index ad_var_##name(Index i0, Index i1) {            \
        using T = GenericArray<S>;                                             \
        uint32_t l0 = lo(i0), l1 = lo(i1),                                     \
                 result_i = jit_var_##name(l0, l1);                            \
        if (is_detached(i0, i1))                                               \
            return result_i;                                                   \
        T result = T::steal(result_i);                                         \
        jit_set_default_backend_from(l0);                                      \
        T v0 = T::borrow(i0), v1 = T::borrow(i1);                              \
        return ad_new(#name, std::move(result), )


OP_2(add) {
    return ad_new("add", o0 + o1, Edge(i0, 1), Edge(i1, 1));
}

template <typename S> Index ad_var_sin(Index i0) {
    using T = GenericArray<S>;

    if (is_detached(i0))
        return jit_var_sin<S>(lo(i0));

    auto [s, c] = dr::sincos(T::borrow(i0));
    return ad_new("sin", std::move(s), i0, std::move(c));
}

template <typename S> Index ad_var_cos(Index i0) {
    using T = GenericArray<S>;

    if (is_detached(i0))
        return jit_var_cos<S>(lo(i0));

    auto [s, c] = dr::sincos(T::borrow(i0));
    return ad_new("cos", std::move(c), i0, std::move(s));
}
