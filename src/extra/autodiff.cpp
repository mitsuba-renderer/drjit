/** Dr.Jit automatic differentiation library
 *
 * This file implements the data structures and traversal routines underlying
 * the Dr.Jit ``drjit::DiffArray<T>`` type. This file is part of the separately
 * compiled ``drjit-extra`` library to reduce compilation overheads in large
 * projects that make extensive use of autodiff. Consequently, most methods of
 * ``drjit::DiffArray<T>`` merely wrap functions implemented here.
 *
 * Forward and reverse-mode traversal build on three main data structures:
 *
 * - ``state.variable``: A hash table to map from variable IDs (uint32_t) to
 *   ``Variable`` instances. It stores the gradient associated with each
 *   variable and links into the ``state.edges`` list to specify the variable's
 *   connectivity.
 *
 * - ``state.edges``: An interlinked array storing edges, which encode the
 *   connectivity between variables. Each edge can be simple or special. A
 *   simple edge records an edge weight that scales gradients flowing along
 *   it. Special edges implement more complex gradient transformation via
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
 * instantanous but whose evaluation is actually symbolic and deferred until
 * the system eventually compiles and runs a fused kernel. While Dr.Jit's AD
 * backend tape-based, the combination with deferred evaluation resembles AD
 * via code transformation. The generated code leverages optimizations such as
 * constant propagation and local value numbering and can support some types of
 * control flow without the usual tape-style unrolling.
 */

#include "common.h"
#include <drjit/jit.h>
#include <drjit/math.h>
#include <drjit/autodiff.h>
#include <tsl/robin_set.h>
#include <queue>
#include <mutex>
#include <memory>

namespace dr = drjit;

/// Reuse the Buffer class from nanobind
#define NB_NAMESPACE drjit
#define NB_NOINLINE DRJIT_NOINLINE
#include "../ext/nanobind/src/buffer.h"
using drjit::detail::Buffer;

using dr::ADScope;

// ==========================================================================
// Aliases for various indices to clarify their use in the code below
// ==========================================================================

/// Index of an AD edge within ``state.edges``
using EdgeIndex     = uint32_t;

/// Index of an AD variable within ``state.variables``
using ADIndex       = uint32_t;

/// Index of a Jit variable managed by drjit-core
using JitIndex      = uint32_t;

/// Combined index that simultaneously stores a Jit index (low part)
/// and an AD index (high part)
using Index         = uint64_t;

/// Return the low (Jit) part of a combined variable index
inline JitIndex jit_index(Index i) { return (JitIndex) i; }

/// Return the high (AD) part of a combined variable index
inline ADIndex ad_index(Index i) { return (ADIndex) (i >> 32); }

/// Merge an AD and JIT index into a combined index
inline Index combine(ADIndex ad_index, JitIndex jit_index) {
    return (((Index) ad_index) << 32) | ((Index) jit_index);
}

template <typename... Args> bool is_detached(Args... args) {
    return ad_index((args | ...)) == 0;
}

// ==========================================================================
// Generic Jit type for arithmetic
// ==========================================================================

/**
 * Declare a generic Jit type with an unknown backend and type. Despite the
 * complete lack of information, Dr.Jit arithmetic in this file is functional
 * since Dr.Jit can generally infer the backend+type from the arguments of
 * arithmetic operations.
 *
 * One exception is the creation of scalars: an operation like ``JitVar(1)`` is
 * simply too abiguous. Instead, use the ``scalar()`` function below.
 */
using JitVar = GenericArray<void>;

/// Associated mask & offset type
using JitMask = GenericArray<bool>;

/// Create a scalar Jit variable with the same floating point type and backend
/// as an already existing variable with the provided ``index``
DRJIT_NOINLINE JitVar scalar(Index index, double value) {
    VarInfo info = jit_set_backend(jit_index(index));

    return JitVar::steal(info.type == VarType::Float32
                             ? jit_var_f32(info.backend, (float) value)
                             : jit_var_f64(info.backend, value));
}

// ==========================================================================
// Central data structures: edges, variables, global state
// ==========================================================================
//
struct Special;

/**
 * Represents an edge in the AD graph that furthermore stores either
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
    ADIndex source = 0;

    /// Source variable index
    ADIndex target = 0;

    /// Link to the next forward edge
    EdgeIndex next_fwd = 0;

    /// Link to the next backward edge
    EdgeIndex next_bwd = 0;

    /// Special edge handler
    std::unique_ptr<Special> special;

    /// Edge weight
    JitVar weight;

    /// Visited flag for DFS traversal
    bool visited = false;
};

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
    JitVar grad;

    /// Size of the associated primal variable
    size_t size = 0;

    /// Descriptive label
    char *label = nullptr;

    /// Value of the ``state.counter`` field when this variable was created
    uint64_t counter = 0;

    /// Gradient reference count for custom operations
    uint16_t ref_count_grad = 0;

    /// Custom flags (see the 'VariableFlag' enum above)
    uint8_t flags = 0;

    /// Floating point type (half/single/double)
    uint8_t type = 0;

    Variable() = default;

    Variable(const Variable &) = delete;
    Variable &operator=(const Variable &) = delete;

    Variable(Variable &&v) noexcept
        : ref_count(v.ref_count), next_fwd(v.next_fwd), next_bwd(v.next_bwd),
          grad(std::move(v.grad)), size(v.size), label(v.label),
          counter(v.counter), ref_count_grad(v.ref_count_grad), flags(v.flags),
          type(v.type) {
        v.label = nullptr;
    }

    Variable &operator=(Variable &&v) noexcept {
        ref_count = v.ref_count; next_fwd = v.next_fwd;
        next_bwd = v.next_bwd; grad = std::move(v.grad);
        size = v.size;
        if (flags & (uint8_t) VariableFlags::FreeLabel)
            free(label);
        label = v.label; v.label = nullptr;
        counter = v.counter; ref_count_grad = v.ref_count_grad;
        flags = v.flags; type = v.type;
        return *this;
    }

    ~Variable() {
        if (flags & (uint8_t) VariableFlags::FreeLabel)
            free(label);
    }

    /**
     * \brief Multiply-accumulate a gradient (i.e., ``grad += v1*v2``), where
     * ``v2`` is typically the weight of an AD edge.
     *
     * The arithmetic is slightly unusual in the sense that ``mul_accum(0,
     * v2)`` leaves ``grad`` unchanged even if the edge weight ``v2`` is
     * infinite or NaN. This is important so that propagating zero gradients
     * through invalid/masked AD subgraphs does not contaminate the final
     * gradient.
     *
     * This is operation is heavily used during AD traversal, hence the
     * implementation considers a few different cases and optimizations.
     */
    void mul_accum(const JitVar &v1, const JitVar &v2, size_t src_size) {
        JitVar zero = scalar(v1.index(), 0.f), weight;

        // Elide the zero check if ``v2`` is known not to be NaN/infinite
        if (jit_var_is_normal_literal(v2.index()))
            weight = v2;
        else
            weight = dr::select(dr::eq(v1, zero), zero, v2);

        if (size == 1 && src_size != 1) {
            /* When this variable is scalar (size == 1) and the source is
               not (src_size != 1), the gradient must be reduced to a single
               value. A special case arises when the source gradient is
               actually scalar after all, in which case it is considered to
               broadcast to all elements. */

            JitVar v3 = v1 * weight;
            if (v3.size() == 1) {
                v3 *= scalar(v1.index(), (double) src_size);
            } else {
                ad_assert(v3.size() == src_size);
                v3 = dr::sum(v3);
            }

            if (grad.valid())
                grad += v3;
            else
                grad = std::move(v3);
        } else {
            if (grad.valid())
                grad = dr::fmadd(v1, weight, grad);
            else
                grad = v1 * weight;
        }
    }

    /**
     * \brief Accumulate a gradient 'v' originating from another variable of
     * size 'src_size' into the current variable.
     *
     * This is a relatively important operation that is heavily used during AD
     * traversal, hence the implementation considers a few different cases and
     * optimizations.
     */
    void accum(const JitVar& v, size_t src_size) {
        if (size == 1 && src_size != 1) {
            /* When this variable is scalar (size == 1) and the source is
               not (src_size != 1), the gradient must be reduced to a single
               value. A special case arises when the source gradient is
               actually scalar after all, in which case it is considered to
               broadcast to all elements. */

            JitVar v2;
            if (v.size() == 1) {
                v2 = v * scalar(v.index(), (double) src_size);
            } else {
                ad_assert(v.size() == src_size);
                v2 = dr::sum(v);
            }

            if (grad.valid())
                grad += v2;
            else
                grad = std::move(v2);
        } else {
            if (grad.valid())
                grad += v;
            else
                grad = v;
        }
    }
};

/// Represents the global state of the AD system
struct State {
    /// std::mutex protecting the state data structure
    std::mutex mutex;

    /// Hash table mapping variable IDs to variable instances
    std::vector<Variable> variables;

    /// List of all edges (used and unused ones)
    std::vector<Edge> edges;

    /// List of currently unused edges
    std::priority_queue<ADIndex, std::vector<ADIndex>, std::greater<uint32_t>> unused_variables;
    std::priority_queue<EdgeIndex, std::vector<EdgeIndex>, std::greater<uint32_t>> unused_edges;

    /// Counter to establish an ordering among variables
    uint64_t counter = 0;

    State() : variables(1), edges(1) { }

    ~State() {
        size_t vars_used  = variables.size() - unused_variables.size() - 1,
               edges_used = edges.size() - unused_edges.size() - 1;

        if (vars_used) {
            ad_warn("Variable leak detected (%zu variables remain in use)!",
                    vars_used);
            size_t counter = 0;

            for (size_t i = 0; i < variables.size(); ++i) {
                if (variables[i].ref_count == 0)
                    continue;

                ad_warn(" - variable a%zu (%u references)", i, variables[i].ref_count);
                if (++counter == 10) {
                    ad_warn(" - (skipping the rest)");
                    break;
                }
            }
        }

        if (edges_used != 0)
            ad_warn("Edge leak detected (%zu edges remain in use)!",
                    edges_used);
    }

    Variable *operator[](ADIndex index) {
        if (unlikely(index > variables.size() || variables[index].ref_count == 0))
            ad_fail("Referenced an unknown variable a%u!", index);
        return &variables[index];
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
    uint64_t source_counter;
    uint64_t target_counter;

    EdgeRef(EdgeIndex id, ADIndex source, ADIndex target,
            uint64_t source_counter, uint64_t target_counter)
        : id(id), source(source), target(target),
          source_counter(source_counter), target_counter(target_counter) { }
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

    // Current ``state.counter`` value when entering this scope
    uint64_t counter = 0;

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
    std::vector<std::unique_ptr<Special>> cleanup;

    ~LocalState() {
        if (!scopes.empty())
            ad_warn("Scope leak detected (%zu scopes remain in use)!",
                    scopes.size());
    }
};

static State state;
static thread_local LocalState local_state;

// Forward declarations
static void ad_free(ADIndex, Variable *);
static void ad_var_inc_ref_int(ADIndex index, Variable *v) noexcept;


// ==========================================================================
// Reference counting and variable cleanup
// ==========================================================================

static void ad_var_inc_ref_int(ADIndex index, Variable *v) noexcept {
    DRJIT_MARK_USED(index);
    ad_log("ad_var_inc_ref(a%u): %u", index, v->ref_count + 1);
    v->ref_count++;
}

static bool ad_var_dec_ref_int(ADIndex index, Variable *v) noexcept {
    DRJIT_MARK_USED(index);
    ad_log("ad_var_dec_ref(a%u): %u", index, v->ref_count - 1);
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
        if (edge.special)
            local_state.cleanup.emplace_back(std::move(edge.special));

        edge = Edge { };

        Variable *v2 = state[source];
        ad_assert(v2->ref_count > 0);

        if (!ad_var_dec_ref_int(source, v2)) {
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

        state.unused_edges.push(edge_id);

        edge_id = next_bwd;
    }

    *v = Variable { };
    state.unused_variables.push(index);
}

Index ad_var_inc_ref_impl(Index index) JIT_NOEXCEPT {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_var_inc_ref(jit_index);

    if (unlikely(ad_index)) {
        const std::vector<Scope> &scopes = local_state.scopes;
        if (!scopes.empty())
            scopes.back().maybe_disable(ad_index);

        if (ad_index) {
            std::lock_guard<std::mutex> guard(state.mutex);
            ad_var_inc_ref_int(ad_index, state[ad_index]);
        }
    }

    return combine(ad_index, jit_index);
}

static void clear_special() {
    std::vector<std::unique_ptr<Special>> &cleanup = local_state.cleanup;
    if (cleanup.empty())
        return;

    std::vector<std::unique_ptr<Special>> temp;
    unlock_guard<std::mutex> guard(state.mutex);
    temp.swap(cleanup);
    temp.clear();
    temp.swap(cleanup);
}

void ad_var_dec_ref_impl(Index index) JIT_NOEXCEPT {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_var_dec_ref(jit_index);

    if (unlikely(ad_index)) {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (ad_var_dec_ref_int(ad_index, state[ad_index]))
            clear_special();
    }
}

// ==========================================================================
// Variable and edge creation
// ==========================================================================

/// Concatenate two strings
char *concat(const char *s1, const char *s2) {
    size_t l1 = strlen(s1),
           l2 = strlen(s1);

    char *buf = (char *) malloc(l1 + l2 + 1);
    if (!buf)
        ad_fail("concat(): memory allocation failed!");
    memcpy(buf, s1, l1);
    memcpy(buf + l1, s2, l2);
    buf[l1 + l2] = '\0';
    return buf;
}

/// Allocate a new variable from the pool
static std::pair<ADIndex, Variable *> ad_var_new(JitBackend backend,
                                                 size_t size, VarType type,
                                                 bool symbolic,
                                                 const char *label) {

    auto &unused = state.unused_variables;
    ADIndex index;

    if (unlikely(unused.empty())) {
        index = state.variables.size();
        state.variables.emplace_back();
    } else {
        index = unused.top();
        unused.pop();
    }

    Variable *v = &state.variables[index];
    v->ref_count = 1;
    v->size = size;
    v->counter = state.counter++;
    v->flags = symbolic ? (uint8_t) VariableFlags::Symbolic : (uint8_t) 0;
    v->type = (uint8_t) type;

    const char *prefix = jit_prefix(backend);
    if (prefix) {
        v->label = concat(prefix, label);
        v->flags |= (uint8_t) VariableFlags::FreeLabel;
    } else {
        v->label = (char *) label;
    }

    return { index, v };
}

/// Allocate a new edge from the pool
static EdgeIndex ad_edge_new() {
    auto &unused = state.unused_edges;
    EdgeIndex index;

    if (unlikely(unused.empty())) {
        index = state.edges.size();
        state.edges.emplace_back();
    } else {
        index = unused.top();
        unused.pop();
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

// This data structure encodes an ordinary dependence on a function argument
struct Arg {
    Arg(Index index, JitVar &&weight)
        : ad_index(::ad_index(index)), weight(std::move(weight)) { }

    Arg(Index index, double value)
        : ad_index(::ad_index(index)), weight(scalar(index, value)) { }

    Arg(Arg &&a) = default;
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
    SpecialArg(SpecialArg &&a) = default;
    SpecialArg(const SpecialArg &a) = delete;
    SpecialArg &operator=(const SpecialArg &a) = delete;
    SpecialArg &operator=(SpecialArg &&a) = delete;

    void release() { special.release(); }

    ADIndex ad_index;
    std::unique_ptr<Special> special;
};

struct Arg;
template <typename...> struct first {
    using type = Arg;
};

template <typename T, typename... Ts> struct first<T, Ts...> {
    using type = T;
};

template <typename... Ts> using first_t = typename first<Ts...>::type;

template <typename... Args>
DRJIT_NOINLINE Index ad_var_new_impl(const char *label, JitVar &&result,
                                     Args &&...args_) {
    constexpr size_t N = sizeof...(Args);
    using ArgType = first_t<Args...>;
    ArgType args[N] { std::move(args_)... };

    std::lock_guard<std::mutex> guard(state.mutex);

    /* Potentially turn off derivative tracking for some of the operands if
       we're within a scope that enables/disables gradient propagation
       (globally, or only for specific variables) */
    std::vector<Scope> &scopes = local_state.scopes;
    if (unlikely(!scopes.empty())) {
        const Scope &scope = scopes.back();

        bool active = false;
        if constexpr (N == 0) {
            // If AD is completely disabled (i.e. this is an dr.suspend_grad()
            // region), don't allow creating new AD variables
            active = scope.complement || !scope.indices.empty();
        } else {
            for (size_t i = 0; i < N; ++i)
                active |= scope.maybe_disable(args[i].ad_index);
        }

        if (!active)
            return (Index) result.release();
    }

    bool symbolic = jit_flag(JitFlag::Recording);

#if 0
    // ReleaseOperandHelper helper;
    if (unlikely(symbolic)) {
        for (size_t i = 0; i < N; ++i) {
            if (args[i].ad_index == 0)
                continue;

            ADIndex arg_index = args[i].ad_index;
            const Variable *arg = state[arg_index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(!(arg->flags & (uint8_t) VariableFlags::Symbolic))) {
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
                //

                // TODO: *if* this is a gather operation, execute the following
                /* Encountered a dependency between recorded/non-recorded computation
                   that will need special handling when the AD graph is traversed at
                   some later point. For now, just keep track of this event. */
                ad_log("ad_var_gather(): recording an implicit dependency (a%u "
                       "<- a%u).", i1, i0);
                local_state.implicit.emplace_back(v1->next_bwd, i0, i1,
                                                  v0->counter, v1->counter);

        return result_index;
            }
        }
    }
#endif

    VarInfo info = jit_set_backend(result.index());
    auto [ad_index, var] =
        ad_var_new(info.backend, info.size, info.type, symbolic, label);

    if constexpr (N == 0) {
        ad_log("ad_var_new(a%u, label=\"%s\")", ad_index, label);
    } else if constexpr (N == 1) {
        ad_log("ad_var_new(a%u <- a%u, label=\"%s\")", ad_index,
               args[0].ad_index, label);
    } else if constexpr (N == 2) {
        ad_log("ad_var_new(a%u <- a%u, a%u, label=\"%s\")", ad_index,
               args[0].ad_index, args[1].ad_index, label);
    } else if constexpr (N == 3) {
        ad_log("ad_var_new(a%u <- a%u, a%u, a%u, label=\"%s\")", ad_index,
               args[0].ad_index, args[1].ad_index, args[2].ad_index, label);
    }

    EdgeIndex edge_index = 0;

    for (size_t i = 0; i < N; ++i) {
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

        if constexpr (std::is_same_v<ArgType, SpecialArg>)
            edge.special = std::move(args[i].special);
        else
            edge.weight = std::move(args[i].weight);

        edge.next_fwd = v_source->next_fwd;
        edge.next_bwd = edge_index;
        edge_index = edge_index_new;

        ad_var_inc_ref_int(source, v_source);
        v_source->next_fwd = edge_index_new;
    }

    if (N > 0 && !edge_index) {
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

JitIndex ad_grad(Index index) {
    ADIndex ad_index = ::ad_index(index);
    const std::vector<Scope> &scopes = local_state.scopes;
    if (unlikely(!scopes.empty()))
        scopes.back().maybe_disable(ad_index);

    JitVar result;
    size_t size = 1;

    if (ad_index) {
        std::lock_guard<std::mutex> guard(state.mutex);
        const Variable *v = state[ad_index];
        result = v->grad;
        size = v->size;
    }

    if (!result.valid())
        result = scalar(index, 0.0);

    if (result.size() != size)
        result.resize(size);

    return result.release();
}

void ad_clear_grad(Index index) {
    ADIndex ad_index = ::ad_index(index);
    if (ad_index == 0)
        return;
    ad_log("ad_clear_grad(a%u)", ad_index);

    std::lock_guard<std::mutex> guard(state.mutex);
    Variable *v = state[ad_index];
    v->grad = JitVar();
}

void ad_accum_grad(Index index, JitIndex value) {
    if (!value)
        return;

    const std::vector<Scope> &scopes = local_state.scopes;
    ADIndex ad_index = ::ad_index(index);

    if (unlikely(!scopes.empty()))
        scopes.back().maybe_disable(ad_index);

    if (unlikely(ad_index == 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    Variable *v = state[ad_index];

    JitVar value_v = JitVar::borrow(value);
    size_t size_in = value_v.size();

    if (v->size != size_in && size_in != 1 && size_in != 0 && v->size != 1)
        ad_raise("ad_set_grad(): attempted to store a gradient of size "
                 "%zu into AD variable a%u, which has size %zu!",
                 size_in, ad_index, v->size);

    ad_log("ad_accum_grad(a%u)", ad_index);

    v->accum(value_v, size_in);
}

Index ad_var_set_label(Index index, const char *label) {
    uint32_t jit_index = ::jit_index(index),
             ad_index = ::ad_index(index);

    jit_index = jit_var_set_label(jit_index, label);

    if (ad_index) {
        std::lock_guard<std::mutex> guard(state.mutex);
        ad_log("ad_var_set_label(a%u, \"%s\")", ad_index,
               label ? label : "(null)");
        Variable *v = state[ad_index];

        if (v->flags & (uint8_t) VariableFlags::FreeLabel)
            free(v->label);

        VarInfo info = jit_set_backend(jit_index);
        const char *prefix = jit_prefix(info.backend);
        if (!prefix || !label) {
            v->label = label ? strdup(label) : nullptr;
        } else {
            v->label = concat(prefix, label);
        }

        const uint8_t flags = (uint8_t) VariableFlags::FreeLabel |
                              (uint8_t) VariableFlags::CustomLabel;

        if (label)
            v->flags |= flags;
        else
            v->flags &= ~flags;

        ad_var_inc_ref_int(ad_index, v);
    }

    return combine(ad_index, jit_index);
}


// ==========================================================================
// Enqueuing of variables and edges
// ==========================================================================

/// Forward-mode DFS starting from 'index'
static void ad_dfs_fwd(std::vector<EdgeRef> &todo, uint32_t index,
                       Variable *v) {
    DRJIT_MARK_USED(index);

    uint32_t edge_id = v->next_fwd;
    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        if (!edge.visited) {
            edge.visited = true;

            ad_log("ad_dfs_fwd(): enqueuing edge a%u -> a%u", index,
                   edge.target);

            Variable *v2 = state[edge.target];
            ad_var_inc_ref_int(edge.target, v2);
            todo.emplace_back(edge_id, edge.source, edge.target, v->counter, v2->counter);

            ad_dfs_fwd(todo, edge.target, v2);
        }

        edge_id = edge.next_fwd;
    }
}

/// Reverse-mode DFS starting from 'index'
static void ad_dfs_bwd(std::vector<EdgeRef> &todo, uint32_t index,
                       Variable *v) {
    uint32_t edge_id = v->next_bwd;
    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        if (!edge.visited) {
            edge.visited = true;

            ad_log("ad_dfs_bwd(): enqueuing edge a%u -> a%u", index,
                   edge.source);

            Variable *v2 = state[edge.source];
            ad_var_inc_ref_int(index, v);
            todo.emplace_back(edge_id, edge.source, edge.target, v2->counter, v->counter);
            ad_dfs_bwd(todo, edge.source, v2);
        }

        edge_id = edge.next_bwd;
    }
}

void ad_enqueue(dr::ADMode mode, Index index) {
    uint32_t ad_index = ::ad_index(index);
    if (ad_index == 0)
        return;

    ad_log("ad_enqueue_node(a%u, mode=%s)", ad_index,
           mode == dr::ADMode::Forward ? "forward" : "backward");

    LocalState &ls = local_state;

    std::lock_guard<std::mutex> guard(state.mutex);
    switch (mode) {
        case dr::ADMode::Forward:
            ad_dfs_fwd(ls.todo, ad_index, state[ad_index]);
            break;

        case dr::ADMode::Backward:
            ad_dfs_bwd(ls.todo, ad_index, state[ad_index]);
            break;

        default:
            ad_raise("ad_enqueue(): invalid mode specified!");
    }
}

// ==========================================================================
// AD graph traversal
// ==========================================================================

void ad_traverse(dr::ADMode mode, uint32_t flags) {
    LocalState &ls = local_state;

    std::vector<EdgeRef> &todo_tls = ls.todo, todo;
    if (todo_tls.empty())
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    todo_tls.swap(todo);

    if (mode != dr::ADMode::Forward && mode != dr::ADMode::Backward)
        ad_raise("ad_traverse(): invalid mode specified!");

    // Bring the edges into the appropriate order
    std::sort(todo.begin(), todo.end(),
              [mode](const EdgeRef &a, const EdgeRef &b) {
                  if (mode == dr::ADMode::Forward)
                      return std::tie(a.source_counter, a.target_counter) <
                             std::tie(b.source_counter, b.target_counter);
                  else
                      return std::tie(a.target_counter, a.source_counter) >
                             std::tie(b.target_counter, b.source_counter);
              });

    ad_log("ad_traverse(): processing %zu edges in %s mode ..", todo.size(),
           mode == dr::ADMode::Forward ? "forward" : "backward");

    /// Any edges with an ID less than this value will be postponed
    uint64_t postpone_before = 0;
    if (!ls.scopes.empty() && ls.scopes.back().isolate)
        postpone_before = ls.scopes.back().counter;

    std::vector<JitVar> dr_loop_todo;
    auto postprocess = [&](uint32_t prev_i, uint32_t cur_i) {
        if (!prev_i || prev_i == cur_i)
            return;

        Variable *cur  = cur_i ? state[cur_i] : nullptr,
                 *prev = state[prev_i];

        /* Wave-front style evaluation of dr.Loop with differentiable
           variables produces nodes with label 'dr_loop' at the boundary of
           each iteration. It's good if we dr::schedule() and then finally
           evaluate the gradient of all such variables at once so that AD
           tarversal produces reasonably sized kernels (i.e. with an evaluation
           granularity matching the loop iterations of the original/primal
           evaluation). The code below does just that. */

        bool dr_loop_prev = prev->label && strstr(prev->label, "dr_loop"),
             dr_loop_cur  = cur && cur->label && strstr(cur->label, "dr_loop");

        if (dr_loop_prev) {
            dr_loop_todo.push_back(prev->grad);
            dr::schedule(prev->grad);

            if (!dr_loop_cur) {
                ad_log("ad_traverse(): evaluating %zi loop variables",
                       dr_loop_todo.size());
                dr::eval();
                dr_loop_todo.clear();
            }
        }

        bool clear_grad = false;
        uint32_t next_edge =
            mode == dr::ADMode::Forward ? prev->next_bwd : prev->next_fwd;

        if (flags & (uint32_t) dr::ADFlag::ClearInterior)
            clear_grad |= next_edge != 0;
        if (flags & (uint32_t) dr::ADFlag::ClearInput)
            clear_grad |= next_edge == 0;

        /* Don't clear the gradient of vertices created *before* entering
           an dr.isolation() scope, or when their gradient is still explicitly
           referenced by some other part of the computation graph */
        if (prev_i < postpone_before || prev->ref_count_grad > 0)
            clear_grad = false;

        // Aggressively clear gradients at intermediate nodes
        if (clear_grad) {
            ad_log("ad_traverse(): clearing gradient at intermediate variable a%u", prev_i);
            prev->grad = JitVar();
        }
    };

    uint32_t v0i_prev = 0;
    uint32_t last_edge_id = 0;

    // This is the main AD traversal loop
    for (EdgeRef &er : todo) {
        Edge &edge = state.edges[er.id];

        uint32_t v0i, v1i;
        if (mode == dr::ADMode::Forward) {
            v0i = edge.source;
            v1i = edge.target;
        } else {
            v0i = edge.target;
            v1i = edge.source;
        }

        if (unlikely(er.id == last_edge_id))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u was "
                    "enqueued twice!", edge.source, edge.target);
        last_edge_id = er.id;

        if (unlikely(edge.source != er.source ||
                     edge.target != er.target))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u was "
                    "garbage collected between enqueuing and traversal steps!",
                    er.source, er.target);

        if (unlikely(!edge.visited))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u is not "
                    "marked as visited! (1)", er.source,
                    er.target);

        Variable *v0 = state[v0i],
                 *v1 = state[v1i];

        size_t grad_size = v0->grad.size();

        if (unlikely(v0i < postpone_before)) {
            if (mode == dr::ADMode::Backward) {
                ad_log("ad_traverse(): postponing edge a%u -> a%u due "
                       "dr.isolate_grad() scope.", v0i, v1i);
                ls.scopes.back().postponed.push_back(er);
                er.id = er.source = er.target = 0;
                continue;
            } else if (v1i < postpone_before) {
                ad_raise(
                    "ad_traverse(): tried to forward-propagate derivatives "
                    "across edge a%u -> a%u, which lies outside of the current "
                    "dr.isolate_grad() scope (a%llu .. a%llu). You must "
                    "enqueue the variables being differentiated and call "
                    "dr.traverse(dr.ADFlag.ClearEdges) *before* entering this "
                    "scope.", v0i, v1i, (unsigned long long) postpone_before,
                    (unsigned long long) state.counter);
            }
        }

        if (unlikely(grad_size != 1 && v0->size != grad_size)) {
            if (grad_size == 0) {
                ad_log("ad_traverse(): skipping edge a%u -> a%u (no source "
                       "gradient).", v0i, v1i);
                continue;
            } else {
                ad_raise("ad_traverse(): gradient propagation encountered "
                         "variable a%u (\"%s\") with an invalid gradient size "
                         "(expected size %zu, actual size %zu)!",
                         v0i, v0->label ? v0->label : "", v0->size, grad_size);
            }
        }

        postprocess(v0i_prev, v0i);
        v0i_prev = v0i;

        ad_log("ad_traverse(): processing edge a%u -> a%u ..", v0i, v1i);

        if (unlikely(v0->flags & (uint8_t) VariableFlags::CustomLabel)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v0->label);
            if (width(v0->grad) != 0)
                dr::set_label(v0->grad, tmp);
        }

        if (unlikely(edge.special)) {
            if (mode == dr::ADMode::Forward)
                edge.special->forward(v0, v1, flags);
            else
                edge.special->backward(v1, v0, flags);

            if (flags & (uint32_t) dr::ADFlag::ClearEdges) {
                // Edge may have been invalidated by callback, look up once more
                Edge &edge2 = state.edges[er.id];
                if (edge2.source == (uint32_t) er.source &&
                    edge2.target == (uint32_t) er.target)
                    ls.cleanup.push_back(std::move(edge2.special));
            }
        } else {
            v1->mul_accum(v0->grad, edge.weight, v0->size);

            if (flags & (uint32_t) dr::ADFlag::ClearEdges)
                edge.weight = JitVar();
        }
    }

    postprocess(v0i_prev, 0);

    ad_log("ad_traverse(): decreasing reference counts %s..",
           (flags & (uint32_t) dr::ADFlag::ClearEdges)
               ? "and removing traversed edges from graph "
               : "");

    // Undo reference count increases performed by ad_enqueue()
    for (EdgeRef &er : todo) {
        if (!er.target)
            continue;

        Edge &edge = state.edges[er.id];
        if (unlikely(edge.source != (uint32_t) er.source ||
                     edge.target != (uint32_t) er.target))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u was "
                    "garbage collected between enqueue and traverse steps!",
                    er.source, er.target);
        else if (unlikely(!edge.visited))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u is not "
                    "marked as visited!", er.source, er.target);

        edge.visited = 0;

        Variable *source = state[er.source],
                 *target = state[er.target];

        if (flags & (uint32_t) dr::ADFlag::ClearEdges) {
            ad_log("ad_traverse(): removing edge a%u -> a%u", er.source,
                   er.target);

            // Clear out forward edge
            uint32_t edge_id_prev = 0,
                     edge_id_cur = source->next_fwd;
            while (edge_id_cur) {
                Edge &e2 = state.edges[edge_id_cur];
                ad_log("ad_traverse(): visiting forward edge a%u -> a%u",
                       e2.source, e2.target);

                if (edge_id_cur == er.id) {
                    if (edge_id_prev)
                        state.edges[edge_id_prev].next_fwd = e2.next_fwd;
                    else
                        source->next_fwd = e2.next_fwd;
                    break;
                } else if (unlikely(e2.source != er.source)) {
                    ad_fail("ad_traverse(): invalid forward edge connectivity!");
                }

                edge_id_prev = edge_id_cur;
                edge_id_cur = e2.next_fwd;
            }

            if (unlikely(!edge_id_cur))
                ad_fail("ad_traverse(): could not find forward edge a%u "
                        "-> a%u", er.source, er.target);

            // Clear out backward edge
            edge_id_prev = 0;
            edge_id_cur = target->next_bwd;
            while (edge_id_cur) {
                Edge &e2 = state.edges[edge_id_cur];

                if (edge_id_cur == er.id) {
                    if (edge_id_prev)
                        state.edges[edge_id_prev].next_bwd = e2.next_bwd;
                    else
                        target->next_bwd = e2.next_bwd;
                    break;
                } else if (unlikely(e2.target != er.target)) {
                    ad_fail("ad_traverse(): invalid backward edge connectivity!");
                }

                edge_id_prev = edge_id_cur;
                edge_id_cur = e2.next_bwd;
            }

            if (unlikely(!edge_id_cur))
                ad_fail("ad_traverse(): could not find backward edge a%u "
                        "-> a%u", er.source, er.target);

            // Postpone deallocation of the edge callback, if there is one
            if (unlikely(edge.special))
                ls.cleanup.push_back(std::move(edge.special));

            edge = Edge { };
            state.unused_edges.push(er.id);

            ad_var_dec_ref_int(er.source, source);
            target = state[er.target]; // pointer might have changed
        }

        ad_var_dec_ref_int(er.target, target);
    }

    ad_log("ad_traverse(): done.");

    todo.clear();
    todo_tls.swap(todo);
    clear_special();
}

// ==========================================================================
// AD traversal callbacks for special operations: masks, gathers, scatters
// ==========================================================================

struct MaskEdge : Special {
    MaskEdge(const JitMask &mask, bool negate)
        : mask(mask), negate(negate) { }

    void backward(Variable *source, const Variable *target,
                  uint32_t) const override {
        source->accum(!negate ? (target->grad & mask)
                              : andnot(target->grad, mask),
                      target->size);
    }

    void forward(const Variable *source, Variable *target,
                 uint32_t) const override {
        target->accum(!negate ? (source->grad & mask)
                              : andnot(source->grad, mask),
                      source->size);
    }

    JitMask mask;
    bool negate;
};

struct CastEdge : Special {
    CastEdge(VarType v1, VarType v2) : v1(v1), v2(v2) { }

    void backward(Variable *source, const Variable *target,
                  uint32_t) const override {
        source->accum(JitVar::steal(jit_var_cast(target->grad.index(), v1, 0)),
                      target->size);
    }

    void forward(const Variable *source, Variable *target,
                 uint32_t) const override {
        target->accum(JitVar::steal(jit_var_cast(source->grad.index(), v2, 0)),
                      source->size);
    }

    VarType v1, v2;
};

struct MaskGuard {
    MaskGuard(JitBackend backend, const JitMask &mask) : backend(backend), mask(mask) {
        if (mask.index())
            jit_var_mask_push(backend, mask.index());
    }
    ~MaskGuard() {
        if (mask.index())
            jit_var_mask_pop(backend);
    }
    JitBackend backend;
    JitMask mask;
};

struct GatherEdge : Special {
    GatherEdge(const GenericArray<uint32_t> &offset, const JitMask &mask, bool permute)
        : offset(offset), mask(mask), permute(permute) {
        backend = jit_set_backend(mask.index()).backend;
        mask_stack = JitMask::steal(jit_var_mask_peek(backend));
    }

    void backward(Variable *source, const Variable *target, uint32_t) const override {
        JitVar &source_grad = (JitVar &) source->grad;
        size_t size = source->size;

        if (source->size == 1 && target->size == 1 &&
            !(target->flags & VariableFlags::Symbolic)) {
            // Downgrade to scalar op
            source->accum(target->grad & mask, 1);
            return;
        }

        if (!source_grad.valid())
            source_grad =
                JitVar::steal((VarType) source->type == VarType::Float32
                                  ? jit_var_f32(backend, 0.f)
                                  : jit_var_f64(backend, 0.0));

        if (source_grad.size() != size)
            source_grad.resize(size);

        MaskGuard guard(backend, mask_stack);
        if (permute)
            scatter(source_grad, target->grad, offset, mask);
        else
            scatter_reduce(ReduceOp::Add, source_grad, target->grad, offset, mask);
    }

    void forward(const Variable *source, Variable *target, uint32_t) const override {
        MaskGuard guard(backend, mask_stack);
        target->accum(dr::gather<JitVar>(source->grad, offset, mask),
                      std::max(width(offset), width(mask)));
    }

    GenericArray<uint32_t> offset;
    JitMask mask;
    bool permute;

    JitBackend backend;
    JitMask mask_stack;
};

Index ad_var_new(JitIndex i0) {
    Index result = ad_var_new(nullptr, JitVar::steal(i0));
    const char *label = jit_var_label(i0);
    if (label)
        ad_var_set_label(result, label);
    return result;
}

// ==========================================================================
// Implementation of arithmetic operations and transcendental functions
// ==========================================================================

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

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = -dr::sqr(result);
        return ad_var_new("rcp", std::move(result),
                          Arg(i0, std::move(w0)));
    }
}

Index ad_var_rsqrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_rsqrt(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::sqr(result) * result * scalar(i0, -.5);
        return ad_var_new("rsqrt", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_cbrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_cbrt(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::sqr(dr::rcp(result)) * scalar(i0, 1.0 / 3.f);
        return ad_var_new("cbrt", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_erf(Index i0) {
    JitVar result = JitVar::steal(jit_var_erf(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = scalar(i0, 2.0 * dr::InvSqrtPi<double>) *
                    dr::exp(-dr::sqr(v0));
        return ad_var_new("erf", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_sin(Index i0) {
    if (is_detached(i0)) {
        return jit_var_sin(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincos(v0);
        return ad_var_new("sin", std::move(s), Arg(i0, std::move(c)));
    }
}

Index ad_var_cos(Index i0) {
    if (is_detached(i0)) {
        return jit_var_cos(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincos(v0);

        return ad_var_new("cos", std::move(c), Arg(i0, -s));
    }
}

UInt64Pair ad_var_sincos(Index i0) {
    if (is_detached(i0)) {
        UInt32Pair p = jit_var_sincos(jit_index(i0));
        return { p.first, p.second };
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));

        auto [s, c] = dr::sincos(v0);

        Index ci = ad_var_new("cos [sincos]", JitVar(c), Arg(i0, -s)),
              si = ad_var_new("sin [sincos]", std::move(s), Arg(i0, std::move(c)));

        return { si, ci };
    }
}

Index ad_var_tan(Index i0) {
    JitVar result = JitVar::steal(jit_var_tan(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        return ad_var_new("tan", std::move(result),
                          Arg(i0, dr::sqr(dr::rcp(dr::cos(v0)))));
    }
}

Index ad_var_asin(Index i0) {
    JitVar result = JitVar::steal(jit_var_asin(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = dr::rsqrt(dr::fmadd(-v0, v0, scalar(i0, 1.0)));
        return ad_var_new("asin", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_acos(Index i0) {
    JitVar result = JitVar::steal(jit_var_acos(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = -dr::rsqrt(dr::fmadd(-v0, v0, scalar(i0, 1.0)));
        return ad_var_new("acos", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_atan(Index i0) {
    JitVar result = JitVar::steal(jit_var_atan(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = dr::rcp(dr::fmadd(v0, v0, scalar(i0, 1.0)));
        return ad_var_new("atan", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_atan2(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_atan2(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1)) {
        return result.release();
    } else {
        JitVar y = JitVar::borrow(jit_index(i0)),
               x = JitVar::borrow(jit_index(i1)),
               s = dr::rcp(dr::fmadd(x, x, sqr(y)));

        return ad_var_new("atan2", std::move(result),
                          Arg(i0, s * x),
                          Arg(i1, -s * y));
    }
}

Index ad_var_exp(Index i0) {
    JitVar result = JitVar::steal(jit_var_exp(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = result;
        return ad_var_new("exp", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_exp2(Index i0) {
    JitVar result = JitVar::steal(jit_var_exp2(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = result * scalar(i0, dr::LogTwo<double>);
        return ad_var_new("exp2", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_log(Index i0) {
    JitVar result = JitVar::steal(jit_var_log(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::rcp(JitVar::borrow(i0));
        return ad_var_new("log", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_log2(Index i0) {
    JitVar result = JitVar::steal(jit_var_log2(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::rcp(JitVar::borrow(i0)) * scalar(i0, dr::InvLogTwo<double>);
        return ad_var_new("log2", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_sinh(Index i0) {
    if (is_detached(i0)) {
        return jit_var_sinh(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincosh(v0);
        return ad_var_new("sinh", std::move(s), Arg(i0, std::move(c)));
    }
}

Index ad_var_cosh(Index i0) {
    if (is_detached(i0)) {
        return jit_var_cosh(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincosh(v0);
        return ad_var_new("cosh", std::move(c), Arg(i0, std::move(s)));
    }
}

UInt64Pair ad_var_sincosh(Index i0) {
    if (is_detached(i0)) {
        UInt32Pair p = jit_var_sincosh(jit_index(i0));
        return { p.first, p.second };
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));

        auto [s, c] = dr::sincosh(v0);

        Index ci = ad_var_new("cosh [sincos]", JitVar(c), Arg(i0, JitVar(s))),
              si = ad_var_new("sinh [sincos]", std::move(s), Arg(i0, std::move(c)));

        return { si, ci };
    }
}

Index ad_var_tanh(Index i0) {
    JitVar result = JitVar::steal(jit_var_tanh(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        return ad_var_new("tanh", std::move(result), Arg(i0, dr::sqr(dr::rcp(dr::cosh(v0)))));
    }
}

Index ad_var_asinh(Index i0) {
    JitVar result = JitVar::steal(jit_var_asinh(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = dr::rsqrt(dr::fmadd(v0, v0, scalar(i0, 1.0)));
        return ad_var_new("asinh", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_acosh(Index i0) {
    JitVar result = JitVar::steal(jit_var_acosh(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = dr::rsqrt(dr::fmadd(v0, v0, scalar(i0, -1.0)));
        return ad_var_new("acosh", std::move(result), Arg(i0, std::move(w0)));
    }
}

Index ad_var_atanh(Index i0) {
    JitVar result = JitVar::steal(jit_var_atanh(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = dr::rcp(dr::fmadd(-v0, v0, scalar(i0, 1.0)));
        return ad_var_new("atanh", std::move(result), Arg(i0, std::move(w0)));
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

        return ad_var_new("minimum", std::move(result),
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

        return ad_var_new("maximum", std::move(result),
                          Arg(i0, dr::select(mask, one, zero)),
                          Arg(i1, dr::select(mask, zero, one)));
    }
}

Index ad_var_select(Index i0, Index i1, Index i2) {
    JitVar result = JitVar::steal(
        jit_var_select(jit_index(i0), jit_index(i1), jit_index(i2)));

    if (is_detached(i1, i2)) {
        return result.release();
    } else if (jit_var_is_literal(jit_index(i0)) || i1 == i2) {
        Index out_index = jit_var_is_literal_zero(jit_index(i0)) ? i2 : i1;

        ad_log("ad_var_select(a%u <- r%u, a%u, a%u): simplified.",
               ad_index(out_index), jit_index(i0), ad_index(i1), ad_index(i2));

        return ad_var_inc_ref_impl(out_index);
    } else {
        JitMask m = JitMask::borrow(i0);
        return ad_var_new("select", std::move(result),
                          SpecialArg(i1, new MaskEdge(m, false)),
                          SpecialArg(i2, new MaskEdge(m, true)));
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

Index ad_var_reduce(JitBackend backend, VarType vt, ReduceOp op, Index i0) {
    JitVar result = JitVar::steal(jit_var_reduce(backend, vt, op, jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        switch (op) {
            case ReduceOp::Add: {
                    return ad_var_new("sum", std::move(result),
                                      Arg(i0, scalar(i0, 1.0)));
                }

            case ReduceOp::Mul: {
                    JitVar v0 = JitVar::borrow(jit_index(i0)),
                           z  = scalar(i0, 0.0),
                           w0 = dr::select(dr::eq(v0, z), z, result / v0);
                    return ad_var_new("prod", std::move(result),
                                      Arg(i0, std::move(w0)));
                }

            /* The cases below introduce duplicate '1' entries when
               multiple entries are equal to the min/max value, which is
               strictly speaking not correct (but getting this right
               would make the operation quite a bit more expensive). */

            case ReduceOp::Min:
            case ReduceOp::Max: {
                    JitVar v0 = JitVar::borrow(jit_index(i0)),
                           z = scalar(i0, 0.0), o = scalar(i0, 1.0),
                           w0 = dr::select(dr::eq(v0, result), o, z);

                    const char *name = op == ReduceOp::Min ? "min" : "max";

                    return ad_var_new(name, std::move(result),
                                      Arg(i0, std::move(w0)));
                }

            default:
                ad_raise("ad_var_reduce(): unsupported reduction!");
        }
    }
}

Index ad_var_cast(Index i0, VarType vt) {
    JitVar result = JitVar::steal(jit_var_cast(jit_index(i0), vt, 0));

    if (is_detached(i0)) {
        return result.release();
    } else {
        return ad_var_new("cast", std::move(result),
                          SpecialArg(i0, new CastEdge(jit_var_type(i0), vt)));
    }
}

uint64_t ad_var_gather(uint64_t source, uint64_t offset, uint64_t mask, bool permute) {
    JitVar result = JitVar::steal(jit_var_gather(source, offset, mask));

    if (is_detached(source)) {
        return result.release();
    } else {
        return ad_var_new(
            "gather", std::move(result),
            SpecialArg(source,
                       new GatherEdge(GenericArray<uint32_t>::borrow(offset),
                                      JitMask::borrow(mask), permute)));
    }
}


// ==========================================================================
// Debugging: GraphViz, variable listing
// ==========================================================================

static Buffer buffer;

static const char *type_name_short[(int) VarType::Count] {
    "void ", "msk", "i8",  "u8",  "i16", "u16", "i32",
    "u32", "i64", "u64", "ptr", "f16", "f32", "f64"
};


const char *ad_var_whos() {
    std::lock_guard<std::mutex> guard(state.mutex);

    std::vector<uint32_t> indices;
    for (size_t i = 1; i < state.variables.size(); ++i) {
        if (state.variables[i].ref_count == 0)
            continue;
        indices.emplace_back(i);
    }

    std::sort(indices.begin(), indices.end(), [](uint32_t i0, uint32_t i1) {
        return state[i0]->counter < state[i1]->counter;
    });

    buffer.clear();
    buffer.put("\n"
               "  ID        Type        Size     Refs    Label\n"
               "  =========================================================\n");
    for (uint32_t id : indices) {
        const Variable *v = state[id];
        buffer.fmt("  %-9i %-3s %12zu %8u    %s\n", id, type_name_short[v->type],
                   v->size, v->ref_count, v->label ? v->label : "");
    }
    buffer.put("  =========================================================\n");
    return buffer.get();
}

const char *ad_var_graphviz() {
    std::lock_guard<std::mutex> guard(state.mutex);

    std::vector<uint32_t> indices;
    for (size_t i = 1; i < state.variables.size(); ++i) {
        if (state.variables[i].ref_count == 0)
            continue;
        indices.emplace_back(i);
    }

    std::sort(indices.begin(), indices.end(), [](uint32_t i0, uint32_t i1) {
        return state[i0]->counter < state[i1]->counter;
    });

    buffer.clear();
    buffer.put("digraph {\n"
               "    rankdir=BT;\n"
               "    graph [dpi=50 fontname=Consolas];\n"
               "    node [shape=record fontname=Consolas];\n"
               "    edge [fontname=Consolas];\n");

    size_t current_hash = 0, current_depth = 1;
    std::hash<std::string> hasher;

    for (uint32_t index : indices) {
        const Variable *v = state[index];
        const char *label = v->label,
                   *label_without_prefix = label;

        size_t prefix_hash = 0;
        if (label) {
            const char *sep = strrchr(label, '/');
            if (sep) {
                prefix_hash = hasher(std::string(label, sep));
                label_without_prefix = sep + 1;
            }
        } else {
            label = label_without_prefix = "(unnamed)";
        }

        if (prefix_hash != current_hash) {
            for (size_t i = current_depth - 1; i > 0; --i) {
                buffer.put(' ', 4 * i);
                buffer.put("}\n");
            }

            current_hash = prefix_hash;
            current_depth = 1;

            const char *p = label;
            while (true) {
                const char *pn = p ? strchr(p, '/') : nullptr;
                if (!pn)
                    break;

                buffer.put(' ', 4 * current_depth);
                buffer.fmt("subgraph cluster_%08llx {\n",
                           (unsigned long long) hasher(std::string(label, pn)));
                current_depth++;
                buffer.put(' ', 4 * current_depth);
                buffer.put("label=\"");
                buffer.put(p, pn - p);
                buffer.put("\";\n");
                buffer.put(' ', 4 * current_depth);
                buffer.put("color=gray95;\n");
                buffer.put(' ', 4 * current_depth);
                buffer.put("style=filled;\n");

                p = pn + 1;
            }
        }

        buffer.put(' ', 4 * current_depth);
        buffer.put_uint32((uint32_t) index);
        buffer.put(" [label=\"{");

        auto print_escape = [&](const char *s) {
            char c;
            while (c = *s++, c != '\0') {
                bool escape = false;
                switch (c) {
                    case '$':
                        if (s[0] == 'n') {
                            s++;
                            buffer.put("\\l");
                            continue;
                        }
                        break;

                    case '\n':
                        buffer.put("\\l");
                        continue;

                    case '"':
                    case '|':
                    case '{':
                    case '}':
                    case '<':
                    case '>':
                        escape = true;
                        break;
                    default:
                        break;
                }
                if (escape)
                    buffer.put('\\');
                buffer.put(c);
            }
        };

        const char *color = nullptr;
        bool labeled = false;
        if (label_without_prefix && strlen(label_without_prefix) != 0) {
            if (v->flags & VariableFlags::CustomLabel) {
                buffer.put("Label: \\\"");
                labeled = true;
            }
            print_escape(label_without_prefix);
            if (v->flags & VariableFlags::CustomLabel)
                buffer.put("\\\"");
        }

        if (v->next_bwd == 0)
            color = "salmon";
        else if (v->next_fwd == 0)
            color = "lightblue2";
        if (labeled && !color)
            color = "wheat";
        if (v->grad.size() > 0)
            color = "yellowgreen";

        if (v->flags & VariableFlags::Symbolic)
            buffer.put("|{Symbolic}");

        buffer.fmt("|{Type: %s|Size: %zu}|{a%u|Refs: %u}}\"",
            type_name_short[v->type], v->size,
            index, (uint32_t) v->ref_count);

        if (color)
            buffer.fmt(" fillcolor=%s style=filled", color);
        buffer.put("];\n");
    }

    for (size_t i = current_depth - 1; i > 0; --i) {
        buffer.put(' ', 4 * i);
        buffer.put("}\n");
    }

    for (uint32_t index : indices) {
        const Variable *v = state[index];

        uint32_t edge = v->next_bwd, edge_count = 0;
        while (edge) {
            edge = state.edges[edge].next_bwd;
            edge_count++;
        }
        edge = v->next_bwd;
        uint32_t edge_ctr = edge_count;
        while (edge) {
            const Edge &e = state.edges[edge];
            if (edge_count == 1)
                buffer.fmt("    %i -> %i%s;\n", e.target, e.source,
                           e.special ? " [color=red]" : "");
            else
                buffer.fmt("    %i -> %i [label=\" %u\"%s];\n", e.target, e.source,
                           edge_ctr--, e.special ? " color=red" : "");
            edge = e.next_bwd;
        }
    }

    buffer.put(
        "    subgraph cluster_legend {\n"
        "        label=\"Legend\";\n"
        "        l4 [style=filled fillcolor=yellowgreen label=\"Gradient present\"];\n"
        "        l3 [style=filled fillcolor=salmon label=\"Input\"];\n"
        "        l2 [style=filled fillcolor=lightblue2 label=\"Output\"];\n"
        "        l1 [style=filled fillcolor=wheat label=\"Labeled\"];\n"
        "    }\n"
        "}\n");

    return buffer.get();
}
