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
 * backend is tape-based, the combination with deferred evaluation resembles AD
 * via code transformation. The generated code leverages optimizations such as
 * constant propagation and local value numbering and can support some types of
 * control flow without the usual tape-style unrolling.
 */

#include "common.h"
#include <drjit-core/half.h>
#include <drjit/jit.h>
#include <drjit/math.h>
#include <drjit/autodiff.h>
#include <drjit/custom.h>
#include <drjit-core/hash.h>
#include <tsl/robin_set.h>
#include <tsl/robin_map.h>
#include <nanobind/intrusive/counter.inl>
#include <queue>
#include <mutex>

namespace dr = drjit;

/// Reuse the Buffer class from nanobind
#define NB_NAMESPACE drjit
#define NB_NOINLINE DRJIT_NOINLINE
#include "../ext/nanobind/src/buffer.h"
using drjit::detail::Buffer;
static Buffer buffer;

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

/// Create a floating-point scalar Jit variable given a backend and type
DRJIT_NOINLINE JitVar scalar(JitBackend backend, VarType type, double value) {
    switch (type) {
        case VarType::Float16:
            return JitVar::steal(jit_var_f16(backend, drjit::half(value)));
        case VarType::Float32:
            return JitVar::steal(jit_var_f32(backend, (float) value));
        case VarType::Float64:
            return JitVar::steal(jit_var_f64(backend, value));
        default:
            ad_fail("scalar(): unsupported AD scalar type");
            return JitVar();
    }
}

/// Create a scalar Jit variable with the same floating point type and backend
/// as an already existing variable with the provided ``index``
DRJIT_INLINE JitVar scalar(Index index, double value) {
    VarInfo info = jit_set_backend(jit_index(index));
    return scalar(info.backend, info.type, value);
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
    dr::unique_ptr<Special> special;

    /// Edge weight
    JitVar weight;

    /// Visited flag for DFS traversal
    bool visited = false;

    /// Does edge.special store an instance of 'CopyGrad'?
    bool copy_grad = false;

    /// Does edge.special store an instance of 'CustomOp'?
    bool is_custom = false;
};

/// Flags characterizing the 'Variable.flags' bit field
enum VariableFlags : uint8_t {
    /// Was this AD node created while capturing symbolic computation in the
    /// JIT compiler? (e.g. a symbolic loop, virtual function call, etc.)
    Symbolic = 1 << 0,

    /// Was the label manually overwritten via drjit.set_label()?
    CustomLabel = 1 << 1,

    /// Should the label be freed when the variable is deallocated?
    FreeLabel = 1 << 2,

    /// Is this variable the output of a 'CustomOp' operation?
    CustomOpOutput = 1 << 3,

    /// Temporary 'visited' flag used in ad_custom_op
    Visited = 1 << 4,

    /// Is this variable on an iteration boundary of an evaluated loop?
    LoopBoundary = 1 << 5
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

    /// JIT backend associated with this variable
    uint8_t backend = 0;

    /// Floating point type (half/single/double)
    uint8_t type = 0;

    /// Custom flags (see the 'VariableFlag' enum above)
    uint8_t flags = 0;

    Variable() = default;

    Variable(const Variable &) = delete;
    Variable &operator=(const Variable &) = delete;

    Variable(Variable &&v) noexcept
        : ref_count(v.ref_count), next_fwd(v.next_fwd), next_bwd(v.next_bwd),
          grad(std::move(v.grad)), size(v.size), label(v.label),
          counter(v.counter), backend(v.backend), type(v.type), flags(v.flags) {
        v.label = nullptr;
    }

    Variable &operator=(Variable &&v) noexcept {
        ref_count = v.ref_count; next_fwd = v.next_fwd;
        next_bwd = v.next_bwd; grad = std::move(v.grad);
        size = v.size;
        if (flags & (uint8_t) VariableFlags::FreeLabel)
            free(label);
        label = v.label; v.label = nullptr;
        counter = v.counter;
        backend = v.backend;
        type = v.type;
        flags = v.flags;
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
        if (jit_var_is_finite_literal(v2.index()))
            weight = v2;
        else
            weight = dr::select(v1 == zero, zero, v2);

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
                ad_assert(v3.size() == src_size, "mul_accum(): size mismatch");
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
                ad_assert(v.size() == src_size, "accum(): size mismatch!");
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

    /// Sorted list of currently unused edges and vertices
    std::priority_queue<ADIndex, std::vector<ADIndex>, std::greater<uint32_t>> unused_variables;
    std::priority_queue<EdgeIndex, std::vector<EdgeIndex>, std::greater<uint32_t>> unused_edges;

    /// Counter to establish an ordering among variables
    uint64_t counter = 0;

    State() {
        variables.resize(1);
        edges.resize(1);
    }

    ~State() {
        size_t vars_used  = variables.size() - unused_variables.size() - 1,
               edges_used = edges.size() - unused_edges.size() - 1;

        if (vars_used) {
            ad_warn("AD variable leak detected (%zu variables remain in use)!",
                    vars_used);
            size_t count = 0;

            for (size_t i = 0; i < variables.size(); ++i) {
                if (variables[i].ref_count == 0)
                    continue;

                ad_warn(" - variable a%zu (%u references)", i, variables[i].ref_count);
                if (++count == 10) {
                    ad_warn(" - (skipping the rest)");
                    break;
                }
            }
        }

        if (edges_used != 0)
            ad_warn("AD sdge leak detected (%zu edges remain in use)!",
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
                          const Variable * /* target */) {
        ad_fail("Special::backward(): not implemented!");
    }

    virtual void forward(const Variable * /* source */,
                         Variable * /* target */) {
        ad_fail("Special::forward(): not implemented!");
    }

    virtual ~Special() = default;
};

// Custom operation that copies the gradient from an input node
struct CopyGrad : Special {
    void backward(Variable *, const Variable *target) override {
        grad = target->grad;
    }

    void forward(const Variable *source, Variable *) override {
        grad = source->grad;
    }

    JitVar grad;
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

    /// Was this scope created in a symbolic execution context?
    bool symbolic = false;

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

    /// Keeps track of implicit input dependencies of symbolic computation
    tsl::robin_set<uint32_t, UInt32Hasher> implicit_in;

    /// Keeps track of implicit output dependencies of symbolic computation
    tsl::robin_set<uint32_t, UInt32Hasher> implicit_out;

    /// Symbolic operations like `dr.if_stmt` and `dr.while_loop` temporarily
    /// replace variable IDs. This map keeps track of this association, which
    /// is needed to resolve the target/source of gatter/scatter operations.
    tsl::robin_map<ADIndex, ADIndex, UInt32Hasher> variable_map;

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

    /// Nested scopes that restrict AD to specific variables
    std::vector<Scope> scopes;

    ~LocalState() {
        if (!scopes.empty())
            ad_warn("Scope leak detected (%zu scopes remain in use)!",
                    scopes.size());
    }
};

static State state;
static thread_local LocalState local_state;

#if defined(DRJIT_SANITIZE_INTENSE)
static void ad_sanitation_checkpoint_variables() {
    state.variables.emplace_back();
    state.variables.pop_back();
    state.variables.shrink_to_fit();
}
static void ad_sanitation_checkpoint_edges() {
    state.edges.emplace_back();
    state.edges.pop_back();
    state.edges.shrink_to_fit();
}
static void ad_sanitation_checkpoint_both() {
    ad_sanitation_checkpoint_variables();
    ad_sanitation_checkpoint_edges();
}
#endif


// Forward declarations
static void ad_free(ADIndex, Variable *);
static void ad_var_inc_ref_int(ADIndex index, Variable *v) noexcept;


// ==========================================================================
// Reference counting and variable cleanup
// ==========================================================================

static bool DRJIT_NOINLINE ad_decref_custom_op_output(Variable *);

static void ad_var_inc_ref_int(ADIndex index, Variable *v) noexcept {
    DRJIT_MARK_USED(index);
    ad_trace("ad_var_inc_ref(a%u): %u", index, v->ref_count + 1);
    v->ref_count++;
}

static bool ad_var_dec_ref_int(ADIndex index, Variable *v) noexcept {
    DRJIT_MARK_USED(index);
    ad_trace("ad_var_dec_ref(a%u): %u", index, v->ref_count - 1);
    ad_assert(v->ref_count > 0, "ad_var_dec_ref_int(): reference count underflow");

    if (--v->ref_count > 0) {
        if (unlikely(v->flags & (uint8_t) VariableFlags::CustomOpOutput))
            return ad_decref_custom_op_output(v);
        else
            return false;
    } else {
        ad_free(index, v);
        return true;
    }
}

static void ad_free_edges(uint32_t index, Variable *v) {
    EdgeIndex edge_id = v->next_bwd;
    v->next_bwd = 0;
    (void) index;

    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        ad_log("ad_free_edges(): freeing edge a%u -> a%u", edge.source,
               edge.target);

        ad_assert(edge.target == index,
                  "ad_free_edges(): invalid edge connectivity!");

        ADIndex source = edge.source;
        EdgeIndex next_bwd = edge.next_bwd,
                  next_fwd = edge.next_fwd;

        edge = Edge { };

        Variable *v2 = state[source];
        if (!ad_var_dec_ref_int(source, v2)) {
            EdgeIndex fwd = v2->next_fwd;

            if (fwd == edge_id) {
                v2->next_fwd = next_fwd;
            } else {
                while (true) {
                    Edge &edge2 = state.edges[fwd];
                    ad_assert(edge2.source == source,
                              "ad_free_edges(): invalid edge connectivity!");
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
}

static void ad_free(ADIndex index, Variable *v) {
    ad_trace("ad_free(a%u)", index);

    ad_free_edges(index, v);

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


uint32_t ad_var_ref(uint64_t index) {
    uint32_t ad_index = ::ad_index(index);
    if (!ad_index)
        return 0;
    std::lock_guard<std::mutex> guard(state.mutex);
    return state[ad_index]->ref_count;
}

void ad_var_dec_ref_impl(Index index) JIT_NOEXCEPT {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_var_dec_ref(jit_index);

    if (unlikely(ad_index)) {
        std::lock_guard<std::mutex> guard(state.mutex);
        ad_var_dec_ref_int(ad_index, state[ad_index]);
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
    if (!buf) {
        ad_fail("concat(): memory allocation failed!");
        return nullptr;
    }
    memcpy(buf, s1, l1);
    memcpy(buf + l1, s2, l2);
    buf[l1 + l2] = '\0';
    return buf;
}

/// Allocate a new variable from the pool
static std::pair<ADIndex, Variable *> ad_var_new(JitBackend backend,
                                                 size_t size, VarType type,
                                                 bool symbolic,
                                                 bool reuse_indices,
                                                 const char *label) {

    auto &unused = state.unused_variables;
    ADIndex index;

    if (unlikely(unused.empty() || !reuse_indices)) {
        index = (ADIndex) state.variables.size();
        state.variables.emplace_back();
    } else {
        index = unused.top();
        unused.pop();
    }

#if defined(DRJIT_SANITIZE_INTENSE)
    ad_sanitation_checkpoint_variables();
#endif

    Variable *v = &state.variables[index];
    v->ref_count = 1;
    v->size = size;
    v->counter = state.counter++;
    v->backend = (uint8_t) backend;
    v->type = (uint8_t) type;
    v->flags = symbolic ? (uint8_t) VariableFlags::Symbolic : (uint8_t) 0;

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
        index = (EdgeIndex) state.edges.size();
        state.edges.emplace_back();
    } else {
        index = unused.top();
        unused.pop();
    }

#if defined(DRJIT_SANITIZE_INTENSE)
    ad_sanitation_checkpoint_edges();
#endif

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
    Arg() = default;

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
    dr::unique_ptr<Special> special;
};

struct Arg;
template <typename...> struct first {
    using type = Arg;
};

template <typename T, typename... Ts> struct first<T, Ts...> {
    using type = T;
};

template <typename... Ts> using first_t = typename first<Ts...>::type;

/// RAII helper to reduce AD reference counts of a set of variables
struct ReleaseHelper {
    uint32_t item = 0;
    ReleaseHelper *next = nullptr;

    void put(uint32_t value) {
        if (item) {
            ReleaseHelper *rl = new ReleaseHelper();
            rl->next = next;
            rl->item = item;
            next = rl;
        }
        item = value;
    }

    ~ReleaseHelper() {
        if (item) {
            ad_var_dec_ref_int(item, state[item]);
            delete next;
        }
    }
};

/// Forward declaration of a helper function defined later on
uint32_t ad_record_implicit_dependence(LocalState &ls, ReleaseHelper &rl,
                                       JitBackend backend, uint32_t source,
                                       Variable *v_source, bool reuse_indices);

/// This helper function is called by essentially all implementations of
/// arithmetic operations below (e.g. ``ad_var_add``). It creates a new
/// AD variable representing an operation along with edges to inputs. It
/// centrally takes care of various special cases, like custom derivative
/// tracking via AD scopes, etc.
template <typename... Args>
DRJIT_NOINLINE Index ad_var_new_impl(const char *label, JitVar &&result,
                                     Args &&...args_) {
#if defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-value"
#endif

    constexpr size_t N = sizeof...(Args);
#if defined(_MSC_VER)
    constexpr size_t M = N == 0 ? 1 : N;
#else
    constexpr size_t M = N;
#endif

    using ArgType = first_t<Args...>;
    ArgType args[M] { std::move(args_)... };

#if defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

    std::lock_guard<std::mutex> guard(state.mutex);

    /* Potentially turn off derivative tracking for some of the operands if
       we're within a scope that enables/disables gradient propagation
       (globally, or only for specific variables) */
    LocalState &ls = local_state;
    std::vector<Scope> &scopes = ls.scopes;
    if (!scopes.empty()) {
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

    uint32_t flags = jit_flags();

    bool symbolic      = flags & (uint32_t) JitFlag::SymbolicScope,
         reuse_indices = flags & (uint32_t) JitFlag::ReuseIndices;

    VarInfo info = jit_set_backend(result.index());
    ReleaseHelper rh;

    /* Turn symbolic reads from non-symbolic variables into gathers,
       and keep track of implicit dependencies while recording virtual
       function calls */
    if (unlikely(symbolic)) {
        if (label && (strncmp(label, "gather", 6) == 0 ||
                      strncmp(label, "scatter", 7) == 0)) {
            // Gatters/scaters are already handled elsewhere
        } else {
            for (size_t i = 0; i < N; ++i) {
                ADIndex source = args[i].ad_index;
                if (source == 0)
                    continue;

                Variable *v_source = state[source];
                bool source_symbolic =
                    v_source->flags & (uint8_t) VariableFlags::Symbolic;

                if (!source_symbolic)
                    args[i].ad_index = ad_record_implicit_dependence(
                        ls, rh, info.backend, source, v_source, reuse_indices);
            }
        }
    }

    auto [ad_index, var] = ad_var_new(info.backend, info.size, info.type,
                                      symbolic, reuse_indices, label);
    const char *tname = jit_type_name(info.type);

    if constexpr (N == 0) {
        if (label)
            ad_log("ad_var_new(): %s a%u[%zu] = %s()", tname, ad_index, info.size, label);
        else
            ad_log("ad_var_new(): %s a%u[%zu] = new()", tname, ad_index, info.size);
    } else if constexpr (N == 1) {
        ad_log("ad_var_new(): %s a%u[%zu] = %s(a%u)", tname, ad_index, info.size,
               label, args[0].ad_index);
    } else if constexpr (N == 2) {
        ad_log("ad_var_new(): %s a%u[%zu] = %s(a%u, a%u)", tname, ad_index, info.size,
               label, args[0].ad_index, args[1].ad_index);
    } else if constexpr (N == 3) {
        ad_log("ad_var_new(): %s a%u[%zu] = %s(a%u, a%u, a%u)", tname, ad_index, info.size,
               label, args[0].ad_index, args[1].ad_index, args[2].ad_index);
    }

    EdgeIndex edge_index = 0;

    for (size_t i = 0; i < N; ++i) {
        ADIndex source = args[i].ad_index;

        if (!source)
            continue;

        if constexpr (!std::is_same_v<ArgType, SpecialArg>) {
            if (jit_var_is_zero_literal(args[i].weight.index())) {
                ad_trace("ad_var_new(a%u <- a%u): weight of edge %zu is zero, skipping!",
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

    if constexpr (N > 0) {
        if (!edge_index) {
            // All edges were pruned, don't create the node after all
            ad_trace("ad_var_new(a%u): all edges pruned, removing variable.", ad_index);
            ad_free(ad_index, var);
            return result.release();
        }
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

JitIndex ad_grad(Index index, bool null_ok) {
    ADIndex ad_index = ::ad_index(index);
    const std::vector<Scope> &scopes = local_state.scopes;
    if (unlikely(!scopes.empty()))
        scopes.back().maybe_disable(ad_index);

    if (index == 0)
        return 0;

    JitVar result;
    JitBackend backend;
    VarType type;
    size_t size;

    if (ad_index) {
        std::lock_guard<std::mutex> guard(state.mutex);
        const Variable *v = state[ad_index];
        result = v->grad;
        backend = (JitBackend) v->backend;
        type = (VarType) v->type;
        size = v->size;
    } else {
        VarInfo info = jit_set_backend(jit_index(index));
        backend = info.backend;
        type = info.type;
        size = info.size;
    }

    if (!result.valid() && null_ok)
        return 0;

    if (!result.valid() &&
        (type == VarType::Float16 || type == VarType::Float32 ||
         type == VarType::Float64))
        result = scalar(backend, type, 0.0);

    if (result.valid() && result.size() != size)
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

    ad_log("ad_accum_grad(a%u, r%u)", ad_index, value);

    v->accum(value_v, size_in);
}

Index ad_var_set_label(Index index, size_t argc, ...) {
    std::lock_guard<std::mutex> guard(state.mutex);

    // First, turn the variable-length argument list into a usable label
    va_list ap;
    va_start(ap, argc);

    const char *label = nullptr;
    if (argc == 1) {
        label = va_arg(ap, const char *);
    } else if (argc > 1) {
        buffer.clear();

        for (size_t i = 0; i < argc; ++i) {
            const char *s = va_arg(ap, const char *);
            bool isnum = s[0] >= '0' || s[1] <= '9';

            if (isnum) {
                buffer.put('[');
                buffer.put(s, strlen(s));
                buffer.put(']');
            } else {
                if (i > 0)
                    buffer.put('.');
                buffer.put(s, strlen(s));
            }
        }
        label = buffer.get();
    }
    va_end(ap);

    uint32_t jit_index = ::jit_index(index),
             ad_index = ::ad_index(index);

    // Set the label at the JIT level
    jit_index = jit_var_set_label(jit_index, 1, label);

    // If this is an AD variable, also set it here
    if (ad_index) {
        ad_log("ad_var_set_label(a%u): \"%s\"", ad_index,
               label ? label : "(null)");

        Variable *v = state[ad_index];

        if (v->flags & (uint8_t) VariableFlags::FreeLabel)
            free(v->label);

        VarInfo info = jit_set_backend(jit_index);
        const char *prefix = jit_prefix(info.backend);
        if (!prefix || !label)
            v->label = label ? strdup(label) : nullptr;
        else
            v->label = concat(prefix, label);

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
            todo.emplace_back(edge_id, edge.source, edge.target, v->counter,
                              v2->counter);

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

// Invariants: edges can be enqueued, which puts them into a todo list
// Different todo lists may be active at any time, but an edge is only
// in one of them, and its 'visited' bit is consequently set to 1. The
// todo list keeps a reference on the target vertex, which keeps the
// edge from being garbage collected.

static std::pair<Variable*, Variable *> ad_lookup_edge(const EdgeRef &er, const Edge &edge) {
    ad_assert(edge.source == er.source && edge.target == er.target,
              "ad_clear_todo(): internal error: edge a%u -> a%u was "
              "in an invalid state! (1)", er.source, er.target);

    ad_assert(edge.visited,
              "ad_clear_todo(): internal error: edge a%u -> a%u was "
              "in an invalid state! (2)", er.source, er.target);

    Variable *source = state[er.source], *target = state[er.target];

    ad_assert(source->counter == er.source_counter &&
              target->counter == er.target_counter,
              "ad_clear_todo(): internal error: edge a%u -> a%u is in an "
              "invalid state! (3)", er.source, er.target);

    (void) edge;

    return { source, target };
}

/**
 * Clear an edge todo list, decreasing reference counts of the target vertices.
 *
 * If desired, the edge can be removed entirely (``remove_edges=true``), which
 * may cause larger parts of the graph to be garbage-collected.
 */
static void ad_clear_todo(std::vector<EdgeRef> &todo, bool remove_edges) {
    for (const EdgeRef &er : todo) {
        if (er.id == 0 && er.source == 0 && er.target == 0)
            continue; // edge has been moved to another todo list

        Variable *source, *target;
        std::tie(source, target) = ad_lookup_edge(er, state.edges[er.id]);

        if (!remove_edges) {
            state.edges[er.id].visited = 0;
        } else {
            ad_log("ad_clear_todo(): removing edge a%u -> a%u", er.source,
                   er.target);

            // Clear out forward edge
            uint32_t edge_id_prev = 0,
                     edge_id_cur = source->next_fwd;

            while (edge_id_cur) {
                Edge &e2 = state.edges[edge_id_cur];

                ad_assert(e2.source == er.source,
                          "ad_clear_todo(): invalid forward edge connectivity!");

                if (edge_id_cur == er.id) {
                    if (edge_id_prev)
                        state.edges[edge_id_prev].next_fwd = e2.next_fwd;
                    else
                        source->next_fwd = e2.next_fwd;
                    break;
                }

                edge_id_prev = edge_id_cur;
                edge_id_cur = e2.next_fwd;
            }

            ad_assert(edge_id_cur,
                      "ad_clear_todo(): could not find forward edge a%u -> a%u",
                      er.source, er.target);

            // Clear out backward edge
            edge_id_prev = 0;
            edge_id_cur = target->next_bwd;

            while (edge_id_cur) {
                Edge &e2 = state.edges[edge_id_cur];
                ad_assert(e2.target == er.target,
                          "ad_clear_todo(): invalid backward edge connectivity!");

                if (edge_id_cur == er.id) {
                    if (edge_id_prev)
                        state.edges[edge_id_prev].next_bwd = e2.next_bwd;
                    else
                        target->next_bwd = e2.next_bwd;
                    break;
                }

                edge_id_prev = edge_id_cur;
                edge_id_cur = e2.next_bwd;
            }

            ad_assert(edge_id_cur,
                      "ad_clear_todo(): could not find backward edge a%u -> a%u",
                      er.source, er.target);

            state.edges[er.id] = Edge { };
            state.unused_edges.push(er.id);

            source = state[er.source];
            ad_var_dec_ref_int(er.source, source);
            target = state[er.target];
        }

        ad_var_dec_ref_int(er.target, target);
    }

    todo.clear();
}

void ad_traverse(dr::ADMode mode, uint32_t flags) {
    if (mode != dr::ADMode::Forward && mode != dr::ADMode::Backward)
        ad_raise("ad_traverse(): invalid mode specified!");

    LocalState &ls = local_state;
    std::vector<EdgeRef> &todo_tls = ls.todo, todo;
    jit_log(LogLevel::InfoSym,
            "ad_traverse(): processing %zu edges in %s mode ..", todo_tls.size(),
            mode == dr::ADMode::Forward ? "forward" : "backward");

    if (todo_tls.empty())
        return;

    todo.swap(todo_tls);
    bool clear_edges = flags & (uint32_t) dr::ADFlag::ClearEdges;

    std::lock_guard<std::mutex> guard(state.mutex);
    try {
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

        // Any edges with an ID less than this value will be postponed
        uint64_t postpone_before = 0;
        if (!ls.scopes.empty()) {
            const Scope &scope = ls.scopes.back();
            if (scope.isolate)
                postpone_before = scope.counter;
        }

        tsl::robin_set<uint32_t, UInt32Hasher> pending;

        auto postprocess = [&](uint32_t prev_i, uint32_t cur_i) {
            if (!prev_i || prev_i == cur_i)
                return;
            pending.erase(prev_i);

            Variable *prev = state[prev_i],
                     *cur = cur_i ? state[cur_i] : nullptr;

            /* Wavefront-style evaluation of loops with differentiable
               variables produces dummy nodes with the 'LoopBoundary' flag set
               after each iteration. It's good if we dr::schedule() and then
               finally evaluate the gradient of everything processed so far
               after each set of such variables so that AD traversal produces
               reasonably sized kernels (i.e. with an evaluation granularity
               matching the loop iterations of the original/primal evaluation).
               The code below does just that. */

            if (prev->flags & (uint8_t) VariableFlags::LoopBoundary &&
                !(cur && (cur->flags & (uint8_t) VariableFlags::LoopBoundary))) {
                for (uint32_t todo: pending)
                    jit_var_schedule(state[todo]->grad.index());
                jit_eval();
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
            if (prev->counter < postpone_before)
                clear_grad = false;

            // Aggressively clear gradients at intermediate nodes
            if (clear_grad) {
                ad_log("ad_traverse(): clearing gradient at intermediate variable a%u", prev_i);
                prev->grad = JitVar();
            }
        };

        uint32_t v0i_prev = 0;

        // This is the main AD traversal loop
        for (EdgeRef &er : todo) {
            Edge &edge = state.edges[er.id];

            Variable *v0, *v1;
            uint32_t v0i = edge.source, v1i = edge.target;
            std::tie(v0, v1) = ad_lookup_edge(er, edge);

            if (mode == dr::ADMode::Backward) {
                std::swap(v0, v1);
                std::swap(v0i, v1i);
            }

            size_t grad_size = v0->grad.size();

            if (unlikely(v0->counter < postpone_before)) {
                if (mode == dr::ADMode::Backward) {
                    ad_log("ad_traverse(): postponing edge a%u -> a%u due "
                           "dr.isolate_grad() scope.", v0i, v1i);

                    ls.scopes.back().postponed.push_back(er);
                    er.id = er.source = er.target = 0;
                    continue;
                } else if (v1->counter < postpone_before) {
                    ad_raise(
                        "ad_traverse(): tried to forward-propagate derivatives "
                        "across edge a%u -> a%u, which lies outside of the current "
                        "dr.isolate_grad() scope. You must "
                        "enqueue the variables being differentiated and call "
                        "dr.traverse(dr.ADFlag.ClearEdges) *before* entering this "
                        "scope.", v0i, v1i);
                }
            }

            if (unlikely(grad_size != 1 && v0->size != grad_size && !edge.is_custom)) {
                if (grad_size == 0) {
                    ad_log("ad_traverse(): skipping edge a%u -> a%u (no source "
                           "gradient).", v0i, v1i);
                    continue;
                } else {
                    ad_raise("ad_traverse(): gradient propagation encountered "
                             "variable a%u (\"%s\") with an invalid gradient size "
                             "(expected=%zu, actual=%zu)!",
                             v0i, v0->label ? v0->label : "", v0->size, grad_size);
                }
            }

            postprocess(v0i_prev, v0i);
            v0i_prev = v0i;

            pending.insert(v1i);

            ad_log("ad_traverse(): processing edge a%u -> a%u ..", v0i, v1i);

            // Only propagate the label to the gradient if this doesn't require
            // a copy of the variable to be made. (This can interfere with
            // some of the symbolic operations)
            if (unlikely(v0->flags & (uint8_t) VariableFlags::CustomLabel) &&
                jit_var_ref(v0->grad.index()) == 1) {
                dr::string tmp;
                tmp.put(v0->label, " [grad]");
                if (v0->grad.valid())
                    dr::set_label(v0->grad, tmp.c_str());
            }

            if (unlikely(edge.special)) {
                if (mode == dr::ADMode::Forward)
                    edge.special->forward(v0, v1);
                else
                    edge.special->backward(v1, v0);

                if (clear_edges) {
                    // Edge may have been invalidated by callback, look up once more
                    Edge &edge2 = state.edges[er.id];

                    // Don't clear ``CopyGrad`` edges, the custom op does this
                    if (edge2.copy_grad)
                        continue;

                    edge2.special.reset();
                }
            } else {
                v1->mul_accum(v0->grad, edge.weight, v0->size);

                if (clear_edges)
                    edge.weight = JitVar();
            }
        }


        postprocess(v0i_prev, 0);
        ad_log("ad_traverse(): done.");
    } catch (...) {
        ad_clear_todo(todo, false);
        throw;
    }

    ad_clear_todo(todo, clear_edges);

    if (todo_tls.empty())
        todo_tls.swap(todo);
}

// ==========================================================================
// AD scope management
// ==========================================================================

void ad_scope_enter(ADScope type, size_t size, const Index *indices) {
    std::vector<Scope> &scopes = local_state.scopes;
    Scope scope;

    if (!scopes.empty())
        scope = scopes.back();

    scope.symbolic = jit_flag(JitFlag::SymbolicScope);
    scope.postponed.clear();
    scope.implicit_in.clear();
    scope.implicit_out.clear();
    scope.type = type;

    switch (type) {
        case ADScope::Suspend:
            ad_log("ad_scope_enter(suspend, %zu indices)", size);

            if (size) {
                for (size_t i = 0; i < size; ++i)
                    scope.disable(ad_index(indices[i]));
            } else {
                scope.complement = false;
                scope.indices.clear();
            }
            break;

        case ADScope::Resume:
            ad_log("ad_scope_enter(resume, %zu indices)", size);

            if (size) {
                for (size_t i = 0; i < size; ++i)
                    scope.enable(ad_index(indices[i]));
                if (!scope.complement && scope.indices.empty())
                    scope.indices.insert(0);
            } else {
                scope.complement = true;
                scope.indices.clear();
            }
            break;

        case ADScope::Isolate:
            if (unlikely(size))
                ad_fail("ad_scope_enter(isolate): variables cannot be "
                        "specified for this scope type!");

            scope.isolate = true;

            /* access state data structure */ {
                std::lock_guard<std::mutex> guard(state.mutex);
                scope.counter = state.counter;
            }

            ad_log("ad_scope_enter(isolate, ctr=%zu)", scope.counter);
            break;

        default:
            ad_fail("ad_scope_enter(): unknown scope type!");
    }

    scopes.push_back(std::move(scope));
}

void ad_scope_leave(bool process_postponed) {
    LocalState &ls = local_state;
    std::vector<Scope> &scopes = ls.scopes;
    if (scopes.empty())
        ad_raise("ad_scope_leave(): scope underflow!");

    Scope &scope = scopes.back();

    const char *type_name = nullptr;

    switch (scope.type) {
        case ADScope::Suspend: type_name = "suspend"; break;
        case ADScope::Resume:  type_name = "resume"; break;
        case ADScope::Isolate: type_name = "isolate"; break;
        default: type_name = "unknown"; break;
    }
    (void) type_name;

    ad_log("ad_scope_leave(%s)", type_name);

    if (scopes.size() < 2 || !scopes[scopes.size() - 2].symbolic) {
        std::lock_guard<std::mutex> guard(state.mutex);
        for (uint32_t i: scope.implicit_in)
            ad_var_dec_ref_int(i, state[i]);
        for (uint32_t i: scope.implicit_out)
            ad_var_dec_ref_int(i, state[i]);
    } else {
        std::lock_guard<std::mutex> guard(state.mutex);
        Scope &prev = scopes[scopes.size() - 2];
        for (uint32_t i : scope.implicit_in) {
            if (!prev.implicit_in.insert(i).second)
                ad_var_dec_ref_int(i, state[i]);
        }
        for (uint32_t i : scope.implicit_out) {
            if (!prev.implicit_out.insert(i).second)
                ad_var_dec_ref_int(i, state[i]);
        }
    }

    if (scope.isolate && !scope.postponed.empty()) {
        ad_log("ad_scope_leave(): %s %zu postponed edges.",
               process_postponed ? "enqueuing" : "discarding",
               scope.postponed.size());

        if (process_postponed) {
            // Need to process postponed edges now..
            if (unlikely(!ls.todo.empty()))
                ad_raise("ad_scope_leave(): internal error: wanted to process "
                         "postponed AD edges, but other edges were already "
                         "enqueued. Did you forget to call dr.traverse() to "
                         "process them?");

            ls.todo.insert(ls.todo.end(), scope.postponed.begin(),
                           scope.postponed.end());
            scopes.pop_back();

            ad_traverse(dr::ADMode::Backward,
                        (uint32_t) dr::ADFlag::ClearVertices);
        } else {
            std::lock_guard<std::mutex> guard(state.mutex);
            for (EdgeRef &er: scope.postponed) {
                ad_var_dec_ref_int(er.target, state[er.target]);
                state.edges[er.id].visited = 0;
            }
            scopes.pop_back();
        }
    } else {
        if (!scope.postponed.empty())
            ad_raise("ad_scope_leave(): internal error: postponed is nonempty");
        scopes.pop_back();
    }
}

/// Check if gradient tracking is enabled for the given variable
int ad_grad_enabled(Index index) {
    ADIndex ad_index = ::ad_index(index);
    if (!ad_index)
        return 0;

    const std::vector<Scope> &scopes = local_state.scopes;
    if (!scopes.empty())
        scopes.back().maybe_disable(ad_index);
    return ad_index != 0;
}

// ==========================================================================
// AD traversal callbacks for special operations: masks, gathers, scatters
// ==========================================================================

struct MaskEdge : Special {
    MaskEdge(const JitMask &mask, bool negate = false)
        : mask(mask), negate(negate) { }

    void backward(Variable *source, const Variable *target) override {
        source->accum(!negate ? (target->grad & mask)
                              : andnot(target->grad, mask),
                      target->size);
    }

    void forward(const Variable *source, Variable *target) override {
        target->accum(!negate ? (source->grad & mask)
                              : andnot(source->grad, mask),
                      source->size);
    }

    JitMask mask;
    bool negate;
};

struct CastEdge : Special {
    CastEdge(VarType v1, VarType v2) : v1(v1), v2(v2) { }

    void backward(Variable *source, const Variable *target) override {
        source->accum(JitVar::steal(jit_var_cast(target->grad.index(), v1, 0)),
                      target->size);
    }

    void forward(const Variable *source, Variable *target) override {
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

struct Gather : Special {
    Gather(const GenericArray<uint32_t> &offset, const JitMask &mask,
           ReduceMode reduce_mode = ReduceMode::Auto)
        : offset(offset), mask(mask), reduce_mode(reduce_mode) {
        backend = jit_set_backend(mask.index()).backend;
        uint32_t mask_idx = jit_var_mask_peek(backend);
        if (!mask_idx)
            mask_idx = jit_var_mask_default(backend, (uint32_t) dr::width(offset, mask));
        mask_stack = JitMask::steal(mask_idx);
    }

    void backward(Variable *source, const Variable *target) override {
        JitVar &source_grad = source->grad;

        if (source->size == 1 && target->size == 1 &&
            !(target->flags & VariableFlags::Symbolic)) {
            // Downgrade to scalar op
            source->accum(target->grad & mask, 1);
            return;
        }

        if (!source_grad.valid()) {
            VarType type = (VarType)source->type;
            source_grad = scalar(backend, type, 0.0);
        }

        if (source_grad.size() != source->size)
            source_grad.resize(source->size);

        MaskGuard guard(backend, mask_stack);
        dr::scatter_reduce(
            reduce_mode == ReduceMode::Permute ? ReduceOp::Identity
                                               : ReduceOp::Add,
            source_grad, target->grad, offset, mask, reduce_mode);
    }

    void forward(const Variable *source, Variable *target) override {
        MaskGuard guard(backend, mask_stack);
        target->accum(dr::gather<JitVar>(source->grad, offset, mask),
                      std::max(width(offset), width(mask)));
    }

    GenericArray<uint32_t> offset;
    JitBackend backend;
    JitMask mask, mask_stack;
    ReduceMode reduce_mode;
};

/// Edge representing a scatter operation
struct Scatter : Special {
    Scatter(const GenericArray<uint32_t> &offset, const JitMask &mask,
            const JitVar &value, const JitVar &result, ReduceOp op, ReduceMode mode)
        : offset(offset), mask(mask), op(op), mode(mode) {
        backend = jit_set_backend(mask.index()).backend;
        uint32_t mask_idx = jit_var_mask_peek(backend);
        if (!mask_idx)
            mask_idx = jit_var_mask_default(backend, (uint32_t) dr::width(offset, mask));
        mask_stack = JitMask::steal(mask_idx);

        if (op != ReduceOp::Identity && op != ReduceOp::Add) {
            this->value = value;
            this->result = result;
        }
    }

    void backward(Variable *source, const Variable *target) override {
        MaskGuard guard(backend, mask_stack);

        JitVar grad;
        switch (op) {
            case ReduceOp::Identity:
            case ReduceOp::Add:
                grad = dr::gather<JitVar>(target->grad, offset, mask);
                break;

            case ReduceOp::Min:
            case ReduceOp::Max: {
                    JitVar result_value = dr::gather<JitVar>(result, offset);
                    JitMask new_mask = (value == result_value) & mask;
                    grad = dr::gather<JitVar>(target->grad, offset, new_mask);
                }
                break;

            default:
                ad_raise("Scatter::backward(): unexpected case!");
        }

        source->accum(grad, width(offset));
    }

    void forward(const Variable *source, Variable *target) override {
        JitVar &target_grad = target->grad;

        if (!target_grad.valid()) {
            VarType type = (VarType) target->type;
            target_grad = scalar(backend, type, 0.0);
        }

        if (target_grad.size() != target->size)
            target_grad.resize(target->size);

        MaskGuard guard(backend, mask_stack);

        switch (op) {
            case ReduceOp::Identity:
            case ReduceOp::Add:
                dr::scatter_reduce(op, target_grad, source->grad, offset, mask,
                                   mode);
                break;

            case ReduceOp::Min:
            case ReduceOp::Max: {
                    JitVar result_value = dr::gather<JitVar>(result, offset);
                    JitMask new_mask = (value == result_value) & mask;
                    dr::scatter(target_grad, source->grad, offset, new_mask, mode);
                }
                break;

            default:
                ad_raise("Scatter::forward(): unexpected case!");
        }
    }

    GenericArray<uint32_t> offset;
    JitMask mask;
    JitVar value, result;
    ReduceOp op;
    ReduceMode mode;
    JitBackend backend;
    JitMask mask_stack;
};

/// Edge representing the target modified by a scatter operation
struct ScatterTarget : Special {
    ScatterTarget(const GenericArray<uint32_t> &offset, const JitMask &mask_,
                  const JitVar &value_before_, const JitVar &value_after_,
                  ReduceOp op, size_t size)
        : offset(offset), op(op) {

        if (op == ReduceOp::Min || op == ReduceOp::Max) {
            value_before = value_before_;
            value_after = value_after_;
        } else if (op == ReduceOp::Identity) {
            mask = dr::zeros<JitMask>(size);
            dr::scatter(mask, JitMask(true), offset, mask_);
        }
    }

    JitMask create_mask() {
        switch (op) {
            case ReduceOp::Identity:
                return !mask;

            case ReduceOp::Add:
                return JitMask(true);

            case ReduceOp::Min:
            case ReduceOp::Max:
                return value_before == value_after;

            default:
                ad_raise("ScatterTarget::create_mask(): unsupported case!");
        }
    }

    void backward(Variable *source, const Variable *target) override {
        source->accum(target->grad & create_mask(), target->size);
    }

    void forward(const Variable *source, Variable *target) override {
        target->accum(source->grad & create_mask(), source->size);
    }

    GenericArray<uint32_t> offset;
    JitMask mask;
    JitVar value_before, value_after;
    ReduceOp op;
};


struct PrefixSumEdge : Special {
    PrefixSumEdge(bool exclusive) : m_exclusive(exclusive) { }

    void forward(const Variable *source, Variable *target) override {
        JitVar value = source->grad;

        if (value.size() != source->size)
            value.resize(source->size);

        value = dr::prefix_sum(value, m_exclusive);
        target->accum(value, source->size);
    }

    void backward(Variable *source, const Variable *target) override {
        JitVar value = target->grad;
        if (!value.valid())
            return;

        if (value.size() != target->size)
            value.resize(target->size);

        jit_set_backend(value.index());
        value = dr::reverse(dr::prefix_sum(dr::reverse(value), m_exclusive));
        source->accum(value, value.size());
    }

    bool m_exclusive;
};

struct BlockReduceEdge : Special {
    BlockReduceEdge(ReduceOp op, uint32_t block_size, int symbolic,
                    JitVar value_in, JitVar value_out)
        : m_op(op), m_block_size(block_size), m_symbolic(symbolic),
          m_value_in(value_in), m_value_out(value_out) {
        if (m_op == ReduceOp::Add) {
            m_value_in = JitVar();
            m_value_out = JitVar();
        }
    }

    void forward(const Variable *source, Variable *target) override {
        JitVar source_grad = source->grad;

        if (source_grad.size() != source->size)
            source_grad.resize(source->size);

        JitVar result;
        switch (m_op) {
            case ReduceOp::Add:
                result = dr::block_sum(source_grad, m_block_size, m_symbolic);
                break;

            case ReduceOp::Mul:
                result = dr::block_sum(
                    source_grad / m_value_in,
                    m_block_size, m_symbolic) * m_value_out;
                break;

            case ReduceOp::Min:
            case ReduceOp::Max: {
                    JitVar value_tile = dr::tile(m_value_out, m_block_size);
                    result = dr::block_sum(source_grad & (value_tile == m_value_in), m_block_size, m_symbolic);
                }
                break;

            default:
                ad_raise("dr.block_reduce(): derivative not implemented for this reduction!");
        }

        target->accum(result, target->size);
    }

    void backward(Variable *source, const Variable *target) override {
        JitVar target_grad = target->grad;
        if (!target_grad.valid())
            return;

        if (target_grad.size() != target->size)
            target_grad.resize(target->size);

        JitVar result;
        switch (m_op) {
            case ReduceOp::Add:
                result = dr::tile(target_grad, m_block_size);
                break;

            case ReduceOp::Mul:
                result = dr::tile(target_grad * m_value_out, m_block_size) / m_value_in;
                break;

            case ReduceOp::Min:
            case ReduceOp::Max:
                result = dr::select(
                    dr::tile(m_value_out, m_block_size) == m_value_in,
                    dr::tile(target_grad, m_block_size), scalar(m_value_in.index(), 0.0));
                break;

            default:
                ad_raise("dr.block_reduce(): derivative not implemented for this reduction!");
        }

        source->accum(result, result.size());
    }

    ReduceOp m_op;
    uint32_t m_block_size;
    int m_symbolic;
    JitVar m_value_in, m_value_out;
};

struct ShrinkEdge : Special {
    void forward(const Variable *source, Variable *target) override {
        JitVar value = source->grad;
        if (value.size() != source->size)
            value.resize(source->size);

        target->accum(
            JitVar::steal(jit_var_shrink(value.index(), target->size)),
            target->size);
    }

    void backward(Variable *source, const Variable *target) override {
        JitVar value = target->grad;
        if (!value.valid())
            return;

        if (value.size() != target->size)
            value.resize(target->size);

        JitBackend backend = (JitBackend) source->backend;

        JitVar ctr = JitVar::steal(jit_var_counter(backend, source->size)),
               bound = JitVar::steal(jit_var_u32(backend, (uint32_t) (target->size))),
               valid = JitVar::steal(jit_var_lt(ctr.index(), bound.index())),
               expanded = JitVar::steal(
                   jit_var_gather(value.index(), ctr.index(), valid.index()));

        source->accum(expanded, source->size);
    }
};

Index ad_var_new(JitIndex i0) {
    if (i0 == 0)
        return 0;

    Index result = ad_var_new(nullptr, JitVar::borrow(i0));

    const char *label = jit_var_label(i0);
    if (label) {
        VarInfo info = jit_set_backend(i0);
        const char *prefix = jit_prefix(info.backend);

        std::lock_guard<std::mutex> guard(state.mutex);
        Variable *v = state[ad_index(result)];

        if (!prefix || !label)
            v->label = label ? strdup(label) : nullptr;
        else
            v->label = concat(prefix, label);

        v->flags |= (uint8_t) VariableFlags::FreeLabel |
                    (uint8_t) VariableFlags::CustomLabel;
    }

    return result;
}

// ==========================================================================
// Convenience wrappers of jit_var_* functions()
// ==========================================================================

Index ad_var_schedule_force(Index index, int *rv) {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_index = jit_var_schedule_force(jit_index, rv);
    if (ad_index) {
        std::lock_guard<std::mutex> guard(state.mutex);
        ad_var_inc_ref_int(ad_index, state[ad_index]);
    }

    return combine(ad_index, jit_index);
}

Index ad_var_data(Index index, void **ptr) {
    JitIndex jit_index = ::jit_index(index);
    ADIndex ad_index = ::ad_index(index);

    jit_index = jit_var_data(jit_index, ptr);
    if (ad_index) {
        std::lock_guard<std::mutex> guard(state.mutex);
        ad_var_inc_ref_int(ad_index, state[ad_index]);
    }

    return combine(ad_index, jit_index);
}

void ad_mark_loop_boundary(Index index) {
    ADIndex ad_index = ::ad_index(index);
    if (ad_index) {
        std::lock_guard<std::mutex> guard(state.mutex);
        state[ad_index]->flags |= (uint8_t) VariableFlags::LoopBoundary;
    }
}


// ==========================================================================
// Implementation of arithmetic operations and transcendental functions
// ==========================================================================

Index ad_var_copy(Index i0) {
    JitVar result = JitVar::borrow((JitIndex) i0);

    if (likely(is_detached(i0)))
        return result.release();
    else
        return ad_var_new("copy", std::move(result), Arg(i0, 1.0));
}

// ==========================================================================

Index ad_var_add(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_add(jit_index(i0), jit_index(i1)));

    if (likely(is_detached(i0, i1)))
        return result.release();
    else
        return ad_var_new("add", std::move(result), Arg(i0, 1.0), Arg(i1, 1.0));
}

// ==========================================================================

Index ad_var_sub(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_sub(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1))
        return result.release();
    else
        return ad_var_new("sub", std::move(result), Arg(i0, 1.0), Arg(i1, -1.0));
}

// ==========================================================================

Index ad_var_mul(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_mul(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1))
        return result.release();
    else
        return ad_var_new("mul", std::move(result),
                          Arg(i0, JitVar::borrow(jit_index(i1))),
                          Arg(i1, JitVar::borrow(jit_index(i0))));
}

// ==========================================================================

Index ad_var_div(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_div(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               v1 = JitVar::borrow(jit_index(i1)),
               w0 = dr::rcp(v1),
               w1 = -v0 * dr::square(w0);

        return ad_var_new("div", std::move(result),
                          Arg(i0, std::move(w0)),
                          Arg(i1, std::move(w1)));
    }
}

// ==========================================================================

Index ad_var_neg(Index i0) {
    JitVar result = JitVar::steal(jit_var_neg(jit_index(i0)));

    if (is_detached(i0))
        return result.release();
    else
        return ad_var_new("neg", std::move(result), Arg(i0, -1.0));
}

// ==========================================================================

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

// ==========================================================================

Index ad_var_rcp(Index i0) {
    JitVar result = JitVar::steal(jit_var_rcp(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = -dr::square(result);
        return ad_var_new("rcp", std::move(result),
                          Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_rsqrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_rsqrt(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::square(result) * result * scalar(i0, -.5);
        return ad_var_new("rsqrt", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_cbrt(Index i0) {
    JitVar result = JitVar::steal(jit_var_cbrt(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::square(dr::rcp(result)) * scalar(i0, 1.0 / 3.f);
        return ad_var_new("cbrt", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_erf(Index i0) {
    JitVar result = JitVar::steal(jit_var_erf(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0)),
               w0 = scalar(i0, 2.0 * dr::InvSqrtPi<double>) *
                    dr::exp(-dr::square(v0));
        return ad_var_new("erf", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_sin(Index i0) {
    if (is_detached(i0)) {
        return jit_var_sin(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincos(v0);
        return ad_var_new("sin", std::move(s), Arg(i0, std::move(c)));
    }
}

// ==========================================================================

Index ad_var_cos(Index i0) {
    if (is_detached(i0)) {
        return jit_var_cos(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincos(v0);

        return ad_var_new("cos", std::move(c), Arg(i0, -s));
    }
}

// ==========================================================================

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

// ==========================================================================

Index ad_var_tan(Index i0) {
    JitVar result = JitVar::steal(jit_var_tan(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        return ad_var_new("tan", std::move(result),
                          Arg(i0, dr::square(dr::rcp(dr::cos(v0)))));
    }
}

// ==========================================================================

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

// ==========================================================================

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

// ==========================================================================

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

// ==========================================================================

Index ad_var_atan2(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_atan2(jit_index(i0), jit_index(i1)));

    if (is_detached(i0, i1)) {
        return result.release();
    } else {
        JitVar y = JitVar::borrow(jit_index(i0)),
               x = JitVar::borrow(jit_index(i1)),
               s = dr::rcp(dr::fmadd(x, x, square(y)));

        return ad_var_new("atan2", std::move(result),
                          Arg(i0, s * x),
                          Arg(i1, -s * y));
    }
}

// ==========================================================================

Index ad_var_exp(Index i0) {
    JitVar result = JitVar::steal(jit_var_exp(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = result;
        return ad_var_new("exp", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_exp2(Index i0) {
    JitVar result = JitVar::steal(jit_var_exp2(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = result * scalar(i0, dr::LogTwo<double>);
        return ad_var_new("exp2", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_log(Index i0) {
    JitVar result = JitVar::steal(jit_var_log(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::rcp(JitVar::borrow((JitIndex) i0));
        return ad_var_new("log", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_log2(Index i0) {
    JitVar result = JitVar::steal(jit_var_log2(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar w0 = dr::rcp(JitVar::borrow((JitIndex) i0)) * scalar(i0, dr::InvLogTwo<double>);
        return ad_var_new("log2", std::move(result), Arg(i0, std::move(w0)));
    }
}

// ==========================================================================

Index ad_var_sinh(Index i0) {
    if (is_detached(i0)) {
        return jit_var_sinh(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincosh(v0);
        return ad_var_new("sinh", std::move(s), Arg(i0, std::move(c)));
    }
}

// ==========================================================================

Index ad_var_cosh(Index i0) {
    if (is_detached(i0)) {
        return jit_var_cosh(jit_index(i0));
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        auto [s, c] = dr::sincosh(v0);
        return ad_var_new("cosh", std::move(c), Arg(i0, std::move(s)));
    }
}

// ==========================================================================

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

// ==========================================================================

Index ad_var_tanh(Index i0) {
    JitVar result = JitVar::steal(jit_var_tanh(jit_index(i0)));

    if (is_detached(i0)) {
        return result.release();
    } else {
        JitVar v0 = JitVar::borrow(jit_index(i0));
        return ad_var_new("tanh", std::move(result), Arg(i0, dr::square(dr::rcp(dr::cosh(v0)))));
    }
}

// ==========================================================================

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

// ==========================================================================

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

// ==========================================================================

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

// ==========================================================================

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

// ==========================================================================

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

// ==========================================================================

Index ad_var_select(Index i0, Index i1, Index i2) {
    JitVar result = JitVar::steal(
        jit_var_select(jit_index(i0), jit_index(i1), jit_index(i2)));

    if (is_detached(i1, i2)) {
        return result.release();
    } else if (jit_var_state(jit_index(i0)) == VarState::Literal || i1 == i2) {
        Index out_index = jit_var_is_zero_literal(jit_index(i0)) ? i2 : i1;

        ad_log("ad_var_select(a%u <- r%u, a%u, a%u): simplified.",
               ad_index(out_index), jit_index(i0), ad_index(i1), ad_index(i2));

        return ad_var_inc_ref_impl(out_index);
    } else {
        JitMask m = JitMask::borrow((JitIndex) i0);
        return ad_var_new("select", std::move(result),
                          SpecialArg(i1, new MaskEdge(m, false)),
                          SpecialArg(i2, new MaskEdge(m, true)));
    }
}

// ==========================================================================

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

// ==========================================================================

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
                           w0 = dr::select(v0 == z, z, result / v0);
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
                           w0 = dr::select(v0 == result, o, z);

                    const char *name = op == ReduceOp::Min ? "min" : "max";

                    return ad_var_new(name, std::move(result),
                                      Arg(i0, std::move(w0)));
                }

            default:
                ad_raise("ad_var_reduce(): unsupported reduction!");
                return 0;
        }
    }
}

Index ad_var_reduce_dot(Index i0, Index i1) {
    JitVar result = JitVar::steal(jit_var_reduce_dot(jit_index(i0), jit_index(i1)));

    if (is_detached(i0) && is_detached(i1)) {
        return result.release();
    } else {
        return ad_var_new("dot", std::move(result),
                          Arg(i0, JitVar::borrow(jit_index(i1))),
                          Arg(i1, JitVar::borrow(jit_index(i0))));
    }
}

// ==========================================================================

Index ad_var_cast(Index i0, VarType vt) {
    JitVar result = JitVar::steal(jit_var_cast(jit_index(i0), vt, 0));

    if (is_detached(i0)) {
        return result.release();
    } else {
        return ad_var_new("cast", std::move(result),
                          SpecialArg(i0, new CastEdge(jit_var_type((JitIndex) i0), vt)));
    }
}

// ==========================================================================

void ad_var_map_put(Index source, Index target) {
    uint32_t ad_index_source = ad_index(source),
             ad_index_target = ad_index(target);

    if ((ad_index_source == 0) != (ad_index_target == 0))
        ad_raise("ad_var_map_put(): mixed attached/detached inputs!");

    if (ad_index_source == 0)
        return;

    ad_log("ad_var_map_put(): a%u -> a%u", ad_index_source, ad_index_target);

    std::vector<Scope> &scopes = local_state.scopes;
    if (scopes.empty())
        ad_raise("ad_var_map_put(): no scope found!");

    Scope &scope = scopes.back();
    auto [it, success] =
        scope.variable_map.emplace(ad_index_target, ad_index_source);

    if (!success)
        ad_raise("ad_var_map_put(): variable already exists!");
}

/// Symbolic operations like `dr.if_stmt` and `dr.while_loop` temporarily
/// replace variable IDs. This function queris this association, which
/// is needed to resolve the target/source of gatter/scatter operations.
Index ad_var_map_get(Index index) {
    std::vector<Scope> &scopes = local_state.scopes;
    uint32_t ad_index = ::ad_index(index);

    if (scopes.empty() || !ad_index)
        return index;

    const Scope &scope = scopes.back();

    while (true) {
        auto it = scope.variable_map.find(ad_index);
        if (it != scope.variable_map.end())
            ad_index = it->second;
        else
            break;
    }

    return combine(ad_index, jit_index(index));
}

/// Potentially use ad_var_map_get to rewrite the source or target of a
/// gatter/scatter operation
static Index ad_var_memop_remap(Index index, bool input) {
    uint32_t flags = jit_flags();
    if (flags & (uint32_t) JitFlag::SymbolicScope) {
        index = ad_var_map_get(index);

        // Add to set of implicit variable dependencies
        std::vector<Scope> &scopes = local_state.scopes;
        if (scopes.empty())
            ad_raise("ad_var_memop_remap(): expected a scope!");

        Scope &scope = scopes.back();
        uint32_t ad_index = ::ad_index(index);
        if (ad_index == 0)
            return index;

        auto &implicit = input ? scope.implicit_in : scope.implicit_out;
        auto [it, success] = implicit.insert(ad_index);
        if (success) {
            ad_var_inc_ref_int(ad_index, state[ad_index]);
            ad_log("ad_var_memop_remap(): registered an implicit %s dependence "
                   "on variable a%u.", input ? "input" : "output", ad_index);
        }
    }

    return index;
}

// ==========================================================================

uint64_t ad_var_gather(Index source, JitIndex offset, JitIndex mask, ReduceMode mode) {
    JitVar result = JitVar::steal(jit_var_gather(jit_index(source), offset, mask));

    if (is_detached(source)) {
        return result.release();
    } else {
        // Track implicit dependencies & potentially remap variable IDs
        source = ad_var_memop_remap(source, true);

        return ad_var_new(
            mode == ReduceMode::Permute ? "gather_permute" : "gather",
            std::move(result),
            SpecialArg(source,
                       new Gather(GenericArray<uint32_t>::borrow(offset),
                                  JitMask::borrow(mask), mode)));
    }
}

// ==========================================================================

static const char *mode_name[] = { "auto",        "direct", "local",
                                   "no_conflict", "expand", "permute" };
static const char *red_name[] = { "identity", "add", "mul", "min", "max", "and", "or" };

/// Perform a differentiable scatter operation. See jit_var_scatter for signature.
Index ad_var_scatter(Index target, Index value, JitIndex offset, JitIndex mask,
                     ReduceOp op, ReduceMode mode) {
    JitVar target_copy;
    if (!is_detached(value) && (op == ReduceOp::Min || op == ReduceOp::Max))
        target_copy = JitVar::borrow(target);

    JitVar result = JitVar::steal(jit_var_scatter(
        jit_index(target), jit_index(value), offset, mask, op, mode));

    bool perm_scatter = op == ReduceOp::Identity && mode == ReduceMode::Permute;
    if (is_detached(value) && (is_detached(target) || perm_scatter)) {
        ADIndex ad_index = ::ad_index(target);
        if (ad_index) {
            std::lock_guard<std::mutex> guard(state.mutex);
            ad_var_inc_ref_int(ad_index, state[ad_index]);
        }

        return combine(ad_index, result.release());
    } else {
        if (op != ReduceOp::Identity && op != ReduceOp::Add &&
            op != ReduceOp::Min && op != ReduceOp::Max)
            ad_raise("ad_var_scatter(): differentiable scatters are only "
                     "supported for op=dr.ReduceOp.{Identity, Add, Min, Max}");

        // Track implicit dependencies & potentially remap variable IDs
        target = ad_var_memop_remap(target, false);

        const char *name;
        if (op == ReduceOp::Identity)
            name = perm_scatter ? "scatter_permute" : "scatter";
        else
            name = "scatter_reduce";

        Index r = ad_var_memop_remap(
            ad_var_new(
                name, std::move(result),
                SpecialArg(value, new Scatter(
                                      GenericArray<uint32_t>::borrow(offset),
                                      JitMask::borrow(mask),
                                      JitVar::borrow(value), result, op, mode)),
                SpecialArg(target, new ScatterTarget(
                                       GenericArray<uint32_t>::borrow(offset),
                                       JitMask::borrow(mask), target_copy,
                                       result, op, result.size()))),
            false);

        ad_log("ad_var_scatter(): (a%u, r%u) = scatter(op=%s, target=(a%u, r%u), value=(a%u, r%u), offset=r%u, mask=r%u, mode=%s)",
               ad_index(r), jit_index(r),
               red_name[(int) op],
               ad_index(target), jit_index(target),
               ad_index(value), jit_index(value),
               offset, mask, mode_name[(int) mode]);
        return r;
    }
}

void ad_var_scatter_add_kahan(Index *target_1, Index *target_2, Index value,
                              JitIndex offset, JitIndex mask) {
    bool detached_1 = is_detached(*target_1),
         detached_2 = is_detached(*target_2);

    if (detached_1 != detached_2)
        ad_raise("ad_var_scatter_kahan: AD status of the two target arrays is "
                 "inconsistent!");

    uint32_t target_1_jit = jit_index(*target_1),
             target_2_jit = jit_index(*target_2);

    jit_var_scatter_add_kahan(&target_1_jit, &target_2_jit, (JitIndex) value, offset,
                              mask);

    if (is_detached(value) && detached_1) {
        *target_1 = (Index) target_1_jit;
        *target_2 = (Index) target_2_jit;
    } else {
        jit_set_backend(mask);

        uint32_t ad_index_1 = ad_index(ad_var_memop_remap(*target_1, false));
        uint32_t ad_index_2 = ad_index(ad_var_memop_remap(*target_2, false));

        Index combined_1 = ad_var_new(
            "scatter_add_kahan", JitVar::steal(target_1_jit),
            SpecialArg(value,
                       new Scatter(GenericArray<uint32_t>::borrow(offset),
                                   JitMask::borrow(mask), JitVar(), JitVar(),
                                   ReduceOp::Add, ReduceMode::Auto)),
            SpecialArg(*target_1, new MaskEdge(JitMask(true))));

        std::lock_guard<std::mutex> guard(state.mutex);
        ad_var_dec_ref_int(ad_index_1, state[ad_index_1]);

        Index combined_2 = combine(ad_index_2, target_2_jit);

        *target_1 = combined_1;
        *target_2 = combined_2;
    }
}

// ==========================================================================

Index ad_var_prefix_sum(Index index, int exclusive) {
    JitVar result =
        JitVar::steal(jit_var_prefix_sum(jit_index(index), exclusive));

    if (is_detached(index))
        return result.release();
    else
        return ad_var_new("prefix_sum", std::move(result),
                          SpecialArg(index, new PrefixSumEdge(exclusive != 0)));
}

Index ad_var_shrink(Index i0, size_t size) {
    JitVar result = JitVar::steal(jit_var_shrink(jit_index(i0), size));

    if (likely(is_detached(i0)))
        return result.release();
    else
        return ad_var_new("shrink", std::move(result),
                          SpecialArg(i0, new ShrinkEdge()));
}

Index ad_var_block_reduce(ReduceOp op, Index index, uint32_t block_size, int symbolic) {
    if (index == 0)
        return index;
    else if (block_size == 1)
        return ad_var_inc_ref(index);

    JitVar result = JitVar::steal(
        jit_var_block_reduce(op, jit_index(index), block_size, symbolic));

    if (likely(is_detached(index)))
        return result.release();
    else
        return ad_var_new(
            "block_reduce", std::move(result),
            SpecialArg(index, new BlockReduceEdge(op, block_size, symbolic,
                                                  JitVar::borrow(jit_index(index)),
                                                  result)));
}

// ==========================================================================
// Debugging: GraphViz, variable listing
// ==========================================================================

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
        indices.emplace_back((uint32_t) i);
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
        indices.emplace_back((uint32_t) i);
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

// ==========================================================================
// Functionality to track implicit inputs of recorded computation
// ==========================================================================

void ad_var_check_implicit(uint64_t index) {
    ADIndex ad_index = ::ad_index(index);
    if (ad_index == 0 || !jit_flag(JitFlag::SymbolicScope))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    Variable *v = state[ad_index];

    if (!(v->flags & (uint8_t) VariableFlags::Symbolic)) {
        std::vector<Scope> &scopes = local_state.scopes;
        if (scopes.empty())
            ad_raise("ad_var_check_implicit(): no scope found!");

        auto [it, success] = scopes.back().implicit_in.insert(ad_index);
        if (success) {
            ad_log("ad_check_implicit(): registered an implicit input dependence on "
                   "variable a%u.", ad_index);
            ad_var_inc_ref_int(ad_index, state[ad_index]);
        }
    }
}

// Turn symbolic reads from non-symbolic variables into gathers,
// and keep track of implicit dependencies of symbolic function calls
uint32_t ad_record_implicit_dependence(LocalState &ls, ReleaseHelper &rh,
                                       JitBackend backend,
                                       uint32_t source, Variable *v_source,
                                       bool reuse_indices) {
    std::vector<Scope> &scopes = ls.scopes;
    if (scopes.empty())
        ad_raise("ad_record_implicit_dependence(): no scope found!");

    if (v_source->size != 1)
        ad_raise(
            "ad_var_new(): You performed a differentiable operation that mixes symbolic\n"
            "and non-symbolic variables in a non-permissible way. The reason is likely\n"
            "one of the following factors:\n"
            "\n"
            "1. Your program performed a *symbolic* operation such as a\n"
            "\n"
            "   - conditional (via drjit.if_stmt())\n"
            "   - loop (via drjit.while_loop())\n"
            "   - call (via drjit.switch(), drjit.dispatch(), C++ method)\n"
            "\n"
            "   (this potentially involved the @drjit.syntax decorator, which\n"
            "    rewrites scalar Python code to make use of these operations)\n"
            "\n"
            "2. The body of this operation accesses a variable *implicitly*, meaning\n"
            "   that the variable wasn't listed as part of 'args' (for conditionals),\n"
            "   'state' (for loops), or as a function argument (for calls).\n"
            "\n"
            "   If you executed a symbolic call, the variable might, e.g., be a field\n"
            "   of an instance targeted by the call. This is fine.\n"
            "\n"
            "3. Dr.Jit then tried to convert the implicit read into a drjit.gather()\n"
            "   operation to legalize this behavior.\n"
            "\n"
            "   However, the problem is that the variable in question (a%u) has size %zu,\n"
            "   and the conversion to drjit.gather() only makes sense for scalar (size 1)\n"
            "   variables.\n"
            "\n"
            "There are two possible solutions:\n"
            "\n"
            "1. \"Pipe\" the variable to the code in question, by explicitly listing it\n"
            "   as part of conditional inputs, loop state varibles, and function inputs.\n"
            "\n"
            "2. Is this potentially a bug in your code? Did you mean to gather an\n"
            "   element from the variable instead of reading it directly? In that case,\n"
            "   please fix the operation referenced in the stack trace.",
            source, v_source->size);

    auto [ad_index, v] = ad_var_new(backend, 1, (VarType) v_source->type,
                                    true, reuse_indices, "gather");
    v_source = state[source];
    EdgeIndex edge_index_new = ad_edge_new();
    Edge &edge = state.edges[edge_index_new];
    edge.source = source;
    edge.target = ad_index;
    edge.next_fwd = v_source->next_fwd;
    v->next_bwd = edge_index_new;
    v_source->next_fwd = edge_index_new;
    edge.next_bwd = 0;
    edge.special = dr::make_unique<Gather>(GenericArray<uint32_t>(0), JitMask(true));
    ad_var_inc_ref_int(source, v_source);
    ad_log(
        "ad_var_new(): a%u = gather(a%u) [converted from scalar read].",
        ad_index, source);

    auto [it, success] = scopes.back().implicit_in.insert(source);
    if (success) {
        ad_var_inc_ref_int(source, v_source);
        ad_log("ad_var_new(): registered an implicit input dependence "
               "on variable a%u.", ad_index);
    }
    rh.put(ad_index);
    return ad_index;
}

void ad_copy_implicit_deps(drjit::vector<uint32_t>& result, bool input) {
    std::vector<Scope> &scopes = local_state.scopes;
    if (scopes.empty())
        return;

    const Scope &scope = scopes.back();

    auto &implicit = input ? scope.implicit_in : scope.implicit_out;
    if (implicit.empty())
        return;

    ad_log("ad_copy_implicit_deps(): returning %zu implicit %s dependencies.",
           implicit.size(), input ? "input" : "output");

    result.reserve(result.size() + implicit.size());
    for (uint32_t index : implicit) {
        ad_log(" - a%u", index);
        ad_var_inc_ref_int(index, state[index]);
        result.push_back(index);
    }
}

// ==========================================================================
// Custom operations
// ==========================================================================

/// Recreate the suspend/resume status in place when this callback edge was created
struct PushScope {
    PushScope(const Scope &scope) {
        std::vector<Scope> &scopes = local_state.scopes;
        scopes.push_back(scope);

        if (scopes.size() >= 2) {
            Scope &child_scope  = scopes[scopes.size() - 1],
                  &parent_scope = scopes[scopes.size() - 2];
            if (parent_scope.isolate) {
                child_scope.isolate = true;
                child_scope.counter =
                    std::max(child_scope.counter, parent_scope.counter);
            }
        }
    }

    ~PushScope() {
        std::vector<Scope> &scopes = local_state.scopes;

        if (scopes.size() >= 2) {
            Scope &child_scope  = scopes[scopes.size() - 1],
                  &parent_scope = scopes[scopes.size() - 2];

            if (child_scope.isolate && !child_scope.postponed.empty()) {
                if (!parent_scope.isolate)
                    ad_fail("PushScope::~PushScope(): internal error!");

                parent_scope.postponed.insert(
                    parent_scope.postponed.end(),
                    child_scope.postponed.begin(),
                    child_scope.postponed.end()
                );
            }

            std::lock_guard<std::mutex> guard(state.mutex);
            for (uint32_t i: child_scope.implicit_in)
                ad_var_dec_ref_int(i, state[i]);
            for (uint32_t i: child_scope.implicit_out)
                ad_var_dec_ref_int(i, state[i]);

        } else if (scopes.size() == 0) {
            ad_fail("PushScope::~PushScope(): underflow!");
        }

        scopes.pop_back();
    }
};

bool ad_release_one_output(dr::detail::CustomOpBase *op) {
    return op->release_one_output();
}

struct scoped_set_flags {
    uint32_t backup;
    scoped_set_flags(uint32_t flags) : backup(jit_flags()) {
        flags &= ~(uint32_t) JitFlag::SymbolicScope;
        flags |= backup & (uint32_t) JitFlag::SymbolicScope;
        jit_set_flags(flags);
    }

    ~scoped_set_flags() {
        jit_set_flags(backup);
    }
};

struct CustomOp : Special {
    nanobind::ref<dr::detail::CustomOpBase> m_op;
    Scope m_scope;
    uint32_t m_flags;

    CustomOp(dr::detail::CustomOpBase *op, Scope &&scope)
        : m_op(op), m_scope(std::move(scope)), m_flags(jit_flags()) { }

    ~CustomOp() {
        if (m_op.get()) {
            nanobind::ref<dr::detail::CustomOpBase> op = std::move(m_op);
            {
                unlock_guard<std::mutex> guard(state.mutex);
                ad_log("ad_free(): freeing custom operation \"%s\"", op->name());
                op.reset();
            }
        }
    }

    bool swap(const Edge &e, Variable *v) {
        if (e.copy_grad) {
            CopyGrad &copy_grad = *(CopyGrad *) e.special.get();
            std::swap(copy_grad.grad, v->grad);
            return true;
        } else {
            return false;
        }
    }

    bool clear(const Edge &e, Variable *v) {
        if (e.copy_grad) {
            CopyGrad &copy_grad = *(CopyGrad *) e.special.get();
            v->grad = copy_grad.grad;
            copy_grad.grad = JitVar();
            return true;
        } else {
            return false;
        }
    }

    void release_one_output() {
        if (m_op.get() && !ad_release_one_output(m_op.get())) {
            nanobind::ref<dr::detail::CustomOpBase> op = std::move(m_op);
            {
                unlock_guard<std::mutex> guard(state.mutex);
                ad_log("ad_free(): freeing custom operation \"%s\"", op->name());
                op.reset();
            }
        }
    }

    void forward(const Variable *source, Variable *) override {
        ad_log("ad_traverse(): evaluating forward derivative of custom "
               "operation \"%s\"..", m_op->name());
        uint32_t next_bwd = source->next_bwd;

        for (uint32_t ei = next_bwd; ei != 0; ) {
            const Edge &e = state.edges[ei];
            if (!swap(e, state[e.source]))
                break;
            ei = e.next_bwd;
        }

        /* leave critical section */ {
            unlock_guard<std::mutex> guard(state.mutex);
            PushScope push(m_scope);
            scoped_set_flags flag_guard(m_flags);
            m_op->forward();
        }

        #if defined(DRJIT_SANITIZE_INTENSE)
            ad_sanitation_checkpoint_both();
        #endif

        for (uint32_t ei = next_bwd; ei != 0; ) {
            const Edge &e = state.edges[ei];
            if (!clear(e, state[e.source]))
                break;
            ei = e.next_bwd;
        }
    }

    void backward(Variable *, const Variable *target) override {
        uint32_t next_fwd = target->next_fwd;

        ad_log("ad_traverse(): evaluating backward derivative of custom "
               "operation \"%s\"..", m_op->name());

        for (uint32_t ei = next_fwd; ei; ) {
            const Edge &e = state.edges[ei];
            if (!swap(e, state[e.target]))
                break;
            ei = e.next_fwd;
        }

        /* leave critical section */ {
            unlock_guard<std::mutex> guard(state.mutex);
            PushScope push(m_scope);
            scoped_set_flags flag_guard(m_flags);
            m_op->backward();
        }

        #if defined(DRJIT_SANITIZE_INTENSE)
            ad_sanitation_checkpoint_both();
        #endif

        for (uint32_t ei = next_fwd; ei; ) {
            const Edge &e = state.edges[ei];
            if (!clear(e, state[e.target]))
                break;
            ei = e.next_fwd;
        }
    }
};

void ad_add_special(uint32_t v0i, uint32_t v1i, bool is_custom,
                    dr::unique_ptr<Special> special) {
    Variable *v0 = state[v0i], *v1 = state[v1i];

    if (v0->counter >= v1->counter)
        ad_fail("ad_add_special(): internal error!");
    ad_log("ad_add_special(a%u <- a%u)", v1i, v0i);

    uint32_t edge_index_new = ad_edge_new();

    Edge &edge = state.edges[edge_index_new];
    edge.source = v0i;
    edge.target = v1i;
    edge.special = std::move(special);
    edge.copy_grad = !is_custom;
    edge.is_custom = is_custom;

    edge.next_fwd = v0->next_fwd;
    edge.next_bwd = v1->next_bwd;

    v0->next_fwd = edge_index_new;
    v1->next_bwd = edge_index_new;

    ad_var_inc_ref_int(v0i, v0);
}

static Variable *ad_custom_output_create(uint32_t index, Variable *v) {
    bool is_scatter = v->label && strncmp(v->label, "scatter", 7) == 0;

    // References should be held by: caller & CustomOp (2x)
    // Side effects can have a higher refcount
    ad_assert(v->ref_count == 3 || is_scatter,
              "ad_custom_op(): invalid reference count %u in variable a%u",
              v->ref_count, index);

    v->flags |= VariableFlags::CustomOpOutput;

    if (!is_scatter)
        return v;

    // From implicit outputs, remove any prior computation traced within the CustomOp
    v->counter = state.counter++;
    ad_free_edges(index, v);
    return state[index];
}

bool ad_custom_op(dr::detail::CustomOpBase *op) {
    const dr::vector<uint32_t> &inputs  = op->m_input_indices,
                               &outputs = op->m_output_indices;

    if (inputs.empty() || outputs.empty() || op->m_counter_offset == 0)
        return false;

    for (uint32_t i: inputs)
        state[i]->flags &= ~(uint8_t) VariableFlags::Visited;
    for (uint32_t o: outputs)
        state[o]->flags &= ~(uint8_t) VariableFlags::Visited;

    const char *name = op->name();

    ad_log("ad_var_custom_op(\"%s\", n_in=%zu, n_out=%zu)",
           name, inputs.size(), outputs.size());

    std::lock_guard<std::mutex> guard(state.mutex);

    uint32_t flags = jit_flags();

    bool symbolic    = flags & (uint32_t) JitFlag::SymbolicScope,
         reuse_indices = flags & (uint32_t) JitFlag::ReuseIndices;

    ADIndex v0i, v1i;
    if (inputs.size() == 1) {
        v0i = inputs[0];
        ad_log(" - in: a%u", v0i);
        ad_var_inc_ref_int(v0i, state[v0i]);
    } else {
        auto [idx, v0] = ad_var_new(op->m_backend, 1, VarType::Void, symbolic,
                                    reuse_indices, "CustomOp[in]");
        ad_log("ad_var_new(a%u, \"%s [in]\")", idx, name);
        v0->counter = op->m_counter_offset;
        v0->size = 0;
        v0i = idx;

        for (uint32_t i: inputs) {
            uint8_t &flags_ref = state[i]->flags;
            if (flags_ref & (uint8_t) VariableFlags::Visited) {
                ad_log(" - in: a%u (ignored)", i);
                continue;
            }

            flags_ref |= (uint8_t) VariableFlags::Visited;
            ad_log(" - in: a%u", i);
            ad_add_special(i, v0i, false, dr::make_unique<CopyGrad>());
        }
    }

    if (outputs.size() == 1) {
        v1i = outputs[0];
        Variable *v1 = ad_custom_output_create(v1i, state[v1i]);
        ad_log(" - out: a%u", v1i);
        ad_var_inc_ref_int(v1i, v1);
    } else {
        auto [idx, v1] = ad_var_new(op->m_backend, 1, VarType::Void, symbolic,
                                    reuse_indices, "CustomOp[out]");
        ad_log("ad_var_new(a%u, \"%s [in]\")", idx, name);
        v1->counter = op->m_counter_offset + 1;
        v1i = idx;

        for (uint32_t o: outputs) {
            uint8_t & flags_ref = state[o]->flags;
            if (flags_ref & (uint8_t) VariableFlags::Visited) {
                ad_log(" - out: a%u (ignored)", o);
                continue;
            }

            flags_ref |= (uint8_t) VariableFlags::Visited;

            Variable *vo = ad_custom_output_create(o, state[o]);

            ad_log(" - out: a%u", o);
            ad_add_special(v1i, o, false, dr::make_unique<CopyGrad>());
            vo = state[o];
            vo->flags |= VariableFlags::CustomOpOutput;
        }
    }

    op->m_outputs_alive = (uint32_t) outputs.size();

    const std::vector<Scope> &scopes = local_state.scopes;
    Scope scope;

    if (!scopes.empty()) {
        scope = scopes.back();
        scope.postponed.clear();
        scope.implicit_in.clear();
        scope.implicit_out.clear();
    }

    ad_add_special(v0i, v1i, true,
                   dr::make_unique<CustomOp>(op, std::move(scope)));

    Variable *v0 = state[v0i], *v1 = state[v1i];

    if (v1->flags & (uint8_t) VariableFlags::FreeLabel)
        free(v1->label);

    const char *prefix = jit_prefix(op->m_backend);
    if (!prefix)
        v1->label = strdup(name);
    else
        v1->label = concat(prefix, name);

    v1->flags |= (uint8_t) VariableFlags::FreeLabel |
                 (uint8_t) VariableFlags::CustomLabel;

    ad_var_dec_ref_int(v0i, v0);
    ad_var_dec_ref_int(v1i, v1);

    op->m_counter_offset = 0;
    return true;
}

// This routine is called when decreasing the reference count of an output node
// following a CustomOp. It breaks what would otherwise be an uncollectable
// reference cycle, since the CustomOp references its own outputs.
static bool DRJIT_NOINLINE ad_decref_custom_op_output(Variable *v) {
    uint32_t next_bwd = v->next_bwd;
    if (v->ref_count != 2 || !next_bwd)
        return false;

    Edge *edge = &state.edges[v->next_bwd];
    if (edge->copy_grad) {
        next_bwd = state[edge->source]->next_bwd;
        if (!next_bwd)
            return false;
        edge = &state.edges[next_bwd];
    }

    ad_assert(edge->is_custom, "ad_decref_custom_op_output(): expected to "
                               "find an edge representing a CustomOp!");

    if (!edge->special)
        return false;

    size_t counter = v->counter;

    ((CustomOp *) edge->special.get())->release_one_output();

    // CustomOp may have been destroyed so check again if output was also freed
    // or reused in the meantime
    return v->ref_count == 0 || v->counter != counter;
}


NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

CustomOpBase::CustomOpBase() {
    std::lock_guard<std::mutex> guard(state.mutex);
    m_backend = JitBackend::None;
    m_counter_offset = state.counter;
    state.counter += 2;
}

CustomOpBase::~CustomOpBase() {
    std::lock_guard<std::mutex> guard(state.mutex);

    for (size_t i = 0, size = m_input_indices.size(); i < size; ++i) {
        ADIndex ad_index = m_input_indices[i];
        ad_var_dec_ref_int(ad_index, state[ad_index]);
    }

    for (size_t i = 0, size = m_output_indices.size(); i < size; ++i) {
        ADIndex ad_index = m_output_indices[i];
        ad_var_dec_ref_int(ad_index, state[ad_index]);
    }
}

bool CustomOpBase::release_one_output() {
    return --m_outputs_alive > 0;
}

bool CustomOpBase::add_index(JitBackend backend, ADIndex index, bool input) {
    if (m_backend != backend) {
        if (m_backend != JitBackend::None)
            ad_raise("CustomOpBase::add_index(): can't mix several backends!");
        m_backend = backend;
    }

    const std::vector<Scope> &scopes = local_state.scopes;
    if (!scopes.empty())
        scopes.back().maybe_disable(index);

    if (!index)
        return false;

    std::lock_guard<std::mutex> guard(state.mutex);
    ad_var_inc_ref_int(index, state[index]);

    dr::vector<uint32_t> &indices = input ? m_input_indices
                                          : m_output_indices;
    indices.push_back(index);

    return true;
}

void CustomOpBase::forward() {
    throw std::runtime_error(std::string(name()) +
                             "::forward(): operation is unimplemented!");
}

void CustomOpBase::backward() {
    throw std::runtime_error(std::string(name()) +
                             "::backward(): operation is unimplemented!");
}

const char *CustomOpBase::name() const { return "CustomOpBase"; }

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
