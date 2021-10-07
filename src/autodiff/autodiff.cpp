/** Enoki automatic differentiation library
 *
 * This file implements the AD data structures and traversal routines
 * underlying templated Enoki types like 'DiffArray<CUDAArray<float>>'. The
 * compilation process explicitly instantiates these templates for
 * scalar/LLVM/CUDA arrays in both single and double precision and merges them
 * into a shared library "enoki-autodiff.so/dll". In this way, the machinery
 * below only needs to be compiled once instead of adding a heavy compilation
 * burden to any code using the AD types.
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
 * and 'ad_traverse()': Arithmetic involving differentiable Enoki arrays
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
 * The combination with JITted (CUDA/LLVM) array types is quite interesting: in
 * this case, AD traversal generates code that can be executed at some later
 * point. While Enoki's AD backend is principally tape-based, this combination
 * then begins to resemble classical AD via code transformation. The JITted
 * modes also exploit their ability to peek into literal constant arrays to
 * optimize generated derivative code.
 *
 * One potential limitation of the implementation here is that a separate AD
 * graph exists for each type (e.g. float/double, Scalar/CUDA/LLVM arrays).
 * Because of this, arithmetic involving multiple different flavors of floating
 * point arrays is not end-to-end differentiable without extra precautions.
 * This is an intentional decision to allow for a simple and performant
 * implementation.
 */

#include "common.h"
#include <enoki/jit.h>
#include <enoki/math.h>
#include <enoki/autodiff.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <assert.h>
#include <mutex>
#include <xxh3.h>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

// ==========================================================================
// Helper and forward declarations
// ==========================================================================

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

#define CONCAT(x,y) x ## _ ## y
#define EVAL(x,y) CONCAT(x,y)
#define RENAME(fun) EVAL(fun, ENOKI_AUTODIFF_NAME)

// Rename various things to avoid symbol clashes
#define Special              RENAME(Special)
#define Edge                 RENAME(Edge)
#define Variable             RENAME(Variable)
#define State                RENAME(State)
#define ReleaseQueueHelper   RENAME(ReleaseQueueHelper)
#define ReleaseOperandHelper RENAME(ReleaseOperandHelper)

using Value = ENOKI_AUTODIFF_VALUE;
using Mask = mask_t<Value>;
using Index = uint32_array_t<Value>;

// Forward declarations
struct Variable;
struct Special;

static void ad_free(uint32_t index, Variable *v);
template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather_impl(const char *label, size_t size, uint32_t src_index,
                           const Index &offset, const Mask &mask, bool permute);

template <typename T> bool is_valid(const T &value) {
    if constexpr (is_jit_array_v<T>)
        return value.valid();
    else
        return true;
}

// ==========================================================================
// Central data structures: edges, variables, global state
// ==========================================================================

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
 * 'next_fwd' and 'next_rev' indices each implement a singly linked list that
 * can be used to iterate through the forward edges (of the 'source' variable)
 * and reverse edges (of the 'target' variable).
 */
struct Edge {
    /// Variable index of source operand
    uint32_t source;

    /// Source variable index
    uint32_t target;

    /// Links to the next forward edge
    uint32_t next_fwd;

    /// Links to the next backward edge
    uint32_t next_rev;

    /// Pointer to a handler for "special" edges
    Special *special;

    /// Weight value (zero/empty for "special" edges)
    Value weight{};

    ENOKI_ARRAY_DEFAULTS(Edge);

    Edge() {
        memset(this, 0, sizeof(uint32_t) * 4 + sizeof(Special *));
    }

    /// Reset the contents of this edge to the default values
    void reset();
};

/**
 * Reference-counted data structure representing an AD variable
 *
 * The data structure associates a gradient with each variable, which may or
 * may not be valid (it's the job of the AD traversal to propagate gradients to
 * variables lacking them). No primal information is stored except for the size
 * of the original program variable.
 *
 * Adjacency, i.e. how the variable is connected to other variables in either
 * direction, is represented using linked lists. The 'next_fwd' and 'next_rev'
 * fields each provide an entry point into such a linked list of edges (see
 * also \ref Edge).
 *
 * The 'placeholder' bit is used to track variables that were created while
 * recording a megakernel (i.e., inside a symbolic loop, virtual function call,
 * etc.). This is a highly restricted execution context where certain
 * operations (e.g. computation that must be partitioned into multiple kernel
 * launches) must be postponed.
 */
struct Variable {
    /// Number of references to this variable
    uint32_t ref_count;

    /// Links to the first forward edge at this node
    uint32_t next_fwd;

    /// Links to the first backward edge at this node
    uint32_t next_rev;

    /// Array size of the associated primal variable
    uint32_t size;

    /// Descriptive label or nullptr
    char *label;

    /// High bits of variable index (unused atm.)
    uint32_t index_hi : 16;

    /// Gradient reference count for custom operations
    uint32_t ref_count_grad : 13;

    /// Was the label manually overwritten via set_label()?
    uint32_t custom_label : 1;

    /// Should the label be freed when the variable is deallocated?
    uint32_t free_label : 1;

    /// Was this graph node created while recording computation?
    uint32_t placeholder : 1;

    /// This field may or may not hold a valid gradient value
    Value grad{};

    Variable() {
        memset(this, 0, sizeof(char *) + 5 * sizeof(uint32_t));
    }

    Variable(const char *label_, size_t size_, bool placeholder_) : Variable() {
        label = (char *) label_;
        if (!label)
            label = (char *) "unnamed";

        if (size_ > 0xffffffffu)
            ad_fail("AD variable is too large (max. size = 2^32)");
        size = (uint32_t) size_;

        const char *prefix = ad_prefix();
        if (prefix) {
            size_t size_2 = strlen(prefix) + strlen(label) + 2;
            char *out = (char *) malloc(size_2);
            snprintf(out, size_2, "%s/%s", prefix, label);
            label = out;
            free_label = 1;
        }

        placeholder = (uint32_t) placeholder_;
    }

    /**
     * \brief Accumulate a gradient 'v' originating from another variable of
     * size 'src_size' into the current variable.
     *
     * This is a relatively important operation that is heavily used during AD
     * traversal, hence the implementation considers a few different cases and
     * optimizations.
     */
    template <typename T = Value>
    void accum(const T& v, uint32_t src_size) {
        if constexpr (is_array_v<T>) {
            bool grad_valid = is_valid(grad);

            if (size == 1 && src_size != 1) {
                /* When this variable is scalar (size == 1) and the source is
                   not (src_size != 1), the gradient must be reduced to a single
                   value. A special case arises when the source gradient is
                   actually scalar after all, in which case it is considered to
                   broadcast to all elements. */

                Value v2;
                if (v.size() == 1) {
                    v2 = v * scalar_t<Value>(src_size);
                } else {
                    assert(v.size() == src_size);
                    v2 = hsum_async(v);
                }

                if (grad_valid)
                    grad += v2;
                else
                    grad = std::move(v2);
            } else {
                if (grad_valid)
                    grad += v;
                else
                    grad = v;
            }
        } else {
            grad += v;
        }
    }

    /**
     * \brief Accumulate a gradient 'v1' originating from another variable of
     * size 'src_size' into the current variable, and scale it by a weight 'v2_'.
     *
     * This is a relatively important operation that is heavily used during AD
     * traversal, hence the implementation considers a few different cases and
     * optimizations.
     */
    template <typename T = Value>
    void mul_accum(const T &v1, const T &v2_, uint32_t src_size) {
        /* The goal of the following logic is to always ensure that
           v1 == 0 implies v1 * v2 == 0, even if multiplication by
           v2 would produce a NaN (e.g. if v2 is infinite or NaN). */

        T z = 0.f, v2;

        if constexpr (is_jit_array_v<T>) {
            if (v2_.is_literal() && std::isnormal(v2_[0]) &&
                jit_flag(JitFlag::ADOptimize)) {
                /* The check can be elided if the edge weight is a normal
                   literal constant. This can save significant amounts of
                   unnecessary eq() and select() operations in generated IR */
                v2 = v2_;
            } else {
                /* Only use this if absolutely necessary (also because it
                   triggers a forced evalution in case any of the input
                   variables have pending scatter operations) */
                v2 = select(eq(v1, z), z, v2_);
            }
        } else {
            // Scalar case
            v2 = select(eq(v1, z), z, v2_);
        }

        if constexpr (is_array_v<T>) {
            bool grad_valid = is_valid(grad);

            if (size == 1 && src_size != 1) {
                /* When this variable is scalar (size == 1) and the source is
                   not (src_size != 1), the gradient must be reduced to a single
                   value. A special case arises when the source gradient is
                   actually scalar after all, in which case it is considered to
                   broadcast to all elements. */

                T v3 = v1 * v2;
                if (v3.size() == 1) {
                    v3 *= scalar_t<Value>(src_size);
                } else {
                    assert(v3.size() == src_size);
                    v3 = hsum_async(v3);
                }

                if (grad_valid)
                    grad += v3;
                else
                    grad = std::move(v3);
            } else {
                if (grad_valid)
                    grad = fmadd(v1, v2, grad);
                else
                    grad = v1 * v2;
            }
        } else {
            grad = fmadd(v1, v2, grad);
        }
    }

    bool is_scalar() const { return size == 1; }

    ENOKI_ARRAY_DEFAULTS(Variable);
};

/// Records the (global) state of the AD graph
struct State {
    using VariableMap = tsl::robin_map<uint32_t, Variable, UInt32Hasher,
                                       std::equal_to<uint32_t>>;
    using EdgeVector  = std::vector<Edge>;

    /// std::mutex protecting the state data structure
    std::mutex mutex;

    /// Hash table mapping variable IDs to variable instances
    VariableMap variables;

    /// List of all edges (used and unused ones)
    EdgeVector edges;

    /// List of currently unused edges
    std::vector<uint32_t> unused_edges;

    /// Counter for variable indices
    uint32_t variable_index = 1;

    State() : edges(1) { }

    ~State() {
        if (!variables.empty()) {
            ad_log(Warn,
                   "enoki-autodiff: variable leak detected (%zu variables "
                   "remain in use)!", variables.size());
            uint32_t counter = 0;
            for (auto kv : variables) {
                ad_log(Warn, " - variable a%i (%u references)", kv.first,
                       kv.second.ref_count);
                if (++counter == 10) {
                    ad_log(Warn, " - (skipping the rest)");
                    break;
                }
            }
        }

        size_t edges_used = edges.size() - unused_edges.size() - 1;
        if (edges_used != 0)
            ad_log(Warn,
                   "enoki-autodiff: edge leak detected (%zu edges "
                   "remain in use)!", edges_used);
    }

    Variable *operator[](uint32_t index) {
        auto it = variables.find(index);
        if (unlikely(index == 0 || it == variables.end()))
            ad_fail("referenced an unknown variable a%i!", index);
        return &it.value();
    }
};

struct EdgeRef {
    uint32_t id;
    uint32_t source;
    uint32_t target;

    EdgeRef() : id(0), source(0), target(0) { }
    EdgeRef(uint32_t id, uint32_t source, uint32_t target)
    : id(id), source(source), target(target) { }
};

using VisitedSet = tsl::robin_set<uint32_t, UInt32Hasher, std::equal_to<uint32_t>>;

// Stores per-thread state
struct LocalState {
    /// Thread-local edge list used by ad_enqueue_*() and ad_traverse()
    std::vector<EdgeRef> todo;

    /// Keeps track of implicit input dependencies of recorded computation
    std::vector<EdgeRef> implicit;

    /// List of AD variables that cannot be processed and must be postponed
    std::vector<EdgeRef> postponed;

    /// List of nodes that were visited by ek::enqueue
    VisitedSet visited;

    /// Requested directionality of differentiation
    ADMode mode = ADMode::Reverse;
};

// Special edge (scatter, gather, scatter_reduce, block_sum, etc.)
struct Special {
    virtual void backward(Variable * /* source */, const Variable * /* target */) const {
        ad_fail("Special::backward(): not implemented!");
    }

    virtual void forward(const Variable * /* source */, Variable * /* target */) const {
        ad_fail("Special::forward(): not implemented!");
    }

    virtual ~Special() = default;
};

constexpr bool IsDouble = std::is_same_v<Value, double>;
static_assert(sizeof(Edge) == 8 * sizeof(uint32_t),
              "Edge data structure has incorrect size. Padding problem?");

static_assert(sizeof(Variable) == ((IsDouble ? 2 : 0) + 8) * sizeof(uint32_t),
              "Variable data structure has incorrect size. Padding problem?");

// ==========================================================================
// Global state variables, thread local storage
// ==========================================================================

/// Global state, protected by a mutex
static State state;

/// Thread-local state
static thread_local LocalState local_state;

void Edge::reset() {
    Special *special_copy = special;
    memset(this, 0, sizeof(uint32_t) * 4 + sizeof(Special *));
    weight = Value();
    if (special_copy) {
        unlock_guard<std::mutex> guard(state.mutex);
        delete special_copy;
    }
}

// ==========================================================================
// Reference counting and variable cleanup
// ==========================================================================

static void ad_inc_ref(uint32_t index, Variable *v) noexcept (true) {
    ENOKI_MARK_USED(index);
    ad_trace("ad_inc_ref(a%i): %u", index, v->ref_count + 1);
    v->ref_count++;
}

static bool ad_dec_ref(uint32_t index, Variable *v) noexcept (true) {
    ENOKI_MARK_USED(index);
    ad_trace("ad_dec_ref(a%i): %u", index, v->ref_count - 1);

    if (unlikely(v->ref_count == 0))
        ad_fail("enoki-autodiff: fatal error: external reference count of "
                "variable a%i became negative!", index);

    if (--v->ref_count == 0) {
        ad_free(index, v);
        return true;
    } else {
        return false;
    }
}

template <typename T> void ad_inc_ref_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_inc_ref(index, state[index]);
}

template <typename T> void ad_dec_ref_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_dec_ref(index, state[index]);
}

/// Clear backward edges of the given variable and decrease int. ref. counts
static void ad_free_edges(uint32_t index, Variable *v) {
    ENOKI_MARK_USED(index);

    uint32_t edge_id = v->next_rev;
    ad_trace("ad_free_edges(a%i)", index);
    v->next_rev = 0;

    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        ad_trace("ad_free_edges(): freeing edge a%i -> a%i", edge.source,
                 edge.target);

        uint32_t source = edge.source;
        uint32_t next_rev = edge.next_rev,
                 next_fwd = edge.next_fwd;

        assert(edge.target == index);
        edge.reset();

        Variable *v2 = state[source];

        if (unlikely(v2->ref_count == 0))
            ad_fail("enoki-autodiff: fatal error: reference count of variable "
                    "a%i became negative!", source);

        if (!ad_dec_ref(source, v2)) {
            uint32_t fwd = v2->next_fwd;
            if (fwd == edge_id) {
                v2->next_fwd = next_fwd;
            } else {
                while (true) {
                    Edge &edge2 = state.edges[fwd];
                    assert(edge2.source == source);
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

        edge_id = next_rev;
    }
}

static void ad_free(uint32_t index, Variable *v) {
    ad_trace("ad_free(a%i)", index);
    if (v->free_label)
        free(v->label);
    if (v->next_rev)
        ad_free_edges(index, v);
    state.variables.erase(index);
}

// ==========================================================================
// Variable/edge creation
// ==========================================================================

/// Allocate a new variable
static std::pair<uint32_t, Variable *> ad_var_new(const char *label,
                                                 size_t size) {
    while (true) {
        uint32_t index = state.variable_index++;

        if (unlikely(index == 0)) { // overflow
            state.variable_index = 1;
            index = state.variable_index++;
        }

        bool rec = false;
        if (is_jit_array_v<Value>)
            rec = jit_flag(JitFlag::Recording);

        auto result = state.variables.try_emplace(index, label, size, rec);
        if (likely(result.second))
            return { index, &result.first.value() };
    }
}

/// Allocate a new edge from the pool
static uint32_t ad_edge_new() {
    uint32_t index;
    if (likely(!state.unused_edges.empty())) {
        index = state.unused_edges.back();
        state.unused_edges.pop_back();
    } else {
        index = (uint32_t) state.edges.size();
        state.edges.emplace_back();
    }
    return index;
}


/// Ensure consistent size of placeholder variables to avoid horiz. reductions
static void ad_propagate_placeholder_size(Variable *v) {
    uint32_t edge = v->next_rev;
    while (edge) {
        Edge &e = state.edges[edge];
        Variable *v2 = state[e.source];
        if (v2->placeholder && v2->size != v->size && v2->size == 1) {
            v2->size = v->size;
            ad_propagate_placeholder_size(v2);
        }
        edge = e.next_rev;
    }
}

/// RAII helper class to clean up reference counts of a limited # of operands
struct ReleaseOperandHelper {
    uint32_t pos = 0;
    uint32_t values[3];

    void put(uint32_t index) {
        if (unlikely(pos == 3))
            ad_fail("ReleaseOperandHelper(): overflow!");
        values[pos++] = index;
    }

    ~ReleaseOperandHelper() {
        for (uint32_t i = 0; i < pos; ++i) {
            uint32_t index = values[i];
            ad_dec_ref(index, state[index]);
        }
    }
};

template <typename T>
uint32_t ad_new(const char *label, size_t size, uint32_t op_count,
               uint32_t *op, T *weights) {
    std::lock_guard<std::mutex> guard(state.mutex);

    bool rec = false;
    if constexpr (is_jit_array_v<Value>)
        rec = jit_flag(JitFlag::Recording);

    ReleaseOperandHelper helper;
    if (unlikely(rec)) {
        for (uint32_t i = 0; i < op_count; ++i) {
            if (op[i] == 0)
                continue;

            uint32_t index = op[i];
            Variable *var = state[index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(!var->placeholder)) {
                if (var->size != 1)
                    ad_raise(
                        "ad_new(): recorded computation performs an implicit "
                        "read of variable (a%i), which has size %u! However, "
                        "only scalar (size == 1) accesses are permitted in "
                        "this manner. You will likely want to convert the "
                        "read into an enoki::gather() operation.",
                        index, var->size);

                ad_trace("ad_new(): implicit read of variable a%i, inserting a "
                         "gather operation..", op[i]);

                index = ad_new_gather_impl<Value>("gather", size, op[i], Index(0),
                                                  Mask(true), false);

                var = state[index];
                op[i] = index;
                helper.put(index);
            }
        }
    }

    auto [index, var] = ad_var_new(label, size);

    if (unlikely(log_level >= Debug)) {
        const char *l1 = label ? ", label=": "";
        const char *l2 = label ? label : "";
        switch (op_count) {
            case 0:
                ad_log(Debug, "ad_new(a%i, size=%zu%s%s)", index, size, l1, l2); break;
            case 1:
                ad_log(Debug, "ad_new(a%i <- a%i, size=%zu%s%s)", index, op[0], size, l1, l2); break;
            case 2:
                ad_log(Debug, "ad_new(a%i <- a%i, a%i, size=%zu%s%s)", index, op[0], op[1], size, l1, l2); break;
            case 3:
                ad_log(Debug, "ad_new(a%i <- a%i, a%i, a%i, size=%zu%s%s)", index, op[0], op[1], op[2], size, l1, l2); break;
            default: break;
        }
    }

    uint32_t edge_index = 0;
    for (uint32_t i = 0; i < op_count; ++i) {
        if (op[i] == 0)
            continue;

        bool weight_is_zero = false;
        if constexpr (is_jit_array_v<T>)
            weight_is_zero = jit_flag(JitFlag::ADOptimize) &&
                             weights[i].is_literal() && weights[i][0] == 0;
        else
            weight_is_zero = weights[i] == 0;

        if (weight_is_zero) {
            ad_trace(
                "ad_new(a%i <- a%i): weight of edge %i is zero, skipping!",
                index, op[i], i);
            continue;
        }

        uint32_t index2 = op[i];
        Variable *var2 = state[index2];

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = index2;
        edge.target = index;
        edge.weight = std::move(weights[i]);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = edge_index;
        edge_index = edge_index_new;

        ad_inc_ref(index2, var2);
        var2->next_fwd = edge_index_new;
    }

    if (op_count > 0 && edge_index == 0) {
        // All edges were pruned, don't create the node after all
        ad_trace(
            "ad_new(a%i): all nodes were pruned, removing variable from graph", index);
        ad_free(index, var);
        return 0;
    }

    var->next_rev = edge_index;
    var->ref_count = 1;

    if (var->placeholder)
        ad_propagate_placeholder_size(var);

    return index;
}

// ==========================================================================
// Implementation of AD for special operations: masks, gathers, scatters
// ==========================================================================

template <typename Value> struct MaskEdge : Special {
    MaskEdge(const Mask &mask, bool negate) : mask(mask), negate(negate) { }

    void backward(Variable *source, const Variable *target) const override {
        source->accum(!negate ? detail::and_(target->grad, mask)
                              : detail::andnot_(target->grad, mask),
                      target->size);
    }

    void forward(const Variable *source, Variable *target) const override {
        target->accum(!negate ? detail::and_(source->grad, mask)
                              : detail::andnot_(source->grad, mask),
                      source->size);
    }

    Mask mask;
    bool negate;
};

template <typename Value> struct SpecialCallback : Special {
    std::unique_ptr<DiffCallback> callback;

    SpecialCallback(DiffCallback* callback) : callback(callback) { }

    void backward(Variable *, const Variable *target) const override {
        uint32_t edge = target->next_fwd;
        if (callback) {
            ad_trace("ad_traverse(): invoking user callback ..");
            /* leave critical section */ {
                unlock_guard<std::mutex> guard(state.mutex);
                callback->backward();
            }
            if (edge && state.edges[edge].next_fwd) { // fan-in > 1, update ref counts
                do {
                    const Edge &e = state.edges[edge];
                    Variable *v = state[e.target];
                    if (v->ref_count_grad > 0) {
                        if (--v->ref_count_grad == 0)
                            v->grad = Value();
                    }
                    edge = e.next_fwd;
                } while (edge);
            }
        } else {
            if (target->size != 0)
                const_cast<Variable *>(target)->ref_count_grad++;
        }
    }

    void forward(const Variable *source, Variable *) const override {
        uint32_t edge = source->next_rev;
        if (callback) {
            ad_trace("ad_traverse(): invoking user callback ..");
            /* leave critical section */ {
                unlock_guard<std::mutex> guard(state.mutex);
                callback->forward();
            }
            if (edge && state.edges[edge].next_rev) { // fan-in > 1, update ref counts
                do {
                    const Edge &e = state.edges[edge];
                    Variable *v = state[e.source];
                    if (v->ref_count_grad > 0) {
                        if (--v->ref_count_grad == 0)
                            v->grad = Value();
                    }
                    edge = e.next_rev;
                } while (edge);
            }
        } else {
            if (source->size != 0)
                const_cast<Variable *>(source)->ref_count_grad++;
        }
    }
};

template <typename Value, typename Mask>
uint32_t ad_new_select(const char *label, size_t size, const Mask &mask,
                      uint32_t t_index, uint32_t f_index) {
    std::lock_guard<std::mutex> guard(state.mutex);
    if constexpr (is_jit_array_v<Mask>) {
        if (jit_flag(JitFlag::ADOptimize) && mask.is_literal()) {
            uint32_t result = mask[0] ? t_index : f_index;
            if (result)
                ad_inc_ref(result, state[result]);
            ad_log(Debug, "ad_new_select(a%i <- a%i, a%i): simplified", result, t_index, f_index);
            return result;
        }

        if (jit_flag(JitFlag::ADOptimize) && f_index == t_index) {
            if (t_index)
                ad_inc_ref(t_index, state[t_index]);
            ad_log(Debug, "ad_new_select(a%i <- a%i, a%i): simplified", t_index, t_index, f_index);
            return t_index;
        }
    }

    bool rec = false;
    if constexpr (is_jit_array_v<Value>)
        rec = jit_flag(JitFlag::Recording);

    uint32_t op[2] = { t_index, f_index };
    ReleaseOperandHelper helper;

    if (rec) {
        for (uint32_t i = 0; i < 2; ++i) {
            if (op[i] == 0)
                continue;

            uint32_t index = op[i];
            Variable *var = state[index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(!var->placeholder)) {
                if (var->size != 1)
                    ad_raise(
                        "ad_new_select(): recorded computation performs an "
                        "implicit read of variable (a%i), which has size %u! "
                        "However, only scalar (size == 1) accesses are "
                        "permitted in this manner. You will likely want to "
                        "conver the read into an enoki::gather() operation.",
                        index, var->size);

                ad_trace("ad_new_select(): implicit read of variable a%i, inserting a "
                         "gather operation..", op[i]);

                index = ad_new_gather_impl<Value>("gather", size, op[i], Index(0),
                                                  Mask(true), false);

                var = state[index];
                op[i] = index;
                helper.put(index);
            }
        }
    }

    auto [index, var] = ad_var_new(label, size);

    ad_log(Debug, "ad_new_select(a%i <- a%i, a%i)", index, t_index, f_index);
    uint32_t edge_index = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        if (op[i] == 0)
            continue;

        uint32_t index2 = op[i];
        Variable *var2 = state[index2];

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = index2;
        edge.target = index;
        edge.special = new MaskEdge<Value>(mask, i != 0);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = edge_index;
        edge_index = edge_index_new;

        ad_inc_ref(index2, var2);

        var2->next_fwd = edge_index_new;
    }

    var->next_rev = edge_index;
    var->ref_count = 1;

    if (var->placeholder)
        ad_propagate_placeholder_size(var);

    return index;
}

template <typename Value> struct GatherEdge : Special {
    GatherEdge(const Index &offset, const Mask &mask, bool permute)
        : offset(offset), mask(mask), permute(permute) { }

    void backward(Variable *source, const Variable *target) const override {
        Value &source_grad = (Value &) source->grad;
        uint32_t size = source->size;

        if (source->size == 1 && target->size == 1 && !target->placeholder) {
            // Downgrade to scalar op
            source->accum(select(mask, target->grad, 0.f), 1);
            return;
        }

        if (!source_grad.valid())
            source_grad = zero<Value>(size);
        else if ((uint32_t) source_grad.size() != size)
            source_grad.resize(size);

        if (permute)
            scatter(source_grad, target->grad, offset, mask);
        else
            scatter_reduce(ReduceOp::Add, source_grad, target->grad, offset, mask);
    }

    void forward(const Variable *source, Variable *target) const override {
        target->accum(gather<Value>(source->grad, offset, mask),
                      (uint32_t) width(offset));
    }

    Index offset;
    Mask mask;
    bool permute;
};

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather_impl(const char *label, size_t size, uint32_t src_index,
                           const Index &offset, const Mask &mask_, bool permute) {
    Mask mask(mask_);

    if constexpr (is_array_v<Value>) {
        if (is_jit_array_v<Value>) {
            // Apply the mask stack (needed for wavefront-mode ek::Loop)
            Mask top = Mask::steal(jit_var_mask_peek(Mask::Backend));
            size_t tsize = top.size();
            if (tsize != 1 && tsize == size)
                mask &= top;
        }

        auto [index, var] = ad_var_new(label, size);

        ad_log(Debug, "ad_new_gather(a%i <- a%i, size=%zu, permute=%i)", index,
               src_index, size, (int) permute);

        Variable *var2 = state[src_index];

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = src_index;
        edge.target = index;
        edge.special = new GatherEdge<Value>(offset, mask, permute);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = 0;
        ad_inc_ref(src_index, var2);
        var2->next_fwd = edge_index_new;
        var->next_rev = edge_index_new;
        var->ref_count = 1;

        /* Encountered a dependency between recorded/non-recorded computation
           that will need special handling when the AD graph is traversed at
           some later point. For now, just keep track of this event. */
        if (unlikely(var->placeholder && !var2->placeholder)) {
            ad_trace("ad_new_gather(): recording an implicit dependency (a%i).", src_index);
            local_state.implicit.emplace_back(edge_index_new, src_index, index);
        }

        return index;
    } else {
        (void) mask; (void) label; (void) size;
        (void) src_index; (void) offset; (void) permute;
        enoki_raise("ad_new_gather(): differentiable gathers not supported by "
                    "this backend!");
    }
}

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather(const char *label, size_t size, uint32_t src_index,
                      const Index &offset, const Mask &mask, bool permute) {
    std::lock_guard<std::mutex> guard(state.mutex);
    return ad_new_gather_impl<Value>(label, size, src_index, offset, mask, permute);
}

template <typename Value> struct ScatterEdge : Special {
    ScatterEdge(const Index &offset, const Mask &mask, ReduceOp op)
        : offset(offset), mask(mask), op(op) {
            if (op != ReduceOp::None && op != ReduceOp::Add)
                enoki_raise("AD only supports ReduceOp::Add in scatter_reduce!");
        }

    void backward(Variable *source, const Variable *target) const override {
        source->accum(gather<Value>(target->grad, offset, mask),
                      (uint32_t) width(offset));
    }

    void forward(const Variable *source, Variable *target) const override {
        Value &target_grad = (Value &) target->grad;
        uint32_t size = target->size;

        if (!target_grad.valid())
            target_grad = zero<Value>(size);
        else if ((uint32_t) target_grad.size() != size)
            target_grad.resize(size);

        if (op != ReduceOp::None)
            scatter_reduce(op, target_grad, source->grad, offset, mask);
        else
            scatter(target_grad, source->grad, offset, mask);
    }

    Index offset;
    Mask mask;
    ReduceOp op;
};

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_scatter(const char *label, size_t size, ReduceOp op,
                       uint32_t src_index, uint32_t dst_index, const Index &offset,
                       const Mask &mask_, bool permute) {

    Mask mask(mask_);
    ENOKI_MARK_USED(mask);

    if constexpr (is_array_v<Value>) {
        std::lock_guard<std::mutex> guard(state.mutex);

        if (is_jit_array_v<Value>) {
            // Apply the mask stack (needed for wavefront-mode ek::Loop)
            Mask top = Mask::steal(jit_var_mask_peek(Mask::Backend));
            size_t tsize = top.size(),
                   ssize = std::max(
                       std::max(
                           offset.size(),
                           (size_t)(src_index ? state[src_index]->size : 0)),
                       mask_.size());
            if (tsize != 1 && tsize == ssize)
                mask &= top;
        }

        auto [index, var] = ad_var_new(label, size);

        ad_log(Debug,
               "ad_new_scatter(op=%i, a%i <- a%i, a%i, permute=%i)",
               (int) op, index, src_index, dst_index, (int) permute);

        uint32_t edge_index = 0;

        if (src_index != 0) {
            Variable *var2 = state[src_index];
            uint32_t edge_index_new = ad_edge_new();
            Edge &edge = state.edges[edge_index_new];
            edge.source = src_index;
            edge.target = index;
            edge.special = new ScatterEdge<Value>(offset, mask, op);
            edge.next_fwd = var2->next_fwd;
            edge.next_rev = var->next_rev;
            ad_inc_ref(src_index, var2);
            var2->next_fwd = edge_index_new;
            edge_index = edge_index_new;
        }

        if (dst_index != 0) {
            Variable *var2 = state[dst_index];

            uint32_t edge_index_new = ad_edge_new();
            Edge &edge2 = state.edges[edge_index_new];
            edge2.source = dst_index;
            edge2.target = index;
            edge2.next_fwd = var2->next_fwd;
            edge2.next_rev = edge_index;
            if (op != ReduceOp::None || permute) {
                edge2.weight = 1;
            } else {
                Mask edge_mask = full<Mask>(false, size);
                scatter(edge_mask, Mask(true), offset, mask);
                edge2.special = new MaskEdge<Value>(edge_mask, true);
            }
            ad_inc_ref(dst_index, var2);
            var2->next_fwd = edge_index_new;
            edge_index = edge_index_new;
        }

        if (edge_index == 0)
            ad_raise("ad_new_scatter(): all inputs were non-differentiable!");

        var->next_rev = edge_index;
        ad_inc_ref(index, var);

        return index;
    } else {
        (void) label; (void) size; (void) op; (void) src_index;
        (void) dst_index; (void) offset; (void) permute;
        enoki_raise("ad_new_scatter(): differentiable scatters not supported "
                    "by this backend!");
    }
}

// ==========================================================================
// Interface for querying and modifying variables
// ==========================================================================

template <typename T> T ad_grad(uint32_t index, bool fail_if_missing) {
    if (unlikely(index == 0))
        return T(0);

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_grad(): referenced an unknown variable a%i!", index);
        return T(0);
    }

    const Variable &v = it->second;
    T result = v.grad;

    if constexpr (is_jit_array_v<T>) {
        if (!is_valid(result))
            result = zero<T>(v.size);
        else if (result.size() != v.size)
            result.resize(v.size);
    }

    return result;
}

template <typename T>
void ad_set_grad(uint32_t index, const T &value, bool fail_if_missing) {
    if (unlikely(index == 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_set_grad(): referenced an unknown variable a%i!", index);
        return;
    }

    size_t size_in = width(value);
    Variable &v = it.value();

    if (v.size != size_in && size_in != 1 && v.size != 1)
        ad_raise("ad_set_grad(): attempted to assign a gradient of size "
                 "%zu to AD variable a%i, which has size %u!",
                 size_in, index, v.size);

    ad_trace("ad_set_grad(a%i)", index);
    if (v.size != 1 || size_in == 1)
        v.grad = value;
    else
        v.grad = hsum_async(value);
}

template <typename T>
void ad_accum_grad(uint32_t index, const T &value, bool fail_if_missing) {
    if (unlikely(index == 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_accum_grad(): referenced an unknown variable a%i!", index);
        return;
    }

    size_t size_in = width(value);
    Variable &v = it.value();

    if (v.size != size_in && size_in != 1 && v.size != 1)
        ad_raise("ad_accum_grad(): attempted to accumulate a gradient of size "
                 "%zu into AD variable a%i, which has size %u!",
                 size_in, index, v.size);

    ad_trace("ad_accum_grad(a%i)", index);
    v.accum(value, (uint32_t) size_in);
}

template <typename T> void ad_set_label(uint32_t index, const char *label) {
    if (index == 0)
        return;
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_log(Debug, "ad_set_label(a%i, \"%s\")", index, label ? label : "(null)");
    Variable *v = state[index];
    if (v->free_label)
        free(v->label);
    v->label = strdup(label);
    v->free_label = true;
    v->custom_label = true;
}

template <typename T> const char *ad_label(uint32_t index) {
    if (index == 0)
        return nullptr;
    std::lock_guard<std::mutex> guard(state.mutex);
    return state[index]->label;
}

template <typename T>
void ad_add_edge(uint32_t source_idx, uint32_t target_idx,
                 DiffCallback *callback) {
    if (source_idx == 0 || target_idx == 0)
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    ad_log(Debug, "ad_add_edge(a%i -> a%i)", source_idx, target_idx);

    Variable *source = state[source_idx],
             *target = state[target_idx];

    uint32_t edge_index_new = ad_edge_new();
    Edge &edge = state.edges[edge_index_new];
    edge.source = source_idx;
    edge.target = target_idx;
    edge.special = new SpecialCallback<Value>(callback);
    edge.next_fwd = source->next_fwd;
    edge.next_rev = target->next_rev;

    source->next_fwd = edge_index_new;
    ad_inc_ref(source_idx, source);
    target->next_rev = edge_index_new;
}


// ==========================================================================
// Enqueuing of variables and edges
// ==========================================================================

/// Forward-mode DFS starting from 'index'
static void ad_dfs_fwd(VisitedSet &visited, std::vector<EdgeRef> &todo, uint32_t index) {
    auto [it, success] = visited.insert(index);
    if (!success)
        return;

    Variable *v = state[index];
    ad_inc_ref(index, v);
    uint32_t edge_id = v->next_fwd;
    while (edge_id) {
        Edge &edge = state.edges[edge_id];
        todo.emplace_back(edge_id, edge.source, edge.target);
        ad_trace("ad_dfs_fwd(): enqueuing edge a%i -> a%i", index,
                 edge.target);
        ad_dfs_fwd(visited, todo, edge.target);
        edge_id = edge.next_fwd;
    }
}

/// Reverse-mode DFS starting from 'index'
static void ad_dfs_rev(VisitedSet &visited, std::vector<EdgeRef> &todo, uint32_t index) {
    auto [it, success] = visited.insert(index);
    if (!success)
        return;

    Variable *v = state[index];
    ad_inc_ref(index, v);
    uint32_t edge_id = v->next_rev;
    while (edge_id) {
        Edge &edge = state.edges[edge_id];
        todo.emplace_back(edge_id, edge.source, edge.target);
        ad_trace("ad_dfs_rev(): enqueuing edge a%i -> a%i", index,
                 edge.source);
        ad_dfs_rev(visited, todo, edge.source);
        edge_id = edge.next_rev;
    }
}

template <typename T> void ad_enqueue(ADMode mode, uint32_t index) {
    if (index == 0)
        return;

    ad_trace("ad_enqueue_node(a%i, mode=%s)", index,
             mode == ADMode::Forward ? "forward" : "reverse");

    LocalState &ls = local_state;

    if (ls.visited.empty() && ls.todo.empty()) {
        ls.mode = mode;
    } else if (ls.mode != mode) {
        ad_raise("ad_enqueue(): attempted to enqueue nodes using "
                 "incompatible 'ADMode' values (i.e. both forward *and* "
                 "reverse-mode differentation)");
    }

    std::lock_guard<std::mutex> guard(state.mutex);
    if (mode == ADMode::Forward)
        ad_dfs_fwd(ls.visited, ls.todo, index);
    else
        ad_dfs_rev(ls.visited, ls.todo, index);
}

// ==========================================================================
// AD graph traversal
// ==========================================================================

template <typename Value>
void ad_traverse(bool retain_graph, bool retain_grad) {
    LocalState &ls = local_state;

    std::vector<EdgeRef> &todo_tls = ls.todo, todo;
    if (todo_tls.empty()) {
        for (uint32_t index : ls.visited)
            ad_dec_ref(index, state[index]);
        ls.visited.clear();
        return;
    }

    /// Are we currently recording a megakernel?
    bool rec = false;
    if (is_jit_array_v<Value>)
        rec = jit_flag(JitFlag::Recording);

    std::lock_guard<std::mutex> guard(state.mutex);
    todo_tls.swap(todo);

    // Bring into the appropriate order
    ADMode mode = ls.mode;
    std::sort(todo.begin(), todo.end(), [mode](EdgeRef e1, EdgeRef e2) {
        if (mode == ADMode::Reverse)
            return std::tie(e1.target, e1.source) > std::tie(e2.target, e2.source);
        else
            return std::tie(e1.source, e1.target) < std::tie(e2.source, e2.target);
    });

    // Remove duplicates
    todo.erase(
        std::unique(todo.begin(), todo.end(),
                    [](EdgeRef e1, EdgeRef e2) { return e1.id == e2.id; }),
        todo.end());

    ad_log(Debug, "ad_traverse(): processing %zu edges in %s mode ..", todo.size(),
           mode == ADMode::Forward ? "forward" : "reverse");

    std::vector<Value> ek_loop_todo;
    auto postprocess = [&](uint32_t prev_i, uint32_t cur_i) {
        if (!prev_i || prev_i == cur_i)
            return;

        Variable *cur  = cur_i ? state[cur_i] : nullptr,
                 *prev = state[prev_i];

        /* Wave-front style evaluation of ek.Loop with differentiable
           variables produces nodes with label 'ek_loop' at the boundary of
           each iteration. It's good if we ek::schedule() and then finally
           evaluate the gradient of all such variables at once so that AD
           tarversal produces reasonably sized kernels (i.e. with an evaluation
           granularity matching the loop iterations of the original/primal
           evaluation). The code below does just that. */

        bool ek_loop_prev = prev->label && strstr(prev->label, "ek_loop"),
             ek_loop_cur  = cur && cur->label && strstr(cur->label, "ek_loop");

        if (ek_loop_prev) {
            ek_loop_todo.push_back(prev->grad);
            schedule(prev->grad);

            if (!ek_loop_cur) {
                ad_trace("ad_traverse(): evaluating %zi loop variables",
                         ek_loop_todo.size());
                eval();
                ek_loop_todo.clear();
            }
        }

        // Aggressively clear gradients at intermediate nodes
        if (!retain_grad && prev->ref_count_grad == 0) {
            ad_trace("ad_traverse(): clearing gradient at intermediate variable a%i", prev_i);
            prev->grad = Value();
        }
    };

    uint32_t v0i_prev = 0;

    // This is the main AD traversal loop
    for (EdgeRef edge_ref : todo) {
        Edge &edge = state.edges[edge_ref.id];

        uint32_t v0i, v1i;
        if (mode == ADMode::Forward) {
            v0i = edge.source;
            v1i = edge.target;
        } else {
            v0i = edge.target;
            v1i = edge.source;
        }

        if (edge.source != edge_ref.source || edge.target != edge_ref.target)
            ad_fail(
                "ad_traverse(): internal error: edge a%i -> a%i was garbage "
                "collected between enqueue and traverse steps!", v0i, v1i);

        Variable *v0 = state[v0i],
                 *v1 = state[v1i];

        ad_trace("ad_traverse(): processing edge a%i -> a%i ..", v0i, v1i);

        uint32_t grad_size = (uint32_t) width(v0->grad);

        if (unlikely(mode == ADMode::Reverse && rec && !v0->placeholder)) {
            ad_trace("ad_traverse(): postponing edge (must be handled outside of megakernel).");
            ls.postponed.push_back(edge_ref);
            ad_inc_ref(v0i, v0);
            continue;
        } else if (unlikely(grad_size != 1 && v0->size != grad_size)) {
            if (grad_size == 0) {
                ad_trace("ad_traverse(): skipping edge (no source gradient).");
                continue;
            } else {
                ad_raise("ad_traverse(): gradient propagation encountered "
                         "variable a%i (\"%s\") with an invalid gradient size "
                         "(expected size %u, actual size %u)!",
                         v0i, v0->label ? v0->label : "", v0->size, grad_size);
            }
        }

        postprocess(v0i_prev, v0i);
        v0i_prev = v0i;

        if (unlikely(v0->custom_label)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v0->label);
            if (width(v0->grad) != 0)
                set_label(v0->grad, tmp);
        }

        if (unlikely(edge.special)) {
            if (mode == ADMode::Forward)
                edge.special->forward(v0, v1);
            else
                edge.special->backward(v1, v0);

            if (!retain_graph) {
                // Edge may have been invalidated by callback, look up once more
                Edge &edge2 = state.edges[edge_ref.id];
                if (edge.source == edge_ref.source && edge.target == edge_ref.target) {
                    Special *special2 = edge2.special;
                    edge2.special = nullptr;
                    unlock_guard<std::mutex> guard2(state.mutex);
                    delete special2;
                }
            }
        } else {
            v1->mul_accum(v0->grad, edge.weight, v0->size);

            if (!retain_graph)
                edge.weight = Value();
        }
    }

    postprocess(v0i_prev, 0);

    if (!retain_graph) {
        ad_log(Debug, "ad_traverse(): clearing graph ..");

        std::sort(todo.begin(), todo.end(), [](EdgeRef e1, EdgeRef e2) {
            return std::tie(e1.target, e1.source) < std::tie(e2.target, e2.source);
        });

        todo.erase(
            std::unique(todo.begin(), todo.end(),
                        [](EdgeRef e1, EdgeRef e2) { return e1.target == e2.target; }),
            todo.end());

        for (EdgeRef edge_ref : todo) {
            auto it = state.variables.find(edge_ref.target);
            if (unlikely(it == state.variables.end()))
                continue;
            if (unlikely(mode == ADMode::Reverse && rec && !it.value().placeholder))
                continue;
            ad_free_edges(edge_ref.target, &it.value());
        }
    }

    for (uint32_t index : ls.visited)
        ad_dec_ref(index, state[index]);
    ls.visited.clear();

    ad_log(Debug, "ad_traverse(): done.");

    todo.clear();
    todo_tls.swap(todo);
}

// ==========================================================================
// Tracking of implicit dependencies. The following functions are used by
// the implementations of differentiable virtual function calls, to
//
// - detect when a virtual function call depends on extra inputs that aren't
//   function inputs (specifically, private instance variables)
//
// - extract that list of variables so that they can be properly attached
//   to the AD graph
//
// - avoid derivative propagation through parts of the graph
// ==========================================================================

template <typename Value> size_t ad_implicit() {
    return local_state.implicit.size();
}

template <typename Value> void ad_extract_implicit(size_t snapshot, uint32_t *out) {
    std::vector<EdgeRef> &implicit = local_state.implicit;
    size_t size = implicit.size();

    if (snapshot == size)
        return;
    else if (unlikely(snapshot > size))
        ad_raise("ad_extract_implicit(): invalid input arguments!");

    size_t count = size - snapshot;
    ad_trace("ad_extract_implicit(): extracting %zu implicit dependencies.",
             count);

    for (size_t i = 0; i < count; ++i) {
        uint32_t index = implicit[snapshot + i].source;
        if (state.variables.find(index) != state.variables.end())
            out[i] = index;
    }

    std::sort(out, out + count);
    uint32_t *ptr = std::unique(out, out + count);
    while (ptr != out + count)
        *ptr++ = 0;
}

template <typename Value> void ad_enqueue_implicit(size_t snapshot) {
    LocalState &ls = local_state;
    std::vector<EdgeRef> &implicit = local_state.implicit;
    size_t size = implicit.size();

    if (snapshot == size)
        return;
    else if (unlikely(snapshot > size))
        ad_raise("ad_enqueue_implicit(): invalid input arguments!");

    if (ls.visited.empty() && ls.todo.empty()) {
        ls.mode = ADMode::Forward;
    } else if (ls.mode != ADMode::Forward) {
        ad_raise("ad_enqueue_implicit(): attempted to enqueue nodes using "
                 "incompatible 'ADMode' values (i.e. both forward *and* "
                 "reverse-mode differentation)");
    }

    ad_trace("ad_enqueue_implicit(): enqueuing %zu implicit dependencies.",
             size - snapshot);

    std::lock_guard<std::mutex> guard(state.mutex);
    for (size_t i = snapshot; i < implicit.size(); ++i) {
        if (state.variables.find(implicit[i].target) == state.variables.end())
            continue;
        ls.todo.push_back(implicit[i]);
        ad_dfs_fwd(ls.visited, ls.todo, implicit[i].target);
        state[implicit[i].source]->ref_count_grad++;
    }
}

template <typename Value> void ad_dequeue_implicit(size_t snapshot) {
    std::vector<EdgeRef> &implicit = local_state.implicit;
    size_t size = implicit.size();

    if (snapshot == size)
        return;
    else if (unlikely(snapshot > size))
        ad_raise("ad_dequeue_implicit(): invalid input arguments!");

    ad_trace("ad_dequeue_implicit(): dequeuing %zu implicit dependencies.",
             size - snapshot);

    std::lock_guard<std::mutex> guard(state.mutex);
    for (size_t i = snapshot; i < implicit.size(); ++i) {
        state[implicit[i].source]->ref_count_grad--;
    }
}

template <typename Value> bool ad_enqueue_postponed() {
    if (is_jit_array_v<Value>) {
        LocalState &ls = local_state;

        if (jit_flag(JitFlag::Recording) || ls.postponed.empty())
            return false;

        // Use this opportunity to also clear the implicit dependency tracker
        ls.implicit.clear();

        if (ls.visited.empty() && ls.todo.empty()) {
            ls.mode = ADMode::Reverse;
        } else if (ls.mode != ADMode::Reverse) {
            ad_raise("ad_enqueue_postponed(): attempted to enqueue nodes using "
                     "incompatible 'ADMode' values (i.e. both forward *and* "
                     "reverse-mode differentation)");
        }

        ad_trace("ad_enqueue_postponed(): enqueuing %zu edges.",
                 ls.postponed.size());

        std::lock_guard<std::mutex> guard(state.mutex);
        for (EdgeRef e : ls.postponed) {
            auto [it, success] = ls.visited.insert(e.target);
            if (!success)
                ad_dec_ref(e.target, state[e.target]);
            ls.todo.push_back(e);
        }

        ls.postponed.clear();

        return true;
    } else {
        return false;
    }
}

// ==========================================================================
// Debugging: GraphViz, variable listing
// ==========================================================================

extern void RENAME(ad_whos)() {
    std::lock_guard<std::mutex> guard(state.mutex);

    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (auto &kv: state.variables)
        indices.push_back(kv.first);
    std::sort(indices.begin(), indices.end());

    for (uint32_t id : indices) {
        const Variable *v = state[id];
        buffer.fmt("  %-7i ", id);
        size_t sz = buffer.fmt("%u", v->ref_count);
        buffer.fmt("%*s%-12u%-8s\n", 11 - (int) sz, "", v->size,
                   v->label ? v->label : "");
    }
}

template <typename Value> const char *ad_graphviz() {
    std::lock_guard<std::mutex> guard(state.mutex);

    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (const auto& it : state.variables)
        indices.push_back(it.first);

    std::sort(indices.begin(), indices.end());
    buffer.clear();
    buffer.put("digraph {\n"
                   "    rankdir=BT;\n"
                   "    graph [dpi=50 fontname=Consolas];\n"
                   "    node [shape=record fontname=Consolas];\n"
                   "    edge [fontname=Consolas];\n");

    size_t current_hash = 0, current_depth = 1;

    for (uint32_t index : indices) {
        const Variable *v = state[index];
        const char *label = v->label;
        const char *label_without_prefix = label;

        size_t prefix_hash = 0;
        if (label) {
            const char *sep = strrchr(label, '/');
            if (sep) {
                prefix_hash = XXH3_64bits(label, sep - label);
                label_without_prefix = sep + 1;
            }
        }

        if (prefix_hash != current_hash) {
            for (size_t i = current_depth - 1; i > 0; --i) {
                buffer.putc(' ', 4 * i);
                buffer.put("}\n");
            }

            current_hash = prefix_hash;
            current_depth = 1;

            const char *p = label;
            while (true) {
                const char *pn = p ? strchr(p, '/') : nullptr;
                if (!pn)
                    break;

                buffer.putc(' ', 4 * current_depth);
                buffer.fmt("subgraph cluster_%08llx {\n",
                               (unsigned long long) XXH3_64bits(label, pn - label));
                current_depth++;
                buffer.putc(' ', 4 * current_depth);
                buffer.put("label=\"");
                buffer.put(p, pn - p);
                buffer.put("\";\n");
                buffer.putc(' ', 4 * current_depth);
                buffer.put("color=gray95;\n");
                buffer.putc(' ', 4 * current_depth);
                buffer.put("style=filled;\n");

                p = pn + 1;
            }
        }

        buffer.putc(' ', 4 * current_depth);
        buffer.put_uint32((uint32_t) index);
        buffer.put(" [label=\"{");

        auto print_escape = [](const char *s) {
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
                    buffer.putc('\\');
                buffer.putc(c);
            }
        };

        const char *color = nullptr;
        bool labeled = false;
        if (label_without_prefix && strlen(label_without_prefix) != 0) {
            if (v->custom_label) {
                buffer.put("Label: \\\"");
                labeled = true;
            }
            print_escape(label_without_prefix);
            if (v->custom_label)
                buffer.put("\\\"");
        }

        if (v->next_rev == 0)
            color = "salmon";
        else if (v->next_fwd == 0)
            color = "lightblue2";
        if (labeled && !color)
            color = "wheat";
        if (is_valid(v->grad))
            color = "yellowgreen";

        buffer.fmt("|{a%i|S:%u|R:%u%s}",
            index, v->size, v->ref_count,
            v->placeholder ? "|P" : "");

        buffer.put("}\"");
        if (color)
            buffer.fmt(" fillcolor=%s style=filled", color);
        buffer.put("];\n");
    }

    for (size_t i = current_depth - 1; i > 0; --i) {
        buffer.putc(' ', 4 * i);
        buffer.put("}\n");
    }

    for (uint32_t index : indices) {
        const Variable *v = state[index];

        uint32_t edge = v->next_rev, edge_count = 0;
        while (edge) {
            edge = state.edges[edge].next_rev;
            edge_count++;
        }
        edge = v->next_rev;
        uint32_t edge_ctr = edge_count;
        while (edge) {
            const Edge &e = state.edges[edge];
            if (edge_count == 1)
                buffer.fmt("    %i -> %i%s;\n", e.target, e.source,
                           e.special ? " [color=red]" : "");
            else
                buffer.fmt("    %i -> %i [label=\" %u\"%s];\n", e.target, e.source,
                           edge_ctr--, e.special ? " color=red" : "");
            edge = e.next_rev;
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
// Export declarations
// ==========================================================================

template ENOKI_EXPORT void ad_inc_ref_impl<Value>(uint32_t) noexcept;
template ENOKI_EXPORT void ad_dec_ref_impl<Value>(uint32_t) noexcept;
template ENOKI_EXPORT uint32_t ad_new<Value>(const char *, size_t, uint32_t,
                                            uint32_t *, Value *);
template ENOKI_EXPORT Value ad_grad<Value>(uint32_t, bool);
template ENOKI_EXPORT void ad_set_grad<Value>(uint32_t, const Value &, bool);
template ENOKI_EXPORT void ad_accum_grad<Value>(uint32_t, const Value &, bool);
template ENOKI_EXPORT void ad_set_label<Value>(uint32_t, const char *);
template ENOKI_EXPORT const char *ad_label<Value>(uint32_t);
template ENOKI_EXPORT void ad_enqueue<Value>(ADMode, uint32_t);
template ENOKI_EXPORT void ad_traverse<Value>(bool, bool);
template ENOKI_EXPORT size_t ad_implicit<Value>();
template ENOKI_EXPORT void ad_extract_implicit<Value>(size_t, uint32_t*);
template ENOKI_EXPORT void ad_enqueue_implicit<Value>(size_t);
template ENOKI_EXPORT void ad_dequeue_implicit<Value>(size_t);
template ENOKI_EXPORT bool ad_enqueue_postponed<Value>();
template ENOKI_EXPORT const char *ad_graphviz<Value>();
template ENOKI_EXPORT uint32_t ad_new_select<Value, Mask>(
    const char *, size_t, const Mask &, uint32_t, uint32_t);
template ENOKI_EXPORT uint32_t ad_new_gather<Value, Mask, Index>(
    const char *, size_t, uint32_t, const Index &, const Mask &, bool);
template ENOKI_EXPORT uint32_t
ad_new_scatter<Value, Mask, Index>(const char *, size_t, ReduceOp, uint32_t,
                                   uint32_t, const Index &, const Mask &, bool);
template ENOKI_EXPORT void ad_add_edge<Value>(uint32_t, uint32_t,
                                              DiffCallback *);
NAMESPACE_END(detail)

template struct ENOKI_EXPORT DiffArray<detail::Value>;

NAMESPACE_END(enoki)
