/** Dr.Jit automatic differentiation library
 *
 * This file implements the AD data structures and traversal routines
 * underlying templated Dr.Jit types like 'DiffArray<CUDAArray<float>>'. The
 * compilation process explicitly instantiates these templates for
 * scalar/LLVM/CUDA arrays in both single and double precision and merges them
 * into a shared library "drjit-autodiff.so/dll". In this way, the machinery
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
 * The combination with JITted (CUDA/LLVM) array types is quite interesting: in
 * this case, AD traversal generates code that can be executed at some later
 * point. While Dr.Jit's AD backend is principally tape-based, this combination
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
#include <drjit/jit.h>
#include <drjit/math.h>
#include <drjit/autodiff.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <assert.h>
#include <mutex>
#include <xxh3.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

// ==========================================================================
// Helper and forward declarations
// ==========================================================================

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

#define CONCAT(x,y) x ## _ ## y
#define EVAL(x,y) CONCAT(x,y)
#define RENAME(fun) EVAL(fun, DRJIT_AUTODIFF_NAME)

// Rename various things to avoid symbol clashes
#define Special              RENAME(Special)
#define Edge                 RENAME(Edge)
#define Variable             RENAME(Variable)
#define State                RENAME(State)
#define ReleaseQueueHelper   RENAME(ReleaseQueueHelper)
#define ReleaseOperandHelper RENAME(ReleaseOperandHelper)

using Value = DRJIT_AUTODIFF_VALUE;
using Mask = mask_t<Value>;
using Index = uint32_array_t<Value>;
using IndexSet = tsl::robin_set<uint32_t, UInt32Hasher>;

// Forward declarations
struct Variable;
struct Special;

static void ad_free(uint32_t index, Variable *v);
template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather_impl(const char *label, size_t size, uint32_t src_index,
                           const Index &offset, const Mask &mask, bool permute);

template <typename T> bool is_valid(const T &value) {
    if constexpr (is_jit_v<T>)
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
 * 'next_fwd' and 'next_bwd' indices each implement a singly linked list that
 * can be used to iterate through the forward edges (of the 'source' variable)
 * and backward edges (of the 'target' variable).
 */
struct Edge {
    /// Variable index of source operand
    uint32_t source;

    /// Source variable index
    uint32_t target;

    /// Links to the next forward edge
    uint32_t next_fwd;

    /// Links to the next backward edge
    uint32_t next_bwd : 31;

    /// Visited bit
    uint32_t visited : 1;

    /// Pointer to a handler for "special" edges
    Special *special;

    /// Weight value (zero/empty for "special" edges)
    Value weight{};

    DRJIT_ARRAY_DEFAULTS(Edge);

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
 * direction, is represented using linked lists. The 'next_fwd' and 'next_bwd'
 * fields each provide an entry point into such a linked list of edges (see
 * also \ref Edge).
 *
 * The 'placeholder' bit is used to track variables that were created while
 * recording a megakernel (i.e., inside a symbolic loop, virtual function call,
 * etc.).
 */
struct Variable {
    /// Number of references to this variable
    uint32_t ref_count;

    /// Links to the first forward edge at this node
    uint32_t next_fwd;

    /// Links to the first backward edge at this node
    uint32_t next_bwd;

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
                    v2 = sum(v);
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

        if constexpr (is_jit_v<T>) {
            if (v2_.is_literal() && std::isnormal(v2_[0]) &&
                jit_flag(JitFlag::ADOptimize)) {
                /* The check can be elided if the edge weight is a normal
                   literal constant. This can save significant amounts of
                   unnecessary eq() and select() operations in generated IR */
                v2 = v2_;
            } else {
                /* Only use this if absolutely necessary (also because it
                   triggers a forced evaluation in case any of the input
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
                    v3 = sum(v3);
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

    DRJIT_ARRAY_DEFAULTS(Variable);
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
                   "drjit-autodiff: variable leak detected (%zu variables "
                   "remain in use)!", variables.size());
            uint32_t counter = 0;
            for (auto kv : variables) {
                ad_log(Warn, " - variable a%u (%u references)", kv.first,
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
                   "drjit-autodiff: edge leak detected (%zu edges "
                   "remain in use)!", edges_used);
    }

    Variable *operator[](uint32_t index) {
        auto it = variables.find(index);
        if (unlikely(index == 0 || it == variables.end()))
            ad_fail("referenced an unknown variable a%u!", index);
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
    uint32_t variable_index = 0;

    /**
     * \brief Depending on the value of 'complement', this set specifies
     * variables for which AD is enabled or disabled.
     */
    IndexSet indices;

    /// List of AD postponed edges that will be traversed when leaving the scope
    std::vector<EdgeRef> postponed;

    Scope() = default;
    Scope(Scope&&) = default;
    Scope(const Scope&) = default;
    Scope& operator=(Scope&&) = default;
    Scope& operator=(const Scope&) = default;

    /// Check if a variable has gradients enabled
    bool enabled(uint32_t index) const {
        return (indices.find(index) != indices.end()) != complement;
    }

    /// Potentially zero out 'index' if the variable has gradients disabled
    bool maybe_disable(uint32_t &index) const {
        if (index && !enabled(index))
            index = 0;
        return index != 0;
    }

    /// Track gradients for the given variable
    void enable(uint32_t index) {
        if (!index)
            return;

        if (complement)
            indices.erase(index);
        else
            indices.insert(index);
    }

    /// Disable gradients for the given variable
    void disable(uint32_t index) {
        if (!index)
            return;

        if (complement)
            indices.insert(index);
        else
            indices.erase(index);
    }
};

// Special edge (scatter, gather, scatter_reduce, block_sum, etc.)
struct Special {
    virtual void backward(Variable * /* source */,
                          const Variable * /* target */,
                          uint32_t /* flags */) const {
        ad_fail("Special::backward(): not implemented!");
    }

    virtual void forward(const Variable * /* source */, Variable * /* target */,
                         uint32_t /* flags */) const {
        ad_fail("Special::forward(): not implemented!");
    }

    virtual ~Special() = default;
};

// Stores per-thread state
struct LocalState {
    /// Thread-local edge list used by ad_enqueue_*() and ad_traverse()
    std::vector<EdgeRef> todo;

    /// Keeps track of implicit input dependencies of recorded computation
    std::vector<EdgeRef> implicit;

    /// Nested scopes that restrict AD to specific variables
    std::vector<Scope> scopes;

    /// List of special edges that should be cleaned up
    std::vector<Special *> cleanup;

    ~LocalState() {
        for (Special *s : cleanup)
            delete s;

        if (!scopes.empty())
            ad_log(Warn,
                   "drjit-autodiff: scope leak detected (%zu scopes "
                   "remain in use)!", scopes.size());
    }
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
    memset(this, 0, sizeof(uint32_t) * 4 + sizeof(Special *));
    weight = Value();
}

// ==========================================================================
// Reference counting and variable cleanup
// ==========================================================================

static void ad_inc_ref(uint32_t index, Variable *v) noexcept (true) {
    DRJIT_MARK_USED(index);
    ad_trace("ad_inc_ref(a%u): %u", index, v->ref_count + 1);
    v->ref_count++;
}

static bool ad_dec_ref(uint32_t index, Variable *v) noexcept (true) {
    DRJIT_MARK_USED(index);
    ad_trace("ad_dec_ref(a%u): %u", index, v->ref_count - 1);

    if (unlikely(v->ref_count == 0))
        ad_fail("drjit-autodiff: fatal error: external reference count of "
                "variable a%u became negative!", index);

    if (--v->ref_count > 0) {
        return false;
    } else {
        ad_free(index, v);
        return true;
    }
}

template <typename T> void ad_inc_ref_impl(uint32_t index) noexcept(true) {
    if (likely(index == 0))
        return;
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_inc_ref(index, state[index]);
}

template <typename T> uint32_t ad_inc_ref_cond_impl(uint32_t index) noexcept(true) {
    if (likely(index == 0))
        return 0;

    auto const &scopes = local_state.scopes;
    if (!scopes.empty()) {
        scopes.back().maybe_disable(index);
        if (index == 0)
            return 0;
    }

    std::lock_guard<std::mutex> guard(state.mutex);
    ad_inc_ref(index, state[index]);
    return index;
}

template <typename T> void ad_dec_ref_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    std::lock_guard<std::mutex> guard(state.mutex);

    if (unlikely(ad_dec_ref(index, state[index]))) {
        /* Extra-careful here: deallocate cleanup queue of
           custom AD edge callbacks (reentrant!) */

        std::vector<Special *> temp, &cleanup = local_state.cleanup;

        if (!cleanup.empty()) {
            temp.swap(cleanup);
            for (Special *special : temp)
                delete special;
            temp.clear();
            temp.swap(cleanup);
        }
    }
}

static void ad_free(uint32_t index, Variable *v) {
    ad_trace("ad_free(a%u)", index);

    if (v->free_label) {
        free(v->label);
        v->label = nullptr;
    }

    uint32_t edge_id = v->next_bwd;
    v->next_bwd = 0;

    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        ad_trace("ad_free(): freeing edge a%u -> a%u", edge.source,
                 edge.target);

        if (unlikely(edge.target != index))
            ad_fail("ad_free(): invalid edge connectivity!");

        uint32_t source = edge.source,
                 next_bwd = edge.next_bwd,
                 next_fwd = edge.next_fwd;

        assert(edge.target == index);

        // Postpone deallocation of the edge callback, if there is one
        if (unlikely(edge.special))
            local_state.cleanup.push_back(edge.special);
        edge.reset();

        Variable *v2 = state[source];

        if (unlikely(v2->ref_count == 0))
            ad_fail("drjit-autodiff: fatal error: reference count of variable "
                    "a%u became negative!", source);

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

        edge_id = next_bwd;
    }

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
        if (is_jit_v<Value>)
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
    uint32_t edge = v->next_bwd;
    while (edge) {
        Edge &e = state.edges[edge];
        Variable *v2 = state[e.source];
        if (v2->placeholder && v2->size != v->size && v2->size == 1) {
            v2->size = v->size;
            ad_propagate_placeholder_size(v2);
        }
        edge = e.next_bwd;
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
void ad_scope_enter(ADScope type, size_t size, const uint32_t *indices) {
    auto &scopes = local_state.scopes;
    Scope scope;

    if (!scopes.empty())
        scope = scopes.back();

    scope.postponed.clear();
    scope.type = type;

    switch (type) {
        case ADScope::Suspend:
            ad_log(Debug, "ad_scope_enter(suspend, %zu indices)", size);

            if (size) {
                for (size_t i = 0; i < size; ++i)
                    scope.disable(indices[i]);
            } else {
                scope.complement = false;
                scope.indices.clear();
            }
            break;

        case ADScope::Resume:
            ad_log(Debug, "ad_scope_enter(resume, %zu indices)", size);

            if (size) {
                for (size_t i = 0; i < size; ++i)
                    scope.enable(indices[i]);
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
                scope.variable_index = state.variable_index;
            }
            ad_log(Debug, "ad_scope_enter(isolate, a%u...)", scope.variable_index);
            break;

        default:
            ad_fail("ad_scope_enter(): unknown scope type!");
    }

    scopes.push_back(std::move(scope));
}

template <typename T> void ad_scope_leave(bool process_postponed) {
    LocalState &ls = local_state;
    auto &scopes = local_state.scopes;
    if (scopes.empty())
        ad_raise("ad_scope_leave(): underflow!");

    Scope &scope = scopes.back();

    const char *type_name = nullptr;

    switch (scope.type) {
        case ADScope::Suspend: type_name = "suspend"; break;
        case ADScope::Resume:  type_name = "resume"; break;
        case ADScope::Isolate: type_name = "isolate"; break;
        default: type_name = "unknown"; break;
    }

    ad_log(Debug, "ad_scope_leave(%s)", type_name);

    if (scope.isolate && !scope.postponed.empty()) {
        // Need to process postponed edges now..
        if (unlikely(!ls.todo.empty()))
            ad_raise("ad_scope_leave(): internal error: wanted to process "
                     "postponed AD edges, but other edges were already "
                     "enqueued. Did you forget to call dr.traverse() to "
                     "process them?");

        if (process_postponed) {
            ad_trace("ad_scope_leave(): enqueuing %zu postponed edges.",
                     scope.postponed.size());

            ls.todo.insert(ls.todo.end(), scope.postponed.begin(),
                           scope.postponed.end());

            scopes.pop_back();

            ad_traverse<Value>(ADMode::Backward,
                               (uint32_t) ADFlag::ClearVertices);
        } else {
            scopes.pop_back();
        }
    } else {
        scopes.pop_back();
    }

    // Use this opportunity to also clear the implicit dependency tracker
    // ls.implicit.clear();
}

template <typename T> bool ad_grad_enabled(uint32_t index) {
    auto const &scopes = local_state.scopes;
    if (!scopes.empty())
        scopes.back().maybe_disable(index);
    return index != 0;
}

template <typename T> bool ad_enabled() noexcept(true) {
    auto const &scopes = local_state.scopes;

    if (!scopes.empty()) {
        const Scope &scope = scopes.back();

        // Check if AD is disabled on the current thread
        if (!scope.complement && scope.indices.empty())
            return false;
    }

    std::lock_guard<std::mutex> guard(state.mutex);
    return !state.variables.empty();
}

template <typename T>
uint32_t ad_new(const char *label, size_t size, uint32_t op_count,
                uint32_t *op, T *weights) {
    std::lock_guard<std::mutex> guard(state.mutex);

    /* Potentially turn off derivative tracking for some of the operands if
       we're within a scope that enables/disables gradient propagation
       (globally, or only for specific variables) */
    std::vector<Scope> &scopes = local_state.scopes;
    if (unlikely(!scopes.empty())) {
        const Scope &scope = scopes.back();

        bool active = false;
        if (op_count == 0) {
            // If AD is completely disabled (i.e. this is an dr.suspend_grad()
            // region), don't allow creating new AD variables
            active = scope.complement || !scope.indices.empty();
        } else {
            for (uint32_t i = 0; i < op_count; ++i)
                active |= scope.maybe_disable(op[i]);
        }

        if (!active)
            return 0;
    }

    bool rec = false;
    if constexpr (is_jit_v<Value>)
        rec = jit_flag(JitFlag::Recording);

    ReleaseOperandHelper helper;
    if (unlikely(rec)) {
        for (uint32_t i = 0; i < op_count; ++i) {
            if (op[i] == 0)
                continue;

            uint32_t index = op[i];
            const Variable *var = state[index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(!var->placeholder)) {
                if (var->size != 1)
                    ad_raise(
                        "ad_new(): recorded computation performs an implicit "
                        "read of variable (a%u), which has size %u! However, "
                        "only scalar (size == 1) accesses are permitted in "
                        "this manner. You will likely want to convert the "
                        "read into an drjit::gather() operation.",
                        index, var->size);

                ad_trace("ad_new(): implicit read of variable a%u, inserting a "
                         "gather operation..", op[i]);

                index = ad_new_gather_impl<Value>("gather", size, op[i], Index(0),
                                                  Mask(true), false);

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
                ad_log(Debug, "ad_new(a%u, size=%zu%s%s)", index, size, l1, l2); break;
            case 1:
                ad_log(Debug, "ad_new(a%u <- a%u, size=%zu%s%s)", index, op[0], size, l1, l2); break;
            case 2:
                ad_log(Debug, "ad_new(a%u <- a%u, a%u, size=%zu%s%s)", index, op[0], op[1], size, l1, l2); break;
            case 3:
                ad_log(Debug, "ad_new(a%u <- a%u, a%u, a%u, size=%zu%s%s)", index, op[0], op[1], op[2], size, l1, l2); break;
            default: break;
        }
    }

    uint32_t edge_index = 0;
    for (uint32_t i = 0; i < op_count; ++i) {
        if (op[i] == 0)
            continue;

        bool weight_is_zero = false;
        if constexpr (is_jit_v<T>)
            weight_is_zero = jit_flag(JitFlag::ADOptimize) &&
                             weights[i].is_literal() && weights[i][0] == 0;
        else
            weight_is_zero = weights[i] == 0;

        if (weight_is_zero) {
            ad_trace(
                "ad_new(a%u <- a%u): weight of edge %i is zero, skipping!",
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
        edge.next_bwd = edge_index;
        edge_index = edge_index_new;

        ad_inc_ref(index2, var2);
        var2->next_fwd = edge_index_new;
    }

    if (op_count > 0 && edge_index == 0) {
        // All edges were pruned, don't create the node after all
        ad_trace(
            "ad_new(a%u): all nodes were pruned, removing variable from graph", index);
        ad_free(index, var);
        return 0;
    }

    var->next_bwd = edge_index;
    var->ref_count = 1;

    if (var->placeholder)
        ad_propagate_placeholder_size(var);

    /* If we're selectively tracking gradients and this operation generates a
       new AD variable, then its index must be added to the index set */
    if (unlikely(!scopes.empty()))
        scopes.back().enable(index);

    return index;
}

// ==========================================================================
// Implementation of AD for special operations: masks, gathers, scatters
// ==========================================================================

template <typename Value> struct MaskEdge : Special {
    MaskEdge(const Mask &mask, bool negate) : mask(mask), negate(negate) { }

    void backward(Variable *source, const Variable *target, uint32_t) const override {
        source->accum(!negate ? detail::and_(target->grad, mask)
                              : detail::andnot_(target->grad, mask),
                      target->size);
    }

    void forward(const Variable *source, Variable *target, uint32_t) const override {
        target->accum(!negate ? detail::and_(source->grad, mask)
                              : detail::andnot_(source->grad, mask),
                      source->size);
    }

    Mask mask;
    bool negate;
};

template <typename Value> struct SpecialConnection : Special {
    void backward(Variable *, const Variable *target, uint32_t) const override {
        if (target->size)
            const_cast<Variable *>(target)->ref_count_grad++;
    }

    void forward(const Variable *source, Variable *, uint32_t) const override {
        if (source->size)
            const_cast<Variable *>(source)->ref_count_grad++;
    }
};

template <typename Value> struct SpecialCallback : Special {
    std::unique_ptr<DiffCallback> callback;
    Scope scope;
    bool clear;

    /// Recreate the suspend/resume status in place when this callback edge was created
    struct PushScope {
        PushScope(const Scope &scope) {
            auto &scopes = local_state.scopes;

            if (!scopes.empty()) {
                bool isolate = scopes.back().isolate;
                scopes.push_back(scope);
                scopes.back().isolate = isolate;
            } else {
                scopes.push_back(scope);
            }

            scopes.back().postponed.clear();
        }

        ~PushScope() {
            auto &scopes = local_state.scopes;
            if (scopes.size() > 1) {
                Scope &scope_child  = scopes[scopes.size() - 1];
                Scope &scope_parent = scopes[scopes.size() - 2];

                if (unlikely(scope_child.isolate == scope_parent.isolate)) {
                    scope_parent.postponed.insert(
                        scope_parent.postponed.end(),
                        scope_child.postponed.begin(),
                        scope_child.postponed.end()
                    );
                }
            } else if (scopes.size() == 0) {
                ad_fail("SpecialCallback::PushScope::~PushScope(): underflow!");
            }

            scopes.pop_back();
        }
    };

    virtual ~SpecialCallback() {
        /* outside of critical section */ {
            unlock_guard<std::mutex> guard(state.mutex);
            callback.reset();
        }
    }

    SpecialCallback(DiffCallback *callback, Scope &&scope)
        : callback(callback), scope(std::move(scope)) { }

    void backward(Variable *, const Variable *target, uint32_t flags) const override {
        ad_trace("ad_traverse(): invoking user callback ..");
        uint32_t edge = target->next_fwd;

        /* leave critical section */ {
            unlock_guard<std::mutex> guard(state.mutex);
            PushScope push(scope);
            callback->backward();
        }
        if (edge && state.edges[edge].next_fwd) { // fan-in > 1, update ref counts
            do {
                const Edge &e = state.edges[edge];
                Variable *v = state[e.target];

                if (v->ref_count_grad > 0 && --v->ref_count_grad == 0) {
                    if (((flags & (uint32_t) ADFlag::ClearInterior) && v->next_fwd != 0) ||
                        ((flags & (uint32_t) ADFlag::ClearInput) && v->next_fwd == 0))
                        v->grad = Value();
                }
                edge = e.next_fwd;
            } while (edge);
        }
    }

    void forward(const Variable *source, Variable *, uint32_t flags) const override {
        ad_trace("ad_traverse(): invoking user callback ..");
        uint32_t edge = source->next_bwd;
        /* leave critical section */ {
            unlock_guard<std::mutex> guard(state.mutex);
            PushScope push(scope);
            callback->forward();
        }
        if (edge && state.edges[edge].next_bwd) { // fan-in > 1, update ref counts
            do {
                const Edge &e = state.edges[edge];
                Variable *v = state[e.source];


                if (v->ref_count_grad > 0 && --v->ref_count_grad == 0) {
                    if (((flags & (uint32_t) ADFlag::ClearInterior) && v->next_bwd != 0) ||
                        ((flags & (uint32_t) ADFlag::ClearInput) && v->next_bwd == 0)) {

                        if (!(scope.isolate && e.source < scope.variable_index))
                            v->grad = Value();
                    }
                }

                edge = e.next_bwd;
            } while (edge);
        }
    }
};

template <typename Value, typename Mask>
uint32_t ad_new_select(const char *label, size_t size, const Mask &mask,
                       uint32_t t_index, uint32_t f_index) {
    std::lock_guard<std::mutex> guard(state.mutex);
    if constexpr (is_jit_v<Mask>) {
        if (jit_flag(JitFlag::ADOptimize) && mask.is_literal()) {
            uint32_t result = mask[0] ? t_index : f_index;
            if (result)
                ad_inc_ref(result, state[result]);
            ad_log(Debug, "ad_new_select(a%u <- a%u, a%u): simplified", result, t_index, f_index);
            return result;
        }

        if (jit_flag(JitFlag::ADOptimize) && f_index == t_index) {
            if (t_index)
                ad_inc_ref(t_index, state[t_index]);
            ad_log(Debug, "ad_new_select(a%u <- a%u, a%u): simplified", t_index, t_index, f_index);
            return t_index;
        }
    }

    /* Potentially turn off derivative tracking for some of the operands if
       we're within a scope that enables/disables gradient propagation
       (globally, or only for specific variables) */
    std::vector<Scope> &scopes = local_state.scopes;
    if (unlikely(!scopes.empty())) {
        const Scope &scope = scopes.back();

        bool active = scope.maybe_disable(t_index);
        active |= scope.maybe_disable(f_index);

        if (!active)
            return 0;
    }

    bool rec = false;
    if constexpr (is_jit_v<Value>)
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
                        "implicit read of variable (a%u), which has size %u! "
                        "However, only scalar (size == 1) accesses are "
                        "permitted in this manner. You will likely want to "
                        "convert the read into an drjit::gather() operation.",
                        index, var->size);

                ad_trace("ad_new_select(): implicit read of variable a%u, inserting a "
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

    ad_log(Debug, "ad_new_select(a%u <- a%u, a%u)", index, t_index, f_index);
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
        edge.next_bwd = edge_index;
        edge_index = edge_index_new;

        ad_inc_ref(index2, var2);

        var2->next_fwd = edge_index_new;
    }

    var->next_bwd = edge_index;
    var->ref_count = 1;

    if (var->placeholder)
        ad_propagate_placeholder_size(var);

    /* If we're selectively tracking gradients and this operation generates a
       new AD variable, then its index must be added to the index set */
    if (unlikely(!scopes.empty()))
        scopes.back().enable(index);

    return index;
}


template <typename Mask> struct MaskGuard {
    MaskGuard(const Mask &mask) {
        if constexpr (is_jit_v<Mask>)
            jit_var_mask_push(Mask::Backend, mask.index());
        else
            DRJIT_MARK_USED(mask);
    }

    ~MaskGuard() {
        if constexpr (is_jit_v<Value>)
            jit_var_mask_pop(Mask::Backend);
    }
};


template <typename Value> struct GatherEdge : Special {
    GatherEdge(const Index &offset, const Mask &mask, bool permute)
        : offset(offset), mask(mask), permute(permute) {
        if constexpr (is_jit_v<Value>) {
            uint32_t mask_idx = jit_var_mask_peek(Value::Backend);
            if (!mask_idx)
                mask_idx = jit_var_mask_default(Value::Backend, (uint32_t) width(offset, mask));
            mask_stack = mask_t<Value>::steal(mask_idx);
        }
    }

    void backward(Variable *source, const Variable *target, uint32_t) const override {
        Value &source_grad = (Value &) source->grad;
        uint32_t size = source->size;

        if (source->size == 1 && target->size == 1 && !target->placeholder) {
            // Downgrade to scalar op
            source->accum(select(mask, target->grad, 0.f), 1);
            return;
        }

        if (!source_grad.valid())
            source_grad = zeros<Value>(size);
        else if ((uint32_t) source_grad.size() != size)
            source_grad.resize(size);

        MaskGuard guard(mask_stack);
        if (permute)
            scatter(source_grad, target->grad, offset, mask);
        else
            scatter_reduce(ReduceOp::Add, source_grad, target->grad, offset, mask);
    }

    void forward(const Variable *source, Variable *target, uint32_t) const override {
        MaskGuard guard(mask_stack);
        target->accum(gather<Value>(source->grad, offset, mask),
                      (uint32_t) width(offset));
    }

    Index offset;
    Mask mask;
    Mask mask_stack;
    bool permute;
};

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather_impl(const char *label, size_t size, uint32_t src_index,
                            const Index &offset, const Mask &mask_, bool permute) {
    Mask mask(mask_);

    if constexpr (is_array_v<Value>) {
        if (is_jit_v<Value>) {
            // Apply the mask stack (needed for wavefront-mode dr::Loop)
            Mask top = Mask::steal(jit_var_mask_peek(Mask::Backend));
            size_t tsize = top.size();
            if (tsize != 1 && tsize == size)
                mask &= top;
        }

        /* Potentially turn off derivative tracking for some of the operands if
           we're within a scope that enables/disables gradient propagation
           (globally, or only for specific variables) */
        std::vector<Scope> &scopes = local_state.scopes;
        if (unlikely(!scopes.empty())) {
            if (!scopes.back().maybe_disable(src_index))
                return 0;
        }

        auto [index, var] = ad_var_new(label, size);

        ad_log(Debug, "ad_new_gather(a%u <- a%u, size=%zu, permute=%i)", index,
               src_index, size, (int) permute);

        Variable *var2 = state[src_index];

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = src_index;
        edge.target = index;
        edge.special = new GatherEdge<Value>(offset, mask, permute);
        edge.next_fwd = var2->next_fwd;
        edge.next_bwd = 0;
        ad_inc_ref(src_index, var2);
        var2->next_fwd = edge_index_new;
        var->next_bwd = edge_index_new;
        var->ref_count = 1;

        /* Encountered a dependency between recorded/non-recorded computation
           that will need special handling when the AD graph is traversed at
           some later point. For now, just keep track of this event. */
        if (unlikely(var->placeholder && !var2->placeholder)) {
            ad_trace("ad_new_gather(): recording an implicit dependency (a%u).", src_index);
            local_state.implicit.emplace_back(edge_index_new, src_index, index);
        }

        /* If we're selectively tracking gradients and this operation generates a
           new AD variable, then its index must be added to the index set */
        if (unlikely(!scopes.empty()))
            scopes.back().enable(index);

        return index;
    } else {
        (void) mask; (void) label; (void) size;
        (void) src_index; (void) offset; (void) permute;
        drjit_raise("ad_new_gather(): differentiable gathers not supported by "
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
                drjit_raise("AD only supports ReduceOp::Add in scatter_reduce!");
        if constexpr (is_jit_v<Value>) {
            uint32_t mask_idx = jit_var_mask_peek(Value::Backend);
            if (!mask_idx)
                mask_idx = jit_var_mask_default(Value::Backend, (uint32_t) width(offset, mask));
            mask_stack = mask_t<Value>::steal(mask_idx);
        }
    }

    void backward(Variable *source, const Variable *target, uint32_t) const override {
        MaskGuard guard(mask_stack);
        source->accum(gather<Value>(target->grad, offset, mask),
                      (uint32_t) width(offset));
    }

    void forward(const Variable *source, Variable *target, uint32_t) const override {
        Value &target_grad = (Value &) target->grad;
        uint32_t size = target->size;

        if (!target_grad.valid())
            target_grad = zeros<Value>(size);
        else if ((uint32_t) target_grad.size() != size)
            target_grad.resize(size);

        MaskGuard guard(mask_stack);
        if (op != ReduceOp::None)
            scatter_reduce(op, target_grad, source->grad, offset, mask);
        else
            scatter(target_grad, source->grad, offset, mask);
    }

    Index offset;
    Mask mask;
    Mask mask_stack;
    ReduceOp op;
};

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_scatter(const char *label, size_t size, ReduceOp op,
                        uint32_t src_index, uint32_t dst_index, const Index &offset,
                        const Mask &mask_, bool permute) {

    Mask mask(mask_);
    DRJIT_MARK_USED(mask);

    if constexpr (is_array_v<Value>) {
        std::lock_guard<std::mutex> guard(state.mutex);

        if (is_jit_v<Value>) {
            // Apply the mask stack (needed for wavefront-mode dr::Loop)
            Mask top = Mask::steal(jit_var_mask_peek(Mask::Backend));
            size_t tsize = top.size(),
                   ssize = (size_t)(src_index ? state[src_index]->size : 0);
            ssize = std::max(std::max(ssize, offset.size()), mask_.size());
            if (tsize != 1 && tsize == ssize)
                mask &= top;
        }

        /* Potentially turn off derivative tracking for some of the operands if
           we're within a scope that enables/disables gradient propagation
           (globally, or only for specific variables) */
        std::vector<Scope> &scopes = local_state.scopes;
        if (unlikely(!scopes.empty())) {
            const Scope &scope = scopes.back();
            bool active = scope.maybe_disable(src_index);
            active |= scope.maybe_disable(dst_index);
            if (!active)
                return 0;
        }

        auto [index, var] = ad_var_new(label, size);

        ad_log(Debug,
               "ad_new_scatter(op=%i, a%u <- a%u, a%u, permute=%i)",
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
            edge.next_bwd = var->next_bwd;
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
            edge2.next_bwd = edge_index;
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

        var->next_bwd = edge_index;
        ad_inc_ref(index, var);

        /* If we're selectively tracking gradients and this operation generates a
           new AD variable, then its index must be added to the index set */
        if (unlikely(!scopes.empty()))
            scopes.back().enable(index);

        return index;
    } else {
        (void) label; (void) size; (void) op; (void) src_index;
        (void) dst_index; (void) offset; (void) permute;
        drjit_raise("ad_new_scatter(): differentiable scatters not supported "
                    "by this backend!");
    }
}

// ==========================================================================
// Interface for querying and modifying variables
// ==========================================================================

template <typename T> T ad_grad(uint32_t index, bool fail_if_missing) {
    auto const &scopes = local_state.scopes;
    if (unlikely(!scopes.empty()))
        scopes.back().maybe_disable(index);
    if (unlikely(index == 0))
        return T(0);

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_grad(): referenced an unknown variable a%u!", index);
        return T(0);
    }

    const Variable &v = it->second;
    T result = v.grad;

    if constexpr (is_jit_v<T>) {
        if (!is_valid(result))
            result = zeros<T>(v.size);
        else if (result.size() != v.size)
            result.resize(v.size);
    }

    return result;
}

template <typename T>
void ad_set_grad(uint32_t index, const T &value, bool fail_if_missing) {
    auto const &scopes = local_state.scopes;
    if (unlikely(!scopes.empty()))
        scopes.back().maybe_disable(index);
    if (unlikely(index == 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_set_grad(): referenced an unknown variable a%u!", index);
        return;
    }

    size_t size_in = width(value);
    Variable &v = it.value();

    if (v.size != size_in && size_in != 1 && v.size != 1)
        ad_raise("ad_set_grad(): attempted to assign a gradient of size "
                 "%zu to AD variable a%u, which has size %u!",
                 size_in, index, v.size);

    ad_trace("ad_set_grad(a%u)", index);
    if (v.size != 1 || size_in == 1)
        v.grad = value;
    else
        v.grad = sum(value);
}

template <typename T>
void ad_accum_grad(uint32_t index, const T &value, bool fail_if_missing) {
    auto const &scopes = local_state.scopes;
    if (unlikely(!scopes.empty()))
        scopes.back().maybe_disable(index);
    if (unlikely(index == 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_accum_grad(): referenced an unknown variable a%u!", index);
        return;
    }

    size_t size_in = width(value);
    Variable &v = it.value();

    if (v.size != size_in && size_in != 1 && v.size != 1)
        ad_raise("ad_accum_grad(): attempted to accumulate a gradient of size "
                 "%zu into AD variable a%u, which has size %u!",
                 size_in, index, v.size);

    ad_trace("ad_accum_grad(a%u)", index);
    v.accum(value, (uint32_t) size_in);
}

template <typename T> void ad_set_label(uint32_t index, const char *label) {
    if (index == 0)
        return;
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_log(Debug, "ad_set_label(a%u, \"%s\")", index, label ? label : "(null)");
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

    /* Potentially turn off derivative tracking for some of the operands if
       we're within a scope that enables/disables gradient propagation
       (globally, or only for specific variables) */
    std::vector<Scope> &scopes = local_state.scopes;
    Scope scope;

    if (unlikely(!scopes.empty())) {
        scope = scopes.back();
        (void) scope.maybe_disable(source_idx);
        (void) scope.maybe_disable(target_idx);
    }

    if (source_idx == 0 || target_idx == 0)
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    ad_log(Debug, "ad_add_edge(a%u -> a%u)", source_idx, target_idx);
    assert(source_idx < target_idx);

    Variable *source = state[source_idx],
             *target = state[target_idx];

    uint32_t edge_index_new = ad_edge_new();
    Edge &edge = state.edges[edge_index_new];
    edge.source = source_idx;
    edge.target = target_idx;

    if (callback)
        edge.special = new SpecialCallback<Value>(callback, std::move(scope));
    else
        edge.special = new SpecialConnection<Value>();

    edge.next_fwd = source->next_fwd;
    edge.next_bwd = target->next_bwd;

    source->next_fwd = edge_index_new;
    target->next_bwd = edge_index_new;
    ad_inc_ref(source_idx, source);
}


// ==========================================================================
// Enqueuing of variables and edges
// ==========================================================================

/// Forward-mode DFS starting from 'index'
static void ad_dfs_fwd(std::vector<EdgeRef> &todo, uint32_t index, Variable *v) {
    DRJIT_MARK_USED(index);

    uint32_t edge_id = v->next_fwd;
    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        if (!edge.visited) {
            edge.visited = 1;

            ad_trace("ad_dfs_fwd(): enqueuing edge a%u -> a%u", index,
                     edge.target);

            Variable *v2 = state[edge.target];
            ad_inc_ref(edge.target, v2);
            todo.emplace_back(edge_id, edge.source, edge.target);
            ad_dfs_fwd(todo, edge.target, v2);
        }

        edge_id = edge.next_fwd;
    }
}

/// Reverse-mode DFS starting from 'index'
static void ad_dfs_bwd(std::vector<EdgeRef> &todo, uint32_t index, Variable *v) {
    uint32_t edge_id = v->next_bwd;
    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        if (!edge.visited) {
            edge.visited = 1;

            ad_trace("ad_dfs_bwd(): enqueuing edge a%u -> a%u", index,
                     edge.source);

            Variable *v2 = state[edge.source];
            ad_inc_ref(index, v);
            todo.emplace_back(edge_id, edge.source, edge.target);
            ad_dfs_bwd(todo, edge.source, v2);
        }

        edge_id = edge.next_bwd;
    }
}

template <typename T> void ad_enqueue(ADMode mode, uint32_t index) {
    if (index == 0)
        return;

    ad_trace("ad_enqueue_node(a%u, mode=%s)", index,
             mode == ADMode::Forward ? "forward" : "backward");

    LocalState &ls = local_state;

    std::lock_guard<std::mutex> guard(state.mutex);
    switch (mode) {
        case ADMode::Forward:
            ad_dfs_fwd(ls.todo, index, state[index]);
            break;

        case ADMode::Backward:
            ad_dfs_bwd(ls.todo, index, state[index]);
            break;

        default:
            ad_raise("ad_enqueue(): invalid mode specified!");
    }
}

// ==========================================================================
// AD graph traversal
// ==========================================================================

template <typename Value>
void ad_traverse(ADMode mode, uint32_t flags) {
    LocalState &ls = local_state;

    std::vector<EdgeRef> &todo_tls = ls.todo, todo;
    if (todo_tls.empty())
        return;

    // Are we currently recording a megakernel?
    bool rec = false;
    if (is_jit_v<Value>)
        rec = jit_flag(JitFlag::Recording);
    DRJIT_MARK_USED(rec);

    std::lock_guard<std::mutex> guard(state.mutex);
    todo_tls.swap(todo);

    if (mode != ADMode::Forward && mode != ADMode::Backward)
        ad_raise("ad_traverse(): invalid mode specified!");

    // Bring the edges into the appropriate order
    std::sort(todo.begin(), todo.end(), [mode](EdgeRef e1, EdgeRef e2) {
        if (mode == ADMode::Backward)
            return std::tie(e1.target, e1.source, e1.id) >
                   std::tie(e2.target, e2.source, e2.id);
        else
            return std::tie(e1.source, e1.target, e1.id) <
                   std::tie(e2.source, e2.target, e2.id);
    });

    ad_log(Debug, "ad_traverse(): processing %zu edges in %s mode ..", todo.size(),
           mode == ADMode::Forward ? "forward" : "backward");

    /// Any edges with an ID less than this value will be postponed
    uint32_t postpone_before = 0;
    if (!ls.scopes.empty() && ls.scopes.back().isolate)
        postpone_before = ls.scopes.back().variable_index;

    std::vector<Value> dr_loop_todo;
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
            schedule(prev->grad);

            if (!dr_loop_cur) {
                ad_trace("ad_traverse(): evaluating %zi loop variables",
                         dr_loop_todo.size());
                eval();
                dr_loop_todo.clear();
            }
        }

        bool clear_grad = false;
        uint32_t next_edge =
            mode == ADMode::Forward ? prev->next_bwd : prev->next_fwd;

        if (flags & (uint32_t) ADFlag::ClearInterior)
            clear_grad |= next_edge != 0;
        if (flags & (uint32_t) ADFlag::ClearInput)
            clear_grad |= next_edge == 0;

        /* Don't clear the gradient of vertices created *before* entering
           an dr.isolation() scope, or when their gradient is still explicitly
           referenced by some other part of the computation graph */
        if (prev_i < postpone_before || prev->ref_count_grad > 0)
            clear_grad = false;

        // Aggressively clear gradients at intermediate nodes
        if (clear_grad) {
            ad_trace("ad_traverse(): clearing gradient at intermediate variable a%u", prev_i);
            prev->grad = Value();
        }
    };

    uint32_t v0i_prev = 0;
    uint32_t last_edge_id = 0;

    // This is the main AD traversal loop
    for (EdgeRef &er : todo) {
        Edge &edge = state.edges[er.id];

        uint32_t v0i, v1i;
        if (mode == ADMode::Forward) {
            v0i = edge.source;
            v1i = edge.target;
        } else {
            v0i = edge.target;
            v1i = edge.source;
        }

        if (unlikely(er.id == last_edge_id))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u was "
                    "enqueued twice!", v0i, v1i);
        last_edge_id = er.id;

        if (unlikely(!edge.visited))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u is not "
                    "marked as visited! (1)", er.source, er.target);

        if (unlikely(edge.source != er.source || edge.target != er.target))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u was "
                    "garbage collected between enqueuing and traversal steps!",
                    v0i, v1i);

        Variable *v0 = state[v0i],
                 *v1 = state[v1i];

        uint32_t grad_size = (uint32_t) width(v0->grad);

        if (unlikely(v0i < postpone_before)) {
            if (mode == ADMode::Backward) {
                ad_trace("ad_traverse(): postponing edge a%u -> a%u due "
                         "dr.isolate_grad() scope.", v0i, v1i);
                ls.scopes.back().postponed.push_back(er);
                er.id = er.source = er.target = 0;
                continue;
            } else if (v1i < postpone_before) {
                ad_raise(
                    "ad_traverse(): tried to forward-propagate derivatives "
                    "across edge a%u -> a%u, which lies outside of the current "
                    "dr.isolate_grad() scope (a%u .. a%u). You must enqueue "
                    "the variables being differentiated and call "
                    "dr.traverse(dr.ADFlag.ClearEdges) *before* entering this "
                    "scope.",
                    v0i, v1i, postpone_before, state.variable_index);
            }
        }

        if (unlikely(grad_size != 1 && v0->size != grad_size)) {
            if (grad_size == 0) {
                ad_trace("ad_traverse(): skipping edge a%u -> a%u (no source "
                         "gradient).", v0i, v1i);
                continue;
            } else {
                ad_raise("ad_traverse(): gradient propagation encountered "
                         "variable a%u (\"%s\") with an invalid gradient size "
                         "(expected size %u, actual size %u)!",
                         v0i, v0->label ? v0->label : "", v0->size, grad_size);
            }
        }

        postprocess(v0i_prev, v0i);
        v0i_prev = v0i;

        ad_trace("ad_traverse(): processing edge a%u -> a%u ..", v0i, v1i);

        if (unlikely(v0->custom_label)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v0->label);
            if (width(v0->grad) != 0)
                set_label(v0->grad, tmp);
        }

        if (unlikely(edge.special)) {
            if (mode == ADMode::Forward)
                edge.special->forward(v0, v1, flags);
            else
                edge.special->backward(v1, v0, flags);

            if (flags & (uint32_t) ADFlag::ClearEdges) {
                // Edge may have been invalidated by callback, look up once more
                Edge &edge2 = state.edges[er.id];
                if (edge2.source == er.source && edge2.target == er.target) {
                    Special *special2 = edge2.special;
                    edge2.special = nullptr;
                    delete special2;
                }
            }
        } else {
            v1->mul_accum(v0->grad, edge.weight, v0->size);

            if (flags & (uint32_t) ADFlag::ClearEdges)
                edge.weight = Value();
        }
    }

    postprocess(v0i_prev, 0);

    ad_log(Debug, (flags & (uint32_t) ADFlag::ClearEdges)
                      ? "ad_traverse(): decreasing reference counts .."
                      : "ad_traverse(): decreasing reference counts "
                        "and removing traversed edges from graph ..");

    // Undo reference count increases performed by ad_enqueue()
    for (EdgeRef er : todo) {
        if (!er.target)
            continue;

        Edge &edge = state.edges[er.id];
        if (unlikely(edge.source != er.source || edge.target != er.target))
            ad_fail(
                "ad_traverse(): internal error: edge a%u -> a%u was garbage "
                "collected between enqueue and traverse steps!", er.source, er.target);
        else if (unlikely(!edge.visited))
            ad_fail("ad_traverse(): internal error: edge a%u -> a%u is not "
                    "marked as visited!", er.source, er.target);

        edge.visited = 0;

        Variable *source = state[er.source],
                 *target = state[er.target];

        if (flags & (uint32_t) ADFlag::ClearEdges) {
            ad_trace("ad_traverse(): removing edge a%u -> a%u", er.source, er.target);

            // Clear out forward edge
            uint32_t edge_id_prev = 0,
                     edge_id_cur = source->next_fwd;
            while (edge_id_cur) {
                Edge &e2 = state.edges[edge_id_cur];
                ad_trace("ad_traverse(): visiting forward edge a%u -> a%u", e2.source, e2.target);

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
                ls.cleanup.push_back(edge.special);
            edge.reset();
            state.unused_edges.push_back(er.id);

            ad_dec_ref(er.source, source);
            target = state[er.target]; // pointer might have changed
        }

        ad_dec_ref(er.target, target);
    }

    ad_log(Debug, "ad_traverse(): done.");

    std::vector<Special *> temp, &cleanup = ls.cleanup;
    if (!cleanup.empty()) {
        /* Extra-careful here: deallocate cleanup queue of
           custom AD edge callbacks (reentrant!) */

        temp.swap(cleanup);
        for (Special *special : temp)
            delete special;
        temp.clear();
        temp.swap(cleanup);
    }

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

    ad_trace("ad_enqueue_implicit(): enqueuing %zu implicit dependencies.",
             size - snapshot);

    std::lock_guard<std::mutex> guard(state.mutex);
    for (size_t i = snapshot; i < implicit.size(); ++i) {
        const EdgeRef &er = implicit[i];
        Edge &e = state.edges[er.id];

        if (e.source != er.source || e.target != er.target || e.visited)
            continue;

        e.visited = 1;
        ad_inc_ref(er.target, state[er.target]);
        ls.todo.push_back(er);
        ad_dfs_fwd(ls.todo, er.target, state[er.target]);
        state[er.source]->ref_count_grad++;
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
    for (size_t i = snapshot; i < implicit.size(); ++i)
        state[implicit[i].source]->ref_count_grad--;
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

        if (v->next_bwd == 0)
            color = "salmon";
        else if (v->next_fwd == 0)
            color = "lightblue2";
        if (labeled && !color)
            color = "wheat";
        if (is_valid(v->grad))
            color = "yellowgreen";

        buffer.fmt("|{a%u|S:%u|R:%u%s}",
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
// Export declarations
// ==========================================================================

template DRJIT_EXPORT void ad_inc_ref_impl<Value>(uint32_t) noexcept;
template DRJIT_EXPORT uint32_t ad_inc_ref_cond_impl<Value>(uint32_t) noexcept;
template DRJIT_EXPORT void ad_dec_ref_impl<Value>(uint32_t) noexcept;
template DRJIT_EXPORT uint32_t ad_new<Value>(const char *, size_t, uint32_t,
                                            uint32_t *, Value *);
template DRJIT_EXPORT Value ad_grad<Value>(uint32_t, bool);
template DRJIT_EXPORT void ad_set_grad<Value>(uint32_t, const Value &, bool);
template DRJIT_EXPORT void ad_accum_grad<Value>(uint32_t, const Value &, bool);
template DRJIT_EXPORT void ad_set_label<Value>(uint32_t, const char *);
template DRJIT_EXPORT const char *ad_label<Value>(uint32_t);
template DRJIT_EXPORT void ad_enqueue<Value>(ADMode, uint32_t);
template DRJIT_EXPORT void ad_traverse<Value>(ADMode, uint32_t);
template DRJIT_EXPORT size_t ad_implicit<Value>();
template DRJIT_EXPORT void ad_extract_implicit<Value>(size_t, uint32_t*);
template DRJIT_EXPORT void ad_enqueue_implicit<Value>(size_t);
template DRJIT_EXPORT void ad_dequeue_implicit<Value>(size_t);
template DRJIT_EXPORT const char *ad_graphviz<Value>();
template DRJIT_EXPORT uint32_t ad_new_select<Value, Mask>(
    const char *, size_t, const Mask &, uint32_t, uint32_t);
template DRJIT_EXPORT uint32_t ad_new_gather<Value, Mask, Index>(
    const char *, size_t, uint32_t, const Index &, const Mask &, bool);
template DRJIT_EXPORT uint32_t
ad_new_scatter<Value, Mask, Index>(const char *, size_t, ReduceOp, uint32_t,
                                   uint32_t, const Index &, const Mask &, bool);
template DRJIT_EXPORT void ad_add_edge<Value>(uint32_t, uint32_t,
                                              DiffCallback *);
template DRJIT_EXPORT void ad_scope_enter<Value>(ADScope, size_t, const uint32_t *);
template DRJIT_EXPORT void ad_scope_leave<Value>(bool);
template DRJIT_EXPORT bool ad_grad_enabled<Value>(uint32_t);
template DRJIT_EXPORT bool ad_enabled<Value>() noexcept;
NAMESPACE_END(detail)

template struct DRJIT_EXPORT DiffArray<detail::Value>;

NAMESPACE_END(drjit)
