#include "common.h"
#include <enoki/jit.h>
#include <enoki/math.h>
#include <enoki/autodiff.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <assert.h>
#include <mutex>
#include <xxh3.h>

#define CONCAT(x,y) x ## _ ## y
#define EVAL(x,y) CONCAT(x,y)
#define RENAME(fun) EVAL(fun, ENOKI_AUTODIFF_NAME)

/// Rename various things to avoid symbol clashes
#define Special              RENAME(Special)
#define Edge                 RENAME(Edge)
#define Variable             RENAME(Variable)
#define State                RENAME(State)
#define ReleaseQueueHelper   RENAME(ReleaseQueueHelper)
#define ReleaseOperandHelper RENAME(ReleaseOperandHelper)

struct Int32Hasher {
    size_t operator()(int32_t v) const {
        // fmix32 from MurmurHash by Austin Appleby (public domain)
        v ^= v >> 16;
        v *= 0x85ebca6b;
        v ^= v >> 13;
        v *= 0xc2b2ae35;
        v ^= v >> 16;
        return (size_t) v;
    }
};

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

using Value = ENOKI_AUTODIFF_VALUE;
using Mask = mask_t<Value>;
using Index = uint32_array_t<Value>;
using Scalar = scalar_t<Value>;
constexpr bool IsDouble = std::is_same_v<Value, double>;

struct Variable;

// Special edge (scatter, gather, scatter_reduce, block_sum, etc.)
struct Special {
    virtual void backward(Variable *source, const Variable *target) const {
        ad_raise("Special::backward(): not implemented!");
    }

    virtual void forward(const Variable *source, Variable *target) const {
        ad_raise("Special::forward(): not implemented!");
    }

    virtual ~Special() = default;
};

// Weighted edge connecting two variables
struct Edge {
    /// Variable index of source operand
    int32_t source;

    /// Source variable index
    int32_t target;

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

static_assert(sizeof(Edge) == 8 * sizeof(uint32_t),
              "Edge data structure has incorrect size. Padding problem?");

template <typename T> bool is_valid(const T &value) {
    if constexpr (is_jit_array_v<T>)
        return value.valid();
    else
        return true;
}

/// Represents a variable in the computation graph
struct Variable {
    /// Descriptive label or nullptr
    char *label;

    /// Number of times this variable is referenced by other variables
    uint64_t ref_count_int : 24;

    /// Number of times this variable is referenced from Python/C++
    uint64_t ref_count_ext : 24;

    /// Gradient reference count for special operations
    uint64_t ref_count_grad : 12;

    /// Was the node visited by the DFS traversal?
    uint64_t visited : 1;

    /// Was the label manually overwritten via set_label()?
    uint64_t custom_label : 1;

    /// Should the label be freed when the variable is deallocated?
    uint64_t free_label : 1;

    /// Was this graph node created while recording computation?
    uint64_t placeholder : 1;

    /// Links to the first forward edge at this node
    uint32_t next_fwd;

    /// Links to the first backward edge at this node
    uint32_t next_rev;

    /// Dimension of the 'grad' field
    uint32_t size;

    /// This will eventually hold a gradient value
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
            size_t size = strlen(prefix) + strlen(label) + 2;
            char *out = (char *) malloc(size);
            snprintf(out, size, "%s/%s", prefix, label);
            label = out;
            free_label = 1;
        }

        placeholder = (uint64_t) placeholder_;
    }

    template <typename T = Value>
    void accum(const T& v, uint32_t src_size) {
        if constexpr (is_array_v<T>) {
            bool grad_valid = is_valid(grad);
            if (size == 1 && src_size != 1) {
                Value v2;
                if (v.size() == 1) {
                    v2 = v * Scalar(src_size);
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

    template <typename T = Value>
    void mul_accum(const T &v1, const T &v2_, uint32_t src_size) {
        /* The goal of the following logic is to always ensure that
           v1 == 0 implies v1 * v2 == 0, even if multiplication by
           v2 would produce a NaN (e.g. if v2 is infinite or NaN). */

        T z = 0.f, v2 = select(eq(v1, z), z, v2_);

        if constexpr (is_jit_array_v<T>) {
            if (v2_.is_literal() && std::isnormal(v2_[0]) &&
                jit_flag(JitFlag::ADOptimize))
                v2 = v2_;
        }

        if constexpr (is_array_v<T>) {
            bool grad_valid = is_valid(grad);

            if (size == 1 && src_size != 1) {
                T v3 = v1 * v2;
                if (v3.size() == 1) {
                    v3 *= Scalar(src_size);
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

static_assert(sizeof(Variable) == ((IsDouble ? 2 : 0) + 8) * sizeof(uint32_t),
              "Variable data structure has incorrect size. Padding problem?");

#if !defined(_MSC_VER)
  /// Thread-local list used by ad_queue() and ad_traverse()
  static __thread std::vector<int32_t> *tls_queue = nullptr;
  /// List of postponed variables that are yet to be processed
  static __thread std::vector<int32_t> *postponed = nullptr;
#else
  static __declspec(thread) std::vector<int32_t>* tls_queue = nullptr;
  static __declspec(thread) std::vector<int32_t>* postponed = nullptr;
#endif

/// Records all internal application state
struct State {
    using VariableMap = tsl::robin_map<int32_t, Variable, Int32Hasher,
                                       std::equal_to<int32_t>>;
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
    int32_t variable_index = 1;

    /// Tracks dependencies between normal and placeholder variables
    uint32_t internal_deps = 0;

    State() : edges(1) { }

    Variable *operator[](int32_t index) {
        auto it = variables.find(index);
        if (unlikely(index < 0 || it == variables.end()))
            ad_fail("referenced an unknown variable a%i!", index);
        return &it.value();
    }

    ~State() {
        if (!variables.empty()) {
            ad_log(Warn,
                   "enoki-ad: variable leak detected (%zu variables "
                   "remain in use)!", variables.size());
            uint32_t counter = 0;
            for (auto kv : variables) {
                ad_log(Warn, " - variable a%i", kv.first);
                if (++counter == 10) {
                    ad_log(Warn, " - (skipping the rest)");
                    break;
                }
            }
        }

        size_t edges_used = edges.size() - unused_edges.size() - 1;
        if (edges_used != 0)
            ad_log(Warn,
                   "enoki-ad: edge leak detected (%zu edges "
                   "remain in use)!", edges_used);

        if (tls_queue) {
            delete tls_queue;
            tls_queue = nullptr;
        }
    }
};

static State state;

template <typename T> uint32_t asize(const T &value) {
    if constexpr (std::is_scalar_v<T>)
        return 1;
    else
        return (uint32_t) value.size();
}

template <typename Value, typename Mask, typename Index>
int32_t ad_new_gather_impl(const char *label, size_t size, int32_t src_index,
                           const Index &offset, const Mask &mask, bool permute);

void Edge::reset() {
    Special *special_copy = special;
    memset(this, 0, sizeof(uint32_t) * 4 + sizeof(Special *));
    weight = Value();
    if (special_copy) {
        unlock_guard<std::mutex> guard(state.mutex);
        delete special_copy;
    }
}

extern void RENAME(ad_whos)() {
    std::vector<int32_t> indices;
    indices.reserve(state.variables.size());
    for (auto &kv: state.variables)
        indices.push_back(kv.first);
    std::sort(indices.begin(), indices.end());

    for (int32_t id : indices) {
        const Variable *v = state[id];
        buffer.fmt("  %-7i ", id);
        size_t sz =
            buffer.fmt("%llu / %llu", (unsigned long long) v->ref_count_ext,
                       (unsigned long long) v->ref_count_int);

        buffer.fmt("%*s%-12u%-8s\n", 11 - (int) sz, "", v->size,
                   v->label ? v->label : "");
    }
}

template <typename T> void ad_enqueue(int32_t index) {
    // no locking needed in this function (all TLS)
    if (index <= 0)
        return;
    std::vector<int32_t> *queue = tls_queue;
    if (unlikely(!queue))
        queue = tls_queue = new std::vector<int32_t>();
    ad_trace("ad_enqueue(a%i)", index);
    queue->push_back(index);
    state[index]->ref_count_int++;
}

/// Forward-mode DFS starting from 'index'
static void ad_dfs_fwd(std::vector<int32_t> &todo, int32_t index, Variable *v) {
    ad_trace("ad_dfs_fwd(): adding a%i to the todo list", index);
    todo.push_back(index);
    v->visited = 1;

    uint32_t edge = v->next_fwd;
    while (edge) {
        Edge &e = state.edges[edge];
        Variable *v2 = state[e.target];
        if (!v2->visited)
            ad_dfs_fwd(todo, e.target, v2);
        edge = e.next_fwd;
    }
}

/// Reverse-mode DFS starting from 'index'
static void ad_dfs_rev(std::vector<int32_t> &todo, uint32_t index, Variable *v) {
    ad_trace("ad_dfs_rev(): adding a%i to the todo list", index);
    todo.push_back(index);
    v->visited = 1;

    uint32_t edge = v->next_rev;
    while (edge) {
        Edge &e = state.edges[edge];
        Variable *v2 = state[e.source];
        if (unlikely(v->placeholder && !v2->placeholder)) {
            ad_trace("ad_dfs_rev(): adding a%i to the postponed list", e.source);
            if (!postponed)
                postponed = new std::vector<int32_t>();
            postponed->push_back(e.source);
            state[e.source]->ref_count_int++;
        } else if (!v2->visited) {
            ad_dfs_rev(todo, e.source, v2);
        }
        edge = e.next_rev;
    }
}

/// Build an ordered list of nodes to traverse given a set of seed vertices
static void ad_build_todo(std::vector<int32_t> &queue,
                          std::vector<int32_t> &todo,
                          bool reverse) {
    // DFS traversal to find all reachable variables
    if (reverse) {
        for (int32_t index: queue) {
            Variable *v = state[index];
            if (!v->visited)
                ad_dfs_rev(todo, index, v);
        }
    } else {
        for (int32_t index: queue) {
            Variable *v = state[index];
            if (!v->visited)
                ad_dfs_fwd(todo, index, v);
        }
    }

    // Clear visited flags
    for (int32_t i : todo)
        state[i]->visited = false;

    // Bring into the appropriate order
    std::sort(
        todo.begin(), todo.end(),
        [reverse](int i, int j) { return reverse ? (i > j) : (i < j); });
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

/// Allocate a new variable
static std::pair<int32_t, Variable *> ad_var_new(const char *label,
                                                 size_t size) {
    while (true) {
        int32_t index = state.variable_index++;

        if (unlikely(index <= 0)) { // overflow
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

static void ad_free(int32_t index, Variable *v);

/// Clear backward edges of the given variable and decrease int. ref. counts
static void ad_free_edges(int32_t index, Variable *v) {
    uint32_t edge_id = v->next_rev;
    ad_trace("ad_free_edges(): freeing edges of variable a%i", index);
    v->next_rev = 0;

    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        ad_trace("ad_free_edges(): freeing edge %u: a%i -> a%i", edge_id,
                 edge.source, edge.target);

        int32_t source = edge.source;
        uint32_t next_rev = edge.next_rev,
                 next_fwd = edge.next_fwd;

        assert(edge.target == index);
        edge.reset();

        Variable *v2 = state[source];
        if (unlikely(v2->ref_count_int == 0))
            ad_fail("enoki-autodiff: fatal error: internal reference count of "
                    "variable a%i became negative!", source);

        if (--v2->ref_count_int == 0 && v2->ref_count_ext == 0) {
            ad_free(source, v2);
        } else {
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

static void ad_free(int32_t index, Variable *v) {
    ad_trace("ad_free(a%i)", index);
    if (v->free_label)
        free(v->label);
    if (v->next_rev)
        ad_free_edges(index, v);
    state.variables.erase(index);
}

/// Ensure that placeholder variables are consistently sized to avoid horizontal ops
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

struct ReleaseOperandHelper {
    std::vector<uint32_t> release;

    void put(uint32_t index) { release.push_back(index); }

    ~ReleaseOperandHelper() {
        for (uint32_t index: release) {
            Variable *v = state[index];
            if (--v->ref_count_ext == 0 && v->ref_count_int == 0)
                ad_free(index, v);
        }
    }
};

template <typename T>
int32_t ad_new(const char *label, size_t size, uint32_t op_count,
               int32_t *op, T *weights) {
    std::lock_guard<std::mutex> guard(state.mutex);

    bool eager_fwd = false, rec = false, check_weights = false;
    if constexpr (is_jit_array_v<Value>) {
        eager_fwd = jit_flag(JitFlag::ADEagerForward);
        rec = jit_flag(JitFlag::Recording);
        check_weights = jit_flag(JitFlag::ADCheckWeights);
    }

    ReleaseOperandHelper helper;
    if (unlikely(rec)) {
        for (uint32_t i = 0; i < op_count; ++i) {
            if (op[i] <= 0)
                continue;

            uint32_t index = op[i];
            Variable *var = state[index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(!var->placeholder)) {
                if (var->size != 1)
                    ad_raise("ad_new(): expression accesses a non-scalar "
                             "private variable (a%i)!", index);

                index = ad_new_gather_impl<Value>("gather", size, op[i], Index(0),
                                                  Mask(true), false);

                ad_trace("ad_new(): placeholder accesses non-placeholder "
                         "variable a%i, inserting a gather operation (a%i).",
                         op[i], index);

                var = state[index];
                op[i] = index;
                helper.put(index);
            }
        }
    }

    auto [index, var] = ad_var_new(label, size);

    if (unlikely(log_level >= Debug)) {
        const char *l = label ? label : "unnamed";
        switch (op_count) {
            case 0:
                ad_log(Debug, "ad_new(a%i, size=%zu): %s", index, size, l); break;
            case 1:
                ad_log(Debug, "ad_new(a%i <- %i, size=%zu): %s", index, op[0], size, l); break;
            case 2:
                ad_log(Debug, "ad_new(a%i <- a%i, a%i, size=%zu): %s", index, op[0], op[1], size, l); break;
            case 3:
                ad_log(Debug, "ad_new(a%i <- a%i, a%i, a%i, size=%zu): %s", index, op[0], op[1], op[2], size, l); break;
            default: break;
        }
    }

    uint32_t edge_index = 0;
    for (uint32_t i = 0; i < op_count; ++i) {
        if (op[i] <= 0)
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

        if constexpr (is_jit_array_v<T>) {
            if (unlikely(check_weights)) {
                bool nan_weights = any(isnan(weights[i])),
                     inf_weights = any(isinf(weights[i]));

                if (nan_weights)
                    ad_log(Warn,
                           "ad_new(a%i <- a%i): \"%s\" -- weight of edge %i "
                           "contains NaNs! Inspect the computation graph via "
                           "enoki::graphviz() or put a breakpoint on "
                           "ad_check_weights_cb() to investigate further.",
                           index, op[i], label ? label : "unnamed", i);

                if (inf_weights)
                    ad_log(Warn,
                           "ad_new(a%i <- a%i): \"%s\": weight of edge %i contains "
                           "infinities! Inspect the computation graph via "
                           "enoki::graphviz() or put a breakpoint on "
                           "ad_check_weights_cb() to investigate further.",
                           index, op[i], label ? label : "unnamed", i);

                if (nan_weights || inf_weights)
                    ad_check_weights_cb();
            }
        }

        uint32_t index2 = op[i];
        Variable *var2 = state[index2];

        if (unlikely(eager_fwd)) {
            var->mul_accum(var2->grad, weights[i], var2->size);
            continue;
        }

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = index2;
        edge.target = index;
        edge.weight = std::move(weights[i]);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = edge_index;
        edge_index = edge_index_new;

        var2->ref_count_int++;
        var2->next_fwd = edge_index_new;
    }

    if (op_count > 0 && edge_index == 0 && !eager_fwd) {
        // All edges were pruned, don't create the node after all
        ad_trace(
            "ad_new(a%i): all nodes were pruned, removing variable from graph", index);
        ad_free(index, var);
        return 0;
    }

    var->next_rev = edge_index;
    var->ref_count_ext = 1;

    if (var->placeholder)
        ad_propagate_placeholder_size(var);

    return index;
}

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
int32_t ad_new_select(const char *label, size_t size, const Mask &mask,
                      int32_t t_index, int32_t f_index) {
    std::lock_guard<std::mutex> guard(state.mutex);
    if constexpr (is_jit_array_v<Mask>) {
        if (jit_flag(JitFlag::ADOptimize) && mask.is_literal()) {
            int32_t result = mask[0] ? t_index : f_index;
            if (result)
                state[result]->ref_count_ext++;
            ad_log(Debug, "ad_new_select(a%i <- a%i, a%i): simplified", result, t_index, f_index);
            return result;
        }
    }

    if (f_index == t_index) {
        if (t_index)
            state[t_index]->ref_count_ext++;
        ad_log(Debug, "ad_new_select(a%i <- a%i, a%i): simplified", t_index, t_index, f_index);
        return t_index;
    }

    bool eager_fwd = false, rec = false;
    if constexpr (is_jit_array_v<Value>) {
        eager_fwd = jit_flag(JitFlag::ADEagerForward);
        rec = jit_flag(JitFlag::Recording);
    }

    if (eager_fwd) {
        Value vt = 0, vf = 0;
        if (t_index)
            vt = state[t_index]->grad;
        if (f_index)
            vf = state[f_index]->grad;
        auto [index, var] = ad_var_new(label, size);
        var->grad = select(mask, vt, vf);
        var->ref_count_ext = 1;
        return index;
    }

    int32_t op[2] = { t_index, f_index };
    ReleaseOperandHelper helper;

    if (rec) {
        for (uint32_t i = 0; i < 2; ++i) {
            if (op[i] <= 0)
                continue;

            uint32_t index = op[i];
            Variable *var = state[index];

            /* When recording AD code (e.g. in a virtual function call),
               convert reads from external/private variables into gathers */
            if (unlikely(!var->placeholder)) {
                if (var->size != 1)
                    ad_raise("ad_new_select(): expression accesses a non-scalar "
                             "private variable (a%i)!", index);

                index = ad_new_gather_impl<Value>("gather", size, op[i], Index(0),
                                                   Mask(true), false);

                ad_trace("ad_new_select(): placeholder accesses non-placeholder "
                         "variable a%i, inserting a gather operation (a%i).",
                         op[i], index);

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
        if (op[i] <= 0)
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

        var2->ref_count_int++;
        var2->next_fwd = edge_index_new;
    }

    var->next_rev = edge_index;
    var->ref_count_ext = 1;

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
                      asize(offset));
    }

    Index offset;
    Mask mask;
    bool permute;
};

template <typename Value, typename Mask, typename Index>
int32_t ad_new_gather_impl(const char *label, size_t size, int32_t src_index,
                           const Index &offset, const Mask &mask_, bool permute) {
    Mask mask(mask_);

    bool eager_fwd = false;
    if constexpr (is_jit_array_v<Value>)
        eager_fwd = jit_flag(JitFlag::ADEagerForward);

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

        if (eager_fwd) {
            var->accum(gather<Value>(var2->grad, offset, mask), asize(offset));
            var->ref_count_ext = 1;
            return index;
        }

        /* When recording some computation (e.g. part of a virtual function call)
           that gathers from a global or private instance variable, keep track of
           the dependency so that the computation graph can be wired up correctly
           following the recording step. See vcall_autodiff.h */
        if (var->placeholder && !var2->placeholder)
            state.internal_deps++;

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = src_index;
        edge.target = index;
        edge.special = new GatherEdge<Value>(offset, mask, permute);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = 0;
        var2->ref_count_int++;
        var2->next_fwd = edge_index_new;
        var->next_rev = edge_index_new;
        var->ref_count_ext = 1;

        return index;
    } else {
        enoki_raise("ad_new_gather(): differentiable gathers not supported by "
                    "this backend!");
    }
}

template <typename Value, typename Mask, typename Index>
int32_t ad_new_gather(const char *label, size_t size, int32_t src_index,
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
                        asize(offset));
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
int32_t ad_new_scatter(const char *label, size_t size, ReduceOp op,
                       int32_t src_index, int32_t dst_index, const Index &offset,
                       const Mask &mask_, bool permute) {

    Mask mask(mask_);

    bool eager_fwd = false;
    if constexpr (is_jit_array_v<Value>)
        eager_fwd = jit_flag(JitFlag::ADEagerForward);

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

        if (eager_fwd) {
            Value vs = 0, vt = 0;
            if (src_index)
                vs = state[src_index]->grad;
            if (dst_index)
                vt = state[dst_index]->grad;

            if (vt.size() != size)
                vt.resize(size);

            if (op != ReduceOp::None)
                scatter_reduce(op, vt, vs, offset, mask);
            else
                scatter(vt, vs, offset, mask);

            var->grad = vt;
            var->ref_count_ext = 1;
            return index;
        }

        uint32_t edge_index = 0;

        if (src_index > 0) {
            Variable *var2 = state[src_index];
            uint32_t edge_index_new = ad_edge_new();
            Edge &edge = state.edges[edge_index_new];
            edge.source = src_index;
            edge.target = index;
            edge.special = new ScatterEdge<Value>(offset, mask, op);
            edge.next_fwd = var2->next_fwd;
            edge.next_rev = var->next_rev;
            var2->ref_count_int++;
            var2->next_fwd = edge_index_new;
            edge_index = edge_index_new;
        }

        if (dst_index > 0) {
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
            var2->ref_count_int++;
            var2->next_fwd = edge_index_new;
            edge_index = edge_index_new;
        }

        if (edge_index == 0)
            ad_raise("ad_new_scatter(): all inputs were non-differentiable!");

        var->next_rev = edge_index;
        var->ref_count_ext++;

        return index;
    } else {
        enoki_raise("ad_new_scatter(): differentiable scatters not supported "
                    "by this backend!");
    }
}

static void ad_traverse_rev(std::vector<int32_t> &todo, bool retain_graph) {
    ad_log(Debug, "ad_traverse_rev(): processing %zu nodes ..", todo.size());

    for (int32_t index : todo) {
        Variable *v = state[index];
        ad_trace("ad_traverse_rev(): processing variable a%i ..", index);

        if constexpr (is_dynamic_v<Value>) {
            uint32_t grad_size = asize(v->grad);
            if (unlikely(v->size != grad_size && grad_size != 1))
                ad_raise("ad_traverse_rev(): variable a%i has an invalid "
                         "gradient size: expected %u, got %u!", index,
                         v->size, grad_size);
        }

        if (unlikely(v->custom_label)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v->label);
            if (width(v->grad) != 0)
                set_label(v->grad, tmp);
        }

        uint32_t edge_id = v->next_rev;
        while (edge_id) {
            Edge &edge = state.edges[edge_id];
            assert(edge.target == index);
            Variable *v2 = state[edge.source];
            uint32_t next_rev = edge.next_rev;

            ad_trace("ad_traverse_rev(): processing edge a%i -> a%i ..", index,
                     edge.source);

            if (unlikely(edge.special)) {
                edge.special->backward(v2, v);

                if (!retain_graph) {
                    // Look up edge, once more, entry in table may have changed
                    Edge &edge2 = state.edges[edge_id];
                    Special *special = edge2.special;
                    edge2.special = nullptr;
                    unlock_guard<std::mutex> guard(state.mutex);
                    delete special;
                }

                // Address of 'v' and 'v2' may have changed due to the above
                v = state[index];
                v2 = state[index];
            } else {
                v2->mul_accum(v->grad, edge.weight, v->size);

                if (!retain_graph)
                    edge.weight = Value();
            }

            edge_id = next_rev;
        }

        if (v->next_rev && v->ref_count_grad == 0) {
            ad_trace("ad_traverse_rev(): clearing gradient at intermediate "
                     "variable a%i (\"%s\")", index, v->label ? v->label : "unnamed");
            v->grad = Value();
        }
    }

    if (!retain_graph) {
        ad_log(Debug, "ad_traverse_rev(): cleaning up ..");
        for (auto it = todo.rbegin(); it != todo.rend(); ++it) {
            int32_t index = *it;
            Variable *v = state[index];
            ad_free_edges(index, v);
        }
    }

    ad_log(Debug, "ad_traverse_rev(): done.");
}

static void ad_traverse_fwd(std::vector<int32_t> &todo, bool retain_graph) {
    ad_log(Debug, "ad_traverse_fwd(): processing %zu nodes ..", todo.size());

    for (int32_t index : todo) {
        Variable *v = state[index];
        ad_trace("ad_traverse_fwd(): processing variable a%i ..", index);

        if constexpr (is_dynamic_v<Value>) {
            uint32_t grad_size = asize(v->grad);
            if (unlikely(v->size != grad_size && grad_size != 1))
                ad_raise("ad_traverse_fwd(): variable a%i has an invalid "
                         "gradient size: expected %u, got %u!", index,
                         v->size, grad_size);
        }

        if (unlikely(v->custom_label)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v->label);
            if (width(v->grad) != 0)
                set_label(v->grad, tmp);
        }

        uint32_t edge_id = v->next_fwd;
        while (edge_id) {
            Edge &edge = state.edges[edge_id];
            assert(edge.source == index);
            Variable *v2 = state[edge.target];
            uint32_t next_fwd = edge.next_fwd;

            ad_trace("ad_traverse_fwd(): processing edge a%i -> a%i ..", index,
                     edge.target);

            if (unlikely(edge.special)) {
                edge.special->forward(v, v2);

                if (!retain_graph) {
                    // Look up edge, once more, entry in table may have changed
                    Edge &edge2 = state.edges[edge_id];
                    Special *special = edge2.special;
                    edge2.special = nullptr;
                    unlock_guard<std::mutex> guard(state.mutex);
                    delete special;
                }

                // Address of 'v' and 'v2' may have changed due to the above
                v = state[index];
                v2 = state[index];
            } else {
                v2->mul_accum(v->grad, edge.weight, v->size);

                if (!retain_graph)
                    edge.weight = Value();
            }

            edge_id = next_fwd;
        }

        if (v->next_fwd && v->ref_count_grad == 0) {
            ad_trace("ad_traverse_fwd(): clearing gradient at intermediate "
                     "variable a%i (\"%s\")", index, v->label ? v->label : "unnamed");
            v->grad = Value();
        }

        if (!retain_graph)
            ad_free_edges(index, v);
    }

    ad_log(Debug, "ad_traverse_fwd(): done.");
}

template <typename Value> const char *ad_graphviz() {
    std::lock_guard<std::mutex> guard(state.mutex);

    std::vector<int32_t> indices;
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

    for (int32_t index : indices) {
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

        buffer.fmt("|{a%i|S:%u|E:%u|I:%u%s}",
            index, v->size, (uint32_t) v->ref_count_ext,
            (uint32_t) v->ref_count_int,
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

    for (int32_t index : indices) {
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

template <typename T> void ad_inc_ref_impl(int32_t index) noexcept(true) {
    if (index == 0)
        return;
    index = std::abs(index);
    std::lock_guard<std::mutex> guard(state.mutex);
    Variable *v = state[index];
    ad_trace("ad_inc_ref(a%i): %u", index, (uint32_t) v->ref_count_ext + 1);
    v->ref_count_ext++;
}

template <typename T> void ad_dec_ref_impl(int32_t index) noexcept(true) {
    if (index == 0)
        return;
    index = std::abs(index);
    std::lock_guard<std::mutex> guard(state.mutex);
    Variable *v = state[index];
    ad_trace("ad_dec_ref(a%i): %u", index, (uint32_t) v->ref_count_ext - 1);
    if (unlikely(v->ref_count_ext == 0))
        ad_fail("enoki-autodiff: fatal error: external reference count of "
                "variable a%i became negative!", index);
    if (--v->ref_count_ext == 0 && v->ref_count_int == 0)
        ad_free(index, v);
}

template <typename T> T ad_grad(int32_t index, bool fail_if_missing) {
    if (unlikely(index <= 0))
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

template <typename T> void ad_set_grad(int32_t index, const T &value, bool fail_if_missing) {
    if (unlikely(index <= 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_set_grad(): referenced an unknown variable a%i!", index);
        return;
    }

    ad_trace("ad_set_grad(a%i)", index);
    Variable &v = it.value();
    if (v.size != 1 || width(value) == 1)
        v.grad = value;
    else
        v.grad = hsum_async(value);
}

template <typename T> void ad_accum_grad(int32_t index, const T &value, bool fail_if_missing) {
    if (unlikely(index <= 0))
        return;

    std::lock_guard<std::mutex> guard(state.mutex);
    auto it = state.variables.find(index);
    if (it == state.variables.end()) {
        if (fail_if_missing)
            ad_raise("ad_accum_grad(): referenced an unknown variable a%i!", index);
        return;
    }
    ad_trace("ad_accum_grad(a%i)", index);
    Variable &v = it.value();
    v.accum(value, width(value));
}

template <typename T> void ad_set_label(int32_t index, const char *label) {
    if (index == 0)
        return;
    index = std::abs(index);
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_log(Debug, "ad_set_label(a%i, \"%s\")", index, label ? label : "(null)");
    Variable *v = state[index];
    if (v->free_label)
        free(v->label);
    v->label = strdup(label);
    v->free_label = true;
    v->custom_label = true;
}

template <typename T> const char *ad_label(int32_t index) {
    if (index == 0)
        return nullptr;
    index = std::abs(index);
    std::lock_guard<std::mutex> guard(state.mutex);
    return state[index]->label;
}

template <typename T>
void ad_add_edge(int32_t source_idx, int32_t target_idx,
                 DiffCallback *callback) {
    std::lock_guard<std::mutex> guard(state.mutex);

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
    source->ref_count_int++;
    target->next_rev = edge_index_new;
}

struct ReleaseQueueHelper {
    std::vector<int32_t> &queue;
    ReleaseQueueHelper(std::vector<int32_t> &queue) : queue(queue) { }

    ~ReleaseQueueHelper() {
        for (int32_t index: queue) {
            Variable *v = state[index];
            if (--v->ref_count_int == 0 && v->ref_count_ext == 0)
                ad_free(index, v);
        }
    }
};

template <typename T>
static void ad_traverse_impl(std::vector<int32_t> &queue_,
                             bool reverse,
                             bool retain_graph) {
    std::vector<int32_t> todo, queue(queue_);
    queue_.clear();

    ReleaseQueueHelper helper(queue);

    ad_build_todo(queue, todo, reverse);

    if (reverse)
        ad_traverse_rev(todo, retain_graph);
    else
        ad_traverse_fwd(todo, retain_graph);
}

template <typename T> void ad_traverse(bool backward, bool retain_graph) {
    std::lock_guard<std::mutex> guard(state.mutex);
    ad_traverse_impl<T>(*tls_queue, backward, retain_graph);
}

template <typename T> void ad_traverse_postponed() {
    std::lock_guard<std::mutex> guard(state.mutex);
    std::vector<int32_t> *queue = postponed;

    if (!postponed || postponed->empty())
        return;
    ad_log(Debug, "ad_traverse_postponed(): reverse mode (%zu indices)",
           postponed->size());
    ad_traverse_impl<T>(*postponed, true, true);
    ad_log(Debug, "ad_traverse_postponed(): reverse mode done.");
}

template <typename T> size_t ad_internal_deps() {
    std::lock_guard<std::mutex> guard(state.mutex);
    return state.internal_deps;
}

template ENOKI_EXPORT void ad_inc_ref_impl<Value>(int32_t) noexcept;
template ENOKI_EXPORT void ad_dec_ref_impl<Value>(int32_t) noexcept;
template ENOKI_EXPORT int32_t ad_new<Value>(const char *, size_t, uint32_t,
                                            int32_t *, Value *);
template ENOKI_EXPORT Value ad_grad<Value>(int32_t, bool);
template ENOKI_EXPORT void ad_set_grad<Value>(int32_t, const Value &, bool);
template ENOKI_EXPORT void ad_accum_grad<Value>(int32_t, const Value &, bool);
template ENOKI_EXPORT void ad_set_label<Value>(int32_t, const char *);
template ENOKI_EXPORT const char *ad_label<Value>(int32_t);
template ENOKI_EXPORT void ad_enqueue<Value>(int32_t);
template ENOKI_EXPORT void ad_traverse<Value>(bool, bool);
template ENOKI_EXPORT void ad_traverse_postponed<Value>();
template ENOKI_EXPORT size_t ad_internal_deps<Value>();
template ENOKI_EXPORT const char *ad_graphviz<Value>();
template ENOKI_EXPORT int32_t ad_new_select<Value, Mask>(
    const char *, size_t, const Mask &, int32_t, int32_t);
template ENOKI_EXPORT int32_t ad_new_gather<Value, Mask, Index>(
    const char *, size_t, int32_t, const Index &, const Mask &, bool);
template ENOKI_EXPORT int32_t
ad_new_scatter<Value, Mask, Index>(const char *, size_t, ReduceOp, int32_t,
                                   int32_t, const Index &, const Mask &, bool);
template ENOKI_EXPORT void ad_add_edge<Value>(int32_t, int32_t,
                                              DiffCallback *);
NAMESPACE_END(detail)

template struct ENOKI_EXPORT DiffArray<detail::Value>;

NAMESPACE_END(enoki)
