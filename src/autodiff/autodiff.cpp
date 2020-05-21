#include "common.h"
#include <enoki/cuda.h>
#include <enoki/llvm.h>
#include <enoki/math.h>
#include <enoki/autodiff.h>
#include <tsl/robin_map.h>
#include <deque>
#include <assert.h>

#if defined(ENOKI_USE_TBB)
#  include <tbb/spin_mutex.h>
#else
#  include <mutex>
#endif

/* Prefer TBB's spin mutex, which are slightly faster in
   single threaded workloads (which are the expectation.) */
#if defined(ENOKI_USE_TBB)
    using Mutex = tbb::spin_mutex;
#else
    using Mutex = std::mutex;
#endif

#define CONCAT(x,y) x ## _ ## y
#define EVAL(x,y) CONCAT(x,y)
#define RENAME(fun) EVAL(fun, ENOKI_AUTODIFF_NAME)

/// Rename various things to avoid symbol clashes
#define Special  RENAME(Special)
#define Edge     RENAME(Edge)
#define Variable RENAME(Variable)
#define State    RENAME(State)

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

using Value = ENOKI_AUTODIFF_VALUE;
using Mask = mask_t<Value>;
using Index = uint32_array_t<Value>;
using Scalar = scalar_t<Value>;
constexpr bool IsDouble = std::is_same_v<Value, double>;

struct Variable;

// Special edge (scatter, gather, scatter_add, block_sum, etc.)
struct Special {
    virtual void backward(Variable *source, const Variable *target) const {
        throw std::runtime_error("Special::backward(): not implemented!");
    }

    virtual void forward(const Variable *source, Variable *target) const {
        throw std::runtime_error("Special::forward(): not implemented!");
    }

    virtual ~Special() = default;
};

// Weighted edge connecting two variables
struct Edge {
    /// Variable index of source operand
    uint32_t source;

    /// Source variable index
    uint32_t target;

    /// Links to the next forward edge
    uint32_t next_fwd;

    /// Links to the next reverse edge
    uint32_t next_rev : 31;

    /// Marks the edge status during topo-sort
    uint32_t visited : 1;

    /// Pointer to a handler for "special" edges
    Special *special = nullptr;

    /// Weight value (zero/empty for "special" edges)
    Value weight{};

    ENOKI_ARRAY_DEFAULTS(Edge);

    Edge() {
        memset(this, 0, sizeof(uint32_t) * 6);
    }

    /// Reset the contents of this edge to the default values
    void reset() {
        assert(!visited);
        delete special;
        memset(this, 0, sizeof(uint32_t) * 6);
        weight = Value();
    }
};

template <typename T> uint32_t asize(const T &value) {
    if constexpr (std::is_scalar_v<T>)
        return 1;
    else
        return (uint32_t) value.size();
}

static_assert(sizeof(Edge) == 8 * sizeof(uint32_t),
              "Edge data structure has incorrect size. Padding problem?");

/// Represents a variable in the computation graph
struct Variable {
    /// Descriptive label or nullptr
    char *label;

    /// Number of times this variable is referenced by other variables
    uint32_t ref_count_int;

    /// Number of times this variable is referenced from Python/C++
    uint32_t ref_count_ext : 30;

    /// Was the label manually overwritten via set_label()?
    uint32_t custom_label : 1;

    /// Should the label variable be freed when the Variable is deallocated?
    uint32_t free_label : 1;

    /// Links to the first forward edge at this node
    uint32_t next_fwd;

    /// Links to the first reverse edge at this node
    uint32_t next_rev;

    /// Number of entries that we expect for the gradient
    uint32_t size;

    /// This will eventually hold a gradient value
    Value grad{};

    Variable() {
        memset(this, 0, 7 * sizeof(uint32_t));
    }

    Variable(const char *label_, uint32_t size_) : Variable() {
        label = (char *) label_;
        if (!label)
            label = (char *) "unnamed";
        size = size_;
        const char *prefix = ad_prefix();
        if (unlikely(prefix)) {
            size_t size = strlen(prefix) + strlen(label) + 2;
            char *out = (char *) malloc(size);
            snprintf(out, size, "%s/%s", prefix, label);
            label = out;
            free_label = 1;
        }
    }

    template <typename T>
    void accum(const T& v, uint32_t src_size) {
        if constexpr (is_array_v<T>) {
            if (size == 1 && src_size != 1) {
                Value v2;
                if (((const T &) v).size() == 1)
                    v2 = v * Scalar(src_size);
                else
                    v2 = hsum_async(v);

                if (((const T &) grad).valid())
                    grad += v2;
                else
                    grad = std::move(v2);
            } else {
                if (((const T &) grad).valid())
                    grad += v;
                else
                    grad = v;
            }
        } else {
            grad += v;
        }
    }

    template <typename T>
    void mul_accum(const T &v1, const T &v2, uint32_t src_size) {
        if constexpr (is_array_v<T>) {
            if (size == 1 && src_size != 1) {
                T v3;
                if (((const T &) v1).size() == 1 &&
                    ((const T &) v2).size() == 1)
                    v3 = v1 * v2 * Scalar(src_size);
                else
                    v3 = hsum_async(v1 * v2);

                if (((const T &) grad).valid())
                    grad += v3;
                else
                    grad = std::move(v3);
            } else {
                if (((const T &) grad).valid())
                    grad = enoki::fmadd(v1, v2, grad);
                else
                    grad = v1 * v2;
            }
        } else {
            grad = enoki::fmadd(v1, v2, grad);
        }
    }

    bool is_scalar() const { return size == 1; }

    ENOKI_ARRAY_DEFAULTS(Variable);
};

static_assert(sizeof(Variable) == ((IsDouble ? 2 : 0) + 8) * sizeof(uint32_t),
              "Variable data structure has incorrect size. Padding problem?");

/// Thread-local list used by ad_queue() and ad_traverse()
static __thread std::deque<uint32_t> *tls_queue = nullptr;

/// Records all internal application state
struct State {
    using VariableMap = tsl::robin_map<uint32_t, Variable>;
    using EdgeVector  = std::vector<Edge>;

    /// Mutex protecting the state data structure
    Mutex mutex;

    /// Hash table mapping variable IDs to variable instances
    VariableMap variables;

    /// List of all edges (used and unused ones)
    EdgeVector edges;

    /// List of currently unused edges
    std::vector<uint32_t> unused_edges;

    /// List of variables to be processed in traverse() / graphviz()
    std::vector<uint32_t> todo;

    /// Counter for variable indices
    uint32_t variable_index = 1;

    State() : edges(1) { }

    Variable *operator[](uint32_t index) {
        auto it = variables.find(index);
        if (unlikely(it == variables.end()))
            ad_fail("referenced an unknown variable %u!", index);
        return &it.value();
    }

    ~State() {
        if (!variables.empty())
            ad_log(Warn,
                   "enoki-ad: variable leak detected (%zu variables "
                   "remain in use)!", variables.size());

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

extern void RENAME(ad_whos)() {
    std::vector<uint32_t> indices;
    indices.reserve(state.variables.size());
    for (auto &kv: state.variables)
        indices.push_back(kv.first);
    std::sort(indices.begin(), indices.end());

    for (uint32_t id : indices) {
        const Variable *v = state[id];
        buffer.fmt("  %-7u ", id);
        size_t sz = buffer.fmt("%u / %u", v->ref_count_ext, v->ref_count_int);

        buffer.fmt("%*s%-12u%-8s\n", 11 - (int) sz, "", v->size,
                   v->label ? v->label : "");
    }
}

/// Forward-mode DFS starting from 'index'
static void ad_dfs_fwd(uint32_t index) {
    Variable *v = state[index];
    uint32_t edge = v->next_fwd;
    while (edge) {
        Edge &e = state.edges[edge];
        if (e.visited == 0) {
            e.visited = 1;
            ad_dfs_fwd(e.target);
        }
        edge = e.next_fwd;
    }
}


/// Reverse-mode DFS starting from 'index'
static void ad_dfs_rev(uint32_t index) {
    Variable *v = state[index];
    uint32_t edge = v->next_rev;
    while (edge) {
        Edge &e = state.edges[edge];
        if (e.visited == 0) {
            e.visited = 1;
            ad_dfs_rev(e.source);
        }
        edge = e.next_rev;
    }
}

template <typename T> void ad_enqueue(uint32_t index) {
    if (index == 0)
        return;
    std::lock_guard<Mutex> guard(state.mutex);
    std::deque<uint32_t> *queue = tls_queue;
    if (unlikely(!queue))
        queue = tls_queue = new std::deque<uint32_t>();
    queue->push_back(index);
}

/// Kahn-style topological sort in forward mode
static void ad_toposort_fwd() {
    state.todo.clear();

    std::deque<uint32_t> *queue = tls_queue;
    if (!queue || queue->empty())
        return;

    /// DFS traversal to tag all reachable edges
    for (uint32_t index: *queue)
        ad_dfs_fwd(index);

    while (!queue->empty()) {
        uint32_t index = queue->front();
        queue->pop_front();
        state.todo.push_back(index);

        uint32_t edge = state[index]->next_fwd;
        while (edge) {
            Edge &e = state.edges[edge];
            e.visited = 0;

            uint32_t edge2 = state[e.target]->next_rev;
            bool ready = true;
            while (edge2) {
                const Edge &e2 = state.edges[edge2];
                if (e2.visited) {
                    ready = false;
                    break;
                }
                edge2 = e2.next_rev;
            }

            if (ready)
                queue->push_back(e.target);

            edge = e.next_fwd;
        }
    }
}

/// Kahn-style topological sort in reverse mode
static void ad_toposort_rev() {
    state.todo.clear();

    std::deque<uint32_t> *queue = tls_queue;
    if (!queue || queue->empty())
        return;

    /// DFS traversal to tag all reachable edges
    for (uint32_t index: *queue)
        ad_dfs_rev(index);

    while (!queue->empty()) {
        uint32_t index = queue->front();
        queue->pop_front();
        state.todo.push_back(index);

        uint32_t edge = state[index]->next_rev;
        while (edge) {
            Edge &e = state.edges[edge];
            e.visited = 0;

            uint32_t edge2 = state[e.source]->next_fwd;
            bool ready = true;
            while (edge2) {
                const Edge &e2 = state.edges[edge2];
                if (e2.visited) {
                    ready = false;
                    break;
                }
                edge2 = e2.next_fwd;
            }

            if (ready)
                queue->push_back(e.source);

            edge = e.next_rev;
        }
    }
}

/// Allocate a new edge from the pool
static uint32_t ad_edge_new() {
    uint32_t index;
    if (likely(!state.unused_edges.empty())) {
        index = state.unused_edges.back();
        state.unused_edges.pop_back();
    } else {
        index = state.edges.size();
        state.edges.emplace_back();
    }
    return index;
}

/// Allocate a new variable
static std::pair<uint32_t, Variable *> ad_var_new(const char *label, uint32_t size) {
    while (true) {
        uint32_t index = state.variable_index++;

        if (unlikely(index == 0)) // overflow
            index = state.variable_index++;

        auto result = state.variables.try_emplace(index, label, size);
        if (likely(result.second))
            return { index, &result.first.value() };
    }
}

static void ad_free(uint32_t index, Variable *v);

/// Clear reverse edges of the given variable and decrease int. ref. counts
static void ad_free_edges(uint32_t index, Variable *v) {
    uint32_t edge_id = v->next_rev;
    ad_log(Trace, "ad_free_edges(): freeing edges of vertex %u", index);
    v->next_rev = 0;

    while (edge_id) {
        Edge &edge = state.edges[edge_id];

        ad_log(Trace,
               "ad_free_edges(): freeing edge %u: %u -> %u",
               edge_id, edge.source, edge.target);

        uint32_t source = edge.source,
                 next_rev = edge.next_rev,
                 next_fwd = edge.next_fwd;

        assert(edge.target == index);
        edge.reset();

        Variable *v2 = state[source];
        if (unlikely(v2->ref_count_int == 0))
            ad_fail("%u: int. reference count became negative!", source);

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

static void ad_free(uint32_t index, Variable *v) {
    ad_log(Trace, "ad_free(%u)", index);
    if (v->free_label)
        free(v->label);
    if (v->next_rev)
        ad_free_edges(index, v);
    state.variables.erase(index);
}

template <typename T>
uint32_t ad_new(const char *label, uint32_t size, uint32_t op_count,
                const uint32_t *op, T *weights) {
    std::lock_guard<Mutex> guard(state.mutex);

    auto [index, var] = ad_var_new(label, size);

    if (unlikely(log_level >= Debug)) {
        const char *l = label ? label : "unnamed";
        switch (op_count) {
            case 0:
                ad_log(Debug, "ad_new(%u): %s", index, l); break;
            case 1:
                ad_log(Debug, "ad_new(%u <- %u): %s", index, op[0], l); break;
            case 2:
                ad_log(Debug, "ad_new(%u <- %u, %u): %s", index, op[0], op[1], l); break;
            case 3:
                ad_log(Debug, "ad_new(%u <- %u, %u, %u): %s", index, op[0], op[1], op[2], l); break;
            default: break;
        }
    }

    uint32_t edge_index = 0;
    for (uint32_t i = 0; i < op_count; ++i) {
        if (op[i] == 0)
            continue;

        bool weight_is_zero = false;
        if constexpr (std::is_scalar_v<T>)
            weight_is_zero = weights[i] == 0;
        else
            weight_is_zero = weights[i].is_literal_zero();

        if (weight_is_zero)
            continue;

        Variable *var2 = state[op[i]];

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = op[i];
        edge.target = index;
        edge.weight = std::move(weights[i]);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = edge_index;
        edge_index = edge_index_new;

        var2->ref_count_int++;
        var2->next_fwd = edge_index_new;
    }

    var->next_rev = edge_index;
    var->ref_count_ext = 1;

    return index;
}


template <typename Value> struct MaskEdge : Special {
    MaskEdge(const Mask &mask, bool negate) : mask(mask), negate(negate) { }

    void backward(Variable *source, const Variable *target) const override {
        if (!negate)
            source->accum(detail::and_(target->grad, mask), target->size);
        else
            source->accum(detail::andnot_(target->grad, mask), target->size);
    }

    void forward(const Variable *source, Variable *target) const override {
        if (!negate)
            target->accum(detail::and_(source->grad, mask), source->size);
        else
            target->accum(detail::andnot_(source->grad, mask), source->size);
    }

    Mask mask;
    bool negate;
};


template <typename Value, typename Mask>
uint32_t ad_new_select(const char *label, uint32_t size, const Mask &mask_,
                       uint32_t t_index, uint32_t f_index) {
    std::lock_guard<Mutex> guard(state.mutex);
    auto [index, var] = ad_var_new(label, size);

    ad_log(Debug, "ad_new_select(%u <- %u, %u)", index, t_index, f_index);
    uint32_t op[2]= { t_index, f_index };

    uint32_t edge_index = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        if (op[i] == 0)
            continue;

        Variable *var2 = state[op[i]];

        uint32_t edge_index_new = ad_edge_new();
        Edge &edge = state.edges[edge_index_new];
        edge.source = op[i];
        edge.target = index;
        edge.special = new MaskEdge<Value>(i == 0 ? mask_ : !mask_, false);
        edge.next_fwd = var2->next_fwd;
        edge.next_rev = edge_index;
        edge_index = edge_index_new;

        var2->ref_count_int++;
        var2->next_fwd = edge_index_new;
    }

    var->next_rev = edge_index;
    var->ref_count_ext = 1;

    return index;
}

template <typename Value> struct GatherEdge : Special {
    GatherEdge(const Index &offset, const Mask &mask, bool permute)
        : offset(offset), mask(mask), permute(permute) { }

    void backward(Variable *source, const Variable *target) const override {
        Value &source_grad = (Value &) source->grad;
        uint32_t size = source->size;

        if (!source_grad.valid())
            source_grad = zero<Value>(size).copy(); // avoid issues with CSA
        else if ((uint32_t) source_grad.size() != size)
            source_grad.resize(size);

        if (permute)
            enoki::scatter(source_grad, target->grad, offset, mask);
        else
            enoki::scatter_add(source_grad, target->grad, offset, mask);
    }

    void forward(const Variable *source, Variable *target) const override {
        target->accum(enoki::gather<Value>(source->grad, offset, mask),
                      asize(offset));
    }

    Index offset;
    Mask mask;
    bool permute;
};

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather(const char *label, uint32_t size, uint32_t src_index,
                       const Index &offset, const Mask &mask, bool permute) {
    if constexpr (is_array_v<Value>) {
        std::lock_guard<Mutex> guard(state.mutex);
        auto [index, var] = ad_var_new(label, size);

        ad_log(Debug, "ad_new_gather(%u <- %u, %u, permute=%i)", index,
               src_index, (int) permute);

        Variable *var2 = state[src_index];
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

template <typename Value> struct ScatterEdge : Special {
    ScatterEdge(const Index &offset, const Mask &mask, bool scatter_add)
        : offset(offset), mask(mask), scatter_add(scatter_add) { }

    void backward(Variable *source, const Variable *target) const override {
        source->accum(enoki::gather<Value>(target->grad, offset, mask),
                      asize(offset));
    }

    void forward(const Variable *source, Variable *target) const override {
        Value &target_grad = (Value &) target->grad;
        uint32_t size = target->size;

        if (!target_grad.valid())
            target_grad = zero<Value>(size).copy(); // avoid issues with CSA
        else if ((uint32_t) target_grad.size() != size)
            target_grad.resize(size);

        if (scatter_add)
            enoki::scatter_add(target_grad, source->grad, offset, mask);
        else
            enoki::scatter(target_grad, source->grad, offset, mask);
    }

    Index offset;
    Mask mask;
    bool scatter_add;
};

template <typename Value, typename Mask, typename Index>
uint32_t ad_new_scatter(const char *label, uint32_t size, uint32_t src_index,
                        uint32_t dst_index, const Index &offset,
                        const Mask &mask, bool permute, bool scatter_add) {

    if constexpr (is_array_v<Value>) {
        std::lock_guard<Mutex> guard(state.mutex);
        Variable *var = nullptr;
        uint32_t index = 0;

        if (permute && dst_index) {
            Variable *var2 = state[dst_index];
            if (strcmp(var2->label, "scatter[permute]") == 0) {
                index = dst_index;
                var = var2;
            }
        }

        if (index == 0)
            std::tie(index, var) = ad_var_new(label, size);

        ad_log(Debug,
               "ad_new_scatter(%u <- %u, %u, permute=%i, scatter_add=%i)",
               index, src_index, dst_index, (int) permute, (int) scatter_add);

        uint32_t edge_index = 0;

        if (src_index) {
            Variable *var2 = state[src_index];
            uint32_t edge_index_new = ad_edge_new();
            Edge &edge = state.edges[edge_index_new];
            edge.source = src_index;
            edge.target = index;
            edge.special = new ScatterEdge<Value>(offset, mask, scatter_add);
            edge.next_fwd = var2->next_fwd;
            edge.next_rev = var->next_rev;
            var2->ref_count_int++;
            var2->next_fwd = edge_index_new;
            edge_index = edge_index_new;
        }

        if (dst_index && dst_index != index) {
            Variable *var2 = state[dst_index];

            uint32_t edge_index_new = ad_edge_new();
            Edge &edge2 = state.edges[edge_index_new];
            edge2.source = dst_index;
            edge2.target = index;
            edge2.next_fwd = var2->next_fwd;
            edge2.next_rev = edge_index;
            if (scatter_add || permute) {
                edge2.weight = 1;
            } else {
                Mask edge_mask = full<Mask>(false, size);
                enoki::scatter(edge_mask, Mask(true), offset, mask);
                edge2.special = new MaskEdge<Value>(edge_mask, true);
            }
            var2->ref_count_int++;
            var2->next_fwd = edge_index_new;
            edge_index = edge_index_new;
        }

        if (edge_index == 0)
            ad_fail("ad_new_scatter(): all inputs were non-differentiable!");

        var->next_rev = edge_index;
        var->ref_count_ext++;

        return index;
    } else {
        enoki_raise("ad_new_scatter(): differentiable scatters not supported "
                    "by this backend!");
    }
}

static void ad_traverse_rev(bool retain_graph) {
    ad_log(Info, "ad_traverse_rev(): processing %zu nodes ..", state.todo.size());

    for (uint32_t index : state.todo) {
        Variable *v = state[index];

        if (is_dynamic_v<Value>) {
            uint32_t grad_size = asize(v->grad);
            if (unlikely(v->size != grad_size && grad_size != 1))
                ad_fail("ad_traverse_rev(): variable %u has an invalid "
                        "gradient size: expected %u, got %u!", index,
                        v->size, grad_size);
        }

        if (unlikely(v->custom_label)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v->label);
            set_label(v->grad, tmp);
        }

        uint32_t edge_id = v->next_rev;
        while (edge_id) {
            Edge &edge = state.edges[edge_id];
            assert(edge.target == index);
            Variable *v2 = state[edge.source];

            if (unlikely(edge.special)) {
                edge.special->backward(v2, v);

                if (!retain_graph) {
                    delete edge.special;
                    edge.special = nullptr;
                }
            } else {
                v2->mul_accum(edge.weight, v->grad, v->size);

                if (!retain_graph)
                    edge.weight = Value();
            }

            edge_id = edge.next_rev;
        }

        if (v->next_rev) {
            /// Clear the grads at interior nodes
            v->grad = Value();
        }
    }

    if (!retain_graph) {
        ad_log(Info, "ad_traverse_rev(): cleaning up ..");
        for (auto it = state.todo.rbegin(); it != state.todo.rend(); ++it) {
            uint32_t index = *it;
            Variable *v = state[index];
            ad_free_edges(index, v);
        }
    }

    ad_log(Info, "ad_traverse_rev(): done.");
}

static void ad_traverse_fwd(bool retain_graph) {
    ad_log(Info, "ad_traverse_fwd(): processing %zu nodes ..", state.todo.size());

    for (uint32_t index : state.todo) {
        Variable *v = state[index];

        if (is_dynamic_v<Value>) {
            uint32_t grad_size = asize(v->grad);
            if (unlikely(v->size != grad_size && grad_size != 1))
                ad_fail("ad_traverse_rev(): variable %u has an invalid "
                        "gradient size: expected %u, got %u!", index,
                        v->size, grad_size);
        }

        if (unlikely(v->custom_label)) {
            char tmp[256];
            snprintf(tmp, 256, "%s_grad", v->label);
            set_label(v->grad, tmp);
        }

        uint32_t edge_id = v->next_fwd;
        while (edge_id) {
            Edge &edge = state.edges[edge_id];
            assert(edge.source == index);
            Variable *v2 = state[edge.target];

            if (unlikely(edge.special)) {
                edge.special->forward(v, v2);

                if (!retain_graph) {
                    delete edge.special;
                    edge.special = nullptr;
                }
            } else {
                v2->mul_accum(edge.weight, v->grad, v->size);

                if (!retain_graph)
                    edge.weight = Value();
            }

            edge_id = edge.next_fwd;
        }

        if (v->next_fwd) {
            /// Clear the grads at interior nodes
            v->grad = Value();
        }

        if (!retain_graph)
            ad_free_edges(index, v);
    }

    ad_log(Info, "ad_traverse_fwd(): done.");
}

template <typename Value> const char *ad_graphviz(bool reverse) {
    std::lock_guard<Mutex> guard(state.mutex);

    if (reverse)
        ad_toposort_rev();
    else
        ad_toposort_fwd();


    buffer.clear();
    buffer.put("digraph {\n");
    buffer.put("  rankdir=BT;\n");
    buffer.put("  graph [dpi=50];\n");
    buffer.put("  node [shape=record fontname=Consolas];\n");
    buffer.put("  edge [fontname=Consolas];\n");

    std::string current_path;
    int current_depth = 0;
    for (uint32_t index : state.todo) {
        Variable *v = state[index];

        std::string label = v->label;

        auto sepidx = label.rfind("/");
        std::string path;
        if (sepidx != std::string::npos) {
            path = label.substr(1, sepidx);
            label = label.substr(sepidx + 1);
        }

        if (current_path != path) {
            for (int i = 0; i < current_depth; ++i)
                buffer.put("  }\n");
            current_depth = 0;
            current_path = path;

            do {
                sepidx = path.find('/');
                std::string graph_label = path.substr(0, sepidx);
                if (graph_label.empty())
                    break;
                buffer.fmt("  subgraph cluster_%x {\n", (uint32_t) std::hash<std::string>()(graph_label));
                buffer.fmt("  label=\"%s\";\n", graph_label.c_str());
                ++current_depth;

                if (sepidx == std::string::npos)
                    break;
                path = path.substr(sepidx + 1, std::string::npos);
            } while (true);
        }

        const char *color = "";
        if (v->next_rev == 0)
            color = " fillcolor=salmon style=filled";
        else if (v->next_fwd == 0)
            color = " fillcolor=cornflowerblue style=filled";

        buffer.fmt("  %u [label=\"{%s%s%s%s|{#%u|E:%u|I:%u}}\"%s];\n", index,
                   v->custom_label ? "\\\"" : "", label.c_str(),
                   v->custom_label ? "\\\"" : "",
                   v->size == 1 ? " [s]" : "",
                   index, v->ref_count_ext, v->ref_count_int, color);

        uint32_t edge = v->next_rev, edge_ctr = 0;
        while (edge) {
            edge = state.edges[edge].next_rev;
            edge_ctr++;
        }

        edge = v->next_rev;
        while (edge) {
            const Edge &e = state.edges[edge];
            buffer.fmt("  %u -> %u [label=\" %u\"%s];\n", e.target, e.source,
                       edge_ctr--, e.special ? " color=red" : "");
            edge = e.next_rev;
        }
    }
    for (int i = 0; i < current_depth; ++i)
        buffer.put("  }\n");
    buffer.put("}\n");

    return buffer.get();
}

template <typename T> void ad_inc_ref_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    std::lock_guard<Mutex> guard(state.mutex);
    state[index]->ref_count_ext++;
}

template <typename T> void ad_dec_ref_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    std::lock_guard<Mutex> guard(state.mutex);
    Variable *v = state[index];
    if (unlikely(v->ref_count_ext == 0))
        ad_fail("%u: ext. reference count became negative!", index);
    if (--v->ref_count_ext == 0 && v->ref_count_int == 0)
        ad_free(index, v);
}

template <typename T> T ad_grad(uint32_t index) {
    if (index == 0)
        enoki_raise("grad(): attempted to retrieve the gradient of a "
                    "variable that was not registered with the AD "
                    "backend. Did you forget to call enable_grad()?");

    std::lock_guard<Mutex> guard(state.mutex);
    return state[index]->grad;
}

template <typename T> void ad_set_grad(uint32_t index, const T &value) {
    if (index == 0)
        enoki_raise("set_grad(): attempted to set the gradient of a "
                    "variable that was not registered with the AD "
                    "backend. Did you forget to call enable_grad()?");

    std::lock_guard<Mutex> guard(state.mutex);
    state[index]->grad = value;
}

template <typename T> void ad_set_label(uint32_t index, const char *label) {
    if (index == 0)
        return;
    std::lock_guard<Mutex> guard(state.mutex);
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
    std::lock_guard<Mutex> guard(state.mutex);
    return state[index]->label;
}

template <typename T> void ad_traverse(bool reverse, bool retain_graph) {
    std::lock_guard<Mutex> guard(state.mutex);

    if (reverse)
        ad_toposort_rev();
    else
        ad_toposort_fwd();

    if (reverse)
        ad_traverse_rev(retain_graph);
    else
        ad_traverse_fwd(retain_graph);
}

template ENOKI_EXPORT void ad_inc_ref_impl<Value>(uint32_t);
template ENOKI_EXPORT void ad_dec_ref_impl<Value>(uint32_t);
template ENOKI_EXPORT uint32_t ad_new<Value>(const char *, uint32_t, uint32_t,
                                             const uint32_t *, Value *);
template ENOKI_EXPORT Value ad_grad<Value>(uint32_t);
template ENOKI_EXPORT void ad_set_grad<Value>(uint32_t, const Value &);
template ENOKI_EXPORT void ad_set_label<Value>(uint32_t, const char *);
template ENOKI_EXPORT const char *ad_label<Value>(uint32_t);
template ENOKI_EXPORT void ad_enqueue<Value>(uint32_t);
template ENOKI_EXPORT void ad_traverse<Value>(bool, bool);
template ENOKI_EXPORT const char *ad_graphviz<Value>(bool);
template ENOKI_EXPORT uint32_t ad_new_select<Value, Mask>(
    const char *, uint32_t, const Mask &, uint32_t, uint32_t);
template ENOKI_EXPORT uint32_t ad_new_gather<Value, Mask, Index>(
    const char *, uint32_t, uint32_t, const Index &, const Mask &, bool);
template ENOKI_EXPORT uint32_t
ad_new_scatter<Value, Mask, Index>(const char *, uint32_t, uint32_t, uint32_t,
                                   const Index &, const Mask &, bool, bool);

NAMESPACE_END(detail)

template struct ENOKI_EXPORT DiffArray<detail::Value>;

NAMESPACE_END(enoki)
