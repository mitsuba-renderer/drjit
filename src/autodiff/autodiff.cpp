#include "common.h"
#include <enoki/cuda.h>
#include <enoki/llvm.h>
#include <enoki/autodiff.h>
#include <tsl/robin_map.h>
#include <type_traits>
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
    uint32_t source = 0;

    /// Source variable index
    uint32_t target = 0;

    /// Links to the next forward edge
    uint32_t next_fwd = 0;

    /// Links to the next reverse edge
    uint32_t next_rev = 0;

    /// Pointer to a handler for "special" edges
    Special *special = nullptr;

    /// Weight value (zero/empty for "special" edges)
    Value weight{};

    ENOKI_ARRAY_DEFAULTS(Edge);

    Edge() = default;

    /// Reset the contents of this edge to the default values
    void reset() {
        delete special;
        memset(this, 0, sizeof(uint32_t) * 6);
        weight = Value();
    }

    template <typename T = Value>
    uint32_t weight_size() const {
        if constexpr (std::is_scalar_v<T>)
            return 1;
        else
            return (uint32_t) ((const T &) weight).size();
    }
};

static_assert(sizeof(Edge) == 8 * sizeof(uint32_t),
              "Edge data structure has incorrect size. Padding problem?");

/// Represents a variable in the computation graph
struct Variable {
    /// Descriptive label or nullptr
    char *label;

    /// Number of times this variable is referenced by other variables
    uint32_t ref_count_int;

    /// Number of times this variable is referenced from Python/C++
    uint32_t ref_count_ext : 29;

    /// Was the label manually overwritten via set_label()?
    uint32_t custom_label : 1;

    /// Should the label variable be freed when the Variable is deallocated?
    uint32_t free_label : 1;

    /// Was the variable queued up for via ad_schedule()?
    uint32_t scheduled : 1;

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

    template <typename T = Value>
    uint32_t grad_size() const {
        if constexpr (std::is_scalar_v<T>)
            return 1;
        else
            return (uint32_t) ((const T &) grad).size();
    }

    template <typename T = Value>
    bool grad_valid() const {
        if constexpr (std::is_scalar_v<T>)
            return true;
        else
            return ((const T &) grad).valid();
    }

    bool is_scalar() const { return size == 1; }

    ENOKI_ARRAY_DEFAULTS(Variable);
};

static_assert(sizeof(Variable) == ((IsDouble ? 2 : 0) + 8) * sizeof(uint32_t),
              "Variable data structure has incorrect size. Padding problem?");

/// Thread-local list used by ad_schedule() and ad_traverse()
static __thread std::vector<uint32_t> *thread_schedule = nullptr;

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

        if (thread_schedule) {
            delete thread_schedule;
            thread_schedule = nullptr;
        }
    }
};

static State state;

extern void RENAME(ad_whos)() {
    std::vector<uint32_t> indices;
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

/// Queue up nodes for a reverse-mode traversal
static void ad_schedule_rev(std::vector<uint32_t> *sched, uint32_t index) {
    Variable *v = state[index];
    if (v->scheduled)
        return;
    v->scheduled = 1;
    sched->push_back(index);

    uint32_t edge = v->next_rev;
    while (edge) {
        const Edge &e = state.edges[edge];
        ad_schedule_rev(sched, e.source);
        edge = e.next_rev;
    }
}

/// Queue up nodes for a forward-mode traversal
static void ad_schedule_fwd(std::vector<uint32_t> *sched, uint32_t index) {
    Variable *v = state[index];
    if (v->scheduled)
        return;
    v->scheduled = 1;
    sched->push_back(index);

    uint32_t edge = v->next_fwd;
    while (edge) {
        const Edge &e = state.edges[edge];
        ad_schedule_fwd(sched, e.target);
        edge = e.next_fwd;
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
                    if (edge2.next_fwd == edge_id) {
                        edge2.next_fwd = next_fwd;
                        break;
                    }
                    fwd = edge2.next_fwd;
                }
            }
        }

        state.unused_edges.push_back(edge_id);

        edge_id = next_rev;
    }

    v->next_rev = 0;
}

static void ad_free(uint32_t index, Variable *v) {
    ad_log(Debug, "ad_free(%u)", index);
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

template <typename Value, typename Mask>
uint32_t ad_new_select(const char *label, uint32_t size, const Mask &mask_,
                       uint32_t t_index, uint32_t f_index) {

    struct SelectEdge : Special {
        SelectEdge(const Mask &mask) : mask(mask) { }

        void backward(Variable *source, const Variable *target) const override {
            Value masked_grad = detail::and_(target->grad, mask);
            if (source->grad_valid())
                source->grad += masked_grad;
            else
                source->grad = masked_grad;
        }

        void forward(const Variable *source, Variable *target) const override {
            Value masked_grad = detail::and_(source->grad, mask);
            if (target->grad_valid())
                target->grad += masked_grad;
            else
                target->grad = masked_grad;
        }

        Mask mask;
    };

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
        edge.special = new SelectEdge(i == 0 ? mask_ : !mask_);
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

static void ad_traverse_rev(std::vector<uint32_t> *sched, bool retain_graph) {
    ad_log(Info, "ad_traverse_rev(): processing %zu nodes ..", sched->size());

    for (uint32_t index : *sched) {
        Variable *v = state[index];
        assert(v->scheduled == 1);
        ad_log(Debug, "Visiting vertex %u", index);

        if (is_dynamic_v<Value>) {
            uint32_t grad_size = (uint32_t) v->grad_size();
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
                if (is_dynamic_v<Value>) {
                    if (!v2->is_scalar() || (edge.weight_size() == 1 && v->is_scalar())) {
                        if (v2->grad_valid())
                            v2->grad = fmadd(edge.weight, v->grad, v2->grad);
                        else
                            v2->grad = edge.weight * v->grad;
                    } else {
                        Value temp = hsum_async(edge.weight * v->grad);
                        if (v2->grad_valid())
                            v2->grad += temp;
                        else
                            v2->grad = temp;
                    }
                } else {
                    v2->grad = fmadd(edge.weight, v->grad, v2->grad);
                }

                if (!retain_graph)
                    edge.weight = Value();
            }

            edge_id = edge.next_rev;
        }

        if (v->next_rev) {
            /// Clear the gradients at interior nodes
            v->grad = Value();
        }
    }

    if (!retain_graph) {
        ad_log(Info, "ad_traverse_rev(): cleaning up ..");
        for (auto it = sched->rbegin(); it != sched->rend(); ++it) {
            uint32_t index = *it;
            Variable *v = state[index];
            v->scheduled = 0;
            ad_free_edges(index, v);
        }
        sched->clear();
    }

    ad_log(Info, "ad_traverse_rev(): done.");
}

static void ad_traverse_fwd(std::vector<uint32_t> *sched, bool retain_graph) {
}

template <typename Value> const char *ad_graphviz() {
    std::lock_guard<Mutex> guard(state.mutex);
    std::vector<uint32_t> *sched = thread_schedule;
    if (!sched)
        return "digraph { }";

    buffer.clear();
    buffer.put("digraph {\n");
    buffer.put("  rankdir=BT;\n");
    buffer.put("  graph [dpi=50];\n");
    buffer.put("  node [shape=record fontname=Consolas];\n");
    buffer.put("  edge [fontname=Consolas];\n");

    std::sort(sched->begin(), sched->end(), std::less<uint32_t>());

    std::string current_path;
    int current_depth = 0;
    for (uint32_t index : *sched) {
        Variable *v = state[index];
        v->scheduled = 0;

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

        uint32_t edge = v->next_rev;
        while (edge) {
            const Edge &e = state.edges[edge];
            buffer.fmt("  %u -> %u [label=\"%u\"%s];\n", e.target, e.source,
                       edge, e.special ? " color=red" : "");
            edge = e.next_rev;
        }
    }
    for (int i = 0; i < current_depth; ++i)
        buffer.put("  }\n");
    buffer.put("}\n");
    sched->clear();

    return buffer.get();
}

template <typename T> void ad_inc_ref(uint32_t index) {
    if (index == 0)
        return;
    std::lock_guard<Mutex> guard(state.mutex);
    state[index]->ref_count_ext++;
}

template <typename T> void ad_dec_ref(uint32_t index) {
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
                    "backend. Did you forget to call requires_grad()?");

    std::lock_guard<Mutex> guard(state.mutex);
    return state[index]->grad;
}

template <typename T> void ad_set_grad(uint32_t index, const T &value) {
    if (index == 0)
        enoki_raise("set_grad(): attempted to set the gradient of a "
                    "variable that was not registered with the AD "
                    "backend. Did you forget to call requires_grad()?");

    std::lock_guard<Mutex> guard(state.mutex);
    state[index]->grad = value;
}

template <typename T> void ad_set_label(uint32_t index, const char *label) {
    std::lock_guard<Mutex> guard(state.mutex);
    Variable *v = state[index];
    if (v->free_label)
        free(v->label);
    v->label = strdup(label);
    v->free_label = true;
    v->custom_label = true;
}

template <typename T> const char *ad_label(uint32_t index) {
    std::lock_guard<Mutex> guard(state.mutex);
    return state[index]->label;
}

template <typename T> void ad_schedule(uint32_t index, bool reverse) {
    if (index == 0)
        return;
    std::lock_guard<Mutex> guard(state.mutex);
    std::vector<uint32_t> *sched = thread_schedule;
    if (!sched)
        sched = thread_schedule = new std::vector<uint32_t>();
    if (reverse)
        ad_schedule_rev(sched, index);
    else
        ad_schedule_fwd(sched, index);
}

template <typename T> void ad_traverse(bool reverse, bool retain_graph) {
    std::lock_guard<Mutex> guard(state.mutex);
    std::vector<uint32_t> *sched = thread_schedule;
    if (!sched || sched->empty())
        return;

    std::sort(sched->begin(), sched->end(), std::greater<uint32_t>());

    if (reverse)
        ad_traverse_rev(sched, retain_graph);
    else
        ad_traverse_fwd(sched, retain_graph);
}

template ENOKI_EXPORT void ad_inc_ref<Value>(uint32_t);
template ENOKI_EXPORT void ad_dec_ref<Value>(uint32_t);
template ENOKI_EXPORT uint32_t ad_new<Value>(const char *, uint32_t, uint32_t,
                                             const uint32_t *, Value *);
template ENOKI_EXPORT Value ad_grad<Value>(uint32_t);
template ENOKI_EXPORT void ad_set_grad<Value>(uint32_t, const Value &);
template ENOKI_EXPORT void ad_set_label<Value>(uint32_t, const char *);
template ENOKI_EXPORT const char *ad_label<Value>(uint32_t);
template ENOKI_EXPORT void ad_schedule<Value>(uint32_t, bool);
template ENOKI_EXPORT void ad_traverse<Value>(bool, bool);
template ENOKI_EXPORT const char *ad_graphviz<Value>();
template ENOKI_EXPORT uint32_t ad_new_select<Value, Mask>(
    const char *, uint32_t, const Mask &, uint32_t, uint32_t);

NAMESPACE_END(detail)

template struct ENOKI_EXPORT DiffArray<detail::Value>;

NAMESPACE_END(enoki)
