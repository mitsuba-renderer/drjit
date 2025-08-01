/*
    detail.cpp -- Bindings for miscellaneous implementation details

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "detail.h"
#include "apply.h"
#include "shape.h"
#include "base.h"
#include "meta.h"
#include "init.h"
#include "traits.h"

/**
 * \brief Create a deep copy of a PyTree
 *
 * This function recursively traverses PyTrees and replaces Dr.Jit arrays with
 * copies created via the ordinary copy constructor. It also rebuilds tuples,
 * lists, dictionaries, and custom data strutures.
 *
 * This function exists for Dr.Jit-internal use. You probably should not call
 * it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.rst -- please keep
 * them in sync in case you make a change here)
 */
nb::object copy(nb::handle h) {
    struct CopyOp : TransformCallback {
        void operator()(nb::handle h1, nb::handle h2) override {
            nb::inst_replace_copy(h2, h1);
        }
    };

    CopyOp c;
    return transform("drjit.detail.copy", c, h);
}

void stash_ref(nb::handle h, dr::vector<StashRef> &v) {
    struct StashRefOp : TraverseCallback {
        dr::vector<StashRef> &v;
        StashRefOp(dr::vector<StashRef> &v) : v(v) { }
        void operator()(nb::handle h) override {
            auto index_fn = supp(h.type()).index;
            if (!index_fn)
                return;
            v.emplace_back((uint32_t) index_fn(inst_ptr(h)));
        }
    };

    StashRefOp vo(v);
    traverse("drjit.detail.stash_ref", vo, h);
}

nb::object reduce_identity(nb::type_object_t<dr::ArrayBase> tp, ReduceOp op, uint32_t size) {
    const ArraySupplement &s = supp(tp);

    ArrayMeta m { };
    m.backend = (uint64_t)JitBackend::None;
    m.ndim = 1;
    m.type = s.type;
    m.shape[0] = DRJIT_DYNAMIC;
    nb::handle tp2 = meta_get_type(m);
    const ArraySupplement &s2 = supp(tp2);

    nb::object id_elem = nb::inst_alloc(tp2);
    uint64_t value = jit_reduce_identity((VarType) s2.type, op);
    s2.init_data(1, &value, inst_ptr(id_elem));
    nb::inst_mark_ready(id_elem);

    nb::object result = nb::inst_alloc(tp);
    s.init_const(size, false, id_elem[0].ptr(), inst_ptr(result));
    nb::inst_mark_ready(result);

    return result;
}

bool can_scatter_reduce(nb::type_object_t<dr::ArrayBase> tp, ReduceOp op) {
    const ArraySupplement &s = supp(tp);
    return jit_can_scatter_reduce((JitBackend) s.backend, (VarType) s.type, op);
}

/**
 * \brief Return Dr.Jit variable indices associated with the provided data structure.
 *
 * This function traverses Dr.Jit arrays, tensors, :ref:`PyTree <pytrees>` (lists,
 * tuples, dicts, custom data structures) and returns the indices of all detected
 * variables (in the order of traversal, may contain duplicates). The index
 * information is returned as a list of encoded 64 bit integers, where each
 * contains the AD variable index in the upper 32 bits and the JIT variable
 * index in the lower 32 bit.
 *
 * This function exists for Dr.Jit-internal use. You probably should not
 * call it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.rst -- please keep
 * them in sync in case you make a change here)
*/
void collect_indices(nb::handle h, dr::vector<uint64_t> &indices, bool inc_ref) {
    struct CollectIndices final : TraverseCallback {
        dr::vector<uint64_t> &result;
        bool inc_ref;

        CollectIndices(dr::vector<uint64_t> &result, bool inc_ref)
            : result(result), inc_ref(inc_ref) { }

        void operator()(nb::handle h) override {
            auto index_fn = supp(h.type()).index;
            if (index_fn)
                operator()(index_fn(inst_ptr(h)), nullptr, nullptr);
        }

        uint64_t operator()(uint64_t index, const char *,
                            const char *) override {
            if (inc_ref)
                ad_var_inc_ref(index);
            result.push_back(index);
            return 0;
        }
    };

    if (!h.is_valid())
        return;

    CollectIndices ci { indices, inc_ref };
    traverse("drjit.detail.collect_indices", ci, h);
}

/**
 * \brief Create a copy of the provided input while replacing Dr.Jit variables
 * with new ones based on a provided set of indices.
 *
 * This function works analogously to ``collect_indices``, except that it
 * consumes an index array and produces an updated output.
 *
 * It recursively traverses and copies an input object that may be a Dr.Jit
 * array, tensor, or :ref:`PyTree <pytrees>` (list, tuple, dict, custom data
 * structure) while replacing any detected Dr.Jit variables with new ones based
 * on the provided index vector. The function returns the resulting object,
 * while leaving the input unchanged. The output array object borrows the
 * provided array references as opposed to stealing them.
 *
 * This function exists for Dr.Jit-internal use. You probably should not call
 * it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.rst -- please keep
 * them in sync in case you make a change here)
 */
nb::object update_indices(nb::handle h, const dr::vector<uint64_t> &indices) {
    struct UpdateIndicesOp final : TransformCallback {
        const dr::vector<uint64_t> &indices;
        size_t counter;

        UpdateIndicesOp(const dr::vector<uint64_t> &indices)
            : indices(indices), counter(0) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());
            if (s.index)
                s.init_index(operator()(s.index(inst_ptr(h1))), inst_ptr(h2));
        }

        uint64_t operator()(uint64_t) override {
            if (counter >= indices.size())
                nb::raise("too few (%zu) indices provided", indices.size());

            return indices[counter++];
        }
    };

    if (!h.is_valid())
        return nb::object();

    UpdateIndicesOp uio { indices };
    nb::object result = transform("drjit.detail.update_indices", uio, h);

    if (uio.counter != indices.size())
        nb::raise("drjit.detail.update_indices(): too many indices "
                  "provided (needed %zu, got %zu)!",
                  uio.counter, indices.size());

    return result;
}

/**
 * \brief Release all Jit variables in a PyTree
 *
 * This function recursively traverses PyTrees and replaces Dr.Jit arrays with
 * empty instances of the same type. :py:func:`drjit.while_loop` uses this
 * function internally to release references held by a temporary copy of the
 * state tuple.
 *
 * (Note: this explanation is also part of src/python/docstr.rst -- please keep
 * them in sync in case you make a change here)
 */
nb::object reset(nb::handle h) {
    struct Reset : TransformCallback {
        void operator()(nb::handle, nb::handle) override { }
    };

    Reset r;
    return transform("drjit.detail.reset", r, h);
}

/**
 * \brief Traverse two pytrees in parallel and ensure that they have an
 * identical structure.
 *
 * Raises an exception is a mismatch is found (e.g., different types, arrays with
 * incompatible numbers of elements, dictionaries with different keys, etc.)
 *
 * When the ``width_consistency`` argument is enabled, an exception will also be
 * raised if there is a mismatch of the vectorization widths of any Dr.Jit type
 * in the pytrees.
 *
 * (Note: this explanation is also part of src/python/docstr.rst -- please keep
 * them in sync in case you make a change here)
 */
void check_compatibility(nb::handle h1, nb::handle h2, bool width_consistency, const char *name) {
    struct CheckCompatibility : TraversePairCallback {
        void operator()(nb::handle, nb::handle) override {
        }
    } cc;
    traverse_pair("drjit.detail.check_compatibility", cc, h1, h2, name, true, width_consistency);
}

static nb::handle trace_func_handle;

/// Python debug tracing callback that informs Dr.Jit about the currently executing line of code
nb::object trace_func(nb::handle frame, nb::handle, nb::handle) {
    const auto f_lineno = frame.attr("f_lineno");
    if (f_lineno.is_none()) {
        // No valid source location available.
        jit_set_source_location(nullptr, (size_t) -1);
    } else {
        const size_t lineno = nb::cast<size_t>(f_lineno);
        const char *filename = nb::cast<const char *>(frame.attr("f_code").attr("co_filename"));
        jit_set_source_location(filename, lineno);
    }
    return nb::borrow(trace_func_handle);
}

/**
 * \brief Returns ``true`` if any of the values in the provided PyTree
 * are symbolic variables.
 */
bool any_symbolic(nb::handle h) {
    struct AnySymbolic : TraverseCallback {
        bool result = false;

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (!s.index)
                return;

            uint32_t index = (uint32_t) s.index(inst_ptr(h));
            if (!index)
                return;

            if (jit_var_state(index) == VarState::Symbolic)
                result = true;
        }
    };

    AnySymbolic as;
    traverse("drjit.detail.any_symbolic()", as, h);
    return as.result;
}

void enable_py_tracing() {
    nb::module_::import_("sys").attr("settrace")(trace_func_handle);
}

void disable_py_tracing() {
    nb::module_::import_("sys").attr("settrace")(nb::none());
}

void set_leak_warnings(bool value) {
    nb::set_leak_warnings(value);
    jit_set_leak_warnings(value);
    ad_set_leak_warnings(value);
}

bool leak_warnings() {
    return nb::leak_warnings() || jit_leak_warnings() || ad_leak_warnings();
}

// Have to wrap this in an unnamed namespace to prevent collisions with the
// other declaration of ``recursion_guard``.
namespace {
static int recursion_level = 0;

// PyTrees could theoretically include cycles. Catch infinite recursion below
struct recursion_guard {
    recursion_guard() {
        if (recursion_level >= 50) {
            PyErr_SetString(PyExc_RecursionError, "runaway recursion detected");
            nb::raise_python_error();
        }
        // NOTE: the recursion_level has to be incremented after potentially
        // throwing an exception, as throwing an exception in the constructor
        // prevents the destructor from being called.
        recursion_level++;
    }
    ~recursion_guard() { recursion_level--; }
};
} // namespace

/**
 * \brief Traverses all variables of a python object.
 *
 * This function is used to traverse variables of python objects, inheriting
 * from trampoline classes. This allows the user to freeze a custom python
 * version of a C++ class, without having to declare its variables.
 */
void traverse_py_cb_ro_impl(nb::handle self, nb::callable c) {
    recursion_guard guard;

    struct PyTraverseCallback : TraverseCallback {
        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            auto index_fn = s.index;
            if (index_fn){
                if (s.is_class){
                    nb::str variant =
                        nb::borrow<nb::str>(nb::getattr(h, "Variant"));
                    nb::str domain =
                        nb::borrow<nb::str>(nb::getattr(h, "Domain"));
                    operator()(index_fn(inst_ptr(h)), variant.c_str(),
                               domain.c_str());
                }
                else
                    operator()(index_fn(inst_ptr(h)), "", "");
            }
        }
        uint64_t operator()(uint64_t index, const char *variant,
                            const char *domain) override {
            m_callback(index, variant, domain);
            return 0;
        }
        nb::callable m_callback;

        PyTraverseCallback(nb::callable c) : m_callback(c) {}
    };

    PyTraverseCallback traverse_cb(std::move(c));

    auto dict    = nb::borrow<nb::dict>(nb::getattr(self, "__dict__"));
    for (auto value : dict.values())
        traverse("traverse_py_cb_ro", traverse_cb, value);
}

/**
 * \brief Traverses all variables of a python object.
 *
 * This function is used to traverse variables of python objects, inheriting
 * from trampoline classes. This allows the user to freeze a custom python
 * version of a C++ class, without having to declare its variables.
 */
void traverse_py_cb_rw_impl(nb::handle self, nb::callable c) {
    recursion_guard guard;

    struct PyTraverseCallback : TraverseCallback {
        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            auto index_fn            = s.index;
            if (index_fn){
                uint64_t new_index;
                if (s.is_class) {
                    nb::str variant =
                        nb::borrow<nb::str>(nb::getattr(h, "Variant"));
                    nb::str domain = nb::borrow<nb::str>(nb::getattr(h, "Domain"));
                    new_index   = operator()(index_fn(inst_ptr(h)),
                                           variant.c_str(), domain.c_str());
                } else
                    new_index = operator()(index_fn(inst_ptr(h)), "", "");
                s.reset_index(new_index, inst_ptr(h));
            }
        }
        uint64_t operator()(uint64_t index, const char *variant, const char *domain) override {
            return nb::cast<uint64_t>(m_callback(index, variant, domain));
        }
        nb::callable m_callback;

        PyTraverseCallback(nb::callable c) : m_callback(c) {}
    };

    PyTraverseCallback traverse_cb(std::move(c));

    auto dict = nb::borrow<nb::dict>(nb::getattr(self, "__dict__"));
    for (auto value : dict.values())
        traverse("traverse_py_cb_rw", traverse_cb, value, true);
}

void export_detail(nb::module_ &) {
    nb::module_ d = nb::module_::import_("drjit.detail");

    d.def("collect_indices",
          [](nb::handle h) {
              dr::vector<uint64_t> result;
              collect_indices(h, result);
              return result;
          },
          doc_detail_collect_indices)

     .def("update_indices", &update_indices, "value"_a, "indices"_a,
          doc_detail_update_indices)

     .def("copy", &copy, "value"_a, doc_detail_copy)

     .def("check_compatibility", &check_compatibility,
          doc_detail_check_compatibility)

     .def("reset", &reset, doc_detail_reset)

     .def("llvm_version",
          []() {
              int major, minor, patch;
              jit_llvm_version(&major, &minor, &patch);
              return nb::make_tuple(major, minor, patch);
          })

     .def("cuda_version",
          []() {
              int major, minor;
              jit_cuda_version(&major, &minor);
              return nb::make_tuple(major, minor);
          })

     .def("trace_func", &trace_func, "frame"_a, "event"_a,
          "arg"_a = nb::none())

     .def("clear_registry", &jit_registry_clear, doc_detail_clear_registry)

     .def("import_tensor",
          [](nb::handle h, bool ad) {
              dr::vector<size_t> shape;
              nb::object flat =
                  import_ndarray(ArrayMeta{}, h.ptr(), &shape, ad);
              return tensor_t(flat.type())(flat, cast_shape(shape));
          },
          "tensor"_a, "ad"_a = false)

     .def("any_symbolic", &any_symbolic, doc_detail_any_symbolic)

     .def("reduce_identity", &reduce_identity,
          nb::sig("def reduce_identity(dtype: typing.Type[drjit.ArrayT], op: drjit.ReduceOp, size: int = 1, /) -> drjit.ArrayT"),
          doc_detail_reduce_identity, "dtype"_a, "op"_a, "size"_a = 1)

     .def("can_scatter_reduce", &can_scatter_reduce, doc_detail_can_scatter_reduce)

     .def("cuda_compute_capability", &jit_cuda_compute_capability)

     .def("new_scope", &jit_new_scope, "backend"_a, doc_detail_new_scope)
     .def("scope", &jit_scope, "backend"_a, doc_detail_scope)
     .def("set_scope", &jit_set_scope, "backend"_a, "scope"_a, doc_detail_set_scope);

#if defined(DRJIT_DISABLE_LEAK_WARNINGS)
    set_leak_warnings(false);
#endif

    d.def("leak_warnings", &leak_warnings, doc_leak_warnings);
    d.def("set_leak_warnings", &set_leak_warnings, doc_set_leak_warnings);
    d.def("traverse_py_cb_ro", &traverse_py_cb_ro_impl);
    d.def("traverse_py_cb_rw", traverse_py_cb_rw_impl);

    nb::enum_<AllocType>(d, "AllocType")
        .value("Host", AllocType::Host)
        .value("HostAsync", AllocType::HostAsync)
        .value("HostPinned", AllocType::HostPinned)
        .value("Device", AllocType::Device);

    d.def("malloc_watermark", &jit_malloc_watermark,
          "Return the peak memory usage (watermark) for a given allocation type");
    d.def("malloc_clear_statistics", &jit_malloc_clear_statistics,
          "Clear memory allocation statistics");
    d.def("launch_stats", []() {
        size_t launches, soft_misses, hard_misses;
        jit_launch_stats(&launches, &soft_misses, &hard_misses);
        return nb::make_tuple(launches, soft_misses, hard_misses);
    }, "Return kernel launch statistics (launches, soft_misses, hard_misses)");

    trace_func_handle = d.attr("trace_func");
}
