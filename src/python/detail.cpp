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
 * lists, dictionaries, and custom data strutures. The purpose of this function
 * is isolate the inputs of :py:func:`drjit.while_loop()` and
 * :py:func:`drjit.if_stmt()` from changes.
 *
 * If the ``copy_map`` parameter is provided, the function furthermore
 * registers created copies, which is useful in combination with the
 * :py:func:`drjit.detail.uncopy()` function.
 *
 * This function exists for Dr.Jit-internal use. You probably should not call
 * it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */
nb::object copy(nb::handle h, CopyMap *copy_map) {
    struct CopyOp : TransformCallback {
        CopyMap *copy_map;
        CopyOp(CopyMap *copy_map) : copy_map(copy_map) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            nb::inst_replace_copy(h2, h1);
        }

        void postprocess(nb::handle h1, nb::handle h2) override {
            if (copy_map && !h1.is(h2))
                copy_map->put(h2, h1);
        }
    };

    CopyOp c(copy_map);
    return transform("drjit.detail.copy", c, h);
}

/**
 * \brief Undo a prior call to :py:func:`drjit.copy()` when the contents of a
 * PyTree are unchanged.
 *
 * This operation recursively traverses a PyTree ``h`` containing copies made
 * by the functions :py:func:`drjit.copy()` and
 * :py:func:`drjit.update_indices()`. Whenever an entire subtree was unchanged
 * (in the sense that the Dr.Jit array indices are still the same), the
 * function "undoes" the change by returning the original Python object prior
 * to the copy.
 *
 * Both :py:func:`drjit.while_loop()` and :py:func:`drjit.if_stmt()`
 * conservatively perform a deep copy of all state variables. When that copy is
 * later discovered to not be necessary, the use :py:func:`uncopy()` to restore
 * the variable to its original Python object.
 *
 * This function exists for Dr.Jit-internal use. You probably should not call
 * it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */
nb::object uncopy_impl(nb::handle h, nb::handle previous_implicit,
                       CopyMap &copy_map, bool &rebuild_parent) {
    nb::handle previous;
    while (true) {
        nb::handle tmp = copy_map.get(previous.is_valid() ? previous : h);
        if (!tmp.is_valid())
            break;
        previous = tmp;
    }

    if (!previous.is_valid())
        previous = previous_implicit;

    // Use newly constructed object if a previous version cannot be found
    bool rebuild = !previous.is_valid();

    nb::handle tp = h.type();
    nb::object result;

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.is_tensor) {
            nb::object array = uncopy_impl(
                nb::steal(s.tensor_array(h.ptr())),
                nb::steal(previous.is_valid() ? s.tensor_array(previous.ptr())
                                              : nb::handle()),
                copy_map, rebuild);
            result = tp(array, shape(h));
        } else if (s.ndim != 1) {
            result = nb::inst_alloc_zero(tp);

            dr::ArrayBase *p1 = inst_ptr(h),
                          *p2 = inst_ptr(result);

            size_t size = s.shape[0];
            if (size == DRJIT_DYNAMIC) {
                size = s.len(p1);
                s.init(size, p2);
            }

            for (size_t i = 0; i < size; ++i)
                result[i] = uncopy_impl(
                    h[i], previous.is_valid() ? previous[i] : nb::handle(),
                    copy_map, rebuild);
        } else {
            result = nb::borrow(h);
            if (previous.is_valid()) {
                if (s.index)
                    rebuild = s.index(inst_ptr(h)) != s.index(inst_ptr(previous));
                else
                    rebuild = !h.is(previous);
            }
        }
    } else if (tp.is(&PyTuple_Type)) {
        nb::tuple t = nb::borrow<nb::tuple>(h);
        size_t size = nb::len(t);
        result = nb::steal(PyTuple_New(size));
        if (!result.is_valid())
            nb::raise_python_error();
        for (size_t i = 0; i < size; ++i)
            NB_TUPLE_SET_ITEM(
                result.ptr(), i,
                uncopy_impl(t[i], nb::handle(), copy_map, rebuild).release().ptr());
    } else if (tp.is(&PyList_Type)) {
        nb::list tmp;
        for (nb::handle item : nb::borrow<nb::list>(h))
            tmp.append(uncopy_impl(item, nb::handle(), copy_map, rebuild));
        result = std::move(tmp);
    } else if (tp.is(&PyDict_Type)) {
        nb::dict tmp;
        for (auto [k, v] : nb::borrow<nb::dict>(h))
            tmp[k] = uncopy_impl(v, nb::handle(), copy_map, rebuild);
        result = std::move(tmp);
    } else {
        nb::object dstruct = nb::getattr(tp, "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
            nb::object tmp = tp();
            for (auto [k, v] : nb::borrow<nb::dict>(dstruct))
                nb::setattr(tmp, k,
                            uncopy_impl(nb::getattr(h, k), nb::handle(),
                                        copy_map, rebuild));
            result = std::move(tmp);
        } else {
            result = nb::borrow(h);
            if (previous.is_valid())
                rebuild = !h.is(previous);
        }
    }

    rebuild_parent |= rebuild;
    return rebuild ? result : nb::borrow(previous);
}

nb::object uncopy(nb::handle h, CopyMap &copy_map) {
    bool changed_parent = false;
    return uncopy_impl(h, nb::handle(), copy_map, changed_parent);
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

nb::object reduce_identity(JitBackend backend, VarType vt, ReduceOp op) {
    ArrayMeta m { };
    m.backend = (uint16_t) backend;
    m.ndim = 1;
    m.type = (uint16_t) vt;
    m.shape[0] = DRJIT_DYNAMIC;
    nb::handle tp = meta_get_type(m);
    nb::object result = nb::inst_alloc(tp);
    uint32_t index = jit_var_reduce_identity(backend, vt, op);
    supp(tp).init_index(index, inst_ptr(result));
    nb::inst_mark_ready(result);
    jit_var_dec_ref(index);
    return result;
}

/**
 * \brief Return Dr.Jit variable indices associated with the provided data structure.
 *
 * This function traverses Dr.Jit arrays, tensors, :ref:`Pytree <pytrees>` (lists,
 * tuples, dicts, custom data structures) and returns the indices of all detected
 * variables (in the order of traversal, may contain duplicates). The index
 * information is returned as a list of encoded 64 bit integers, where each
 * contains the AD variable index in the upper 32 bits and the JIT variable
 * index in the lower 32 bit.
 *
 * This function exists for Dr.Jit-internal use. You probably should not
 * call it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
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
                operator()(index_fn(inst_ptr(h)));
        }

        void operator()(uint64_t index) override {
            if (inc_ref)
                ad_var_inc_ref(index);
            result.push_back(index);
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
 * array, tensor, or :ref:`Pytree <pytrees>` (list, tuple, dict, custom data
 * structure) while replacing any detected Dr.Jit variables with new ones based
 * on the provided index vector. The function returns the resulting object,
 * while leaving the input unchanged. The output array object borrows the
 * provided array references as opposed to stealing them.
 *
 * If the ``copy_map`` parameter is provided, the function furthermore
 * registers created copies, which is useful in combination with the
 * :py:func:`drjit.detail.uncopy()` function.
 *
 * This function exists for Dr.Jit-internal use. You probably should not call
 * it in your own application code.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */
nb::object update_indices(nb::handle h, const dr::vector<uint64_t> &indices,
                          CopyMap *copy_map, bool preserve_dirty) {
    struct UpdateIndicesOp final : TransformCallback {
        const dr::vector<uint64_t> &indices;
        CopyMap *copy_map;
        bool preserve_dirty;
        size_t counter;

        UpdateIndicesOp(const dr::vector<uint64_t> &indices,
                        CopyMap *copy_map, bool preserve_dirty)
            : indices(indices), copy_map(copy_map),
              preserve_dirty(preserve_dirty), counter(0) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());
            if (s.index)
                s.init_index(operator()(s.index(inst_ptr(h1))), inst_ptr(h2));
        }

        uint64_t operator()(uint64_t index) override {
            if (counter >= indices.size())
                nb::raise("too few (%zu) indices provided", indices.size());

            uint64_t new_index = indices[counter++];
            if (preserve_dirty && jit_var_is_dirty((uint32_t) index))
                new_index = index;

            return new_index;
        }

        void postprocess(nb::handle h1, nb::handle h2) override {
            if (copy_map && !h1.is(h2))
                copy_map->put(h2, h1);
        }
    };

    if (!h.is_valid())
        return nb::object();

    UpdateIndicesOp uio { indices, copy_map, preserve_dirty };
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
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
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
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */
void check_compatibility(nb::handle h1, nb::handle h2, const char *name) {
    struct CheckCompatibility : TraversePairCallback {
        void operator()(nb::handle, nb::handle) override {
        }
    } cc;
    traverse_pair("drjit.detail.check_compatibility", cc, h1, h2, name, true);
}

static nb::handle trace_func_handle;

/// Python debug tracing callback that informs Dr.Jit about the currently executing line of code
nb::object trace_func(nb::handle frame, nb::handle, nb::handle) {
    const size_t lineno = nb::cast<size_t>(frame.attr("f_lineno"));
    const char *filename = nb::cast<const char *>(frame.attr("f_code").attr("co_filename"));
    jit_set_source_location(filename, lineno);
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

void export_detail(nb::module_ &) {
    nb::module_ d = nb::module_::import_("drjit.detail");

    nb::class_<CopyMap>(d, "CopyMap")
        .def(nb::init<>());

    d.def("collect_indices",
          [](nb::handle h) {
              dr::vector<uint64_t> result;
              collect_indices(h, result);
              return result;
          },
          doc_detail_collect_indices)

     .def("update_indices", &update_indices,
          "value"_a, "indices"_a, "copy_map"_a = nb::none(),
          "preserve_dirty"_a = false, doc_detail_update_indices)

     .def("copy", &copy, "value"_a,
          "copy_map"_a = nb::none(), doc_detail_copy)

     .def("uncopy", &uncopy, "value"_a, "copy_map"_a,
          doc_detail_uncopy)

     .def("check_compatibility", &check_compatibility,
          doc_detail_check_compatibility)

     .def("reset", &reset, doc_detail_reset)

     .def("llvm_version",
          []() {
              int major, minor, patch;
              jit_llvm_version(&major, &minor, &patch);
              return nb::str("{}.{}.{}").format(major, minor, patch);
          })

     .def("trace_func", &trace_func, "frame"_a,
          "event"_a, "arg"_a = nb::none())

     .def("clear_registry", &jit_registry_clear,
          doc_detail_clear_registry)

     .def("import_tensor",
          [](nb::handle h, bool ad) {
              dr::vector<size_t> shape;
              nb::object flat = import_ndarray(ArrayMeta{}, h.ptr(), &shape, ad);
              return tensor_t(flat.type())(
                  flat,
                  cast_shape(shape)
              );
          }, "tensor"_a, "ad"_a = false)

     .def("any_symbolic", &any_symbolic, doc_detail_any_symbolic)
     .def("reduce_identity", &reduce_identity)
     .def("cuda_compute_capability", &jit_cuda_compute_capability);

    trace_func_handle = d.attr("trace_func");
}
