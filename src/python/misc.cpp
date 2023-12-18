/*
    misc.cpp -- Bindings for miscellaneous implementation details

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "misc.h"
#include "apply.h"
#include "shape.h"
#include "base.h"

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

void stash_ref(nb::handle h, std::vector<StashRef> &v) {
    struct StashRefOp : TraverseCallback {
        std::vector<StashRef> &v;
        StashRefOp(std::vector<StashRef> &v) : v(v) { }
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
 * Intended purely for internal Dr.Jit use, you probably should not call this in
 * your own application.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
*/
void collect_indices(nb::handle h, dr::dr_vector<uint64_t> &indices, bool inc_ref) {
    struct CollectIndices : TraverseCallback {
        dr::dr_vector<uint64_t> &result;
        bool inc_ref;
        CollectIndices(dr::dr_vector<uint64_t> &result, bool inc_ref)
            : result(result), inc_ref(inc_ref) { }

        void operator()(nb::handle h) override {
            auto index_fn = supp(h.type()).index;
            if (index_fn) {
                uint64_t index = index_fn(inst_ptr(h));
                if (inc_ref)
                    ad_var_inc_ref(index);
                result.push_back(index);
            }
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
nb::object update_indices(nb::handle h, const dr::dr_vector<uint64_t> &indices,
                          CopyMap *copy_map) {
    struct UpdateIndicesOp : TransformCallback {
        const dr::dr_vector<uint64_t> &indices;
        CopyMap *copy_map;
        size_t counter;

        UpdateIndicesOp(const dr::dr_vector<uint64_t> &indices, CopyMap *copy_map)
            : indices(indices), copy_map(copy_map), counter(0) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());
            if (s.index) {
                if (counter >= indices.size())
                    nb::raise("too few (%zu) indices provided", indices.size());

                s.init_index(indices[counter++], inst_ptr(h2));
            }
        }

        void postprocess(nb::handle h1, nb::handle h2) override {
            if (copy_map && !h1.is(h2))
                copy_map->put(h2, h1);
        }
    };

    if (!h.is_valid())
        return nb::object();

    UpdateIndicesOp uio { indices, copy_map };
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
        void operator()(nb::handle, nb::handle) override { }
    } cc;
    traverse_pair("drjit.detail.check_compatibility", cc, h1, h2, name);
}

static nb::handle trace_func_handle;

/// Python debug tracing callback that informs Dr.Jit about the currently executing line of code
nb::object trace_func(nb::handle frame, nb::handle, nb::handle) {
    const size_t lineno = nb::cast<size_t>(frame.attr("f_lineno"));
    const char *filename = nb::cast<const char *>(frame.attr("f_code").attr("co_filename"));
    jit_set_source_location(filename, lineno);
    return nb::borrow(trace_func_handle);
}

void enable_py_tracing() {
    nb::module_::import_("sys").attr("settrace")(trace_func_handle);
}

void disable_py_tracing() {
    nb::module_::import_("sys").attr("settrace")(nb::none());
}

void export_misc(nb::module_ &) {
    nb::module_ d = nb::module_::import_("drjit.detail");

    nb::class_<CopyMap>(d, "CopyMap")
        .def(nb::init<>());

    d.def("collect_indices",
          [](nb::handle h) {
              dr::dr_vector<uint64_t> result;
              collect_indices(h, result);
              return result;
          },
          doc_detail_collect_indices)

     .def("update_indices", &update_indices,
          "value"_a, "indices"_a, "copy_map"_a = nb::none(),
          doc_detail_update_indices)

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
          "event"_a, "arg"_a = nb::none());

    trace_func_handle = d.attr("trace_func");
}
