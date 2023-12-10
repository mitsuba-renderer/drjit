/*
    misc.cpp -- Bindings for miscellaneous implementation details

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "misc.h"
#include "apply.h"
#include "base.h"

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
void collect_indices(nb::handle h, dr::dr_vector<uint64_t> &indices) {
    struct CollectIndices : TraverseCallback {
        dr::dr_vector<uint64_t> &result;
        CollectIndices(dr::dr_vector<uint64_t> &result) : result(result) { }

        void operator()(nb::handle h) override {
            auto index = supp(h.type()).index;
            if (index)
                result.push_back(index(inst_ptr(h)));
        }
    };

    if (!h.is_valid())
        return;

    CollectIndices ci { indices };
    traverse("drjit.detail.collect_indices", ci, h);
}

dr::dr_vector<uint64_t> collect_indices(nb::handle h) {
    dr::dr_vector<uint64_t> result;
    collect_indices(h, result);
    return result;
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
 * Intended purely for internal Dr.Jit use, you probably should not call this in
 * your own application.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */

nb::object update_indices(nb::handle h, const dr::dr_vector<uint64_t> &indices_) {
    struct UpdateIndices : TransformCallback {
        const dr::dr_vector<uint64_t> &indices;
        size_t counter = 0;
        UpdateIndices(const dr::dr_vector<uint64_t> &indices) : indices(indices) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());
            if (s.index) {
                if (counter >= indices.size())
                    nb::raise("too few (%zu) indices provided", indices.size());

                s.init_index(indices[counter++], inst_ptr(h2));
            }
        }
    };

    if (!h.is_valid())
        return nb::object();

    UpdateIndices ui { indices_ };
    nb::object result = transform("drjit.detail.update_indices", ui, h);

    if (ui.counter != indices_.size())
        nb::raise("drjit.detail.update_indices(): too many indices "
                  "provided (needed %zu, got %zu)!",
                  ui.counter, indices_.size());

    return result;
}

/**
 * \brief Create a deep copy of a PyTree
 *
 * This function recursively traverses PyTrees. It replaces Dr.Jit arrays with
 * copies that are created via the ordinary copy constructor. This function is
 * used to isolate the input variables of drjit.while_loop() from changes.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */
nb::object copy(nb::handle h) {
    struct Copy : TransformCallback {
        void operator()(nb::handle h1, nb::handle h2) override {
            nb::inst_replace_copy(h2, h1);
        }
    };

    Copy c;
    return transform("drjit.detail.copy", c, h);
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

/**
 * \brief Undo a prior call to :py:func:`drjit.copy()` when the contents
 * of a PyTree are unchanged.
 *
 * This operation recursively traverses a PyTree ``arg1`` that was previously
 * copied from a PyTree ``arg0`` (using :py:func:`drjit.copy()`) and then
 * potentially modified. Whenever an entire subtree was unchanged (in the sense
 * that the Dr.Jit array indices are still the same), the function "undoes" the
 * change by returning the original Python object prior to the copy.
 *
 * This function is internally used by :py:func:`drjit.while_loop()` and
 * :py:func:`drjit.if_stmt()`, which both conservatively perform a deep copy of
 * all state variables. When that copy is later discovered to not be necessary,
 * the use :py:func:`uncopy()` to restore the variable to its original Python
 * object.
 *
 * (Note: this explanation is also part of src/python/docstr.h -- please keep
 * them in sync in case you make a change here)
 */
nb::object uncopy(nb::handle arg0, nb::handle arg1) {
    nb::handle tp1 = arg0.type(), tp2 = arg1.type();

    try {
        if (!tp1.is(tp2))
            nb::raise("incompatible types (%s and %s).",
                      nb::type_name(tp1).c_str(),
                      nb::type_name(tp2).c_str());

        bool unchanged = true;
        if (is_drjit_type(tp1)) {
            const ArraySupplement &s = supp(tp1);

            if (s.is_tensor) {
                nb::object a1 = nb::steal(s.tensor_array(arg0.ptr())),
                           a2 = nb::steal(s.tensor_array(arg1.ptr())),
                           a3 = uncopy(a1, a2);
                unchanged = a3.is(a1);
            } else if (s.ndim > 1) {
                Py_ssize_t l1 = s.shape[0], l2 = l1;

                if (l1 == DRJIT_DYNAMIC) {
                    l1 = s.len(inst_ptr(arg0));
                    l2 = s.len(inst_ptr(arg1));
                }

                if (l1 != l2)
                    nb::raise("incompatible input lengths (%zu and %zu).", l1, l2);

                for (Py_ssize_t i = 0; i < l1; ++i) {
                    nb::object o1 = nb::steal(s.item(arg0.ptr(), i)),
                               o2 = nb::steal(s.item(arg1.ptr(), i)),
                               o3 = uncopy(o1, o2);
                    if (!o3.is(o1)) {
                        unchanged = false;
                        break;
                    }
                }
            } else  {
                unchanged = s.index(inst_ptr(arg0)) == s.index(inst_ptr(arg1));
            }

            return unchanged ? nb::borrow(arg0) : nb::borrow(arg1);
        } else if (tp1.is(&PyTuple_Type) || tp1.is(&PyList_Type)) {
            size_t l1 = nb::len(arg0), l2 = nb::len(arg1);
            if (l1 != l2)
                nb::raise("incompatible list/tuple size (%zu and %zu)", l1, l2);

            nb::list result;
            for (size_t i = 0; i < l1; ++i) {
                nb::object o1 = arg0[i], o2 = arg1[i], o3 = uncopy(o1, o2);
                if (!o3.is(o1))
                    unchanged = false;
                result.append(o3);
            }

            return unchanged ? nb::borrow(arg0) : result;
        } else if (tp1.is(&PyDict_Type)) {
            nb::object k1 = nb::borrow<nb::dict>(arg0).keys(),
                       k2 = nb::borrow<nb::dict>(arg1).keys();

            if (!k1.equal(k2))
                nb::raise("incompatible dictionary keys (%s and %s)",
                          nb::str(k1).c_str(), nb::str(k2).c_str());

            nb::dict result;
            for (nb::handle k : k1) {
                nb::object o1 = arg0[k], o2 = arg1[k], o3 = uncopy(o1, o2);
                if (!o3.is(o1))
                    unchanged = false;
                result[k] = o3;
            }

            return unchanged ? nb::borrow(arg0) : result;
        } else {
            nb::object dstruct = nb::getattr(tp1, "DRJIT_STRUCT", nb::handle());
            if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
                nb::object result = tp1();
                for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                    nb::object o1 = nb::getattr(arg0, k),
                               o2 = nb::getattr(arg1, k),
                               o3 = uncopy(o1, o2);
                    if (!o3.is(o1))
                        unchanged = false;
                    nb::setattr(result, k, o3);
                }
                return unchanged ? nb::borrow(arg0) : result;
            } else {
                return nb::borrow(arg1);
            }
        }
    } catch (nb::python_error &e) {
        nb::str tp1_name = nb::type_name(tp1),
                tp2_name = nb::type_name(tp2);
        nb::raise_from(
            e, PyExc_RuntimeError,
            "drjit.detail.uncopy(<%U>, <%U>): encountered an exception (see above).",
            tp1_name.ptr(), tp2_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp1_name = nb::type_name(tp1),
                tp2_name = nb::type_name(tp2);
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.detail.uncopy(<%U>, <%U>): %s.", tp1_name.ptr(),
                        tp2_name.ptr(), e.what());
        nb::raise_python_error();
    }
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

void export_misc(nb::module_ &m) {
    nb::module_ detail = nb::module_::import_("drjit.detail");
    detail.def("collect_indices",
               nb::overload_cast<nb::handle>(&collect_indices),
               doc_detail_collect_indices);
    detail.def("update_indices", &update_indices, doc_detail_update_indices);
    detail.def("check_compatibility", &check_compatibility,
               doc_detail_check_compatibility);
    m.def("copy", &copy, doc_copy);
    detail.def("uncopy", &uncopy, doc_detail_uncopy);
    detail.def("reset", &reset, doc_detail_reset);
    detail.def("llvm_version", []() {
        int major, minor, patch;
        jit_llvm_version(&major, &minor, &patch);
        return nb::str("{}.{}.{}").format(major, minor, patch);
    });
    detail.def("trace_func", &trace_func, "frame"_a, "event"_a, "arg"_a = nb::none());
    trace_func_handle = detail.attr("trace_func");
}
