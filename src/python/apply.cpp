/*
    apply.cpp -- Implementation of the internal apply() function,
    which recursively propagates operations through Dr.Jit arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "apply.h"
#include "meta.h"
#include "base.h"

static const char *op_names[] = {
    // Unary operations
    "__neg__",
    "__invert__",
    "abs",
    "sqrt",

    // Binary arithetic operations
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__lshift__",
    "__rshift__",

    "minimum",
    "maximum",

    // Binary bit/mask operations
    "__and__",
    "__or__",
    "__xor__",

    // Ternary operations
    "fma",
    "select",

    // Horizontal reductions
    "all",
    "any",

    // Miscellaneous
    "__richcmp__",
};

static_assert(sizeof(op_names) == sizeof(const char *) * (int) ArrayOp::Count,
              "op_names array is missing entries!");

static void raise_incompatible_size_error(Py_ssize_t *sizes, size_t N) {
    std::string msg = "invalid input array sizes (";
    for (size_t i = 0; i < N; ++i) {
        msg += std::to_string(sizes[i]);
        if (i + 2 < N)
            msg += ", ";
        else if (i + 2 == N)
            msg += ", and ";
    }
    msg += ")";
    throw std::runtime_error(msg);
}

/// Alternative to std::max() that also work when only a single argument is given
template <typename... Args> NB_INLINE Py_ssize_t maxv(Py_ssize_t arg, Args... args) {
    if constexpr (sizeof...(Args) > 0) {
        Py_ssize_t other = maxv(args...);
        return other > arg ? other : arg;
    } else {
        return arg;
    }
}

template <typename T1, typename T2> using first_t = T1;

template <ApplyMode Mode, typename Slot, typename... Args, size_t... Is>
PyObject *apply(ArrayOp op, Slot slot, std::index_sequence<Is...>,
                Args... args) noexcept {
    nb::object o[] = { nb::borrow(args)... };
    nb::handle tp = o[0].type();
    constexpr size_t N = sizeof...(Args);

    try {
        // All arguments must first be promoted to the same type
        if (!(o[Is].type().is(tp) && ...)) {
            promote(o, sizeof...(Args), Mode == Select);
            tp = o[Mode == Select ? 1 : 0].type();
        }

        const ArraySupplement &s = supp(tp);
        void *impl = s[op];

        if (impl == DRJIT_OP_NOT_IMPLEMENTED)
            return nb::not_implemented().release().ptr();

        ArraySupplement::Item item = s.item, item_mask = nullptr;
        ArraySupplement::SetItem set_item;
        nb::handle result_type;

        if constexpr (Mode == RichCompare) {
            raise_if(((s.is_matrix || s.is_complex || s.is_quaternion) &&
                      (slot != Py_EQ && slot != Py_NE)) ||
                         (VarType) s.type == VarType::Pointer,
                     "Inequality comparisons are only permitted on ordinary "
                     "arithmetic arrays. They are suppressed for complex "
                     "arrays, quaternions, matrices, and arrays of pointers.");
            result_type = s.mask;
            set_item = supp(result_type).set_item;
        } else if constexpr (Mode == Select) {
            result_type = tp;
            set_item = s.set_item;
            item_mask = supp(o[0].type()).item;
        } else {
            result_type = tp;
            set_item = s.set_item;
        }
        (void) item_mask;

        drjit::ArrayBase *p[N] = { inst_ptr(o[Is])... };
        nb::object result;

        // In 'InPlace' mode
        bool move = true;

        if (impl != DRJIT_OP_DEFAULT) {
            result = nb::inst_alloc(result_type);
            drjit::ArrayBase *pr = inst_ptr(result);

            if constexpr (Mode == RichCompare) {
                using Impl =
                    void (*)(const dr::ArrayBase *, const dr::ArrayBase *, int,
                             dr::ArrayBase *);
                ((Impl) impl)(p[0], p[1], slot, pr);
            } else {
                using Impl = void (*)(first_t<const dr::ArrayBase *, Args>...,
                                      dr::ArrayBase *);
                ((Impl) impl)(p[Is]..., pr);
            }

            nb::inst_mark_ready(result);
        } else {
            /// Initialize an output array of the right size. In 'InPlace'
            /// mode, try to place the output into o[0] if compatible.

            Py_ssize_t l[N], i[N] { }, lr;
            if (s.shape[0] != DRJIT_DYNAMIC) {
                ((l[Is] = s.shape[0]), ...);
                lr = s.shape[0];

                if constexpr (Mode == InPlace) {
                    result = borrow(o[0]);
                    move = false; // can directly construct output into o[0]
                } else {
                    result = nb::inst_alloc_zero(result_type);
                }
            } else {
                ((l[Is] = s.len(p[Is])), ...);
                lr = maxv(l[Is]...);

                if (((l[Is] != lr && l[Is] != 1) || ...))
                    raise_incompatible_size_error(l, N);

                if (Mode == InPlace && lr == l[0]) {
                    result = borrow(o[0]);
                    move = false; // can directly construct output into o[0]
                } else {
                    result = nb::inst_alloc(result_type);
                    s.init(lr, inst_ptr(result));
                    nb::inst_mark_ready(result);
                }
            }

            void *py_impl;
            nb::object py_impl_o;

            // Fetch pointer/handle to function to be applied recursively
            if constexpr (Mode == RichCompare) {
                py_impl = PyType_GetSlot((PyTypeObject *) s.value,
                                         Py_tp_richcompare);
            } else {
                if constexpr (std::is_same_v<Slot, int>)
                    py_impl = PyType_GetSlot((PyTypeObject *) s.value, slot);
                else
                    py_impl_o = array_module.attr(slot);
            }

            for (Py_ssize_t j = 0; j < lr; ++j) {
                // Fetch the j-th element from each array. In 'Select' mode,
                // o[0] is a mask requiring a different accessor function.
                nb::object v[] = { nb::steal(
                    ((Mode == Select && Is == 0) ? item_mask : item)(
                        o[Is].ptr(), i[Is]))... };

                raise_if(!(v[Is].is_valid() && ...), "Item retrival failed!");

                // Recurse
                nb::object vr;
                if constexpr (Mode == RichCompare) {
                    using PyImpl = PyObject *(*)(PyObject *, PyObject *, int);
                    vr = nb::steal(((PyImpl) py_impl)(v[0].ptr(), v[1].ptr(), slot));
                } else {
                    if constexpr (std::is_same_v<Slot, int>) {
                        using PyImpl = PyObject *(*)(first_t<PyObject *, Args>...);
                        vr = nb::steal(((PyImpl) py_impl)(v[Is].ptr()...));
                    } else {
                        vr = py_impl_o(v[Is]...);
                    }
                }

                raise_if(!vr.is_valid(), "Nested operation failed!");

                // Assign result
                raise_if(set_item(result.ptr(), j, vr.ptr()),
                         "Item assignment failed!");

                // Advance to next element, broadcast size-1 arrays
                ((i[Is] += (l[Is] == 1 ? 0 : 1)), ...);
            }
        }

        if constexpr (Mode == InPlace) {
            // In in-place mode, if a separate result object had to be
            // constructed, use it to now replace the contents of o[0]

            if (move) {
                nb::inst_destruct(o[0]);
                nb::inst_move(o[0], result);
                result = borrow(o[0]);
            }
        } else {
            (void) move;
        }

        return result.release().ptr();
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        if constexpr (std::is_same_v<Slot, const char *>)
            nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): failed (see above)!",
                            op_names[(int) op], tp_name.ptr());
        else
            nb::chain_error(PyExc_RuntimeError, "%U.%s(): failed (see above)!",
                            tp_name.ptr(), op_names[(int) op]);
        throw nb::python_error();
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        if constexpr (std::is_same_v<Slot, const char *>)
            nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): %s",
                            op_names[(int) op], tp_name.ptr(), e.what());
        else
            nb::chain_error(PyExc_RuntimeError, "%U.%s(): %s", tp_name.ptr(),
                            op_names[(int) op], e.what());
        return nullptr;
    }
}

void traverse(const char *op, TraverseCallback &tc, nb::handle h) {
    nb::handle tp = h.type();

    try {
        if (is_drjit_array(h)) {
            const ArraySupplement &s = supp(tp);
            if (s.ndim == 1) {
                tc(h);
            } else {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(h.ptr()));

                for (Py_ssize_t i = 0; i < len; ++i) {
                    nb::object item = nb::steal(s.item(h.ptr(), i));
                    traverse(op, tc, item);
                }
            }
        } else if (tp.is(&PyTuple_Type)) {
            for (nb::handle h2 : nb::borrow<nb::tuple>(h))
                traverse(op, tc, h2);
        } else if (tp.is(&PyList_Type)) {
            for (nb::handle h2 : nb::borrow<nb::list>(h))
                traverse(op, tc, h2);
        } else if (tp.is(&PyDict_Type)) {
            for (nb::handle h2 : nb::borrow<nb::dict>(h).values())
                traverse(op, tc, h2);
        } else {
            nb::object dstruct = nb::getattr(tp, "DRJIT_STRUCT", nb::handle());
            if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
                nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
                for (auto [k, v] : dstruct_dict)
                    traverse(op, tc, nb::getattr(h, k));
            }
        }
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "%s(): error encountered while processing an argument "
                        "of type '%U' (see above)!", op, tp_name.ptr());
        throw nb::python_error();
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        nb::chain_error(PyExc_RuntimeError,
                        "%s(): error encountered while processing an argument "
                        "of type '%U': %s", op, tp_name.ptr(), e.what());
        throw nb::python_error();
    }
}


template PyObject *apply<Normal>(ArrayOp, int, std::index_sequence<0>,
                                 PyObject *) noexcept;
template PyObject *apply<Normal>(ArrayOp, int, std::index_sequence<0, 1>,
                                 PyObject *, PyObject *) noexcept;
template PyObject *apply<Normal>(ArrayOp, int, std::index_sequence<0, 1, 2>,
                                 PyObject *, PyObject *, PyObject *) noexcept;
template PyObject *apply<Normal>(ArrayOp, const char *, std::index_sequence<0>,
                                 PyObject *) noexcept;
template PyObject *apply<Normal>(ArrayOp, const char *, std::index_sequence<0, 1>,
                                 PyObject *, PyObject *) noexcept;
template PyObject *apply<Normal>(ArrayOp, const char *, std::index_sequence<0, 1, 2>,
                                 PyObject *, PyObject *, PyObject *) noexcept;
template PyObject *apply<Select>(ArrayOp, const char *, std::index_sequence<0, 1, 2>,
                                 PyObject *, PyObject *, PyObject *) noexcept;
template PyObject *apply<RichCompare>(ArrayOp, int, std::index_sequence<0, 1>,
                                      PyObject *, PyObject *) noexcept;
template PyObject *apply<InPlace>(ArrayOp, int, std::index_sequence<0, 1>,
                                  PyObject *, PyObject *) noexcept;
