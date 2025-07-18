/*
    apply.cpp -- Implementation of the internal ``apply()``, ``traverse()``,
    and ``transform()`` functions, which recursively perform operations on
    Dr.Jit arrays and Python object trees ("pytrees")

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "apply.h"
#include "meta.h"
#include "base.h"
#include "memop.h"
#include "shape.h"
#include "init.h"
#include <algorithm>

static const char *op_names[] = {
    // Unary operations
    "__neg__",
    "__invert__",
    "abs",
    "sqrt",
    "rcp",
    "rsqrt",
    "cbrt",

    "popcnt",
    "lzcnt",
    "tzcnt",
    "brev",

    "exp",
    "exp2",
    "log",
    "log2",

    "sin",
    "cos",
    "sincos",
    "tan",
    "asin",
    "acos",
    "atan",

    "sinh",
    "cosh",
    "sincosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",

    "erf",
    "erfinv",
    "lgamma",

    "round",
    "trunc",
    "ceil",
    "floor",

    // Binary arithetic operations
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__lshift__",
    "__rshift__",

    "mul_hi",
    "mul_wide",

    "minimum",
    "maximum",
    "atan2",

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
    "count",
    "sum",
    "prod",
    "min",
    "max",

    // Miscellaneous
    "__richcmp__",
};

static_assert(sizeof(op_names) == sizeof(const char *) * (int) ArrayOp::OpCount,
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

/// Forward declaration: specialization of apply() for tensor arguments
template <ApplyMode Mode, typename Func, typename... Args, size_t... Is>
PyObject *apply_tensor(ArrayOp op, Func func, std::index_sequence<Is...>,
                Args... args) noexcept;


/// Alternative to std::max() that also work when only a single argument is given
template <typename... Args> NB_INLINE Py_ssize_t maxv(Py_ssize_t arg, Args... args) {
    if constexpr (sizeof...(Args) > 0) {
        Py_ssize_t other = maxv(args...);
        return other > arg ? other : arg;
    } else {
        return arg;
    }
}


namespace detail {

struct Arg;
template <typename...> struct first {
    using type = Arg;
};

template <typename T, typename... Ts> struct first<T, Ts...> {
    using type = T;
};

}

template <typename... Ts> using first_t = typename detail::first<Ts...>::type;
template <typename... T>
nb::handle first(nb::handle h, T&...) { return h; }


/**
 * A significant portion of Dr.Jit operations pass through the central apply()
 * function below. It performs arithmetic operation (e.g. addition, FMA) by
 *
 * 1.  Casting operands into compatible representations, and
 * 2a. Calling an existing "native" implementation of the operation if
 *     available (see drjit/python.h), or alternatively:
 * 2b. Executing a fallback loop that recursively invokes the operation
 *     on array elements.
 *
 * The ApplyMode template parameter slightly adjusts the functions' operation
 * (see the definition of the ApplyMode enumeration for details).
 */
template <ApplyMode Mode, typename Slot, typename... Args, size_t... Is>
PyObject *apply(ArrayOp op, Slot slot, std::index_sequence<Is...> is,
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
        if (s.is_tensor)
            return apply_tensor<Mode, Slot>(op, slot, is, o[Is].ptr()...);

        void *impl = s[op];

        if (impl == DRJIT_OP_NOT_IMPLEMENTED) {
            if constexpr (std::is_same_v<Slot, int>)
                return nb::not_implemented().release().ptr();
            else
                nb::raise("operation not supported for this type.");
        }

        ArraySupplement::Item item = s.item, item_mask = nullptr;
        ArraySupplement::SetItem set_item;
        ArraySupplement::Init init;
        nb::handle result_type;

        if constexpr (Mode == RichCompare) {
            raise_if(((s.is_matrix || s.is_complex || s.is_quaternion) &&
                      (slot != Py_EQ && slot != Py_NE)) ||
                         (VarType) s.type == VarType::Pointer,
                     "inequality comparisons are only permitted on ordinary "
                     "arithmetic arrays. They are suppressed for complex "
                     "arrays, quaternions, matrices, and arrays of pointers.");
            result_type = s.mask;
            const ArraySupplement &s2 = supp(result_type);
            set_item = s2.set_item;
            init = s2.init;
        } else if constexpr (Mode == Select) {
            result_type = tp;
            set_item = s.set_item;
            item_mask = supp(o[0].type()).item;
            init = s.init;
        } else if constexpr (Mode == MulWide) {
            ArrayMeta m2 = s;
            switch ((VarType) s.type) {
                case VarType::Int32: m2.type = (uint16_t) VarType::Int64; break;
                case VarType::UInt32: m2.type = (uint16_t) VarType::UInt64; break;
                default: nb::raise("only signed/unsigned 32 bit integer types are supported.");
            }
            result_type = meta_get_type(m2);
            const ArraySupplement &s2 = supp(result_type);
            set_item = s2.set_item;
            init = s2.init;
        } else {
            result_type = tp;
            set_item = s.set_item;
            init = s.init;
        }
        (void) item_mask;

        drjit::ArrayBase *p[N] = { inst_ptr(o[Is])... };
        nb::object result;

        // In 'InPlace' mode, try to update the 'self' argument when it makes sense
        bool move = Mode == InPlace && o[0].is(first(args...));

        if (impl != DRJIT_OP_DEFAULT) {
            result = nb::inst_alloc(result_type);
            drjit::ArrayBase *pr = inst_ptr(result);

            if constexpr (Mode == RichCompare) {
                using Impl =
                    void (*)(const ArrayBase *, const ArrayBase *, int,
                             ArrayBase *);
                ((Impl) impl)(p[0], p[1], slot, pr);
            } else {
                using Impl = void (*)(first_t<const ArrayBase *, Args>...,
                                      ArrayBase *);
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
                ((l[Is] = (Py_ssize_t) s.len(p[Is])), ...);
                lr = maxv(l[Is]...);

                if (((l[Is] != lr && l[Is] != 1) || ...))
                    raise_incompatible_size_error(l, N);

                if (Mode == InPlace && lr == l[0]) {
                    result = borrow(o[0]);
                    move = false; // can directly construct output into o[0]
                } else {
                    result = nb::inst_alloc(result_type);
                    init(lr, inst_ptr(result));
                    nb::inst_mark_ready(result);
                }
            }

            void *py_impl;
            nb::object py_impl_o;

            // Fetch pointer/handle to function to be applied recursively
            if constexpr (Mode == RichCompare) {
                py_impl = nb::type_get_slot((PyTypeObject *) s.value,
                                            Py_tp_richcompare);
            } else {
                if constexpr (std::is_same_v<Slot, int>)
                    py_impl = nb::type_get_slot((PyTypeObject *) s.value, slot);
                else
                    py_impl_o = array_module.attr(slot);
            }
            (void) py_impl;

            for (Py_ssize_t j = 0; j < lr; ++j) {
                // Fetch the j-th element from each array. In 'Select' mode,
                // o[0] is a mask requiring a different accessor function.
                nb::object v[] = { nb::steal(
                    ((Mode == Select && Is == 0) ? item_mask : item)(
                        o[Is].ptr(), i[Is]))... };

                raise_if(!(v[Is].is_valid() && ...), "item retrival failed!");

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

                raise_if(!vr.is_valid(), "nested operation failed!");

                // Assign result
                raise_if(set_item(result.ptr(), j, vr.ptr()),
                         "item assignment failed!");

                // Advance to next element, broadcast size-1 arrays
                ((i[Is] += (l[Is] == 1 ? 0 : 1)), ...);
            }
        }

        // In in-place mode, if a separate result object had to be
        // constructed, use it to now replace the contents of o[0]
        if (move) {
            nb::inst_replace_move(o[0], result);
            result = borrow(o[0]);
        }

        return result.release().ptr();
    } catch (nb::python_error &e) {
        e.restore();
        if constexpr (std::is_same_v<Slot, const char *>)
            nb::chain_error(PyExc_RuntimeError,
                            "drjit.%s(<%U>): failed (see above)!",
                            op_names[(int) op], nb::type_name(tp).ptr());
        else
            nb::chain_error(PyExc_RuntimeError,
                            "%U.%s(): failed (see above)!",
                            nb::type_name(tp).ptr(), op_names[(int) op]);
    } catch (const std::exception &e) {
        if constexpr (std::is_same_v<Slot, const char *>)
            nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): %s",
                            op_names[(int) op], nb::type_name(tp).ptr(),
                            e.what());
        else
            nb::chain_error(PyExc_RuntimeError, "%U.%s(): %s",
                            nb::type_name(tp).ptr(), op_names[(int) op],
                            e.what());
    }

    return nullptr;
}

// Broadcast a tensor to a desired shape. The size of 'shape_src' and 'shape_dst' must match.
void tensor_broadcast(nb::object &tensor, nb::object &array,
                      const vector<size_t> &shape_src,
                      const vector<size_t> &shape_dst) {
    size_t ndim = shape_src.size(), src_size = 1, dst_size = 1;
    if (ndim == 0 || memcmp(shape_src.data(), shape_dst.data(), sizeof(size_t) * ndim) == 0)
        return;

    for (size_t i = 0; i < ndim; ++i)
        src_size *= shape_src[i];
    for (size_t i = 0; i < ndim; ++i)
        dst_size *= shape_dst[i];

    nb::handle tp = tensor.type();
    const ArraySupplement &s = supp(tp);

    if (src_size == 1) {
        if (dst_size != 1) {
            if (s.type == (uint16_t) VarType::Bool)
                array = array | full("zeros", array.type(), nb::bool_(0), dst_size);
            else
                array = array + full("zeros", array.type(), nb::int_(0), dst_size);
        }
        return;
    }

    nb::type_object_t<ArrayBase> index_type =
        nb::borrow<nb::type_object_t<ArrayBase>>(s.tensor_index);

    nb::object index  = arange(index_type, 0, (Py_ssize_t) dst_size, 1),
               size_o = index_type(dst_size);

    for (size_t i = 0; i < ndim; ++i) {
        size_t size_next = dst_size / shape_dst[i];

        nb::object size_next_o = index_type(size_next);

        if (shape_src[i] == 1 && shape_dst[i] != 1)
            index = (index % size_next_o) + index.floor_div(size_o) * size_next_o;

        dst_size = size_next;
        size_o = std::move(size_next_o);
    }

    array = gather(nb::borrow<nb::type_object>(array.type()),
                   array, index, nb::borrow(Py_True));
}

/// Apply an element-wise operation to the given tensor(s)
template <ApplyMode Mode, typename Slot, typename... Args, size_t... Is>
NB_NOINLINE PyObject *apply_tensor(ArrayOp op, Slot slot,
                                   std::index_sequence<Is...> is,
                                   Args... args) noexcept {

    nb::object o[] = { nb::borrow(args)... };
    nb::handle tp = o[0].type();

    try {
        constexpr size_t N = sizeof...(Args);

        // All arguments must first be promoted to the same type
        if (!(o[Is].type().is(tp) && ...)) {
            promote(o, sizeof...(Args), Mode == Select);
            tp = o[Mode == Select ? 1 : 0].type();
        }

        // In 'InPlace' mode, try to update the 'self' argument when it makes sense
        bool move = Mode == InPlace && o[0].is(first(args...));

        const ArraySupplement *s[] = { &supp(o[Is].type())... };

        nb::object arrays[] = {
            nb::steal(s[Is]->tensor_array(o[Is].ptr()))...
        };

        const vector<size_t> *shapes[N] = {
            &s[Is]->tensor_shape(inst_ptr(o[Is]))...
        };

        size_t ndims[] = { shapes[Is]->size()... };
        size_t ndim = maxv(ndims[Is]...);
        bool compatible = true;

        // Left-fill with dummy dimensions of size 1
        vector<size_t> expanded_shapes_alloc[N] = {};
        auto expand = [&](size_t index, const vector<size_t>* shape) {
            size_t src_ndim = shape->size();
            if (src_ndim == ndim)
                return shape;

            expanded_shapes_alloc[index] = vector<size_t>(ndim, 1);
            vector<size_t>& expanded_shape = expanded_shapes_alloc[index];
            size_t offset = ndim - src_ndim;
            if (src_ndim)
                memcpy(&expanded_shape[offset], shape->data(), sizeof(size_t) * src_ndim);
            return (const vector<size_t>*)&expanded_shape;
        };

        const vector<size_t>* expanded_shapes[N] = { expand(Is, shapes[Is])... };
        vector<size_t> shape(ndim, 0);

        if (compatible) {
            for (size_t i = 0; i < ndim; ++i) {
                size_t shape_i[] = { expanded_shapes[Is]->operator[](i)... };
                size_t value = maxv(shape_i[Is]...);
                if (((shape_i[Is] != value && shape_i[Is] != 1) || ...)) {
                    compatible = false;
                    break;
                }
                shape[i] = value;
            }
        }

        if (!compatible) {
            nb::str shape_str[] = { nb::str(cast_shape(*shapes[Is]))... };

            const char *fmt = N == 2
                ? "operands have incompatible shapes: %s and %s."
                : "operands have incompatible shapes: %s, %s, and %s.";

            nb::raise(fmt, shape_str[Is].c_str()...);
        }

        if constexpr (N > 1) {
            // Broadcast to compatible shape for binary/ternary operations
            (tensor_broadcast(o[Is], arrays[Is], *expanded_shapes[Is], shape), ...);
        }

        constexpr ApplyMode NestedMode = Mode == InPlace ? Normal : Mode;

        nb::object result_array = nb::steal(apply<NestedMode, Slot>(
            op, slot, is, arrays[Is].ptr()...));

        raise_if(!result_array.is_valid(),
                 "operation on underlying array failed.");

        nb::handle result_type = tp;
        if (Mode == RichCompare)
            result_type = s[0]->mask;

        nb::object result = result_type(result_array, cast_shape(shape));

        if (move) {
            nb::inst_replace_move(o[0], result);
            result = borrow(o[0]);
        }

        return result.release().ptr();
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        if constexpr (std::is_same_v<Slot, const char *>)
            nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): failed (see above).",
                            op_names[(int) op], tp_name.ptr());
        else
            nb::chain_error(PyExc_RuntimeError, "%U.%s(): failed (see above).",
                            tp_name.ptr(), op_names[(int) op]);
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        if constexpr (std::is_same_v<Slot, const char *>)
            nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): %s",
                            op_names[(int) op], tp_name.ptr(), e.what());
        else
            nb::chain_error(PyExc_RuntimeError, "%U.%s(): %s", tp_name.ptr(),
                            op_names[(int) op], e.what());
    }
    return nullptr;
}

/// Like apply(), but returns a pair of results. Used for dr.sincos, dr.sincosh, dr.frexp()
nb::object apply_ret_pair(ArrayOp op, const char *name, nb::handle_t<dr::ArrayBase> h) {
    using Impl = void (*)(const ArrayBase *, ArrayBase *, ArrayBase *);
    nb::handle tp = h.type();

    try {
        const ArraySupplement &s = supp(tp);
        nb::object r0 = nb::inst_alloc_zero(tp),
                   r1 = nb::inst_alloc_zero(tp);

        if (s.is_tensor) {
            nb::object result_shape = shape(h),
                       result_array = apply_ret_pair(
                           op, name, nb::steal(s.tensor_array(h.ptr())));
            return nb::make_tuple(tp(result_array[0], result_shape),
                                  tp(result_array[1], result_shape));
        }

        Impl impl = (Impl) s[op];

        raise_if(impl == DRJIT_OP_NOT_IMPLEMENTED,
                 "operation not supported for this type.");

        if (impl != DRJIT_OP_DEFAULT) {
            impl(inst_ptr(h), inst_ptr(r0), inst_ptr(r1));
            return nb::make_tuple(r0, r1);
        }
        dr::ArrayBase *ph = inst_ptr(h),
                      *p0 = inst_ptr(r0),
                      *p1 = inst_ptr(r1);

        Py_ssize_t lr = s.shape[0];
        if (lr == DRJIT_DYNAMIC) {
            lr = (Py_ssize_t) s.len(ph);
            s.init(lr, p0);
            s.init(lr, p1);
        }

        ArraySupplement::Item item = s.item;
        ArraySupplement::SetItem set_item = s.set_item;

        nb::object py_impl_o = array_module.attr(name);

        for (Py_ssize_t i = 0; i < lr; ++i) {
            nb::object v = nb::steal(item(h.ptr(), i));
            raise_if(!v.is_valid(), "item retrival failed!");
            nb::object vr = py_impl_o(v);

            // Assign result
            raise_if(set_item(r0.ptr(), i, vr[0].ptr()) ||
                     set_item(r1.ptr(), i, vr[1].ptr()),
                     "item assignment failed!");
        }

        return nb::make_tuple(r0, r1);
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.%s(<%U>): failed (see above)!", name,
                        tp_name.ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): %s",
                        name, nb::type_name(tp).ptr(), e.what());
    }

    return nb::object();
}

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

uint64_t TraverseCallback::operator()(uint64_t, const char *, const char *) { return 0; }
void TraverseCallback::traverse_unknown(nb::handle) { }

/// Invoke the given callback on leaf elements of the pytree 'h'
void traverse(const char *op, TraverseCallback &tc, nb::handle h, bool rw) {
    recursion_guard guard;

    nb::handle tp = h.type();

    try {
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);
            if (s.is_tensor) {
                tc(nb::steal(s.tensor_array(h.ptr())));
            } else if (s.ndim > 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(h));

                for (Py_ssize_t i = 0; i < len; ++i)
                    traverse(op, tc, nb::steal(s.item(h.ptr(), i)), rw);
            } else  {
                tc(h);
            }
        } else if (tp.is(&PyTuple_Type)) {
            for (nb::handle h2 : nb::borrow<nb::tuple>(h))
                traverse(op, tc, h2, rw);
        } else if (tp.is(&PyList_Type)) {
            for (nb::handle h2 : nb::borrow<nb::list>(h))
                traverse(op, tc, h2, rw);
        } else if (tp.is(&PyDict_Type)) {
            for (nb::handle h2 : nb::borrow<nb::dict>(h).values())
                traverse(op, tc, h2, rw);
        } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            for (auto [k, v] : ds)
                traverse(op, tc, nb::getattr(h, k), rw);
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                traverse(op, tc, nb::getattr(h, k), rw);
            }
        } else if (auto traversable = get_traversable_base(h); traversable) {
            struct Payload {
                TraverseCallback &tc;
            };
            Payload p{ tc };
            if (rw) {
                traversable->traverse_1_cb_rw(
                    (void *) &p,
                    [](void *p, uint64_t index, const char *variant,
                       const char *domain) -> uint64_t {
                        Payload *payload = (Payload *) p;
                        uint64_t new_index =
                            payload->tc(index, variant, domain);
                        return new_index;
                    });
            } else {
                traversable->traverse_1_cb_ro(
                    (void *) &p, [](void *p, uint64_t index,
                                    const char *variant, const char *domain) {
                        Payload *payload = (Payload *) p;
                        payload->tc(index, variant, domain);
                    });
            }
        } else if (auto cb = get_traverse_cb_ro(tp); cb.is_valid() && !rw) {
            cb(h, nb::cpp_function(
                      [&](uint64_t index, const char *variant,
                          const char *domain) { tc(index, variant, domain); }));
        } else if (nb::object cb_rw = get_traverse_cb_rw(tp);
                   cb_rw.is_valid() && rw) {
            cb_rw(h, nb::cpp_function([&](uint64_t index, const char *variant,
                                       const char *domain) {
                   return tc(index, variant, domain);
               }));
        } else {
            tc.traverse_unknown(h);
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "%s(): error encountered while processing an argument "
                       "of type '%U' (see above).", op, nb::type_name(tp).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "%s(): error encountered while processing an argument "
                        "of type '%U': %s", op, nb::type_name(tp).ptr(), e.what());
        nb::raise_python_error();
    }
}

void traverse_pair_impl(const char *op, TraversePairCallback &tc, nb::handle h1,
                        nb::handle h2, drjit::string &name,
                        dr::vector<PyObject *> &stack,
                        bool report_inconsistencies,
                        bool width_consistency) {
    if (std::find(stack.begin(), stack.end(), h1.ptr()) != stack.end()) {
        PyErr_Format(PyExc_RecursionError, "detected a cycle in field '%s'.",
                     name.c_str());
        nb::raise_python_error();
    }

    nb::handle tp1 = h1.type(), tp2 = h2.type();
    stack.push_back(h1.ptr());

    if (!tp1.is(tp2)) {
        if (report_inconsistencies)
            nb::raise("inconsistent types for field '%s' ('%s' and '%s').",
                      name.c_str(), nb::type_name(tp1).c_str(),
                      nb::type_name(tp2).c_str());
        else
            return;
    }

    size_t name_size = name.size();
    if (is_drjit_type(tp1)) {
        const ArraySupplement &s = supp(tp1);
        if (s.is_tensor) {
            nb::object sh1 = shape(h1),
                       sh2 = shape(h2);

            name.put(".array");
            traverse_pair_impl(
                op, tc,
                nb::steal(s.tensor_array(h1.ptr())),
                nb::steal(s.tensor_array(h2.ptr())),
                name, stack, report_inconsistencies, width_consistency
            );
            name.resize(name_size);
        } else {
            size_t s1 = nb::len(h1), s2 = nb::len(h2);

            if (s.ndim > 1) {
                if (s1 != s2) {
                    if (report_inconsistencies)
                        nb::raise("inconsistent sizes for field '%s' (%zu and %zu).",
                                  name.c_str(), s1, s2);
                    else if (s1 != 1 && s2 != 1)
                        return;
                }

                for (size_t i = 0; i < s1; ++i) {
                    name.put('[', i, ']');
                    traverse_pair_impl(
                        op, tc,
                        nb::steal(s.item(h1.ptr(), s1 == 1 ? 0 : i)),
                        nb::steal(s.item(h2.ptr(), s2 == 1 ? 0 : i)),
                        name, stack, report_inconsistencies, width_consistency
                    );
                    name.resize(name_size);
                }
            } else {
                if (report_inconsistencies) {
                    if (s1 == 0 || s2 == 0) {
                        nb::raise("field '%s' is uninitialized.", name.c_str());
                    } else if (s1 != s2 && s1 != 1 && s2 != 1 && width_consistency) {
                        nb::raise("incompatible sizes for field '%s' (%zu and %zu).",
                                  name.c_str(), s1, s2);
                    }
                }
                tc(h1, h2);
            }
        }
    } else if (tp1.is(&PyTuple_Type) || tp1.is(&PyList_Type)) {
        size_t s1 = nb::len(h1), s2 = nb::len(h2);
        if (s1 != s2) {
            if (report_inconsistencies)
                nb::raise("inconsistent sizes for field '%s' (%zu and %zu).",
                          name.c_str(), s1, s2);
            else if (s1 != 1 && s2 != 1)
                return;
        }
        for (size_t i = 0; i < s1; ++i) {
            name.put('[', i, ']');
            traverse_pair_impl(
                op, tc,
                h1[s1 == 1 ? 0 : i], h2[s2 == 1 ? 0 : i],
                name, stack, report_inconsistencies, width_consistency
            );
            name.resize(name_size);
        }
    } else if (tp1.is(&PyDict_Type)) {
        nb::dict d1 = nb::borrow<nb::dict>(h1),
                 d2 = nb::borrow<nb::dict>(h2);
        nb::object k1 = d1.keys(), k2 = d2.keys();
        if (!k1.equal(k2)) {
            if (report_inconsistencies)
                nb::raise(
                    "inconsistent dictionary keys for field '%s' (%s and %s).",
                    name.c_str(), nb::str(k1).c_str(), nb::str(k2).c_str());
            else
                return;
        }
        for (nb::handle k : k1) {
            name.put('[', nb::str(k).c_str(), ']');
            traverse_pair_impl(op, tc, d1[k], d2[k], name, stack,
                               report_inconsistencies, width_consistency);
            name.resize(name_size);
        }
    } else {
        if (nb::dict ds = get_drjit_struct(tp1); ds.is_valid()) {
            for (auto [k, v] : ds) {
                name.put('.', nb::str(k).c_str());
                traverse_pair_impl(op, tc, nb::getattr(h1, k),
                                   nb::getattr(h2, k), name, stack,
                                   report_inconsistencies, width_consistency);
                name.resize(name_size);
            }
        } else if (nb::object df = get_dataclass_fields(tp1); df.is_valid()) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                name.put('.', nb::str(k).c_str());
                traverse_pair_impl(op, tc, nb::getattr(h1, k),
                                   nb::getattr(h2, k), name, stack,
                                   report_inconsistencies, width_consistency);
                name.resize(name_size);
            }
        } else if (!h1.is(h2) && !h1.equal(h2)) {
            if (report_inconsistencies)
                nb::raise(
                    "inconsistent scalar Python object of type '%s' for field '%s'.",
                    nb::type_name(tp1).c_str(), name.c_str());
            else
                return;
        }
    }
    stack.pop_back();
}

/// Parallel traversal of two compatible PyTrees 'h1' and 'h2'
void traverse_pair(const char *op, TraversePairCallback &tc, nb::handle h1,
                   nb::handle h2, const char *name_,
                   bool report_inconsistencies, bool width_consistency) {
    drjit::string name = name_;
    dr::vector<PyObject *> stack;

    try {
        traverse_pair_impl(op, tc, h1, h2, name, stack,
                           report_inconsistencies, width_consistency);
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "%s(): error encountered while processing arguments "
                       "of type '%U' and '%U' (see above).",
                       op, nb::inst_name(h1).ptr(), nb::inst_name(h2).ptr());
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string(op) + "(): " + e.what());
    }
}

nb::handle TransformCallback::transform_type(nb::handle tp) const {
    return tp;
}

nb::object TransformCallback::transform_unknown(nb::handle h) const {
    return nb::borrow(h);
}

/// Transform an input pytree 'h' into an output pytree, potentially of a different type
nb::object transform(const char *op, TransformCallback &tc, nb::handle h) {
    recursion_guard guard;
    nb::handle tp = h.type();

    try {
        nb::object result;

        if (is_drjit_type(tp)) {
            nb::handle tp2 = tc.transform_type(tp);
            if (!tp2.is_valid())
                return nb::none();

            const ArraySupplement &s1 = supp(tp),
                                  &s2 = supp(tp2);

            if (s1.is_tensor) {
                nb::object array = nb::steal(s1.tensor_array(h.ptr())),
                           array_t = transform(op, tc, array);

                nb::object s = shape(h);
                if (nb::len(array) == nb::len(array_t))
                    result = tp2(array_t, shape(h));
                else
                    result = tp2(array_t);
            } else if (s1.ndim != 1) {
                result = nb::inst_alloc_zero(tp2);
                dr::ArrayBase *p1 = inst_ptr(h),
                              *p2 = inst_ptr(result);

                size_t size = s1.shape[0];
                if (size == DRJIT_DYNAMIC) {
                    size = s1.len(p1);
                    s2.init(size, p2);
                }

                for (size_t i = 0; i < size; ++i)
                    result[i] = transform(op, tc, h[i]);
            } else {
                result = nb::inst_alloc_zero(tp2);
                tc(h, result);
            }
        } else if (tp.is(&PyTuple_Type)) {
            nb::tuple t = nb::borrow<nb::tuple>(h);
            size_t size = nb::len(t);
            result = nb::steal(PyTuple_New(size));
            if (!result.is_valid())
                nb::raise_python_error();
            for (size_t i = 0; i < size; ++i)
                NB_TUPLE_SET_ITEM(result.ptr(), i,
                                  transform(op, tc, t[i]).release().ptr());
        } else if (tp.is(&PyList_Type)) {
            nb::list tmp;
            for (nb::handle item : nb::borrow<nb::list>(h))
                tmp.append(transform(op, tc, item));
            result = std::move(tmp);
        } else if (tp.is(&PyDict_Type)) {
            nb::dict tmp;
            for (auto [k, v] : nb::borrow<nb::dict>(h))
                tmp[k] = transform(op, tc, v);
            result = std::move(tmp);
        } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            nb::object tmp = tp();
            for (auto [k, v] : ds)
                nb::setattr(tmp, k, transform(op, tc, nb::getattr(h, k)));
            result = std::move(tmp);
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            nb::object tmp = nb::dict();
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                tmp[k]       = transform(op, tc, nb::getattr(h, k));
            }
            result = tp(**tmp);
        } else if (nb::object cb = get_traverse_cb_rw(tp); cb.is_valid()) {
            cb(h, nb::cpp_function([&](uint64_t index, const char *,
                                       const char *) { return tc(index); }));
            result = nb::borrow(h);
        } else if (!result.is_valid()) {
            result = tc.transform_unknown(h);
        }
        return result;
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "%s(): error encountered while processing an argument "
                       "of type '%U' (see above).", op, nb::type_name(tp).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "%s(): error encountered while processing an argument "
                        "of type '%U': %s", op, nb::type_name(tp).ptr(), e.what());
        nb::raise_python_error();
    }
}

nb::handle TransformPairCallback::transform_type(nb::handle tp) const {
    return tp;
}

uint64_t TransformCallback::operator()(uint64_t index) { return index; }

nb::object TransformPairCallback::transform_unknown(nb::handle, nb::handle) const {
    return nb::none();
}

/// Transform a pair of input pytrees 'h1' and 'h2' into an output pytree, potentially of a different type
nb::object transform_pair(const char *op, TransformPairCallback &tc,
                          nb::handle h1, nb::handle h2) {
    recursion_guard guard;
    nb::handle tp1 = h1.type(), tp2 = h2.type();

    try {
        if (!tp1.is(tp2))
            nb::raise("incompatible input types.");

        if (is_drjit_type(tp1)) {
            nb::handle tp3 = tc.transform_type(tp1);
            if (!tp3.is_valid())
                return nb::none();

            const ArraySupplement &s1 = supp(tp1),
                                  &s3 = supp(tp3);

            nb::object h3;
            if (s1.is_tensor) {
                nb::object sh1 = shape(h1),
                           sh2 = shape(h2);

                if (!sh1.equal(sh2))
                    nb::raise("incompatible tensor shape (%s and %s)",
                              nb::str(sh1).c_str(), nb::str(sh2).c_str());

                h3 = tp3(transform_pair(op, tc,
                                        nb::steal(s1.tensor_array(h1.ptr())),
                                        nb::steal(s1.tensor_array(h2.ptr()))),
                         sh1);
            } else if (s1.ndim != 1) {
                h3 = nb::inst_alloc_zero(tp3);

                dr::ArrayBase *p1 = inst_ptr(h1),
                              *p2 = inst_ptr(h2),
                              *p3 = inst_ptr(h3);

                Py_ssize_t len1 = s1.shape[0], len2 = len1;
                if (len1 == DRJIT_DYNAMIC) {
                    len1 = s1.len(p1);
                    len2 = s1.len(p2);

                    if (len1 != len2)
                        nb::raise("incompatible input lengths (%zu and %zu).", len1, len2);

                    s3.init(len1, p3);
                }

                for (Py_ssize_t i = 0; i < len1; ++i) {
                    nb::object o1 = h1[i], o2 = h2[i];
                    h3[i] = transform_pair(op, tc, o1, o2);
                }
            } else {
                h3 = nb::inst_alloc_zero(tp3);
                tc(h1, h2, h3);
            }

            return h3;
        } else if (tp1.is(&PyTuple_Type)) {
            nb::tuple t1 = nb::borrow<nb::tuple>(h1),
                      t2 = nb::borrow<nb::tuple>(h2);
            size_t len1 = nb::len(t1), len2 = nb::len(t2);
            if (len1 != len2)
                nb::raise("incompatible input lengths (%zu and %zu).", len1, len2);

            nb::object result = nb::steal(PyTuple_New(len1));
            if (!result.is_valid())
                nb::raise_python_error();

            for (size_t i = 0; i < len1; ++i)
                NB_TUPLE_SET_ITEM(result.ptr(), i,
                                  transform_pair(op, tc, t1[i], t2[i]).release().ptr());

            return result;
        } else if (tp1.is(&PyList_Type)) {
            nb::list l1 = nb::borrow<nb::list>(h1),
                     l2 = nb::borrow<nb::list>(h2);
            size_t len1 = nb::len(l1), len2 = nb::len(l2);
            if (len1 != len2)
                nb::raise("incompatible input lengths (%zu and %zu).", len1, len2);

            nb::list result;
            for (size_t i = 0; i < len1; ++i)
                result.append(transform_pair(op, tc, l1[i], l2[i]));

            return std::move(result);
        } else if (tp1.is(&PyDict_Type)) {
            nb::dict d1 = nb::borrow<nb::dict>(h1),
                     d2 = nb::borrow<nb::dict>(h2);

            nb::object k1 = d1.keys(), k2 = d2.keys();
            if (!k1.equal(k2))
                nb::raise(
                    "dictionaries have incompatible keys (%s vs %s).",
                    nb::str(k1).c_str(), nb::str(k2).c_str());

            nb::dict result;
            for (nb::handle k : k1)
                result[k] = transform_pair(op, tc, d1[k], d2[k]);

            return std::move(result);
        } else {
            if (nb::dict ds = get_drjit_struct(tp1); ds.is_valid()) {
                nb::object result = tp1();
                for (auto [k, v] : ds)
                    nb::setattr(result, k,
                                transform_pair(op, tc, nb::getattr(h1, k),
                                               nb::getattr(h2, k)));
                return result;
            } else if (nb::object df = get_dataclass_fields(tp1); df.is_valid()) {
                nb::dict result;
                for (nb::handle field : df) {
                    nb::object k = field.attr(DR_STR(name));
                    result[k] = transform_pair(op, tc, nb::getattr(h1, k),
                                               nb::getattr(h2, k));
                }
                return tp1(**result);
            } else {
                return tc.transform_unknown(h1, h2);
            }
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "%s(): error encountered while processing arguments "
                       "of type '%U' and '%U' (see above).",
                       op, nb::type_name(tp1).ptr(), nb::type_name(tp2).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "%s(): error encountered while processing arguments "
                        "of type '%U' and '%U': %s",
                        op, nb::type_name(tp1).ptr(), nb::type_name(tp2).ptr(),
                        e.what());
        nb::raise_python_error();
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
template PyObject *apply<MulWide>(ArrayOp, const char *, std::index_sequence<0, 1>,
                                  PyObject *, PyObject *) noexcept;
