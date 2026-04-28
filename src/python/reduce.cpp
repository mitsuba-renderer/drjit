/*
    reduce.cpp -- Bindings for horizontal reduction operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "reduce.h"
#include "base.h"
#include "drjit/python.h"
#include "meta.h"
#include "shape.h"
#include "memop.h"
#include "init.h"
#include "apply.h"
#include "detail.h"
#include "coop_vec.h"
#include <nanobind/stl/optional.h>

using ReduceInit = nb::object();
using ReduceCombine = nb::object(nb::handle, nb::handle);

// The python bindings expose a few more reduction operations that aren't
// available as part of the drjit::ReduceOp list.
enum class ReduceOpExt : uint32_t {
    All = (uint32_t) ReduceOp::Count,
    Any = (uint32_t) ReduceOp::Count + 1,
    Count = (uint32_t) ReduceOp::Count + 2,
    OpCount = (uint32_t) ReduceOp::Count + 3
};

struct Reduction {
    ArrayOp op;
    const char *name;
    bool (*skip)(nb::handle);
    nb::object (*init)();
    nb::object (*combine)(nb::handle, nb::handle);
};

static Reduction reductions[] = {
    { (ArrayOp) 0, nullptr, nullptr, nullptr, nullptr },
    {
        ArrayOp::Sum,
        "sum",
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::int_(0); },
        [](nb::handle h1, nb::handle h2) { return h1 + h2; }
    },
    {
        ArrayOp::Prod,
        "prod",
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::int_(1); },
        [](nb::handle h1, nb::handle h2) { return h1 * h2; }
    },
    {
        ArrayOp::Min,
        "min",
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::float_(INFINITY); },
        [](nb::handle h1, nb::handle h2) { return array_module.attr("minimum")(h1, h2); }
    },
    {
        ArrayOp::Max,
        "max",
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::float_(-INFINITY); },
        [](nb::handle h1, nb::handle h2) { return array_module.attr("maximum")(h1, h2); }
    },
    {
        ArrayOp::And,
        "and",
        [](nb::handle tp) { return tp.is(&PyLong_Type); },
        []() -> nb::object { return nb::int_(1); },
        [](nb::handle h1, nb::handle h2) { return h1 & h2; }
    },
    {
        ArrayOp::Or,
        "or",
        [](nb::handle tp) { return tp.is(&PyLong_Type); },
        []() -> nb::object { return nb::int_(0); },
        [](nb::handle h1, nb::handle h2) { return h1 | h2; }
    },
    {
        ArrayOp::All,
        "all",
        [](nb::handle tp) { return tp.is(&PyBool_Type); },
        []() -> nb::object { return nb::borrow(Py_True); },
        [](nb::handle h1, nb::handle h2) { return h1 & h2; }
    },
    {
        ArrayOp::Any,
        "any",
        [](nb::handle tp) { return tp.is(&PyBool_Type); },
        []() -> nb::object { return nb::borrow(Py_False); },
        [](nb::handle h1, nb::handle h2) { return h1 | h2; }
    },
    {
        ArrayOp::Count,
        "count",
        [](nb::handle tp) { return tp.is(&PyBool_Type); },
        []() -> nb::object { return nb::int_(0); },
        [](nb::handle h1, nb::handle h2) {
            return h1 + array_module.attr("select")(h2, nb::int_(1), nb::int_(0));
        }
    }
};

// Sanity checks to catch modifications in the ReduceOp enumeration
static_assert(
    (size_t) ReduceOp::Identity == 0 &&
    (size_t) ReduceOp::Add == 1 &&
    (size_t) ReduceOp::Mul == 2 &&
    (size_t) ReduceOp::Min == 3 &&
    (size_t) ReduceOp::Max == 4 &&
    (size_t) ReduceOp::And == 5 &&
    (size_t) ReduceOp::Or == 6 &&
    (size_t) ReduceOpExt::All == 7 &&
    (size_t) ReduceOpExt::Any == 8 &&
    (size_t) ReduceOpExt::Count == 9 &&
    (size_t) ReduceOpExt::OpCount == 10
);

static_assert(sizeof(reductions) == sizeof(Reduction) * (size_t) ReduceOpExt::OpCount);

// Forward declaration
nb::object reduce(uint32_t op, nb::handle h, nb::handle axis, nb::handle mode, bool keepdims);

nb::object reduce_seq(uint32_t op, nb::handle h, nb::handle axis, nb::handle mode) {
    Reduction red = reductions[(size_t) op];

    if (red.skip(h.type()))
        return nb::borrow(h);

    nb::object it;
    try {
        it = iter(h);
    } catch (...) {
        nb::raise("the input must be a Dr.Jit array, iterable type, or a "
                  "Python scalar compatible with the requested reduction.");
    }

    if (!(axis.is_none() || (nb::isinstance<int>(axis) && nb::cast<int>(axis) == 0)))
        nb::raise("for reductions over (non-Dr.Jit) iterable types, 'axis' "
                  "must equal 0 or None.");

    nb::object result = red.init();
    size_t i = 0;
    for (nb::handle h2 : it) {
        nb::object o = nb::borrow(h2);
        if (axis.is_none())
            o = reduce(op, o, axis, mode, false);

        if (i++ == 0)
            result = std::move(o);
        else
            result = red.combine(result, o);

        if (!result.is_valid())
            nb::raise_python_error();
    }

    return result;
}

nb::object prefix_reduce_seq(ReduceOp op, nb::handle h, int axis, bool exclusive, bool reverse) {
    Reduction red = reductions[(size_t) op];

    size_t size = nb::len(h);

    if (axis != 0)
        nb::raise("for reductions over (non-Dr.Jit) iterable types, 'axis' must equal 0.");

    nb::object value = red.init();
    nb::list result;

    for (size_t i = 0; i < size; ++i) {
        nb::object o = nb::borrow(h[reverse ? size - 1 - i : i]);
        nb::object value_prev = value;

        if (i == 0)
            value = std::move(o);
        else
            value = red.combine(value, o);

        if (!result.is_valid())
            nb::raise_python_error();

        result.append(exclusive ? value_prev : value);
    }

    if (is_drjit_array(h))
        return h.type()(result);

    return std::move(result);
}

nb::object reduce(uint32_t op, nb::handle h, nb::handle axis_, nb::handle mode, bool keepdims) {
    nb::handle tp = h.type();
    if (axis_.type().is(&PyEllipsis_Type)) {
        if (!is_drjit_type(tp) || !supp(tp).is_tensor)
            axis_ = nb::int_(0);
        else
            axis_ = nb::none();
    }

    if (op >= (size_t) ReduceOpExt::OpCount || !reductions[op].skip)
        nb::raise("drjit.reduce(): unsupported reduction type.");

    const Reduction &red = reductions[op];

    try {
        if (!is_drjit_type(tp)) {
            if (keepdims)
                nb::raise("'keepdims=True' is only supported for Dr.Jit "
                          "arrays and tensors.");
            return reduce_seq(op, h, axis_, mode);
        }

        const ArraySupplement &s = supp(tp);

        int ndim = (int)::ndim(h);
        if (ndim == 0) {
            int value;
            // Accept 0-dim tensors and axis==0 (the default); return the
            // tensor without changes instead of failing with an error message.
            if (nb::try_cast(axis_, value) && value == 0)
                return nb::borrow(h);
        }

        // Number of axes along which to reduce (-1: all of them)
        int axis_len;

        // First axis along which to reduce
        int red_axis;

        const char *axis_type_msg =
            "'axis' argument must be of type 'int', 'tuple[int, ...]', "
            "or None.";

        nb::object axis = nb::borrow(axis_);
        if (axis.is_none()) {
            red_axis = 0;
            axis_len = -1;
        } else if (nb::try_cast(axis, red_axis)) {
            if (red_axis < 0)
                red_axis = red_axis + ndim;
            if (red_axis < 0 || red_axis >= ndim)
                nb::raise("out-of-bounds axis (got %i, ndim=%i)", red_axis, ndim);
            if (red_axis == 0 && ndim == 1) {
                axis_len = -1;
                axis = nb::none();
            } else {
                axis_len = 1;
                axis = nb::make_tuple(red_axis);
            }
        } else if (nb::isinstance<nb::tuple>(axis)) {
            // Normalize the axis tuple: convert negative indices,
            // bounds-check, then deduplicate and sort. Sorting is
            // required by the recursive reduction below, which assumes
            // ascending order; dedup avoids reducing a remapped axis.
            nb::list new_axis;
            for (nb::handle h2 : nb::borrow<nb::tuple>(axis)) {
                int value;
                if (!nb::try_cast(h2, value))
                    nb::raise("%s", axis_type_msg);
                if (value < 0)
                    value += ndim;
                if (value < 0 || value >= ndim)
                    nb::raise("out-of-bounds axis (got %i, ndim=%i)",
                              value, ndim);
                new_axis.append(value);
            }
            nb::list l = nb::list(nb::set(new_axis));
            l.sort();
            axis = nb::tuple(l);
            axis_len = (int) nb::len(axis);

            if (axis_len == 0) {
                // Nothing to do
                return nb::borrow(h);
            } else if (axis_len == ndim) {
                // Special case: reducing over all dims
                axis = nb::none();
                red_axis = 0;
                axis_len = -1;
            } else {
                red_axis = nb::cast<int>(axis[0]);
            }
        } else {
            nb::raise("%s", axis_type_msg);
        }

        if (s.is_tensor) {
            if (axis_len == -1) {
                // Directly process the underlying 1D array
                nb::object value = nb::steal(s.tensor_array(h.ptr()));
                value = reduce(op, value, axis, mode, false);
                if (op == (uint32_t) ReduceOpExt::Count) {
                    ArrayMeta m = s;
                    m.type = (uint16_t) VarType::UInt32;
                    tp = meta_get_type(m);
                }

                if (keepdims && ndim > 0)
                    return tp(value, cast_shape(dr::vector<size_t>(ndim, 1)));

                return tp(value, nb::tuple());
            } else {
                if (op >= (uint32_t) ReduceOp::Count) {
                    if (axis_len == 1 && red_axis == 0)
                        return reduce_seq(op, h, nb::int_(0), mode);
                    nb::raise_type_error("tensor type is not compatible with "
                                         "the requested reduction.");
                }
                // Complex case, defer to a separate Python implementation
                return nb::module_::import_("drjit._reduce")
                    .attr("tensor_reduce")(ReduceOp(op), h, axis, mode,
                                            nb::bool_(keepdims));
            }
        }

        int symbolic = -1;
        if (!mode.is_none()) {
            if (nb::isinstance<nb::str>(mode)) {
                const char *s_ = nb::borrow<nb::str>(mode).c_str();
                if (strcmp(s_, "symbolic") == 0)
                    symbolic = 1;
                else if (strcmp(s_, "evaluated") == 0)
                    symbolic = 0;
            }
            if (symbolic == -1)
                nb::raise("'mode' must be \'symbolic\", \"evaluated\", or None.");
        }

        // 'keepdims=True' on a non-tensor is only supported when the
        // existing reduction already preserves a size-1 stub at the
        // reduced axis -- which happens iff the trailing axis is dynamic
        // and is the (only) reduced axis.
        if (keepdims) {
            bool only_trailing_reduced =
                (axis_len == 1 && red_axis == s.ndim - 1) ||
                (axis_len == -1 && s.ndim == 1);
            if (!only_trailing_reduced ||
                s.shape[s.ndim - 1] != DRJIT_DYNAMIC)
                nb::raise("'keepdims=True' is only supported for tensor "
                          "types or reductions of a trailing dynamic axis.");
        }

        // Reduce along the first specified axis
        nb::object result;
        if (red_axis == 0) {
            /// Reduce among the outermost axis
            void *op_fn = s.op[(int) red.op];

            if (op_fn == DRJIT_OP_NOT_IMPLEMENTED)
                nb::raise_type_error("array type is not compatible with the "
                                     "requested reduction.");

            nb::type_object_t<dr::ArrayBase> tpa =
                nb::borrow<nb::type_object_t<dr::ArrayBase>>(tp);

            if (s.ndim == 1 && op_fn != DRJIT_OP_DEFAULT) {
                const ArrayBase *hp = inst_ptr(h);

                if (symbolic == -1) {
                    VarState state = VarState::Evaluated;
                    if (s.index) {
                        uint32_t index = (uint32_t) s.index(hp);
                        state = jit_var_state(index);

                        // Reducing a symbolic variable is probably a bad idea.
                        // As a default policy, let's error out here by trying
                        // to evaluate the variable, which will display a long
                        // and informative error message to the user.
                        //
                        // If a symbolic reduction of a symbolic variable is
                        // truly desired, the user may specify mode="symbolic".

                        if (state == VarState::Symbolic)
                            jit_var_eval(index);
                    }

                    bool can_reduce = op < (uint32_t) ReduceOp::Count &&
                                      can_scatter_reduce(tpa, (ReduceOp) op);

                    if (!can_reduce) {
                        symbolic = 0;
                    } else {
                        // Would it be reasonable to evaluate the array?
                        bool is_evaluated = state == VarState::Evaluated ||
                                            state == VarState::Dirty;
                        bool is_big_array =
                            jit_type_size((VarType) s.type) * nb::len(h) >
                            1024 * 1024 * 1024; // 1 GiB

                        symbolic = !is_evaluated && is_big_array;
                    }
                }
                if (symbolic) {
                    // Symbolic, via scatter
                    result = reduce_identity(tpa, (ReduceOp) op, 1);
                    ::scatter_reduce((ReduceOp) op, result, nb::borrow(h),
                                     nb::int_(0), nb::bool_(true),
                                     ReduceMode::Auto);
                } else {
                    // Evaluate the array, then use a specialized reduction kernel
                    if (op == (uint32_t) ReduceOpExt::Count) {
                        ArrayMeta m = s;
                        m.type = (uint16_t) VarType::UInt32;
                        tp = meta_get_type(m);
                    }

                    result = nb::inst_alloc(tp);
                    ((ArraySupplement::UnaryOp) op_fn)(hp, inst_ptr(result));
                    nb::inst_mark_ready(result);
                }
            } else {
                // Use general sequence implementation
                result = reduce_seq(op, h, nb::int_(0), mode);
            }
        } else {
            // Reduce along an intermediate axis

            ArrayMeta m = s;
            dr::vector<size_t> shape;
            shape_impl(h, shape);

            if (red_axis >= (int) shape.size())
                nb::raise("nb::reduce(): internal error, 'red_axis' is out of bounds!");

            if (red_axis == (int) shape.size() - 1 && m.shape[red_axis] == DRJIT_DYNAMIC) {
                shape[red_axis] = 1;
            } else {
                m.is_matrix = m.is_quaternion = m.is_complex = false;
                for (int i = red_axis; i < m.ndim - 1; ++i) {
                    m.shape[i] = m.shape[i + 1];
                    shape[i] = shape[i + 1];
                }
                m.ndim--;
                shape.resize(shape.size() - 1);
            }

            result =
                array_module.attr("empty")(meta_get_type(m), cast_shape(shape));

            size_t i = 0;
            for (nb::handle h2 : h)
                result[i++] = reduce(op, h2, nb::int_(red_axis - 1), mode, false);
        }

        if (ndim == 1 || axis_len == 1)
            return result; // All done!

        if (axis_len != -1) {
            // Drop the just-reduced axis, then decrement the remaining
            // ones since their positions in ``result`` have shifted by
            // one. Axes are sorted ascending, so all entries beyond
            // index 0 are strictly greater than the reduced axis.
            nb::list shifted;
            for (size_t i = 1; i < nb::len(axis); ++i)
                shifted.append(nb::int_(nb::cast<int>(axis[i]) - 1));
            axis = nb::tuple(shifted);
        }

        return reduce(op, result, axis, mode, false);
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.%s(<%U>): failed (see above)!",
                        red.name, tp_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): %s",
                        red.name, tp_name.ptr(), e.what());
    }

    return nb::object();
}

nb::object sum(nb::handle value, nb::handle axis, nb::handle mode, bool keepdims) {
    return reduce((uint32_t) ReduceOp::Add, value, axis, mode, keepdims);
}

nb::object prod(nb::handle value, nb::handle axis, nb::handle mode, bool keepdims) {
    return reduce((uint32_t) ReduceOp::Mul, value, axis, mode, keepdims);
}

nb::object min(nb::handle value, nb::handle axis, nb::handle mode, bool keepdims) {
    return reduce((uint32_t) ReduceOp::Min, value, axis, mode, keepdims);
}

nb::object max(nb::handle value, nb::handle axis, nb::handle mode, bool keepdims) {
    return reduce((uint32_t) ReduceOp::Max, value, axis, mode, keepdims);
}

nb::object all(nb::handle value, nb::handle axis, bool keepdims) {
    return reduce((uint32_t) ReduceOpExt::All, value, axis, nb::none(), keepdims);
}

nb::object any(nb::handle value, nb::handle axis, bool keepdims) {
    return reduce((uint32_t) ReduceOpExt::Any, value, axis, nb::none(), keepdims);
}

nb::object count(nb::handle value, nb::handle axis, bool keepdims) {
    return reduce((uint32_t) ReduceOpExt::Count, value, axis, nb::none(), keepdims);
}

nb::object reduce_py(ReduceOp op, nb::handle value, nb::handle axis,
                     nb::handle mode, bool keepdims) {
    return reduce((uint32_t) op, value, axis, mode, keepdims);
}

nb::object none(nb::handle h, nb::handle axis, bool keepdims) {
    nb::object result = any(h, axis, keepdims);

    if (!result.ptr()) {
        nb::chain_error(PyExc_RuntimeError,
            "dr.none(): encountered an exception (see above).");
        return result;
    }

    if (result.type().is(&PyBool_Type))
        return nb::borrow(result.is(Py_True) ? Py_False : Py_True);
    else
        return ~result;
}

nb::object mean(nb::handle value, nb::handle axis, nb::handle mode, bool keepdims) {
    nb::object out = sum(value, axis, mode, keepdims);

    if (!out.ptr()) {
        nb::chain_error(PyExc_RuntimeError,
            "dr.mean(): encountered an exception (see above).");
        return out;
    }

    if (jit_flag(JitFlag::FreezingScope) && width(out) == 1 &&
        width(value) > 1) {
        // To avoid incorrect values when replaying frozen functions, we have to
        // avoid baking the size of the array into the kernel as a literal. We
        // therefore use the functions ``jit_opaque_width`` to compute the
        // number of elements.
        auto num_input  = opaque_n_elements(value);
        auto num_output = prod(shape(out), nb::none(), nb::none(), false);

        return (out * num_output) / num_input;
    }

    // mean = sum / (num_input/num_output)
    return (out * prod(shape(out), nb::none(), nb::none(), false))
         / prod(shape(value), nb::none(), nb::none(), false);
}

nb::object dot(nb::handle h0, nb::handle h1) {
    try {
        size_t l0 = nb::len(h0),
               l1 = nb::len(h1),
               lr = std::max(l0, l1);

        if (l0 != l1 && l0 != 1 && l1 != 1)
            nb::raise("invalid input array sizes (%zu and %zu)", l0, l1);

        bool use_fma = true;
        bool use_native_op = false;

        nb::handle tp0 = h0.type(), tp1 = h1.type();
        if (is_drjit_type(tp0)) {
            const ArraySupplement &s0 = supp(tp0);
            if (s0.ndim == 1 && s0.shape[0] == DRJIT_DYNAMIC) {
                use_fma = false;
                if (tp0.is(tp1) && s0.index)
                    use_native_op = true;
            }
        }

        if (is_drjit_type(tp1)) {
            const ArraySupplement &s1 = supp(tp1);
            if (s1.ndim == 1 && s1.shape[0] == DRJIT_DYNAMIC)
                use_fma = false;
        }

        if (use_fma) {
            if (tp0.is(coop_vector_type) || tp1.is(coop_vector_type)) {
                nb::list o0 = nb::list(h0), o1 = nb::list(h1);
                return dot(o1, o1);
            }
            nb::object result = h0[0] * h1[0],
                       fma = array_module.attr("fma");
            for (size_t i = 1; i < lr; ++i)
                result = fma(h0[l0 == 1 ? 0 : i],
                             h1[l1 == 1 ? 0 : i], result);
            return result;
        } else if (use_native_op) {
            const ArraySupplement &s = supp(tp0);

            uint64_t index = ad_var_reduce_dot(
                s.index(inst_ptr(h0)),
                s.index(inst_ptr(h1))
            );

            nb::object result = nb::inst_alloc(tp1);
            s.init_index(index, inst_ptr(result));
            nb::inst_mark_ready(result);
            ad_var_dec_ref(index);
            return result;
        } else {
            return sum(h0 * h1, nb::int_(0), nb::none(), false);
        }
    } catch (nb::python_error &e) {
        nb::str tp0_name = nb::inst_name(h0),
                tp1_name = nb::inst_name(h1);

        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.dot(<%U>, <%U>): failed (see above)!",
                        tp0_name.ptr(), tp1_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp0_name = nb::inst_name(h0),
                tp1_name = nb::inst_name(h1);

        nb::chain_error(PyExc_RuntimeError, "drjit.dot(<%U>, <%U>): %s",
                        tp0_name.ptr(), tp1_name.ptr(), e.what());
    }

    return { };
}

nb::object compress(nb::handle_t<dr::ArrayBase> h) {
    nb::handle tp = h.type();
    const ArraySupplement &s = supp(tp);
    if (!s.compress)
        nb::raise(
            "drjit.compress(<%s>): 'arg' must be a flat (1D) boolean array!",
            nb::type_name(tp).c_str());
    ArrayMeta m = s;
    m.type = (uint16_t) VarType::UInt32;
    nb::handle t = meta_get_type(m);
    nb::object result = nb::inst_alloc(t);
    s.compress(inst_ptr(h), inst_ptr(result));
    nb::inst_mark_ready(result);
    return result;
}


static nb::object prefix_reduce(ReduceOp op, nb::handle h, nb::handle axis, bool exclusive, bool reverse) {

    if (nb::isinstance<nb::tuple>(axis)) {
        nb::object o = nb::borrow(h);
        nb::tuple t = nb::cast<nb::tuple>(axis);
        for (size_t i = 0, s = t.size(); i < s; ++i)
            o = prefix_reduce(op, h, t[s - 1 - i], exclusive, reverse);
        return o;
    }
    int axis_i;

    if (!nb::try_cast(axis, axis_i))
        nb::raise("drjit.prefix_reduce(): 'axis' must be of type 'int' or 'tuple[int, ...]'!");

    nb::handle tp = h.type();
    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.is_tensor || (s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC))
            return nb::module_::import_("drjit._reduce")
                .attr("prefix_reduce")(op, h, axis, exclusive, reverse);
    }

    if (nb::isinstance<nb::sequence>(h))
        return prefix_reduce_seq(op, h, axis_i, exclusive, reverse);

    nb::raise_type_error("drjit.prefix_reduce(): 'value' must be a sequence type!");
}

static nb::object block_reduce(ReduceOp op,
                               nb::handle h, uint32_t block_size,
                               std::optional<dr::string> mode) {
    struct BlockReduceOp : TransformCallback {
        ReduceOp op;
        uint32_t block_size;
        int symbolic;

        BlockReduceOp(ReduceOp op, uint32_t block_size, int symbolic)
            : op(op), block_size(block_size), symbolic(symbolic) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());

            if (!s.block_reduce)
                nb::raise_type_error("drjit.block_reduce(): type '%s' does not "
                                     "implement the block reduction operation!",
                                     inst_name(h1).c_str());

            s.block_reduce(inst_ptr(h1), op, block_size, symbolic, inst_ptr(h2));
            inst_mark_ready(h2);
        }
    };

    int symbolic = -1;
    if (mode.has_value()) {
        if (mode.value() == "symbolic")
            symbolic = 1;
        else if (mode.value() == "evaluated")
            symbolic = 0;
        else
            nb::raise("drjit.block_reduce(): 'mode' parameter must either equal 'symbolic' or 'evaluated'!");
    }

    BlockReduceOp r(op, block_size, symbolic);
    return transform("drjit.block_reduce", r, h);
}

static nb::object block_prefix_reduce(ReduceOp op, nb::handle h,
                                      uint32_t block_size, bool exclusive,
                                      bool reverse) {
    struct BlockPrefixReduceOp : TransformCallback {
        ReduceOp op;
        uint32_t block_size;
        bool exclusive;
        bool reverse;

        BlockPrefixReduceOp(ReduceOp op, uint32_t block_size, bool exclusive, bool reverse)
            : op(op), block_size(block_size), exclusive(exclusive), reverse(reverse) { }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());

            if (!s.block_reduce)
                nb::raise_type_error("drjit.block_prefix_reduce(): type '%s' does not "
                                     "implement the block prefix reduction operation!",
                                     inst_name(h1).c_str());

            s.block_prefix_reduce(inst_ptr(h1), op, block_size, exclusive, reverse, inst_ptr(h2));
            inst_mark_ready(h2);
        }
    };

    BlockPrefixReduceOp r(op, block_size, exclusive, reverse);
    return transform("drjit.block_prefix_reduce", r, h);
}

static nb::object block_sum(nb::handle h, uint32_t block_size,
                            std::optional<dr::string> mode) {
    return block_reduce(ReduceOp::Add, h, block_size, mode);
}

/// Returns the shape that ``sum(x, axis, keepdims)`` would produce for an
/// input of shape ``in_shape``. The default ellipsis resolves to all axes
/// for tensors and to axis 0 otherwise. Returns ``std::nullopt`` for
/// malformed axis specs or for ranks outside [1, 63].
static std::optional<dr::vector<size_t>> reduction_output_shape(
    nb::handle axis, const dr::vector<size_t> &in_shape, bool is_tensor,
    bool keepdims) {
    int n = (int) in_shape.size();
    if (n <= 0 || n >= 64) return std::nullopt;
    uint64_t full = ((uint64_t) 1 << n) - 1;
    uint64_t reduced;

    if (axis.type().is(&PyEllipsis_Type)) {
        reduced = is_tensor ? full : (uint64_t) 1;
    } else if (axis.is_none()) {
        reduced = full;
    } else {
        reduced = 0;
        auto add = [n, &reduced](int v) {
            if (v < 0) v += n;
            if (v < 0 || v >= n) return false;
            reduced |= (uint64_t) 1 << v;
            return true;
        };
        int v;
        if (nb::isinstance<nb::tuple>(axis)) {
            for (auto a : nb::cast<nb::tuple>(axis))
                if (!nb::try_cast(a, v) || !add(v)) return std::nullopt;
        } else if (!nb::try_cast(axis, v) || !add(v)) {
            return std::nullopt;
        }
    }

    dr::vector<size_t> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        if ((reduced >> i) & 1) {
            if (keepdims) out.push_back(1);
        } else {
            out.push_back(in_shape[i]);
        }
    }
    return out;
}

/// Shared implementation for ``norm`` and ``squared_norm``. We route
/// through ``dot()`` whenever the reduction effectively collapses the
/// input to a scalar -- detected by asking ``reduction_output_shape``
/// what shape ``sum`` would produce and checking that it is "all ones"
/// (i.e., the result has a single element). For tensors this routes via
/// the underlying 1D buffer to engage the fused dot kernel. For non-
/// tensor inputs ``dot(h, h)`` itself only reduces axis 0, so we use it
/// when the default axis is requested *or* when ``shape[1:]`` is
/// trivial.
static nb::object norm_impl(nb::handle h, nb::handle axis, nb::handle mode,
                            bool keepdims, bool sqrt_wrap) {
    auto wrap = [sqrt_wrap](nb::object v) -> nb::object {
        return sqrt_wrap ? array_module.attr("sqrt")(v) : v;
    };

    auto all_ones = [](auto begin, auto end) {
        for (auto it = begin; it != end; ++it)
            if (*it != 1) return false;
        return true;
    };

    if (!keepdims && mode.is_none()) {
        nb::handle tp = h.type();
        bool default_axis = axis.type().is(&PyEllipsis_Type);

        if (!is_drjit_type(tp)) {
            // Python sequences: ``square`` fails on plain lists, so only
            // the iterating dot fallback applies.
            if (default_axis)
                return wrap(array_module.attr("dot")(h, h));
        } else {
            const ArraySupplement &s = supp(tp);
            dr::vector<size_t> shape;
            shape_impl(h, shape);

            auto out = reduction_output_shape(axis, shape, s.is_tensor,
                                                /* keepdims = */ false);
            bool whole = out.has_value() && all_ones(out->begin(), out->end());

            if (whole && s.is_tensor) {
                nb::object flat = nb::steal(s.tensor_array(h.ptr()));
                return tp(wrap(array_module.attr("dot")(flat, flat)),
                          nb::tuple());
            }
            if (!s.is_tensor &&
                (default_axis ||
                 (whole && all_ones(shape.begin() + 1, shape.end()))))
                return wrap(array_module.attr("dot")(h, h));
        }
    }

    nb::object sq = array_module.attr("square")(h);
    return wrap(sum(sq, axis, mode, keepdims));
}


void export_reduce(nb::module_ & m) {
    m.def("reduce", &reduce_py, "op"_a, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_reduce,
          nb::sig("def reduce(op: ReduceOp, value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("all", &all, "value"_a, "axis"_a.none() = nb::ellipsis(), "keepdims"_a = false, doc_all,
          nb::sig("def all(value: object, axis: int | tuple[int, ...] | ... | None = ..., keepdims: bool = False) -> object"))
     .def("any", &any, "value"_a, "axis"_a.none() = nb::ellipsis(), "keepdims"_a = false, doc_any,
          nb::sig("def any(value: object, axis: int | tuple[int, ...] | ... | None = ..., keepdims: bool = False) -> object"))
     .def("none", &none, "value"_a, "axis"_a.none() = nb::ellipsis(), "keepdims"_a = false, doc_none,
          nb::sig("def none(value: object, axis: int | tuple[int, ...] | ... | None = ..., keepdims: bool = False) -> object"))
     .def("count", &count, "value"_a, "axis"_a.none() = nb::ellipsis(), "keepdims"_a = false, doc_count,
          nb::sig("def count(value: object, axis: int | tuple[int, ...] | ... | None = ..., keepdims: bool = False) -> object"))
     .def("sum", &sum, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_sum,
          nb::sig("def sum(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("prod", &prod, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_prod,
          nb::sig("def prod(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("min", &min, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_min,
          nb::sig("def min(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("max", &max, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_max,
          nb::sig("def max(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("mean", &mean, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_mean,
          nb::sig("def mean(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("prefix_reduce", &prefix_reduce, "op"_a, "value"_a, "axis"_a = 0, "exclusive"_a = true, "reverse"_a = false, doc_prefix_reduce,
          nb::sig("def prefix_reduce(op: ReduceOp, value: T, axis: int | tuple[int, ...] = 0, exclusive: bool = True, reverse: bool = False) -> T"))
     .def("dot", &dot, doc_dot)
     .def("abs_dot",
          [](nb::handle h0, nb::handle h1) -> nb::object {
              return array_module.attr("abs")(
                  array_module.attr("dot")(h0, h1));
          }, doc_abs_dot)
     .def("norm",
          [](nb::handle h, nb::handle axis, nb::handle mode, bool keepdims) -> nb::object {
              return norm_impl(h, axis, mode, keepdims, /* sqrt_wrap = */ true);
          }, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_norm,
          nb::sig("def norm(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("squared_norm",
          [](nb::handle h, nb::handle axis, nb::handle mode, bool keepdims) -> nb::object {
              return norm_impl(h, axis, mode, keepdims, /* sqrt_wrap = */ false);
          }, "value"_a, "axis"_a.none() = nb::ellipsis(), "mode"_a = nb::none(), "keepdims"_a = false, doc_squared_norm,
          nb::sig("def squared_norm(value: object, axis: int | tuple[int, ...] | ... | None = ..., mode: str | None = None, keepdims: bool = False) -> object"))
     .def("block_prefix_reduce", &block_prefix_reduce, "op"_a, "value"_a, "block_size"_a, "exclusive"_a = true, "reverse"_a = false,
          doc_block_prefix_reduce,
          nb::sig("def block_prefix_reduce(op: ReduceOp, value: ArrayT, block_size: int, exclusive: bool = True, reverse: bool = False) -> ArrayT"))
     .def("block_reduce", &block_reduce, "op"_a, "value"_a, "block_size"_a, "mode"_a = nb::none(), doc_block_reduce,
          nb::sig("def block_reduce(op: ReduceOp, value: T, block_size: int, mode: Literal['evaluated', 'symbolic', None] = None) -> T"))
     .def("block_sum", &block_sum, "value"_a, "block_size"_a, "mode"_a = nb::none(), doc_block_sum,
          nb::sig("def block_sum(value: T, block_size: int, mode: Literal['evaluated', 'symbolic', None] = None) -> T"))
     .def("compress", &compress, doc_compress)
     .def("cumsum", [](nb::handle value, nb::handle axis, bool reverse) {
             return prefix_reduce(ReduceOp::Add, value, axis, false, reverse);
          }, "value"_a, "axis"_a = 0, "reverse"_a = false, doc_cumsum,
          nb::sig("def cumsum(value: T, axis: Union[int, tuple[int, ...]] = 0, reverse: bool = False) -> T"))
     .def("prefix_sum", [](nb::handle value, nb::handle axis, bool reverse) {
             return prefix_reduce(ReduceOp::Add, value, axis, true, reverse);
          }, "value"_a, "axis"_a = 0, "reverse"_a = false, doc_prefix_sum,
          nb::sig("def prefix_sum(value: T, axis: Union[int, tuple[int, ...]] = 0, reverse: bool = False) -> T"))
     .def("block_prefix_sum",
          [](nb::handle value, uint32_t block_size, bool exclusive, bool reverse) {
              return block_prefix_reduce(ReduceOp::Add, value, block_size, exclusive, reverse);
          }, "value"_a, "block_size"_a, "exclusive"_a = true, "reverse"_a = false, doc_block_prefix_sum,
          nb::sig("def block_prefix_sum(value: T, block_size: int, exclusive: bool = True, reverse: bool = False) -> T"));
}
