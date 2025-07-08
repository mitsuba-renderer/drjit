/*
    src/coop_vec.cpp -- Python bindings for Cooperative CoopVecs

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include "base.h"
#include "init.h"
#include "meta.h"
#include "apply.h"
#include "coop_vec.h"
#include <drjit/autodiff.h>
#include "nanobind/nanobind.h"
#include "nanobind/nb_defs.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/typing.h>


/// Cooperative vector constructor
CoopVec::CoopVec(nb::handle arg) {
    construct(arg);
}

void CoopVec::construct(nb::handle arg) {
    nb::handle single_arg = nb::none();
    if (nb::len(arg) == 1)
        single_arg = arg[0];

    if (CoopVec *v = nullptr; nb::try_cast(single_arg, v, false), v != nullptr) {
        m_index = ad_var_inc_ref(v->m_index);
        m_size = v->m_size;
        m_type = v->m_type;
        return;
    }

    nb::handle arg_tp = single_arg.type();
    if (is_drjit_type(arg_tp)) {
        const ArraySupplement &s = supp(arg_tp);
        if (s.is_tensor) {
            const dr::vector<size_t> &shape = s.tensor_shape(inst_ptr(single_arg));
            if (shape.size() <= 2) {
                construct(nb::list(single_arg));
                return;
            }
        }
    }

    /// Flatten a PyTree into a set of 1D arrays used to construct a cooperative vector
    struct Flatten: TraverseCallback {
        std::vector<nb::object> result;

        void operator()(nb::handle h) {
            if ((JitBackend) supp(h.type()).backend != JitBackend::None)
                result.push_back(nb::borrow(h));
        }

        void traverse_unknown(nb::handle h) {
            if (PyIter_Check(h.ptr()))
                traverse("drjit.nn.CoopVec", *this, nb::list(h));
            else if (PyLong_CheckExact(h.ptr()) || PyFloat_CheckExact(h.ptr()))
                result.push_back(nb::borrow(h));
            else
                nb::raise("encountered an unknown type \"%s\"", nb::inst_name(h).c_str());
        }
    };

    Flatten cb;
    traverse("drjit.nn.CoopVec", cb, arg);

    uint32_t size = (uint32_t) cb.result.size();

    if (cb.result.empty())
        nb::raise("drjit.nn.CoopVec(): cannot be empty!");

    // Identify type
    for (uint32_t i = 0; i < size; ++i) {
        nb::handle tp = cb.result[i].type();
        if (is_drjit_type(tp)) {
            m_type = tp;
            break;
        }
    }

    // Check that this type makes sense
    if (!m_type.is_valid())
        nb::raise_type_error(
            "drjit.nn.CoopVec(): at least one Jit-compiled 1D array is required as input "
            "(e.g., of type 'drjit.cuda.Float16')!");

    const ArraySupplement &s = supp(m_type);
    if (s.ndim != 1 || (JitBackend) s.backend == JitBackend::None)
        nb::raise_type_error(
            "drjit.nn.CoopVec(): expected Jit-compiled 1D arrays as input "
            "(e.g., of type 'drjit.cuda.Float16')!");

    // Check/cast the other arguments
    uint64_t *tmp = (uint64_t *) alloca(sizeof(uint64_t) * size);
    for (uint32_t i = 0; i < size; ++i) {
        nb::object value = cb.result[i];
        try {
            if (!value.type().is(m_type)) {
                value = m_type(value);
                cb.result[i] = value;
            }
            tmp[i] = s.index(inst_ptr(value));
        } catch (...) {
            nb::raise_type_error(
                "drjit.nn.CoopVec.__init__(): encountered an incompatible "
                "argument of type \"%s\" (expected \"%s\")!",
                nb::inst_name(value).c_str(),
                nb::type_name(m_type).c_str());
        }
    }

    m_index = ad_coop_vec_pack(size, tmp);
    m_size = size;
}

/// Unpack a cooperative vector into a Python list
nb::list CoopVec::expand_to_list() const {
    if (m_size == 0)
        return nb::list();

    uint64_t *tmp = (uint64_t *) alloca(m_size * sizeof(uint64_t));
    ad_coop_vec_unpack(m_index, m_size, tmp);

    nb::list result;
    const ArraySupplement &s = supp(m_type);
    for (uint32_t i = 0; i < m_size; ++i) {
        nb::object o = nb::inst_alloc(m_type);
        s.init_index(tmp[i], inst_ptr(o));
        ad_var_dec_ref(tmp[i]);
        nb::inst_mark_ready(o);
        result.append(std::move(o));
    }
    return result;
}

/// Unpack a cooperative vector into a Dr.Jit array type like ArrayXf
nb::object CoopVec::expand_to_vector() const {
    ArrayMeta m = supp(m_type);
    m.ndim = 2;
    m.shape[0] = DRJIT_DYNAMIC;
    m.shape[1] = DRJIT_DYNAMIC;
    return meta_get_type(m)(expand_to_list());
}

/// Perform one of several supported unary operations
template <JitOp Op> static CoopVec coop_vec_unary_op(const CoopVec &arg) {
    if ((JitBackend) supp(arg.m_type).backend == JitBackend::LLVM) {
        nb::object unpacked = arg.expand_to_vector(), func;

        switch (Op) {
            case JitOp::Exp2: func = array_module.attr("exp2"); break;
            case JitOp::Tanh: func = array_module.attr("tanh"); break;
            case JitOp::Log2: func = array_module.attr("log2"); break;
            default:
                nb::raise("Unsupported operation!");
        }

        return CoopVec(func(unpacked));
    }

    return CoopVec(
        ad_coop_vec_unary_op(Op, arg.m_index),
        arg.m_size,
        arg.m_type
    );
}

/// Perform one of several supported binary operations
template <JitOp Op>
static nb::object coop_vec_binary_op(nb::handle h0, nb::handle h1) {
    nb::object o[2] { nb::borrow(h0), nb::borrow(h1) };
    CoopVec *ptr[2] { };
    CoopVec *c = nullptr;

    for (uint32_t i = 0; i < 2; ++i) {
        if (nb::try_cast(o[i], ptr[i], false))
            c = ptr[i];
    }
    if (!c)
        return nb::steal(NB_NEXT_OVERLOAD);

    for (uint32_t i = 0; i < 2; ++i) {
        if (ptr[i])
            continue;

        nb::list args;
        nb::object oi = c->m_type(o[i]);
        for (uint32_t j = 0; j < c->m_size; ++j)
            args.append(oi);

        o[i] = nb::cast(CoopVec(nb::borrow<nb::args>(nb::tuple(args))));
        if (!nb::try_cast(o[i], ptr[i], false))
            nb::raise("CoopVec::binary_op(): internal error");
    }

    return nb::cast(CoopVec(
        ad_coop_vec_binary_op(
            Op,
            ptr[0]->m_index,
            ptr[1]->m_index
        ),
        c->m_size,
        c->m_type
    ));
}

/// Perform a ternary operation (currently only FMA)
template <JitOp Op>
static nb::object coop_vec_ternary_op(nb::handle h0, nb::handle h1,
                                      nb::handle h2) {
    nb::object o[3] { nb::borrow(h0), nb::borrow(h1), nb::borrow(h2) };
    CoopVec *ptr[3] { };
    CoopVec *c = nullptr;

    for (uint32_t i = 0; i < 3; ++i) {
        if (nb::try_cast(o[i], ptr[i], false))
            c = ptr[i];
    }
    if (!c)
        return nb::steal(NB_NEXT_OVERLOAD);

    for (uint32_t i = 0; i < 3; ++i) {
        if (ptr[i])
            continue;

        nb::list args;
        for (uint32_t j = 0; j < c->m_size; ++j)
            args.append(c->m_type(o[i]));

        o[i] = nb::cast(CoopVec(nb::borrow<nb::args>(nb::tuple(args))));
        if (!nb::try_cast(o[i], ptr[i], false))
            nb::raise("CoopVec::ternary_op(): internal error");
    }

    return nb::cast(CoopVec(
        ad_coop_vec_ternary_op(
            Op,
            ptr[0]->m_index,
            ptr[1]->m_index,
            ptr[2]->m_index
        ),
        c->m_size,
        c->m_type
    ));
}

/// Matrix-vector product
static CoopVec matvec(const MatrixView &A,
                     const CoopVec &x,
                     std::optional<const MatrixView *> b,
                     bool transpose) {

    return {
        ad_coop_vec_matvec(
            A.index(),
            &A.descr,
            x.m_index,
            b.has_value() ? b.value()->index() : 0,
            b.has_value() ? &b.value()->descr : nullptr,
            ((int) transpose) ^ ((int) A.transpose)
        ),
        transpose ? A.descr.cols : A.descr.rows,
        x.m_type
    };
}

nb::str MatrixView::repr() const {
    const char *layout;
    switch (descr.layout) {
       case MatrixLayout::InferencingOptimal: layout = "inference"; break;
       case MatrixLayout::TrainingOptimal: layout = "training"; break;
       case MatrixLayout::RowMajor: layout = "row_major"; break;
       default: layout = "unknown"; break;
    }
    return nb::str(
        "drjit.nn.MatrixView[\n"
        "    dtype={},\n"
        "    layout=\"{}\",\n"
        "    shape=({}, {}),\n"
        "    stride={},\n"
        "    offset={}\n"
        "    size={}\n"
        "    buffer=<{} instance>\n"
        "]"
    ).format(
        descr.dtype,
        layout,
        descr.rows,
        descr.cols,
        descr.stride,
        descr.offset,
        descr.size,
        inst_name(buffer)
    );
}

uint64_t MatrixView::index() const {
    return supp(buffer.type()).index(inst_ptr(buffer));
}

MatrixView MatrixView::getitem(nb::object arg) const {
    nb::object s[2];

    if (descr.layout == MatrixLayout::InferencingOptimal ||
        descr.layout == MatrixLayout::TrainingOptimal)
        nb::raise("drjit.MatrixView.__getitem__(): slicing is not permitted for "
                  "training/inferencing-optimal layouts!");

    if (nb::isinstance<nb::tuple>(arg)) {
        size_t l = nb::len(arg);
        if (l == 0 || l > 2)
            nb::raise("drjit.MatrixView.__getitem__(): expected 1 or 2 terms in "
                      "slice expression (got %zu)!", l);
        s[0] = arg[0];
        if (l == 2)
            s[1] = arg[1];
    } else {
        s[0] = arg;
    }

    if (!s[1].is_valid())
        s[1] = nb::slice(nb::none(), nb::none(), nb::none());

    Py_ssize_t start[2], step[2];
    size_t len[2];

    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t value;
        if (nb::try_cast(s[i], value, false))
            s[i] = nb::slice(nb::int_(value), nb::int_(value + 1), nb::int_(1));
        nb::slice sl;
        if (!nb::try_cast(s[i], sl, false))
            nb::raise("drjit.MatrixView.__getitem__(): expected 'int' or 'slice' "
                      "in slice expression, got '%s'!",
                      nb::inst_name(s[i]).c_str());
        size_t limit = i == 0 ? descr.rows : descr.cols;
        auto [start_i, stop_i, step_i, len_i] =
            sl.compute(limit);
        start[i] = start_i; step[i] = step_i; len[i] = len_i;
    }

    if (step[1] != 1)
        nb::raise("drjit.MatrixView.__getitem__(): rows elements must be contiguous!");

    if (len[0] == 0 || len[1] == 0)
        nb::raise("drjit.MatrixView.__getitem__(): input array may not be empty!");

    MatrixView result;
    result.descr.rows = (uint32_t) len[0];
    result.descr.cols = (uint32_t) len[1];
    result.descr.offset = (uint32_t) (descr.offset + start[0] * descr.stride + start[1]);
    result.descr.dtype = descr.dtype;
    result.descr.layout = descr.layout;
    result.descr.stride = (uint32_t) (descr.stride * step[0]);
    result.descr.size = (uint32_t) ((len[0] - 1) * result.descr.stride + len[1]);
    result.buffer = buffer;
    return result;
}

static MatrixView view(nb::handle_t<dr::ArrayBase> arg) {
    MatrixView result { };
    MatrixDescr &d = result.descr;

    const ArraySupplement &s = supp(arg.type());

    d.dtype = (VarType) s.type;
    d.layout = MatrixLayout::RowMajor;

    if (s.is_tensor) {
        const dr::vector<size_t> &shape = s.tensor_shape(inst_ptr(arg));
        if (shape.size() != 1 && shape.size() != 2)
            nb::raise("drjit.view(): tensor must have 1 or 2 dimensions!");
        d.rows = (uint32_t) shape[0];
        d.cols = (uint32_t) (shape.size() > 1 ? shape[1] : 1);
        result.buffer = nb::steal(s.tensor_array(arg.ptr()));
    } else if (s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC) {
        d.rows = (uint32_t) nb::len(arg);
        d.cols = 1u;
        result.buffer = nb::borrow(arg);
    } else {
        nb::raise("Unsupported input type!");
    }

    d.stride = d.cols;
    d.size = d.rows * d.cols;
    d.offset = 0;

    if (d.rows == 0 || d.cols == 0)
        nb::raise("drjit.view(): input array/tensor may not be empty!");

    return result;
}

struct RepackItem {
    nb::object in_o;
    nb::object out_o;
    MatrixView *in;
    MatrixView *out;

    RepackItem(nb::handle in_o, nb::handle out_o, MatrixView *in, MatrixView *out)
        : in_o(nb::borrow(in_o)), out_o(nb::borrow(out_o)), in(in), out(out) { }
    RepackItem(RepackItem&&) = default;
    RepackItem(const RepackItem&) = default;
};

nb::handle view_type;
nb::handle coop_vector_type;

static nb::object repack_impl(const char *name, MatrixLayout layout,
                              nb::handle arg_, uint32_t &offset,
                              std::vector<RepackItem> &items) {
    nb::handle arg_tp = arg_.type();
    nb::object arg = nb::borrow(arg_);

    if (is_drjit_type(arg_tp) && layout != MatrixLayout::RowMajor) {
        arg = nb::cast(view(nb::handle_t<dr::ArrayBase>(arg)));
        arg_tp = view_type;
    }

    if (arg_tp.is(view_type)) {
        MatrixView *in_view = nb::cast<MatrixView *>(arg, false);
        uint64_t in_index = supp(in_view->buffer.type()).index(inst_ptr(in_view->buffer));
        MatrixDescr out_descr =
            jit_coop_vec_compute_layout((uint32_t) in_index, &in_view->descr, layout, offset);
        MatrixView *out_view = new MatrixView{out_descr, nb::none()};
        nb::object result = nb::cast(out_view, nb::rv_policy::take_ownership);
        items.emplace_back(arg, result, in_view, out_view);
        offset = out_descr.offset + out_descr.size;
        return result;
    } else if (arg_tp.is(&PyTuple_Type)) {
        nb::tuple t = nb::borrow<nb::tuple>(arg);
        nb::list result;
        for (nb::handle h : t)
            result.append(repack_impl(name, layout, h, offset, items));
        return nb::tuple(result);
    } else if (arg_tp.is(&PyList_Type)) {
        nb::list l = nb::borrow<nb::list>(arg);
        nb::list result;
        for (nb::handle h : l)
            result.append(repack_impl(name, layout, h, offset, items));
        return std::move(result);
    } else if (arg_tp.is(&PyDict_Type)) {
        nb::dict d = nb::borrow<nb::dict>(arg);
        nb::dict result;
        for (auto [k, v] : d)
            result[k] = repack_impl(name, layout, v, offset, items);
        return std::move(result);
    } else if (nb::dict ds = get_drjit_struct(arg_tp); ds.is_valid()) {
        nb::object tmp = arg_tp();
        for (auto [k, v] : ds)
            nb::setattr(tmp, k, repack_impl(name, layout, nb::getattr(arg, k), offset, items));
        return tmp;
    } else if (nb::object df = get_dataclass_fields(arg_tp); df.is_valid()) {
        nb::object tmp = nb::dict();
        for (nb::handle field : df) {
            nb::object k = field.attr(DR_STR(name));
            tmp[k] = repack_impl(name, layout, nb::getattr(arg, k), offset, items);
        }
        return arg_tp(**tmp);
    } else {
        return nb::borrow(arg);
    }
}

static std::pair<nb::object, nb::object> repack(const char *name, const char *layout_str, nb::handle arg) {
    uint32_t offset = 0;
    std::vector<RepackItem> items;
    MatrixLayout layout;

    if (layout_str) {
        if (strcmp(layout_str, "inference") == 0)
            layout = MatrixLayout::InferencingOptimal;
        else if (strcmp(layout_str, "training") == 0)
            layout = MatrixLayout::TrainingOptimal;
        else
            nb::raise("drjit.%s(): 'mode' must equal \"inference\" or \"training\"!", name);
    } else {
        layout = MatrixLayout::RowMajor;
    }

    nb::object result = repack_impl(name, layout, arg, offset, items);
    nb::object buffer = nb::none();

    if (items.size() > 0) {
        nb::handle buf_cur = items[0].in->buffer,
                   buf_tp = buf_cur.type();

        buffer = full("zeros", buf_tp, nb::int_(0), offset, true);
        const ArraySupplement &s = supp(buf_tp);

        std::vector<MatrixDescr> in, out;
        in.reserve(items.size());
        out.reserve(items.size());

        auto submit = [&] {
            jit_coop_vec_pack_matrices(
                (uint32_t) in.size(),
                (uint32_t) s.index(inst_ptr(buf_cur)),
                in.data(),
                (uint32_t) s.index(inst_ptr(buffer)),
                out.data()
            );
        };

        for (size_t i = 0; i < items.size(); ++i) {
            nb::handle buf_i = items[i].in->buffer,
                       buf_i_tp = buf_i.type();

            if (!buf_i_tp.is(buf_tp)) {
                nb::raise_type_error(
                    "drjit.%s(): encountered different input formats (%s vs %s)", name,
                    nb::type_name(buf_tp).c_str(),
                    nb::type_name(buf_i_tp).c_str());
            }

            if (!buf_cur.is(buf_i)) {
                submit();
                in.clear();
                out.clear();
                buf_cur = buf_i;
            }

            items[i].out->buffer = buffer;

            in.push_back(items[i].in->descr);
            out.push_back(items[i].out->descr);
        }

        if (!in.empty())
            submit();
    }

    return { buffer, result };
}

static CoopVec coopvec_abs_workaround(nb::handle_t<CoopVec> &v) {
    nb::list result;
    for (nb::handle h: v)
        result.append(nb::steal(PyNumber_Absolute(h.ptr())));
    return CoopVec(result);
}

void export_coop_vec(nb::module_ &m) {
    nb::module_ nn = m.def_submodule("detail").def_submodule("nn");
    nn.attr("__name__") = "drjit.nn";

    nn.attr("ArrayT") = nb::type_var("ArrayT", "bound"_a = "drjit.ArrayBase");
    for (const char *name :
         { "T", "SelfT", "SelfCpT", "ValT", "ValCpT", "RedT", "PlainT", "MaskT" })
        nn.attr(name) = nb::type_var(name);

    coop_vector_type = nb::class_<CoopVec>(nn, "CoopVec", nb::is_generic(), nb::sig("class CoopVec(typing.Generic[T])"))
        .def(nb::init<nb::args>(),
             nb::sig("def __init__(self, *args: typing.Unpack[typing.Tuple[typing.Union[drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, T, PlainT, MaskT], float, int], ...]]) -> None"),
             doc_nn_CoopVec_init)
        .def("__iter__", [](const CoopVec &v) { return iter(v.expand_to_list()); },
             nb::sig("def __iter__(self, /) -> typing.Iterator[T]"))
        .def("__add__", &coop_vec_binary_op<JitOp::Add>,
             nb::sig("def __add__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]"))
        .def("__radd__", &coop_vec_binary_op<JitOp::Add>,
             nb::sig("def __radd__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]"))
        .def("__sub__", &coop_vec_binary_op<JitOp::Sub>,
             nb::sig("def __sub__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]"))
        .def("__rsub__", &coop_vec_binary_op<JitOp::Sub>,
             nb::sig("def __rsub__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]"))
        .def("__mul__", &coop_vec_binary_op<JitOp::Mul>,
             nb::sig("def __mul__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]"))
        .def("__rmul__", &coop_vec_binary_op<JitOp::Mul>,
             nb::sig("def __rmul__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]"))
        .def_prop_ro("index", [](const CoopVec &v) { return v.m_index; })
        .def_prop_ro("type", [](const CoopVec &v) { return v.m_type; })
        .def("__len__", [](const CoopVec &v) { return v.m_size; })
        .def("__abs__", &coopvec_abs_workaround)
        .def("__repr__",
             [](const CoopVec &v) {
                 return nb::str("drjit.nn.CoopVec[{}, shape=({}, {})]")
                     .format(nb::type_name(v.m_type), v.m_size,
                             jit_var_size((uint32_t) v.m_index));
             });

    view_type = nb::class_<MatrixView>(nn, "MatrixView", doc_nn_MatrixView)
        .def(nb::init<>())
        .def("__repr__", &MatrixView::repr)
        .def("__getitem__", &MatrixView::getitem,
             nb::sig("def __getitem__(self, arg: typing.Union[int, slice, typing.Tuple[typing.Union[int, slice], typing.Union[int, slice]]]) -> MatrixView"))
        .def_prop_rw("dtype",
                     [](MatrixView &v) { return v.descr.dtype; },
                     [](MatrixView &v, VarType v2) { v.descr.dtype = v2; }, doc_nn_MatrixView_dtype)
        .def_prop_rw("offset",
                     [](MatrixView &v) { return v.descr.offset; },
                     [](MatrixView &v, uint32_t v2) { v.descr.offset = v2; }, doc_nn_MatrixView_offset)
        .def_prop_rw("stride",
                     [](MatrixView &v) { return v.descr.stride; },
                     [](MatrixView &v, uint32_t v2) { v.descr.stride = v2; }, doc_nn_MatrixView_stride)
        .def_prop_rw("size",
                     [](MatrixView &v) { return v.descr.size; },
                     [](MatrixView &v, uint32_t v2) { v.descr.size = v2; }, doc_nn_MatrixView_size)
        .def_prop_rw("layout",
                     [](MatrixView &v) {
                         switch (v.descr.layout) {
                            case MatrixLayout::InferencingOptimal: return "inference";
                            case MatrixLayout::TrainingOptimal: return "training";
                            case MatrixLayout::RowMajor: return "row_major";
                            default: return "unknown";
                         }
                     },
                     [](MatrixView &v, const char *s) {
                         if (strcmp(s, "inference") == 0)
                             v.descr.layout = MatrixLayout::InferencingOptimal;
                         else if (strcmp(s, "training") == 0)
                             v.descr.layout = MatrixLayout::TrainingOptimal;
                         else if (strcmp(s, "row_major") == 0)
                             v.descr.layout = MatrixLayout::RowMajor;
                         else
                             nb::raise("Unknown layout!");
                     },
                     nb::for_getter(nb::sig("def layout(self) -> typing.Literal['inference', 'training', 'row_major']")),
                     nb::for_setter(nb::sig("def layout(self, value: typing.Literal['inference', 'training', 'row_major']) -> None")),
                     doc_nn_MatrixView_layout)
        .def_prop_rw("transpose",
                     [](MatrixView &v) { return v.transpose; },
                     [](MatrixView &v, bool v2) { v.transpose = v2; },
                     doc_nn_MatrixView_transpose)
        .def_prop_rw("shape",
                     [](MatrixView &v) {
                         return std::make_pair(v.descr.rows, v.descr.cols);
                     },
                     [](MatrixView &v, std::pair<uint32_t, uint32_t> v2) {
                         v.descr.rows = v2.first;
                         v.descr.cols = v2.second;
                     },
                     doc_nn_MatrixView_shape)
        .def("__matmul__", [](const MatrixView &self, const CoopVec &x) { return matvec(self, x, {}, false); },
             nb::sig("def __matmul__(self, arg: CoopVec[T], /) -> CoopVec[T]"))
        .def_rw("buffer", &MatrixView::buffer,
                doc_nn_MatrixView_buffer)
        .def_prop_ro("T",
                     [](MatrixView &v) {
                         MatrixView r;
                         r.descr = v.descr;
                         r.buffer = v.buffer;
                         r.transpose = !v.transpose;
                         return r;
                     })
        .def_prop_ro("grad",
                     [](MatrixView &v) {
                         MatrixView r;
                         r.descr = v.descr;
                         r.buffer = v.buffer.attr("grad");
                         r.transpose = v.transpose;
                         return r;
                     });


    nb::dict drjit_struct;
    drjit_struct["layout"] = nb::handle(&PyUnicode_Type);
    drjit_struct["buffer"] = nb::none();
    drjit_struct["dtype"] = nb::type<VarType>();
    drjit_struct["shape"] = nb::handle(&PyTuple_Type);
    drjit_struct["offset"] = nb::handle(&PyLong_Type);
    drjit_struct["size"] = nb::handle(&PyLong_Type);
    drjit_struct["stride"] = nb::handle(&PyLong_Type);
    drjit_struct["transpose"] = nb::handle(&PyBool_Type);
    view_type.attr("DRJIT_STRUCT") = drjit_struct;

    nn.def("view", &view,
           doc_nn_view);

    nn.def("pack", [](nb::handle arg, const char *layout) { return repack("pack", layout, arg); },
           nb::arg(), "layout"_a = "inference",
           nb::sig("def pack(arg: MatrixView | drjit.AnyArray, *, layout: typing.Literal['inference', 'training'] = 'inference') -> typing.Tuple[drjit.ArrayBase, MatrixView]"),
           doc_nn_pack);

    nn.def("pack",
           [](nb::args args, const char *layout) {
               auto temp = repack("pack", layout, args);
               nb::list l;
               l.append(temp.first);
               l.extend(temp.second);
               return nb::tuple(l);
           },
           "args"_a, "layout"_a = "inference",
           nb::sig("def pack(*args: PyTree, layout: typing.Literal['inference', "
                   "'training'] = 'inference') -> typing.Tuple[drjit.ArrayBase, "
                   "typing.Unpack[typing.Tuple[PyTree, ...]]]"));

    nn.def("unpack", [](nb::handle arg) {
        return repack("unpack", nullptr, arg); },
           nb::sig("def unpack(arg: MatrixView | drjit.AnyArray, /) -> typing.Tuple[drjit.ArrayBase, MatrixView]"),
           doc_nn_unpack);

    nn.def("unpack",
           [](nb::args args) {
               auto temp = repack("unpack", nullptr, args);
               nb::list l;
               l.append(temp.first);
               l.extend(temp.second);
               return nb::tuple(l);
           },
           "args"_a,
           nb::sig("def unpack(*args: PyTree) -> typing.Tuple[drjit.ArrayBase, "
                   "typing.Unpack[typing.Tuple[PyTree, ...]]]"));

    nn.def("matvec", &matvec, "A"_a.noconvert(), "x"_a.noconvert(),
           "b"_a.noconvert() = nb::none(), "transpose"_a = false,
            nb::sig("def matvec(A: MatrixView, x: drjit.nn.CoopVec[T], b: typing.Optional[MatrixView] = "
                    "None, /, transpose: bool = False) -> drjit.nn.CoopVec[T]"),
            doc_nn_matvec);

    nn.def("cast",
           [](CoopVec vec, nb::type_object_t<drjit::ArrayBase> tp) {
               const ArraySupplement &s = supp(tp);
               ArrayMeta m = supp(vec.m_type);
               m.type = s.type;
               nb::handle new_type = meta_get_type(m);
               return CoopVec(ad_coop_vec_cast(vec.m_index, (VarType) s.type),
                              vec.m_size, new_type);
           }, nb::sig("def cast(arg0: CoopVec[T], arg1: typing.Type[ArrayT], /) -> CoopVec[ArrayT]"),
           doc_nn_cast
    );

    m.def("fma", &coop_vec_ternary_op<JitOp::Fma>);
    m.def("minimum", &coop_vec_binary_op<JitOp::Min>);
    m.def("maximum", &coop_vec_binary_op<JitOp::Max>);
    m.def("step", &coop_vec_binary_op<JitOp::Step>, doc_step);
    m.def("log2", &coop_vec_unary_op<JitOp::Log2>);
    m.def("exp2", &coop_vec_unary_op<JitOp::Exp2>);
    m.def("tanh", &coop_vec_unary_op<JitOp::Tanh>);
    m.def("step", [](nb::handle h0, nb::handle h1) {
        return select(
            nb::steal(PyObject_RichCompare(h0.ptr(), h1.ptr(), Py_LT)),
            nb::int_(0), nb::int_(1));
    });
    m.def("abs", coopvec_abs_workaround);
}
