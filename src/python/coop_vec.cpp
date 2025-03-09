/*
    src/coop_vec.cpp -- Python bindings for Cooperative Vectors

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include "base.h"
#include "init.h"
#include "meta.h"
#include "coop_vec.h"
#include <drjit/autodiff.h>
#include "nanobind/nanobind.h"
#include "nanobind/nb_defs.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/typing.h>

/// Flatten a PyTree into a set of 1D arrays used to construct a cooperative vector
void vector_collect(nb::handle arg, dr::vector<nb::object> &result) {
    nb::handle arg_tp = arg.type();

    if (is_drjit_type(arg_tp)) {
        const ArraySupplement &s = supp(arg_tp);
        if (s.ndim > 1) {
            size_t len = nb::len(arg);
            for (size_t i = 0; i < len; ++i)
                vector_collect(arg[i], result);
        } else {
            result.push_back(nb::borrow(arg));
        }
    } else if (arg_tp.is(&PyLong_Type) || arg_tp.is(&PyFloat_Type)) {
        result.push_back(nb::borrow(arg));
    } else if (PyGen_CheckExact(arg.ptr())) {
        vector_collect(nb::list(arg), result);
    } else if (arg_tp.is(&PyTuple_Type)) {
        nb::tuple t = nb::borrow<nb::tuple>(arg);
        for (nb::handle h : t)
            vector_collect(h, result);
    } else if (arg_tp.is(&PyList_Type)) {
        nb::list l = nb::borrow<nb::list>(arg);
        for (nb::handle h : l)
            vector_collect(h, result);
    } else if (arg_tp.is(&PyDict_Type)) {
        nb::dict d = nb::borrow<nb::dict>(arg);
        for (auto [k, v] : d)
            vector_collect(v, result);
    } else {
        nb::raise("drjit.coop.Vector(): encountered an unknown type \"%s\"", nb::type_name(arg_tp).c_str());
    }
}

/// Cooperative vector constructor
Vector::Vector(nb::handle arg) {
    dr::vector<nb::object> result;
    vector_collect(arg, result);
    uint32_t size = (uint32_t) result.size();

    if (result.empty())
        nb::raise("drjit.coop.Vector(): cannot be empty!");

    // Identify type
    for (uint32_t i = 0; i < size; ++i) {
        nb::handle tp = result[i].type();
        if (is_drjit_type(tp)) {
            m_type = tp;
            break;
        }
    }

    // Check that this type makes sense
    if (!m_type.is_valid())
        nb::raise_type_error(
            "drjit.coop.Vector(): at least one Jit-compiled 1D array is required as input "
            "(e.g., of type 'drjit.cuda.Float16')!");

    const ArraySupplement &s = supp(m_type);
    if (s.ndim != 1 || (JitBackend) s.backend == JitBackend::None)
        nb::raise_type_error(
            "drjit.coop.Vector(): expected Jit-compiled 1D arrays as input "
            "(e.g., of type 'drjit.cuda.Float16')!");

    // Check/cast the other arguments
    uint64_t *tmp = (uint64_t *) alloca(sizeof(uint64_t) * size);
    for (uint32_t i = 0; i < size; ++i) {
        nb::object value = result[i];
        try {
            if (!value.type().is(m_type)) {
                value = m_type(value);
                result[i] = value;
            }
            tmp[i] = s.index(inst_ptr(value));
        } catch (...) {
            nb::raise_type_error(
                "drjit.coop.Vector.__init__(): encountered an incompatible "
                "argument of type \"%s\" (expected \"%s\")!",
                nb::inst_name(value).c_str(),
                nb::type_name(m_type).c_str());
        }
    }

    m_index = ad_coop_vec_pack(size, tmp);
    m_size = size;
}

/// Unpack a cooperative vector into a Python list
nb::list Vector::expand_to_list() const {
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

/// Unpack a cooperative vecotr into a Dr.Jit array type like VectorXf
nb::object Vector::expand_to_vector() const {
    ArrayMeta m = supp(m_type);
    m.ndim = 2;
    m.shape[0] = DRJIT_DYNAMIC;
    m.shape[1] = DRJIT_DYNAMIC;
    return meta_get_type(m)(expand_to_list());
}

/// Perform one of several supported unary operations
template <JitOp Op> static Vector coop_vec_unary_op(const Vector &arg) {
    if ((JitBackend) supp(arg.m_type).backend == JitBackend::LLVM) {
        nb::object unpacked = arg.expand_to_vector(), func;

        switch (Op) {
            case JitOp::Exp2: func = array_module.attr("exp2"); break;
            case JitOp::Tanh: func = array_module.attr("tanh"); break;
            case JitOp::Log2: func = array_module.attr("log2"); break;
            default:
                nb::raise("Unsupported operation!");
        }

        return Vector(func(unpacked));
    }

    return Vector(
        jit_coop_vec_unary_op(Op, arg.m_index),
        arg.m_size,
        arg.m_type
    );
}

/// Perform one of several supported binary operations
template <JitOp Op>
static nb::object coop_vec_binary_op(nb::handle h0, nb::handle h1) {
    nb::object o[2] { nb::borrow(h0), nb::borrow(h1) };
    Vector *ptr[2] { };
    Vector *c = nullptr;

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

        o[i] = nb::cast(Vector(nb::borrow<nb::args>(nb::tuple(args))));
        if (!nb::try_cast(o[i], ptr[i], false))
            nb::raise("Vector::binary_op(): internal error");
    }

    return nb::cast(Vector(
        ad_coop_vec_binary_op(
            Op,
            ptr[0]->m_index,
            ptr[1]->m_index
        ),
        c->m_size,
        c->m_type
    ));
}

/// Implements the step() operation for ordinary arrays and cooperative vectors
static nb::object coop_vec_step(nb::handle h0, nb::handle h1) {
    nb::object o[2] { nb::borrow(h0), nb::borrow(h1) };
    Vector *ptr[2] { };
    Vector *c = nullptr;

    for (uint32_t i = 0; i < 2; ++i) {
        if (nb::try_cast(o[i], ptr[i], false))
            c = ptr[i];
    }

    if (!c) {
        return select(
            nb::steal(PyObject_RichCompare(h0.ptr(), h1.ptr(), Py_LT)),
            nb::int_(0), nb::int_(1));
    }

    for (uint32_t i = 0; i < 2; ++i) {
        if (ptr[i])
            continue;

        nb::list args;
        nb::object oi = c->m_type(o[i]);
        for (uint32_t j = 0; j < c->m_size; ++j)
            args.append(oi);

        o[i] = nb::cast(Vector(nb::borrow<nb::args>(nb::tuple(args))));
        if (!nb::try_cast(o[i], ptr[i], false))
            nb::raise("Vector::binary_op(): internal error");
    }

    if ((JitBackend) supp(c->m_type).backend == JitBackend::LLVM) {
        return nb::cast(Vector(coop_vec_step(ptr[0]->expand_to_vector(),
                                             ptr[1]->expand_to_vector())));
    } else {
        return nb::cast(Vector(
            ad_coop_vec_binary_op(
                JitOp::Step,
                ptr[0]->m_index,
                ptr[1]->m_index
            ),
            c->m_size,
            c->m_type
        ));
    }
}

/// Perform a ternary operation (currently only FMA)
template <JitOp Op>
static nb::object coop_vec_ternary_op(nb::handle h0, nb::handle h1,
                                      nb::handle h2) {
    nb::object o[3] { nb::borrow(h0), nb::borrow(h1), nb::borrow(h2) };
    Vector *ptr[3] { };
    Vector *c = nullptr;

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

        o[i] = nb::cast(Vector(nb::borrow<nb::args>(nb::tuple(args))));
        if (!nb::try_cast(o[i], ptr[i], false))
            nb::raise("Vector::ternary_op(): internal error");
    }

    return nb::cast(Vector(
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
static Vector matvec(const View &A,
                     const Vector &x,
                     std::optional<const View *> b,
                     bool transpose) {

    return {
        jit_coop_vec_matvec(
            A.index(),
            &A.descr,
            x.m_index,
            b.has_value() ? b.value()->index() : 0,
            b.has_value() ? &b.value()->descr : nullptr,
            (int) transpose
        ),
        transpose ? A.descr.cols : A.descr.rows,
        x.m_type
    };
}

nb::str View::repr() const {
    return nb::str(
        "drjit.coop.View[\n"
        "    dtype={},\n"
        "    layout={},\n"
        "    shape=({}, {}),\n"
        "    stride={},\n"
        "    offset={}\n"
        "    size={}\n"
        "    buffer=<{} instance>\n"
        "]"
    ).format(
        descr.dtype,
        descr.layout,
        descr.rows,
        descr.cols,
        descr.stride,
        descr.offset,
        descr.size,
        inst_name(buffer)
    );
}

uint64_t View::index() const {
    return supp(buffer.type()).index(inst_ptr(buffer));
}

View View::getitem(nb::object arg) const {
    nb::object s[2];

    if (descr.layout == MatrixLayout::InferencingOptimal ||
        descr.layout == MatrixLayout::TrainingOptimal)
        nb::raise("drjit.View.__getitem__(): slicing is not permitted for "
                  "training/inferencing-optimal layouts!");

    if (nb::isinstance<nb::tuple>(arg)) {
        size_t l = nb::len(arg);
        if (l == 0 || l > 2)
            nb::raise("drjit.View.__getitem__(): expected 1 or 2 terms in "
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
            nb::raise("drjit.View.__getitem__(): expected 'int' or 'slice' "
                      "in slice expression, got '%s'!",
                      nb::inst_name(s[i]).c_str());
        size_t limit = i == 0 ? descr.rows : descr.cols;
        auto [start_i, stop_i, step_i, len_i] =
            sl.compute(limit);
        start[i] = start_i; step[i] = step_i; len[i] = len_i;
    }

    if (step[1] != 1)
        nb::raise("drjit.View.__getitem__(): rows elements must be contiguous!");

    if (len[0] == 0 || len[1] == 0)
        nb::raise("drjit.View.__getitem__(): input array may not be empty!");

    View result;
    result.descr.rows = len[0];
    result.descr.cols = len[1];
    result.descr.offset = descr.offset + start[0] * descr.stride + start[1];
    result.descr.dtype = descr.dtype;
    result.descr.layout = descr.layout;
    result.descr.stride = descr.stride * step[0];
    result.descr.size = (len[0] - 1) * result.descr.stride + len[1];
    result.buffer = buffer;
    return result;
}

static View view(nb::handle_t<dr::ArrayBase> arg) {
    View result { };
    MatrixDescr &d = result.descr;

    const ArraySupplement &s = supp(arg.type());

    d.dtype = (VarType) s.type;
    d.layout = MatrixLayout::RowMajor;

    if (s.is_tensor) {
        const dr::vector<size_t> &shape = s.tensor_shape(inst_ptr(arg));
        if (shape.size() != 1 && shape.size() != 2)
            nb::raise("drjit.view(): tensor must have 1 or 2 dimensions!");
        d.rows = shape[0];
        d.cols = shape.size() > 1 ? shape[1] : 1;
        result.buffer = nb::steal(s.tensor_array(arg.ptr()));
    } else if (s.ndim == 1) {
        d.rows = s.shape[0];
        d.cols = 1;
        result.buffer = nb::borrow(arg);
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
    View *in;
    View *out;

    RepackItem(nb::handle in_o, nb::handle out_o, View *in, View *out)
        : in_o(nb::borrow(in_o)), out_o(nb::borrow(out_o)), in(in), out(out) { }
    RepackItem(RepackItem&&) = default;
    RepackItem(const RepackItem&) = default;
};

static nb::handle view_type;

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
        View *in_view = nb::cast<View *>(arg, false);
        uint64_t in_index = supp(in_view->buffer.type()).index(inst_ptr(in_view->buffer));
        MatrixDescr out_descr =
            jit_coop_vec_compute_layout(in_index, &in_view->descr, layout, offset);
        View *out_view = new View{out_descr, nb::none()};
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
        return result;
    } else if (arg_tp.is(&PyDict_Type)) {
        nb::dict d = nb::borrow<nb::dict>(arg);
        nb::dict result;
        for (auto [k, v] : d)
            result[k] = repack_impl(name, layout, v, offset, items);
        return result;
    } else {
        nb::raise_type_error("drjit.%s(): unsupported type %s",
                             name, nb::type_name(arg_tp).c_str());
    }
}

static nb::object repack(const char *name, const char *layout_str, nb::handle arg) {
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
    if (items.size() > 0) {
        nb::handle buf_cur = items[0].in->buffer,
                   buf_tp = buf_cur.type();

        nb::object buf_o = full("zeros", buf_tp, nb::int_(0), offset, true);
        const ArraySupplement &s = supp(buf_tp);

        std::vector<MatrixDescr> in, out;
        in.reserve(items.size());
        out.reserve(items.size());

        auto submit = [&] {
            jit_coop_vec_pack_matrices(
                (uint32_t) in.size(),
                s.index(inst_ptr(buf_cur)),
                in.data(),
                s.index(inst_ptr(buf_o)),
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

            items[i].out->buffer = buf_o;

            in.push_back(items[i].in->descr);
            out.push_back(items[i].out->descr);
        }

        if (!in.empty())
            submit();
    }

    return result;
}

void export_coop_vec(nb::module_ &m) {
    nb::module_ coop = nb::module_::import_("drjit.coop");

    for (const char *name :
         { "T", "SelfT", "SelfCpT", "ValT", "ValCpT", "RedT", "PlainT", "MaskT" })
        m.attr(name) = nb::type_var(name);

    nb::class_<Vector>(coop, "Vector", nb::is_generic(), nb::sig("class Vector(typing.Generic[T])"))
        .def(nb::init<nb::args>(),
             nb::sig("def __init__(self, *args: *tuple[ArrayBase[SelfT, SelfCpT, ValT, ValCpT, T, PlainT, MaskT] | float | int, ...], /) -> None"),
             doc_coop_Vector_init)
        .def("__iter__", [](const Vector &v) { return iter(v.expand_to_list()); },
             nb::sig("def __iter__(self, /) -> Iterator[T]"))
        .def("__add__", &coop_vec_binary_op<JitOp::Add>,
             nb::sig("def __add__(self, arg: Vector[T] | T | float | int, /) -> Vector[T]"))
        .def("__sub__", &coop_vec_binary_op<JitOp::Sub>,
             nb::sig("def __sub__(self, arg: Vector[T] | T | float | int, /) -> Vector[T]"))
        .def("__mul__", &coop_vec_binary_op<JitOp::Mul>,
             nb::sig("def __mul__(self, arg: Vector[T] | T | float | int, /) -> Vector[T]"))
        .def_prop_ro("index", [](const Vector &v) { return v.m_index; })
        .def_prop_ro("type", [](const Vector &v) { return v.m_type; })
        .def("__len__", [](const Vector &v) { return v.m_size; })
        .def("__repr__",
             [](const Vector &v) {
                 return nb::str("drjit.coop.Vector[{}, {}]").format(nb::type_name(v.m_type), v.m_size);
             });

    view_type = nb::class_<View>(coop, "View", doc_coop_View)
        .def("__repr__", &View::repr)
        .def("__getitem__", &View::getitem,
             nb::sig("def __getitem__(self, int | slice | tuple[int | slice, int | slice]) -> View"))
        .def_prop_ro("dtype", [](View &r) { return r.descr.dtype; })
        .def_prop_ro("offset", [](View &r) { return r.descr.offset; })
        .def_prop_ro("stride", [](View &r) { return r.descr.stride; })
        .def_prop_ro("size", [](View &r) { return r.descr.size; })
        .def_prop_ro("layout", [](View &r) { return r.descr.layout; })
        .def_prop_ro("shape", [](View &r) { return std::make_pair(r.descr.rows, r.descr.cols); })
        .def_rw("buffer", &View::buffer);

    nb::enum_<MatrixLayout>(view_type, "MatrixLayout")
        .value("RowMajor", MatrixLayout::RowMajor)
        .value("InferencingOptimal", MatrixLayout::InferencingOptimal)
        .value("TrainingOptimal", MatrixLayout::TrainingOptimal);

    coop.def("view", &view,
             doc_coop_view);

    coop.def("pack", [](nb::handle arg, const char *layout) { return repack("pack", layout, arg); },
             nb::arg(), "layout"_a = "inference",
             nb::sig("def pack(arg: View | dr.AnyArray, *, layout: typing.Literal['inference', 'training'] = 'inference') -> View"),
             doc_coop_pack);

    coop.def("pack", [](nb::args args, const char *layout) { return repack("pack", layout, args); },
             "args"_a, "layout"_a = "inference");
             nb::sig("def pack(*args: *tuple[View | dr.AnyArray, ...], layout: typing.Literal['inference', 'training'] = 'inference') -> tuple[View, ...]");

    coop.def("unpack", [](nb::handle arg) { return repack("unpack", nullptr, arg); },
             nb::sig("def unpack(arg: View | dr.AnyArray, /) -> View"),
             doc_coop_unpack);

    coop.def("unpack", [](nb::args args) { return repack("unpack", nullptr, args); });
             nb::sig("def unpack(*args: *tuple[View | dr.AnyArray, ...]) -> tuple[View, ...]");

    coop.def("matvec", &matvec, "A"_a, "x"_a, "b"_a = nb::none(),
             "transpose"_a = false,
             nb::sig("def matvec(A: View, x: drjit.coop.Vector[T], b: typing.Optional[View] = "
                     "None, /, transpose: bool = False) -> drjit.coop.Vector[T]"),
             doc_coop_matvec);

    m.def("fma", &coop_vec_ternary_op<JitOp::Fma>);
    m.def("minimum", &coop_vec_binary_op<JitOp::Min>);
    m.def("maximum", &coop_vec_binary_op<JitOp::Max>);
    m.def("log2", &coop_vec_unary_op<JitOp::Log2>);
    m.def("exp2", &coop_vec_unary_op<JitOp::Exp2>);
    m.def("tanh", &coop_vec_unary_op<JitOp::Tanh>);
    m.def("step", &coop_vec_step, doc_step);
}
