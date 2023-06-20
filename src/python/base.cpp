#include "base.h"
#include "meta.h"
#include "repr.h"
#include "shape.h"
#include <nanobind/stl/string.h>

nb::handle array_base;

static const char *op_names[(int) ArrayOp::Count] = {
    "__add__",
    "__sub__",
    "__mul__"
};

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

template <typename T1, typename T2> using first_t = T1;

template <typename Func, typename... Args, size_t... Is>
static PyObject *apply(ArrayOp op, Func func, std::index_sequence<Is...>,
                       Args... args) noexcept {
    nb::object o[] = { nb::borrow(args)... };
    nb::handle tp = o[0].type();
    constexpr size_t N = sizeof...(Args);

    try {
        // All arguments must first be promoted to the same type
        if (!(o[Is].type().is(tp) && ...)) {
            promote(o, sizeof...(Args), false);
            tp = o[0].type();
        }

        const ArraySupplement &s = supp(tp);
        void *impl = s[op];

        if (impl == DRJIT_OP_NOT_IMPLEMENTED)
            return nb::not_implemented().release().ptr();

        nb::object result = nb::inst_alloc(tp);

        drjit::ArrayBase *p[N+1] = {
            nb::inst_ptr<dr::ArrayBase>(args)...,
            nb::inst_ptr<dr::ArrayBase>(result)
        };

        if (impl != DRJIT_OP_DEFAULT) {
            using Impl = void (*)(first_t<const dr::ArrayBase *, Args>..., dr::ArrayBase *);

            ((Impl) impl)(p[Is]..., p[N]);
            nb::inst_mark_ready(result);
        } else {
            nb::inst_zero(result);

            Py_ssize_t l[N + 1], i[N] { };
            if (s.shape[0] != DRJIT_DYNAMIC) {
                ((l[Is] = s.shape[0]), ...);
                l[N] = s.shape[0];
            } else {
                ((l[Is] = s.len(p[Is])), ...);
                l[N] = std::max(l[Is]...);

                if (((l[Is] != l[N] && l[Is] != 1) || ...))
                    raise_incompatible_size_error(l, N);

                s.init(p[N], l[N]);
            }

            using PyImpl = PyObject *(*)(first_t<PyObject *, Args>...);
            PyImpl py_impl;
            if constexpr (std::is_same_v<Func, int>)
                py_impl = (PyImpl) PyType_GetSlot((PyTypeObject *) tp.ptr(), func);
            else
                py_impl = func;

            const ArraySupplement::Item item = s.item;
            const ArraySupplement::SetItem set_item = s.set_item;

            for (Py_ssize_t j = 0; j < l[N]; ++j) {
                nb::object v[] = { nb::steal(item(o[Is].ptr(), i[Is]))... };

                if (!(v[Is].is_valid() && ...)) {
                    result.reset();
                    break;
                }

                nb::object vr = nb::steal(py_impl(v[Is].ptr()...));
                if (!vr.is_valid() || set_item(result.ptr(), j, vr.ptr())) {
                    result.reset();
                    break;
                }

                ((i[Is] += (l[Is] == 1 ? 0 : 1)), ...);
            }
        }

        return result.release().ptr();
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        PyErr_Format(PyExc_RuntimeError, "%U.%s(): %s!", tp_name.ptr(),
                     op_names[(int) op], e.what());
        return nullptr;
    }
}

static PyObject *nb_add(PyObject *h0, PyObject *h1) noexcept {
    return apply(ArrayOp::Add, Py_nb_add, std::make_index_sequence<2>(), h0, h1);
}

static PyObject *nb_subtract(PyObject *h0, PyObject *h1) noexcept {
    return apply(ArrayOp::Sub, Py_nb_subtract, std::make_index_sequence<2>(), h0, h1);
}

static PyObject *nb_multiply(PyObject *h0, PyObject *h1) noexcept {
    return apply(ArrayOp::Mul, Py_nb_multiply, std::make_index_sequence<2>(), h0, h1);
}

template <int Index> nb::object xyzw_getter(nb::handle_t<dr::ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY((!s.is_vector && !s.is_quaternion) || s.ndim == 0 ||
                    s.shape[0] == DRJIT_DYNAMIC || Index >= s.shape[0])) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%c' component!", name.c_str(),
                          "xyzw"[Index]);
    }

    return nb::steal(s.item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index> void xyzw_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY((!s.is_vector && !s.is_quaternion) || s.ndim == 0 ||
                    s.shape[0] == DRJIT_DYNAMIC || Index >= s.shape[0])) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%c' component!", name.c_str(),
                          "xyzw"[Index]);
    }

    if (s.set_item(h.ptr(), (Py_ssize_t) Index, value.ptr()))
        nb::detail::raise_python_error();
}

template <int Index> nb::object complex_getter(nb::handle_t<dr::ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%s' component!", name.c_str(),
                          Index == 0 ? "real" : "imaginary");
    }

    return nb::steal(s.item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index> void complex_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%s' component!", name.c_str(),
                          Index == 0 ? "real" : "imaginary");
    }

    if (s.set_item(h.ptr(), (Py_ssize_t) Index, value.ptr()))
        nb::detail::raise_python_error();
}


#define DR_ARRAY_SLOT(name) { Py_##name, (void *) name }

static PyType_Slot array_base_slots[] = {
    DR_ARRAY_SLOT(nb_add),
    DR_ARRAY_SLOT(nb_subtract),
    DR_ARRAY_SLOT(nb_multiply),
    DR_ARRAY_SLOT(sq_length),
    DR_ARRAY_SLOT(tp_repr),
    { 0, nullptr }
};


void export_base(nb::module_ &m) {
    nb::class_<dr::ArrayBase> ab(m, "ArrayBase",
                                 nb::type_slots(array_base_slots),
                                 nb::supplement<ArraySupplement>());

    ab.def_prop_ro_static("__meta__", [](nb::handle h) {
        return meta_str(nb::type_supplement<ArraySupplement>(h));
    });

    ab.def_prop_ro("shape", &shape, nb::raw_doc(doc_ArrayBase_shape));
    ab.def_prop_rw("x", xyzw_getter<0>, xyzw_setter<0>, nb::raw_doc(doc_ArrayBase_x));
    ab.def_prop_rw("y", xyzw_getter<1>, xyzw_setter<1>, nb::raw_doc(doc_ArrayBase_y));
    ab.def_prop_rw("z", xyzw_getter<2>, xyzw_setter<2>, nb::raw_doc(doc_ArrayBase_z));
    ab.def_prop_rw("w", xyzw_getter<3>, xyzw_setter<3>, nb::raw_doc(doc_ArrayBase_w));
    ab.def_prop_rw("real", complex_getter<0>, complex_setter<0>, nb::raw_doc(doc_ArrayBase_real));
    ab.def_prop_rw("imag", complex_getter<1>, complex_setter<1>, nb::raw_doc(doc_ArrayBase_imag));

    array_base = ab;
}
