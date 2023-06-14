#include "base.h"
#include "meta.h"
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

        const ArraySupplement &supp = nb::type_supplement<ArraySupplement>(o[0]);
        void *impl = supp.ops[(int) op];

        if (impl == DRJIT_OP_NOT_IMPLEMENTED)
            return nb::not_implemented().release().ptr();

        nb::object result = nb::inst_alloc(tp);

        drjit::ArrayBase *p[N+1] = {
            nb::inst_ptr<dr::ArrayBase>(args)...,
            nb::inst_ptr<dr::ArrayBase>(result)
        };

        if (impl != DRJIT_OP_DEFAULT) {
            using Impl = void (*)(first_t<const dr::ArrayBase *, Args>..., dr::ArrayBase *);
            ((Impl) impl)(p[N], p[Is]...);
            nb::inst_mark_ready(result);
        } else {
            nb::inst_zero(result);

            Py_ssize_t l[N + 1], i[N] { };
            if (supp.shape[0] != DRJIT_DYNAMIC) {
                ((l[Is] = supp.shape[0]), ...);
                l[N] = supp.shape[0];
            } else {
                ((l[Is] = supp.len(p[Is])), ...);
                l[N] = std::max(l[Is]...);

                if (((l[Is] != l[N] && l[Is] != 1) || ...))
                    raise_incompatible_size_error(l, N);

                supp.init(p[N], l[N]);
            }

            using PyImpl = PyObject *(*)(first_t<PyObject *, Args>...);
            PyImpl py_impl;
            if constexpr (std::is_same_v<Func, int>)
                py_impl = (PyImpl) PyType_GetSlot((PyTypeObject *) tp.ptr(), func);
            else
                py_impl = func;

            for (Py_ssize_t j = 0; j < l[N]; ++j) {
                nb::object v[] = { nb::steal(supp.item(o[Is].ptr(), i[Is]))... };

                if (!(v[Is].is_valid() && ...)) {
                    result.reset();
                    break;
                }

                nb::object vr = nb::steal(py_impl(v[Is].ptr()...));
                if (!vr.is_valid() || supp.set_item(result.ptr(), j, vr.ptr())) {
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


#define DR_ARRAY_SLOT(name) { Py_##name, (void *) name }

static PyType_Slot array_base_slots[] = {
    DR_ARRAY_SLOT(nb_add),
    DR_ARRAY_SLOT(nb_subtract),
    DR_ARRAY_SLOT(nb_multiply),
    { 0, nullptr }
};


void export_base(nb::module_ &m) {
    nb::class_<dr::ArrayBase> ab(m, "ArrayBase",
                                 nb::type_slots(array_base_slots),
                                 nb::supplement<ArraySupplement>());

    ab.def_prop_ro_static("__meta__", [](nb::handle h) {
        return meta_str(nb::type_supplement<ArraySupplement>(h));
    });
    array_base = ab;
}
