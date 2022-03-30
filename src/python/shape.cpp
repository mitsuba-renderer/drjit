#include "python.h"

Py_ssize_t len(PyObject *o) noexcept {
    PyTypeObject *tp = Py_TYPE(o);
    const supp &s = nb::type_supplement<supp>(tp);
    Py_ssize_t length = s.meta.shape[0];

    if (length == 0xFF)
        length = (Py_ssize_t) s.ops.len(nb::inst_ptr<void>(o));

    return length;
}

bool shape_impl(nb::handle h, int i, Py_ssize_t *shape) noexcept {
    if (i >= 4)
        nb::detail::fail("drjit.shape(): internal error!");
    nb::handle tp = h.type();

    const supp &s = nb::type_supplement<supp>(tp);
    Py_ssize_t size = s.meta.shape[0], cur = shape[i];

    if (size == 0xFF)
        size = (Py_ssize_t) s.ops.len(nb::inst_ptr<void>(h));

    Py_ssize_t max_size = size > cur ? size : cur;
    if (max_size != size && size != 1)
        return false;

    shape[i] = max_size;

    if (s.meta.shape[1]) {
        auto sq_item = ((PyTypeObject *) tp.ptr())->tp_as_sequence->sq_item;

        for (Py_ssize_t j = 0; j < size; ++j) {
            PyObject *o = sq_item(h.ptr(), j);

            if (!shape_impl(o, i + 1, shape)) {
                Py_DECREF(o);
                return false;
            }

            Py_DECREF(o);
        }
    }

    return true;
}

nb::object shape(nb::handle_of<dr::ArrayBase> h) noexcept {
    Py_ssize_t shape[4] { -1, -1, -1, -1 };
    if (!shape_impl(h, 0, shape))
        return nb::none();

    Py_ssize_t ndim = 0;
    for (Py_ssize_t i = 0; i < 4; ++i) {
        if (shape[i] == -1)
            break;
        ndim++;
    }

    PyObject *result = PyTuple_New(ndim);
    for (Py_ssize_t i = 0; i < ndim; ++i)
        PyTuple_SET_ITEM(result, i, PyLong_FromSize_t(shape[i]));

    return nb::steal(result);
}
