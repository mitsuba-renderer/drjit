#include "python.h"

extern void bind_ops(nb::module_ m) {
    m.def("all", [](nb::handle h) -> nb::object {
        PyTypeObject *tp = (PyTypeObject *) h.type().ptr();

        if (tp == &PyBool_Type)
            return borrow(h);

        if (is_drjit_array(h)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce_mask op = s.ops.op_all;
            if (!op)
                throw nb::type_error(
                    "drjit.all(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                PyObject *result = tp->tp_new(tp, nullptr, nullptr);
                if (!result)
                    nb::detail::raise_python_error();
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::detail::nb_inst_ready(result);
                return nb::steal(result);
            }
        }

        nb::object result = nb::borrow(Py_True);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = result & h2;
        }

        return result;
    });

    m.def("any", [](nb::handle h) -> nb::object {
        PyTypeObject *tp = (PyTypeObject *) h.type().ptr();

        if (tp == &PyBool_Type)
            return borrow(h);

        if (is_drjit_array(h)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce_mask op = s.ops.op_any;
            if (!op)
                throw nb::type_error(
                    "drjit.any(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                PyObject *result = tp->tp_new(tp, nullptr, nullptr);
                if (!result)
                    nb::detail::raise_python_error();
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::detail::nb_inst_ready(result);
                return nb::steal(result);
            }
        }

        nb::object result = nb::borrow(Py_False);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = result | h2;
        }

        return result;
    });
}

