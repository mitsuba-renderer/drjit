#include "python.h"
#include "docstr.h"

extern void bind_traits(nb::module_ m) {
    m.attr("Dynamic") = (Py_ssize_t) -1;

    m.def("is_array_v", [](nb::handle h) -> bool {
        return is_drjit_type(h.is_type() ? h : h.type());
    }, nb::raw_doc(doc_is_array_v));

    m.def("array_size_v", [](nb::handle h) -> Py_ssize_t {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            Py_ssize_t shape = nb::type_supplement<supp>(tp).meta.shape[0];
            if (shape == 0xFF)
                shape = -1;
            return shape;
        } else {
            return 1;
        }
    }, nb::raw_doc(doc_array_size_v));

    m.def("array_depth_v", [](nb::handle h) -> size_t {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            const uint8_t *shape = nb::type_supplement<supp>(tp).meta.shape;
            int depth = 1;
            for (int i = 1; i < 4; ++i) {
                if (!shape[i])
                    break;
                depth++;
            }
            return depth;
        } else {
            return 0;
        }
    }, nb::raw_doc(doc_array_depth_v));
}
