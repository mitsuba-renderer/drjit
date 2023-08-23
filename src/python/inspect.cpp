/*
    inspect.cpp -- operations to label Jit/AD graph nodes and
    visualize their structure using GraphViz.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "inspect.h"
#include "base.h"
#include <string>

static nb::object graphviz(bool as_str = false) {
    nb::str string = nb::str(jit_var_graphviz());

    if (as_str)
        return std::move(string);

    try {
        return nb::module_::import_("graphviz").attr("Source")(string);
    } catch (...) {
        throw nb::type_error(
            "drjit.graphviz(): The 'graphviz' Python package not available! "
            "Install via 'python -m pip install graphviz'. Alternatively, "
            "you can call ``drjit.graphviz(as_str=True)`` to obtain a "
            "string representation..");
    }
}

void set_label(nb::handle h, nb::str label) {
    nb::handle tp = h.type();

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if ((JitBackend) s.backend == JitBackend::None)
            return;
        if (s.ndim == 1) {
            uint64_t index = s.index(inst_ptr(h));
            uint64_t new_index = ad_var_set_label(index, label.c_str());
            nb::object tmp = nb::inst_alloc(tp);
            s.init_index(new_index, inst_ptr(tmp));
            nb::inst_mark_ready(tmp);
            nb::inst_replace_move(h, tmp);
            ad_var_dec_ref(new_index);
            return;
        }
    }

    if (nb::isinstance<nb::sequence>(h)) {
        size_t size = nb::len(h);
        for (size_t i = 0; i < size; ++i)
            set_label(h[i], nb::str("{}_{}").format(label, nb::int_(i)));
    } else if (nb::isinstance<nb::dict>(h)) {
        for (auto [k, v] : nb::borrow<nb::dict>(h))
            set_label(v, nb::str("{}_{}").format(label, k));
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            for (auto [k, v] : nb::borrow<nb::dict>(dstruct))
                set_label(nb::getattr(h, k), nb::str("{}_{}").format(label, k));
        }
    }
}

void set_label_2(nb::kwargs kwargs) {
    for (auto [k, v] : kwargs)
        set_label(v, nb::str(k));
}

static nb::object label(nb::handle h) {
    if (is_drjit_array(h)) {
        const ArraySupplement &s = supp(h.type());
        if ((JitBackend) s.backend != JitBackend::None) {
            const char *str = ad_var_label(s.index(inst_ptr(h)));
            if (str)
                return nb::str(str);
        }
    }

    return nb::none();
}

void export_inspect(nb::module_ &m) {
    m.def("graphviz", &graphviz, "as_str"_a = false, doc_graphviz)
     .def("label", &label, doc_label)
     .def("set_label", &set_label, doc_set_label)
     .def("set_label", &set_label_2);
}
