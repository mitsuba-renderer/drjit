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

static nb::object graphviz(bool ad, bool as_string) {
    nb::str string;

    if (ad)
        string = nb::str(ad_var_graphviz());
    else
        string = nb::str(jit_var_graphviz());

    if (as_string)
        return std::move(string);

    try {
        return nb::module_::import_("graphviz").attr("Source")(string);
    } catch (...) {
        throw nb::type_error(
            "drjit.graphviz(): The 'graphviz' Python package not available! "
            "Install via 'python -m pip install graphviz'. Alternatively, "
            "you can call ``drjit.graphviz(as_string=True)`` to obtain a "
            "string representation..");
    }
}

static nb::object whos(bool ad, bool as_string) {
    nb::str string;

    if (ad)
        string = nb::str(ad_var_whos());
    else
        string = nb::str(jit_var_whos());

    if (as_string) {
        return std::move(string);
    } else {
        nb::print(string);
        return nb::none();
    }
}

void set_label(nb::handle h, nb::str label) {
    nb::handle tp = h.type();
    bool is_drjit = is_drjit_type(tp);

    if (is_drjit) {
        const ArraySupplement &s = supp(tp);
        if ((JitBackend) s.backend == JitBackend::None)
            return;
        if (s.ndim == 1) {
            uint64_t new_index =
                ad_var_set_label(s.index(inst_ptr(h)), label.c_str());
            nb::object tmp = nb::inst_alloc(tp);
            s.init_index(new_index, inst_ptr(tmp));
            nb::inst_mark_ready(tmp);
            ad_var_dec_ref(new_index);

            nb::inst_replace_move(h, tmp);
            return;
        }
    }

    if (is_drjit || tp.is(&PyList_Type) || tp.is(&PyTuple_Type)) {
        size_t size = nb::len(h);
        for (size_t i = 0; i < size; ++i)
            set_label(h[i], nb::str("{}_{}").format(label, nb::int_(i)));
    } else if (tp.is(&PyDict_Type)) {
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
    for (auto [k, v] : kwargs) {
        if (v.type().is(&PyUnicode_Type))
            nb::detail::raise("drjit.set_label(): You passed a ``str``-valued "
                              "keyword argument where an array was expected. "
                              "To use the kwargs-style interface, call this "
                              "function as follows: ``set_label(x=x, y=y)``");
        set_label(v, nb::borrow<nb::str>(k));
    }
}

static nb::object label(nb::handle h) {
    if (is_drjit_array(h)) {
        const ArraySupplement &s = supp(h.type());
        if ((JitBackend) s.backend != JitBackend::None) {
            const char *str = jit_var_label((uint32_t) s.index(inst_ptr(h)));
            if (str)
                return nb::str(str);
        }
    }

    return nb::none();
}

void export_inspect(nb::module_ &m) {
    m.def("graphviz", [](bool as_string) { return graphviz(false, as_string); },
          "as_string"_a = false, doc_graphviz)
     .def("graphviz_ad",
          [](bool as_string) { return graphviz(true, as_string); },
          "as_string"_a = false, doc_graphviz_ad)
     .def("whos", [](bool as_string) { return whos(false, as_string); },
          "as_string"_a = false, doc_whos)
     .def("whos_ad", [](bool as_string) { return whos(true, as_string); },
          "as_string"_a = false, doc_whos_ad)
     .def("label", &label, doc_label)
     .def("set_label", &set_label, doc_set_label)
     .def("set_label", &set_label_2);
}
