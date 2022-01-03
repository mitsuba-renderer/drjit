#pragma once

#include "common.h"
#include <enoki/texture.h>

template <typename Type, size_t Dimension>
void bind_texture(py::module &m, const char *name) {
    using Tex = ek::Texture<Type, Dimension>;

    auto tex = py::class_<Tex>(m, name)
        .def(py::init([](const std::array<size_t, Dimension> &shape,
                         size_t channels, bool migrate,
                         ek::FilterMode filter_mode, ek::WrapMode wrap_mode) {
                 return new Tex(shape.data(), channels, migrate, filter_mode,
                                wrap_mode);
             }),
             "shape"_a, "channels"_a, "migrate"_a = true,
             "filter_mode"_a = ek::FilterMode::Linear,
             "wrap_mode"_a   = ek::WrapMode::Repeat)
        .def(py::init<const typename Tex::TensorXf &, bool, ek::FilterMode,
                      ek::WrapMode>(),
             "tensor"_a, "migrate"_a = true,
             "filter_mode"_a = ek::FilterMode::Linear,
             "wrap_mode"_a   = ek::WrapMode::Repeat)
        .def("set_value", &Tex::set_value, "value"_a)
        .def("set_tensor", &Tex::set_tensor, "tensor"_a)
        .def("value", &Tex::value, py::return_value_policy::reference_internal)
        .def("tensor", &Tex::tensor,
             py::return_value_policy::reference_internal)
        .def("filter_mode", &Tex::filter_mode)
        .def("wrap_mode", &Tex::wrap_mode)
        .def("eval_cuda", &Tex::eval_cuda, "pos"_a, "active"_a = true)
        .def("eval_enoki", &Tex::eval_enoki, "pos"_a, "active"_a = true)
        .def("eval_cubic", &Tex::eval_cubic, "pos"_a, "active"_a = true, "force_enoki"_a = false)
        .def("eval_cubic_grad", &Tex::eval_cubic_grad, "pos"_a, "active"_a = true)
        .def("eval", &Tex::eval, "pos"_a, "active"_a = true);

    tex.attr("IsTexture") = true;
}

template <typename Type> void bind_texture_all(py::module &m) {
    bind_texture<Type, 1>(m, "Texture1f");
    bind_texture<Type, 2>(m, "Texture2f");
    bind_texture<Type, 3>(m, "Texture3f");
}
