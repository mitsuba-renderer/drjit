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
             "wrap_mode"_a = ek::WrapMode::Clamp)
        .def(py::init<const typename Tex::TensorXf &, bool, ek::FilterMode,
                      ek::WrapMode>(),
             "tensor"_a, "migrate"_a = true,
             "filter_mode"_a = ek::FilterMode::Linear,
             "wrap_mode"_a = ek::WrapMode::Clamp)
        .def("set_value", &Tex::set_value, "value"_a)
        .def("set_tensor", &Tex::set_tensor, "tensor"_a)
        .def("value", &Tex::value, py::return_value_policy::reference_internal)
        .def("tensor", &Tex::tensor, py::return_value_policy::reference_internal)
        .def("filter_mode", &Tex::filter_mode)
        .def("wrap_mode", &Tex::wrap_mode)
        .def("eval_cuda",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval_cuda(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_enoki",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval_enoki(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_fetch_cuda",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    constexpr size_t ResultSize = 1 << Dimension;
                    size_t channels = texture.shape()[Dimension];

                    ek::Array<Type *, ResultSize> result_ptrs;
                    std::array<std::vector<Type>, ResultSize> result;
                    for (size_t i = 0; i < ResultSize; ++i) {
                        std::vector<Type> result_i(channels);
                        result[i] = std::move(result_i);
                        result_ptrs[i] = result[i].data();
                    }
                    texture.eval_fetch_cuda(pos, result_ptrs, active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_fetch_enoki",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    constexpr size_t ResultSize = 1 << Dimension;
                    size_t channels = texture.shape()[Dimension];

                    ek::Array<Type *, ResultSize> result_ptrs;
                    std::array<std::vector<Type>, ResultSize> result;
                    for (size_t i = 0; i < ResultSize; ++i) {
                        std::vector<Type> result_i(channels);
                        result[i] = std::move(result_i);
                        result_ptrs[i] = result[i].data();
                    }
                    texture.eval_fetch_enoki(pos, result_ptrs, active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_fetch",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    constexpr size_t ResultSize = 1 << Dimension;
                    size_t channels = texture.shape()[Dimension];

                    ek::Array<Type *, ResultSize> result_ptrs;
                    std::array<std::vector<Type>, ResultSize> result;
                    for (size_t i = 0; i < ResultSize; ++i) {
                        std::vector<Type> result_i(channels);
                        result[i] = std::move(result_i);
                        result_ptrs[i] = result[i].data();
                    }
                    texture.eval_fetch(pos, result_ptrs, active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_cubic",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active, bool force_enoki) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval_cubic(pos, result.data(), active, force_enoki);

                    return result;
                }, "pos"_a, "active"_a = true, "force_enoki"_a = false)
        .def("eval_cubic_grad",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<ek::Array<Type, Dimension>> result(channels);
                    texture.eval_cubic_grad(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_cubic_hessian",
                [](const Tex &texture, const ek::Array<Type, Dimension> &pos,
                   const ek::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<ek::Array<Type, Dimension>> gradient(channels);
                    std::vector<ek::Matrix<Type, Dimension>> hessian(channels);
                    texture.eval_cubic_hessian(pos, gradient.data(),
                                               hessian.data(), active);

                    return std::make_tuple(gradient, hessian);
                }, "pos"_a, "active"_a = true);

    tex.attr("IsTexture") = true;
}

template <typename Type>
void bind_texture_all(py::module &m) {
    bind_texture<Type, 1>(m, "Texture1f");
    bind_texture<Type, 2>(m, "Texture2f");
    bind_texture<Type, 3>(m, "Texture3f");
}
