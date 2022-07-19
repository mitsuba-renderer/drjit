#pragma once

#include "common.h"
#include <drjit/texture.h>

template <typename Type, size_t Dimension>
void bind_texture(py::module &m, const char *name) {
    using Tex = dr::Texture<Type, Dimension>;

    auto tex = py::class_<Tex>(m, name)
        .def(py::init([](const std::array<size_t, Dimension> &shape,
                         size_t channels, bool use_accel,
                         dr::FilterMode filter_mode, dr::WrapMode wrap_mode) {
                 return new Tex(shape.data(), channels, use_accel, filter_mode, wrap_mode);
             }),
             "shape"_a, "channels"_a, "use_accel"_a = true,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp)
        .def(py::init<const typename Tex::TensorXf &, bool, bool, dr::FilterMode,
                      dr::WrapMode>(),
             "tensor"_a, "use_accel"_a = true, "migrate"_a = true,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp)
        .def("set_value",  &Tex::set_value,  "value"_a,  "migrate"_a = false)
        .def("set_tensor", &Tex::set_tensor, "tensor"_a, "migrate"_a = false)
        .def("value", &Tex::value, py::return_value_policy::reference_internal)
        .def("tensor",
             py::overload_cast<>(&Tex::tensor, py::const_),
             py::return_value_policy::reference_internal)
        .def("filter_mode", &Tex::filter_mode)
        .def("wrap_mode", &Tex::wrap_mode)
        .def("use_accel", &Tex::use_accel)
        .def("migrated", &Tex::migrated)
        .def_property_readonly("shape", [](const Tex &t) {
            PyObject *shape = PyTuple_New(t.ndim());
            for (size_t i = 0; i < t.ndim(); ++i)
                PyTuple_SET_ITEM(shape, i, PyLong_FromLong((long) t.shape()[i]));
            return py::reinterpret_steal<py::tuple>(shape);
        })
        .def("eval_cuda",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval_cuda(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_nonaccel",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval_nonaccel(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval(pos, result.data(), active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_fetch_cuda",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    constexpr size_t ResultSize = 1 << Dimension;
                    size_t channels = texture.shape()[Dimension];

                    dr::Array<Type *, ResultSize> result_ptrs;
                    std::array<std::vector<Type>, ResultSize> result;
                    for (size_t i = 0; i < ResultSize; ++i) {
                        std::vector<Type> result_i(channels);
                        result[i] = std::move(result_i);
                        result_ptrs[i] = result[i].data();
                    }
                    texture.eval_fetch_cuda(pos, result_ptrs, active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_fetch_drjit",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    constexpr size_t ResultSize = 1 << Dimension;
                    size_t channels = texture.shape()[Dimension];

                    dr::Array<Type *, ResultSize> result_ptrs;
                    std::array<std::vector<Type>, ResultSize> result;
                    for (size_t i = 0; i < ResultSize; ++i) {
                        std::vector<Type> result_i(channels);
                        result[i] = std::move(result_i);
                        result_ptrs[i] = result[i].data();
                    }
                    texture.eval_fetch_nonaccel(pos, result_ptrs, active);

                    return result;
                }, "pos"_a, "active"_a = true)
        .def("eval_fetch",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    constexpr size_t ResultSize = 1 << Dimension;
                    size_t channels = texture.shape()[Dimension];

                    dr::Array<Type *, ResultSize> result_ptrs;
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
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active, bool force_drjit) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> result(channels);
                    texture.eval_cubic(pos, result.data(), active, force_drjit);

                    return result;
                }, "pos"_a, "active"_a = true, "force_drjit"_a = false)
        .def("eval_cubic_grad",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];
                    std::vector<Type> value(channels);
                    std::vector<dr::Array<Type, Dimension>> gradient(channels);
                    texture.eval_cubic_grad(pos, value.data(), gradient.data(), active);

                    return std::make_tuple(value, gradient);
                }, "pos"_a, "active"_a = true)
        .def("eval_cubic_hessian",
                [](const Tex &texture, const dr::Array<Type, Dimension> &pos,
                   const dr::mask_t<Type> active) {
                    size_t channels = texture.shape()[Dimension];

                    std::vector<Type> value(channels);
                    std::vector<dr::Array<Type, Dimension>> gradient(channels);
                    std::vector<dr::Matrix<Type, Dimension>> hessian(channels);
                    texture.eval_cubic_hessian(pos, value.data(), gradient.data(),
                                               hessian.data(), active);

                    return std::make_tuple(value, gradient, hessian);
                }, "pos"_a, "active"_a = true);

    tex.attr("IsTexture") = true;
}

template <typename Type>
void bind_texture_all(py::module &m) {
    using Type64 = dr::float64_array_t<Type>;
    bind_texture<Type, 1>(m, "Texture1f");
    bind_texture<Type, 2>(m, "Texture2f");
    bind_texture<Type, 3>(m, "Texture3f");
    bind_texture<Type64, 1>(m, "Texture1f64");
    bind_texture<Type64, 2>(m, "Texture2f64");
    bind_texture<Type64, 3>(m, "Texture3f64");
}
