#pragma once

#include "common.h"
#include <drjit/texture.h>
#include <nanobind/stl/optional.h>

template <typename Type, size_t Dimension>
void bind_texture(nb::module_ &m, const char *name) {
    using Tex = dr::Texture<Type, Dimension>;
    using Float16 = dr::replace_scalar_t<Type, dr::half>;
    using Float32 = dr::replace_scalar_t<Type, float>;
    using Float64 = dr::replace_scalar_t<Type, double>;

    auto tex = nb::class_<Tex>(m, name)
        .def("__init__", [](Tex* t, const dr::vector<size_t>& shape,
                         size_t channels, bool use_accel,
                         dr::FilterMode filter_mode, dr::WrapMode wrap_mode,
                         bool writable) {
                 new (t) Tex(shape.data(), channels, use_accel, filter_mode,
                             wrap_mode, writable); },
             "shape"_a, "channels"_a, "use_accel"_a = true,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp,
             "writable"_a = false,
             doc_Texture_init)
        .def(nb::init<const typename Tex::TensorXf &, bool, bool, dr::FilterMode,
                      dr::WrapMode>(),
             "tensor"_a, "use_accel"_a = true, "migrate"_a = true,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp,
             doc_Texture_init_tensor)
        .def("set_value", &Tex::template set_value<const typename Tex::Storage &>, "value"_a, "migrate"_a = false, doc_Texture_set_value)
        .def("set_tensor", &Tex::template set_tensor<const typename Tex::TensorXf &>, "tensor"_a,  "migrate"_a = false, doc_Texture_set_tensor)
        .def("update_inplace", &Tex::update_inplace, "migrate"_a = false, doc_Texture_update_inplace)
        .def("value", &Tex::value, nb::rv_policy::reference_internal, doc_Texture_value)
        .def("tensor",
             nb::overload_cast<>(&Tex::tensor, nb::const_),
             nb::rv_policy::reference_internal, doc_Texture_tensor)
        .def("filter_mode", &Tex::filter_mode, doc_Texture_filter_mode)
        .def("wrap_mode", &Tex::wrap_mode, doc_Texture_wrap_mode)
        .def("use_accel", &Tex::use_accel, doc_Texture_use_accel)
        .def("writable", &Tex::writable, doc_Texture_writable)
        .def_static("from_native_handle", &Tex::from_native_handle, "handle"_a,
             "writable"_a = false,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp,
             doc_Texture_from_native_handle)
        .def("map", &Tex::map, doc_Texture_map)
        .def("unmap", &Tex::unmap, doc_Texture_unmap)
        .def("native_handle", &Tex::native_handle, "sub_index"_a = 0,
             doc_Texture_native_handle)
        .def("migrated", &Tex::migrated, doc_Texture_migrated)
        .def_prop_ro("shape", [](const Tex &t) {
            PyObject *shape = PyTuple_New(t.ndim());
            for (size_t i = 0; i < t.ndim(); ++i)
                PyTuple_SetItem(shape, i, PyLong_FromLong((long) t.shape()[i]));
            return nb::steal<nb::tuple>(shape);
        }, doc_Texture_shape)
        #define def_tex_eval(T)                                                \
            def("eval",                                                        \
                [](const Tex &texture, const dr::Array<T, Dimension> &pos,     \
                   const std::optional<dr::mask_t<T>> active_) {               \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    size_t channels = texture.shape()[Dimension];              \
                    dr::vector<T> result(channels);                            \
                    texture.eval(pos, result.data(), active);                  \
                                                                               \
                    return result;                                             \
                }, "pos"_a, "active"_a.sig("Bool(True)") = nb::none(),         \
                doc_Texture_eval)
        .def_tex_eval(Float32)
        .def_tex_eval(Float16)
        .def_tex_eval(Float64)
        #undef def_tex_eval
        #define def_tex_write(T)                                               \
            def("write",                                                       \
                [](Tex &texture,                                               \
                   const dr::Array<dr::uint32_array_t<T>, Dimension> &pos,     \
                   const dr::vector<T> &value,                                 \
                   const std::optional<dr::mask_t<T>> active_) {               \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    size_t channels = texture.shape()[Dimension];              \
                    if (value.size() != channels)                              \
                        nb::raise("Texture.write(): expected %zu channel "     \
                                  "values, got %zu.", channels, value.size()); \
                    if constexpr (Tex::HasGPUTexture)                          \
                        texture.write(pos, value.data(), active);              \
                    else                                                       \
                        nb::raise("Texture.write(): requires a CUDA or Metal " \
                                  "float16/float32 texture.");                 \
                }, "pos"_a, "value"_a,                                         \
                "active"_a.sig("Bool(True)") = nb::none(),                     \
                doc_Texture_write)
        .def_tex_write(Float32)
        .def_tex_write(Float16)
        .def_tex_write(Float64)
        #undef def_tex_write
        #define def_tex_eval_fetch(T)                                          \
            def("eval_fetch",                                                  \
                [](const Tex &texture, const dr::Array<T, Dimension> &pos,     \
                   const std::optional<dr::mask_t<T>> active_) {               \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    constexpr size_t ResultSize = 1 << Dimension;              \
                    size_t channels = texture.shape()[Dimension];              \
                                                                               \
                    dr::Array<T *, ResultSize> result_ptrs;                    \
                    dr::vector<dr::vector<T>> result(ResultSize);              \
                    for (size_t i = 0; i < ResultSize; ++i) {                  \
                        result[i].resize(channels);                            \
                        result_ptrs[i] = result[i].data();                     \
                    }                                                          \
                    texture.eval_fetch(pos, result_ptrs, active);              \
                                                                               \
                    return result;                                             \
                }, "pos"_a, "active"_a.sig("Bool(True)") = nb::none(),         \
                doc_Texture_eval_fetch)
        .def_tex_eval_fetch(Float32)
        .def_tex_eval_fetch(Float16)
        .def_tex_eval_fetch(Float64)
        #undef def_tex_eval_fetch
        #define def_tex_eval_cubic(T)                                          \
            def("eval_cubic",                                                  \
                [](const Tex &texture, const dr::Array<T, Dimension> &pos,     \
                   const std::optional<dr::mask_t<T>> active_,                 \
                   bool force_nonaccel) {                                      \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    size_t channels = texture.shape()[Dimension];              \
                    dr::vector<T> result(channels);                            \
                    texture.eval_cubic(                                        \
                        pos, result.data(), active, force_nonaccel);           \
                                                                               \
                    return result;                                             \
                }, "pos"_a, "active"_a.sig("Bool(True)") = nb::none(),         \
                "force_nonaccel"_a = false, doc_Texture_eval_cubic)
        .def_tex_eval_cubic(Float32)
        .def_tex_eval_cubic(Float16)
        .def_tex_eval_cubic(Float64)
        #undef def_tex_eval_cubic
        #define def_tex_eval_cubic_grad(T)                                     \
            def("eval_cubic_grad",                                             \
                [](const Tex &texture, const dr::Array<T, Dimension> &pos,     \
                   const std::optional<dr::mask_t<T>> active_) {               \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    size_t channels = texture.shape()[Dimension];              \
                    dr::vector<T> value(channels);                             \
                    dr::vector<dr::Array<T, Dimension>> gradient(channels);    \
                    texture.eval_cubic_grad(                                   \
                        pos, value.data(), gradient.data(), active);           \
                                                                               \
                    return nb::make_tuple(value, gradient);                    \
                }, "pos"_a, "active"_a.sig("Bool(True)") = nb::none(),         \
                doc_Texture_eval_cubic_grad)
        .def_tex_eval_cubic_grad(Float32)
        .def_tex_eval_cubic_grad(Float16)
        .def_tex_eval_cubic_grad(Float64)
        #undef def_tex_eval_cubic_grad
        #define def_tex_eval_cubic_hessian(T)                                  \
            def("eval_cubic_hessian",                                          \
                [](const Tex &texture, const dr::Array<T, Dimension> &pos,     \
                   const std::optional<dr::mask_t<T>> active_) {               \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    size_t channels = texture.shape()[Dimension];              \
                    dr::vector<T> value(channels);                             \
                    dr::vector<dr::Array<T, Dimension>> gradient(channels);    \
                    dr::vector<dr::Matrix<T, Dimension>> hessian(channels);    \
                    texture.eval_cubic_hessian(pos, value.data(),              \
                        gradient.data(), hessian.data(), active);              \
                                                                               \
                    return nb::make_tuple(value, gradient, hessian);           \
                }, "pos"_a, "active"_a.sig("Bool(True)") = nb::none(),         \
                doc_Texture_eval_cubic_hessian)
        .def_tex_eval_cubic_hessian(Float32)
        .def_tex_eval_cubic_hessian(Float16)
        .def_tex_eval_cubic_hessian(Float64)
        #undef def_tex_eval_cubic_hessian
        #define def_tex_eval_cubic_helper(T)                                   \
            def("eval_cubic_helper",                                           \
                [](const Tex &texture, const dr::Array<T, Dimension> &pos,     \
                   const std::optional<dr::mask_t<T>> active_) {               \
                    dr::mask_t<T> active = active_.has_value() ?               \
                                                     active_.value() :         \
                                                     true;                     \
                                                                               \
                    size_t channels = texture.shape()[Dimension];              \
                    dr::vector<T> result(channels);                            \
                    texture.eval_cubic_helper(pos, result.data(), active);     \
                                                                               \
                    return result;                                             \
                }, "pos"_a, "active"_a.sig("Bool(True)") = nb::none(),         \
                doc_Texture_eval_cubic_helper)
        .def_tex_eval_cubic_helper(Float32)
        .def_tex_eval_cubic_helper(Float16)
        .def_tex_eval_cubic_helper(Float64);
        #undef def_tex_eval_cubic_helper

    tex.attr("IsTexture") = true;

    drjit::bind_traverse(tex);
}

template <typename Type>
void bind_texture_all(nb::module_ &m) {
    using Type16 = dr::float16_array_t<Type>;
    using Type32 = dr::float32_array_t<Type>;
    using Type64 = dr::float64_array_t<Type>;
    bind_texture<Type16, 1>(m, "Texture1f16");
    bind_texture<Type16, 2>(m, "Texture2f16");
    bind_texture<Type16, 3>(m, "Texture3f16");
    bind_texture<Type32, 1>(m, "Texture1f");
    bind_texture<Type32, 2>(m, "Texture2f");
    bind_texture<Type32, 3>(m, "Texture3f");
    bind_texture<Type64, 1>(m, "Texture1f64");
    bind_texture<Type64, 2>(m, "Texture2f64");
    bind_texture<Type64, 3>(m, "Texture3f64");
}
