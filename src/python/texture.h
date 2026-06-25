#pragma once

#include "common.h"
#include "event.h"
#include <drjit/texture.h>
#include <nanobind/stl/optional.h>

/// Helper macro to dynamically adapt the output type to the channel count
#define DR_TEX_DISPATCH(texture, expr)                                         \
    switch ((texture).shape()[Dimension]) {                                    \
        case 1:  { using Output = dr::Array<T, 1>;      return (expr); }       \
        case 2:  { using Output = dr::Array<T, 2>;      return (expr); }       \
        case 3:  { using Output = dr::Array<T, 3>;      return (expr); }       \
        case 4:  { using Output = dr::Array<T, 4>;      return (expr); }       \
        default: { using Output = dr::DynamicArray<T>;  return (expr); }       \
    }

// Build a ``pos``/``active`` signature with extra args and return type
#define DR_TEX_SIG(name, extra, ret)                                           \
    "def " name "(self, pos: drjit.AnyArray, "                                 \
    "active: drjit.AnyArray | bool = True" extra ") -> " ret

// Bind one ``Texture`` method, specialized to the query precision ``T``
#define def_tex_eval(T)                                                        \
    def("eval", &tex_eval<T, Dimension, Tex>, "pos"_a,                         \
        "active"_a = nb::none(),                                               \
        nb::sig(DR_TEX_SIG("eval", "", "drjit.AnyArray")), doc_Texture_eval)
#define def_tex_write(T)                                                       \
    def("write", &tex_write<T, Dimension, Tex>, "pos"_a, "value"_a,            \
        "active"_a.sig("Bool(True)") = nb::none(), doc_Texture_write)
#define def_tex_eval_fetch(T)                                                  \
    def("eval_fetch", &tex_eval_fetch<T, Dimension, Tex>, "pos"_a,             \
        "active"_a = nb::none(),                                               \
        nb::sig(DR_TEX_SIG("eval_fetch", "", "tuple[drjit.AnyArray, ...]")),   \
        doc_Texture_eval_fetch)
#define def_tex_eval_cubic(T)                                                  \
    def("eval_cubic", &tex_eval_cubic<T, Dimension, Tex>, "pos"_a,             \
        "active"_a = nb::none(), "force_nonaccel"_a = false,                   \
        nb::sig(DR_TEX_SIG("eval_cubic", ", force_nonaccel: bool = False",     \
                           "drjit.AnyArray")),                                 \
        doc_Texture_eval_cubic)
#define def_tex_eval_cubic_grad(T)                                             \
    def("eval_cubic_grad", &tex_eval_cubic_grad<T, Dimension, Tex>, "pos"_a,   \
        "active"_a = nb::none(),                                               \
        nb::sig(DR_TEX_SIG("eval_cubic_grad", "",                              \
                           "tuple[drjit.AnyArray, list[drjit.AnyArray]]")),    \
        doc_Texture_eval_cubic_grad)
#define def_tex_eval_cubic_hessian(T)                                          \
    def("eval_cubic_hessian", &tex_eval_cubic_hessian<T, Dimension, Tex>,      \
        "pos"_a, "active"_a = nb::none(),                                      \
        nb::sig(DR_TEX_SIG("eval_cubic_hessian", "",                           \
                           "tuple[drjit.AnyArray, list[drjit.AnyArray], "      \
                           "list[drjit.AnyArray]]")),                          \
        doc_Texture_eval_cubic_hessian)
#define def_tex_eval_cubic_helper(T)                                           \
    def("eval_cubic_helper", &tex_eval_cubic_helper<T, Dimension, Tex>,        \
        "pos"_a, "active"_a = nb::none(),                                      \
        nb::sig(DR_TEX_SIG("eval_cubic_helper", "", "drjit.AnyArray")),        \
        doc_Texture_eval_cubic_helper)

/// Resolve an optional mask, defaulting to ``true`` when unset
template <typename T>
static dr::mask_t<T> mask_or_true(const std::optional<dr::mask_t<T>> &active) {
    return active.has_value() ? active.value() : dr::mask_t<T>(true);
}

template <typename T, size_t Dimension, typename Tex>
static nb::object tex_eval(const Tex &texture,
                           const dr::Array<T, Dimension> &pos,
                           const std::optional<dr::mask_t<T>> &active_) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    DR_TEX_DISPATCH(texture,
                    nb::cast(texture.template eval<Output>(pos, active)));
}

template <typename T, size_t Dimension, typename Tex>
static nb::object tex_eval_cubic(const Tex &texture,
                                 const dr::Array<T, Dimension> &pos,
                                 const std::optional<dr::mask_t<T>> &active_,
                                 bool force_nonaccel) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    DR_TEX_DISPATCH(texture, nb::cast(texture.template eval_cubic<Output>(
                                 pos, active, force_nonaccel)));
}

template <typename T, size_t Dimension, typename Tex>
static nb::object tex_eval_cubic_helper(const Tex &texture,
                                        const dr::Array<T, Dimension> &pos,
                                        const std::optional<dr::mask_t<T>> &active_) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    DR_TEX_DISPATCH(texture, nb::cast(texture.template eval_cubic_helper<Output>(
                                 pos, active)));
}

template <typename T, size_t Dimension, typename Tex>
static nb::object tex_eval_fetch(const Tex &texture,
                                 const dr::Array<T, Dimension> &pos,
                                 const std::optional<dr::mask_t<T>> &active_) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    auto to_tuple = [](auto &&corners) {
        constexpr size_t ResultSize = 1 << Dimension;
        nb::object out = nb::steal(PyTuple_New((Py_ssize_t) ResultSize));
        for (size_t i = 0; i < ResultSize; ++i)
            NB_TUPLE_SET_ITEM(out.ptr(), (Py_ssize_t) i,
                              nb::cast(corners.entry(i)).release().ptr());
        return out;
    };
    DR_TEX_DISPATCH(
        texture, to_tuple(texture.template eval_fetch<Output>(pos, active)));
}

template <typename T, size_t Dimension, typename Tex>
static nb::object tex_eval_cubic_grad(const Tex &texture,
                                      const dr::Array<T, Dimension> &pos,
                                      const std::optional<dr::mask_t<T>> &active_) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    size_t channels = texture.shape()[Dimension];
    auto build = [&](auto &&res) {
        nb::list gradient;
        for (size_t ch = 0; ch < channels; ++ch)
            gradient.append(nb::cast(res.gradient.entry(ch)));
        return nb::make_tuple(nb::cast(res.value), gradient);
    };
    DR_TEX_DISPATCH(
        texture, build(texture.template eval_cubic_grad<Output>(pos, active)));
}

template <typename T, size_t Dimension, typename Tex>
static nb::object tex_eval_cubic_hessian(const Tex &texture,
                                         const dr::Array<T, Dimension> &pos,
                                         const std::optional<dr::mask_t<T>> &active_) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    size_t channels = texture.shape()[Dimension];
    auto build = [&](auto &&res) {
        nb::list gradient, hessian;
        for (size_t ch = 0; ch < channels; ++ch) {
            gradient.append(nb::cast(res.gradient.entry(ch)));
            hessian.append(nb::cast(res.hessian.entry(ch)));
        }
        return nb::make_tuple(nb::cast(res.value), gradient, hessian);
    };
    DR_TEX_DISPATCH(texture, build(texture.template eval_cubic_hessian<Output>(
                                 pos, active)));
}

template <typename Type, size_t Dimension, typename Tex>
static auto tex_wrap(const Tex &texture,
                     const dr::Array<dr::int32_array_t<Type>, Dimension> &pos) {
    return texture.wrap(pos);
}

template <typename T, size_t Dimension, typename Tex>
static void tex_write(Tex &texture,
                      const dr::Array<dr::uint32_array_t<T>, Dimension> &pos,
                      const dr::vector<T> &value,
                      const std::optional<dr::mask_t<T>> &active_) {
    dr::mask_t<T> active = mask_or_true<T>(active_);
    size_t channels = texture.shape()[Dimension];
    if (value.size() != channels)
        nb::raise("Texture.write(): expected %zu channel values, got %zu.",
                  channels, value.size());
    if constexpr (dr::is_jit_v<typename Tex::Storage>)
        texture.write(pos, value.data(), active);
    else
        nb::raise("Texture.write(): requires a JIT backend (CUDA, Metal, or LLVM).");
}

template <typename Type, size_t Dimension, typename QueryArray = Type>
void bind_texture(nb::module_ &m, const char *name) {
    using Tex = dr::Texture<Type, Dimension>;
    // Query/output precisions; for 8-bit textures these come from a separate
    // floating-point guide (\c QueryArray) since the storage type is integral.
    using Float16 = dr::replace_scalar_t<QueryArray, dr::half>;
    using Float32 = dr::replace_scalar_t<QueryArray, float>;
    using Float64 = dr::replace_scalar_t<QueryArray, double>;

    auto tex = nb::class_<Tex>(m, name)
        .def("__init__", [](Tex* t, const dr::vector<size_t>& shape,
                         size_t channels, bool use_accel,
                         dr::FilterMode filter_mode, dr::WrapMode wrap_mode,
                         bool writable, bool srgb) {
                 new (t) Tex(shape.data(), channels, use_accel, filter_mode,
                             wrap_mode, writable, srgb); },
             "shape"_a, "channels"_a, "use_accel"_a = true,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp,
             "writable"_a = false, "srgb"_a = false,
             doc_Texture_init)
        .def(nb::init<const typename Tex::TensorXf &, bool, bool, dr::FilterMode,
                      dr::WrapMode, bool>(),
             "tensor"_a, "use_accel"_a = true, "migrate"_a = true,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp, "srgb"_a = false,
             doc_Texture_init_tensor)
        .def("set_value",
             &Tex::template set_value<const typename Tex::Storage &>,
             "value"_a, "migrate"_a = false, doc_Texture_set_value)
        .def("set_value_with_event",
             [](Tex &t, const typename Tex::Storage &value,
                Event<Tex::Backend> &event, bool migrate) {
                 t.set_value(value, migrate);
                 event.record();
             },
             "value"_a, "event"_a, "migrate"_a = false, doc_Texture_set_value_2)
        .def("set_tensor", &Tex::template set_tensor<const typename Tex::TensorXf &>, "tensor"_a,  "migrate"_a = false, doc_Texture_set_tensor)
        .def("update_inplace", &Tex::update_inplace, "migrate"_a = false, doc_Texture_update_inplace)
        .def("value", &Tex::value, nb::rv_policy::reference_internal, doc_Texture_value)
        .def("tensor",
             nb::overload_cast<>(&Tex::tensor, nb::const_),
             nb::rv_policy::reference_internal, doc_Texture_tensor)
        .def("filter_mode", &Tex::filter_mode, doc_Texture_filter_mode)
        .def("wrap_mode", &Tex::wrap_mode, doc_Texture_wrap_mode)
        .def("wrap", &tex_wrap<Type, Dimension, Tex>, "pos"_a,
             nb::sig("def wrap(self, pos: drjit.AnyArray) -> drjit.AnyArray"),
             doc_Texture_wrap)
        .def("use_accel", &Tex::use_accel, doc_Texture_use_accel)
        .def("writable", &Tex::writable, doc_Texture_writable)
        .def("srgb", &Tex::srgb, doc_Texture_srgb)
        .def_static("from_native_handle", &Tex::from_native_handle, "handle"_a,
             "writable"_a = false,
             "filter_mode"_a = dr::FilterMode::Linear,
             "wrap_mode"_a = dr::WrapMode::Clamp, "srgb"_a = false,
             doc_Texture_from_native_handle)
        .def("map", &Tex::map, doc_Texture_map)
        .def("unmap", &Tex::unmap, doc_Texture_unmap)
        .def("native_handle", &Tex::native_handle, "sub_index"_a = 0,
             doc_Texture_native_handle)
        .def("migrated", &Tex::migrated, doc_Texture_migrated)
        .def_prop_ro("shape", [](const Tex &t) {
            PyObject *shape = PyTuple_New(t.ndim());
            for (size_t i = 0; i < t.ndim(); ++i)
                NB_TUPLE_SET_ITEM(shape, i, PyLong_FromLong((long) t.shape()[i]));
            return nb::steal<nb::tuple>(shape);
        }, doc_Texture_shape)
        .def("channel_count", &Tex::channel_count, doc_Texture_channel_count)
        .def_tex_eval(Float32)
        .def_tex_eval(Float16)
        .def_tex_eval(Float64)
        .def_tex_write(Float32)
        .def_tex_write(Float16)
        .def_tex_write(Float64)
        .def_tex_eval_fetch(Float32)
        .def_tex_eval_fetch(Float16)
        .def_tex_eval_fetch(Float64)
        .def_tex_eval_cubic(Float32)
        .def_tex_eval_cubic(Float16)
        .def_tex_eval_cubic(Float64)
        .def_tex_eval_cubic_grad(Float32)
        .def_tex_eval_cubic_grad(Float16)
        .def_tex_eval_cubic_grad(Float64)
        .def_tex_eval_cubic_hessian(Float32)
        .def_tex_eval_cubic_hessian(Float16)
        .def_tex_eval_cubic_hessian(Float64)
        .def_tex_eval_cubic_helper(Float32)
        .def_tex_eval_cubic_helper(Float16)
        .def_tex_eval_cubic_helper(Float64);

    tex.attr("IsTexture") = true;

    drjit::bind_traverse(tex);
}

template <typename Type>
void bind_texture_all(nb::module_ &m) {
    using Type16 = dr::float16_array_t<Type>;
    using Type32 = dr::float32_array_t<Type>;
    using Type64 = dr::float64_array_t<Type>;
    // 8-bit storage is integral; queries still return single precision via the
    // \c Type32 guide (differentiable in the AD variants).
    using Type8 = dr::uint8_array_t<Type>;
    bind_texture<Type16, 1>(m, "Texture1f16");
    bind_texture<Type16, 2>(m, "Texture2f16");
    bind_texture<Type16, 3>(m, "Texture3f16");
    bind_texture<Type32, 1>(m, "Texture1f");
    bind_texture<Type32, 2>(m, "Texture2f");
    bind_texture<Type32, 3>(m, "Texture3f");
    bind_texture<Type64, 1>(m, "Texture1f64");
    bind_texture<Type64, 2>(m, "Texture2f64");
    bind_texture<Type64, 3>(m, "Texture3f64");
    bind_texture<Type8, 1, Type32>(m, "Texture1f8u");
    bind_texture<Type8, 2, Type32>(m, "Texture2f8u");
    bind_texture<Type8, 3, Type32>(m, "Texture3f8u");
}

#undef DR_TEX_DISPATCH
#undef DR_TEX_SIG
#undef def_tex_eval
#undef def_tex_write
#undef def_tex_eval_fetch
#undef def_tex_eval_cubic
#undef def_tex_eval_cubic_grad
#undef def_tex_eval_cubic_hessian
#undef def_tex_eval_cubic_helper
