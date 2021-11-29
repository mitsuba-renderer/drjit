#include <enoki/texture.h>

template <typename Type, size_t Dimension>
void bind_texture(py::module &m, const char *name) {
    using Tex = ek::Texture<Type, Dimension>;

    py::class_<Tex>(m, name)
        .def(py::init([](std::array<size_t, Dimension> shape, size_t channels,
                         bool) { return new Tex(shape.data(), channels); }),
             "shape"_a, "channels"_a, "migrate"_a = true)
        .def(py::init<const typename Tex::TensorXf &, bool>(), "tensor"_a,
             "migrate"_a = true)
        .def("set_value", &Tex::set_value, "value"_a)
        .def("set_tensor", &Tex::set_tensor, "tensor"_a)
        .def("value", &Tex::value)
        .def("tensor", &Tex::tensor)
        .def("eval_cuda", &Tex::eval_cuda, "pos"_a, "active"_a = true)
        .def("eval_enoki", &Tex::eval_enoki, "pos"_a, "active"_a = true)
        .def("eval", &Tex::eval, "pos"_a, "active"_a = true);
}

template <typename Type>
void bind_texture_all(py::module &m) {
    bind_texture<Type, 1>(m, "Texture1f");
    bind_texture<Type, 2>(m, "Texture2f");
    bind_texture<Type, 3>(m, "Texture3f");
}
