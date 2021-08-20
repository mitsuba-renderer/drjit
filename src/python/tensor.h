#include <enoki/tensor.h>
#include <pybind11/stl.h>

template <typename T> auto bind_tensor(py::module m) {
    using Tensor = ek::Tensor<T>;
    auto cls = bind_type<Tensor>(m);

    cls.attr("Index") = py::type::of<typename Tensor::Index>();
    cls.attr("Array") = py::type::of<typename Tensor::Array>();

    cls.def(py::init<>())
       .def(py::init([](py::object o) -> Tensor {
            std::string mod = py::cast<std::string>(o.get_type().attr("__module__"));
            const char *mod_s = mod.c_str();
            if (strncmp(mod_s, "numpy", 5) == 0 ||
                strncmp(mod_s, "torch", 5) == 0 ||
                strncmp(mod_s, "jaxlib", 6) == 0 ||
                strncmp(mod_s, "tensorflow", 10) == 0 ||
                (py::hasattr(o, "__array_interface__") &&
                 strncmp(mod_s, "enoki", 5) != 0)) {
                o = tensor_init(py::type::of<Tensor>(), o);
                return py::cast<Tensor>(o);
            } else {
                // trick to get pybind11 to ignore this overload
                throw py::reference_cast_error();
            }
       }), "array"_a)
       .def(py::init<T>(), "array"_a)
       .def(py::init([](const T &array, const std::vector<size_t> &shape) {
               return Tensor(array, shape.size(), shape.data());
            }), "array"_a, "shape"_a)
       .def("__len__", &Tensor::size)
       .def_property_readonly("ndim", &Tensor::ndim)
       .def_property_readonly("array", [](Tensor &t) { return &(t.array()); })
       .def_property_readonly("shape", [](const Tensor &t) {
            PyObject *shape = PyTuple_New(t.ndim());
            for (size_t i = 0; i < t.ndim(); ++i)
                PyTuple_SET_ITEM(shape, i, PyLong_FromLong(t.shape(i)));
            return py::reinterpret_steal<py::tuple>(shape);
        })
       .def("data_", [](const Tensor &a) {
            return (uintptr_t) a.data();
        });

    cls.def("or_",     [](const Tensor &a, const Tensor &b) { return a.or_(b); });
    cls.def("and_",    [](const Tensor &a, const Tensor &b) { return a.and_(b); });
    cls.def("xor_",    [](const Tensor &a, const Tensor &b) { return a.xor_(b); });
    cls.def("andnot_", [](const Tensor &a, const Tensor &b) { return a.andnot_(b); });
    cls.def("not_", &Tensor::not_);

    cls.def("ior_",    [](Tensor *a, const Tensor &b) { *a = a->or_(b);  return a;});
    cls.def("iand_",   [](Tensor *a, const Tensor &b) { *a = a->and_(b); return a;});
    cls.def("ixor_",   [](Tensor *a, const Tensor &b) { *a = a->xor_(b); return a;});

    if constexpr (!ek::is_mask_v<T>) {
        cls.def("add_", &Tensor::add_)
        .def("sub_", &Tensor::sub_)
        .def("mul_", &Tensor::mul_)
        .def("neg_", &Tensor::neg_)
        .def("fmadd_", &Tensor::div_);

        cls.def("lt_", &Tensor::lt_)
        .def("gt_", &Tensor::gt_)
        .def("le_", &Tensor::le_)
        .def("ge_", &Tensor::ge_);

        cls.def(Tensor::IsFloat ? "truediv_" : "floordiv_", &Tensor::div_);

        cls.def("iadd_", [](Tensor *a, const Tensor &b) { *a = a->add_(b); return a; });
        cls.def("isub_", [](Tensor *a, const Tensor &b) { *a = a->sub_(b); return a; });
        cls.def("imul_", [](Tensor *a, const Tensor &b) { *a = a->mul_(b); return a; });
        cls.def("imod_", [](Tensor *a, const Tensor &b) { *a = a->mod_(b); return a; });

        cls.def(Tensor::IsFloat ? "itruediv_" : "ifloordiv_",
                [](Tensor *a, const Tensor &b) { *a = a->div_(b); return a; });

        cls.def("abs_", &Tensor::abs_)
           .def("min_", &Tensor::min_)
           .def("max_", &Tensor::max_);
    }

    if constexpr (ek::is_floating_point_v<Tensor>) {
        cls.def("rcp_", &Tensor::rcp_)
           .def("rsqrt_", &Tensor::rsqrt_);
    }

    if constexpr (ek::is_integral_v<Tensor>) {
        cls.def("mod_", &Tensor::mod_);
        cls.def("mulhi_", &Tensor::mulhi_);
    }

    cls.attr("select_") = py::cpp_function([](const ek::mask_t<Tensor> &m,
                                              const Tensor &t,
                                              const Tensor &f) {
        return Tensor::select_(m, t, f);
    });

    cls.def("neq_", &Tensor::neq_)
       .def("eq_", &Tensor::eq_);

    return cls;
}

template <typename Tensor> auto bind_tensor_conversions(py::class_<Tensor> &cls) {
    if constexpr (!ek::is_mask_v<Tensor>) {
        cls.def(py::init<const ek::int32_array_t<Tensor> &>(), py::arg().noconvert())
           .def(py::init<const ek::uint32_array_t<Tensor> &>(), py::arg().noconvert())
           .def(py::init<const ek::int64_array_t<Tensor> &>(), py::arg().noconvert())
           .def(py::init<const ek::uint64_array_t<Tensor> &>(), py::arg().noconvert())
           .def(py::init<const ek::float32_array_t<Tensor> &>(), py::arg().noconvert())
           .def(py::init<const ek::float64_array_t<Tensor> &>(), py::arg().noconvert());
    } else {
        cls.def(py::init<const Tensor &>(), py::arg().noconvert());
    }

    if constexpr (ek::is_diff_array_v<Tensor>)
        cls.def(py::init<const ek::detached_t<Tensor> &>(), py::arg().noconvert());

    using Scalar = ek::scalar_t<Tensor>;
    if constexpr (sizeof(Scalar) == 4) {
        cls.def_static("reinterpret_array_",
                       [](const ek::int32_array_t<Tensor> &a) {
                           return ek::reinterpret_array<Tensor>(a);
                       });
        cls.def_static("reinterpret_array_",
                       [](const ek::uint32_array_t<Tensor> &a) {
                           return ek::reinterpret_array<Tensor>(a);
                       });
        cls.def_static("reinterpret_array_",
                       [](const ek::float32_array_t<Tensor> &a) {
                           return ek::reinterpret_array<Tensor>(a);
                       });
    } else if constexpr (sizeof(Scalar) == 8) {
        cls.def_static("reinterpret_array_",
                       [](const ek::int64_array_t<Tensor> &a) {
                           return ek::reinterpret_array<Tensor>(a);
                       });
        cls.def_static("reinterpret_array_",
                       [](const ek::uint64_array_t<Tensor> &a) {
                           return ek::reinterpret_array<Tensor>(a);
                       });
        cls.def_static("reinterpret_array_",
                       [](const ek::float64_array_t<Tensor> &a) {
                           return ek::reinterpret_array<Tensor>(a);
                       });
    }
}

#define ENOKI_BIND_TENSOR_TYPES(module)                                        \
    auto tensor_b = bind_tensor<ek::mask_t<Guide>>(module);                    \
    auto tensor_f32 = bind_tensor<ek::float32_array_t<Guide>>(module);         \
    auto tensor_f64 = bind_tensor<ek::float64_array_t<Guide>>(module);         \
    auto tensor_u32 = bind_tensor<ek::uint32_array_t<Guide>>(module);          \
    auto tensor_i32 = bind_tensor<ek::int32_array_t<Guide>>(module);           \
    auto tensor_u64 = bind_tensor<ek::uint64_array_t<Guide>>(module);          \
    auto tensor_i64 = bind_tensor<ek::int64_array_t<Guide>>(module);           \
    bind_tensor_conversions(tensor_b);                                         \
    bind_tensor_conversions(tensor_f32);                                       \
    bind_tensor_conversions(tensor_f64);                                       \
    bind_tensor_conversions(tensor_u32);                                       \
    bind_tensor_conversions(tensor_i32);                                       \
    bind_tensor_conversions(tensor_u64);                                       \
    bind_tensor_conversions(tensor_i64);
