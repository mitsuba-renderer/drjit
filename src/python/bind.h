#pragma once

#include "common.h"
#include <enoki/math.h>

extern py::handle array_name, array_init, array_configure;

template <typename Array>
auto bind_type(py::module &m, bool scalar_mode = false) {
    using Scalar = std::conditional_t<Array::IsMask, bool, ek::scalar_t<Array>>;
    using Value = std::conditional_t<Array::IsMask, ek::mask_t<ek::value_t<Array>>,
                                     ek::value_t<Array>>;
    constexpr VarType Type = ek::var_type_v<Scalar>;

    py::str name = array_name(Type, Array::Depth, Array::Size, scalar_mode);
    auto cls = py::class_<Array, ek::ArrayBase>(
        m, PyUnicode_AsUTF8AndSize(name.ptr(), nullptr));

    cls.attr("Value") = py::cast(Value()).get_type();
    cls.attr("Type") = py::cast(Type);
    cls.attr("Size") = py::cast(Array::Size);

    if constexpr (Array::Size == ek::Dynamic) {
        cls.def("init_", [](Array &a, size_t size) {
            if (!a.empty())
                ek::enoki_raise("Dynamic array was already initialized!");
            a.init_(size);
        });
    }

    array_configure(cls);
    register_implicit_conversions(typeid(Array));

    return cls;
}

template <typename Array>
void bind_basic_methods(py::class_<Array, ek::ArrayBase> &cls) {
    using Value = std::conditional_t<Array::IsMask, ek::mask_t<ek::value_t<Array>>,
                                     ek::value_t<Array>>;
    if (hasattr(cls, "entry"))
        return;

    cls.def("__len__", &Array::size)
       .def("entry", [](const Array &a, size_t i) -> Value { return a.entry(i); })
       .def("set_entry", [](Array &a, size_t i, const Value &value) {
           a.set_entry(i, value);
       })
       .def_static("empty_", &Array::empty_);
}

template <typename Array>
void bind_generic_constructor(py::class_<Array, ek::ArrayBase> &cls) {
    cls.def(
        "__init__",
        [](py::detail::value_and_holder &v_h, py::args args) {
            v_h.value_ptr() = new Array();
            array_init(py::handle((PyObject *) v_h.inst), args);
        },
        py::detail::is_new_style_constructor());
}

template <typename Array> auto bind(py::module &m, bool scalar_mode = false) {
    auto cls = bind_type<Array>(m, scalar_mode);
    bind_generic_constructor(cls);
    bind_basic_methods(cls);
    return cls;
};

template <typename Array>
auto bind_full(py::class_<Array, ek::ArrayBase> &cls,
               bool scalar_mode = false) {
    static_assert(ek::array_depth_v<Array> == 1);
    bind_basic_methods(cls);

    using Scalar = std::conditional_t<Array::IsMask, bool, ek::scalar_t<Array>>;
    using Mask = ek::mask_t<ek::float32_array_t<Array>>;

    cls.def(py::init<Scalar>());
    if constexpr (Array::IsFloat)
        cls.def(py::init([](ssize_t value) { return new Array((Scalar) value); }));

    if constexpr (!Array::IsMask) {
        cls.def(py::init<const ek::  int32_array_t<Array> &>());
        cls.def(py::init<const ek:: uint32_array_t<Array> &>());
        cls.def(py::init<const ek::  int64_array_t<Array> &>());
        cls.def(py::init<const ek:: uint64_array_t<Array> &>());
        cls.def(py::init<const ek::float32_array_t<Array> &>());
        cls.def(py::init<const ek::float64_array_t<Array> &>());
    }

    cls.def("or_",     [](const Array &a, const Array &b) { return a.or_(b); });
    cls.def("and_",    [](const Array &a, const Array &b) { return a.and_(b); });
    cls.def("xor_",    [](const Array &a, const Array &b) { return a.xor_(b); });
    cls.def("andnot_", [](const Array &a, const Array &b) { return a.andnot_(b); });
    cls.def("not_", &Array::not_);

    cls.def("ior_",    [](Array *a, const Array &b) { *a = a->or_(b);  return a;});
    cls.def("iand_",   [](Array *a, const Array &b) { *a = a->and_(b); return a;});
    cls.def("ixor_",   [](Array *a, const Array &b) { *a = a->xor_(b); return a;});

    if constexpr (std::is_same_v<Mask, ek::mask_t<Array>>) {
        cls.def("eq_", &Array::eq_);
        cls.def("neq_", &Array::neq_);
    } else {
        cls.def("eq_", [](const Array &a, const Array &b) -> Mask { return a.eq_(b); });
        cls.def("neq_", [](const Array &a, const Array &b) -> Mask { return a.neq_(b); });
    }

    cls.attr("zero_") = py::cpp_function(&Array::zero_);
    cls.attr("full_") = py::cpp_function(&Array::full_);

    if constexpr (!Array::IsMask) {
        cls.attr("arange_") = py::cpp_function(&Array::arange_);
        cls.attr("linspace_") = py::cpp_function(&Array::linspace_);
    }

    cls.attr("select_") = py::cpp_function([](const Mask &m, const Array &t,
                                              const Array &f) {
        return Array::select_(static_cast<const ek::mask_t<Array>>(m), t, f);
    });

    if constexpr (Array::IsMask) {
        cls.def("all_", &Array::all_);
        cls.def("any_", &Array::any_);
        cls.def("count_", &Array::count_);
    } else {
        if constexpr (sizeof(Scalar) == 4) {
            cls.def_static("reinterpret_array_",
                           [](const ek::int32_array_t<Array> &a) {
                               return ek::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const ek::uint32_array_t<Array> &a) {
                               return ek::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const ek::float32_array_t<Array> &a) {
                               return ek::reinterpret_array<Array>(a);
                           });
        } else {
            cls.def_static("reinterpret_array_",
                           [](const ek::int64_array_t<Array> &a) {
                               return ek::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const ek::uint64_array_t<Array> &a) {
                               return ek::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const ek::float64_array_t<Array> &a) {
                               return ek::reinterpret_array<Array>(a);
                           });
        }

        cls.def("add_", &Array::add_);
        cls.def("sub_", &Array::sub_);
        cls.def("mul_", &Array::mul_);
        cls.def("mod_", &Array::mod_);
        cls.def(Array::IsFloat ? "truediv_" : "floordiv_", &Array::div_);

        cls.def("iadd_", [](Array *a, const Array &b) { *a = a->add_(b); return a; });
        cls.def("isub_", [](Array *a, const Array &b) { *a = a->sub_(b); return a; });
        cls.def("imul_", [](Array *a, const Array &b) { *a = a->mul_(b); return a; });
        cls.def("imod_", [](Array *a, const Array &b) { *a = a->mod_(b); return a; });

        cls.def(Array::IsFloat ? "itruediv_" : "ifloordiv_",
                [](Array *a, const Array &b) { *a = a->div_(b); return a; });

        cls.def("dot_", &Array::dot_);
        cls.def("hsum_", &Array::hsum_);
        cls.def("hprod_", &Array::hprod_);
        cls.def("hmin_", &Array::hmin_);
        cls.def("hmax_", &Array::hmax_);

        if constexpr (ek::is_dynamic_v<Array> &&
                      ek::array_depth_v<Array> == 1) {
            if constexpr (ek::is_jit_array_v<Array>) {
                cls.def("dot_async_", &Array::dot_async_);
                cls.def("hsum_async_", &Array::hsum_async_);
                cls.def("hprod_async_", &Array::hprod_async_);
                cls.def("hmin_async_", &Array::hmin_async_);
                cls.def("hmax_async_", &Array::hmax_async_);
                cls.def("migrate_", &Array::migrate);
            }
        }

        cls.def("and_", [](const Array &a, const Mask &b) {
            return a.and_(static_cast<const ek::mask_t<Array> &>(b));
        });

        cls.def("iand_", [](Array *a, const Mask &b) {
            *a = a->and_(static_cast<const ek::mask_t<Array> &>(b));
            return a;
        });

        cls.def("or_", [](const Array &a, const Mask &b) {
            return a.or_(static_cast<const ek::mask_t<Array> &>(b));
        });

        cls.def("ior_", [](Array *a, const Mask &b) {
            *a = a->or_(static_cast<const ek::mask_t<Array> &>(b));
            return a;
        });

        cls.def("xor_", [](const Array &a, const Mask &b) {
            return a.xor_(static_cast<const ek::mask_t<Array> &>(b));
        });

        cls.def("ixor_", [](Array *a, const Mask &b) {
            *a = a->xor_(static_cast<const ek::mask_t<Array> &>(b));
            return a;
        });

        cls.def("andnot_", [](const Array &a, const Mask &b) {
            return a.andnot_(static_cast<const ek::mask_t<Array> &>(b));
        });

        cls.def("abs_", &Array::abs_);
        cls.def("min_", &Array::min_);
        cls.def("max_", &Array::max_);

        if constexpr (std::is_same_v<Mask, ek::mask_t<Array>>) {
            cls.def("lt_", &Array::lt_);
            cls.def("le_", &Array::le_);
            cls.def("gt_", &Array::gt_);
            cls.def("ge_", &Array::ge_);
        } else {
            cls.def("lt_", [](const Array &a, const Array &b) -> Mask { return a.lt_(b); });
            cls.def("le_", [](const Array &a, const Array &b) -> Mask { return a.le_(b); });
            cls.def("gt_", [](const Array &a, const Array &b) -> Mask { return a.gt_(b); });
            cls.def("ge_", [](const Array &a, const Array &b) -> Mask { return a.ge_(b); });
        }

        cls.def("fmadd_", &Array::fmadd_);

        cls.def("neg_", &Array::neg_);
        cls.def("hsum_", &Array::hsum_);
        cls.def("hprod_", &Array::hprod_);
        cls.def("hmax_", &Array::hmax_);
        cls.def("hmin_", &Array::hmin_);
    }

    if constexpr (ek::is_dynamic_v<Array>) {
        using UInt32 = ek::uint32_array_t<Array>;
        cls.def_static("gather_",
                [](const Array &source, const UInt32 &index, const Mask &mask) {
                    return ek::gather<Array>(source, index, mask);
                });
        cls.def("scatter_",
                [](const Array &value, Array &target, const UInt32 &index, const Mask &mask) {
                    ek::scatter(target, value, index, mask);
                });
        cls.def("scatter_add_",
                [](const Array &value, Array &target, const UInt32 &index, const Mask &mask) {
                    ek::scatter_add(target, value, index, mask);
                });
    }

    if constexpr (Array::IsFloat) {
        cls.def("sqrt_",  &Array::sqrt_);
        cls.def("floor_", &Array::floor_);
        cls.def("ceil_",  &Array::ceil_);
        cls.def("round_", &Array::round_);
        cls.def("trunc_", &Array::trunc_);
        cls.def("rcp_",   &Array::rcp_);
        cls.def("rsqrt_", &Array::rsqrt_);
    } else if constexpr(Array::IsIntegral) {
        // cls.def("mulhi_", &Array::mulhi_);
        cls.def("sl_", [](const Array &a, const Array &b) { return a.sl_(b); });
        cls.def("sr_", [](const Array &a, const Array &b) { return a.sr_(b); });
        cls.def("isl_", [](Array *a, const Array &b) { *a = a->sl_(b); return a; });
        cls.def("isr_", [](Array *a, const Array &b) { *a = a->sr_(b); return a; });
    }

    if constexpr (Array::IsFloat) {
        cls.def("sin_", [](const Array &a) { return ek::sin(a); });
        cls.def("cos_", [](const Array &a) { return ek::cos(a); });
        cls.def("sincos_", [](const Array &a) { return ek::sincos(a); });
        cls.def("tan_", [](const Array &a) { return ek::tan(a); });
        cls.def("cot_", [](const Array &a) { return ek::cot(a); });
        cls.def("asin_", [](const Array &a) { return ek::asin(a); });
        cls.def("acos_", [](const Array &a) { return ek::acos(a); });
        cls.def("atan_", [](const Array &a) { return ek::atan(a); });
        cls.def("atan2_", [](const Array &y, const Array &x) { return ek::atan2(y, x); });
    }

    if constexpr (Array::IsJIT || Array::IsDiff) {
        cls.def("index_", [](const Array &a) { return a.index(); });
        cls.def("set_label_", [](const Array &a, const char *name) { a.set_label(name); });
        cls.def("label_", [](const Array &a) { return a.label(); });
    }

    if constexpr (Array::IsDiff) {
        using Detached = decltype(ek::detach(std::declval<Array>()));
        cls.def(py::init<Detached>());
        cls.def("value_", &Array::value);
        if constexpr (Array::IsFloat) {
            cls.def("grad_", [](const Array &a) -> py::object {
                if (a.index() == 0)
                    return py::none();
                else
                    return py::cast(a.grad());
            });
            cls.def("set_grad_", &Array::set_grad);
            cls.def(
                "requires_grad_",
                [](Array *a, bool value) -> Array * {
                    ek::requires_grad(*a, value);
                    return a;
                },
                "value"_a = true);
            cls.def("ad_schedule_", &Array::ad_schedule);
            cls.def("graphviz_", &Array::graphviz_);
            cls.def_static("traverse_", &Array::traverse);
        }
    }

    bind_generic_constructor(cls);

    return cls;
}

#define ENOKI_BIND_ARRAY_TYPES_DYN(Module, Guide, Scalar)                      \
    auto d_b = bind<ek::mask_t<ek::DynamicArray<ek::float32_array_t<Guide>>>>( \
        Module, Scalar);                                                       \
    auto d_i32 =                                                               \
        bind<ek::DynamicArray<ek::int32_array_t<Guide>>>(Module, Scalar);      \
    auto d_u32 =                                                               \
        bind<ek::DynamicArray<ek::uint32_array_t<Guide>>>(Module, Scalar);     \
    auto d_i64 =                                                               \
        bind<ek::DynamicArray<ek::int64_array_t<Guide>>>(Module, Scalar);      \
    auto d_u64 =                                                               \
        bind<ek::DynamicArray<ek::uint64_array_t<Guide>>>(Module, Scalar);     \
    auto d_f32 =                                                               \
        bind<ek::DynamicArray<ek::float32_array_t<Guide>>>(Module, Scalar);    \
    auto d_f64 =                                                               \
        bind<ek::DynamicArray<ek::float64_array_t<Guide>>>(Module, Scalar);    \
    (void) d_i32; (void) d_u32; (void) d_i64; (void) d_u64; (void) d_f32;      \
    (void) d_f64; (void) d_b;

#define ENOKI_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, Dim)                 \
    bind<ek::mask_t<ek::Array<ek::float32_array_t<Guide>, Dim>>>(Module,       \
                                                                 Scalar);      \
    bind<ek::Array<ek::int32_array_t<Guide>, Dim>>(Module, Scalar);            \
    bind<ek::Array<ek::uint32_array_t<Guide>, Dim>>(Module, Scalar);           \
    bind<ek::Array<ek::int64_array_t<Guide>, Dim>>(Module, Scalar);            \
    bind<ek::Array<ek::uint64_array_t<Guide>, Dim>>(Module, Scalar);           \
    bind<ek::Array<ek::float32_array_t<Guide>, Dim>>(Module, Scalar);          \
    bind<ek::Array<ek::float64_array_t<Guide>, Dim>>(Module, Scalar);

#define ENOKI_BIND_ARRAY_TYPES(Module, Guide, Scalar)                          \
    ENOKI_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 0)                       \
    ENOKI_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 1)                       \
    ENOKI_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 2)                       \
    ENOKI_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 3)                       \
    ENOKI_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 4)                       \
    ENOKI_BIND_ARRAY_TYPES_DYN(Module, Guide, Scalar)

#define ENOKI_BIND_ARRAY_BASE_1(Module, Guide, Scalar)                         \
    auto a_msk =                                                               \
        bind_type<ek::mask_t<ek::float32_array_t<Guide>>>(Module, Scalar);     \
    auto a_i32 = bind_type<ek::int32_array_t<Guide>>(Module, Scalar);          \
    auto a_u32 = bind_type<ek::uint32_array_t<Guide>>(Module, Scalar);         \
    auto a_i64 = bind_type<ek::int64_array_t<Guide>>(Module, Scalar);          \
    auto a_u64 = bind_type<ek::uint64_array_t<Guide>>(Module, Scalar);         \
    auto a_f32 = bind_type<ek::float32_array_t<Guide>>(Module, Scalar);        \
    auto a_f64 = bind_type<ek::float64_array_t<Guide>>(Module, Scalar);        \
    Module.attr("Int32") = Module.attr("Int");                                 \
    Module.attr("UInt32") = Module.attr("UInt");                               \
    Module.attr("Float32") = Module.attr("Float");

#define ENOKI_BIND_ARRAY_BASE_2(Scalar)                                        \
    bind_full(a_i32, Scalar);                                                  \
    bind_full(a_u32, Scalar);                                                  \
    bind_full(a_i64, Scalar);                                                  \
    bind_full(a_u64, Scalar);                                                  \
    bind_full(a_f32, Scalar);                                                  \
    bind_full(a_f64, Scalar);                                                  \
    bind_full(a_msk, Scalar);
