#pragma once

#include "common.h"
#include <drjit/math.h>
#include <drjit/complex.h>
#include <drjit/matrix.h>
#include <drjit/quaternion.h>
#include <drjit/autodiff.h>
#include <drjit/sh.h>
#include <pybind11/functional.h>

extern py::handle array_base, array_name, array_init, tensor_init, array_configure;

template <typename Array>
auto bind_type(py::module_ &m, bool scalar_mode = false) {
    using Scalar = std::conditional_t<Array::IsMask, bool, dr::scalar_t<Array>>;
    using Value = std::conditional_t<Array::IsMask, dr::mask_t<dr::value_t<Array>>,
                                     dr::value_t<Array>>;
    constexpr VarType Type = dr::var_type_v<Scalar>;

    const char *prefix = "Array";
    if constexpr (dr::is_complex_v<Array>)
        prefix = "Complex";
    else if constexpr (dr::is_quaternion_v<Array>)
        prefix = "Quaternion";
    else if constexpr (dr::is_matrix_v<Array>)
        prefix = "Matrix";
    else if constexpr (dr::is_tensor_v<Array>)
        prefix = "Tensor";

    py::tuple shape;
    if constexpr (Array::Depth == 1)
        shape = py::make_tuple(Array::Size);
    else if constexpr (Array::Depth == 2)
        shape = py::make_tuple(Array::Size, Value::Size);
    else if constexpr (Array::Depth == 3)
        shape = py::make_tuple(Array::Size, Value::Size, Value::Value::Size);
    else if constexpr (Array::Depth == 4)
        shape = py::make_tuple(Array::Size, Value::Size, Value::Value::Size,
                               Value::Value::Value::Size);

    PyTypeObject *value_obj;
    if constexpr (std::is_scalar_v<Value>) {
        if constexpr (std::is_same_v<bool, Value>)
            value_obj = &PyBool_Type;
        else if constexpr (std::is_integral_v<Value>)
            value_obj = &PyLong_Type;
        else
            value_obj = &PyFloat_Type;
    } else {
        auto &reg_types = py::detail::get_internals().registered_types_cpp;
        auto it = reg_types.find(std::type_index(typeid(Value)));
        if (it == reg_types.end())
            dr::drjit_raise("bind_type(): value type was not bound!");
        value_obj = it->second->type;
    }

    py::object type = py::cast(Type),
               name = array_name(prefix, type, shape, scalar_mode);

    auto cls = py::class_<Array>(
        m, PyUnicode_AsUTF8AndSize(name.ptr(), nullptr), array_base);

    array_configure(cls, shape, type, py::handle((PyObject *) value_obj));
    register_implicit_conversions(typeid(Array));

    return cls;
}

template <typename Array>
void bind_basic_methods(py::class_<Array> &cls) {
    using Value = std::conditional_t<Array::IsMask, dr::mask_t<dr::value_t<Array>>,
                                     dr::value_t<Array>>;
    cls.def("entry_", [](const Array &a, size_t i) -> Value { return a.entry(i); })
       .def("set_entry_", [](Array &a, size_t i, const Value &value) {
           a.set_entry(i, value);
       });

    if constexpr (!Array::IsMask && dr::is_dynamic_array_v<Array> &&
                  dr::array_depth_v<Array> == 1 && dr::is_unsigned_v<Array>) {
        cls.def("set_entry_", [](Array &a, size_t i, const
        std::make_signed_t<dr::scalar_t<Value>> &value) {
            a.set_entry(i, value);
        });
    }

    if constexpr (dr::is_dynamic_array_v<Array> ||
                  (!dr::is_jit_v<Array> && !dr::is_mask_v<Array>))
        cls.def("data_", [](const Array &a) {
            return (uintptr_t) a.data();
        });

    if constexpr (dr::is_dynamic_array_v<Array>) {
        cls.def("__len__", &Array::size);
        cls.def("init_", [](Array &a, size_t size) { a.init_(size); });
    }

    if constexpr (dr::array_depth_v<Array> > 1) {
        cls.def(
            "entry_ref_",
            [](Array &a, size_t i) -> Value & { return a.entry(i); },
            py::return_value_policy::reference_internal);
    }
}

template <typename Array>
void bind_other_methods(py::class_<Array> &cls) {
    if constexpr (Array::Size == 3 && Array::IsFloat) {
        cls.def("sh_eval_", [](Array &a, size_t order) {
            std::vector<dr::value_t<Array>> out((order+1)*(order+1));
            sh_eval(a, order, out.data());
            return out;
        });
    }
}

template <typename Array>
void bind_generic_constructor(py::class_<Array> &cls) {
    cls.def(
        "__init__",
        [](py::detail::value_and_holder &v_h, py::args args) {
            Array * a = new Array();
            v_h.value_ptr() = a;
            try {
                array_init(py::handle((PyObject *) v_h.inst), args);
            } catch(const std::exception&) {
                delete a;
                v_h.value_ptr() = nullptr;
                throw;
            }
        }, py::detail::is_new_style_constructor());
}

template <typename Array> auto bind(py::module_ &m, bool scalar_mode = false) {
    auto cls = bind_type<Array>(m, scalar_mode);
    bind_generic_constructor(cls);
    bind_basic_methods(cls);
    bind_other_methods(cls);
    return cls;
};

template <typename Array>
auto bind_full(py::class_<Array> &cls, bool /* scalar_mode */ = false) {
    static_assert(dr::array_depth_v<Array> == 1);
    bind_basic_methods(cls);

    using Scalar = std::conditional_t<Array::IsMask, bool, dr::scalar_t<Array>>;
    using Mask = dr::mask_t<dr::float32_array_t<Array>>;

    cls.def(py::init<Scalar>())
       .def("assign", [](Array &a, const Array &b) {
           if (&a != &b)
               a = b;
       });

    if constexpr (Array::IsFloat) {
        cls.def(py::init([](dr::ssize_t value) { return new Array((Scalar) value); }));
    } else if constexpr (Array::IsIntegral) {
        if constexpr (std::is_unsigned_v<Scalar>)
            cls.def(py::init<std::make_signed_t<Scalar>>());
        else
            cls.def(py::init<std::make_unsigned_t<Scalar>>());
    }

    if constexpr (!Array::IsMask) {
        cls.def(py::init<const dr::  int32_array_t<Array> &>(), py::arg().noconvert());
        cls.def(py::init<const dr:: uint32_array_t<Array> &>(), py::arg().noconvert());
        cls.def(py::init<const dr::  int64_array_t<Array> &>(), py::arg().noconvert());
        cls.def(py::init<const dr:: uint64_array_t<Array> &>(), py::arg().noconvert());
        cls.def(py::init<const dr::float32_array_t<Array> &>(), py::arg().noconvert());
        cls.def(py::init<const dr::float64_array_t<Array> &>(), py::arg().noconvert());
    } else {
        cls.def(py::init<const dr::bool_array_t<Array> &>(), py::arg().noconvert());
    }

#if defined(DRJIT_ENABLE_AUTODIFF)
    if constexpr (Array::IsJIT && !Array::IsFloat && !Array::IsDiff) {
        cls.def(py::init([](const dr::DiffArray<Array> &value) {
            return new Array(dr::detach(value)); }));
    }
#endif

    cls.def("or_",     [](const Array &a, const Array &b) { return a.or_(b); });
    cls.def("and_",    [](const Array &a, const Array &b) { return a.and_(b); });
    cls.def("xor_",    [](const Array &a, const Array &b) { return a.xor_(b); });
    cls.def("andnot_", [](const Array &a, const Array &b) { return a.andnot_(b); });
    cls.def("not_", &Array::not_);

    cls.def("ior_",    [](Array *a, const Array &b) { *a = a->or_(b);  return a;});
    cls.def("iand_",   [](Array *a, const Array &b) { *a = a->and_(b); return a;});
    cls.def("ixor_",   [](Array *a, const Array &b) { *a = a->xor_(b); return a;});

    if constexpr (std::is_same_v<Mask, dr::mask_t<Array>>) {
        cls.def("eq_", &Array::eq_);
        cls.def("neq_", &Array::neq_);
    } else {
        cls.def("eq_", [](const Array &a, const Array &b) -> Mask { return a.eq_(b); });
        cls.def("neq_", [](const Array &a, const Array &b) -> Mask { return a.neq_(b); });
    }

    cls.attr("zero_") = py::cpp_function(&Array::zero_);
    cls.attr("full_") = py::cpp_function(
        [](Scalar v, size_t size) { return dr::full<Array>(v, size); });
    cls.attr("opaque_") = py::cpp_function(
        [](Scalar v, size_t size) { return dr::opaque<Array>(v, size); });

    if constexpr (!Array::IsMask) {
        cls.attr("arange_") = py::cpp_function(&Array::arange_);
        cls.attr("linspace_") = py::cpp_function(&Array::linspace_);
    }

    cls.attr("select_") = py::cpp_function([](const Mask &m, const Array &t,
                                              const Array &f) {
        return Array::select_(static_cast<const dr::mask_t<Array>>(m), t, f);
    });

    if constexpr (Array::IsMask) {
        cls.def("all_", &Array::all_);
        cls.def("any_", &Array::any_);
        cls.def("count_", &Array::count_);
    } else {
        if constexpr (sizeof(Scalar) == 4) {
            cls.def_static("reinterpret_array_",
                           [](const dr::int32_array_t<Array> &a) {
                               return dr::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const dr::uint32_array_t<Array> &a) {
                               return dr::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const dr::float32_array_t<Array> &a) {
                               return dr::reinterpret_array<Array>(a);
                           });
        } else {
            cls.def_static("reinterpret_array_",
                           [](const dr::int64_array_t<Array> &a) {
                               return dr::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const dr::uint64_array_t<Array> &a) {
                               return dr::reinterpret_array<Array>(a);
                           });
            cls.def_static("reinterpret_array_",
                           [](const dr::float64_array_t<Array> &a) {
                               return dr::reinterpret_array<Array>(a);
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

        cls.def("dot_",  &Array::dot_);
        cls.def("min_",  &Array::min_);
        cls.def("max_",  &Array::max_);
        cls.def("sum_",  &Array::sum_);
        cls.def("prod_", &Array::prod_);

        cls.def("and_", [](const Array &a, const Mask &b) {
            return a.and_(static_cast<const dr::mask_t<Array> &>(b));
        });

        cls.def("iand_", [](Array *a, const Mask &b) {
            *a = a->and_(static_cast<const dr::mask_t<Array> &>(b));
            return a;
        });

        cls.def("or_", [](const Array &a, const Mask &b) {
            return a.or_(static_cast<const dr::mask_t<Array> &>(b));
        });

        cls.def("ior_", [](Array *a, const Mask &b) {
            *a = a->or_(static_cast<const dr::mask_t<Array> &>(b));
            return a;
        });

        cls.def("xor_", [](const Array &a, const Mask &b) {
            return a.xor_(static_cast<const dr::mask_t<Array> &>(b));
        });

        cls.def("ixor_", [](Array *a, const Mask &b) {
            *a = a->xor_(static_cast<const dr::mask_t<Array> &>(b));
            return a;
        });

        cls.def("andnot_", [](const Array &a, const Mask &b) {
            return a.andnot_(static_cast<const dr::mask_t<Array> &>(b));
        });

        cls.def("abs_", &Array::abs_);
        cls.def("minimum_", &Array::minimum_);
        cls.def("maximum_", &Array::maximum_);

        if constexpr (std::is_same_v<Mask, dr::mask_t<Array>>) {
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

        cls.def("fma_", &Array::fmadd_);

        cls.def("neg_",  &Array::neg_);
    }

    if constexpr (dr::is_dynamic_v<Array>) {
        using UInt32 = dr::uint32_array_t<Array>;
        cls.def_static("gather_",
                [](const Array &source, const UInt32 &index, const Mask &mask, bool permute) {
                    if (permute)
                        return dr::gather<Array, true>(source, index, mask);
                    else
                        return dr::gather<Array, false>(source, index, mask);
                }, "source"_a, "index"_a, "mask"_a, "permute"_a=false);
        cls.def("scatter_",
                [](const Array &value, Array &target, const UInt32 &index, const Mask &mask, bool permute) {
                    if (permute)
                        dr::scatter<true>(target, value, index, mask);
                    else
                        dr::scatter<false>(target, value, index, mask);
                }, "target"_a.noconvert(), "index"_a, "mask"_a, "permute"_a=false);

        if constexpr (Array::IsMask) {
            cls.def("compress_", [](const Array &source) {
                return dr::compress(source);
            });
        } else {
            cls.def("scatter_reduce_",
                    [](const Array& value, ReduceOp op, Array& target, const UInt32& index, const Mask& mask) {
                        dr::scatter_reduce(op, target, value, index, mask);
                    }, "op"_a, "target"_a.noconvert(), "index"_a, "mask"_a);
        }
    }

    if constexpr (dr::is_jit_v<Array>) {
        cls.def("resize_", [](Array &value, size_t size) { value.resize(size); });
        cls.def("is_literal_", [](Array &value) { return value.is_literal(); });
        cls.def("is_evaluated_", [](Array &value) { return value.is_evaluated(); });

        if constexpr (!Array::IsMask) {
            cls.def("prefix_sum_", &Array::prefix_sum_);
            cls.def("block_sum_", &Array::block_sum_);
        }
    }

    if constexpr (dr::is_dynamic_array_v<Array> || Array::IsDiff)
        cls.def("copy_", [](Array &value) { return value.copy(); });

    if constexpr (Array::IsFloat) {
        cls.def("sqrt_",  &Array::sqrt_);
        cls.def("floor_", &Array::floor_);
        cls.def("ceil_",  &Array::ceil_);
        cls.def("round_", &Array::round_);
        cls.def("trunc_", &Array::trunc_);
        cls.def("rcp_",   &Array::rcp_);
        cls.def("rsqrt_", &Array::rsqrt_);
    } else if constexpr(Array::IsIntegral) {
        cls.def("mulhi_", &Array::mulhi_);
        cls.def("tzcnt_", [](const Array &a) { return dr::tzcnt(a); });
        cls.def("lzcnt_", [](const Array &a) { return dr::lzcnt(a); });
        cls.def("popcnt_", [](const Array &a) { return dr::popcnt(a); });
        cls.def("sl_", [](const Array &a, const Array &b) { return a.sl_(b); });
        cls.def("sr_", [](const Array &a, const Array &b) { return a.sr_(b); });
        cls.def("isl_", [](Array *a, const Array &b) { *a = a->sl_(b); return a; });
        cls.def("isr_", [](Array *a, const Array &b) { *a = a->sr_(b); return a; });
    }

    if constexpr (Array::IsFloat) {
        cls.def("sin_", [](const Array &a) { return dr::sin(a); });
        cls.def("cos_", [](const Array &a) { return dr::cos(a); });
        cls.def("sincos_", [](const Array &a) { return dr::sincos(a); });
        cls.def("tan_", [](const Array &a) { return dr::tan(a); });
        cls.def("csc_", [](const Array &a) { return dr::csc(a); });
        cls.def("sec_", [](const Array &a) { return dr::sec(a); });
        cls.def("cot_", [](const Array &a) { return dr::cot(a); });
        cls.def("asin_", [](const Array &a) { return dr::asin(a); });
        cls.def("acos_", [](const Array &a) { return dr::acos(a); });
        cls.def("atan_", [](const Array &a) { return dr::atan(a); });
        cls.def("atan2_", [](const Array &y, const Array &x) { return dr::atan2(y, x); });
        cls.def("exp_", [](const Array &a) { return dr::exp(a); });
        cls.def("exp2_", [](const Array &a) { return dr::exp2(a); });
        cls.def("log_", [](const Array &a) { return dr::log(a); });
        cls.def("log2_", [](const Array &a) { return dr::log2(a); });
        cls.def("power_", [](const Array &x, Scalar y) { return dr::pow(x, y); });
        cls.def("power_", [](const Array &x, const Array &y) { return dr::pow(x, y); });
        cls.def("sinh_", [](const Array &a) { return dr::sinh(a); });
        cls.def("cosh_", [](const Array &a) { return dr::cosh(a); });
        cls.def("sincosh_", [](const Array &a) { return dr::sincosh(a); });
        cls.def("tanh_", [](const Array &a) { return dr::tanh(a); });
        cls.def("asinh_", [](const Array &a) { return dr::asinh(a); });
        cls.def("acosh_", [](const Array &a) { return dr::acosh(a); });
        cls.def("atanh_", [](const Array &a) { return dr::atanh(a); });
        cls.def("cbrt_", [](const Array &a) { return dr::cbrt(a); });
        cls.def("erf_", [](const Array &a) { return dr::erf(a); });
        cls.def("erfinv_", [](const Array &a) { return dr::erfinv(a); });
        cls.def("lgamma_", [](const Array &a) { return dr::lgamma(a); });
        cls.def("tgamma_", [](const Array &a) { return dr::tgamma(a); });
    }

    if constexpr (Array::IsJIT || Array::IsDiff) {
        cls.def("set_label_", [](Array &a, const char *name) { a.set_label_(name); });
        cls.def("label_", [](const Array &a) { return a.label_(); });
    }

    if constexpr (!dr::is_mask_v<Array> || dr::is_dynamic_v<Array>) {
        cls.def_static("load_", [](uintptr_t ptr, size_t size) {
            return drjit::load<Array>((const void *) ptr, size);
        });
    }

    if constexpr (dr::is_jit_v<Array>) {
        cls.def_static("map_", [](uintptr_t ptr, size_t size, std::function<void (void)> callback) {
            Array result = Array::map_((void *) ptr, size, false);
            if (callback) {
                std::function<void (void)> *func = new std::function<void (void)>(std::move(callback));
                jit_var_set_callback(result.index(), [](uint32_t /* i */, int free, void *arg) {
                    if (free) {
                        std::function<void(void)> *func2 = (std::function<void(void)> *) arg;
                        (*func2)();
                        delete func2;
                    }
                }, func);
            }
            return result;
        }, "ptr"_a, "size"_a, "callback"_a = py::none());
    }

    if constexpr (Array::IsJIT)
        cls.def("migrate_", &Array::migrate_);

    if constexpr (Array::IsJIT) {
        cls.def_property_readonly("index", &Array::index);
        cls.def("set_index_",
                [](Array &a, uint32_t index) { *a.index_ptr() = index; });
    }

    if constexpr (Array::IsDiff) {
        cls.def_property_readonly("index_ad", &Array::index_ad);
        cls.def("set_index_ad_",
                [](Array &a, int32_t index) { *a.index_ad_ptr() = index; });
    }

    if constexpr (Array::IsDiff) {
        cls.def(py::init<dr::detached_t<Array>>(), py::arg().noconvert());
        cls.def("detach_", [](const Array &a) { return dr::detach(a); });
        cls.def("detach_ref_", py::overload_cast<>(&Array::detach_),
                py::return_value_policy::reference_internal);

        if constexpr (Array::IsFloat) {
            cls.def("grad_", [](const Array &a) { return a.grad_(); });
            cls.def("set_grad_", [](Array &a, dr::detached_t<Array> &value) { a.set_grad_(value); });
            cls.def("accum_grad_", [](Array &a, dr::detached_t<Array> &value) { a.accum_grad_(value); });
            cls.def("set_grad_enabled_", &Array::set_grad_enabled_);
            cls.def("grad_enabled_", &Array::grad_enabled_);
            cls.def("enqueue_", &Array::enqueue_);

            cls.def_static("traverse_", &Array::traverse_,
                           py::call_guard<py::gil_scoped_release>());

            cls.def_static("create_", [](uint32_t index,
                                         const dr::detached_t<Array> &value) {
                dr::detail::ad_inc_ref_impl<dr::detached_t<Array>>(index);
                return Array::create(index, dr::detached_t<Array>(value));
            });

            cls.def_static(
                "scope_enter_",
                [](dr::detail::ADScope type, const std::vector<uint32_t> &indices) {
                    dr::detail::ad_scope_enter<dr::detached_t<Array>>(
                        type, indices.size(), indices.data());
                });

            cls.def_static("scope_leave_", [](bool process_postoned) {
                dr::detail::ad_scope_leave<dr::detached_t<Array>>(process_postoned);
            });
        }
    }

    bind_generic_constructor(cls);

    return cls;
}

struct CustomOp : dr::detail::DiffCallback {
    CustomOp(py::handle handle) : m_handle(handle) {
        m_handle.inc_ref();
    }

    virtual void forward() override {
        py::gil_scoped_acquire gsa;
        m_handle.attr("forward")();
    }

    virtual void backward() override {
        py::gil_scoped_acquire gsa;
        m_handle.attr("backward")();
    }

    ~CustomOp() {
        py::gil_scoped_acquire gsa;
        m_handle.dec_ref();
    }

    py::handle m_handle;
};

template <typename T>
void bind_ad_details(py::class_<dr::DiffArray<T>> &cls) {
    cls.def_static(
        "ad_add_edge_",
        [](int32_t src_index, int32_t dst_index, py::handle cb) {
            dr::detail::ad_add_edge<T>(
                src_index, dst_index,
                cb.is_none() ? nullptr : new CustomOp(cb));
        },
        "src_index"_a, "dst_index"_a, "cb"_a = py::none());

    cls.def("ad_dec_ref_", [](dr::DiffArray<T> &v) {
        dr::detail::ad_dec_ref<T>(v.index_ad());
    });
    cls.def("ad_inc_ref_", [](dr::DiffArray<T> &v) {
        dr::detail::ad_inc_ref<T>(v.index_ad());
    });

    cls.def("ad_implicit_", []() {
        return dr::detail::ad_implicit<T>();
    });
    cls.def("ad_extract_implicit_", [](size_t snapshot) {
        std::vector<uint32_t> implicit_in(dr::detail::ad_implicit<T>() - snapshot, 0);
        dr::detail::ad_extract_implicit<T>(snapshot, implicit_in.data());
        return implicit_in;
    });
    cls.def("ad_enqueue_implicit_", [](size_t snapshot) {
        return dr::detail::ad_enqueue_implicit<T>(snapshot);
    });
    cls.def("ad_dequeue_implicit_", [](size_t snapshot) {
        return dr::detail::ad_dequeue_implicit<T>(snapshot);
    });
}

#define DRJIT_BIND_ARRAY_TYPES_DYN(Module, Guide, Scalar)                      \
    auto d_b = bind<dr::mask_t<dr::DynamicArray<dr::float32_array_t<Guide>>>>( \
        Module, Scalar);                                                       \
    auto d_i32 =                                                               \
        bind<dr::DynamicArray<dr::int32_array_t<Guide>>>(Module, Scalar);      \
    auto d_u32 =                                                               \
        bind<dr::DynamicArray<dr::uint32_array_t<Guide>>>(Module, Scalar);     \
    auto d_i64 =                                                               \
        bind<dr::DynamicArray<dr::int64_array_t<Guide>>>(Module, Scalar);      \
    auto d_u64 =                                                               \
        bind<dr::DynamicArray<dr::uint64_array_t<Guide>>>(Module, Scalar);     \
    auto d_f32 =                                                               \
        bind<dr::DynamicArray<dr::float32_array_t<Guide>>>(Module, Scalar);    \
    auto d_f64 =                                                               \
        bind<dr::DynamicArray<dr::float64_array_t<Guide>>>(Module, Scalar);    \
    (void) d_i32; (void) d_u32; (void) d_i64; (void) d_u64; (void) d_f32;      \
    (void) d_f64; (void) d_b;

#define DRJIT_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, Dim)                 \
    bind<dr::mask_t<dr::Array<dr::float32_array_t<Guide>, Dim>>>(Module,       \
                                                                 Scalar);      \
    bind<dr::Array<dr::int32_array_t<Guide>, Dim>>(Module, Scalar);            \
    bind<dr::Array<dr::uint32_array_t<Guide>, Dim>>(Module, Scalar);           \
    bind<dr::Array<dr::int64_array_t<Guide>, Dim>>(Module, Scalar);            \
    bind<dr::Array<dr::uint64_array_t<Guide>, Dim>>(Module, Scalar);           \
    bind<dr::Array<dr::float32_array_t<Guide>, Dim>>(Module, Scalar);          \
    bind<dr::Array<dr::float64_array_t<Guide>, Dim>>(Module, Scalar);

#define DRJIT_BIND_COMPLEX_TYPES(Module, Guide, Scalar)                        \
    bind<dr::Complex<dr::float32_array_t<Guide>>>(Module, Scalar);             \
    bind<dr::Complex<dr::float64_array_t<Guide>>>(Module, Scalar);             \

#define DRJIT_BIND_QUATERNION_TYPES(Module, Guide, Scalar)                     \
    bind<dr::Quaternion<dr::float32_array_t<Guide>>>(Module, Scalar);          \
    bind<dr::Quaternion<dr::float64_array_t<Guide>>>(Module, Scalar);

#define DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide, Scalar, Dim)                \
    bind<dr::Matrix<dr::int32_array_t<Guide>, Dim>>(Module, Scalar);           \
    bind<dr::Matrix<dr::uint32_array_t<Guide>, Dim>>(Module, Scalar);          \
    bind<dr::Matrix<dr::float32_array_t<Guide>, Dim>>(Module, Scalar);         \
    bind<dr::Matrix<dr::float64_array_t<Guide>, Dim>>(Module, Scalar);

#define DRJIT_BIND_ARRAY_TYPES(Module, Guide, Scalar)                          \
    DRJIT_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 0)                       \
    DRJIT_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 1)                       \
    DRJIT_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 2)                       \
    DRJIT_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 3)                       \
    DRJIT_BIND_ARRAY_TYPES_DIM(Module, Guide, Scalar, 4)                       \
    DRJIT_BIND_ARRAY_TYPES_DYN(Module, Guide, Scalar)                          \
    bind<dr::mask_t<dr::Array<dr::Array<Guide, 2>, 2>>>(Module, Scalar);       \
    bind<dr::mask_t<dr::Array<dr::Array<Guide, 3>, 3>>>(Module, Scalar);       \
    bind<dr::mask_t<dr::Array<dr::Array<Guide, 4>, 4>>>(Module, Scalar);       \
    DRJIT_BIND_COMPLEX_TYPES(Module, Guide, Scalar)                            \
    DRJIT_BIND_QUATERNION_TYPES(Module, Guide, Scalar)                         \
    DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide, Scalar, 2)                      \
    DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide, Scalar, 3)                      \
    DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide, Scalar, 4)                      \
                                                                               \
    using Guide1 = dr::Array<Guide, 1>;                                        \
    using Guide3 = dr::Array<Guide, 3>;                                        \
    using Guide4 = dr::Array<Guide, 4>;                                        \
    bind<dr::mask_t<dr::Array<dr::float32_array_t<Guide1>, 2>>>(Module,        \
                                                                Scalar);       \
    bind<dr::mask_t<dr::Array<dr::float32_array_t<Guide3>, 2>>>(Module,        \
                                                                Scalar);       \
    bind<dr::mask_t<dr::Array<dr::float32_array_t<Guide4>, 2>>>(Module,        \
                                                                Scalar);       \
    bind<dr::mask_t<dr::Array<Guide1, 4>>>(Module, Scalar);                    \
    bind<dr::mask_t<dr::Array<Guide3, 4>>>(Module, Scalar);                    \
    bind<dr::mask_t<dr::Array<dr::Array<Guide1, 4>, 4>>>(Module, Scalar);      \
    bind<dr::mask_t<dr::Array<dr::Array<Guide3, 4>, 4>>>(Module, Scalar);      \
    bind<dr::mask_t<dr::Array<dr::Array<Guide4, 4>, 4>>>(Module, Scalar);      \
    bind<dr::Array<dr::int32_array_t<Guide1>,   4>>(Module, Scalar);           \
    bind<dr::Array<dr::uint32_array_t<Guide1>,  4>>(Module, Scalar);           \
    bind<dr::Array<dr::float32_array_t<Guide1>, 4>>(Module, Scalar);           \
    bind<dr::Array<dr::float64_array_t<Guide1>, 4>>(Module, Scalar);           \
    bind<dr::Array<dr::int32_array_t<Guide3>,   4>>(Module, Scalar);           \
    bind<dr::Array<dr::uint32_array_t<Guide3>,  4>>(Module, Scalar);           \
    bind<dr::Array<dr::float32_array_t<Guide3>, 4>>(Module, Scalar);           \
    bind<dr::Array<dr::float64_array_t<Guide3>, 4>>(Module, Scalar);           \
    bind<dr::Array<dr::int32_array_t<Guide4>,   4>>(Module, Scalar);           \
    bind<dr::Array<dr::uint32_array_t<Guide4>,  4>>(Module, Scalar);           \
    bind<dr::Array<dr::float32_array_t<Guide4>, 4>>(Module, Scalar);           \
    bind<dr::Array<dr::float64_array_t<Guide4>, 4>>(Module, Scalar);           \
    DRJIT_BIND_COMPLEX_TYPES(Module, Guide1, Scalar)                           \
    DRJIT_BIND_COMPLEX_TYPES(Module, Guide3, Scalar)                           \
    DRJIT_BIND_COMPLEX_TYPES(Module, Guide4, Scalar)                           \
    DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide1, Scalar, 4)                     \
    DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide3, Scalar, 4)                     \
    DRJIT_BIND_MATRIX_TYPES_DIM(Module, Guide4, Scalar, 4)                     \

#define DRJIT_BIND_ARRAY_BASE(Module, Guide, Scalar)                           \
    auto a_msk =                                                               \
        bind_type<dr::mask_t<dr::float32_array_t<Guide>>>(Module, Scalar);     \
    auto a_i32 = bind_type<dr::int32_array_t<Guide>>(Module, Scalar);          \
    auto a_u32 = bind_type<dr::uint32_array_t<Guide>>(Module, Scalar);         \
    auto a_i64 = bind_type<dr::int64_array_t<Guide>>(Module, Scalar);          \
    auto a_u64 = bind_type<dr::uint64_array_t<Guide>>(Module, Scalar);         \
    auto a_f32 = bind_type<dr::float32_array_t<Guide>>(Module, Scalar);        \
    auto a_f64 = bind_type<dr::float64_array_t<Guide>>(Module, Scalar);        \
    Module.attr("Int32") = Module.attr("Int");                                 \
    Module.attr("UInt32") = Module.attr("UInt");                               \
    Module.attr("Float32") = Module.attr("Float");                             \
    bind_full(a_i32, Scalar);                                                  \
    bind_full(a_u32, Scalar);                                                  \
    bind_full(a_i64, Scalar);                                                  \
    bind_full(a_u64, Scalar);                                                  \
    bind_full(a_f32, Scalar);                                                  \
    bind_full(a_f64, Scalar);                                                  \
    bind_full(a_msk, Scalar);
