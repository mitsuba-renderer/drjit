#pragma once

#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit/dynamic.h>
#include <drjit/tensor.h>
#include <drjit/math.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/detail/nb_list.h>

#if defined(DRJIT_PYTHON_BUILD)
#  define DRJIT_PYTHON_EXPORT DRJIT_EXPORT
#else
#  define DRJIT_PYTHON_EXPORT DRJIT_IMPORT
#endif

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

using array_unop = void (*) (const void *, void *);
using array_unop_2 = void (*) (const void *, void *, void *);
using array_binop = void (*) (const void *, const void *, void *);
using array_ternop = void (*) (const void *, const void *, const void *, void *);
using array_richcmp = void (*) (const void *, const void *, int, void *);
using array_reduce_mask = void (*) (const void *, void *);
using array_index = uint32_t (*) (const void *);
using array_set_index = void (*) (void *, uint32_t);
using array_full = void (*) (nanobind::handle, size_t, void *);
using array_empty = void (*) (size_t, void *);
using array_counter = void (*) (uint32_t size, void *);
using array_cast = int (*) (const void *, VarType, void *);
using array_ad_create = void (*) (void *, uint32_t, void *);
using array_set_label = void (*) (void *, const char *);

using array_set_bool = void (*) (void *, bool);
using array_get_bool = bool (*) (const void *);
using array_set_grad = void (*) (void *, const void *);

struct array_metadata {
    uint16_t is_vector     : 1;
    uint16_t is_complex    : 1;
    uint16_t is_quaternion : 1;
    uint16_t is_matrix     : 1;
    uint16_t is_tensor     : 1;
    uint16_t is_diff       : 1;
    uint16_t is_llvm       : 1;
    uint16_t is_cuda       : 1;
    uint16_t is_valid      : 1;
    uint16_t type          : 4;
    uint16_t ndim          : 3;
    uint8_t tsize_rel;  // type size as multiple of 'talign'
    uint8_t talign;     // type alignment
    uint8_t shape[4];
};

struct array_supplement {
    array_metadata meta;
    PyTypeObject *value;
    PyTypeObject *mask;
    PyTypeObject *array;

    size_t (*len)(const void *) noexcept;
    void (*init)(void *, size_t);
    void *(*ptr)(void *);
    dr_vector<size_t> & (*op_tensor_shape)(void *) noexcept;
    PyObject *(*op_tensor_array)(PyObject *) noexcept;
    array_cast op_cast;

    array_full op_full;
    array_empty op_empty;
    array_counter op_counter;
    array_binop op_add;
    array_binop op_subtract;
    array_binop op_multiply;
    array_binop op_remainder;
    array_binop op_floor_divide;
    array_binop op_true_divide;
    array_binop op_and;
    array_binop op_or;
    array_binop op_xor;
    array_binop op_lshift;
    array_binop op_rshift;
    array_unop op_negative;
    array_unop op_invert;
    array_unop op_absolute;
    array_reduce_mask op_all;
    array_reduce_mask op_any;
    array_richcmp op_richcmp;
    array_ternop op_fma;
    array_ternop op_select;
    array_index op_index, op_index_ad;
    array_set_index op_set_index;
    array_ternop op_gather;
    array_ternop op_scatter;

    array_set_label op_set_label;

    array_unop op_sqrt, op_cbrt;
    array_unop op_sin, op_cos, op_tan;
    array_unop op_sinh, op_cosh, op_tanh;
    array_unop op_asin, op_acos, op_atan;
    array_unop op_asinh, op_acosh, op_atanh;
    array_unop op_exp, op_exp2, op_log, op_log2;
    array_unop op_floor, op_ceil, op_round, op_trunc;
    array_unop op_rcp, op_rsqrt;
    array_binop op_min, op_max, op_atan2, op_ldexp;
    array_unop_2 op_sincos, op_sincosh, op_frexp;
    array_unop op_detach;
    array_ad_create op_ad_create;

    array_set_bool op_set_grad_enabled;
    array_get_bool op_grad_enabled;
    array_unop op_grad;
    array_unop op_detach;
    array_set_grad op_set_grad, op_accum_grad;
    void (*op_enqueue) (const void *, const void *);
};

static_assert(sizeof(array_metadata) == 8);

extern DRJIT_PYTHON_EXPORT const nanobind::handle array_get(array_metadata meta);

extern DRJIT_PYTHON_EXPORT nanobind::handle
bind(const char *name, array_supplement &supp, const std::type_info *type,
     const std::type_info *value_type, void (*copy)(void *, const void *),
     void (*move)(void *, void *) noexcept, void (*destruct)(void *) noexcept,
     void (*type_callback)(PyTypeObject *) noexcept) noexcept;

extern DRJIT_PYTHON_EXPORT int array_init(PyObject *self, PyObject *args,
                                          PyObject *kwds);
extern DRJIT_PYTHON_EXPORT int tensor_init(PyObject *self, PyObject *args,
                                           PyObject *kwds);

template <typename T>
constexpr uint8_t size_or_zero_v = std::is_scalar_v<T> ? 0 : (uint8_t) array_size_v<T>;

template <typename T> void type_callback_array(PyTypeObject *tp) noexcept {
    namespace nb = nanobind;
    using Value = std::decay_t<decltype(std::declval<T>().entry(0))>;

    tp->tp_init = array_init;

    PySequenceMethods *sm = tp->tp_as_sequence;
    sm->sq_item = [](PyObject *o, Py_ssize_t i_) noexcept -> PyObject * {
        T *inst = nb::inst_ptr<T>(o);
        size_t i = (size_t) i_, size = inst->size();

        if (size == 1) // Broadcast
            i = 0;

        PyObject *result = nullptr;
        if (i < size) {
            nb::detail::cleanup_list cleanup(o);
            result = nb::detail::make_caster<Value>::from_cpp(
                         inst->entry(i),
                         nb::rv_policy::reference_internal,
                         &cleanup).ptr();
            cleanup.release();
        } else {
            PyErr_Format(
                PyExc_IndexError,
                "%s.__getitem__(): entry %zu is out of bounds (the array is of size %zu).",
                Py_TYPE(o)->tp_name, i, size);
        }
        return result;
    };

    sm->sq_ass_item = [](PyObject *o, Py_ssize_t i_, PyObject *value) noexcept -> int {
        T *inst = nb::inst_ptr<T>(o);
        size_t i = (size_t) i_, size = inst->size();
        if (i < size) {
            nb::detail::cleanup_list cleanup(o);
            nb::detail::make_caster<Value> in;
            bool success = in.from_python(
                value, (uint8_t) nb::detail::cast_flags::convert, &cleanup);
            if (success)
                inst->set_entry(i, in.operator Value & ());
            cleanup.release();

            if (success) {
                return 0;
            } else {
                PyErr_Format(
                    PyExc_TypeError,
                    "%s.__setitem__(): could not initialize element with a value of type '%s'.",
                    Py_TYPE(o)->tp_name, Py_TYPE(value)->tp_name);
                return -1;
            }
        } else {
            PyErr_Format(
                PyExc_IndexError,
                "%s.__setitem__(): entry %zu is out of bounds (the array is of size %zu).",
                Py_TYPE(o)->tp_name, i, size);
            return -1;
        }
    };
}

template <typename T> void type_callback_tensor(PyTypeObject *tp) noexcept {
    tp->tp_init = tensor_init;
}

NAMESPACE_END(detail)


template <typename T> nanobind::class_<T> bind(const char *name = nullptr) {
    namespace nb = nanobind;

    static_assert(
        std::is_copy_constructible_v<T> &&
        std::is_move_constructible_v<T> &&
        std::is_destructible_v<T>,
        "drjit::bind(): type must be copy/move constructible and destructible!"
    );

    constexpr uint8_t RelSize = (uint8_t) (sizeof(T) / alignof(T));

    static_assert(alignof(T) <= 0xFF && RelSize * alignof(T) == sizeof(T),
                  "drjit::bind(): type is too large!");

    detail::array_supplement s;
    memset(&s, 0, sizeof(detail::array_supplement));

    s.meta.is_vector = T::IsVector;
    s.meta.is_complex = T::IsComplex;
    s.meta.is_quaternion = T::IsQuaternion;
    s.meta.is_matrix = T::IsMatrix;
    s.meta.is_tensor = T::IsTensor;
    s.meta.is_diff = T::IsDiff;
    s.meta.is_llvm = T::IsLLVM;
    s.meta.is_cuda = T::IsCUDA;
    s.meta.is_valid = 1;

    if (T::IsMask)
        s.meta.type = (uint16_t) VarType::Bool;
    else
        s.meta.type = (uint16_t) var_type_v<scalar_t<T>>;

    s.meta.tsize_rel = (uint8_t) RelSize;
    s.meta.talign = (uint8_t) alignof(T);

    if constexpr (!T::IsTensor) {
        s.meta.ndim = (uint16_t) array_depth_v<T>;
        s.meta.shape[0] = detail::size_or_zero_v<T>;
        s.meta.shape[1] = detail::size_or_zero_v<value_t<T>>;
        s.meta.shape[2] = detail::size_or_zero_v<value_t<value_t<T>>>;
        s.meta.shape[3] = detail::size_or_zero_v<value_t<value_t<value_t<T>>>>;
    }

    void (*copy)(void *, const void *) = nullptr;
    void (*move)(void *, void *) noexcept = nullptr;
    void (*destruct)(void *) noexcept = nullptr;
    void (*type_callback)(PyTypeObject *) noexcept = nullptr;

    if constexpr (!std::is_trivially_copy_constructible_v<T>)
        copy = nb::detail::wrap_copy<T>;

    if constexpr (!std::is_trivially_move_constructible_v<T>)
        move = nb::detail::wrap_move<T>;

    if constexpr (!std::is_trivially_destructible_v<T>)
        destruct = nb::detail::wrap_destruct<T>;

    using Value = typename T::Value;

    if constexpr (!T::IsTensor)
        type_callback = detail::type_callback_array<T>;
    else
        type_callback = detail::type_callback_tensor<T>;

    return nb::steal<nb::class_<T>>(detail::bind(
        name, s, &typeid(T), std::is_scalar_v<Value> ? nullptr : &typeid(Value),
        copy, move, destruct, type_callback));
}

template <typename T> nanobind::class_<T> bind_array(const char *name = nullptr) {
    namespace nb = nanobind;
    using Mask = mask_t<T>;

    nb::class_<T> tp = bind<T>(name);

    detail::array_supplement &s =
        nb::type_supplement<detail::array_supplement>(tp);

    if constexpr (T::Size == Dynamic) {
        s.len = [](const void *a) noexcept -> size_t {
            return ((const T *) a)->size();
        };

        s.init = [](void *a, size_t size) {
            ((T *) a)->init_(size);
        };

        if constexpr (T::Depth == 1 && !T::IsJIT)
            s.ptr = [](void *a) -> void * { return ((T *) a)->data(); };
    }

    if constexpr (T::Depth == 1 && T::Size == Dynamic) {
        s.op_full = [](nb::handle a, size_t b, void *c) {
            new ((T *) c) T(full<T>(nb::cast<scalar_t<T>>(a), b));
        };

        s.op_empty = [](size_t b, void *c) {
            new ((T *) c) T(empty<T>(b));
        };

        if constexpr (std::is_same_v<scalar_t<T>, uint32_t>) {
            s.op_counter = [](uint32_t size, void *a) {
                new ((T *) a) T(T::counter(size));
            };
        }

        s.op_select = [](const void *a, const void *b, const void *c, void *d) {
            new ((T *) d) T(select(*(const mask_t<T> *) a, *(const T *) b, *(const T *) c));
        };

        using UInt32 = uint32_array_t<T>;

        s.op_gather = [](const void *a, const void *b, const void *c, void *d) {
            new ((T *) d) T(gather<T>(*(const T *) a,
                                      *(const UInt32 *) b,
                                      *(const Mask *) c));
        };

        s.op_scatter = [](const void *a, const void *b, const void *c,
                          void *d) {
            scatter(*(T *) d, *(const T *) a, *(const UInt32 *) b,
                    *(const Mask *) c);
        };

        if constexpr (T::IsArithmetic) {
            using Int32  = int32_array_t<T>;
            using UInt64 = uint64_array_t<T>;
            using Int64  = int64_array_t<T>;
            using Float32 = float32_array_t<T>;
            using Float64 = float64_array_t<T>;

            s.op_cast = [](const void *a, VarType vt, void *b) {
                switch (vt) {
                    case VarType::Int32:   new (b) T(*(const Int32 *)   a); break;
                    case VarType::UInt32:  new (b) T(*(const UInt32 *)  a); break;
                    case VarType::Int64:   new (b) T(*(const Int64 *)   a); break;
                    case VarType::UInt64:  new (b) T(*(const UInt64 *)  a); break;
                    case VarType::Float32: new (b) T(*(const Float32 *) a); break;
                    case VarType::Float64: new (b) T(*(const Float64 *) a); break;
                    default: return -1;
                }
                return 0;
            };

            s.op_add = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a + *(const T *) b);
            };

            s.op_subtract = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a - *(const T *) b);
            };

            s.op_multiply = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a * *(const T *) b);
            };

            s.op_min = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(drjit::min(*(const T *) a, *(const T *) b));
            };

            s.op_max = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(drjit::max(*(const T *) a, *(const T *) b));
            };

            s.op_fma = [](const void *a, const void *b, const void *c, void *d) {
                new ((T *) d) T(fmadd(*(const T *) a, *(const T *) b, *(const T *) c));
            };

            if constexpr (std::is_signed_v<scalar_t<T>>) {
                s.op_absolute = [](const void *a, void *b) {
                    new ((T *) b) T(((const T *) a)->abs_());
                };
                s.op_negative = [](const void *a, void *b) {
                    new ((T *) b) T(-*(const T *) a);
                };
            }
        }

        if constexpr (T::IsIntegral) {
            s.op_remainder = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a % *(const T *) b);
            };

            s.op_floor_divide = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a / *(const T *) b);
            };

            s.op_lshift = [](const void *a, const void *b, void *c) {
                new ((T *) c) T((*(const T *) a) << (*(const T *) b));
            };

            s.op_rshift = [](const void *a, const void *b, void *c) {
                new ((T *) c) T((*(const T *) a) >> (*(const T *) b));
            };
        } else {
            s.op_true_divide = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a / *(const T *) b);
            };
        }

        if constexpr (T::IsIntegral || T::IsMask) {
            s.op_and = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a & *(const T *) b);
            };

            s.op_or = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(*(const T *) a | *(const T *) b);
            };

            s.op_xor = [](const void *a, const void *b, void *c) {
                if constexpr (T::IsIntegral)
                    new ((T *) c) T(*(const T *) a ^ *(const T *) b);
                else
                    new ((T *) c) T(neq(*(const T *) a, *(const T *) b));
            };

            s.op_invert = [](const void *a, void *b) {
                if constexpr (T::IsIntegral)
                    new ((T *) b) T(~*(const T *) a);
                else
                    new ((T *) b) T(!*(const T *) a);
            };
        }

        if constexpr (T::IsArithmetic || T::IsMask) {
            s.op_richcmp = [](const void *a, const void *b, int op, void *c) {
                switch (op) {
                    case Py_LT:
                        new ((Mask *) c) Mask((*(const T *) a) < (*(const T *) b));
                        break;
                    case Py_LE:
                        new ((Mask *) c) Mask(*(const T *) a <= *(const T *) b);
                        break;
                    case Py_GT:
                        new ((Mask *) c) Mask(*(const T *) a > *(const T *) b);
                        break;
                    case Py_GE:
                        new ((Mask *) c) Mask(*(const T *) a >= *(const T *) b);
                        break;
                    case Py_EQ:
                        new ((Mask *) c) Mask(eq(*(const T *) a, *(const T *) b));
                        break;
                    case Py_NE:
                        new ((Mask *) c) Mask(neq(*(const T *) a, *(const T *) b));
                        break;
                }
            };
        }

        if constexpr (T::IsFloat) {
            s.op_sqrt  = [](const void *a, void *b) { new ((T *) b) T(sqrt(*(const T *) a)); };
            s.op_cbrt  = [](const void *a, void *b) { new ((T *) b) T(cbrt(*(const T *) a)); };
            s.op_sin   = [](const void *a, void *b) { new ((T *) b) T(sin(*(const T *) a)); };
            s.op_cos   = [](const void *a, void *b) { new ((T *) b) T(cos(*(const T *) a)); };
            s.op_tan   = [](const void *a, void *b) { new ((T *) b) T(tan(*(const T *) a)); };
            s.op_asin  = [](const void *a, void *b) { new ((T *) b) T(asin(*(const T *) a)); };
            s.op_acos  = [](const void *a, void *b) { new ((T *) b) T(acos(*(const T *) a)); };
            s.op_atan  = [](const void *a, void *b) { new ((T *) b) T(atan(*(const T *) a)); };
            s.op_sinh  = [](const void *a, void *b) { new ((T *) b) T(sinh(*(const T *) a)); };
            s.op_cosh  = [](const void *a, void *b) { new ((T *) b) T(cosh(*(const T *) a)); };
            s.op_tanh  = [](const void *a, void *b) { new ((T *) b) T(tanh(*(const T *) a)); };
            s.op_asinh = [](const void *a, void *b) { new ((T *) b) T(asinh(*(const T *) a)); };
            s.op_acosh = [](const void *a, void *b) { new ((T *) b) T(acosh(*(const T *) a)); };
            s.op_atanh = [](const void *a, void *b) { new ((T *) b) T(atanh(*(const T *) a)); };
            s.op_exp   = [](const void *a, void *b) { new ((T *) b) T(exp(*(const T *) a)); };
            s.op_exp2  = [](const void *a, void *b) { new ((T *) b) T(exp2(*(const T *) a)); };
            s.op_log   = [](const void *a, void *b) { new ((T *) b) T(log(*(const T *) a)); };
            s.op_log2  = [](const void *a, void *b) { new ((T *) b) T(log2(*(const T *) a)); };
            s.op_floor = [](const void *a, void *b) { new ((T *) b) T(floor(*(const T *) a)); };
            s.op_ceil  = [](const void *a, void *b) { new ((T *) b) T(ceil(*(const T *) a)); };
            s.op_round = [](const void *a, void *b) { new ((T *) b) T(round(*(const T *) a)); };
            s.op_trunc = [](const void *a, void *b) { new ((T *) b) T(trunc(*(const T *) a)); };
            s.op_rcp   = [](const void *a, void *b) { new ((T *) b) T(rcp(*(const T *) a)); };
            s.op_rsqrt = [](const void *a, void *b) { new ((T *) b) T(rsqrt(*(const T *) a)); };
            s.op_ldexp = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(drjit::ldexp(*(const T *) a, *(const T *) b));
            };
            s.op_atan2 = [](const void *a, const void *b, void *c) {
                new ((T *) c) T(drjit::atan2(*(const T *) a, *(const T *) b));
            };
            s.op_sincos = [](const void *a, void *b, void *c) {
                auto [b_, c_] = sincos(*(const T *) a);
                new ((T *) b) T(b_);
                new ((T *) c) T(c_);
            };
            s.op_sincosh = [](const void *a, void *b, void *c) {
                auto [b_, c_] = sincosh(*(const T *) a);
                new ((T *) b) T(b_);
                new ((T *) c) T(c_);
            };
            s.op_frexp = [](const void *a, void *b, void *c) {
                auto [b_, c_] = frexp(*(const T *) a);
                new ((T *) b) T(b_);
                new ((T *) c) T(c_);
            };
        }
    } else {
        // Default implementations of everything
        const detail::array_unop default_unop =
            (detail::array_unop) uintptr_t(1);
        const detail::array_unop_2 default_unop_2 =
            (detail::array_unop_2) uintptr_t(1);
        const detail::array_binop default_binop =
            (detail::array_binop) uintptr_t(1);
        const detail::array_ternop default_ternop =
            (detail::array_ternop) uintptr_t(1);
        (void) default_unop; (void) default_unop_2;
        (void) default_binop; (void) default_ternop;

        s.op_select = default_ternop;

        if constexpr (T::IsArithmetic) {
            s.op_add = default_binop;
            s.op_subtract = default_binop;
            s.op_multiply = default_binop;
            s.op_min = default_binop;
            s.op_max = default_binop;
            s.op_fma = default_ternop;

            if constexpr (std::is_signed_v<scalar_t<T>>) {
                s.op_absolute = default_unop;
                s.op_negative = default_unop;
            }
        }

        if constexpr (T::IsIntegral) {
            s.op_remainder = default_binop;
            s.op_floor_divide = default_binop;
            s.op_lshift = default_binop;
            s.op_rshift = default_binop;
        } else {
            s.op_true_divide = default_binop;
        }

        if constexpr (T::IsIntegral || T::IsMask) {
            s.op_and = default_binop;
            s.op_or = default_binop;
            s.op_xor = default_binop;
            s.op_invert = default_unop;
        }

        if constexpr (T::IsArithmetic || T::IsMask)
            s.op_richcmp = (detail::array_richcmp) uintptr_t(1);

        if constexpr (T::IsFloat) {
            s.op_sqrt  = default_unop;
            s.op_cbrt  = default_unop;
            s.op_sin   = default_unop;
            s.op_cos   = default_unop;
            s.op_tan   = default_unop;
            s.op_asin  = default_unop;
            s.op_acos  = default_unop;
            s.op_atan  = default_unop;
            s.op_sinh  = default_unop;
            s.op_cosh  = default_unop;
            s.op_tanh  = default_unop;
            s.op_asinh = default_unop;
            s.op_acosh = default_unop;
            s.op_atanh = default_unop;
            s.op_exp   = default_unop;
            s.op_exp2  = default_unop;
            s.op_log   = default_unop;
            s.op_log2  = default_unop;
            s.op_floor = default_unop;
            s.op_ceil  = default_unop;
            s.op_round = default_unop;
            s.op_trunc = default_unop;
            s.op_rcp   = default_unop;
            s.op_rsqrt = default_unop;
            s.op_ldexp = default_binop;
            s.op_atan2 = default_binop;
            s.op_sincos = default_unop_2;
            s.op_sincosh = default_unop_2;
            s.op_frexp = default_unop_2;
        }
    }

    if constexpr (T::Depth == 1 && T::IsDynamic && T::IsMask) {
        s.op_all = [](const void *a, void *b) {
            new (b) T(((const T *) a)->all_());
        };
        s.op_any = [](const void *a, void *b) {
            new (b) T(((const T *) a)->any_());
        };
    } else {
        const detail::array_reduce_mask default_reduce_mask =
            (detail::array_reduce_mask) uintptr_t(1);

        s.op_all = default_reduce_mask;
        s.op_any = default_reduce_mask;
    }

    if (T::IsMask && T::Depth == 1 && T::Size != Dynamic) {
        s.op_invert = [](const void *a, void *b) {
            new ((T *) b) T(!*(const T *) a);
        };
    }

    if constexpr (T::IsJIT && T::Depth == 1) {
        s.op_index = [](const void *a) { return ((const T *) a)->index(); };
        s.op_set_index = [](void *a, uint32_t index) { *(((T *) a)->index_ptr()) = index; };
    }

    if constexpr (T::IsDiff && T::Depth == 1 && T::IsFloat) {
        s.op_index_ad = [](const void *a) { return ((const T *) a)->index_ad(); };
        s.op_set_grad_enabled = [](void *a, bool v) { set_grad_enabled(*(T *) a, v); };
        s.op_grad_enabled = [](const void *a) { return grad_enabled(*(const T *) a); };
        s.op_grad = [](const void *a, void *b) { new (b) T(grad(*(const T *) a)); };
        s.op_set_grad = [](void *a, const void *b) { set_grad(*(T *) a, *(const T *) b); };
        s.op_accum_grad = [](void *a, const void *b) { accum_grad(*(T *) a, *(const T *) b); };
        s.op_detach = [](const void *a, void *b) { new (b) detached_t<T>(detach(*(const T *) a)); };
        s.op_enqueue = [](const void *a, const void *b) { enqueue(*(drjit::ADMode *) a, *(const T *) b); };
    }

    if constexpr (T::IsDiff && T::Depth == 1) {
        s.op_detach = [](const void *a, void *b) {
            new (b) detached_t<T>(((const T *) a)->detach_());
        };
        s.op_ad_create = [](void *a, uint32_t index, void *c) {
            if constexpr (T::IsFloat)
                detail::ad_inc_ref_impl<detached_t<T>>(index);
            new (c) T(T::create(index, detached_t<T>(((T *) a)->detach_())));
        };
    }

    if constexpr (T::Depth == 1 && (T::IsDiff || T::IsJIT))
        s.op_set_label = [](void *a, const char *b) { return ((T *) a)->set_label_(b); };

    return tp;
}

template <typename T> nanobind::class_<T> bind_tensor(const char *name = nullptr) {
    namespace nb = nanobind;
    using Array = typename T::Array;

    nb::class_<T> tp = bind<T>(name);

    detail::array_supplement &s =
        nb::type_supplement<detail::array_supplement>(tp);

    s.op_tensor_shape = [](void *o) noexcept -> dr_vector<size_t> & {
        return ((T *) o)->shape();
    };

    s.op_tensor_array = [](PyObject *o) noexcept -> PyObject * {
        T *inst = nb::inst_ptr<T>(o);
        nb::detail::cleanup_list cleanup(o);
        PyObject *result = nb::detail::make_caster<typename T::Array>::from_cpp(
                     inst->array(),
                     nb::rv_policy::reference_internal,
                     &cleanup).ptr();
        cleanup.release();
        return result;
    };

    return tp;
}

// Run bind_array() and bind_tensor() for many different types
template <typename T> void bind_all() {
    if constexpr (is_jit_array_v<T>)
        bind_array<bool_array_t<T>>();
    else
        bind_array<mask_t<T>>();

    bind_array<float32_array_t<T>>();
    bind_array<float64_array_t<T>>();
    bind_array<uint32_array_t<T>>();
    bind_array<int32_array_t<T>>();
    bind_array<uint64_array_t<T>>();
    bind_array<int64_array_t<T>>();

    if constexpr (T::Size == Dynamic && T::Depth == 1) {
        bind_tensor<Tensor<mask_t<T>>>();
        bind_tensor<Tensor<float32_array_t<T>>>();
        bind_tensor<Tensor<float64_array_t<T>>>();
        bind_tensor<Tensor<int32_array_t<T>>>();
        bind_tensor<Tensor<int64_array_t<T>>>();
        bind_tensor<Tensor<uint32_array_t<T>>>();
        bind_tensor<Tensor<uint64_array_t<T>>>();
    }
}

// .. and for many different types
template <typename T> void bind_all_types() {
    if constexpr (!std::is_scalar_v<T>)
        bind_all<T>();

    bind_all<Array<T, 0>>();
    bind_all<Array<T, 1>>();
    bind_all<Array<T, 2>>();
    bind_all<Array<T, 3>>();
    bind_all<Array<T, 4>>();
    bind_all<DynamicArray<T>>();
}

NAMESPACE_END(drjit)
