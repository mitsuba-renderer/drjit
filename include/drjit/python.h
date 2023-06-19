#pragma once

#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit/dynamic.h>
#include <drjit/matrix.h>
#include <drjit/complex.h>
#include <drjit/quaternion.h>
#include <drjit/tensor.h>
#include <drjit-core/traits.h>
#include <nanobind/nanobind.h>


NAMESPACE_BEGIN(drjit)
struct ArrayBinding;
NAMESPACE_END(drjit)

/// Publish a Dr.Jit type binding in Python
extern nanobind::object bind(const drjit::ArrayBinding &);

NAMESPACE_BEGIN(drjit)

/// Constant indicating a dynamically sized component in ``ArrayMeta``
#define DRJIT_DYNAMIC            0xFF

/// Constant indicating fallback to the default implementation in ``ArraySupplement``
#define DRJIT_OP_DEFAULT         ((void *) 0)

/// Constant indicating an unimplemented operation in ``ArraySupplement``
#define DRJIT_OP_NOT_IMPLEMENTED ((void *) 1)

/// Metadata describing the backend/type/shape/.. of a Dr.Jit array binding
struct ArrayMeta {
    uint16_t backend       : 2;
    uint16_t type          : 4;
    uint16_t ndim          : 3;
    uint16_t is_vector     : 1;
    uint16_t is_sequence   : 1;
    uint16_t is_complex    : 1;
    uint16_t is_quaternion : 1;
    uint16_t is_matrix     : 1;
    uint16_t is_tensor     : 1;
    uint16_t is_diff       : 1;
    uint8_t is_valid       : 1;
    uint8_t tsize_rel      : 7;  // type size as multiple of 'talign'
    uint8_t talign;              // type alignment
    uint8_t shape[4];
};

static_assert(sizeof(ArrayMeta) == 8, "Structure packing issue");
enum class ArrayOp {
    Add,
    Sub,
    Mul,
    Count
};

/**
 * \brief Supplemental data stored in Dr.Jit Python array types
 *
 * This is essentially a big table of function pointers to array-specific
 * implementations of various standard operations.
 *
 * The top entries (``len``, ``init``, ``item``, and ``set_item``) are always
 * available (the former two only for dynamic arrays).
 *
 * The ``op`` array contains additional operations enumerated in ``ArrayOp``.
 * Besides pointing to an implementations, entries of this array can taken on
 * two special values:
 *
 * - ``DRJIT_OP_NOT_IMPLEMENTED`` indicates that the operation is not supported
 *   by the specified array type.
 *
 * - ``DRJIT_OP_DEFAULT`` indicates that Dr.Jit should fall back to a default
 *   implementation that recursively invokes the operation on array elements.
 */
struct ArraySupplement : ArrayMeta {
    using Item    = PyObject* (*)(PyObject *, Py_ssize_t) noexcept;
    using SetItem = int       (*)(PyObject *, Py_ssize_t, PyObject *) noexcept;

    using Len  = size_t    (*)(const ArrayBase *) noexcept;
    using Init = void      (*)(ArrayBase *, size_t);
    using Cast = void      (*)(const ArrayBase *, VarType, ArrayBase *);

    using TensorShape = dr_vector<size_t> & (*) (ArrayBase *) noexcept;
    using TensorArray = PyObject * (*) (PyObject *) noexcept;

    // Pointer to the associated array, mask, and element type
    PyObject *array, *mask, *value;

    union {
        struct {
            /// Return an entry as a Python object
            Item item;

            /// Assign a Python object to the given entry
            SetItem set_item;

            /// Determine the length of the given array (if dynamically sized)
            Len len;

            /// Initialize the dynamically sized array to the given size
            Init init;
        };

        struct {
            TensorShape tensor_shape;
            TensorArray tensor_array;
        };
    };

    /// Cast an array into a different format
    Cast cast;

    /// Additional operations
    void *op[(int) ArrayOp::Count];

    inline void *& operator[](ArrayOp o) { return op[(int) o]; }
    inline void * operator[](ArrayOp o) const { return op[(int) o]; }
};

/**
 * \brief Temporary record used by the ``drjit.detail.bind(..)`` function, which
 * registers a new Array type with the Dr.Jit Python bindings
 */
struct ArrayBinding : ArraySupplement {
    nanobind::handle scope;
    const char *name;
    const std::type_info *array_type;
    const std::type_info *value_type;
    void (*copy)(void *, const void *);
    void (*move)(void *, void *) noexcept;
    void (*destruct)(void *) noexcept;
};

NAMESPACE_BEGIN(detail)

template <typename T>
constexpr uint8_t size_or_zero_v = std::is_scalar_v<T> ? 0 : (uint8_t) array_size_v<T>;

NAMESPACE_END(detail)

template <typename T> NB_INLINE void bind_init(ArrayBinding &b, nanobind::handle scope = {}, const char *name = nullptr) {
    namespace nb = nanobind;

    static_assert(
        std::is_copy_constructible_v<T> &&
        std::is_move_constructible_v<T> &&
        std::is_destructible_v<T>,
        "drjit::bind(): type must be copy/move constructible and destructible!"
    );

    using Value = value_t<T>;

    constexpr size_t Align = alignof(T),
                     Size = sizeof(T),
                     RelSize = Size / Align;

    static_assert(Align < 0xFF && RelSize <= 0xFF && RelSize * Align == Size,
                  "drjit::bind(): type is too large!");

    memset(&b, 0, sizeof(ArrayBinding));
    b.backend = (uint16_t) backend_v<T>;

    if (is_mask_v<T>)
        b.type = (uint16_t) VarType::Bool;
    else
        b.type = (uint16_t) var_type_v<scalar_t<T>>;

    if constexpr (!T::IsTensor) {
        b.ndim = (uint16_t) array_depth_v<T>;
        b.shape[0] = detail::size_or_zero_v<T>;
        b.shape[1] = detail::size_or_zero_v<Value>;
        b.shape[2] = detail::size_or_zero_v<value_t<Value>>;
        b.shape[3] = detail::size_or_zero_v<value_t<value_t<Value>>>;
    }

    b.is_vector = T::IsVector;
    b.is_complex = T::IsComplex;
    b.is_quaternion = T::IsQuaternion;
    b.is_matrix = T::IsMatrix;
    b.is_tensor = T::IsTensor;
    b.is_diff = T::IsDiff;
    b.is_valid = 1;

    b.tsize_rel = (uint8_t) RelSize;
    b.talign = (uint8_t) Align;

    b.scope = scope;
    b.name = name;
    b.array_type = &typeid(T);
    b.value_type = std::is_scalar_v<Value> ? nullptr : &typeid(Value);

    if constexpr (!std::is_trivially_copy_constructible_v<T>)
        b.copy = nb::detail::wrap_copy<T>;

    if constexpr (!std::is_trivially_move_constructible_v<T>)
        b.move = nb::detail::wrap_move<T>;

    if constexpr (!std::is_trivially_destructible_v<T>)
        b.destruct = nb::detail::wrap_destruct<T>;
}

template <typename T> NB_INLINE void bind_base(ArrayBinding &b) {
    namespace nb = nanobind;
    using Value = std::conditional_t<is_mask_v<T> && array_depth_v<T> == 1,
                                     bool, value_t<T>>;

    b.item = [](PyObject *o, Py_ssize_t i_) noexcept -> PyObject * {
        T *inst = nb::inst_ptr<T>(o);
        size_t i = (size_t) i_, size = inst->size();

        if (size == 1) // Broadcast
            i = 0;

        nb::handle result;
        if (i < size) {
            nb::detail::cleanup_list cleanup(o);
            result = nb::detail::make_caster<Value>::from_cpp(
                inst->entry(i), nb::rv_policy::reference_internal, &cleanup);
            assert(!cleanup.used());
        } else {
            nb::str tp_name = nb::inst_name(o);
            PyErr_Format(
                PyExc_IndexError,
                "%U.__getitem__(): entry %zu is out of bounds (the array is of size %zu).",
                tp_name.ptr(), i, size);
        }
        return result.ptr();
    };

    b.set_item = [](PyObject *o, Py_ssize_t i_, PyObject *value) noexcept -> int {
        T *inst = nb::inst_ptr<T>(o);
        size_t i = (size_t) i_, size = inst->size();
        if (i < size) {
            nb::detail::cleanup_list cleanup(o);
            nb::detail::make_caster<Value> in;
            bool success =  value != Py_None && in.from_python(
                value, (uint8_t) nb::detail::cast_flags::convert, &cleanup);
            if (success)
                inst->set_entry(i, in.operator Value & ());
            cleanup.release();

            if (success) {
                return 0;
            } else {
                nb::str tp_name = nb::inst_name(o),
                        val_name = nb::inst_name(value);
                PyErr_Format(
                    PyExc_TypeError,
                    "%U.__setitem__(): could not initialize element with a value of type '%U'.",
                    tp_name.ptr(), val_name.ptr());
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

    if constexpr (T::Size == Dynamic) {
        b.len = (ArraySupplement::Len) + [](const T *a) noexcept -> size_t {
            return a->size();
        };

        b.init = (ArraySupplement::Init) +[](T *a, size_t size) { a->init_(size); };
    }

}

template <typename T> void bind_arithmetic(ArrayBinding &b) {
    using UInt32  = uint32_array_t<T>;
    using Int32   = int32_array_t<T>;
    using UInt64  = uint64_array_t<T>;
    using Int64   = int64_array_t<T>;
    using Float32 = float32_array_t<T>;
    using Float64 = float64_array_t<T>;

    b[ArrayOp::Add] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a + *b); };
    b[ArrayOp::Sub] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a - *b); };
    b[ArrayOp::Mul] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a * *b); };

    b.cast = (ArrayBinding::Cast) +[](const ArrayBase *a, VarType vt, T *b) {
        switch (vt) {
            case VarType::Int32:   new (b) T(*(const Int32 *)   a); break;
            case VarType::UInt32:  new (b) T(*(const UInt32 *)  a); break;
            case VarType::Int64:   new (b) T(*(const Int64 *)   a); break;
            case VarType::UInt64:  new (b) T(*(const UInt64 *)  a); break;
            case VarType::Float32: new (b) T(*(const Float32 *) a); break;
            case VarType::Float64: new (b) T(*(const Float64 *) a); break;
            default: nanobind::detail::raise("Unsupported cast.");
        }
    };
}

template <typename T> void disable_arithmetic(ArrayBinding &b) {
    b[ArrayOp::Add] = b[ArrayOp::Sub] = b[ArrayOp::Mul] =
        DRJIT_OP_NOT_IMPLEMENTED;
    b.cast = (ArrayBinding::Cast) DRJIT_OP_NOT_IMPLEMENTED;
}

template <typename T>
void bind_tensor(ArrayBinding &b) {
    b.tensor_shape = (ArrayBinding::TensorShape) +[](T *o) noexcept -> dr_vector<size_t> & {
        return o->shape();
    };

    b.tensor_array = (ArrayBinding::TensorArray) +[](PyObject *o) noexcept -> PyObject * {
        T *inst = nanobind::inst_ptr<T>(o);
        nanobind::detail::cleanup_list cleanup(o);
        PyObject *result = nanobind::detail::make_caster<typename T::Array>::from_cpp(
                   inst->array(), nanobind::rv_policy::reference_internal,
                   &cleanup)
            .ptr();
        assert(!cleanup.used());
        return result;
    };
}

template <typename T> void bind_array(ArrayBinding &b) {
    bind_init<T>(b);

    if constexpr (T::IsTensor) {
        bind_tensor<T>(b);
    } else {
        bind_base<T>(b);

        if constexpr (T::Depth == 1) {
            if constexpr (T::IsArithmetic)
                bind_arithmetic<T>(b);
        }
    }

    if constexpr (!T::IsArithmetic)
        disable_arithmetic<T>(b);

    bind(b);
}

// Run bind_array() for many different plain array types
template <typename T> void bind_array_types(ArrayBinding &b) {
    bind_array<mask_t<T>>(b);
    bind_array<float32_array_t<T>>(b);
    bind_array<float64_array_t<T>>(b);
    bind_array<uint32_array_t<T>>(b);
    bind_array<int32_array_t<T>>(b);
    bind_array<uint64_array_t<T>>(b);
    bind_array<int64_array_t<T>>(b);
}

// Run bind_array() for many different matrix types
template <typename T, size_t Size> void bind_matrix_types(ArrayBinding &b) {
    using VecF32 = Array<float32_array_t<T>, Size>;
    using VecF64 = Array<float64_array_t<T>, Size>;
    using VecMask = mask_t<VecF32>;

    bind_array<Mask<VecMask, Size>>(b);
    bind_array<Array<VecF32, Size>>(b);
    bind_array<Array<VecF64, Size>>(b);
    bind_array<Matrix<float32_array_t<T>, Size>>(b);
    bind_array<Matrix<float64_array_t<T>, Size>>(b);
}

template <typename T> void bind_all(ArrayBinding &b) {
    if constexpr (!std::is_scalar_v<T>)
        bind_array_types<T>(b);

    bind_array_types<Array<T, 0>>(b);
    bind_array_types<Array<T, 1>>(b);
    bind_array_types<Array<T, 2>>(b);
    bind_array_types<Array<T, 3>>(b);
    bind_array_types<Array<T, 4>>(b);
    bind_array_types<DynamicArray<T>>(b);

    bind_matrix_types<T, 2>(b);
    bind_matrix_types<T, 3>(b);
    bind_matrix_types<T, 4>(b);

    bind_array<Complex<float32_array_t<T>>>(b);
    bind_array<Complex<float64_array_t<T>>>(b);
    bind_array<Quaternion<float32_array_t<T>>>(b);
    bind_array<Quaternion<float64_array_t<T>>>(b);

    using T2 = std::conditional_t<std::is_scalar_v<T>, DynamicArray<T>, T>;
    bind_array<Tensor<mask_t<T2>>>(b);
    bind_array<Tensor<float32_array_t<T2>>>(b);
    bind_array<Tensor<float64_array_t<T2>>>(b);
    bind_array<Tensor<int32_array_t<T2>>>(b);
    bind_array<Tensor<int64_array_t<T2>>>(b);
    bind_array<Tensor<uint32_array_t<T2>>>(b);
    bind_array<Tensor<uint64_array_t<T2>>>(b);
}

NAMESPACE_END(drjit)
