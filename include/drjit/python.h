/*
   drjit/python.h -- Public interface for creating Python bindings of custom
   Dr.Jit types.

   To dynamically construct a new type binding for a class ``MyType``, call
   the ``dr::bind_array_t<>`` function as follows:

   ```python
   dr::ArrayBinding b;
   auto py_type_object = dr::bind_array_t<MyType>(b, m, "MyType")
   ```

   The ``ArrayBinding`` variable is scratch space and can be reused
   following this call.

   You do not need to link against the Dr.Jit Python bindings to be able
   to use this function, it is realized using "pure" nanobind code.

   This code below has the job of populating the ``ArrayBinding`` data
   structure, which consists of two main parts: the ``ArrayMeta`` metadata
   block describes the shape and type of the array.

   The ``ArraySupplement`` part consists of a huge list of function pointers.
   Each pointer provides the possibility to override how the array realizes a
   certain operation (e.g., addition, trigometry, etc.). This pointer table is
   directly copied into the newly created Python type object using the "type
   supplement" feature of nanobind.

   Entries can be initialized with two special values besides a pointer to an
   implementation: ``DRJIT_OP_DEFAULT`` (equal to 0) means that the operation
   is supported but no special implementation is given. Usually, this means
   that the Dr.Jit Python bindings will recursively traverse the type and then
   perform the operation on the elements.

   The other option (``DRJIT_OP_NOT_IMPLEMENTED``) causes the bindings to
   raise an exception when the user attempts to perform such an operation.

   Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
   Copyright 2023, Realistic Graphics Lab, EPFL.

   All rights reserved. Use of this source code is governed by a
   BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit/dynamic.h>
#include <drjit/matrix.h>
#include <drjit/complex.h>
#include <drjit/quaternion.h>
#include <drjit/tensor.h>
#include <drjit/math.h>
#include <drjit-core/python.h>
#include <nanobind/stl/array.h>

NAMESPACE_BEGIN(drjit)
struct ArrayBinding;
NAMESPACE_END(drjit)

#if defined(DRJIT_PYTHON_BUILD)
/// Publish a Dr.Jit type binding in Python
extern nanobind::object bind(const drjit::ArrayBinding &);
#endif

NAMESPACE_BEGIN(drjit)

/// Constant indicating a dynamically sized component in ``ArrayMeta``
#define DRJIT_DYNAMIC            0xFF

/// Constant indicating fallback to the default implementation in ``ArraySupplement``
#define DRJIT_OP_DEFAULT         ((void *) 0)

/// Constant indicating an unimplemented operation in ``ArraySupplement``
#define DRJIT_OP_NOT_IMPLEMENTED ((void *) 1)

/**
 * \brief Metadata describing the backend/type/shape/.. of a Dr.Jit array binding
 *
 * All Dr.Jit types include a metadata descriptor that characterizes the type
 * in more detail. This contains essentially the same information as a type
 * name like 'drjit.cuda.ad.Array3f', but using an easier-to-manipulate
 * representation.
 */
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
    uint16_t is_class      : 1;
    uint16_t is_valid      : 1;
    uint16_t tsize_rel     : 7;  // type size as multiple of 'talign'
    uint16_t talign        : 7; // type alignment
    uint8_t shape[4];
};

static_assert(sizeof(ArrayMeta) == 8, "Structure packing issue");

/// A large set of Dr.Jit operations are handled generically. This
/// enumeration encodes indices into the ArraySupplement::op field
/// which points to the underlying implementations.
enum class ArrayOp {
    // Unary operations
    Neg,
    Invert,

    Abs,
    Sqrt,
    Rcp,
    Rsqrt,
    Cbrt,

    Popcnt,
    Lzcnt,
    Tzcnt,
    Brev,

    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Sincos,
    Tan,
    Asin,
    Acos,
    Atan,

    Sinh,
    Cosh,
    Sincosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,

    Erf,
    ErfInv,
    LGamma,

    Round,
    Trunc,
    Ceil,
    Floor,

    // Binary arithetic operations
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    Mod,
    LShift,
    RShift,

    Minimum,
    Maximum,
    Atan2,

    // Binary bit/mask operations
    And,
    Or,
    Xor,

    // Ternary operations
    Fma,
    Select,

    // Horizontal reductions
    All,
    Any,
    Count,
    Sum,
    Prod,
    Min,
    Max,

    // Miscellaneous
    Richcmp,

    OpCount
};

#if defined(_MSC_VER)
#  pragma warning(disable: 4201) // nonstandard extension used: nameless struct/union
#endif

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
 * Besides pointing to an implementations, entries of this array can take on
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

    using Len = size_t (*)(const ArrayBase *) noexcept;
    using Init = void (*)(size_t, ArrayBase *);
    using InitData = void (*)(size_t, const void *, ArrayBase *);
    using InitIndex = void (*)(uint64_t, ArrayBase *);
    using InitConst = void (*)(size_t, bool, PyObject *, ArrayBase *);
    using Cast = void (*)(const ArrayBase *, VarType, bool reinterpret, ArrayBase *);
    using Index = uint64_t (*)(const ArrayBase *) noexcept;
    using Data = void *(*)(const ArrayBase *) noexcept;
    using Gather = void (*)(ReduceMode, const ArrayBase *, const ArrayBase *,
                            const ArrayBase *, ArrayBase *);
    using ScatterReduce = void (*)(ReduceOp, ReduceMode, const ArrayBase *,
                                   const ArrayBase *, const ArrayBase *,
                                   const ArrayBase *);
    using ScatterInc = void (*)(const ArrayBase *, ArrayBase *, ArrayBase *, ArrayBase *);
    using ScatterAddKahan = void (*)(const ArrayBase *, const ArrayBase *,
                                     const ArrayBase *, ArrayBase *,
                                     ArrayBase *);
    using UnaryOp  = void (*)(const ArrayBase *, ArrayBase *);
    using BinaryOp = void (*)(const ArrayBase *, const ArrayBase *, ArrayBase *);

    using TensorShape = vector<size_t> & (*) (ArrayBase *) noexcept;
    using TensorArray = PyObject * (*) (PyObject *) noexcept;
    using BlockReduceOp = void (*)(const ArrayBase *, ReduceOp, uint32_t, int, ArrayBase *);
    using BlockPrefixReduceOp = void (*)(const ArrayBase *, ReduceOp, uint32_t, bool, bool, ArrayBase *);

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

            /// Create a counter variable
            Init init_counter;

            /// Initialize from a Python constant value
            InitConst init_const;

            /// Initialize from a given memory region on the CPU
            InitData init_data;

            /// Initialize from a JIT variable index
            InitIndex init_index;

            /// Replace with another JIT variable
            InitIndex reset_index;

            /// Gather operation
            Gather gather;

            /// Scatter reduction operation
            ScatterReduce scatter_reduce;

            /// Scatter-increment operation
            ScatterInc scatter_inc;

            /// Kahan-compensated scatter-addition
            ScatterAddKahan scatter_add_kahan;

            /// Return a pointer to the underlying storage
            Data data;

            /// Return the JIT variable index
            Index index;

            /// Cast an array into a different format
            Cast cast;

            /// Compress a mask vector
            UnaryOp compress;

            /// Reduce an array within blocks
            BlockReduceOp block_reduce;

            /// Prefix-reduce an array within blocks
            BlockPrefixReduceOp block_prefix_reduce;

            /// Additional operations
            void *op[(int) ArrayOp::OpCount];
        };

        // Tensors expose a different set of operations
        struct {
            TensorShape tensor_shape;
            TensorArray tensor_array;

            /// Python type object for indexing calculations
            PyObject *tensor_index;
        };
    };

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
constexpr uint8_t size_or_zero_v = drjit::detail::is_scalar_v<T> ? 0 : (uint8_t) (size_v<T> == (size_t) - 1 ? 255 : size_v<T>);

NAMESPACE_END(detail)

template <typename T>
NB_INLINE void bind_init(ArrayBinding &b, nanobind::handle scope = {},
                         const char *name = nullptr) {
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

    static_assert(Align < 0x80 && RelSize < 0x80 && RelSize * Align == Size,
                  "drjit::bind(): type is too large!");

    memset((void *) &b, 0, sizeof(ArrayBinding));
    b.backend = (uint16_t) backend_v<T>;

    if (is_mask_v<T>)
        b.type = (uint16_t) VarType::Bool;
    else
        b.type = (uint16_t) type_v<scalar_t<T>>;

    if constexpr (!T::IsTensor) {
        b.ndim = (uint16_t) depth_v<T>;
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
    b.is_class = T::IsClass;
    b.is_valid = 1;
    b.tsize_rel = (uint16_t) RelSize;
    b.talign = (uint16_t) Align;

    b.scope = scope;
    b.name = name;
    b.array_type = &typeid(T);
    b.value_type = drjit::detail::is_scalar_v<Value> && !std::is_pointer_v<Value>
                       ? nullptr
                       : &typeid(std::remove_pointer_t<Value>);

    if constexpr (!std::is_trivially_copy_constructible_v<T>)
        b.copy = nb::detail::wrap_copy<T>;

    if constexpr (!std::is_trivially_move_constructible_v<T>)
        b.move = nb::detail::wrap_move<T>;

    if constexpr (!std::is_trivially_destructible_v<T>)
        b.destruct = nb::detail::wrap_destruct<T>;
}

template <typename T> NB_INLINE void bind_base(ArrayBinding &b) {
    namespace nb = nanobind;
    using Value = std::conditional_t<is_mask_v<T> && depth_v<T> == 1,
                                     bool, value_t<T>>;

    b.item = [](PyObject *o, Py_ssize_t i_) noexcept -> PyObject * {
        T *inst = nb::inst_ptr<T>(o);
        size_t i = (size_t) i_, size = inst->size();

        if (size == 1) // Broadcast
            i = 0;

        nb::handle result;
        if (i < size) {
            nb::detail::cleanup_list cleanup(o);
            try {
                auto &&value = inst->entry(i);
                result = nb::detail::make_caster<Value>::from_cpp(
                    value, nb::rv_policy::reference_internal, &cleanup);
            } catch (const std::exception &e) {
                nb::str tp_name = nb::inst_name(o);
                PyErr_Format(PyExc_RuntimeError, "%U.__getitem__(): %s",
                             tp_name.ptr(), e.what());
            }
            assert(!cleanup.used());
        } else {
            nb::str tp_name = nb::inst_name(o);
            PyErr_Format(
                PyExc_IndexError,
                "%U.__getitem__(): entry %zd is out of bounds (the array is of size %zu).",
                tp_name.ptr(), i_, size);
        }
        return result.ptr();
    };

    b.set_item = [](PyObject *o, Py_ssize_t i_, PyObject *value) noexcept -> int {
        T *inst = nb::inst_ptr<T>(o);
        size_t i = (size_t) i_, size = inst->size();
        if (i < size) {
            nb::detail::cleanup_list cleanup(o);
            nb::detail::make_caster<Value> in;

            bool success = value != Py_None && in.from_python(
                value, (uint8_t) nb::detail::cast_flags::convert, &cleanup);
            if (success) {
                using Intrinsic = nb::detail::intrinsic_t<Value>;
                using Out = std::conditional_t<std::is_pointer_v<Value>, Intrinsic*, Intrinsic &>;

                Out out = in.operator Out();
                inst->set_entry(i, (value_t<T>) out);
            }
            cleanup.release();

            if (success) {
                return 0;
            } else {
                PyErr_Format(
                    PyExc_TypeError,
                    "%U.__setitem__(): could not initialize element with a value of type '%U'.",
                    nb::inst_name(o).ptr(), nb::inst_name(value).ptr());
                return -1;
            }
        } else {
            PyErr_Format(
                PyExc_IndexError,
                "%U.__setitem__(): entry %zd is out of bounds (the array is of size %zu).",
                nb::inst_name(o).ptr(), i_, size);
            return -1;
        }
    };

    if constexpr (T::Size == Dynamic) {
        b.len = (ArraySupplement::Len) + [](const T *a) noexcept -> size_t {
            return a->size();
        };

        b.init = (ArraySupplement::Init) + [](size_t size, T *a) {
            new (a) T(empty<T>(size));
        };

        if constexpr (T::Depth == 1) {
            b.init_data = (ArraySupplement::InitData) +
                          [](size_t size, const void *ptr, T *a) {
                              new (a) T(load<T>(ptr, size));
                          };

            b.init_const = (ArraySupplement::InitConst) + [](size_t size, bool opaque, PyObject *h, T *a) {
                scalar_t<T> scalar;
                if (T::IsClass) {
                    if (nb::handle(h).equal(nb::int_(0)))
                        h = Py_None;
                }
                if (!nb::try_cast(nb::handle(h), scalar)) {
                    nb::str tp_name = nb::inst_name(h);
                    nb::detail::raise("Could not initialize element with a "
                                      "value of type '%s'.", tp_name.c_str());
                }
                if (opaque)
                    new (a) T(drjit::opaque<T>(scalar, size));
                else
                    new (a) T(drjit::full<T>(scalar, size));
            };

            if constexpr (std::is_same_v<scalar_t<T>, uint32_t>) {
                b.init_counter = (ArraySupplement::Init) +[](size_t size, T *a) {
                    new (a) T(T::counter(size));
                };
            }
        }

        b.data = (ArraySupplement::Data) + [](const T *a) { return a->data(); };
    } else {
        if constexpr (!is_dynamic_v<T> && !T::IsKMask)
            b.data = (ArraySupplement::Data) + [](const T *a) { return a->data(); };
    }
}

template <typename T> void bind_arithmetic(ArrayBinding &b) {
    b[ArrayOp::Abs] = (void *) +[](const T *a, T *b) { new (b) T(abs(*a)); };
    b[ArrayOp::Neg] = (void *) +[](const T *a, T *b) { new (b) T(-*a); };

    // Binary arithetic operations
    b[ArrayOp::Add] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a + *b); };
    b[ArrayOp::Sub] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a - *b); };
    b[ArrayOp::Mul] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a * *b); };

    b[ArrayOp::Minimum] = (void *) +[](const T *a, const T *b, T *c) {
        new (c) T(drjit::minimum(*a, *b));
    };

    b[ArrayOp::Maximum] = (void *) +[](const T *a, const T *b, T *c) {
        new (c) T(drjit::maximum(*a, *b));
    };

    // Ternary arithetic operations
    b[ArrayOp::Fma] = (void *) +[](const T *a, const T *b, const T *c, T *d) {
        new (d) T(fmadd(*a, *b, *c));
    };

    if constexpr (is_jit_v<T>) {
        b[ArrayOp::Sum] = (void *) +[](const T *a, T *b) { new (b) T(a->sum_()); };
        b[ArrayOp::Prod] = (void *) +[](const T *a, T *b) { new (b) T(a->prod_()); };
        b[ArrayOp::Min] = (void *) +[](const T *a, T *b) { new (b) T(a->min_()); };
        b[ArrayOp::Max] = (void *) +[](const T *a, T *b) { new (b) T(a->max_()); };
    }
}


template <typename T> void bind_cast(ArrayBinding &b) {
    using UInt32  = uint32_array_t<T>;
    using Int32   = int32_array_t<T>;
    using UInt64  = uint64_array_t<T>;
    using Int64   = int64_array_t<T>;
    using Float16 = float16_array_t<T>;
    using Float32 = float32_array_t<T>;
    using Float64 = float64_array_t<T>;

    if constexpr (!T::IsClass) {
        b.cast = (ArrayBinding::Cast) +[](const ArrayBase *a, VarType vt, bool reinterpret, T *b) {
            if (!reinterpret) {
                switch (vt) {
                    case VarType::Int32:   new (b) T(*(const Int32 *)   a); break;
                    case VarType::UInt32:  new (b) T(*(const UInt32 *)  a); break;
                    case VarType::Int64:   new (b) T(*(const Int64 *)   a); break;
                    case VarType::UInt64:  new (b) T(*(const UInt64 *)  a); break;
                    case VarType::Float16: new (b) T(*(const Float16 *) a); break;
                    case VarType::Float32: new (b) T(*(const Float32 *) a); break;
                    case VarType::Float64: new (b) T(*(const Float64 *) a); break;
                    default: nanobind::raise("Unsupported cast.");
                }
            } else {
                switch (vt) {
                    case VarType::Int32:   new (b) T(reinterpret_array<T>(*(const Int32 *)   a)); break;
                    case VarType::UInt32:  new (b) T(reinterpret_array<T>(*(const UInt32 *)  a)); break;
                    case VarType::Int64:   new (b) T(reinterpret_array<T>(*(const Int64 *)   a)); break;
                    case VarType::UInt64:  new (b) T(reinterpret_array<T>(*(const UInt64 *)  a)); break;
                    case VarType::Float16: new (b) T(reinterpret_array<T>(*(const Float16 *) a)); break;
                    case VarType::Float32: new (b) T(reinterpret_array<T>(*(const Float32 *) a)); break;
                    case VarType::Float64: new (b) T(reinterpret_array<T>(*(const Float64 *) a)); break;
                    default: nanobind::raise("Unsupported cast.");
                }
            }
        };
    } else {
        // Only allow reinterpreting Class arrays to UInt32
        b.cast = (ArrayBinding::Cast) +[](const ArrayBase *a, VarType vt, bool reinterpret, T *b) {
            if (!reinterpret) {
                nanobind::raise("Unsupported cast.");
            } else {
                if (vt == VarType::UInt32)
                    new (b) T(reinterpret_array<T>(*(const UInt32 *)  a));
                else
                    nanobind::raise("Unsupported cast.");
            }
        };
    }
}

inline void disable_cast(ArrayBinding &b) {
    b.cast = (ArrayBinding::Cast) DRJIT_OP_NOT_IMPLEMENTED;
}

inline void disable_arithmetic(ArrayBinding &b) {
    b[ArrayOp::Abs] = b[ArrayOp::Neg] = b[ArrayOp::Add] = b[ArrayOp::Sub] =
        b[ArrayOp::Mul] = b[ArrayOp::Minimum] = b[ArrayOp::Maximum] =
        b[ArrayOp::Fma] = b[ArrayOp::Sum] = b[ArrayOp::Prod] =
        b[ArrayOp::Min] = b[ArrayOp::Max] = DRJIT_OP_NOT_IMPLEMENTED;
}

template <typename T> void bind_int_arithmetic(ArrayBinding &b) {
    b[ArrayOp::FloorDiv] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a / *b); };
    b[ArrayOp::LShift] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a << *b); };
    b[ArrayOp::RShift] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a >> *b); };
    b[ArrayOp::Mod] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a % *b); };
    b[ArrayOp::Popcnt] = (void *) +[](const T *a, T *b) { new (b) T(popcnt(*a)); };
    b[ArrayOp::Lzcnt] = (void *) +[](const T *a, T *b) { new (b) T(lzcnt(*a)); };
    b[ArrayOp::Tzcnt] = (void *) +[](const T *a, T *b) { new (b) T(tzcnt(*a)); };
    b[ArrayOp::Brev] = (void *) +[](const T *a, T *b) { new (b) T(brev(*a)); };
}

inline void disable_int_arithmetic(ArrayBinding &b) {
    b[ArrayOp::FloorDiv] = b[ArrayOp::LShift] = b[ArrayOp::RShift] =
        b[ArrayOp::Mod] = b[ArrayOp::Popcnt] = b[ArrayOp::Lzcnt] =
        b[ArrayOp::Tzcnt] = b[ArrayOp::Brev] = DRJIT_OP_NOT_IMPLEMENTED;
}

template <typename T> void bind_float_arithmetic(ArrayBinding &b) {
    b[ArrayOp::TrueDiv] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a / *b); };
    b[ArrayOp::Sqrt] = (void *) +[](const T *a, T *b) { new (b) T(sqrt(*a)); };
    b[ArrayOp::Rcp] = (void *) +[](const T *a, T *b) { new (b) T(rcp(*a)); };
    b[ArrayOp::Rsqrt] = (void *) +[](const T *a, T *b) { new (b) T(rsqrt(*a)); };
    b[ArrayOp::Cbrt] = (void *) +[](const T *a, T *b) { new (b) T(cbrt(*a)); };
    b[ArrayOp::Exp] = (void *) +[](const T *a, T *b) { new (b) T(exp(*a)); };
    b[ArrayOp::Exp2] = (void *) +[](const T *a, T *b) { new (b) T(exp2(*a)); };
    b[ArrayOp::Log] = (void *) +[](const T *a, T *b) { new (b) T(log(*a)); };
    b[ArrayOp::Log2] = (void *) +[](const T *a, T *b) { new (b) T(log2(*a)); };
    b[ArrayOp::Sin] = (void *) +[](const T *a, T *b) { new (b) T(sin(*a)); };
    b[ArrayOp::Cos] = (void *) +[](const T *a, T *b) { new (b) T(cos(*a)); };
    b[ArrayOp::Sincos] = (void *) +[](const T *a, T *b, T *c) {
        auto [sa, ca] = sincos(*a);
        new (b) T(std::move(sa));
        new (c) T(std::move(ca));
    };
    b[ArrayOp::Tan] = (void *) +[](const T *a, T *b) { new (b) T(tan(*a)); };
    b[ArrayOp::Asin] = (void *) +[](const T *a, T *b) { new (b) T(asin(*a)); };
    b[ArrayOp::Acos] = (void *) +[](const T *a, T *b) { new (b) T(acos(*a)); };
    b[ArrayOp::Atan] = (void *) +[](const T *a, T *b) { new (b) T(atan(*a)); };
    b[ArrayOp::Sinh] = (void *) +[](const T *a, T *b) { new (b) T(sinh(*a)); };
    b[ArrayOp::Cosh] = (void *) +[](const T *a, T *b) { new (b) T(cosh(*a)); };
    b[ArrayOp::Sincosh] = (void *) +[](const T *a, T *b, T *c) {
        auto [sa, ca] = sincosh(*a);
        new (b) T(std::move(sa));
        new (c) T(std::move(ca));
    };
    b[ArrayOp::Tanh] = (void *) +[](const T *a, T *b) { new (b) T(tanh(*a)); };
    b[ArrayOp::Asinh] = (void *) +[](const T *a, T *b) { new (b) T(asinh(*a)); };
    b[ArrayOp::Acosh] = (void *) +[](const T *a, T *b) { new (b) T(acosh(*a)); };
    b[ArrayOp::Atanh] = (void *) +[](const T *a, T *b) { new (b) T(atanh(*a)); };
    b[ArrayOp::Erf] = (void *) +[](const T *a, T *b) { new (b) T(erf(*a)); };
    b[ArrayOp::ErfInv] = (void *) +[](const T *a, T *b) { new (b) T(erfinv(*a)); };
    b[ArrayOp::LGamma] = (void *) +[](const T *a, T *b) { new (b) T(lgamma(*a)); };
    b[ArrayOp::Atan2] = (void *) +[](const T *a, const T *b, T *c) {
        new (c) T(atan2(*a, *b));
    };

    b[ArrayOp::Round] = (void *) +[](const T *a, T *b) { new (b) T(round(*a)); };
    b[ArrayOp::Trunc] = (void *) +[](const T *a, T *b) { new (b) T(trunc(*a)); };
    b[ArrayOp::Ceil] = (void *) +[](const T *a, T *b) { new (b) T(ceil(*a)); };
    b[ArrayOp::Floor] = (void *) +[](const T *a, T *b) { new (b) T(floor(*a)); };
}

inline void disable_float_arithmetic(ArrayBinding &b) {
    b[ArrayOp::TrueDiv] = b[ArrayOp::Sqrt] = b[ArrayOp::Rcp] =
    b[ArrayOp::Rsqrt] = b[ArrayOp::Cbrt] = b[ArrayOp::Exp] =
    b[ArrayOp::Exp2] = b[ArrayOp::Log] = b[ArrayOp::Log2] =
    b[ArrayOp::Sin] = b[ArrayOp::Cos] = b[ArrayOp::Sincos] =
    b[ArrayOp::Tan] = b[ArrayOp::Asin] = b[ArrayOp::Acos] =
    b[ArrayOp::Atan] = b[ArrayOp::Sinh] = b[ArrayOp::Cosh] =
    b[ArrayOp::Sincosh] = b[ArrayOp::Tanh] = b[ArrayOp::Asinh] =
    b[ArrayOp::Acosh] = b[ArrayOp::Atanh] = b[ArrayOp::Erf] =
    b[ArrayOp::ErfInv] = b[ArrayOp::LGamma] =
    b[ArrayOp::Atan2] = DRJIT_OP_NOT_IMPLEMENTED;
}

template <typename T>
void bind_tensor(ArrayBinding &b) {
    namespace nb = nanobind;

    b.tensor_shape = (ArrayBinding::TensorShape) +[](T *o) noexcept -> vector<size_t> & {
        return o->shape();
    };

    b.tensor_array = (ArrayBinding::TensorArray) +[](PyObject *o) noexcept -> PyObject * {
        T *inst = nanobind::inst_ptr<T>(o);
        nanobind::detail::cleanup_list cleanup(o);
        nb::handle result =
            nanobind::detail::make_caster<typename T::Array>::from_cpp(
                inst->array(), nanobind::rv_policy::reference_internal,
                &cleanup);
        assert(!cleanup.used());
        return result.ptr();
    };
}

template <typename T> void bind_mask_reductions(ArrayBinding &b) {
    if constexpr (is_jit_v<T>) {
        b[ArrayOp::All] = (void *) +[](const T *a, T *b) { new (b) T(a->all_()); };
        b[ArrayOp::Any] = (void *) +[](const T *a, T *b) { new (b) T(a->any_()); };
        b[ArrayOp::Count] = (void *) +[](const T *a, uint32_array_t<T> *b) {
            new (b) uint32_array_t<T>(a->count_());
        };
    }

    b.compress =
        (ArraySupplement::UnaryOp) +
        [](const T *a, uint32_array_t<T> *b) { new (b) uint32_array_t<T>(a->compress_()); };
}

inline void disable_mask_reductions(ArrayBinding &b) {
    b[ArrayOp::All] = b[ArrayOp::Any] = DRJIT_OP_NOT_IMPLEMENTED;
}

template <typename T> void bind_bit_ops(ArrayBinding &b) {
    b[ArrayOp::And] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a & *b); };
    b[ArrayOp::Or]  = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a | *b); };
    b[ArrayOp::Xor] = (void *) +[](const T *a, const T *b, T *c) {
        if constexpr (T::IsIntegral)
            new (c) T(*a ^ *b);
        else
            new (c) T(*a != *b);
    };
}

template <typename T> void bind_bit_invert(ArrayBinding &b) {
    b[ArrayOp::Invert] = (void *) +[](const T *a, T *b) {
        if constexpr (T::IsIntegral)
            new (b) T(~*a);
        else
            new (b) T(!*a);
    };
}

inline void disable_bit_ops(ArrayBinding &b) {
    b[ArrayOp::And] = b[ArrayOp::Or] = b[ArrayOp::Xor] = b[ArrayOp::Invert] =
        DRJIT_OP_NOT_IMPLEMENTED;
}

template <typename T> void bind_richcmp(ArrayBinding &b) {
    using Mask = mask_t<T>;

    b[ArrayOp::Richcmp] = (void *) +[](const T *a, const T *b, int op, Mask *c) {
        switch (op) {
            case Py_LT: new (c) Mask(*a < *b); break;
            case Py_LE: new (c) Mask(*a <= *b); break;
            case Py_GT: new (c) Mask(*a > *b); break;
            case Py_GE: new (c) Mask(*a >= *b); break;
            case Py_EQ: new (c) Mask(*a == *b); break;
            case Py_NE: new (c) Mask(*a != *b); break;
        }
    };
}

template <typename T> void bind_select(ArrayBinding &b) {
    b[ArrayOp::Select] = (void *) +[](const mask_t<T> *a, const T *b, const T *c, T *d) {
        new (d) T(select(*a, *b, *c));
    };
}

template <typename T> void bind_jit_ops(ArrayBinding &b) {
    b.index = (ArraySupplement::Index)
        +[](const T *v) { return v->index_combined(); };
    b.init_index = (ArraySupplement::InitIndex) + [](uint64_t index, T *v) {
        new (v) T(T::borrow((typename T::Index) index));
    };
    b.reset_index = (ArraySupplement::InitIndex) + [](uint64_t index, T *v) {
        *v = T::borrow((typename T::Index) index);
    };
}

template <typename T> void bind_memop(ArrayBinding &b) {
    using UInt32 = uint32_array_t<T>;
    using Mask = mask_t<T>;
    using Scalar = scalar_t<T>;

    b.gather = (ArraySupplement::Gather)
        +[](ReduceMode mode, const T *a, const UInt32 *b, const Mask *c, T *d) {
            new (d) T(gather<T>(*a, *b, *c, mode));
        };

    b.scatter_reduce = (ArraySupplement::ScatterReduce)
        +[](ReduceOp op, ReduceMode mode, const T *a, const UInt32 *b, const Mask *c, T *d) {
            scatter_reduce(op, *d, *a, *b, *c, mode);
        };

    if constexpr (std::is_same_v<scalar_t<T>, uint32_t> && is_jit_v<T>) {
        b.scatter_inc = (ArraySupplement::ScatterInc)
            +[](const UInt32 *a, const Mask *b, UInt32 *c, UInt32 *d) {
                new (d) T(scatter_inc(*c, *a, *b));
            };
    }

    if constexpr ((std::is_same_v<Scalar, float> ||
                   std::is_same_v<Scalar, double>) && is_jit_v<T>) {
        b.scatter_add_kahan =
            (ArraySupplement::ScatterAddKahan) +
            [](const T *a, const UInt32 *b, const Mask *c, T *d, T *e) {
                scatter_add_kahan(*d, *e, *a, *b, *c);
            };
    }

    if constexpr (!is_mask_v<T>) {
        b.block_reduce = (ArraySupplement::BlockReduceOp) +
            [](const T *a, ReduceOp op, uint32_t block_size, int symbolic, T *out) {
                new (out) T(block_reduce(op, *a, block_size, symbolic));
            };

        b.block_prefix_reduce = (ArraySupplement::BlockPrefixReduceOp) +
            [](const T *a, ReduceOp op, uint32_t block_size, bool exclusive, bool reverse, T *out) {
                new (out) T(block_prefix_reduce(op, *a, block_size, exclusive, reverse));
            };
    }
}

template <typename T> void bind_special(ArrayBinding &b) {
    b[ArrayOp::Mul] = (void *) +[](const T *a, const T *b, T *c) { new (c) T(*a * *b); };
    b[ArrayOp::Rcp] = (void *) +[](const T *a, T *b) { new (b) T(rcp(*a)); };
    b[ArrayOp::Fma] = (void *) +[](const T *a, const T *b, const T *c,
                                   T *d) { new (d) T(fmadd(*a, *b, *c)); };
}

template <typename T> void bind_complex_and_quaternion(ArrayBinding &b) {
    b[ArrayOp::Abs] = (void *) +[](const T *a, T *b) { new (b) T(abs(*a)); };
    b[ArrayOp::Sqrt] = (void *) +[](const T *a, T *b) { new (b) T(sqrt(*a)); };
    b[ArrayOp::Rsqrt] = (void *) +[](const T *a, T *b) { new (b) T(rsqrt(*a)); };
    b[ArrayOp::Log2] = (void *) +[](const T *a, T *b) { new (b) T(log2(*a)); };
    b[ArrayOp::Log] = (void *) +[](const T *a, T *b) { new (b) T(log(*a)); };
    b[ArrayOp::Exp2] = (void *) +[](const T *a, T *b) { new (b) T(exp2(*a)); };
    b[ArrayOp::Exp] = (void *) +[](const T *a, T *b) { new (b) T(exp(*a)); };
}

template <typename T> void bind_complex(ArrayBinding &b) {
    b[ArrayOp::Sin] = (void *) +[](const T *a, T *b) { new (b) T(sin(*a)); };
    b[ArrayOp::Cos] = (void *) +[](const T *a, T *b) { new (b) T(cos(*a)); };
    b[ArrayOp::Tan] = (void *) +[](const T *a, T *b) { new (b) T(tan(*a)); };
    b[ArrayOp::Sincos] = (void *) +[](const T *a, T *b, T *c) {
        auto [sa, ca] = sincos(*a);
        new (b) T(std::move(sa));
        new (c) T(std::move(ca));
    };
    b[ArrayOp::Asin] = (void *) +[](const T *a, T *b) { new (b) T(asin(*a)); };
    b[ArrayOp::Acos] = (void *) +[](const T *a, T *b) { new (b) T(acos(*a)); };
    b[ArrayOp::Atan] = (void *) +[](const T *a, T *b) { new (b) T(atan(*a)); };
    b[ArrayOp::Sinh] = (void *) +[](const T *a, T *b) { new (b) T(sinh(*a)); };
    b[ArrayOp::Cosh] = (void *) +[](const T *a, T *b) { new (b) T(cosh(*a)); };
    b[ArrayOp::Tanh] = (void *) +[](const T *a, T *b) { new (b) T(tanh(*a)); };
    b[ArrayOp::Sincosh] = (void *) +[](const T *a, T *b, T *c) {
        auto [sa, ca] = sincosh(*a);
        new (b) T(std::move(sa));
        new (c) T(std::move(ca));
    };
    b[ArrayOp::Asinh] = (void *) +[](const T *a, T *b) { new (b) T(asinh(*a)); };
    b[ArrayOp::Acosh] = (void *) +[](const T *a, T *b) { new (b) T(acosh(*a)); };
    b[ArrayOp::Atanh] = (void *) +[](const T *a, T *b) { new (b) T(atanh(*a)); };
}


template <typename T>
nanobind::object bind_array(ArrayBinding &b, nanobind::handle scope = {},
                            const char *name = nullptr) {
    namespace nb = nanobind;

    bind_init<T>(b, scope, name);

    if constexpr (T::IsTensor) {
        bind_tensor<T>(b);
    } else {
        bind_base<T>(b);

        if constexpr ((T::IsArithmetic || T::IsClass) && T::Depth == 1 && !is_special_v<T>)
            bind_cast<T>(b);

        if constexpr (T::Depth == 1 && T::IsDynamic) {
            if constexpr (T::IsArithmetic) {
                bind_arithmetic<T>(b);
            }

            if constexpr (T::IsIntegral)
                bind_int_arithmetic<T>(b);

            if constexpr (T::IsFloat)
                bind_float_arithmetic<T>(b);

            if constexpr (T::IsMask)
                bind_mask_reductions<T>(b);

            if constexpr (T::IsMask || T::IsIntegral)
                bind_bit_ops<T>(b);

            if constexpr (T::IsJIT)
                bind_jit_ops<T>(b);

            bind_select<T>(b);
            bind_richcmp<T>(b);
            bind_memop<T>(b);
        }

        if constexpr (T::Depth == 1 && (T::IsMask || T::IsIntegral))
            bind_bit_invert<T>(b);
    }


    if constexpr (T::IsComplex || T::IsQuaternion || T::IsMatrix)
        bind_special<T>(b);

    if constexpr (T::IsComplex || T::IsQuaternion)
        bind_complex_and_quaternion<T>(b);

    if constexpr (T::IsComplex)
        bind_complex<T>(b);

    if constexpr (!T::IsArithmetic)
        disable_arithmetic(b);

    if constexpr (!T::IsArithmetic && !T::IsClass)
        disable_cast(b);

    if constexpr (!T::IsIntegral)
        disable_int_arithmetic(b);

    if constexpr (!T::IsFloat)
        disable_float_arithmetic(b);

    if constexpr (!T::IsMask)
        disable_mask_reductions(b);

    if constexpr (!T::IsMask && !T::IsIntegral)
        disable_bit_ops(b);

    #if defined(DRJIT_PYTHON_BUILD)
        nb::object result = bind(b);
    #else
        nb::object bind_func = nb::module_::import_("drjit.detail").attr("bind");
        nb::object result = bind_func(nb::cast((void *) &b));
    #endif

    if constexpr (std::is_pointer_v<value_t<T>>)
        result.attr("Domain") = T::CallSupport::Domain;

    return result;
}

template <typename T>
nanobind::class_<T> bind_array_t(ArrayBinding &b, nanobind::handle scope = {},
                                 const char *name = nullptr) {
        return nanobind::borrow<nanobind::class_<T>>(
            bind_array<T>(b, scope, name));
}

/// Run bind_array() for many different plain array types
template <typename T> void bind_array_types(ArrayBinding &b) {
    bind_array<mask_t<T>>(b);
    bind_array<int32_array_t<T>>(b);
    bind_array<uint32_array_t<T>>(b);
    bind_array<int64_array_t<T>>(b);
    bind_array<uint64_array_t<T>>(b);
    bind_array<float16_array_t<T>>(b);
    bind_array<float32_array_t<T>>(b);
    bind_array<float64_array_t<T>>(b);
}

/// Run bind_array() for many different matrix types
template <typename T, size_t Size> void bind_matrix_types(ArrayBinding &b) {
    using VecF16 = Array<float16_array_t<T>, Size>;
    using VecF32 = Array<float32_array_t<T>, Size>;
    using VecF64 = Array<float64_array_t<T>, Size>;
    using VecMask = mask_t<VecF32>;

    bind_array<Mask<VecMask, Size>>(b);
    bind_array<Array<VecF16, Size>>(b);
    bind_array<Array<VecF32, Size>>(b);
    bind_array<Array<VecF64, Size>>(b);
    bind_array<Matrix<float16_array_t<T>, Size>>(b);
    bind_array<Matrix<float32_array_t<T>, Size>>(b);
    bind_array<Matrix<float64_array_t<T>, Size>>(b);
}

template <typename T, size_t VecSize, size_t Size>
void bind_matrix_vec_types(ArrayBinding &b) {
    using VecF16 = Array<float16_array_t<T>, VecSize>;
    using VecF32 = Array<float32_array_t<T>, VecSize>;
    using VecF64 = Array<float64_array_t<T>, VecSize>;
    using VecMask = mask_t<VecF32>;

    bind_array<mask_t<Array<VecMask, Size>>>(b);
    bind_array<Array<VecF16, Size>>(b);
    bind_array<Array<VecF32, Size>>(b);
    bind_array<Array<VecF64, Size>>(b);

    bind_array<mask_t<Array<Array<VecMask, Size>, Size>>>(b);
    bind_array<Array<Array<VecF16, Size>,Size>>(b);
    bind_array<Array<Array<VecF32, Size>,Size>>(b);
    bind_array<Array<Array<VecF64, Size>,Size>>(b);
    bind_array<Matrix<VecF16, Size>>(b);
    bind_array<Matrix<VecF32, Size>>(b);
    bind_array<Matrix<VecF64, Size>>(b);
}

/// Run bind_array() for arrays, matrices, quaternions, complex numbers, and tensors
template <typename T> void bind_all(ArrayBinding &b) {
    if constexpr (!drjit::detail::is_scalar_v<T>)
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

    bind_matrix_vec_types<T, 1, 4>(b);
    bind_matrix_vec_types<T, 3, 4>(b);
    bind_matrix_vec_types<T, 4, 3>(b);
    bind_matrix_vec_types<T, 4, 4>(b);

    bind_array<Complex<float32_array_t<T>>>(b);
    bind_array<Complex<float64_array_t<T>>>(b);
    bind_array<Quaternion<float16_array_t<T>>>(b);
    bind_array<Quaternion<float32_array_t<T>>>(b);
    bind_array<Quaternion<float64_array_t<T>>>(b);

    using T2 = std::conditional_t<drjit::detail::is_scalar_v<T>, DynamicArray<T>, T>;
    bind_array<Tensor<mask_t<T2>>>(b);
    bind_array<Tensor<int32_array_t<T2>>>(b);
    bind_array<Tensor<int64_array_t<T2>>>(b);
    bind_array<Tensor<uint32_array_t<T2>>>(b);
    bind_array<Tensor<uint64_array_t<T2>>>(b);
    bind_array<Tensor<float16_array_t<T2>>>(b);
    bind_array<Tensor<float32_array_t<T2>>>(b);
    bind_array<Tensor<float64_array_t<T2>>>(b);
}

// Expose already existing object tree traversal callbacks (T::traverse_1_..) in Python.
// This functionality is needed to traverse custom/opaque C++ classes and correctly
// update their members when they are used in vectorized loops, function calls, etc.
template <typename T, typename... Args> auto& bind_traverse(nanobind::class_<T, Args...> &cls) {
    namespace nb = nanobind;
    struct Payload { nb::callable c; };

    cls.def("_traverse_1_cb_ro", [](const T *self, nb::callable c) {
        Payload payload{ std::move(c) };
        self->traverse_1_cb_ro((void *) &payload, [](void *p, uint64_t index) {
            ((Payload *) p)->c(index);
        });
    });

    cls.def("_traverse_1_cb_rw", [](T *self, nb::callable c) {
        Payload payload{ std::move(c) };
        self->traverse_1_cb_rw((void *) &payload, [](void *p, uint64_t index) {
            return nb::cast<uint64_t>(((Payload *) p)->c(index));
        });
    });

    return cls;
}

NAMESPACE_END(drjit)
