/*
    drjit/array_traits.h -- Type traits for Dr.Jit arrays

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/fwd.h>
#include <utility>
#include <stdint.h>

NAMESPACE_BEGIN(drjit)

using ssize_t = std::make_signed_t<size_t>;

// -----------------------------------------------------------------------
//! @{ \name General type traits (not specific to Dr.Jit arrays)
// -----------------------------------------------------------------------

/// Convenience wrapper around std::enable_if
template <bool B> using enable_if_t = std::enable_if_t<B, int>;

namespace detail {
    /// Identity function for types
    template <typename T, typename...> struct identity {
        using type = T;
    };

    /// Detector pattern that is used to drive many type traits below
    template <typename SFINAE, template <typename> typename Op, typename Arg>
    struct detector : std::false_type { };

    template <template <typename> typename Op, typename Arg>
    struct detector<std::void_t<Op<Arg>>, Op, Arg>
        : std::true_type { };

    template <typename...> constexpr bool false_v = false;

    template <typename T>
    constexpr bool is_integral_ext_v =
        std::is_integral_v<T> || std::is_enum_v<T> || std::is_pointer_v<T>;

    /// Relaxed type equivalence to work around 'long' vs 'long long' differences
    template <typename T0, typename T1>
    static constexpr bool is_same_v =
        sizeof(T0) == sizeof(T1) &&
        std::is_floating_point_v<T0> == std::is_floating_point_v<T1> &&
        std::is_signed_v<T0> == std::is_signed_v<T1> &&
        is_integral_ext_v<T0> == is_integral_ext_v<T1>;

    /// SFINAE checker for component-based array constructors
    template <size_t Size, typename... Ts>
    using enable_if_components_t = enable_if_t<sizeof...(Ts) == Size && Size != 1 &&
              (!std::is_same_v<Ts, reinterpret_flag> && ...)>;

    template <bool... Args> constexpr bool and_v = (Args && ...);
}

/// True for any type that can reasonably be packed into a 32 bit integer array
template <typename T>
using enable_if_int32_t = enable_if_t<sizeof(T) == 4 && detail::is_integral_ext_v<T>>;

/// True for any type that can reasonably be packed into a 64 bit integer array
template <typename T>
using enable_if_int64_t = enable_if_t<sizeof(T) == 8 && detail::is_integral_ext_v<T>>;

template <typename... Ts> using identity_t = typename detail::identity<Ts...>::type;

template <template<typename> class Op, typename Arg>
constexpr bool is_detected_v = detail::detector<void, Op, Arg>::value;

constexpr size_t Dynamic = (size_t) -1;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Type traits for querying the properties of Dr.Jit arrays
// -----------------------------------------------------------------------

namespace detail {
    template <typename T> using is_array_det           = std::enable_if_t<T::IsDrJit>;
    template <typename T> using is_masked_array_det    = std::enable_if_t<T::IsDrJit && T::Derived::IsMaskedArray>;
    template <typename T> using is_static_array_det    = std::enable_if_t<T::IsDrJit && T::Derived::Size != Dynamic>;
    template <typename T> using is_dynamic_array_det   = std::enable_if_t<T::IsDrJit && T::Derived::Size == Dynamic>;
    template <typename T> using is_packed_array_det    = std::enable_if_t<T::IsDrJit && T::Derived::IsPacked>;
    template <typename T> using is_recursive_array_det = std::enable_if_t<T::IsDrJit && T::Derived::IsRecursive>;
    template <typename T> using is_cuda_det            = std::enable_if_t<T::IsDrJit && T::Derived::IsCUDA>;
    template <typename T> using is_llvm_det            = std::enable_if_t<T::IsDrJit && T::Derived::IsLLVM>;
    template <typename T> using is_jit_det             = std::enable_if_t<T::IsDrJit && T::Derived::IsJIT>;
    template <typename T> using is_diff_det            = std::enable_if_t<T::IsDrJit && T::Derived::IsDiff>;
    template <typename T> using is_mask_det            = std::enable_if_t<T::IsDrJit && T::Derived::IsMask>;
    template <typename T> using is_kmask_det           = std::enable_if_t<T::IsDrJit && T::Derived::IsKMask>;
    template <typename T> using is_complex_det         = std::enable_if_t<T::Derived::IsComplex>;
    template <typename T> using is_matrix_det          = std::enable_if_t<T::Derived::IsMatrix>;
    template <typename T> using is_vector_det          = std::enable_if_t<T::Derived::IsVector>;
    template <typename T> using is_quaternion_det      = std::enable_if_t<T::Derived::IsQuaternion>;
    template <typename T> using is_tensor_det          = std::enable_if_t<T::Derived::IsTensor>;
    template <typename T> using is_special_det         = std::enable_if_t<T::Derived::IsSpecial>;
    template <typename T> using is_dynamic_det         = std::enable_if_t<T::IsDynamic>;
    template <typename T> using is_drjit_struct_det    = std::enable_if_t<T::IsDrJitStruct>;
}

template <typename T> using enable_if_scalar_t = enable_if_t<std::is_scalar_v<T>>;

template <typename T>
constexpr bool is_array_v = is_detected_v<detail::is_array_det, std::decay_t<T>>;
template <typename T> using enable_if_array_t = enable_if_t<is_array_v<T>>;
template <typename T> using enable_if_not_array_t = enable_if_t<!is_array_v<T>>;

template <typename T>
constexpr bool is_masked_array_v = is_detected_v<detail::is_masked_array_det, std::decay_t<T>>;
template <typename T> using enable_if_masked_array_t = enable_if_t<is_masked_array_v<T>>;

template <typename T>
constexpr bool is_static_array_v = is_detected_v<detail::is_static_array_det, std::decay_t<T>>;
template <typename T> using enable_if_static_array_t = enable_if_t<is_static_array_v<T>>;

template <typename T>
constexpr bool is_dynamic_array_v = is_detected_v<detail::is_dynamic_array_det, std::decay_t<T>>;
template <typename T> using enable_if_dynamic_array_t = enable_if_t<is_dynamic_array_v<T>>;

template <typename T>
constexpr bool is_dynamic_v = is_detected_v<detail::is_dynamic_det, std::decay_t<T>>;
template <typename T> using enable_if_dynamic_t = enable_if_t<is_dynamic_v<T>>;

template <typename T>
constexpr bool is_packed_array_v = is_detected_v<detail::is_packed_array_det, std::decay_t<T>>;
template <typename T> using enable_if_packed_array_t = enable_if_t<is_packed_array_v<T>>;

template <typename T>
constexpr bool is_cuda_v = is_detected_v<detail::is_cuda_det, std::decay_t<T>>;
template <typename T> using enable_if_cuda_array_t = enable_if_t<is_cuda_v<T>>;

template <typename T>
constexpr bool is_llvm_v = is_detected_v<detail::is_llvm_det, std::decay_t<T>>;
template <typename T> using enable_if_llvm_array_t = enable_if_t<is_llvm_v<T>>;

template <typename T>
constexpr bool is_jit_v = is_detected_v<detail::is_jit_det, std::decay_t<T>>;
template <typename T> using enable_if_jit_array_t = enable_if_t<is_jit_v<T>>;

template <typename T>
constexpr bool is_diff_v = is_detected_v<detail::is_diff_det, std::decay_t<T>>;
template <typename T> using enable_if_diff_array_t = enable_if_t<is_diff_v<T>>;

template <typename T>
constexpr bool is_recursive_array_v = is_detected_v<detail::is_recursive_array_det, std::decay_t<T>>;
template <typename T> using enable_if_recursive_array_t = enable_if_t<is_recursive_array_v<T>>;

template <typename T>
constexpr bool is_mask_v = std::is_same_v<T, bool> || is_detected_v<detail::is_mask_det, std::decay_t<T>>;
template <typename T>
constexpr bool is_kmask_v = is_detected_v<detail::is_kmask_det, std::decay_t<T>>;
template <typename T> using enable_if_mask_t = enable_if_t<is_mask_v<T>>;

template <typename... Ts> constexpr bool is_array_any_v = (is_array_v<Ts> || ...);
template <typename... Ts> using enable_if_array_any_t = enable_if_t<is_array_any_v<Ts...>>;

template <typename T>
constexpr bool is_complex_v = is_detected_v<detail::is_complex_det, std::decay_t<T>>;
template <typename T> using enable_if_complex_t = enable_if_t<is_complex_v<T>>;

template <typename T>
constexpr bool is_matrix_v = is_detected_v<detail::is_matrix_det, std::decay_t<T>>;
template <typename T> using enable_if_matrix_t = enable_if_t<is_matrix_v<T>>;

template <typename T>
constexpr bool is_vector_v = is_detected_v<detail::is_vector_det, std::decay_t<T>>;
template <typename T> using enable_if_vector_t = enable_if_t<is_vector_v<T>>;

template <typename T>
constexpr bool is_quaternion_v = is_detected_v<detail::is_quaternion_det, std::decay_t<T>>;
template <typename T> using enable_if_quaternion_t = enable_if_t<is_quaternion_v<T>>;

template <typename T>
constexpr bool is_tensor_v = is_detected_v<detail::is_tensor_det, std::decay_t<T>>;
template <typename T> using enable_if_tensor_t = enable_if_t<is_tensor_v<T>>;

template <typename T>
constexpr bool is_special_v = is_detected_v<detail::is_special_det, std::decay_t<T>>;
template <typename T> using enable_if_special_t = enable_if_t<is_special_v<T>>;

template <typename T> struct struct_support {
    using type = T;
    static constexpr bool Defined =
        is_detected_v<detail::is_drjit_struct_det, type>;
};

template <typename T> constexpr bool is_drjit_struct_v = struct_support<std::decay_t<T>>::Defined;
template <typename T> using enable_if_drjit_struct_t = enable_if_t<is_drjit_struct_v<T>>;

namespace detail {
    template <typename T, typename = int> struct scalar {
        using type = std::decay_t<T>;
    };

    template <typename T> struct scalar<T, enable_if_array_t<T>> {
        using type = typename std::decay_t<T>::Derived::Scalar;
    };

    template <typename T, typename = int> struct value {
        using type = std::decay_t<T>;
    };

    template <typename T> struct value<T, enable_if_array_t<T>> {
        using type = typename std::decay_t<T>::Derived::Value;
    };

    template <typename T, typename = int> struct array_depth {
        static constexpr size_t value = 0;
    };

    template <typename T> struct array_depth<T, enable_if_array_t<T>> {
        static constexpr size_t value = std::decay_t<T>::Derived::Depth;
    };

    template <typename T, typename = int> struct array_size {
        static constexpr size_t value = 1;
    };

    template <typename T> struct array_size<T, enable_if_array_t<T>> {
        static constexpr size_t value = std::decay_t<T>::Derived::Size;
    };
}

/// Type trait to access the base scalar type underlying a potentially nested array
template <typename T> using scalar_t = typename detail::scalar<T>::type;

/// Type trait to access the value type of an array
template <typename T> using value_t = typename detail::value<T>::type;

/// Determine the depth of a nested Dr.Jit array (scalars evaluate to zero)
template <typename T> constexpr size_t array_depth_v = detail::array_depth<T>::value;

/// Determine the size of a nested Dr.Jit array (scalars evaluate to one)
template <typename T> constexpr size_t array_size_v = detail::array_size<T>::value;

template <typename T> constexpr bool is_floating_point_v = std::is_floating_point_v<scalar_t<T>> && !is_mask_v<T>;
template <typename T> constexpr bool is_integral_v = std::is_integral_v<scalar_t<T>> && !is_mask_v<T>;
template <typename T> constexpr bool is_arithmetic_v = std::is_arithmetic_v<scalar_t<T>> && !is_mask_v<T>;
template <typename T> constexpr bool is_signed_v = std::is_signed_v<scalar_t<T>>;
template <typename T> constexpr bool is_unsigned_v = std::is_unsigned_v<scalar_t<T>>;

namespace detail {
    template <typename T, typename = int> struct mask {
        using type = bool;
    };

    template <typename T> struct mask<MaskedArray<T>> {
        using type = MaskedArray<typename mask<T>::type>;
    };

    template <typename T> struct mask<T, enable_if_t<is_array_v<T> && !is_masked_array_v<T>>> {
        using type = typename std::decay_t<T>::Derived::MaskType;
    };

    template <typename T, typename = int> struct array {
        using type = T;
    };

    template <typename T> struct array<T, enable_if_array_t<T>> {
        using type = typename std::decay_t<T>::Derived::ArrayType;
    };

    template <typename T, typename = int> struct plain {
        using type = T;
    };

    template <typename T> struct plain<T, enable_if_special_t <T>> {
        using type = typename std::decay_t<T>::Derived::PlainArrayType;
    };
}

/// Type trait to access the mask type underlying an array
template <typename T> using mask_t = typename detail::mask<T>::type;

/// Type trait to access the array type underlying a mask
template <typename T> using array_t = typename detail::array<T>::type;

/// Type trait to access the plain array type underlying an special array
template <typename T> using plain_t = typename detail::plain<T>::type;

template <typename T>
using struct_support_t = typename struct_support<std::decay_t<T>>::type;

namespace detail {
    template <typename T, typename = int> struct backend {
        static constexpr JitBackend value = (JitBackend) 0u;
    };

    template <typename T> struct backend<T, enable_if_cuda_array_t<T>> {
        static constexpr JitBackend value = JitBackend::CUDA;
    };

    template <typename T> struct backend<T, enable_if_llvm_array_t<T>> {
        static constexpr JitBackend value = JitBackend::LLVM;
    };
}

/// Determine the backend of an Dr.Jit array (scalars evaluate to 0)
template <typename T> constexpr JitBackend backend_v = detail::backend<T>::value;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Traits for differentiable array
// -----------------------------------------------------------------------

namespace detail {
    template <typename = int, typename... Ts> struct leaf_array;
};

/* Get the lowest-level array type underlying a potentially nested array,
   prefer floating point arrays when multiple options are available. */
template <typename... Ts>
using leaf_array_t = typename detail::leaf_array<int, std::decay_t<Ts>...>::type;

namespace detail {
    template <> struct leaf_array<int> {
        using type = void;
    };

    template <typename T>
    struct leaf_array<enable_if_t<!is_array_v<T> && !is_drjit_struct_v<T>>, T> {
        using type = T;
    };

    template <typename T> struct leaf_array<enable_if_array_t<T>, T> {
        using type = std::conditional_t<
            is_array_v<value_t<T>>,
            leaf_array_t<value_t<T>>, T
        >;
    };

    template <template <typename...> typename Base, typename... Ts>
    struct leaf_array<enable_if_drjit_struct_t<Base<Ts...>>, Base<Ts...>> {
        using type = leaf_array_t<Ts...>;
    };

    template <template <typename, size_t> typename Base, typename T, size_t S>
    struct leaf_array<enable_if_drjit_struct_t<Base<T, S>>, Base<T, S>> {
        using type = leaf_array_t<T>;
    };

    template <typename T0, typename T1, typename... Ts> struct leaf_array<int, T0, T1, Ts...> {
        using T0L = leaf_array_t<T0>;
        using TsL = leaf_array_t<T1, Ts...>;

        using type = std::conditional_t<
            is_array_v<T0L> && std::is_floating_point_v<scalar_t<T0L>>,
            T0L,
            TsL
        >;
    };

    template <typename T, typename = int> struct diff_array { using type = void; };

    template <typename T> struct diff_array<T, enable_if_t<!T::IsDiff && T::IsJIT && T::Depth != 1>> {
        using type = typename std::decay_t<T>::Derived::template ReplaceValue<
            typename diff_array<value_t<T>>::type>;
    };

    template <typename T> struct diff_array<T, enable_if_t<!T::IsDiff && T::IsJIT && T::Depth == 1>> {
        using type = DiffArray<T>;
    };

    template <typename T, typename = int> struct detached {
        using type = T;
    };

    template <typename T> struct detached<T, enable_if_t<T::IsDiff && T::Depth != 1>> {
        using type = typename std::decay_t<T>::Derived::template ReplaceValue<
            typename detached<value_t<T>>::type>;
    };

    template <typename T> struct detached<T, enable_if_t<T::IsDiff && T::Depth == 1 && !T::IsTensor>> {
        using type = typename std::decay_t<T>::Type;
    };

    template <typename T> struct detached<T, enable_if_t<T::IsDiff && T::Depth == 1 && T::IsTensor>> {
        using type = Tensor<typename detached<typename std::decay_t<T>::Array>::type>;
    };

    template <template <typename...> typename Base, typename... Ts>
    struct detached<Base<Ts...>, enable_if_drjit_struct_t<Base<Ts...>>> {
        using type = Base<typename detached<Ts>::type...>;
    };

    template <template <typename, size_t> typename Base, typename T, size_t S>
    struct detached<Base<T, S>, enable_if_drjit_struct_t<Base<T, S>>> {
        using type = Base<typename detached<T>::type, S>;
    };

    template <typename T, typename = int> struct masked {
        using type = MaskedArray<T>;
    };

    template <template <typename...> typename Base, typename... Ts>
    struct masked<Base<Ts...>, enable_if_drjit_struct_t<Base<Ts...>>> {
        using type = Base<typename masked<Ts>::type...>;
    };

    template <template <typename, size_t> typename Base, typename T, size_t S>
    struct masked<Base<T, S>, enable_if_drjit_struct_t<Base<T, S>>> {
        using type = Base<typename masked<T>::type, S>;
    };
};

/// Convert a non-differentiable array type into a differentiable one
template <typename T>
using diff_array_t = typename detail::diff_array<T>::type;

/// Convert a differentiable array type into a non-differentiable one
template <typename T>
using detached_t = typename detail::detached<std::decay_t<T>>::type;

/// Get the type of the masked(..) expression
template <typename T>
using masked_t = typename detail::masked<std::decay_t<T>>::type;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Traits for determining the types of desired Dr.Jit arrays
// -----------------------------------------------------------------------

namespace detail {
    /// Convenience class to choose an arithmetic type based on its size and flavor
    template <size_t Size> struct sized_types { };

    template <> struct sized_types<1> {
        using Int = int8_t;
        using UInt = uint8_t;
    };

    template <> struct sized_types<2> {
        using Int = int16_t;
        using UInt = uint16_t;
        using Float = half;
    };

    template <> struct sized_types<4> {
        using Int = int32_t;
        using UInt = uint32_t;
        using Float = float;
    };

    template <> struct sized_types<8> {
        using Int = int64_t;
        using UInt = uint64_t;
        using Float = double;
    };

    template <typename T, typename Value, typename = int>
    struct replace_scalar {
        using type = Value;
    };

    template <typename T, typename Value> struct replace_scalar<T, Value, enable_if_array_t<T>> {
        using Entry = typename replace_scalar<value_t<T>, Value>::type;
        using type = typename std::decay_t<T>::Derived::template ReplaceValue<Entry>;
    };

    template <typename T, typename Value, typename = int>
    struct replace_value {
        using type = Value;
    };

    template <typename T, typename Value> struct replace_value<T, Value, enable_if_array_t<T>> {
        using type = typename std::decay_t<T>::Derived::template ReplaceValue<Value>;
    };
};

/// Replace the base scalar type of a (potentially nested) array
template <typename T, typename Value>
using replace_scalar_t = typename detail::replace_scalar<T, Value>::type;

/// Replace the value type of an array
template <typename T, typename Value>
using replace_value_t = typename detail::replace_value<T, Value>::type;

/// Integer-based version of a given array class
template <typename T>
using int_array_t = replace_scalar_t<T, typename detail::sized_types<sizeof(scalar_t<T>)>::Int>;

/// Unsigned integer-based version of a given array class
template <typename T>
using uint_array_t = replace_scalar_t<T, typename detail::sized_types<sizeof(scalar_t<T>)>::UInt>;

/// Floating point-based version of a given array class
template <typename T>
using float_array_t = replace_scalar_t<T, typename detail::sized_types<sizeof(scalar_t<T>)>::Float>;

template <typename T> using int8_array_t    = replace_scalar_t<T, int8_t>;
template <typename T> using uint8_array_t   = replace_scalar_t<T, uint8_t>;
template <typename T> using int16_array_t   = replace_scalar_t<T, int16_t>;
template <typename T> using uint16_array_t  = replace_scalar_t<T, uint16_t>;
template <typename T> using int32_array_t   = replace_scalar_t<T, int32_t>;
template <typename T> using uint32_array_t  = replace_scalar_t<T, uint32_t>;
template <typename T> using int64_array_t   = replace_scalar_t<T, int64_t>;
template <typename T> using uint64_array_t  = replace_scalar_t<T, uint64_t>;
template <typename T> using float16_array_t = replace_scalar_t<T, half>;
template <typename T> using float32_array_t = replace_scalar_t<T, float>;
template <typename T> using float64_array_t = replace_scalar_t<T, double>;
template <typename T> using bool_array_t    = replace_scalar_t<T, bool>;
template <typename T> using size_array_t    = replace_scalar_t<T, size_t>;
template <typename T> using ssize_array_t   = replace_scalar_t<T, ssize_t>;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Trait for determining the type of an expression
// -----------------------------------------------------------------------

namespace detail {
    /// Extract the most deeply nested Dr.Jit array type from a list of arguments
    template <typename... Args> struct deepest;
    template <> struct deepest<> { using type = void; };

    template <typename Arg, typename... Args> struct deepest<Arg, Args...> {
    private:
        using T0 = Arg;
        using T1 = typename deepest<Args...>::type;

        // Give precedence to differentiable arrays
        static constexpr size_t D0 = array_depth_v<T0> * 2 + (is_diff_v<T0> ? 1 : 0);
        static constexpr size_t D1 = array_depth_v<T1> * 2 + (is_diff_v<T1> ? 1 : 0);

    public:
        using type = std::conditional_t<(D1 > D0 || D0 == 0), T1, T0>;
    };

    template <typename... Ts> struct expr {
        using type = decltype((std::declval<Ts>() + ...));
    };

    template <typename T> struct expr<T> {
        using type = std::decay_t<T>;
    };

    template <typename T> struct expr<T, T> : expr<T> { };
    template <typename T> struct expr<T, T, T> : expr<T> { };
    template <typename T> struct expr<T*, std::nullptr_t> : expr<T*> { };
    template <typename T> struct expr<const T*, T*> : expr<const T*> { };
    template <typename T> struct expr<T*, const T*> : expr<const T*> { };
    template <typename T> struct expr<std::nullptr_t, T*> : expr<T*> { };

    template <typename ... Ts> using deepest_t = typename deepest<Ts...>::type;
}

/// Type trait to compute the type of an arithmetic expression involving Ts...
template <typename... Ts>
using expr_t = replace_scalar_t<typename detail::deepest_t<Ts...>,
                                typename detail::expr<scalar_t<Ts>...>::type>;

/// Intermediary for performing a cast from 'const Source &' to 'const Target &'
template <typename Source, typename Target>
using ref_cast_t =
    std::conditional_t<std::is_same_v<Source, Target>, const Target &, Target>;

/// As above, but move-construct if possible. Convert values with the wrong type.
template <typename Source, typename Target>
using move_cast_t = std::conditional_t<
    std::is_same_v<std::decay_t<Source>, Target>,
    std::conditional_t<std::is_reference_v<Source>, Source, Source &&>, Target>;


//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(drjit)
