/*
    drjit/fwd.h -- Preprocessor definitions and forward declarations

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#  define DRJIT_NOINLINE               __declspec(noinline)
#  define DRJIT_INLINE                 __forceinline
#  define DRJIT_INLINE_LAMBDA
#  define DRJIT_MALLOC                 __declspec(restrict)
#  define DRJIT_THREAD                 __declspec(thread)
#  define DRJIT_MAY_ALIAS
#  define DRJIT_ASSUME_ALIGNED(x, s)   x
#  define DRJIT_PACK
#  define DRJIT_LIKELY(x)              x
#  define DRJIT_UNLIKELY(x)            x
#  define DRJIT_IMPORT                 __declspec(dllimport)
#  define DRJIT_EXPORT                 __declspec(dllexport)
#  define DRJIT_TRIVIAL_ABI
#  define DRJIT_NO_UBSAN
#else
#  define DRJIT_NOINLINE               __attribute__ ((noinline))
#  define DRJIT_INLINE                 __attribute__ ((always_inline)) inline
#  define DRJIT_INLINE_LAMBDA          __attribute__ ((always_inline))
#  define DRJIT_MALLOC                 __attribute__ ((malloc))
#  define DRJIT_ASSUME_ALIGNED(x, s)   __builtin_assume_aligned(x, s)
#  define DRJIT_LIKELY(x)              __builtin_expect(!!(x), 1)
#  define DRJIT_UNLIKELY(x)            __builtin_expect(!!(x), 0)
#  define DRJIT_PACK                   __attribute__ ((packed))
#  define DRJIT_MAY_ALIAS              __attribute__ ((may_alias))
#  define DRJIT_THREAD                 __thread
#  if defined(__clang__)
#    define DRJIT_TRIVIAL_ABI            __attribute__ ((trivial_abi))
#  else
#    define DRJIT_TRIVIAL_ABI
#  endif
#  define DRJIT_IMPORT                 __attribute__ ((visibility("default")))
#  define DRJIT_EXPORT                 __attribute__ ((visibility("default")))
#  define DRJIT_NO_UBSAN               __attribute__ ((no_sanitize("undefined")))
#endif

#define DRJIT_MARK_USED(x) (void) x

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

#define DRJIT_VERSION_MAJOR 1
#define DRJIT_VERSION_MINOR 0
#define DRJIT_VERSION_PATCH 0

#define DRJIT_STRINGIFY(x) #x
#define DRJIT_TOSTRING(x)  DRJIT_STRINGIFY(x)
#define DRJIT_VERSION                                                          \
    (DRJIT_TOSTRING(DRJIT_VERSION_MAJOR) "."                                   \
     DRJIT_TOSTRING(DRJIT_VERSION_MINOR) "."                                   \
     DRJIT_TOSTRING(DRJIT_VERSION_PATCH))

#if defined(__clang__) && defined(__apple_build_version__)
#  if __clang_major__ < 10
#    error Dr.Jit requires a very recent version of AppleClang (XCode >= 10.0)
#  endif
#elif defined(__clang__)
#  if __clang_major__ < 7 && !defined(EMSCRIPTEN)
#    error Dr.Jit requires a very recent version of Clang/LLVM (>= 7.0)
#  endif
#elif defined(__GNUC__)
#  if (__GNUC__ < 8) || (__GNUC__ == 8 && __GNUC_MINOR__ < 2)
#    error Dr.Jit requires a very recent version of GCC (>= 8.2)
#  endif
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  define DRJIT_X86_64 1
#endif

#if (defined(__i386__) || defined(_M_IX86)) && !defined(DRJIT_X86_64)
#  define DRJIT_X86_32 1
#endif

#if defined(__aarch64__)
#  define DRJIT_ARM_64 1
#elif defined(__arm__)
#  define DRJIT_ARM_32 1
#endif

#if (defined(_MSC_VER) && defined(DRJIT_X86_32)) && !defined(DRJIT_DISABLE_VECTORIZATION)
/* Dr.Jit does not support vectorization on 32-bit Windows due to various
   platform limitations (unaligned stack, calling conventions don't allow
   passing vector registers, etc.). */
# define DRJIT_DISABLE_VECTORIZATION 1
#endif

# if !defined(DRJIT_DISABLE_VECTORIZATION) && !defined(DRJIT_DISABLE_AVX512)
#  if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
#    define DRJIT_X86_AVX512 1
#  endif
#  if defined(__AVX512VBMI__)
#    define DRJIT_X86_AVX512VBMI 1
#  endif
#  if defined(__AVX512VPOPCNTDQ__)
#    define DRJIT_X86_AVX512VPOPCNTDQ 1
#  endif
#endif

# if !defined(DRJIT_DISABLE_VECTORIZATION)
#  if defined(__AVX2__)
#    define DRJIT_X86_AVX2 1
#  endif
#  if defined(__FMA__)
#    define DRJIT_X86_FMA 1
#  endif
#  if defined(__F16C__)
#    define DRJIT_X86_F16C 1
#  endif
#  if defined(__BMI__)
#    define DRJIT_X86_BMI 1
#  endif
#  if defined(__BMI2__)
#    define DRJIT_X86_BMI2 1
#  endif
#  if defined(__AVX__)
#    define DRJIT_X86_AVX 1
#  endif
#  if defined(__SSE4_2__)
#    define DRJIT_X86_SSE42 1
#  endif
#  if defined(__ARM_NEON)
#    define DRJIT_ARM_NEON 1
#  endif
#  if defined(__ARM_FEATURE_FMA)
#    define DRJIT_ARM_FMA 1
#  endif
#endif

// Fix missing/inconsistent preprocessor flags
#if defined(DRJIT_X86_AVX512) && !defined(DRJIT_X86_AVX2)
#  define DRJIT_X86_AVX2 1
#endif
#if defined(DRJIT_X86_AVX2) && !defined(DRJIT_X86_AVX)
#  define DRJIT_X86_AVX 1
#endif
#if defined(DRJIT_X86_AVX) && !defined(DRJIT_X86_SSE42)
#  define DRJIT_X86_SSE42 1
#endif

#if defined(_MSC_VER)
  #if defined(DRJIT_X86_AVX2) && !defined(DRJIT_X86_F16C)
  #  define DRJIT_X86_F16C 1
  #endif
  #if defined(DRJIT_X86_AVX2) && !defined(DRJIT_X86_BMI)
  #  define DRJIT_X86_BMI 1
  #endif
  #if defined(DRJIT_X86_AVX2) && !defined(DRJIT_X86_BMI2)
  #  define DRJIT_X86_BMI2 1
  #endif
  #if defined(DRJIT_X86_AVX2) && !defined(DRJIT_X86_FMA)
  #  define DRJIT_X86_FMA 1
  #endif
#endif

/* The following macro is used by the test suite to detect
   unimplemented methods in vectorized backends */
#if !defined(DRJIT_TRACK_SCALAR)
#  define DRJIT_TRACK_SCALAR(reason) { }
#endif

#define DRJIT_CHKSCALAR(reason)                                                \
    if (drjit::detail::is_scalar_v<std::decay_t<Value>>)                                 \
        DRJIT_TRACK_SCALAR(reason)

NAMESPACE_BEGIN(drjit)

/// Maximum hardware-supported packet size in bytes
#if defined(DRJIT_X86_AVX512)
    static constexpr size_t DefaultSize = 16;
#elif defined(DRJIT_X86_AVX)
    static constexpr size_t DefaultSize = 8;
#elif defined(DRJIT_X86_SSE42) || defined(DRJIT_ARM_NEON)
    static constexpr size_t DefaultSize = 4;
#else
    static constexpr size_t DefaultSize = 1;
#endif

/// Base class of all arrays (via the Curiously recurring template pattern)
template <typename Value_, bool IsMask_, typename Derived_> struct ArrayBaseT;

/// Base class of all statically sized arrays
template <typename Value_, size_t Size_, bool IsMask_, typename Derived_>
struct StaticArrayBase;

/// Generic array class, which broadcasts from the outer to inner dimensions
template <typename Value_, size_t Size_ = DefaultSize>
struct Array;

/// Generic array class, which broadcasts from the inner to outer dimensions
template <typename Value_, size_t Size_ = DefaultSize>
struct Packet;

/// Generic mask class, which broadcasts from the outer to inner dimensions
template <typename Value_, size_t Size_ = DefaultSize>
struct Mask;

/// Generic mask class, which broadcasts from the inner to outer dimensions
template <typename Value_, size_t Size_ = DefaultSize>
struct PacketMask;

/// Naive dynamic array
template <typename Value_> struct DynamicArray;

/// JIT-compiled dynamically sized generic array
template <JitBackend Backend_, typename Value_> struct JitArray;

/// Forward- and backward-mode automatic differentiation wrapper
template <JitBackend Backend_, typename Value_> struct DiffArray;

/// Generic square matrix type
template <typename Value_, size_t Size_> struct Matrix;

/// Generic complex number type
template <typename Value_> struct Complex;

/// Generic quaternion type
template <typename Value_> struct Quaternion;

/// Generic tensor type
template <typename Array_> struct Tensor;

/// Generic texture type
template <typename Value_, size_t Dimension_> class Texture;

/// Helper class for custom data structures
template <typename T>
struct struct_support;

template <typename T, typename Array>
struct call_support {
    call_support(const Array &) { }
};

struct StringBuffer;

/// Half-precision floating point value
struct half;

namespace detail {
    struct reinterpret_flag { };
    template <typename T> struct MaskedValue;
    template <typename T> struct MaskedArray;
    template <typename T> struct MaskBit;
}

/// Enumeration to distinguish different types of primal/derivative computation
enum class ADMode { Primal, Forward, Backward };

NAMESPACE_END(drjit)
