/*
    drjit/packet.h -- Packet arrays for various instruction sets

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>
#include <drjit/packet_intrin.h>
#include <drjit/packet_recursive.h>

#if defined(DRJIT_X86_AVX512)
#  include <drjit/packet_kmask.h>
#endif

#if defined(DRJIT_X86_SSE42)
#  include <drjit/packet_sse42.h>
#endif

#if defined(DRJIT_X86_AVX)
#  include <drjit/packet_avx.h>
#endif

#if defined(DRJIT_X86_AVX2)
#  include <drjit/packet_avx2.h>
#endif

#if defined(DRJIT_X86_AVX512)
#  include <drjit/packet_avx512.h>
#endif

NAMESPACE_BEGIN(drjit)

template <typename Value_, size_t Size_>
struct Packet : StaticArrayImpl<Value_, Size_, false, Packet<Value_, Size_>> {
    using Base = StaticArrayImpl<Value_, Size_, false, Packet<Value_, Size_>>;

    static_assert(!is_mask_v<Value_>,
                  "Can't create an 'drjit::Packet', whose elements are masks "
                  "(use 'drjit::Mask' instead)");

    using ArrayType = Packet;
    using MaskType = PacketMask<Value_, Size_>;

    /// Packet types prefer to be broadcasted to the *inner* dimensions of a N-D array
    static constexpr bool BroadcastOuter = false;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = Packet<T, Size_>;

    DRJIT_ARRAY_IMPORT(Packet, Base)
};

template <typename Value_, size_t Size_>
struct PacketMask : MaskBase<Value_, Size_, PacketMask<Value_, Size_>> {
    using Base = MaskBase<Value_, Size_, PacketMask<Value_, Size_>>;

    using MaskType = PacketMask;
    using ArrayType = Packet<Value_, Size_>;
    using Value = Value_;
    using Scalar = scalar_t<Value_>;

    /// Packet types prefer to be broadcasted to the *inner* dimensions of a N-D array
    static constexpr bool BroadcastOuter = false;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = PacketMask<T, Size_>;

    DRJIT_ARRAY_IMPORT(PacketMask, Base)
};

#if defined(DRJIT_X86_SSE42)
/// Flush denormalized numbers to zero
inline void set_flush_denormals(bool value) {
    _MM_SET_FLUSH_ZERO_MODE(value ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(value ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
}

inline bool flush_denormals() {
    return _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
}

#else
inline void set_flush_denormals(bool) { }
inline bool flush_denormals() { return false; }
#endif

struct scoped_flush_denormals {
public:
    scoped_flush_denormals(bool value) {
        m_old_value = flush_denormals();
        set_flush_denormals(value);

    }

    ~scoped_flush_denormals() {
        set_flush_denormals(m_old_value);
    }
private:
    bool m_old_value;
};

NAMESPACE_END(drjit)
