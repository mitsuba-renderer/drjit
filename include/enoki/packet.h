/*
    enoki/packet.h -- Packet arrays for various instruction sets

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>
#include <enoki/packet_intrin.h>
#include <enoki/packet_recursive.h>

#if defined(ENOKI_X86_AVX512)
#  include <enoki/packet_kmask.h>
#endif

#if defined(ENOKI_X86_SSE42)
#  include <enoki/packet_sse42.h>
#endif

#if defined(ENOKI_X86_AVX)
#  include <enoki/packet_avx.h>
#endif

#if defined(ENOKI_X86_AVX2)
#  include <enoki/packet_avx2.h>
#endif

#if defined(ENOKI_X86_AVX512)
#  include <enoki/packet_avx512.h>
#endif

NAMESPACE_BEGIN(enoki)

template <typename Value_, size_t Size_>
struct Packet : StaticArrayImpl<Value_, Size_, false, Packet<Value_, Size_>> {
    using Base = StaticArrayImpl<Value_, Size_, false, Packet<Value_, Size_>>;

    static_assert(!is_mask_v<Value_>,
                  "Can't create an 'enoki::Packet', whose elements are masks "
                  "(use 'enoki::Mask' instead)");

    using ArrayType = Packet;
    using MaskType = PacketMask<Value_, Size_>;

    /// Packet types prefer to be broadcasted to the *inner* dimensions of a N-D array
    static constexpr bool BroadcastOuter = false;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = Packet<T, Size_>;

    ENOKI_ARRAY_IMPORT(Packet, Base)
};

template <typename Value_, size_t Size_>
struct PacketMask : MaskBase<Value_, Size_, PacketMask<Value_, Size_>> {
    using Base = MaskBase<Value_, Size_, PacketMask<Value_, Size_>>;

    using MaskType = PacketMask;
    using ArrayType = Packet<Value_, Size_>;

    /// Packet types prefer to be broadcasted to the *inner* dimensions of a N-D array
    static constexpr bool BroadcastOuter = false;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = PacketMask<T, Size_>;

    ENOKI_ARRAY_IMPORT(PacketMask, Base)
};

NAMESPACE_END(enoki)
