/*
    drjit/packet_neon.h -- Packet arrays, ARM NEON specialization

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
DRJIT_PACKET_DECLARE(16)
DRJIT_PACKET_DECLARE(12)

NAMESPACE_BEGIN(detail)
static constexpr uint64_t arm_shuffle_helper_(int i) {
    if (i == 0)
        return 0x03020100;
    else if (i == 1)
        return 0x07060504;
    else if (i == 2)
        return 0x0B0A0908;
    else
        return 0x0F0E0D0C;
}
NAMESPACE_END(detail)

DRJIT_INLINE uint64x2_t vmvnq_u64(uint64x2_t a) {
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)));
}

DRJIT_INLINE int64x2_t vmvnq_s64(int64x2_t a) {
    return vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a)));
}

/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (single precision).
template <bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<float, 4, IsMask_, Derived_>
  : StaticArrayBase<float, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(float, 4, float32x4_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(vdupq_n_f32((float)value)) {}

    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m{v0, v1, v2, v3} {}

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------
    DRJIT_CONVERT(float) : m(a.derived().m) {}
    DRJIT_CONVERT(int32_t) : m(vcvtq_f32_s32(vreinterpretq_s32_u32(a.derived().m))) {}
    DRJIT_CONVERT(uint32_t) : m(vcvtq_f32_u32(a.derived().m)) {}
    // DRJIT_CONVERT(half) : m(vcvt_f32_f16(vld1_f16((const __fp16 *)a.data()))) {}
#if defined(DRJIT_ARM_64)
    DRJIT_CONVERT(double) : m(vcvtx_high_f32_f64(vcvtx_f32_f64(low(a).m), high(a).m)) {}
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

#define DRJIT_REINTERPRET_BOOL(type, target)                   \
    DRJIT_REINTERPRET(type) {                                  \
        m = vreinterpretq_##target##_u32(uint32x4_t {          \
            reinterpret_array<uint32_t>(a.derived().entry(0)), \
            reinterpret_array<uint32_t>(a.derived().entry(1)), \
            reinterpret_array<uint32_t>(a.derived().entry(2)), \
            reinterpret_array<uint32_t>(a.derived().entry(3))  \
        });                                                    \
    }

    DRJIT_REINTERPRET(float) : m(a.derived().m) {}
    DRJIT_REINTERPRET(int32_t) : m(vreinterpretq_f32_u32(a.derived().m)) {}
    DRJIT_REINTERPRET(uint32_t) : m(vreinterpretq_f32_u32(a.derived().m)) {}

#if defined(DRJIT_ARM_64)
    DRJIT_REINTERPRET(int64_t) : m(vreinterpretq_f32_u32(vcombine_u32(vmovn_u64(low(a).m), vmovn_u64(high(a).m)))) { }
    DRJIT_REINTERPRET(uint64_t) : m(vreinterpretq_f32_u32(vcombine_u32(vmovn_u64(low(a).m), vmovn_u64(high(a).m)))) { }
    DRJIT_REINTERPRET(double) : m(vreinterpretq_f32_u32(vcombine_u32(
        vmovn_u64(vreinterpretq_u64_f64(low(a).m)),
        vmovn_u64(vreinterpretq_u64_f64(high(a).m))))) { }
#else
    DRJIT_REINTERPRET_BOOL(int64_t, f32)
    DRJIT_REINTERPRET_BOOL(uint64_t, f32)
    DRJIT_REINTERPRET_BOOL(double, f32)
#endif

    DRJIT_REINTERPRET_BOOL(bool, f32)

#undef DRJIT_REINTERPRET_BOOL

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m{a1.entry(0), a1.entry(1), a2.entry(0), a2.entry(1)} {}

    DRJIT_INLINE Array1 low_() const { return Array1(entry(0), entry(1)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(2), entry(3)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return vaddq_f32(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const { return vsubq_f32(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const { return vmulq_f32(m, a.m); }
    DRJIT_INLINE Derived div_(Ref a) const {
#if defined(DRJIT_ARM_64)
      return vdivq_f32(m, a.m);
#else
      return Base::div_(a);
#endif
    }

#if defined(DRJIT_ARM_FMA)
    DRJIT_INLINE Derived fmadd_(Ref b, Ref c) const { return vfmaq_f32(c.m, m, b.m); }
    DRJIT_INLINE Derived fnmadd_(Ref b, Ref c) const { return vfmsq_f32(c.m, m, b.m); }
    DRJIT_INLINE Derived fmsub_(Ref b, Ref c) const { return vfmaq_f32(vnegq_f32(c.m), m, b.m); }
    DRJIT_INLINE Derived fnmsub_(Ref b, Ref c) const { return vfmsq_f32(vnegq_f32(c.m), m, b.m); }
#else
    DRJIT_INLINE Derived fmadd_(Ref b, Ref c) const { return vmlaq_f32(c.m, m, b.m); }
    DRJIT_INLINE Derived fnmadd_(Ref b, Ref c) const { return vmlsq_f32(c.m, m, b.m); }
    DRJIT_INLINE Derived fmsub_(Ref b, Ref c) const { return vmlaq_f32(vnegq_f32(c.m), m, b.m); }
    DRJIT_INLINE Derived fnmsub_(Ref b, Ref c) const { return vmlsq_f32(vnegq_f32(c.m), m, b.m); }
#endif

    template <typename T> DRJIT_INLINE Derived or_ (const T &a) const { return vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }
    template <typename T> DRJIT_INLINE Derived and_(const T &a) const { return vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }
    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const { return vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }
    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const { return vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }

    DRJIT_INLINE auto lt_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcltq_f32(m, a.m))); }
    DRJIT_INLINE auto gt_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcgtq_f32(m, a.m))); }
    DRJIT_INLINE auto le_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcleq_f32(m, a.m))); }
    DRJIT_INLINE auto ge_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcgeq_f32(m, a.m))); }

    DRJIT_INLINE auto eq_ (Ref a) const {
        if constexpr (!IsMask_)
            return mask_t<Derived>(vreinterpretq_f32_u32(vceqq_f32(m, a.m)));
        else
            return mask_t<Derived>(vceqq_u32(vreinterpretq_u32_f32(m), vreinterpretq_u32_f32(a.m)));
    }

    DRJIT_INLINE auto neq_ (Ref a) const {
        if constexpr (!IsMask_)
            return mask_t<Derived>(vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(m, a.m))));
        else
            return mask_t<Derived>(vmvnq_u32(vceqq_u32(vreinterpretq_u32_f32(m), vreinterpretq_u32_f32(a.m))));
    }

    DRJIT_INLINE Derived abs_()      const { return vabsq_f32(m); }
    DRJIT_INLINE Derived neg_()      const { return vnegq_f32(m); }
    DRJIT_INLINE Derived not_()      const { return vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(m))); }

    DRJIT_INLINE Derived minimum_(Ref b) const { return vminq_f32(b.m, m); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return vmaxq_f32(b.m, m); }

#if defined(DRJIT_ARM_64)
    DRJIT_INLINE Derived round_()    const { return vrndnq_f32(m);     }
    DRJIT_INLINE Derived floor_()    const { return vrndmq_f32(m);     }
    DRJIT_INLINE Derived ceil_()     const { return vrndpq_f32(m);     }
#endif

    DRJIT_INLINE Derived sqrt_() const {
        #if defined(DRJIT_ARM_64)
            return vsqrtq_f32(m);
        #else
            const float32x4_t inf = vdupq_n_f32(std::numeric_limits<float>::infinity());
            float32x4_t r = vrsqrteq_f32(m);
            uint32x4_t inf_or_zero = vorrq_u32(vceqq_f32(r, inf), vceqq_f32(m, inf));
            r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(r, r), m));
            r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(r, r), m));
            r = vmulq_f32(r, m);
            return vbslq_f32(inf_or_zero, m, r);
        #endif
    }

    DRJIT_INLINE Derived rcp_() const {
        float32x4_t r = vrecpeq_f32(m);
        r = vmulq_f32(r, vrecpsq_f32(r, m));
        r = vmulq_f32(r, vrecpsq_f32(r, m));
        return r;
    }

    DRJIT_INLINE Derived rsqrt_() const {
        float32x4_t r = vrsqrteq_f32(m);
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(r, r), m));
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(r, r), m));
        return r;
    }

    template <typename Mask_>
    static DRJIT_INLINE Derived select_(const Mask_ &m, Ref t, Ref f) {
        return vbslq_f32(vreinterpretq_u32_f32(m.m), t.m, f.m);
    }

    template <int I0, int I1, int I2, int I3>
    DRJIT_INLINE Derived shuffle_() const {
        /// Based on https://stackoverflow.com/a/32537433/1130282
        switch (I3 + I2*10 + I1*100 + I0*1000) {
            case 0123: return m;
            case 0000: return vdupq_lane_f32(vget_low_f32(m), 0);
            case 1111: return vdupq_lane_f32(vget_low_f32(m), 1);
            case 2222: return vdupq_lane_f32(vget_high_f32(m), 0);
            case 3333: return vdupq_lane_f32(vget_high_f32(m), 1);
            case 1032: return vrev64q_f32(m);
            case 0101: { float32x2_t vt = vget_low_f32(m); return vcombine_f32(vt, vt); }
            case 2323: { float32x2_t vt = vget_high_f32(m); return vcombine_f32(vt, vt); }
            case 1010: { float32x2_t vt = vrev64_f32(vget_low_f32(m)); return vcombine_f32(vt, vt); }
            case 3232: { float32x2_t vt = vrev64_f32(vget_high_f32(m)); return vcombine_f32(vt, vt); }
            case 0132: return vcombine_f32(vget_low_f32(m), vrev64_f32(vget_high_f32(m)));
            case 1023: return vcombine_f32(vrev64_f32(vget_low_f32(m)), vget_high_f32(m));
            case 2310: return vcombine_f32(vget_high_f32(m), vrev64_f32(vget_low_f32(m)));
            case 3201: return vcombine_f32(vrev64_f32(vget_high_f32(m)), vget_low_f32(m));
            case 3210: return vcombine_f32(vrev64_f32(vget_high_f32(m)), vrev64_f32(vget_low_f32(m)));
#if defined(DRJIT_ARM_64)
            case 0022: return vtrn1q_f32(m, m);
            case 1133: return vtrn2q_f32(m, m);
            case 0011: return vzip1q_f32(m, m);
            case 2233: return vzip2q_f32(m, m);
            case 0202: return vuzp1q_f32(m, m);
            case 1313: return vuzp2q_f32(m, m);
#endif
            case 1230: return vextq_f32(m, m, 1);
            case 2301: return vextq_f32(m, m, 2);
            case 3012: return vextq_f32(m, m, 3);

            default: {
                constexpr uint64_t prec0 = detail::arm_shuffle_helper_(I0) |
                                          (detail::arm_shuffle_helper_(I1) << 32);
                constexpr uint64_t prec1 = detail::arm_shuffle_helper_(I2) |
                                          (detail::arm_shuffle_helper_(I3) << 32);

                uint8x8x2_t tbl;
                tbl.val[0] = vreinterpret_u8_f32(vget_low_f32(m));
                tbl.val[1] = vreinterpret_u8_f32(vget_high_f32(m));

                uint8x8_t idx1 = vreinterpret_u8_u32(vcreate_u32(prec0));
                uint8x8_t idx2 = vreinterpret_u8_u32(vcreate_u32(prec1));

                float32x2_t l = vreinterpret_f32_u8(vtbl2_u8(tbl, idx1));
                float32x2_t h = vreinterpret_f32_u8(vtbl2_u8(tbl, idx2));

                return vcombine_f32(l, h);
            }
        }
    }

    template <typename Index>
    DRJIT_INLINE Derived shuffle_(const Index &index) const {
        return Base::shuffle_(index);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

#if defined(DRJIT_ARM_64)
    DRJIT_INLINE Value hmax_() const { return vmaxvq_f32(m); }
    DRJIT_INLINE Value hmin_() const { return vminvq_f32(m); }
    DRJIT_INLINE Value hsum_() const { return vaddvq_f32(m); }

    bool all_() const {
        if constexpr (Derived::Size == 4)
            return vmaxvq_s32(vreinterpretq_s32_f32(m)) < 0;
        else
            return Base::all_();
    }

    bool any_() const {
        if constexpr (Derived::Size == 4)
            return vminvq_s32(vreinterpretq_s32_f32(m)) < 0;
        else
            return Base::any_();
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        vst1q_f32((Value *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        vst1q_f32((Value *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return vld1q_f32((const Value *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return vld1q_f32((const Value *) ptr);
    }

    static DRJIT_INLINE Derived zero_(size_t) { return vdupq_n_f32(0.f); }

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

#if defined(DRJIT_ARM_64)
/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (double precision)
template <bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<double, 2, IsMask_, Derived_>
  : StaticArrayBase<double, 2, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(double, 2, float64x2_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    DRJIT_INLINE StaticArrayImpl(Value value) : m(vdupq_n_f64(value)) { }
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1) : m{v0, v1} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    DRJIT_CONVERT(double) : m(a.derived().m) { }
    DRJIT_CONVERT(int64_t) : m(vcvtq_f64_s64(vreinterpretq_s64_u64(a.derived().m))) { }
    DRJIT_CONVERT(uint64_t) : m(vcvtq_f64_u64(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET(double) : m(a.derived().m) { }
    DRJIT_REINTERPRET(int64_t) : m(vreinterpretq_f64_u64(a.derived().m)) { }
    DRJIT_REINTERPRET(uint64_t) : m(vreinterpretq_f64_u64(a.derived().m)) { }
    DRJIT_REINTERPRET(bool) {
        m = vreinterpretq_f64_u64(uint64x2_t {
            reinterpret_array<uint64_t>(a.derived().entry(0)),
            reinterpret_array<uint64_t>(a.derived().entry(1))
        });
    }
    DRJIT_REINTERPRET(float) {
        auto v0 = memcpy_cast<uint32_t>(a.derived().entry(0)),
             v1 = memcpy_cast<uint32_t>(a.derived().entry(1));
        m = vreinterpretq_f64_u32(uint32x4_t { v0, v0, v1, v1 });
    }

    DRJIT_REINTERPRET(int32_t) {
        auto v0 = memcpy_cast<uint32_t>(a.derived().entry(0)),
             v1 = memcpy_cast<uint32_t>(a.derived().entry(1));
        m = vreinterpretq_f64_u32(uint32x4_t { v0, v0, v1, v1 });
    }

    DRJIT_REINTERPRET(uint32_t) {
        auto v0 = memcpy_cast<uint32_t>(a.derived().entry(0)),
             v1 = memcpy_cast<uint32_t>(a.derived().entry(1));
        m = vreinterpretq_f64_u32(uint32x4_t { v0, v0, v1, v1 });
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m{a1.entry(0), a2.entry(0)} { }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return vaddq_f64(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const { return vsubq_f64(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const { return vmulq_f64(m, a.m); }
    DRJIT_INLINE Derived div_(Ref a) const { return vdivq_f64(m, a.m); }

#if defined(DRJIT_ARM_FMA)
    DRJIT_INLINE Derived fmadd_(Ref b, Ref c) const { return vfmaq_f64(c.m, m, b.m); }
    DRJIT_INLINE Derived fnmadd_(Ref b, Ref c) const { return vfmsq_f64(c.m, m, b.m); }
    DRJIT_INLINE Derived fmsub_(Ref b, Ref c) const { return vfmaq_f64(vnegq_f64(c.m), m, b.m); }
    DRJIT_INLINE Derived fnmsub_(Ref b, Ref c) const { return vfmsq_f64(vnegq_f64(c.m), m, b.m); }
#else
    DRJIT_INLINE Derived fmadd_(Ref b, Ref c) const { return vmlaq_f64(c.m, m, b.m); }
    DRJIT_INLINE Derived fnmadd_(Ref b, Ref c) const { return vmlsq_f64(c.m, m, b.m); }
    DRJIT_INLINE Derived fmsub_(Ref b, Ref c) const { return vmlaq_f64(vnegq_f64(c.m), m, b.m); }
    DRJIT_INLINE Derived fnmsub_(Ref b, Ref c) const { return vmlsq_f64(vnegq_f64(c.m), m, b.m); }
#endif

    template <typename T> DRJIT_INLINE Derived or_ (const T &a) const { return vreinterpretq_f64_s64(vorrq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }
    template <typename T> DRJIT_INLINE Derived and_(const T &a) const { return vreinterpretq_f64_s64(vandq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }
    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const { return vreinterpretq_f64_s64(vbicq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }
    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const { return vreinterpretq_f64_s64(veorq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }

    DRJIT_INLINE auto lt_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcltq_f64(m, a.m))); }
    DRJIT_INLINE auto gt_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcgtq_f64(m, a.m))); }
    DRJIT_INLINE auto le_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcleq_f64(m, a.m))); }
    DRJIT_INLINE auto ge_ (Ref a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcgeq_f64(m, a.m))); }

    DRJIT_INLINE auto eq_ (Ref a) const {
        if constexpr (!IsMask_)
            return mask_t<Derived>(vreinterpretq_f64_u64(vceqq_f64(m, a.m)));
        else
            return mask_t<Derived>(vceqq_u64(vreinterpretq_u64_f64(m), vreinterpretq_u64_f64(a.m)));
    }

    DRJIT_INLINE auto neq_ (Ref a) const {
        if constexpr (!IsMask_)
            return mask_t<Derived>(vreinterpretq_f64_u64(vmvnq_u64(vceqq_f64(m, a.m))));
        else
            return mask_t<Derived>(vmvnq_u64(vceqq_u64(vreinterpretq_u64_f64(m), vreinterpretq_u64_f64(a.m))));
    }

    DRJIT_INLINE Derived abs_()      const { return vabsq_f64(m); }
    DRJIT_INLINE Derived neg_()      const { return vnegq_f64(m); }
    DRJIT_INLINE Derived not_()      const { return vreinterpretq_f64_s64(vmvnq_s64(vreinterpretq_s64_f64(m))); }

    DRJIT_INLINE Derived minimum_(Ref b) const { return vminq_f64(b.m, m); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return vmaxq_f64(b.m, m); }

#if defined(DRJIT_ARM_64)
    DRJIT_INLINE Derived sqrt_()     const { return vsqrtq_f64(m);     }
    DRJIT_INLINE Derived round_()    const { return vrndnq_f64(m);     }
    DRJIT_INLINE Derived floor_()    const { return vrndmq_f64(m);     }
    DRJIT_INLINE Derived ceil_()     const { return vrndpq_f64(m);     }
#endif

    DRJIT_INLINE Derived rcp_() const {
        float64x2_t r = vrecpeq_f64(m);
        r = vmulq_f64(r, vrecpsq_f64(r, m));
        r = vmulq_f64(r, vrecpsq_f64(r, m));
        r = vmulq_f64(r, vrecpsq_f64(r, m));
        return r;
    }

    DRJIT_INLINE Derived rsqrt_() const {
        float64x2_t r = vrsqrteq_f64(m);
        r = vmulq_f64(r, vrsqrtsq_f64(vmulq_f64(r, r), m));
        r = vmulq_f64(r, vrsqrtsq_f64(vmulq_f64(r, r), m));
        r = vmulq_f64(r, vrsqrtsq_f64(vmulq_f64(r, r), m));
        return r;
    }

    template <typename Mask_>
    static DRJIT_INLINE Derived select_(const Mask_ &m, Ref t, Ref f) {
        return vbslq_f64(vreinterpretq_u64_f64(m.m), t.m, f.m);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Value hmax_() const { return vmaxvq_f64(m); }
    DRJIT_INLINE Value hmin_() const { return vminvq_f64(m); }
    DRJIT_INLINE Value hsum_() const { return vaddvq_f64(m); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        vst1q_f64((Value *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        vst1q_f64((Value *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return vld1q_f64((const Value *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return vld1q_f64((const Value *) ptr);
    }

    static DRJIT_INLINE Derived zero_(size_t) { return vdupq_n_f64(0.0); }

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;
#endif

/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (32-bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<Value_, 4, IsMask_, Derived_, enable_if_int32_t<Value_>>
  : StaticArrayBase<Value_, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(Value_, 4, uint32x4_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    DRJIT_INLINE StaticArrayImpl(Value value) : m(vdupq_n_u32((uint32_t) value)) { }
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m{(uint32_t) v0, (uint32_t) v1, (uint32_t) v2, (uint32_t) v3} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    DRJIT_CONVERT(int32_t) : m(a.derived().m) { }
    DRJIT_CONVERT(uint32_t) : m(a.derived().m) { }
    DRJIT_CONVERT(float) : m(std::is_signed_v<Value> ?
          vreinterpretq_u32_s32(vcvtq_s32_f32(a.derived().m))
        : vcvtq_u32_f32(a.derived().m)) { }
#if defined(DRJIT_ARM_64)
    DRJIT_CONVERT(int64_t) : m(vmovn_high_u64(vmovn_u64(low(a).m), high(a).m)) { }
    DRJIT_CONVERT(uint64_t) : m(vmovn_high_u64(vmovn_u64(low(a).m), high(a).m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

#define DRJIT_REINTERPRET_BOOL(type, target)                   \
    DRJIT_REINTERPRET(type) {                                  \
        m = uint32x4_t {                                       \
            reinterpret_array<uint32_t>(a.derived().entry(0)), \
            reinterpret_array<uint32_t>(a.derived().entry(1)), \
            reinterpret_array<uint32_t>(a.derived().entry(2)), \
            reinterpret_array<uint32_t>(a.derived().entry(3))  \
        };                                                     \
    }

    DRJIT_REINTERPRET(int32_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(uint32_t) : m(a.derived().m) { }
#if defined(DRJIT_ARM_64)
    DRJIT_REINTERPRET(int64_t) : m(vcombine_u32(vmovn_u64(low(a).m), vmovn_u64(high(a).m))) { }
    DRJIT_REINTERPRET(uint64_t) : m(vcombine_u32(vmovn_u64(low(a).m), vmovn_u64(high(a).m))) { }
    DRJIT_REINTERPRET(double) : m(vcombine_u32(
        vmovn_u64(vreinterpretq_u64_f64(low(a).m)),
        vmovn_u64(vreinterpretq_u64_f64(high(a).m)))) { }
#else
    DRJIT_REINTERPRET_BOOL(int64_t, u32)
    DRJIT_REINTERPRET_BOOL(uint64_t, u32)
    DRJIT_REINTERPRET_BOOL(double, u32)
#endif
    DRJIT_REINTERPRET(float) : m(vreinterpretq_u32_f32(a.derived().m)) { }
    DRJIT_REINTERPRET_BOOL(bool, u32)

#undef DRJIT_REINTERPRET_BOOL

    //! @}
    // -----------------------------------------------------------------------


    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m{(uint32_t) a1.entry(0), (uint32_t) a1.entry(1), (uint32_t) a2.entry(0), (uint32_t) a2.entry(1)} { }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0), entry(1)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(2), entry(3)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return vaddq_u32(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const { return vsubq_u32(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const { return vmulq_u32(m, a.m); }

    template <typename T> DRJIT_INLINE Derived or_ (const T &a) const { return vorrq_u32(m, a.m); }
    template <typename T> DRJIT_INLINE Derived and_(const T &a) const { return vandq_u32(m, a.m); }
    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const { return vbicq_u32(m, a.m); }
    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const { return veorq_u32(m, a.m); }

    DRJIT_INLINE auto lt_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcltq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcltq_u32(m, a.m));
    }

    DRJIT_INLINE auto gt_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcgtq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcgtq_u32(m, a.m));
    }

    DRJIT_INLINE auto le_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcleq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcleq_u32(m, a.m));
    }

    DRJIT_INLINE auto ge_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcgeq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcgeq_u32(m, a.m));
    }

    DRJIT_INLINE auto eq_ (Ref a) const { return mask_t<Derived>(vceqq_u32(m, a.m)); }
    DRJIT_INLINE auto neq_(Ref a) const { return mask_t<Derived>(vmvnq_u32(vceqq_u32(m, a.m))); }

    DRJIT_INLINE Derived abs_() const {
        if (!std::is_signed<Value>())
            return m;
        return vreinterpretq_u32_s32(vabsq_s32(vreinterpretq_s32_u32(m)));
    }

    DRJIT_INLINE Derived neg_() const {
        return vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(m)));
    }

    DRJIT_INLINE Derived not_()      const { return vmvnq_u32(m); }

    DRJIT_INLINE Derived maximum_(Ref b) const {
        if constexpr (std::is_signed_v<Value>)
            return vreinterpretq_u32_s32(vmaxq_s32(vreinterpretq_s32_u32(b.m), vreinterpretq_s32_u32(m)));
        else
            return vmaxq_u32(b.m, m);
    }

    DRJIT_INLINE Derived minimum_(Ref b) const {
        if constexpr (std::is_signed_v<Value>)
            return vreinterpretq_u32_s32(vminq_s32(vreinterpretq_s32_u32(b.m), vreinterpretq_s32_u32(m)));
        else
            return vminq_u32(b.m, m);
    }

    template <typename Mask_>
    static DRJIT_INLINE Derived select_(const Mask_ &m, Ref t, Ref f) {
        return vbslq_u32(m.m, t.m, f.m);
    }

    template <size_t Imm> DRJIT_INLINE Derived sr_() const {
        if constexpr (Imm == 0) {
            return derived();
        } else {
            if constexpr (std::is_signed_v<Value>)
                return vreinterpretq_u32_s32(
                    vshrq_n_s32(vreinterpretq_s32_u32(m), (int) Imm));
            else
                return vshrq_n_u32(m, (int) Imm);
        }
    }

    template <size_t Imm> DRJIT_INLINE Derived sl_() const {
        if constexpr (Imm == 0)
            return derived();
        else
            return vshlq_n_u32(m, (int) Imm);
    }

    DRJIT_INLINE Derived sr_(size_t k) const {
        if constexpr (std::is_signed_v<Value>)
            return vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(m), vdupq_n_s32(-(int) k)));
        else
            return vshlq_u32(m, vdupq_n_s32(-(int) k));
    }

    DRJIT_INLINE Derived sl_(size_t k) const {
        return vshlq_u32(m, vdupq_n_s32((int) k));
    }

    DRJIT_INLINE Derived sr_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(m),
                          vnegq_s32(vreinterpretq_s32_u32(a.m))));
        else
            return vshlq_u32(m, vnegq_s32(vreinterpretq_s32_u32(a.m)));
    }

    DRJIT_INLINE Derived sl_(Ref a) const {
        return vshlq_u32(m, vreinterpretq_s32_u32(a.m));
    }

#if defined(DRJIT_ARM_64)
    DRJIT_INLINE Derived mulhi_(Ref a) const {
    uint32x4_t ll, hh;
        if constexpr (std::is_signed_v<Value>) {
            int64x2_t l = vmull_s32(vreinterpret_s32_u32(vget_low_u32(m)),
                                    vreinterpret_s32_u32(vget_low_u32(a.m)));

            int64x2_t h = vmull_high_s32(vreinterpretq_s32_u32(m),
                                         vreinterpretq_s32_u32(a.m));

            ll = vreinterpretq_u32_s64(l);
            hh = vreinterpretq_u32_s64(h);
        } else {
            uint64x2_t l = vmull_u32(vget_low_u32(m),
                                     vget_low_u32(a.m));

            uint64x2_t h = vmull_high_u32(m, a.m);

            ll = vreinterpretq_u32_u64(l);
            hh = vreinterpretq_u32_u64(h);
        }
        return vuzp2q_u32(ll, hh);
    }
#endif

    DRJIT_INLINE Derived lzcnt_() const { return vclzq_u32(m); }
    DRJIT_INLINE Derived tzcnt_() const { return Value(32) - lzcnt(~derived() & (derived() - Value(1))); }
    DRJIT_INLINE Derived popcnt_() const { return vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(m)))); }

    template <int I0, int I1, int I2, int I3>
    DRJIT_INLINE Derived shuffle_() const {
        /// Based on https://stackoverflow.com/a/32537433/1130282
        switch (I3 + I2*10 + I1*100 + I0*1000) {
            case 0123: return m;
            case 0000: return vdupq_lane_u32(vget_low_u32(m), 0);
            case 1111: return vdupq_lane_u32(vget_low_u32(m), 1);
            case 2222: return vdupq_lane_u32(vget_high_u32(m), 0);
            case 3333: return vdupq_lane_u32(vget_high_u32(m), 1);
            case 1032: return vrev64q_u32(m);
            case 0101: { uint32x2_t vt = vget_low_u32(m); return vcombine_u32(vt, vt); }
            case 2323: { uint32x2_t vt = vget_high_u32(m); return vcombine_u32(vt, vt); }
            case 1010: { uint32x2_t vt = vrev64_u32(vget_low_u32(m)); return vcombine_u32(vt, vt); }
            case 3232: { uint32x2_t vt = vrev64_u32(vget_high_u32(m)); return vcombine_u32(vt, vt); }
            case 0132: return vcombine_u32(vget_low_u32(m), vrev64_u32(vget_high_u32(m)));
            case 1023: return vcombine_u32(vrev64_u32(vget_low_u32(m)), vget_high_u32(m));
            case 2310: return vcombine_u32(vget_high_u32(m), vrev64_u32(vget_low_u32(m)));
            case 3201: return vcombine_u32(vrev64_u32(vget_high_u32(m)), vget_low_u32(m));
            case 3210: return vcombine_u32(vrev64_u32(vget_high_u32(m)), vrev64_u32(vget_low_u32(m)));
#if defined(DRJIT_ARM_64)
            case 0022: return vtrn1q_u32(m, m);
            case 1133: return vtrn2q_u32(m, m);
            case 0011: return vzip1q_u32(m, m);
            case 2233: return vzip2q_u32(m, m);
            case 0202: return vuzp1q_u32(m, m);
            case 1313: return vuzp2q_u32(m, m);
#endif
            case 1230: return vextq_u32(m, m, 1);
            case 2301: return vextq_u32(m, m, 2);
            case 3012: return vextq_u32(m, m, 3);

            default: {
                constexpr uint64_t prec0 = detail::arm_shuffle_helper_(I0) |
                                          (detail::arm_shuffle_helper_(I1) << 32);
                constexpr uint64_t prec1 = detail::arm_shuffle_helper_(I2) |
                                          (detail::arm_shuffle_helper_(I3) << 32);

                uint8x8x2_t tbl;
                tbl.val[0] = vreinterpret_u8_u32(vget_low_u32(m));
                tbl.val[1] = vreinterpret_u8_u32(vget_high_u32(m));

                uint8x8_t idx1 = vreinterpret_u8_u32(vcreate_u32(prec0));
                uint8x8_t idx2 = vreinterpret_u8_u32(vcreate_u32(prec1));

                uint32x2_t l = vreinterpret_u32_u8(vtbl2_u8(tbl, idx1));
                uint32x2_t h = vreinterpret_u32_u8(vtbl2_u8(tbl, idx2));

                return vcombine_u32(l, h);
            }
        }
    }

    template <typename Index>
    DRJIT_INLINE Derived shuffle_(const Index &index) const {
        return Base::shuffle_(index);
    }


    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

#if defined(DRJIT_ARM_64)
    DRJIT_INLINE Value hmax_() const {
        if constexpr (std::is_signed_v<Value>)
            return Value(vmaxvq_s32(vreinterpretq_s32_u32(m)));
        else
            return Value(vmaxvq_u32(m));
    }

    DRJIT_INLINE Value hmin_() const {
        if constexpr (std::is_signed_v<Value>)
            return Value(vminvq_s32(vreinterpretq_s32_u32(m)));
        else
            return Value(vminvq_u32(m));
    }

    DRJIT_INLINE Value hsum_() const { return Value(vaddvq_u32(m)); }

    bool all_() const {
        if constexpr (Derived::Size == 4)
            return vmaxvq_s32(vreinterpretq_s32_u32(m)) < 0;
        else
            return Base::all_();
    }

    bool any_() const {
        if constexpr (Derived::Size == 4)
            return vminvq_s32(vreinterpretq_s32_u32(m)) < 0;
        else
            return Base::any_();
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        vst1q_u32((uint32_t *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        vst1q_u32((uint32_t *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return vld1q_u32((const uint32_t *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return vld1q_u32((const uint32_t *) ptr);
    }

    static DRJIT_INLINE Derived zero_(size_t) { return vdupq_n_u32(0); }

    //! @}
    // -----------------------------------------------------------------------
};

#if defined(DRJIT_ARM_64)
/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (64-bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<Value_, 2, IsMask_, Derived_, enable_if_int64_t<Value_>>
  : StaticArrayBase<Value_, 2, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(Value_, 2, uint64x2_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    DRJIT_INLINE StaticArrayImpl(Value value) : m(vdupq_n_u64((uint64_t) value)) { }
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1) : m{(uint64_t) v0, (uint64_t) v1} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    DRJIT_CONVERT(int64_t) : m(a.derived().m) { }
    DRJIT_CONVERT(uint64_t) : m(a.derived().m) { }
    DRJIT_CONVERT(double) : m(std::is_signed_v<Value> ?
          vreinterpretq_u64_s64(vcvtq_s64_f64(a.derived().m))
        : vcvtq_u64_f64(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET(int64_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(uint64_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(double) : m(vreinterpretq_u64_f64(a.derived().m)) { }
    DRJIT_REINTERPRET(bool) {
        m = uint64x2_t {
            reinterpret_array<uint64_t>(a.derived().entry(0)),
            reinterpret_array<uint64_t>(a.derived().entry(1))
        };
    }
    DRJIT_REINTERPRET(float) {
        auto v0 = memcpy_cast<uint32_t>(a.derived().entry(0)),
             v1 = memcpy_cast<uint32_t>(a.derived().entry(1));
        m = vreinterpretq_u64_u32(uint32x4_t { v0, v0, v1, v1 });
    }

    DRJIT_REINTERPRET(int32_t) {
        auto v0 = memcpy_cast<uint32_t>(a.derived().entry(0)),
             v1 = memcpy_cast<uint32_t>(a.derived().entry(1));
        m = vreinterpretq_u64_u32(uint32x4_t { v0, v0, v1, v1 });
    }

    DRJIT_REINTERPRET(uint32_t) {
        auto v0 = memcpy_cast<uint32_t>(a.derived().entry(0)),
             v1 = memcpy_cast<uint32_t>(a.derived().entry(1));
        m = vreinterpretq_u64_u32(uint32x4_t { v0, v0, v1, v1 });
    }

    //! @}
    // -----------------------------------------------------------------------


    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m{(uint64_t) a1.entry(0), (uint64_t) a2.entry(0)} { }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return vaddq_u64(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const { return vsubq_u64(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a_) const {
#if 1
        // Native ARM instructions + cross-domain penalities still
        // seem to be faster than the NEON approach below
        return Derived(
            entry(0) * a_.entry(0),
            entry(1) * a_.entry(1)
        );
#else
        // inp: [ah0, al0, ah1, al1], [bh0, bl0, bh1, bl1]
        uint32x4_t a = vreinterpretq_u32_u64(m),
                   b = vreinterpretq_u32_u64(a_.m);

        // uzp: [al0, al1, bl0, bl1], [bh0, bh1, ah0, ah1]
        uint32x4_t l = vuzp1q_u32(a, b);
        uint32x4_t h = vuzp2q_u32(b, a);

        uint64x2_t accum = vmull_u32(vget_low_u32(l), vget_low_u32(h));
        accum = vmlal_high_u32(accum, h, l);
        accum = vshlq_n_u64(accum, 32);
        accum = vmlal_u32(accum, vget_low_u32(l), vget_high_u32(l));

        return accum;
#endif
    }

    template <typename T> DRJIT_INLINE Derived or_ (const T &a) const { return vorrq_u64(m, a.m); }
    template <typename T> DRJIT_INLINE Derived and_(const T &a) const { return vandq_u64(m, a.m); }
    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const { return vbicq_u64(m, a.m); }
    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const { return veorq_u64(m, a.m); }

    DRJIT_INLINE auto lt_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcltq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcltq_u64(m, a.m));
    }

    DRJIT_INLINE auto gt_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcgtq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcgtq_u64(m, a.m));
    }

    DRJIT_INLINE auto le_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcleq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcleq_u64(m, a.m));
    }

    DRJIT_INLINE auto ge_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return mask_t<Derived>(vcgeq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcgeq_u64(m, a.m));
    }

    DRJIT_INLINE auto eq_ (Ref a) const { return mask_t<Derived>(vceqq_u64(m, a.m)); }
    DRJIT_INLINE auto neq_(Ref a) const { return mask_t<Derived>(vmvnq_u64(vceqq_u64(m, a.m))); }

    DRJIT_INLINE Derived abs_() const {
        if (!std::is_signed<Value>())
            return m;
        return vreinterpretq_u64_s64(vabsq_s64(vreinterpretq_s64_u64(m)));
    }

    DRJIT_INLINE Derived neg_() const {
        return vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(m)));
    }

    DRJIT_INLINE Derived not_()      const { return vmvnq_u64(m); }

    DRJIT_INLINE Derived minimum_(Ref b) const { return Derived(minimum(entry(0), b.entry(0)), minimum(entry(1), b.entry(1))); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return Derived(maximum(entry(0), b.entry(0)), maximum(entry(1), b.entry(1))); }

    template <typename Mask_>
    static DRJIT_INLINE Derived select_(const Mask_ &m, Ref t, Ref f) {
        return vbslq_u64(m.m, t.m, f.m);
    }

    template <size_t Imm> DRJIT_INLINE Derived sr_() const {
        if constexpr (Imm == 0) {
            return derived();
        } else {
            if constexpr (std::is_signed_v<Value>)
                return vreinterpretq_u64_s64(
                    vshrq_n_s64(vreinterpretq_s64_u64(m), (int) Imm));
            else
                return vshrq_n_u64(m, (int) Imm);
        }
    }

    template <size_t Imm> DRJIT_INLINE Derived sl_() const {
        if constexpr (Imm == 0)
            return derived();
        else
            return vshlq_n_u64(m, (int) Imm);
    }

    DRJIT_INLINE Derived sr_(size_t k) const {
        if constexpr (std::is_signed_v<Value>)
            return vreinterpretq_u64_s64(
                vshlq_s64(vreinterpretq_s64_u64(m), vdupq_n_s64(-(int) k)));
        else
            return vshlq_u64(m, vdupq_n_s64(-(int) k));
    }

    DRJIT_INLINE Derived sl_(size_t k) const {
        return vshlq_u64(m, vdupq_n_s64((int) k));
    }

    DRJIT_INLINE Derived sr_(Ref a) const {
        if constexpr (std::is_signed_v<Value>)
            return vreinterpretq_u64_s64(
                vshlq_s64(vreinterpretq_s64_u64(m),
                          vnegq_s64(vreinterpretq_s64_u64(a.m))));
        else
            return vshlq_u64(m, vnegq_s64(vreinterpretq_s64_u64(a.m)));
    }

    DRJIT_INLINE Derived sl_(Ref a) const {
        return vshlq_u64(m, vreinterpretq_s64_u64(a.m));
    }

    DRJIT_INLINE Derived popcnt_() const {
        return vpaddlq_u32(
            vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(m)))));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Value hsum_() const { return Value(vaddvq_u64(m)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        vst1q_u64((uint64_t *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        vst1q_u64((uint64_t *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return vld1q_u64((const uint64_t *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return vld1q_u64((const uint64_t *) ptr);
    }

    static DRJIT_INLINE Derived zero_(size_t) { return vdupq_n_u64(0); }

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;
#endif

/// Partial overload of StaticArrayImpl for the n=3 case (single precision)
template <bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<float, 3, IsMask_, Derived_>
  : StaticArrayImpl<float, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE_3D(float)

    // template <typename Derived2>
    // DRJIT_INLINE StaticArrayImpl(
    //     const StaticArrayBase<half, 3, IsMask_, Derived2> &a) {
    //     float16x4_t value;
    //     memcpy(&value, a.data(), sizeof(uint16_t)*3);
    //     m = vcvt_f32_f16(value);
    // }

    template <int I0, int I1, int I2>
    DRJIT_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    template <typename Index>
    DRJIT_INLINE Derived shuffle_(const Index &index) const {
        return Base::shuffle_(index);
    }


    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    using Base::entry;                                              \
    DRJIT_INLINE Value hmax_() const { return maximum(maximum(entry(0), entry(1)), entry(2)); }
    DRJIT_INLINE Value minimum() const { return minimum(minimum(entry(0), entry(1)), entry(2)); }
    DRJIT_INLINE Value hsum_() const { return entry(0) + entry(1) + entry(2); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        memcpy(ptr, &m, sizeof(Value) * 3);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        store_aligned_(ptr);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t size) {
        return Base::load_(ptr, size);
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Value) * 3);
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl for the n=3 case (32 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<Value_, 3, IsMask_, Derived_, enable_if_int32_t<Value_>>
  : StaticArrayImpl<Value_, 4, IsMask_, Derived_> {
    DRJIT_PACKET_TYPE_3D(Value_)
    using Base::entry;

    template <int I0, int I1, int I2>
    DRJIT_INLINE Derived shuffle_() const {
        return Derived(entry(I0), entry(I1), entry(I2));
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    DRJIT_INLINE Value hmax_() const { return maximum(maximum(entry(0), entry(1)), entry(2)); }
    DRJIT_INLINE Value hmin_() const { return minimum(minimum(entry(0), entry(1)), entry(2)); }
    DRJIT_INLINE Value hsum_() const { return entry(0) + entry(1) + entry(2); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        memcpy(ptr, &m, sizeof(Value) * 3);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        store_aligned_(ptr);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t size) {
        return Base::load_(ptr, size);
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Value) * 3);
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

NAMESPACE_END(drjit)
