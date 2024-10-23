
/*
    drjit/autodiff.h -- Forward/reverse-mode automatic differentiation

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/jit.h>
#include <drjit/extra.h>
#include <stdexcept>

NAMESPACE_BEGIN(drjit)

/**
 * By default, Dr.Jit's AD system destructs the enqueued input graph during
 * forward/backward mode traversal. This frees up resources, which is useful
 * when working with large wavefronts or very complex computation graphs.
 * However, this also prevents repeated propagation of gradients through a
 * shared subgraph that is being differentiated multiple times.
 *
 * To support more fine-grained use cases that require this, the following
 * flags can be used to control what should and should not be destructed.
 */
enum class ADFlag : uint32_t {
    /// None: clear nothing.
    ClearNone = 0,

    /// Delete all traversed edges from the computation graph
    ClearEdges = 1,

    /// Clear the gradients of processed input vertices (in-degree == 0)
    ClearInput = 2,

    /// Clear the gradients of processed interior vertices (out-degree != 0)
    ClearInterior = 4,

    /// Clear gradients of processed vertices only, but leave edges intact
    ClearVertices = (uint32_t) ClearInput | (uint32_t) ClearInterior,

    /// Default: clear everything (edges, gradients of processed vertices)
    Default = (uint32_t) ClearEdges | (uint32_t) ClearVertices,

    // --------------- Other flags influencing the AD traversal ---------------

    /// Don't fail when the input to a ``dr::forward`` or ``backward`` operation
    /// is not a differentiable array.
    AllowNoGrad = 8,
};

constexpr uint32_t operator |(ADFlag f1, ADFlag f2)   { return (uint32_t) f1 | (uint32_t) f2; }
constexpr uint32_t operator |(uint32_t f1, ADFlag f2) { return f1 | (uint32_t) f2; }
constexpr uint32_t operator &(ADFlag f1, ADFlag f2)   { return (uint32_t) f1 & (uint32_t) f2; }
constexpr uint32_t operator &(uint32_t f1, ADFlag f2) { return f1 & (uint32_t) f2; }
constexpr uint32_t operator ~(ADFlag f1)              { return ~(uint32_t) f1; }
constexpr uint32_t operator +(ADFlag e)               { return (uint32_t) e; }

template <JitBackend Backend_, typename Value_>
struct DRJIT_TRIVIAL_ABI DiffArray
    : ArrayBaseT<Value_, is_mask_v<Value_>, DiffArray<Backend_, Value_>> {
    static_assert(drjit::detail::is_scalar_v<Value_> || std::is_same_v<void, Value_>,
                  "Differentiable arrays can only be created over scalar types!");

    template <JitBackend, typename> friend struct DiffArray;

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Value = Value_;
    using Base = ArrayBaseT<Value_, is_mask_v<Value_>, DiffArray<Backend_, Value_>>;

    static constexpr JitBackend Backend = Backend_;

    static constexpr bool IsDiff = true;
    static constexpr bool IsArray = true;
    static constexpr bool IsDynamic = true;
    static constexpr bool IsJIT = true;
    static constexpr bool IsCUDA = Backend == JitBackend::CUDA;
    static constexpr bool IsLLVM = Backend == JitBackend::LLVM;
    static constexpr bool IsFloat = drjit::is_floating_point_v<Value_> || std::is_same_v<void, Value_>;
    static constexpr bool IsClass =
        std::is_pointer_v<Value_> &&
        std::is_class_v<std::remove_pointer_t<Value_>>;
    static constexpr size_t Size = Dynamic;

    static constexpr VarType Type =
        IsClass ? VarType::UInt32 : type_v<Value>;

    using ActualValue = std::conditional_t<IsClass, uint32_t, Value>;

    using CallSupport =
        call_support<std::decay_t<std::remove_pointer_t<Value_>>, DiffArray>;

    template <typename T> using ReplaceValue = DiffArray<Backend, T>;
    using MaskType = DiffArray<Backend, bool>;
    using ArrayType = DiffArray;

    using Index = std::conditional_t<IsFloat, uint64_t, uint32_t>;
    using Detached = JitArray<Backend, Value>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    DiffArray() = default;

    ~DiffArray() noexcept {
        if constexpr (IsFloat)
            ad_var_dec_ref(m_index);
        else
            jit_var_dec_ref(m_index);
    }

    DiffArray(const DiffArray &a) {
        if constexpr (IsFloat)
            m_index = ad_var_copy_ref(a.m_index);
        else
            m_index = jit_var_inc_ref(a.m_index);
    }

    DiffArray(DiffArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T>
    DiffArray(const DiffArray<Backend, T> &v) {
        if constexpr (IsFloat && DiffArray<Backend, T>::IsFloat)
            m_index = ad_var_cast(v.m_index, Type);
        else
            m_index = jit_var_cast((uint32_t) v.m_index, Type, 0);
    }

    template <typename T>
    DiffArray(const DiffArray<Backend, T> &v, detail::reinterpret_flag) {
        m_index = jit_var_cast((uint32_t) v.m_index, Type, 1);
    }

    DiffArray(const Detached &v) {
        m_index = jit_var_inc_ref((uint32_t) v.index());
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    DiffArray(T value) : m_index(Detached(value).release()) { }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    DiffArray(Ts&&... ts) : m_index(Detached(ts...).release()) { }

    DiffArray &operator=(const DiffArray &a) {
        Index old_index = m_index;

        if constexpr (IsFloat) {
            m_index = ad_var_copy_ref(a.m_index);
            ad_var_dec_ref(old_index);
        } else {
            m_index = jit_var_inc_ref(a.m_index);
            jit_var_dec_ref(old_index);
        }

        return *this;
    }

    DiffArray &operator=(DiffArray &&a) {
        Index temp = m_index;
        m_index = a.m_index;
        a.m_index = temp;
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations with derivative tracking
    // -----------------------------------------------------------------------

    DiffArray add_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_add(m_index, a.m_index));
        else
            return steal(jit_var_add(m_index, a.m_index));
    }

    DiffArray sub_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_sub(m_index, a.m_index));
        else
            return steal(jit_var_sub(m_index, a.m_index));
    }

    DiffArray mul_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_mul(m_index, a.m_index));
        else
            return steal(jit_var_mul(m_index, a.m_index));
    }

    DiffArray mulhi_(const DiffArray &a) const {
        return steal(jit_var_mulhi((uint32_t) m_index,
                                   (uint32_t) a.m_index));
    }

    DiffArray div_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_div(m_index, a.m_index));
        else
            return steal(jit_var_div(m_index, a.m_index));
    }

    DiffArray neg_() const {
        if constexpr (IsFloat)
            return steal(ad_var_neg(m_index));
        else
            return steal(jit_var_neg(m_index));
    }

    DiffArray not_() const {
        if (grad_enabled_())
            jit_raise("DiffArray::not_(): not permitted on attached variables!");
        return steal(jit_var_not((uint32_t) m_index));
    }

    template <typename T> DiffArray or_(const T &v) const {
        if (grad_enabled_())
            jit_raise("DiffArray::or_(): not permitted on attached variables!");
        return steal(jit_var_or((uint32_t) m_index, (uint32_t) v.m_index));
    }

    template <typename T> DiffArray and_(const T &v) const {
        if constexpr (is_mask_v<T> && !is_mask_v<DiffArray>)
            return select(v, *this, zeros<DiffArray>(size()));

        if (grad_enabled_())
            jit_raise("DiffArray::and_(): not permitted on attached variables!");
        return steal(jit_var_and((uint32_t) m_index, (uint32_t) v.m_index));
    }

    template <typename T> DiffArray xor_(const T &v) const {
        if (grad_enabled_())
            jit_raise("DiffArray::xor_(): not permitted on attached variables!");
        return steal(jit_var_xor((uint32_t) m_index, (uint32_t) v.m_index));
    }

    template <typename T> DiffArray andnot_(const T &a) const {
        return and_(a.not_());
    }

    DiffArray abs_() const {
        if constexpr (IsFloat)
            return steal(ad_var_abs(m_index));
        else
            return steal(jit_var_abs(m_index));
    }

    DiffArray rcp_() const { return steal(ad_var_rcp(m_index)); }
    DiffArray rsqrt_() const { return steal(ad_var_rsqrt(m_index)); }
    DiffArray sqrt_() const { return steal(ad_var_sqrt(m_index)); }
    DiffArray cbrt_() const { return steal(ad_var_cbrt(m_index)); }
    DiffArray erf_() const { return steal(ad_var_erf(m_index)); }
    DiffArray sin_() const { return steal(ad_var_sin(m_index)); }
    DiffArray cos_() const { return steal(ad_var_cos(m_index)); }
    std::pair<DiffArray, DiffArray> sincos_() const {
        UInt64Pair p = ad_var_sincos(m_index);
        return { steal(p.first), steal(p.second) };
    }
    DiffArray tan_() const { return steal(ad_var_tan(m_index)); }
    DiffArray csc_() const { return steal(ad_var_csc(m_index)); }
    DiffArray sec_() const { return steal(ad_var_sec(m_index)); }
    DiffArray cot_() const { return steal(ad_var_cot(m_index)); }
    DiffArray asin_() const { return steal(ad_var_asin(m_index)); }
    DiffArray acos_() const { return steal(ad_var_acos(m_index)); }
    DiffArray atan_() const { return steal(ad_var_atan(m_index)); }
    DiffArray atan2_(const DiffArray &x) const {
        return steal(ad_var_atan2(m_index, x.m_index));
    }

    DiffArray exp_() const { return steal(ad_var_exp(m_index)); }
    DiffArray exp2_() const { return steal(ad_var_exp2(m_index)); }
    DiffArray log_() const { return steal(ad_var_log(m_index)); }
    DiffArray log2_() const { return steal(ad_var_log2(m_index)); }

    DiffArray sinh_() const { return steal(ad_var_sinh(m_index)); }
    DiffArray cosh_() const { return steal(ad_var_cosh(m_index)); }
    std::pair<DiffArray, DiffArray> sincosh_() const {
        UInt64Pair p = ad_var_sincosh(m_index);
        return { steal(p.first), steal(p.second) };
    }
    DiffArray tanh_() const { return steal(ad_var_tanh(m_index)); }
    DiffArray asinh_() const { return steal(ad_var_asinh(m_index)); }
    DiffArray acosh_() const { return steal(ad_var_acosh(m_index)); }
    DiffArray atanh_() const { return steal(ad_var_atanh(m_index)); }

    DiffArray minimum_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_min(m_index, a.m_index));
        else
            return steal(jit_var_min(m_index, a.m_index));
    }

    DiffArray maximum_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_max(m_index, a.m_index));
        else
            return steal(jit_var_max(m_index, a.m_index));
    }

    DiffArray fmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (IsFloat)
            return steal(ad_var_fma(m_index, a.m_index, b.m_index));
        else
            return steal(jit_var_fma(m_index, a.m_index, b.m_index));
    }

    DiffArray fmsub_(const DiffArray &a, const DiffArray &b) const {
        DiffArray neg_b = -b;
        if constexpr (IsFloat)
            return steal(ad_var_fma(m_index, a.m_index, neg_b.m_index));
        else
            return steal(jit_var_fma(m_index, a.m_index, neg_b.m_index));
    }

    DiffArray fnmadd_(const DiffArray &a, const DiffArray &b) const {
        DiffArray neg_a = -a;
        if constexpr (IsFloat)
            return steal(ad_var_fma(m_index, neg_a.m_index, b.m_index));
        else
            return steal(jit_var_fma(m_index, neg_a.m_index, b.m_index));
    }

    DiffArray fnmsub_(const DiffArray &a, const DiffArray &b) const {
        DiffArray neg_a = -a, neg_b = -b;
        if constexpr (IsFloat)
            return steal(ad_var_fma(m_index, neg_a.m_index, neg_b.m_index));
        else
            return steal(jit_var_fma(m_index, neg_a.m_index, neg_b.m_index));
    }

    static DiffArray select_(const MaskType m,
                             const DiffArray &t,
                             const DiffArray &f) {
        if constexpr (IsFloat)
            return steal(ad_var_select(m.m_index, t.m_index, f.m_index));
        else
            return steal(jit_var_select(m.m_index, t.m_index, f.m_index));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const { return jit_var_all((uint32_t) m_index); }
    bool any_() const { return jit_var_any((uint32_t) m_index); }

    #define DRJIT_HORIZONTAL_OP(name, op)                                      \
        DiffArray name##_() const {                                            \
            if constexpr (IsFloat)                                             \
                return steal(ad_var_reduce(Backend, Type, op, m_index));       \
            else                                                               \
                return steal(jit_var_reduce(Backend, Type, op, m_index));      \
        }

    DRJIT_HORIZONTAL_OP(sum,  ReduceOp::Add)
    DRJIT_HORIZONTAL_OP(prod, ReduceOp::Mul)
    DRJIT_HORIZONTAL_OP(min,  ReduceOp::Min)
    DRJIT_HORIZONTAL_OP(max,  ReduceOp::Max)

    #undef DRJIT_HORIZONTAL_OP

    DiffArray block_reduce_(ReduceOp op, size_t block_size, int symbolic) const {
        if constexpr (IsFloat)
            return steal(ad_var_block_reduce(op, m_index, (uint32_t) block_size, symbolic));
        else
            return steal(jit_var_block_reduce(op, m_index, (uint32_t) block_size, symbolic));
    }

    DiffArray block_prefix_reduce_(ReduceOp op, uint32_t block_size, bool exclusive, bool reverse) const {
        if constexpr (IsFloat)
            return steal(ad_var_block_prefix_reduce(op, m_index, block_size, exclusive, reverse));
        else
            return steal(jit_var_block_prefix_reduce(op, m_index, block_size, exclusive, reverse));
    }

    DiffArray tile_(size_t count) const {
        if constexpr (IsFloat)
            return steal(ad_var_tile(m_index, count));
        else
            return steal(jit_var_tile(m_index, count));
    }

    DiffArray dot_(const DiffArray &a) const { return sum(*this * a); }

    auto count_() const {
        return Detached::borrow((uint32_t) m_index).count_();
    }

    auto compress_() const {
        return Detached::borrow((uint32_t) m_index).compress_();
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Operations that aren't tracked by the AD backend
    // -----------------------------------------------------------------------

    DiffArray mod_(const DiffArray &a) const {
        return steal(jit_var_mod((uint32_t) m_index, (uint32_t) a.m_index));
    }

    MaskType eq_(const DiffArray &d) const {
        return MaskType::steal(
            jit_var_eq((uint32_t) m_index, (uint32_t) d.m_index));
    }

    MaskType neq_(const DiffArray &d) const {
        return MaskType::steal(
            jit_var_neq((uint32_t) m_index, (uint32_t) d.m_index));
    }

    MaskType lt_(const DiffArray &d) const {
        return MaskType::steal(
            jit_var_lt((uint32_t) m_index, (uint32_t) d.m_index));
    }

    MaskType le_(const DiffArray &d) const {
        return MaskType::steal(
            jit_var_le((uint32_t) m_index, (uint32_t) d.m_index));
    }

    MaskType gt_(const DiffArray &d) const {
        return MaskType::steal(
            jit_var_gt((uint32_t) m_index, (uint32_t) d.m_index));
    }

    MaskType ge_(const DiffArray &d) const {
        return MaskType::steal(
            jit_var_ge((uint32_t) m_index, (uint32_t) d.m_index));
    }

    DiffArray sl_(const DiffArray &v) const {
        return steal(jit_var_shl((uint32_t) m_index, (uint32_t) v.m_index));
    }

    DiffArray sr_(const DiffArray &v) const {
        return steal(jit_var_shr((uint32_t) m_index, (uint32_t) v.m_index));
    }

    template <int Imm> DiffArray sr_() const { return sr_(Imm); }
    template <int Imm> DiffArray sl_() const { return sl_(Imm); }

    DiffArray popcnt_() const { return steal(jit_var_popc((uint32_t) m_index)); }
    DiffArray lzcnt_() const { return steal(jit_var_clz((uint32_t) m_index)); }
    DiffArray tzcnt_() const { return steal(jit_var_ctz((uint32_t) m_index)); }
    DiffArray brev_() const { return steal(jit_var_brev((uint32_t) m_index)); }

    DiffArray round_() const { return steal(jit_var_round((uint32_t) m_index)); }
    template <typename T> T round2int_() const { return T(round(*this)); }

    DiffArray floor_() const { return steal(jit_var_floor((uint32_t) m_index)); }
    template <typename T> T floor2int_() const { return T(floor(*this)); }

    DiffArray ceil_() const { return steal(jit_var_ceil((uint32_t) m_index)); }
    template <typename T> T ceil2int_() const { return T(ceil(*this)); }

    DiffArray trunc_() const { return steal(jit_var_trunc((uint32_t) m_index)); }
    template <typename T> T trunc2int_() const { return T(trunc(*this)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather support
    // -----------------------------------------------------------------------

    template <typename Index, typename Mask>
    static DiffArray gather_(const DiffArray &src, const Index &index,
                             const Mask &mask, ReduceMode mode) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);
        (void) mode;

        if constexpr (IsFloat)
            return steal(ad_var_gather(src.m_index, index.m_index, mask.m_index, mode));
        else
            return steal(jit_var_gather(src.m_index, index.m_index, mask.m_index));
    }

    template <size_t N, typename Index, typename Mask>
    static Array<DiffArray, N> gather_packet_(const DiffArray &src, const Index &index,
                                              const Mask &mask, ReduceMode mode) {
        if constexpr ((N & (N-1)) > 0) {
            return Base::template gather_packet_<N>(src, index, mask, mode);
        } else {
            static_assert(
                std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);

            Array<DiffArray, N> result;
            if constexpr (IsFloat) {
                uint64_t tmp[N];
                ad_var_gather_packet(N, src.index_combined(), index.index(), mask.index(), tmp, mode);
                for (size_t i = 0; i < N; ++i)
                    result[i] = steal(tmp[i]);
            } else {
                uint32_t tmp[N];
                jit_var_gather_packet(N, src.index_combined(), index.index(), mask.index(), tmp);
                for (size_t i = 0; i < N; ++i)
                    result[i] = steal(tmp[i]);
            }
            return result;
        }
    }

    template <typename Index, typename Mask>
    void scatter_(DiffArray &dst, const Index &index, const Mask &mask, ReduceMode mode) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);

        if constexpr (IsFloat)
            dst = steal(ad_var_scatter(dst.m_index, m_index, index.m_index,
                                       mask.m_index, ReduceOp::Identity, mode));
        else
            dst =
                steal(jit_var_scatter(dst.m_index, m_index, index.m_index,
                                      mask.m_index, ReduceOp::Identity, mode));
    }

    template <size_t N, typename Source, typename Index, typename Mask>
    static void scatter_packet_(DiffArray &dst, const Source &source,
                                const Index &index, const Mask &mask,
                                ReduceMode mode) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);
        if constexpr ((N & (N-1)) > 0) {
            Base::template scatter_packet_<N>(dst, source, index, mask, mode);
        } else if constexpr (IsFloat) {
            uint64_t indices[N];
            for (uint32_t i = 0; i < N; ++i)
                indices[i] = source[i].index_combined();

            dst = steal(ad_var_scatter_packet(N, dst.index(), indices,
                                              index.index(), mask.index(),
                                              ReduceOp::Identity, mode));
        } else {
            uint32_t indices[N];
            for (uint32_t i = 0; i < N; ++i)
                indices[i] = source[i].index();

            dst = steal(jit_var_scatter_packet(N, dst.index(), indices,
                                               index.index(), mask.index(),
                                               ReduceOp::Identity, mode));
        }
    }

    template <typename Index, typename Mask>
    void scatter_reduce_(DiffArray &dst, const Index &index,
                         const Mask &mask, ReduceOp op, ReduceMode mode) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);

        if constexpr (IsFloat)
            dst = steal(ad_var_scatter(dst.m_index, m_index, index.m_index,
                                       mask.m_index, op, mode));
        else
            dst = steal(jit_var_scatter(dst.m_index, m_index, index.m_index,
                                        mask.m_index, op, mode));
    }

    template <size_t N, typename Source, typename Index, typename Mask>
    static void scatter_reduce_packet_(DiffArray &dst, const Source &source,
                                       const Index &index, const Mask &mask,
                                       ReduceOp op, ReduceMode mode) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);
        if constexpr ((N & (N-1)) > 0) {
            Base::template scatter_reduce_packet_<N>(dst, source, index, mask, op, mode);
        } else if constexpr (IsFloat) {
            uint64_t indices[N];
            for (uint32_t i = 0; i < N; ++i)
                indices[i] = source[i].index_combined();

            dst = steal(ad_var_scatter_packet(N, dst.index(), indices,
                                              index.index(), mask.index(),
                                              op, mode));
        } else {
            uint32_t indices[N];
            for (uint32_t i = 0; i < N; ++i)
                indices[i] = source[i].index();

            dst = steal(jit_var_scatter_packet(N, dst.index(), indices,
                                               index.index(), mask.index(),
                                               op, mode));
        }
    }

    template <typename Mask>
    DiffArray scatter_inc_(const DiffArray &index, const Mask &mask) {
        return steal(jit_var_scatter_inc(&m_index, index.index(), mask.index()));
    }

    template <typename Index, typename Mask>
    void scatter_add_kahan_(DiffArray &dst_1, DiffArray &dst_2,
                            const Index &offset, const Mask &mask) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);

        ad_var_scatter_add_kahan(&dst_1.m_index, &dst_2.m_index, m_index,
                                 offset.index(), mask.index());
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fancy array initialization
    // -----------------------------------------------------------------------

    static DiffArray empty_(size_t size) {
        return steal(Detached::empty_(size).release());
    }

    static DiffArray zero_(size_t size) {
        return steal(Detached::zero_(size).release());
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    static DiffArray full_(T value, size_t size) {
        return steal(Detached::full_(value, size).release());
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    static DiffArray opaque_(T value, size_t size) {
        return steal(Detached::opaque_(value, size).release());
    }

    static DiffArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        return steal(Detached::arange_(start, stop, step).release());
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    static DiffArray linspace_(T min, T max, size_t size, bool endpoint) {
        return steal(Detached::linspace_(min, max, size, endpoint).release());
    }

    static DiffArray map_(void *ptr, size_t size, bool free = false) {
        return steal(Detached::map_(ptr, size, free).release());
    }

    static DiffArray load_(const void *ptr, size_t size) {
        return steal(Detached::load_(ptr, size).release());
    }

    static auto counter(size_t size) {
        return steal(Detached::counter(size).release());
    }

    void store_(void *ptr) const {
        Detached::borrow((uint32_t) m_index).store_(ptr);
    }

    //! @}
    // -----------------------------------------------------------------------

    static DRJIT_INLINE DiffArray steal(Index index) {
        DiffArray result;
        result.m_index = index;
        return result;
    }

    static DRJIT_INLINE DiffArray borrow(Index index) {
        DiffArray result;

        if constexpr (IsFloat)
            result.m_index = ad_var_inc_ref(index);
        else
            result.m_index = jit_var_inc_ref(index);

        return result;
    }

    Index release() {
        Index tmp = m_index;
        m_index = 0;
        return tmp;
    }

    size_t size() const { return jit_var_size((uint32_t) m_index); }

    bool grad_enabled_() const {
        if constexpr (IsFloat)
            return (m_index >> 32) != 0;
        else
            return false;
    }

    void clear_grad_() {
        if constexpr (IsFloat)
            ad_clear_grad(index_combined());
    }

    void accum_grad_(uint32_t index) {
        if constexpr (IsFloat)
            ad_accum_grad(index_combined(), index);
    }

    DiffArray copy() const {
        if constexpr (IsFloat)
            return steal(ad_var_copy(m_index));
        else
            return steal(jit_var_copy(m_index));
    }

    DiffArray migrate_(AllocType type) const {
        return steal(jit_var_migrate((uint32_t) m_index, type));
    }

    void set_grad_enabled_(bool value) {
        DRJIT_MARK_USED(value);
        if constexpr (IsFloat) {
            uint32_t jit_index = (uint32_t) m_index;

            if (value) {
                if (grad_enabled_())
                    return;
                m_index = ad_var_new(jit_index);
                jit_var_dec_ref(jit_index);
            } else {
                jit_index = jit_var_inc_ref(jit_index);
                ad_var_dec_ref(m_index);
                m_index = jit_index;
            }
        }
    }

    void new_grad_() {
        if constexpr (IsFloat) {
            Index old_index = m_index;
            m_index = ad_var_new((uint32_t) m_index);
            ad_var_dec_ref(old_index);
        }
    }

    Value entry(size_t offset) const {
        ActualValue out;
        jit_var_read((uint32_t) m_index, offset, &out);

        if constexpr (!IsClass)
            return out;
        else
            return (Value) jit_registry_ptr(Backend, CallSupport::Domain, out);
    }

    bool schedule_() const { return jit_var_schedule((uint32_t) m_index); }
    bool eval_() const { return jit_var_eval((uint32_t) m_index) != 0; }

    bool schedule_force_() {
        int rv = 0;
        *this = steal((Index) ad_var_schedule_force(m_index, &rv));
        return rv;
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    void set_entry(size_t offset, T value) {
        if (grad_enabled_())
            jit_raise("DiffArray::set_entry(): not permitted on attached variables!");

        uint32_t index;
        if constexpr (!IsClass) {
            index = jit_var_write((uint32_t) m_index, offset, &value);
        } else {
            ActualValue av = jit_registry_id(value);
            index = jit_var_write((uint32_t) m_index, (uint32_t) offset, &av);
        }
        jit_var_dec_ref((uint32_t) m_index);
        m_index = index;
    }

    const Value *data() const {
        void *p = nullptr;
        const_cast<DiffArray&>(*this) = steal((Index) ad_var_data(m_index, &p));
        return (const Value *) p;
    }

    Value *data() {
        void *p = nullptr;
        *this = steal((Index) ad_var_data(m_index, &p));
        return (Value *) p;
    }

    bool valid() const { return m_index != 0; }

    uint32_t index() const { return (uint32_t) m_index; }
    uint32_t index_ad() const { return (uint32_t) (((uint64_t) m_index) >> 32); }
    uint64_t index_combined() const { return m_index; }
    uint32_t grad_() const { return ad_grad(m_index); }
    VarState state() const { return jit_var_state((uint32_t) m_index); }

    void swap(DiffArray &a) {
        Index index = m_index;
        m_index = a.m_index;
        a.m_index = index;
    }

    template <typename... Ts> void set_label_(const Ts*... args) {
        Index index = (Index) ad_var_set_label(m_index, sizeof...(Ts), args...);
        ad_var_dec_ref(m_index);
        m_index = index;
	}

private:
    Index m_index = 0;
};

template <typename Value> using CUDADiffArray = DiffArray<JitBackend::CUDA, Value>;
template <typename Value> using LLVMDiffArray = DiffArray<JitBackend::LLVM, Value>;

template <typename T> void enqueue(ADMode mode, const T &value) {
    if constexpr (is_diff_v<T> && depth_v<T> == 1)
        ad_enqueue(mode, value.index_combined());
    else if constexpr (is_traversable_v<T>)
        traverse_1(fields(value), [mode](auto const &x) { enqueue(mode, x); });
    DRJIT_MARK_USED(mode);
    DRJIT_MARK_USED(value);
}

template <typename T1, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
void enqueue(ADMode mode, const T1 &value, const Ts&... values) {
    enqueue(mode, value);
    enqueue(mode, values...);
}

DRJIT_INLINE void enqueue(ADMode) { }

template <typename... Ts>
void traverse(ADMode mode, uint32_t flags = (uint32_t) ADFlag::Default) {
    ad_traverse(mode, flags);
}

namespace detail {
    template <bool IncRef, typename T>
    void collect_indices(const T &value, vector<uint64_t> &indices);
}

template <typename T> struct suspend_grad {
    static constexpr bool Enabled =
        is_diff_v<T> && std::is_floating_point_v<scalar_t<T>>;
    suspend_grad() : condition(true) {
        if constexpr (Enabled)
            ad_scope_enter(ADScope::Suspend, 0, nullptr, -1);
    }
    template <typename... Args>
    suspend_grad(bool when, const Args &... args) : condition(when) {
        if constexpr (Enabled) {
            if (condition) {
                vector<uint64_t> indices;
                (detail::collect_indices<false>(args, indices), ...);
                ad_scope_enter(
                    ADScope::Suspend, indices.size(), indices.data(), -1);
            }
        } else {
            (((void) args), ...);
        }
    }

    ~suspend_grad() {
        if constexpr (Enabled) {
            if (condition)
                ad_scope_leave(true);
        }
    }

    bool condition;
};

namespace detail {
    template <typename T>
    void check_grad_enabled(const char *name, const T &value) {
        if (!grad_enabled(value))
            drjit_fail(
                "drjit::%s(): the argument does not depend on the input "
                "variable(s) being differentiated. Throwing an exception since "
                "this is usually indicative of a bug (for example, you may "
                "have forgotten to call drjit::enable_grad(..)). If this is "
                "expected behavior, skip the call to drjit::%s(..) if "
                "drjit::grad_enabled(..) returns 'false'.", name, name);
    }
}

template <typename T>
void backward_from(T &value, uint32_t flags = (uint32_t) ADFlag::Default) {
    detail::check_grad_enabled("backward_from", value);

    // Handle case where components of an N-d vector map to the same AD variable
    if constexpr (depth_v<T> > 1)
        value = value + T(0);

    if constexpr (is_complex_v<T>)
        set_grad(value, T(1.f, 1.f));
    else if constexpr (is_quaternion_v<T>)
        set_grad(value, T(1.f, 1.f, 1.f, 1.f));
    else
        set_grad(value, full<T>(1.f));

    enqueue(ADMode::Backward, value);
    traverse<T>(ADMode::Backward, flags);
}

template <typename T>
void backward_to(const T &value, uint32_t flags = (uint32_t) ADFlag::Default) {
    detail::check_grad_enabled("backward_to", value);
    enqueue(ADMode::Forward, value);
    traverse(ADMode::Backward, flags);
}

template <typename T>
void forward_from(T &value, uint32_t flags = (uint32_t) ADFlag::Default) {
    detail::check_grad_enabled("forward_from", value);
    
    if constexpr (is_complex_v<T>)
        set_grad(value, T(1.f, 1.f));
    else if constexpr (is_quaternion_v<T>)
        set_grad(value, T(1.f, 1.f, 1.f, 1.f));
    else
        set_grad(value, full<T>(1.f));

    enqueue(ADMode::Forward, value);
    traverse(ADMode::Forward, flags);
}

template <typename T>
void forward_to(T &value, uint32_t flags = (uint32_t) ADFlag::Default) {
    detail::check_grad_enabled("forward_to", value);
    enqueue(ADMode::Backward, value);
    traverse(ADMode::Forward, flags);
}

template <typename T>
void backward(T &value, uint32_t flags = (uint32_t) ADFlag::Default) {
    backward_from(value, flags);
}

template <typename T>
void forward(T &value, uint32_t flags = (uint32_t) ADFlag::Default) {
    forward_from(value, flags);
}

NAMESPACE_BEGIN(detail)

/// Internal operations for traversing nested data structures and fetching or
/// storing indices. Used in ``call.h`` and ``loop.h``.

template <bool IncRef> void collect_indices_fn(void *p, uint64_t index) {
    vector<uint64_t> &indices = *(vector<uint64_t> *) p;
    if constexpr (IncRef)
        index = ad_var_inc_ref(index);
    indices.push_back(index);
}

struct update_indices_payload {
    const vector<uint64_t> &indices;
    size_t &pos;
};

inline uint64_t update_indices_fn(void *p, uint64_t) {
    update_indices_payload &payload = *(update_indices_payload *) p;
    return payload.indices[payload.pos++];
}

template <bool IncRef, typename Value>
void collect_indices(const Value &value, vector<uint64_t> &indices) {
    traverse_1_fn_ro(value, (void *) &indices, collect_indices_fn<IncRef>);
}

template <typename Value>
void update_indices(Value &value, const vector<uint64_t> &indices, size_t &pos) {
    update_indices_payload payload { indices, pos };
    traverse_1_fn_rw(value, (void *) &payload, update_indices_fn);
}

template <typename T> void update_indices(T &value, const vector<uint64_t> &indices) {
    size_t pos = 0;
    update_indices(value, indices, pos);
#if !defined(NDEBUG)
    if (pos != indices.size())
        drjit_fail("update_indices(): did not consume the expected number of indices!");
#endif
}

/// Index vector that decreases AD refcounts when destructed
struct ad_index32_vector : drjit::vector<uint32_t> {
    using Base = drjit::vector<uint32_t>;
    using Base::Base;
    ad_index32_vector(ad_index32_vector &&a) : Base(std::move(a)) { }
    ad_index32_vector &operator=(ad_index32_vector &&a) {
        Base::operator=(std::move(a));
        return *this;
    }

    ~ad_index32_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(uint64_t(operator[](i)) << 32);
        Base::clear();
    }

    void push_back_steal(uint32_t index) { push_back(index); }
    void push_back_borrow(uint32_t index) {
        push_back((uint32_t) (ad_var_inc_ref(uint64_t(index) << 32) >> 32));
    }
};

/// Index vector that decreases JIT + AD refcounts when destructed
struct index64_vector : drjit::vector<uint64_t> {
    using Base = drjit::vector<uint64_t>;
    using Base::Base;
    index64_vector(index64_vector &&a) : Base(std::move(a)) { }
    index64_vector &operator=(index64_vector &&a) {
        Base::operator=(std::move(a));
        return *this;
    }

    ~index64_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
        Base::clear();
    }

    void push_back_steal(uint64_t index) { push_back(index); }
    void push_back_borrow(uint64_t index) { push_back(ad_var_inc_ref(index)); }
};

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
