
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

NAMESPACE_BEGIN(drjit)

template <JitBackend Backend_, typename Value_>
struct DRJIT_TRIVIAL_ABI DiffArray
    : ArrayBaseT<Value_, is_mask_v<Value_>, DiffArray<Backend_, Value_>> {
    static_assert(std::is_scalar_v<Value_>,
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
    static constexpr bool IsFloat = std::is_floating_point_v<Value_>;
    static constexpr bool IsClass =
        std::is_pointer_v<Value_> &&
        std::is_class_v<std::remove_pointer_t<Value_>>;
    static constexpr size_t Size = Dynamic;

    static constexpr VarType Type =
        IsClass ? VarType::UInt32 : var_type_v<Value>;

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
        if constexpr (IsFloat) {
            m_index = ad_var_inc_ref(a.m_index);
        } else {
            m_index = a.m_index;
            jit_var_inc_ref(m_index);
        }
    }

    DiffArray(DiffArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T>
    DiffArray(const DiffArray<Backend, T> &v) {
        if constexpr (IsFloat && std::is_floating_point_v<T>)
            m_index = ad_var_cast(v.m_index, Type);
        else
            m_index = jit_var_cast((uint32_t) v.m_index, Type, 0);
    }

    template <typename T>
    DiffArray(const DiffArray<Backend, T> &v, detail::reinterpret_flag) {
        m_index = jit_var_cast((uint32_t) v.m_index, Type, 1);
    }

    DiffArray(const Detached &v) : m_index(v.m_index) {
        jit_var_inc_ref((uint32_t) m_index);
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    DiffArray(T value) : m_index(Detached(value).release()) { }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    DiffArray(Ts&&... ts) : m_index(Detached(ts...).release()) { }

    DiffArray &operator=(const DiffArray &a) {
        Index old_index = m_index;
        if constexpr (IsFloat) {
            m_index = ad_var_inc_ref(a.m_index);
            ad_var_dec_ref(old_index);
        } else {
            m_index = a.m_index;
            jit_var_inc_ref(m_index);
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
    DiffArray sin_() const { return steal(ad_var_sin(m_index)); }
    DiffArray cos_() const { return steal(ad_var_cos(m_index)); }
    std::pair<DiffArray, DiffArray> sincos_() const {
        UInt64Pair p = jit_var_sincos(m_index);
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
        UInt64Pair p = jit_var_sincosh(m_index);
        return { steal(p.first), steal(p.second) };
    }
    DiffArray tanh_() const { return steal(ad_var_tanh(m_index)); }
    DiffArray asinh_() const { return steal(ad_var_asinh(m_index)); }
    DiffArray acosh_() const { return steal(ad_var_acosh(m_index)); }
    DiffArray atanh_() const { return steal(ad_var_atanh(m_index)); }

    DiffArray min_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_min(m_index, a.m_index));
        else
            return steal(jit_var_min(m_index, a.m_index));
    }

    DiffArray max_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_max(m_index, a.m_index));
        else
            return steal(jit_var_max(m_index, a.m_index));
    }

    DiffArray fma_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (IsFloat)
            return steal(ad_var_fma(m_index, a.m_index, b.m_index));
        else
            return steal(jit_var_fma(m_index, a.m_index, b.m_index));
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

    DiffArray dot_(const DiffArray &a) const { return sum(*this * a); }

    uint32_t count_() const {
        if constexpr (!is_mask_v<Value>)
            drjit_raise("Unsupported operand type");

        return sum(select(*this, (uint32_t) 1, (uint32_t) 0)).entry(0);
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

    template <bool, typename Index, typename Mask>
    static DiffArray gather_(const void * /*src*/, const Index & /*index*/,
                             const Mask & /*mask*/) {
        drjit_raise("Not implemented, please use gather() variant that takes a "
                    "array source argument.");
    }

    template <bool, typename Index, typename Mask>
    static DiffArray gather_(const DiffArray &src, const Index &index,
                             const Mask &mask) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);
        return steal(jit_var_gather(src.m_index, index.m_index, mask.m_index));
    }

    template <bool, typename Index, typename Mask>
    void scatter_(void * /* dst */, const Index & /*index*/,
                  const Mask & /*mask*/) const {
        drjit_raise("Not implemented, please use scatter() variant that takes "
                    "a array target argument.");
    }

    template <bool, typename Index, typename Mask>
    void scatter_(DiffArray &dst, const Index &index, const Mask &mask) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);
        dst = steal(jit_var_scatter(dst.m_index, m_index, index.m_index,
                                    mask.m_index, ReduceOp::None));
    }

    template <typename Index, typename Mask>
    void scatter_reduce_(ReduceOp /*op*/, void * /*dst*/,
                         const Index & /*index*/,
                         const Mask & /* mask */) const {
        drjit_raise("Not implemented, please use scatter_reduce() variant that "
                    "takes a array target argument.");
    }

    template <typename Index, typename Mask>
    void scatter_reduce_(ReduceOp op, DiffArray &dst, const Index &index,
                         const Mask &mask) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<DiffArray>>>);
        dst = steal(jit_var_scatter(dst.m_index, m_index, index.m_index,
                                    mask.m_index, op));
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

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_same_v<T, Value>> = 0>
    static DiffArray full_(T value, size_t size) {
        return steal(Detached::full_(value, size).release());
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_same_v<T, Value>> = 0>
    static DiffArray opaque_(T value, size_t size) {
        return steal(Detached::opaque_(value, size).release());
    }

    static DiffArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        return steal(Detached::arange_(start, stop, step).release());
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_same_v<T, Value>> = 0>
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

        if constexpr (IsFloat) {
            result.m_index = ad_var_inc_ref(index);
        } else {
            jit_var_inc_ref(index);
            result.m_index = index;
        }

        return result;
    }

    Index release() {
        Index tmp = m_index;
        m_index = 0;
        return tmp;
    }

    size_t size() const { return jit_var_size(m_index); }

    bool grad_enabled_() const {
        if constexpr (IsFloat)
            return (m_index >> 32) != 0;
        else
            return false;
    }

    void set_grad_enabled_(bool value) {
        DRJIT_MARK_USED(value);
        if constexpr (IsFloat) {
            if (value) {
                if (grad_enabled_())
                    return;
                m_index = ad_var_new(m_index);
            } else {
                jit_var_inc_ref(m_index);
                ad_var_dec_ref(m_index);
                m_index = (uint32_t) m_index;
            }
        }
    }

    Value entry(size_t offset) const {
        ActualValue out;
        jit_var_read((uint32_t) m_index, offset, &out);

        if constexpr (!IsClass)
            return out;
        else
            return (Value) jit_registry_get_ptr(Backend, CallSupport::Domain, out);
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_same_v<T, Value>> = 0>
    void set_entry(size_t offset, T value) {
        if (grad_enabled_())
            jit_raise("DiffArray::set_entry(): not permitted on attached variables!");

        uint32_t index;
        if constexpr (!IsClass) {
            index = jit_var_write((uint32_t) m_index, offset, &value);
        } else {
            ActualValue av = jit_registry_get_id(Backend, value);
            index = jit_var_write((uint32_t) m_index, (uint32_t) offset, &av);
        }
        jit_var_dec_ref((uint32_t) m_index);
        m_index = index;
    }

    const Value *data() const { return (const Value *) jit_var_ptr((uint32_t) m_index); }
    Value *data() { return (Value *) jit_var_ptr((uint32_t) m_index); }

    bool valid() const { return m_index != 0; }

    uint32_t index() const { return (uint32_t) m_index; }
    uint32_t index_ad() const { return (uint32_t) (((uint64_t) m_index) >> 32); }
    uint64_t index_combined() const { return m_index; }

private:
    Index m_index = 0;
};

template <typename Value> using CUDADiffArray = DiffArray<JitBackend::CUDA, Value>;
template <typename Value> using LLVMDiffArray = DiffArray<JitBackend::LLVM, Value>;

NAMESPACE_END(drjit)
