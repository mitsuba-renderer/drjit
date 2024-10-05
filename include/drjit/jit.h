/*
    drjit/jit.h -- Dr.Jit dynamic array with JIT compilation

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>
#include <drjit/extra.h>
#include <drjit-core/half.h>
#include <drjit-core/traits.h>

NAMESPACE_BEGIN(drjit)

template <JitBackend Backend_, typename Value_>
struct DRJIT_TRIVIAL_ABI JitArray
    : ArrayBaseT<Value_, is_mask_v<Value_>, JitArray<Backend_, Value_>> {
    static_assert(drjit::detail::is_scalar_v<Value_> || std::is_void_v<Value_>,
                  "JIT Arrays can only be created over scalar types!");

    template <JitBackend, typename> friend struct JitArray;

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Value = Value_;
    using Base = ArrayBaseT<Value_, is_mask_v<Value_>, JitArray<Backend_, Value_>>;

    static constexpr JitBackend Backend = Backend_;

    static constexpr bool IsArray = true;
    static constexpr bool IsJIT = true;
    static constexpr bool IsCUDA = Backend == JitBackend::CUDA;
    static constexpr bool IsLLVM = Backend == JitBackend::LLVM;
    static constexpr bool IsDynamic = true;
    static constexpr size_t Size = Dynamic;

    static constexpr bool IsClass =
        std::is_pointer_v<Value_> &&
        std::is_class_v<std::remove_pointer_t<Value_>>;

    static constexpr VarType Type =
        IsClass ? VarType::UInt32 : type_v<Value>;

    using ActualValue = std::conditional_t<IsClass, uint32_t, Value>;

    using CallSupport =
        call_support<std::decay_t<std::remove_pointer_t<Value_>>, JitArray>;

    template <typename T> using ReplaceValue = JitArray<Backend, T>;
    using MaskType = JitArray<Backend, bool>;
    using ArrayType = JitArray;
    using Index = uint32_t;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    JitArray() = default;

    ~JitArray() noexcept { jit_var_dec_ref(m_index); }

    JitArray(const JitArray &a) {
        m_index = jit_var_inc_ref(a.m_index);
    }

    JitArray(JitArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> JitArray(const JitArray<Backend, T> &v) {
        m_index = jit_var_cast(v.m_index, Type, 0);
    }

    template <typename T> JitArray(const JitArray<Backend, T> &v,
                                   detail::reinterpret_flag) {
        m_index = jit_var_cast(v.m_index, Type, 1);
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    JitArray(T value) {
        if constexpr (IsClass) {
            m_index = jit_var_class(Backend, (void *) value);
        } else {
            switch (Type) {
                case VarType::Bool:    m_index = jit_var_bool(Backend, (bool) value); break;
                case VarType::Int32:   m_index = jit_var_i32 (Backend, (int32_t) value); break;
                case VarType::UInt32:  m_index = jit_var_u32 (Backend, (uint32_t) value); break;
                case VarType::Int64:   m_index = jit_var_i64 (Backend, (int64_t) value); break;
                case VarType::UInt64:  m_index = jit_var_u64 (Backend, (uint64_t) value); break;
                case VarType::Float16: m_index = jit_var_f16 (Backend, half(value)); break;
                case VarType::Float32: m_index = jit_var_f32 (Backend, (float) value); break;
                case VarType::Float64: m_index = jit_var_f64 (Backend, (double) value); break;
                default: jit_fail("JitArray(): tried to initialize scalar array with unsupported type!");
            }
        }
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    JitArray(Ts&&... ts) {
        if constexpr (!IsClass) {
            Value data[] = { (Value) ts... };
            m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                       sizeof...(Ts));
        } else {
            uint32_t data[] = { jit_registry_id(ts)... };
            m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                       sizeof...(Ts));
        }
    }

    JitArray &operator=(const JitArray &a) {
        uint32_t index = jit_var_inc_ref(a.m_index);
        jit_var_dec_ref(m_index);
        m_index = index;
        return *this;
    }

    JitArray &operator=(JitArray &&a) {
        Index temp = m_index;
        m_index = a.m_index;
        a.m_index = temp;
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    JitArray add_(const JitArray &v) const {
        return steal(jit_var_add(m_index, v.m_index));
    }

    JitArray sub_(const JitArray &v) const {
        return steal(jit_var_sub(m_index, v.m_index));
    }

    JitArray mul_(const JitArray &v) const {
        return steal(jit_var_mul(m_index, v.m_index));
    }

    JitArray mulhi_(const JitArray &v) const {
        return steal(jit_var_mulhi(m_index, v.m_index));
    }

    JitArray div_(const JitArray &v) const {
        return steal(jit_var_div(m_index, v.m_index));
    }

    JitArray mod_(const JitArray &v) const {
        return steal(jit_var_mod(m_index, v.m_index));
    }

    auto gt_(const JitArray &v) const {
        return mask_t<JitArray>::steal(jit_var_gt(m_index, v.m_index));
    }

    auto ge_(const JitArray &v) const {
        return mask_t<JitArray>::steal(jit_var_ge(m_index, v.m_index));
    }

    auto lt_(const JitArray &v) const {
        return mask_t<JitArray>::steal(jit_var_lt(m_index, v.m_index));
    }

    auto le_(const JitArray &v) const {
        return mask_t<JitArray>::steal(jit_var_le(m_index, v.m_index));
    }

    auto eq_(const JitArray &v) const {
        return mask_t<JitArray>::steal(jit_var_eq(m_index, v.m_index));
    }

    auto neq_(const JitArray &v) const {
        return mask_t<JitArray>::steal(jit_var_neq(m_index, v.m_index));
    }

    JitArray neg_() const { return steal(jit_var_neg(m_index)); }

    JitArray not_() const { return steal(jit_var_not(m_index)); }

    template <typename T> JitArray or_(const T &v) const {
        return steal(jit_var_or(m_index, v.m_index));
    }

    template <typename T> JitArray and_(const T &v) const {
        return steal(jit_var_and(m_index, v.m_index));
    }

    template <typename T> JitArray xor_(const T &v) const {
        return steal(jit_var_xor(m_index, v.m_index));
    }

    template <typename T> JitArray andnot_(const T &a) const {
        return and_(a.not_());
    }

    JitArray sl_(const JitArray &v) const {
        return steal(jit_var_shl(m_index, v.m_index));
    }

    JitArray sr_(const JitArray &v) const {
        return steal(jit_var_shr(m_index, v.m_index));
    }

    template <int Imm> JitArray sr_() const { return sr_(Imm); }
    template <int Imm> JitArray sl_() const { return sl_(Imm); }

    JitArray abs_() const { return steal(jit_var_abs(m_index)); }
    JitArray sqrt_() const { return steal(jit_var_sqrt(m_index)); }
    JitArray rcp_() const { return steal(jit_var_rcp(m_index)); }
    JitArray rsqrt_() const { return steal(jit_var_rsqrt(m_index)); }
    JitArray exp2_() const { return steal(jit_var_exp2(m_index)); }
    JitArray exp_() const { return steal(jit_var_exp(m_index)); }
    JitArray log2_() const { return steal(jit_var_log2(m_index)); }
    JitArray log_() const { return steal(jit_var_log(m_index)); }
    JitArray sin_() const { return steal(jit_var_sin(m_index)); }
    JitArray cos_() const { return steal(jit_var_cos(m_index)); }
    JitArray tan_() const { return steal(jit_var_tan(m_index)); }
    JitArray asin_() const { return steal(jit_var_asin(m_index)); }
    JitArray acos_() const { return steal(jit_var_acos(m_index)); }
    JitArray atan_() const { return steal(jit_var_atan(m_index)); }
    JitArray sinh_() const { return steal(jit_var_sinh(m_index)); }
    JitArray cosh_() const { return steal(jit_var_cosh(m_index)); }
    JitArray tanh_() const { return steal(jit_var_tanh(m_index)); }
    JitArray asinh_() const { return steal(jit_var_asinh(m_index)); }
    JitArray acosh_() const { return steal(jit_var_acosh(m_index)); }
    JitArray atanh_() const { return steal(jit_var_atanh(m_index)); }
    JitArray cbrt_() const { return steal(jit_var_cbrt(m_index)); }
    JitArray erf_() const { return steal(jit_var_erf(m_index)); }

    JitArray atan2_(const JitArray &x) const {
        return steal(jit_var_atan2(m_index, x.index()));
    }

    JitArray ldexp_(const JitArray &x) const {
        return steal(jit_var_ldexp(m_index, x.index()));
    }

    std::pair<JitArray, JitArray> frexp_() const {
        UInt32Pair p = jit_var_frexp(m_index);
        return { steal(p.first), steal(p.second) };
    }

    std::pair<JitArray, JitArray> sincos_() const {
        UInt32Pair p = jit_var_sincos(m_index);
        return { steal(p.first), steal(p.second) };
    }

    std::pair<JitArray, JitArray> sincosh_() const {
        UInt32Pair p = jit_var_sincosh(m_index);
        return { steal(p.first), steal(p.second) };
    }

    JitArray minimum_(const JitArray &v) const {
        return steal(jit_var_min(m_index, v.m_index));
    }

    JitArray maximum_(const JitArray &v) const {
        return steal(jit_var_max(m_index, v.m_index));
    }

    JitArray round_() const { return steal(jit_var_round(m_index)); }
    template <typename T> T round2int_() const { return T(round(*this)); }

    JitArray floor_() const { return steal(jit_var_floor(m_index)); }
    template <typename T> T floor2int_() const { return T(floor(*this)); }

    JitArray ceil_() const { return steal(jit_var_ceil(m_index)); }
    template <typename T> T ceil2int_() const { return T(ceil(*this)); }

    JitArray trunc_() const { return steal(jit_var_trunc(m_index)); }
    template <typename T> T trunc2int_() const { return T(trunc(*this)); }

    JitArray fmadd_(const JitArray &b, const JitArray &c) const {
        return steal(jit_var_fma(m_index, b.index(), c.index()));
    }

    JitArray fmsub_(const JitArray &b, const JitArray &c) const {
        return fmadd_(b, -c);
    }

    JitArray fnmadd_(const JitArray &b, const JitArray &c) const {
        return fmadd_(-b, c);
    }

    JitArray fnmsub_(const JitArray &b, const JitArray &c) const {
        return fmadd_(-b, -c);
    }

    template <typename Mask>
    static JitArray select_(const Mask &m, const JitArray &t, const JitArray &f) {
        static_assert(std::is_same_v<Mask, mask_t<JitArray>>);
        return steal(jit_var_select(m.index(), t.index(), f.index()));
    }

    JitArray popcnt_() const { return steal(jit_var_popc(m_index)); }
    JitArray lzcnt_() const { return steal(jit_var_clz(m_index)); }
    JitArray brev_() const { return steal(jit_var_brev(m_index)); }
    JitArray tzcnt_() const { return steal(jit_var_ctz(m_index)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const { return jit_var_all(m_index); }
    bool any_() const { return jit_var_any(m_index); }

    #define DRJIT_HORIZONTAL_OP(name, op)                                  \
        JitArray name##_() const {                                         \
            return steal(jit_var_reduce(Backend, Type, op, m_index));      \
        }

    DRJIT_HORIZONTAL_OP(sum,  ReduceOp::Add)
    DRJIT_HORIZONTAL_OP(prod, ReduceOp::Mul)
    DRJIT_HORIZONTAL_OP(min,  ReduceOp::Min)
    DRJIT_HORIZONTAL_OP(max,  ReduceOp::Max)

    #undef DRJIT_HORIZONTAL_OP

    JitArray block_prefix_reduce_(ReduceOp op, uint32_t block_size, bool exclusive, bool reverse) const {
        return steal(jit_var_block_prefix_reduce(op, m_index, block_size, exclusive, reverse));
    }

    JitArray dot_(const JitArray &a) const { return sum(*this * a); }

    auto count_() const {
        if constexpr (!is_mask_v<Value>)
            drjit_fail("Unsupported operand type");

        if (!m_index)
            return uint32_array_t<JitArray>(0);

        return sum(select(*this, (uint32_t) 1, (uint32_t) 0));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fancy array initialization
    // -----------------------------------------------------------------------

    static JitArray empty_(size_t size) {
        return steal(jit_var_undefined(Backend, Type, size));
    }

    static JitArray zero_(size_t size) {
        Value value = 0;
        return steal(jit_var_literal(Backend, Type, &value, size));
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    static JitArray full_(T value, size_t size) {
        ActualValue av;
        if constexpr (!IsClass)
            av = (ActualValue) value;
        else
            av = jit_registry_id(value);

        return steal(jit_var_literal(Backend, Type, &av, size, false));
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    static JitArray opaque_(T value, size_t size) {
        ActualValue av;
        if constexpr (!IsClass)
            av = (ActualValue) value;
        else
            av = jit_registry_id(value);

        return steal(jit_var_literal(Backend, Type, &av, size, true));
    }

    static JitArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
        if (size == 0)
            return JitArray();
        return fmadd(JitArray(uint32_array_t<JitArray>::counter(size)),
                     JitArray((Value) step),
                     JitArray((Value) start));
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_convertible_v<T, Value>> = 0>
    static JitArray linspace_(T min, T max, size_t size, bool endpoint) {
        T step = (max - min) / T(size - ((endpoint && size > 1) ? 1 : 0));
        return fmadd(JitArray(uint32_array_t<JitArray>::counter(size)),
                     JitArray(step),
                     JitArray(min));
    }

    static JitArray map_(void *ptr, size_t size, bool free = false) {
         return steal(jit_var_mem_map(Backend, Type, ptr, size, free ? 1 : 0));
    }

    static JitArray load_(const void *ptr, size_t size) {
        if constexpr (!IsClass) {
            return steal(jit_var_mem_copy(Backend, AllocType::Host, Type, ptr,
                                          (uint32_t) size));
        } else {
            uint32_t *temp = new uint32_t[size];
            for (uint32_t i = 0; i < size; i++)
                temp[i] = jit_registry_id(((const void **) ptr)[i]);
            JitArray result = steal(
                jit_var_mem_copy(Backend, AllocType::Host, Type, temp, (uint32_t) size));
            delete[] temp;
            return result;
        }
    }

    void store_(void *ptr) const {
        eval_();
        if constexpr (!IsClass) {
            jit_memcpy(Backend, ptr, data(), size() * sizeof(Value));
        } else {
            uint32_t size = this->size();
            uint32_t *temp = new uint32_t[size];
            jit_memcpy(Backend, temp, data(), size * sizeof(uint32_t));
            for (uint32_t i = 0; i < size; i++)
                ((void **) ptr)[i] =
                    jit_registry_ptr(Backend, CallSupport::Domain, temp[i]);
            delete[] temp;
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather support
    // -----------------------------------------------------------------------

    template <typename Index, typename Mask>
    static JitArray gather_(const JitArray &src, const Index &index,
                            const Mask &mask, ReduceMode /* mode */) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
        return steal(jit_var_gather(src.index(), index.index(), mask.index()));
    }

    template <size_t N, typename Index, typename Mask>
    static Array<JitArray, N> gather_packet_(const JitArray &src, const Index &index,
                                             const Mask &mask, ReduceMode mode) {
        if constexpr ((N & (N-1)) > 0) {
            return Base::template gather_packet_<N>(src, index, mask, mode);
        } else {
            static_assert(
                std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
            uint32_t tmp[N];
            jit_var_gather_packet(N, src.index(), index.index(), mask.index(), tmp);

            Array<JitArray, N> result;
            for (size_t i = 0; i < N; ++i)
                result[i] = steal(tmp[i]);
            return result;
        }
    }

    template <typename Index, typename Mask>
    void scatter_(JitArray &dst, const Index &index, const Mask &mask, ReduceMode mode) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
        dst = steal(jit_var_scatter(dst.index(), m_index, index.index(),
                                    mask.index(), ReduceOp::Identity, mode));
    }

    template <size_t N, typename Source, typename Index, typename Mask>
    static void scatter_packet_(JitArray &dst, const Source &source,
                                const Index &index, const Mask &mask,
                                ReduceMode mode) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
        if constexpr ((N & (N-1)) > 0) {
            Base::template scatter_packet_<N>(dst, source, index, mask, mode);
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
    void scatter_reduce_(JitArray &dst, const Index &index, const Mask &mask,
                         ReduceOp op, ReduceMode mode) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
        dst = steal(jit_var_scatter(dst.index(), m_index, index.index(),
                                    mask.index(), op, mode));
    }

    template <size_t N, typename Source, typename Index, typename Mask>
    static void scatter_reduce_packet_(JitArray &dst, const Source &source,
                                       const Index &index, const Mask &mask,
                                       ReduceOp op, ReduceMode mode) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
        if constexpr ((N & (N-1)) > 0) {
            Base::template scatter_reduce_packet_<N>(dst, source, index, mask, op, mode);
        } else {
            uint32_t indices[N];
            for (uint32_t i = 0; i < N; ++i)
                indices[i] = source[i].index();

            dst = steal(jit_var_scatter_packet(N, dst.index(), indices,
                                               index.index(), mask.index(),
                                               op, mode));
        }
    }

    template <typename Index, typename Mask>
    void scatter_add_kahan_(JitArray &dst_1, JitArray &dst_2,
                            const Index &index, const Mask &mask) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<JitArray>>>);
        jit_var_scatter_add_kahan(&dst_1.m_index, &dst_2.m_index,
                                  m_index, index.index(), mask.index());
    }

    template <typename Mask>
    JitArray scatter_inc_(const JitArray &index, const Mask &mask) {
        return steal(jit_var_scatter_inc(&m_index, index.index(), mask.index()));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    std::pair<CallBucket *, uint32_t> vcall_() const {
        if constexpr (!IsClass) {
            drjit_fail("Unsupported operand type");
        } else {
            uint32_t bucket_count = 0;
            CallBucket *buckets = jit_var_call_reduce(
                Backend, CallSupport::Domain, m_index, &bucket_count);
            return { buckets, bucket_count };
        }
    }

    auto compress_() const {
        if constexpr (!is_mask_v<Value>)
            drjit_fail("Unsupported operand type");
        else
            return uint32_array_t<JitArray>::steal(jit_var_compress(m_index));
    }

    JitArray block_reduce_(ReduceOp op, size_t block_size, int symbolic) const {
        return steal(jit_var_block_reduce(op, m_index, (uint32_t) block_size, symbolic));
    }

    JitArray tile_(size_t count) const {
        return steal(jit_var_tile(m_index, (uint32_t) count));
    }

    JitArray copy() const { return steal(jit_var_copy(m_index)); }

    bool schedule_() const { return jit_var_schedule(m_index) != 0; }
    bool schedule_force_() {
        int rv = 0;
        *this = steal(jit_var_schedule_force(m_index, &rv));
        return rv;
    }

    bool eval_() const { return jit_var_eval(m_index) != 0; }

    bool valid() const { return m_index != 0; }
    size_t size() const { return jit_var_size(m_index); }
    uint32_t index() const { return m_index; }
    uint32_t index_ad() const { return 0; }
    uint64_t index_combined() const { return m_index; }

    void swap(JitArray &a) {
        uint32_t index = m_index;
        m_index = a.m_index;
        a.m_index = index;
    }

    const Value *data() const {
        void *p = nullptr;
        const_cast<JitArray&>(*this) = steal(jit_var_data(m_index, &p));
        return (const Value *) p;
    }

    Value *data() {
        void *p = nullptr;
        *this = steal(jit_var_data(m_index, &p));
        return (Value *) p;
    }

    const char *str() { return jit_var_str(m_index); }

    VarState state() const { return jit_var_state(m_index); }

    Value entry(size_t offset) const {
        ActualValue out;
        jit_var_read(m_index, offset, &out);

        if constexpr (!IsClass)
            return out;
        else
            return (Value) jit_registry_ptr(Backend, CallSupport::Domain, out);
    }

    template <typename T, enable_if_t<!std::is_void_v<T> && std::is_same_v<T, Value>> = 0>
    void set_entry(size_t offset, T value) {
        Index index;
        if constexpr (!IsClass) {
            index = jit_var_write(m_index, offset, &value);
        } else {
            uint32_t av = jit_registry_id(value);
            index = jit_var_write(m_index, (uint32_t) offset, &av);
        }
        jit_var_dec_ref(m_index);
        m_index = index;
    }

    void resize(size_t size) {
        Index index = jit_var_resize(m_index, size);
        jit_var_dec_ref(m_index);
        m_index = index;
    }

    JitArray migrate_(AllocType type) const {
        return steal(jit_var_migrate(m_index, type));
    }

    static auto counter(size_t size) {
        return uint32_array_t<JitArray>::steal(jit_var_counter(Backend, size));
    }

    template <typename... Ts> void set_label_(const Ts*... args) {
        Index index = jit_var_set_label(m_index, sizeof...(Ts), args...);
        jit_var_dec_ref(m_index);
        m_index = index;
    }

    const char *label_() const {
        return jit_var_label(m_index);
    }

    const CallSupport operator->() const {
        return CallSupport(*this);
    }

    //! @}
    // -----------------------------------------------------------------------

    static DRJIT_INLINE JitArray steal(Index index) {
        JitArray result;
        result.m_index = index;
        return result;
    }

    static DRJIT_INLINE JitArray borrow(Index index) {
        JitArray result;
        result.m_index = jit_var_inc_ref(index);
        return result;
    }

    DRJIT_INLINE Index release() {
        Index tmp = m_index;
        m_index = 0;
        return tmp;
    }

protected:
    Index m_index = 0;
};

template <typename Value> using CUDAArray = JitArray<JitBackend::CUDA, Value>;
template <typename Value> using LLVMArray = JitArray<JitBackend::LLVM, Value>;

template <typename T> T tile(const T &value, size_t count) {
    if constexpr (is_traversable_v<T>) {
        T result;
        traverse_2(
            fields(result), fields(value),
            [count](auto &x, const auto &y) {
                x = tile(y, count);
            });
        return result;
    } if constexpr (is_jit_v<T>) {
        return value.tile_(count);
    } else {
        return value;
    }
}

NAMESPACE_BEGIN(detail)

/// Index vector that decreases JIT refcounts when destructed
struct index32_vector : drjit::vector<uint32_t> {
    using Base = drjit::vector<uint32_t>;
    using Base::Base;

    index32_vector(index32_vector &&a) : Base(std::move(a)) { }
    index32_vector &operator=(index32_vector &&a) {
        Base::operator=(std::move(a));
        return *this;
    }
    ~index32_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            jit_var_dec_ref(operator[](i));
        Base::clear();
    }

    void push_back_steal(uint32_t index) { push_back(index); }
    void push_back_borrow(uint32_t index) {
        push_back(jit_var_inc_ref(index));
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
