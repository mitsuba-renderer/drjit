/*
    enoki/jit.h -- Enoki dynamic array with JIT compilation

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once
#define ENOKI_CUDA_H

#include <enoki/array.h>
#include <enoki-jit/traits.h>

NAMESPACE_BEGIN(enoki)

template <JitBackend Backend_, typename Value_>
struct JitArray : ArrayBase<Value_, is_mask_v<Value_>, JitArray<Backend_, Value_>> {
    static_assert(std::is_scalar_v<Value_>,
                  "JIT Arrays can only be created over scalar types!");

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Value = Value_;
    using MaskType = JitArray<Backend_, bool>;
    using ArrayType = JitArray;

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
        IsClass ? VarType::UInt32 : var_type_v<Value>;

    using ActualValue = std::conditional_t<IsClass, uint32_t, Value>;

    using CallSupport =
        call_support<std::decay_t<std::remove_pointer_t<Value_>>, JitArray>;

    template <typename T> using ReplaceValue = JitArray<Backend, T>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    JitArray() = default;

    ~JitArray() noexcept { jit_var_dec_ref_ext(m_index); }

    JitArray(const JitArray &a) : m_index(a.m_index) {
        jit_var_inc_ref_ext(m_index);
    }

    JitArray(JitArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> JitArray(const JitArray<Backend, T> &v) {
        m_index = jit_var_new_cast(v.index(), Type, 0);
    }

    template <typename T>
    JitArray(const JitArray<Backend, T> &v, detail::reinterpret_flag) {
        m_index = jit_var_new_cast(v.index(), Type, 1);
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    JitArray(T v) {
        scalar_t<Value> value(v);
        m_index = jit_var_new_literal(Backend, Type, &value);
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    JitArray(Ts&&... ts) {
        if constexpr (!IsClass) {
            Value data[] = { (Value) ts... };
            m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                       sizeof...(Ts));
        } else {
            uint32_t data[] = { jit_registry_get_id(ts)... };
            m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                       sizeof...(Ts));
        }
    }

    JitArray &operator=(const JitArray &a) {
        jit_var_inc_ref_ext(a.m_index);
        jit_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    JitArray &operator=(JitArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    JitArray add_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Add, m_index, v.m_index));
    }

    JitArray sub_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Sub, m_index, v.m_index));
    }

    JitArray mul_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Mul, m_index, v.m_index));
    }

    JitArray mulhi_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Mulhi, m_index, v.m_index));
    }

    JitArray div_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Div, m_index, v.m_index));
    }

    JitArray mod_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Mod, m_index, v.m_index));
    }

    MaskType gt_(const JitArray &v) const {
        return MaskType::steal(jit_var_new_op_2(JitOp::Gt, m_index, v.m_index));
    }

    MaskType ge_(const JitArray &v) const {
        return MaskType::steal(jit_var_new_op_2(JitOp::Ge, m_index, v.m_index));
    }

    MaskType lt_(const JitArray &v) const {
        return MaskType::steal(jit_var_new_op_2(JitOp::Lt, m_index, v.m_index));
    }

    MaskType le_(const JitArray &v) const {
        return MaskType::steal(jit_var_new_op_2(JitOp::Le, m_index, v.m_index));
    }

    MaskType eq_(const JitArray &v) const {
        return MaskType::steal(jit_var_new_op_2(JitOp::Eq, m_index, v.m_index));
    }

    MaskType neq_(const JitArray &v) const {
        return MaskType::steal(jit_var_new_op_2(JitOp::Neq, m_index, v.m_index));
    }

    JitArray neg_() const {
        return steal(jit_var_new_op_1(JitOp::Neg, m_index));
    }

    JitArray not_() const {
        return steal(jit_var_new_op_1(JitOp::Not, m_index));
    }

    template <typename T> JitArray or_(const T &v) const {
        return steal(jit_var_new_op_2(JitOp::Or, m_index, v.index()));
    }

    template <typename T> JitArray and_(const T &v) const {
        return steal(jit_var_new_op_2(JitOp::And, m_index, v.index()));
    }

    template <typename T> JitArray xor_(const T &v) const {
        return steal(jit_var_new_op_2(JitOp::Xor, m_index, v.index()));
    }

    template <typename T> JitArray andnot_(const T &a) const {
        return and_(a.not_());
    }

    template <int Imm> JitArray sl_() const {
        return sl_(Imm);
    }

    JitArray sl_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Shl, m_index, v.index()));
    }

    template <int Imm> JitArray sr_() const {
        return sr_(Imm);
    }

    JitArray sr_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Shr, m_index, v.index()));
    }

    JitArray abs_() const {
        return steal(jit_var_new_op_1(JitOp::Abs, m_index));
    }

    JitArray sqrt_() const {
        return steal(jit_var_new_op_1(JitOp::Sqrt, m_index));
    }

    JitArray rcp_() const {
        return steal(jit_var_new_op_1(JitOp::Rcp, m_index));
    }

    JitArray rsqrt_() const {
        return steal(jit_var_new_op_1(JitOp::Rsqrt, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    JitArray exp2_() const {
        return steal(jit_var_new_op_1(JitOp::Exp2, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    JitArray exp_() const {
        return exp2(InvLogTwo<T> * (*this));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    JitArray log2_() const {
        return steal(jit_var_new_op_1(JitOp::Log2, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    JitArray log_() const {
        return log2(*this) * LogTwo<T>;
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    JitArray sin_() const {
        return steal(jit_var_new_op_1(JitOp::Sin, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    JitArray cos_() const {
        return steal(jit_var_new_op_1(JitOp::Cos, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    std::pair<JitArray, JitArray> sincos_() const {
        return { sin_(), cos_() };
    }

    JitArray min_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Min, m_index, v.index()));
    }

    JitArray max_(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Max, m_index, v.index()));
    }

    JitArray round_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Round, m_index));
    }

    template <typename T> T round2int_() const {
        return T(round(*this));
    }

    JitArray floor_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Floor, m_index));
    }

    template <typename T> T floor2int_() const {
        return T(floor(*this));
    }

    JitArray ceil_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Ceil, m_index));
    }

    template <typename T> T ceil2int_() const {
        return T(ceil(*this));
    }

    JitArray trunc_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Trunc, m_index));
    }

    template <typename T> T trunc2int_() const {
        return T(trunc(*this));
    }

    JitArray fmadd_(const JitArray &b, const JitArray &c) const {
        return steal(
            jit_var_new_op_3(JitOp::Fmadd, m_index, b.index(), c.index()));
    }

    JitArray fmsub_(const JitArray &b, const JitArray &c) const {
        return fmadd_(b, -c);
    }

    JitArray fnmadd_(const JitArray &b, const JitArray &c) const {
        return fmadd_(-b, c);
    }

    JitArray fnmsub_(const JitArray &b, const JitArray &c) const {
        return fmsub_(-b, -c);
    }

    static JitArray select_(const MaskType &m, const JitArray &t,
                             const JitArray &f) {
        return steal(
            jit_var_new_op_3(JitOp::Select, m.index(), t.index(), f.index()));
    }

    JitArray popcnt_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Popc, m_index));
    }

    JitArray lzcnt_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Clz, m_index));
    }

    JitArray tzcnt_() const {
        return MaskType::steal(jit_var_new_op_1(JitOp::Ctz, m_index));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const {
        return jit_var_all(m_index);
    }

    bool any_() const {
        return jit_var_any(m_index);
    }

    #define ENOKI_HORIZONTAL_OP(name, op)                                        \
        JitArray name##_async_() const {                                         \
            if (size() == 0)                                                     \
                enoki_raise(#name "_async_(): zero-sized array!");               \
            return steal(jit_var_reduce(m_index, op));                           \
        }                                                                        \
        Value name##_() const { return name##_async_().entry(0); }

    ENOKI_HORIZONTAL_OP(hsum,  ReduceOp::Add)
    ENOKI_HORIZONTAL_OP(hprod, ReduceOp::Mul)
    ENOKI_HORIZONTAL_OP(hmin,  ReduceOp::Min)
    ENOKI_HORIZONTAL_OP(hmax,  ReduceOp::Max)

    #undef ENOKI_HORIZONTAL_OP

    Value dot_(const JitArray &a) const {
        return hsum(*this * a);
    }

    JitArray dot_async_(const JitArray &a) const {
        return hsum_async(*this * a);
    }

    uint32_t count_() const {
        if constexpr (!is_mask_v<Value>)
            enoki_raise("Unsupported operand type");

        return hsum(select(*this, (uint32_t) 1, (uint32_t) 0));
    }

    //! @}
    // -----------------------------------------------------------------------

   // -----------------------------------------------------------------------
    //! @{ \name Fancy array initialization
    // -----------------------------------------------------------------------

    JitArray placeholder_(bool propagate_literals) const {
        return steal(
            jit_var_new_placeholder(m_index, propagate_literals));
    }

    static JitArray empty_(size_t size) {
        size_t byte_size = size * sizeof(Value);
        void *ptr =
            jit_malloc(Backend == JitBackend::CUDA ? AllocType::Device
                                                   : AllocType::HostAsync,
                       byte_size);
        return steal(
            jit_var_mem_map(Backend, Type, ptr, size, 1));
    }

    static JitArray zero_(size_t size) {
        Value value = 0;
        return steal(jit_var_new_literal(Backend, Type, &value, size));
    }

    static JitArray full_(Value value, size_t size) {
        return steal(
            jit_var_new_literal(Backend, Type, &value, size, false));
    }

    static JitArray opaque_(const Value &value, size_t size = 1) {
        return steal(
            jit_var_new_literal(Backend, Type, &value, size, true));
    }

    static JitArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
        return fmadd(JitArray(JitArray<Backend, uint32_t>::counter(size)),
                     JitArray((Value) step),
                     JitArray((Value) start));
    }

    static JitArray linspace_(Value min, Value max, size_t size) {
        Value step = (max - min) / Value(size - 1);
        return fmadd(JitArray(JitArray<Backend, uint32_t>::counter(size)),
                     JitArray(step),
                     JitArray(min));
    }

    static JitArray map_(void *ptr, size_t size, bool free = false) {
         return steal(jit_var_mem_map(Backend, Type, ptr, size, free ? 1 : 0));
    }

    static JitArray load_(const void *ptr, size_t size) {
        if constexpr (!IsClass) {
            return steal(
                jit_var_mem_copy(Backend, AllocType::Host, Type, ptr, (uint32_t) size));
        } else {
            uint32_t *temp = new uint32_t[size];
            for (uint32_t i = 0; i < size; i++)
                temp[i] = jit_registry_get_id(((const void **) ptr)[i]);
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
                ((void **) ptr)[i] = jit_registry_get_ptr(CallSupport::Domain, temp[i]);
            delete[] temp;
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather support
    // -----------------------------------------------------------------------

    template <bool, typename Index>
    static JitArray gather_(const void */*src*/,
                            const JitArray<Backend, Index> &/*index*/,
                            const MaskType &/*mask*/ = true) {
        enoki_raise("Not implemented, please use gather(JitArray src, index, mask) instead!");
    }

    template <bool, typename Index>
    static JitArray gather_(const JitArray &src, const JitArray<Backend, Index> &index,
                             const MaskType &mask = true) {
        return steal(
            jit_var_new_gather(src.index(), index.index(), mask.index()));
    }

    template <bool, typename Index>
    void scatter_(void */*dst*/, const JitArray<Backend, Index> &/*index*/,
                  const MaskType &/*mask*/ = true) const {
        enoki_raise("Not implemented, please use scatter(JitArray src, index, mask) instead!");
    }

    template <bool, typename Index>
    void scatter_(JitArray &dst, const JitArray<Backend, Index> &index,
                  const MaskType &mask = true) const {
        dst = steal(jit_var_new_scatter(dst.index(), m_index,
                                        index.index(), mask.index(),
                                        ReduceOp::None));
    }

    template <typename Index>
    void scatter_reduce_(ReduceOp /*op*/, void */*dst*/,
                         const JitArray<Backend, Index> &/*index*/,
                         const MaskType &/*mask*/ = true) const {
        enoki_raise("Not implemented, please use scatter_reduce(JitArray src, index, op, mask) instead!");
    }

    template <typename Index>
    void scatter_reduce_(ReduceOp op, JitArray &dst,
                         const JitArray<Backend, Index> &index,
                         const MaskType &mask = true) const {
        dst = steal(jit_var_new_scatter(dst.index(), m_index, index.index(),
                                        mask.index(), op));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    std::pair<VCallBucket *, uint32_t> vcall_() const {
        if constexpr (!IsClass) {
            enoki_raise("Unsupported operand type");
        } else {
            uint32_t bucket_count = 0;
            VCallBucket *buckets =
                jit_vcall(Backend, CallSupport::Domain, m_index, &bucket_count);
            return { buckets, bucket_count };
        }
    }

    JitArray<Backend, uint32_t> compress_() const {
        if constexpr (!is_mask_v<Value>) {
            enoki_raise("Unsupported operand type");
        } else {
            uint32_t size_in = (uint32_t) size();
            uint32_t *indices = (uint32_t *) jit_malloc(
                AllocType::Device, size_in * sizeof(uint32_t));

            eval_();
            uint32_t size_out = jit_compress(Backend, (const uint8_t *) data(), size_in, indices);
            return JitArray<Backend, uint32_t>::steal(
                jit_var_mem_map(Backend, VarType::UInt32, indices, size_out, 1));
        }
    }

    JitArray copy() const { return steal(jit_var_copy(m_index)); }


    bool schedule_() const { return jit_var_schedule(m_index) != 0; }
    bool eval_() const { return jit_var_eval(m_index) != 0; }

    bool valid() const { return m_index != 0; }
    size_t size() const { return jit_var_size(m_index); }
    uint32_t index() const { return m_index; }
    uint32_t* index_ptr() { return &m_index; }

    const Value *data() const { return (const Value *) jit_var_ptr(m_index); }
    Value *data() { return (Value *) jit_var_ptr(m_index); }

    const char *str() { return jit_var_str(m_index); }

    bool is_literal() const { return (bool) jit_var_is_literal(m_index); }

    Value entry(size_t offset) const {
        ActualValue out;
        jit_var_read(m_index, offset, &out);

        if constexpr (!IsClass)
            return out;
        else
            return (Value) jit_registry_get_ptr(CallSupport::Domain, out);
    }

    void set_entry(size_t offset, Value value) {
        uint32_t index;
        if constexpr (!IsClass) {
            index = jit_var_write(m_index, offset, &value);
        } else {
            ActualValue av = jit_registry_get_id(value);
            index = jit_var_write(m_index, (uint32_t) offset, &av);
        }
        jit_var_dec_ref_ext(m_index);
        m_index = index;
    }

	void resize(size_t size) {
        uint32_t index = jit_var_resize(m_index, size);
        jit_var_dec_ref_ext(m_index);
        m_index = index;
    }

    JitArray migrate_(AllocType type) const {
        return steal(jit_var_migrate(m_index, type));
    }

    static JitArray<Backend, uint32_t> counter(size_t size) {
        return JitArray<Backend, uint32_t>::steal(
            jit_var_new_counter(Backend, size));
    }

	void set_label_(const char *label) const {
		jit_var_set_label(m_index, label);
	}

	const char *label_() const {
		return jit_var_label(m_index);
	}

    const CallSupport operator->() const {
        return CallSupport(*this);
    }

    //! @}
    // -----------------------------------------------------------------------

    static JitArray steal(uint32_t index) {
        JitArray result;
        result.m_index = index;
        return result;
    }

    static JitArray borrow(uint32_t index) {
        JitArray result;
        jit_var_inc_ref_ext(index);
        result.m_index = index;
        return result;
    }

    void init_(size_t size) {
        *this = empty_(size);
    }

protected:
    uint32_t m_index = 0;
};

template <typename T> using CUDAArray = JitArray<JitBackend::CUDA, T>;
template <typename T> using LLVMArray = JitArray<JitBackend::LLVM, T>;

// #if defined(ENOKI_AUTODIFF_H)
// ENOKI_DECLARE_EXTERN_TEMPLATE(CUDAArray<float>, CUDAArray<bool>, CUDAArray<uint32_t>)
// ENOKI_DECLARE_EXTERN_TEMPLATE(CUDAArray<double>, CUDAArray<bool>, CUDAArray<uint32_t>)

// ENOKI_DECLARE_EXTERN_TEMPLATE(LLVMArray<float>, LLVMArray<bool>, LLVMArray<uint32_t>)
// ENOKI_DECLARE_EXTERN_TEMPLATE(LLVMArray<double>, LLVMArray<bool>, LLVMArray<uint32_t>)
// #endif

NAMESPACE_END(enoki)
