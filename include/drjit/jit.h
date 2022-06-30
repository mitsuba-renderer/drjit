/*
    drjit/jit.h -- Dr.Jit dynamic array with JIT compilation

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once
#define DRJIT_H

#include <drjit/array.h>
#include <drjit-core/traits.h>

NAMESPACE_BEGIN(drjit)

template <JitBackend Backend_, typename Value_, typename Derived_>
struct JitArray : ArrayBase<Value_, is_mask_v<Value_>, Derived_> {
    static_assert(std::is_scalar_v<Value_>,
                  "JIT Arrays can only be created over scalar types!");

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Value = Value_;
    using Derived = Derived_;
    using Base = ArrayBase<Value_, is_mask_v<Value_>, Derived_>;
    using Base::derived;

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
        call_support<std::decay_t<std::remove_pointer_t<Value_>>, Derived>;

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

    template <typename T, typename Derived2>
    JitArray(const JitArray<Backend, T, Derived2> &v) {
        m_index = jit_var_new_cast(v.index(), Type, 0);
    }

    template <typename T, typename Derived2>
    JitArray(const JitArray<Backend, T, Derived2> &v,
             detail::reinterpret_flag) {
        m_index = jit_var_new_cast(v.index(), Type, 1);
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    JitArray(T value) {
        ActualValue av;

        if constexpr (!IsClass)
            av = (ActualValue) value;
        else
            av = jit_registry_get_id(Backend, value);

        m_index = jit_var_new_literal(Backend, Type, &av, 1, 0, IsClass);
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    JitArray(Ts&&... ts) {
        if constexpr (!IsClass) {
            Value data[] = { (Value) ts... };
            m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                       sizeof...(Ts));
        } else {
            uint32_t data[] = { jit_registry_get_id(Backend, ts)... };
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

    Derived add_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Add, m_index, v.m_index));
    }

    Derived sub_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Sub, m_index, v.m_index));
    }

    Derived mul_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Mul, m_index, v.m_index));
    }

    Derived mulhi_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Mulhi, m_index, v.m_index));
    }

    Derived div_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Div, m_index, v.m_index));
    }

    Derived mod_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Mod, m_index, v.m_index));
    }

    auto gt_(const Derived &v) const {
        return mask_t<Derived>::steal(jit_var_new_op_2(JitOp::Gt, m_index, v.m_index));
    }

    auto ge_(const Derived &v) const {
        return mask_t<Derived>::steal(jit_var_new_op_2(JitOp::Ge, m_index, v.m_index));
    }

    auto lt_(const Derived &v) const {
        return mask_t<Derived>::steal(jit_var_new_op_2(JitOp::Lt, m_index, v.m_index));
    }

    auto le_(const Derived &v) const {
        return mask_t<Derived>::steal(jit_var_new_op_2(JitOp::Le, m_index, v.m_index));
    }

    auto eq_(const Derived &v) const {
        return mask_t<Derived>::steal(jit_var_new_op_2(JitOp::Eq, m_index, v.m_index));
    }

    auto neq_(const Derived &v) const {
        return mask_t<Derived>::steal(jit_var_new_op_2(JitOp::Neq, m_index, v.m_index));
    }

    Derived neg_() const {
        return steal(jit_var_new_op_1(JitOp::Neg, m_index));
    }

    Derived not_() const {
        return steal(jit_var_new_op_1(JitOp::Not, m_index));
    }

    template <typename T> Derived or_(const T &v) const {
        return steal(jit_var_new_op_2(JitOp::Or, m_index, v.index()));
    }

    template <typename T> Derived and_(const T &v) const {
        return steal(jit_var_new_op_2(JitOp::And, m_index, v.index()));
    }

    template <typename T> Derived xor_(const T &v) const {
        return steal(jit_var_new_op_2(JitOp::Xor, m_index, v.index()));
    }

    template <typename T> Derived andnot_(const T &a) const {
        return and_(a.not_());
    }

    template <int Imm> Derived sl_() const {
        return sl_(Imm);
    }

    Derived sl_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Shl, m_index, v.index()));
    }

    template <int Imm> Derived sr_() const {
        return sr_(Imm);
    }

    Derived sr_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Shr, m_index, v.index()));
    }

    Derived abs_() const {
        return steal(jit_var_new_op_1(JitOp::Abs, m_index));
    }

    Derived sqrt_() const {
        return steal(jit_var_new_op_1(JitOp::Sqrt, m_index));
    }

    Derived rcp_() const {
        return steal(jit_var_new_op_1(JitOp::Rcp, m_index));
    }

    Derived rsqrt_() const {
        return steal(jit_var_new_op_1(JitOp::Rsqrt, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    Derived exp2_() const {
        return steal(jit_var_new_op_1(JitOp::Exp2, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    Derived exp_() const {
        return exp2(InvLogTwo<T> * derived());
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    Derived log2_() const {
        return steal(jit_var_new_op_1(JitOp::Log2, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    Derived log_() const {
        return log2(derived()) * LogTwo<T>;
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    Derived sin_() const {
        return steal(jit_var_new_op_1(JitOp::Sin, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    Derived cos_() const {
        return steal(jit_var_new_op_1(JitOp::Cos, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float> && IsCUDA> = 0>
    std::pair<Derived, Derived> sincos_() const {
        return { sin_(), cos_() };
    }

    Derived minimum_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Min, m_index, v.index()));
    }

    Derived maximum_(const Derived &v) const {
        return steal(jit_var_new_op_2(JitOp::Max, m_index, v.index()));
    }

    Derived round_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Round, m_index));
    }

    template <typename T> T round2int_() const {
        return T(round(derived()));
    }

    Derived floor_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Floor, m_index));
    }

    template <typename T> T floor2int_() const {
        return T(floor(derived()));
    }

    Derived ceil_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Ceil, m_index));
    }

    template <typename T> T ceil2int_() const {
        return T(ceil(derived()));
    }

    Derived trunc_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Trunc, m_index));
    }

    template <typename T> T trunc2int_() const {
        return T(trunc(derived()));
    }

    Derived fmadd_(const Derived &b, const Derived &c) const {
        return steal(
            jit_var_new_op_3(JitOp::Fmadd, m_index, b.index(), c.index()));
    }

    Derived fmsub_(const Derived &b, const Derived &c) const {
        return fmadd_(b, -c);
    }

    Derived fnmadd_(const Derived &b, const Derived &c) const {
        return fmadd_(-b, c);
    }

    Derived fnmsub_(const Derived &b, const Derived &c) const {
        return fmadd_(-b, -c);
    }

    template <typename Mask>
    static Derived select_(const Mask &m, const Derived &t, const Derived &f) {
        static_assert(std::is_same_v<Mask, mask_t<Derived>>);
        return steal(
            jit_var_new_op_3(JitOp::Select, m.index(), t.index(), f.index()));
    }

    Derived popcnt_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Popc, m_index));
    }

    Derived lzcnt_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Clz, m_index));
    }

    Derived tzcnt_() const {
        return Derived::steal(jit_var_new_op_1(JitOp::Ctz, m_index));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const {
        if (size() == 0)
            return true;
        return jit_var_all(m_index);
    }

    bool any_() const {
        if (size() == 0)
            return false;
        return jit_var_any(m_index);
    }

    #define DRJIT_HORIZONTAL_OP(name, op, default_op)                          \
        Derived name##_() const {                                              \
            if (size() == 0)                                                   \
                default_op;                                                    \
            return steal(jit_var_reduce(m_index, op));                         \
        }

    DRJIT_HORIZONTAL_OP(sum,  ReduceOp::Add, return Derived(0))
    DRJIT_HORIZONTAL_OP(prod, ReduceOp::Mul, return Derived(1))
    DRJIT_HORIZONTAL_OP(min,  ReduceOp::Min, drjit_raise("min_(): zero-sized array!"))
    DRJIT_HORIZONTAL_OP(max,  ReduceOp::Max, drjit_raise("max_(): zero-sized array!"))

    #undef DRJIT_HORIZONTAL_OP

    Derived dot_(const Derived &a) const {
        return sum(derived() * a);
    }

    uint32_t count_() const {
        if constexpr (!is_mask_v<Value>)
            drjit_raise("Unsupported operand type");

        return sum(select(derived(), (uint32_t) 1, (uint32_t) 0)).entry(0);
    }

    //! @}
    // -----------------------------------------------------------------------

   // -----------------------------------------------------------------------
    //! @{ \name Fancy array initialization
    // -----------------------------------------------------------------------

    static Derived empty_(size_t size) {
        size_t byte_size = size * sizeof(Value);
        void *ptr =
            jit_malloc(Backend == JitBackend::CUDA ? AllocType::Device
                                                   : AllocType::HostAsync,
                       byte_size);
        return steal(
            jit_var_mem_map(Backend, Type, ptr, size, 1));
    }

    static Derived zero_(size_t size) {
        Value value = 0;
        return steal(jit_var_new_literal(Backend, Type, &value, size));
    }

    static Derived full_(Value value, size_t size) {
        ActualValue av;
        if constexpr (!IsClass)
            av = (ActualValue) value;
        else
            av = jit_registry_get_id(Backend, value);

        return steal(
            jit_var_new_literal(Backend, Type, &av, size, false, IsClass));
    }

    static Derived opaque_(Value value, size_t size) {
        ActualValue av;
        if constexpr (!IsClass)
            av = (ActualValue) value;
        else
            av = jit_registry_get_id(Backend, value);

        return steal(
            jit_var_new_literal(Backend, Type, &av, size, true, IsClass));
    }

    static Derived arange_(ssize_t start, ssize_t stop, ssize_t step) {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
        if (size == 0)
            return Derived();
        return fmadd(Derived(uint32_array_t<Derived>::counter(size)),
                     Derived((Value) step),
                     Derived((Value) start));
    }

    static Derived linspace_(Value min, Value max, size_t size, bool endpoint) {
        Value step = (max - min) / Value(size - ((endpoint && size > 1) ? 1 : 0));
        return fmadd(Derived(uint32_array_t<Derived>::counter(size)),
                     Derived(step),
                     Derived(min));
    }

    static Derived map_(void *ptr, size_t size, bool free = false) {
         return steal(jit_var_mem_map(Backend, Type, ptr, size, free ? 1 : 0));
    }

    static Derived load_(const void *ptr, size_t size) {
        if constexpr (!IsClass) {
            return steal(
                jit_var_mem_copy(Backend, AllocType::Host, Type, ptr, (uint32_t) size));
        } else {
            uint32_t *temp = new uint32_t[size];
            for (uint32_t i = 0; i < size; i++)
                temp[i] = jit_registry_get_id(Backend, ((const void **) ptr)[i]);
            Derived result = steal(
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
                    jit_registry_get_ptr(Backend, CallSupport::Domain, temp[i]);
            delete[] temp;
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather support
    // -----------------------------------------------------------------------

    template <bool, typename Index, typename Mask>
    static Derived gather_(const void * /*src*/, const Index & /*index*/,
                           const Mask & /*mask*/) {
        drjit_raise("Not implemented, please use gather() variant that takes a "
                    "array source argument.");
    }

    template <bool, typename Index, typename Mask>
    static Derived gather_(const Derived &src, const Index &index,
                           const Mask &mask) {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<Derived>>>);
        return steal(
            jit_var_new_gather(src.index(), index.index(), mask.index()));
    }

    template <bool, typename Index, typename Mask>
    void scatter_(void * /* dst */, const Index & /*index*/,
                  const Mask & /*mask*/) const {
        drjit_raise("Not implemented, please use scatter() variant that takes "
                    "a array target argument.");
    }

    template <bool, typename Index, typename Mask>
    void scatter_(Derived &dst, const Index &index, const Mask &mask) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<Derived>>>);
        dst = steal(jit_var_new_scatter(dst.index(), m_index, index.index(),
                                        mask.index(), ReduceOp::None));
    }

    template <typename Index, typename Mask>
    void scatter_reduce_(ReduceOp /*op*/, void * /*dst*/,
                         const Index & /*index*/,
                         const Mask & /* mask */) const {
        drjit_raise("Not implemented, please use scatter_reduce() variant that "
                    "takes a array target argument.");
    }

    template <typename Index, typename Mask>
    void scatter_reduce_(ReduceOp op, Derived &dst, const Index &index,
                         const Mask &mask) const {
        static_assert(
            std::is_same_v<detached_t<Mask>, detached_t<mask_t<Derived>>>);
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
            drjit_raise("Unsupported operand type");
        } else {
            uint32_t bucket_count = 0;
            VCallBucket *buckets = jit_var_vcall_reduce(
                Backend, CallSupport::Domain, m_index, &bucket_count);
            return { buckets, bucket_count };
        }
    }

    auto compress_() const {
        if constexpr (!is_mask_v<Value>) {
            drjit_raise("Unsupported operand type");
        } else {
            uint32_t size_in = (uint32_t) size();
            uint32_t *indices = (uint32_t *) jit_malloc(
                Backend == JitBackend::CUDA ? AllocType::Device
                                            : AllocType::HostAsync,
                size_in * sizeof(uint32_t));

            eval_();
            uint32_t size_out = jit_compress(Backend, (const uint8_t *) data(),
                                             size_in, indices);
            if (size_out > 0) {
                return int32_array_t<Derived>::steal(
                    jit_var_mem_map(Backend, VarType::UInt32, indices, size_out, 1));
            } else {
                jit_free(indices);
                return int32_array_t<Derived>();
            }
        }
    }

    Derived block_sum_(size_t block_size) {
        size_t input_size  = size(),
               block_count = input_size / block_size;

        if (block_count * block_size != input_size)
            drjit_raise("block_sum(): input size must be a multiple of block_size!");

        Derived output = empty_(block_count);

        jit_block_sum(Derived::Backend, Derived::Type, data(), output.data(),
                      (uint32_t) block_count, (uint32_t) block_size);

        return output;
    }

    Derived copy() const { return steal(jit_var_copy(m_index)); }

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
    bool is_evaluated() const { return (bool) jit_var_is_evaluated(m_index); }
    bool is_placeholder() const { return (bool) jit_var_is_placeholder(m_index); }

    Value entry(size_t offset) const {
        ActualValue out;
        jit_var_read(m_index, offset, &out);

        if constexpr (!IsClass)
            return out;
        else
            return (Value) jit_registry_get_ptr(Backend, CallSupport::Domain, out);
    }

    void set_entry(size_t offset, Value value) {
        uint32_t index;
        if constexpr (!IsClass) {
            index = jit_var_write(m_index, offset, &value);
        } else {
            ActualValue av = jit_registry_get_id(Backend, value);
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

    Derived migrate_(AllocType type) const {
        return steal(jit_var_migrate(m_index, type));
    }

    static auto counter(size_t size) {
        return uint32_array_t<Derived>::steal(
            jit_var_new_counter(Backend, size));
    }

	void set_label_(const char *label) {
        uint32_t index = jit_var_set_label(m_index, label);
        jit_var_dec_ref_ext(m_index);
        m_index = index;
	}

	const char *label_() const {
		return jit_var_label(m_index);
	}

    const CallSupport operator->() const {
        return CallSupport(derived());
    }

    //! @}
    // -----------------------------------------------------------------------

    static Derived steal(uint32_t index) {
        Derived result;
        result.m_index = index;
        return result;
    }

    static Derived borrow(uint32_t index) {
        Derived result;
        jit_var_inc_ref_ext(index);
        result.m_index = index;
        return result;
    }

    void init_(size_t size) {
        derived() = empty_(size);
    }

protected:
    uint32_t m_index = 0;
};

template <typename Value>
struct CUDAArray : JitArray<JitBackend::CUDA, Value, CUDAArray<Value>> {
    using Base = JitArray<JitBackend::CUDA, Value, CUDAArray<Value>>;
    using MaskType = CUDAArray<bool>;
    using ArrayType = CUDAArray;
    template <typename T> using ReplaceValue = CUDAArray<T>;
    DRJIT_ARRAY_IMPORT(CUDAArray, Base)
};

template <typename Value>
struct LLVMArray : JitArray<JitBackend::LLVM, Value, LLVMArray<Value>> {
    using Base = JitArray<JitBackend::LLVM, Value, LLVMArray<Value>>;
    using MaskType = LLVMArray<bool>;
    using ArrayType = LLVMArray;
    template <typename T> using ReplaceValue = LLVMArray<T>;
    DRJIT_ARRAY_IMPORT(LLVMArray, Base)
};

#if defined(DRJIT_AUTODIFF_H)
DRJIT_DECLARE_EXTERN_TEMPLATE(CUDAArray<float>, CUDAArray<bool>, CUDAArray<uint32_t>)
DRJIT_DECLARE_EXTERN_TEMPLATE(CUDAArray<double>, CUDAArray<bool>, CUDAArray<uint32_t>)
DRJIT_DECLARE_EXTERN_TEMPLATE(LLVMArray<float>, LLVMArray<bool>, LLVMArray<uint32_t>)
DRJIT_DECLARE_EXTERN_TEMPLATE(LLVMArray<double>, LLVMArray<bool>, LLVMArray<uint32_t>)
#endif

template <typename Mask, typename... Ts>
void printf_async(const Mask &mask, const char *fmt, const Ts &... ts) {
    constexpr bool Active = is_jit_v<Mask> || (is_jit_v<Ts> || ...);
    static_assert(!Active || (is_jit_v<Mask> && array_depth_v<Mask> == 1 && is_mask_v<Mask>),
                  "printf_async(): 'mask' argument must be CUDA/LLVM mask "
                  "array of depth 1");
    static_assert(!Active || ((is_jit_v<Ts> && array_depth_v<Ts> == 1) && ...),
                  "printf_async(): variadic arguments must be CUDA/LLVM arrays "
                  "of depth 1");
    if constexpr (Active) {
        uint32_t indices[] = { ts.index()... };
        jit_var_printf(detached_t<Mask>::Backend, mask.index(), fmt,
                       (uint32_t) sizeof...(Ts), indices);
    }
}

template <typename Array>
Array block_sum(const Array &array, size_t block_size) {
    if constexpr (array_depth_v<Array> > 1) {
        Array result;
        if constexpr (Array::Size == Dynamic)
            result = empty<Array>(array.size());

        for (size_t i = 0; i < array.size(); ++i)
            result.entry(i) = block_sum(array.entry(i), block_size);

        return result;
    } else if constexpr (is_jit_v<Array>) {
        return array.block_sum_(block_size);
    } else {
        static_assert(detail::false_v<Array>, "block_sum(): requires a JIT array!");
    }
}

NAMESPACE_END(drjit)

#if defined(DRJIT_VCALL_H)
#  include <drjit/vcall_jit_reduce.h>
#  include <drjit/vcall_jit_record.h>
#endif
