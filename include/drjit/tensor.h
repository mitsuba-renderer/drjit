/*
    drjit/tensor.h -- Tensorial wrapper of a dynamic 1D array

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>
#include <drjit-core/containers.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

template <typename Index, typename T>
void tensor_broadcast_impl(const char *op, T &t, const dr_vector<size_t> &shape) {
    DRJIT_MARK_USED(op);
    int ndim = (int) t.ndim();
    if (ndim == 0 || memcmp(t.shape().data(), shape.data(), sizeof(size_t) * ndim) == 0)
        return;

    uint32_t size = 1;
    for (int i = 0; i < ndim; ++i)
        size *= (uint32_t) shape[i];

    Index index = arange<Index>(size);
    size = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        uint32_t size_next = size * (uint32_t) shape[i];
        if (t.shape(i) == 1 && shape[i] != 1)
            index = (index % size) + (index / size_next) * size;
        size = size_next;
    }

    t = T(gather<typename T::Array>(t.array(), index), shape);
}

template <typename T0, typename T1>
dr_vector<size_t> tensor_broadcast(const char *op, T0 &t0, T1 &t1) {
    size_t t0d = t0.ndim(), t1d = t1.ndim(),
           ndim = drjit::maximum(t0d, t1d);

    if ((t0d != ndim && t0d != 0) || (t1d != ndim && t1d != 0))
        drjit_raise("drjit::Tensor::%s(): incompatible tensor dimensions "
                    "(%zu and %zu)!", op, t0d, t1d);

    dr_vector<size_t> shape(ndim, 0);
    for (size_t i = 0; i < ndim; ++i) {
        size_t t0_i = t0d > 0 ? t0.shape(i) : 0;
        size_t t1_i = t1d > 0 ? t1.shape(i) : 0;
        shape[i] = drjit::maximum(t0_i, t1_i);
        if (t0_i > 1 && t1_i > 1 && t0_i != t1_i)
            drjit_raise("drjit::Tensor::%s(): incompatible tensor shapes "
                        "for dimension %zu (%zu and %zu)!", op, i, t0_i, t1_i);
    }

    using Index = typename T0::Index;
    tensor_broadcast_impl<Index>(op, t0, shape);
    tensor_broadcast_impl<Index>(op, t1, shape);

    return shape;
}

template <typename T0, typename T1, typename T2>
dr_vector<size_t> tensor_broadcast(const char *op, T0 &t0, T1 &t1, T2 &t2) {
    size_t t0d = t0.ndim(), t1d = t1.ndim(), t2d = t2.ndim();
    size_t ndim = drjit::maximum(drjit::maximum(t0d, t1d), t2d);

    if ((t0d != ndim && t0d != 0) || (t1d != ndim && t1d != 0) ||
        (t2d != ndim && t2d != 0))
        drjit_raise("drjit::Tensor::%s(): incompatible tensor dimensions "
                    "(%zu, %zu, and %zu)!", op, t0d, t1d, t2d);

    dr_vector<size_t> shape(ndim, 0);
    for (size_t i = 0; i < ndim; ++i)
        shape[i] = drjit::maximum(drjit::maximum(t0d > 0 ? t0.shape(i) : 0,
                                                 t1d > 0 ? t1.shape(i) : 0),
                                                 t2d > 0 ? t2.shape(i) : 0);

    using Index = typename T0::Index;
    tensor_broadcast_impl<Index>(op, t0, shape);
    tensor_broadcast_impl<Index>(op, t1, shape);
    tensor_broadcast_impl<Index>(op, t2, shape);

    return shape;
}

NAMESPACE_END(detail)


template <typename Array_>
struct Tensor
    : ArrayBase<value_t<Array_>, is_mask_v<Array_>, Tensor<Array_>> {

    template <typename Array2> friend struct Tensor;

    template <typename Index, typename T>
    friend void detail::tensor_broadcast_impl(const char *op, T &t,
                                              const dr_vector<size_t> &shape);

    using Base = ArrayBase<value_t<Array_>, is_mask_v<Array_>, Tensor<Array_>>;
    using Array = Array_;
    using Value = typename Array::Value;
    using Index = uint32_array_t<Array>;

    using ArrayType = Tensor<array_t<Array>>;
    using MaskType  = Tensor<mask_t<Array>>;
    using Shape     = dr_vector<size_t>;

    static constexpr bool IsMask = is_mask_v<Array_>;
    static constexpr bool IsTensor = true;
    static constexpr bool IsDynamic = true;
    static constexpr bool IsDiff = is_diff_v<Array_>;
    static constexpr bool IsJIT  = is_jit_v<Array_>;
    static constexpr bool IsCUDA = is_cuda_v<Array_>;
    static constexpr bool IsLLVM = is_llvm_v<Array_>;
    static constexpr size_t Size = Dynamic;

    template <typename T>
    using ReplaceValue = Tensor<typename Array::template ReplaceValue<T>>;

    DRJIT_ARRAY_IMPORT(Tensor, Base)

    template <typename T2>
    Tensor(const Tensor<T2> &t2) : m_array(t2.m_array), m_shape(t2.m_shape) { }

    template <typename T2>
    Tensor(const Tensor<T2> &t2, detail::reinterpret_flag)
        : m_array(t2.m_array, detail::reinterpret_flag()), m_shape(t2.m_shape) { }

    Tensor(const Array &data) : m_array(data) {
        size_t size = m_array.size();
        if (size != 0 && size != 1)
            drjit_raise("Tensor(): initialization with a non-trivial array "
                        "(size %u) requires specifying the 'shape' parameter.", size);
    }

    Tensor(const Array &data, size_t ndim, const size_t *shape)
        : m_array(data), m_shape(shape, shape + ndim) {
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i)
            size *= shape[i];
        if (size != m_array.size()) {
            if (m_array.size() == 1)
                resize(m_array, size);
            else
                drjit_raise("Tensor(): invalid size specified (%zu vs %zu)!",
                            size, m_array.size());
        }
    }

    Tensor(const void *ptr, size_t ndim, const size_t *shape)
        : m_shape(shape, shape + ndim) {
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i)
            size *= shape[i];
        m_array = load<Array>(ptr, size);
    }

    template <typename T, enable_if_t<std::is_scalar_v<T> && !std::is_pointer_v<T>> = 0>
    Tensor(T value) : m_array(value) { }

    operator Array() const { return m_array; }

    Tensor add_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        auto shape = detail::tensor_broadcast("add_", t0, t1);
        return Tensor(t0.m_array + t1.m_array, std::move(shape));
    }

    Tensor sub_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        auto shape = detail::tensor_broadcast("add_", t0, t1);
        return Tensor(t0.m_array - t1.m_array, std::move(shape));
    }

    Tensor mul_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("mul_", t0, t1);
        return Tensor(t0.m_array * t1.m_array, std::move(shape));
    }

    Tensor mulhi_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("mulhi_", t0, t1);
        return Tensor(mulhi(t0.m_array, t1.m_array), std::move(shape));
    }

    Tensor div_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("div_", t0, t1);
        return Tensor(t0.m_array / t1.m_array, std::move(shape));
    }

    Tensor mod_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("mod_", t0, t1);
        return Tensor(t0.m_array % t1.m_array, std::move(shape));
    }

    Tensor or_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("or_", t0, t1);
        return Tensor(t0.m_array | t1.m_array, std::move(shape));
    }

    Tensor and_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("and_", t0, t1);
        return Tensor(t0.m_array & t1.m_array, std::move(shape));
    }

    Tensor andnot_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("andnot_", t0, t1);
        return Tensor(andnot(t0.m_array, t1.m_array), std::move(shape));
    }

    Tensor xor_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("xor_", t0, t1);
        return Tensor(t0.m_array ^ t1.m_array, std::move(shape));
    }

    Tensor fmadd_(const Tensor &b, const Tensor &c) const {
        Tensor t0 = *this, t1 = b, t2 = c;
        Shape shape = detail::tensor_broadcast("fmadd_", t0, t1, t2);
        return Tensor(fmadd(t0.m_array, t1.m_array, t2.m_array), std::move(shape));
    }

    Tensor fmsub_(const Tensor &b, const Tensor &c) const {
        return fmadd_(b, -c);
    }

    Tensor fnmadd_(const Tensor &b, const Tensor &c) const {
        return fmadd_(-b, c);
    }

    Tensor fnmsub_(const Tensor &b, const Tensor &c) const {
        return fmadd_(-b, -c);
    }

    Tensor abs_() const { return Tensor(abs(m_array), m_shape); }

    Tensor minimum_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("minimum_", t0, t1);
        return Tensor(drjit::minimum(t0.m_array, t1.m_array), std::move(shape));
    }

    Tensor maximum_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("maximum_", t0, t1);
        return Tensor(drjit::maximum(t0.m_array, t1.m_array), std::move(shape));
    }

    auto gt_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("gt_", t0, t1);
        return mask_t<Tensor>(t0.m_array > t1.m_array, std::move(shape));
    }

    auto ge_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("ge_", t0, t1);
        return mask_t<Tensor>(t0.m_array >= t1.m_array, std::move(shape));
    }

    auto lt_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("lt_", t0, t1);
        return mask_t<Tensor>(t0.m_array < t1.m_array, std::move(shape));
    }

    auto le_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("le_", t0, t1);
        return mask_t<Tensor>(t0.m_array <= t1.m_array, std::move(shape));
    }

    auto eq_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("eq_", t0, t1);
        return mask_t<Tensor>(eq(t0.m_array, t1.m_array), std::move(shape));
    }

    auto neq_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("neq_", t0, t1);
        return mask_t<Tensor>(neq(t0.m_array, t1.m_array), std::move(shape));
    }

    Tensor neg_() const { return Tensor(-m_array, m_shape); }
    Tensor not_() const { return Tensor(~m_array, m_shape); }

    #define F(op) Tensor op##_() const { return Tensor(op(m_array), m_shape); }
    F(rcp) F(sqrt) F(rsqrt) F(sin) F(cos) F(tan) F(csc) F(sec) F(cot) F(asin)
    F(acos) F(atan) F(exp) F(exp2) F(log) F(log2) F(cbrt) F(erf) F(erfinv)
    F(lgamma) F(tgamma) F(sinh) F(cosh) F(tanh) F(csch) F(sech)
    F(coth) F(asinh) F(acosh) F(atanh)
    #undef F

    std::pair<Tensor, Tensor> sincos_() const {
        auto [s, c] = sincos(m_array);
        return { Tensor(std::move(s), m_shape),  Tensor(std::move(c), m_shape) };
    }

    std::pair<Tensor, Tensor> sincosh_() const {
        auto [s, c] = sincosh(m_array);
        return { Tensor(std::move(s), m_shape),  Tensor(std::move(c), m_shape) };
    }

    Tensor atan2_(const Tensor &b) const {
        Tensor t0 = *this, t1 = b;
        Shape shape = detail::tensor_broadcast("atan2_", t0, t1);
        return Tensor(drjit::atan2(t0.m_array, t1.m_array), std::move(shape));
    }

    template <typename Mask>
    static Tensor select_(const Mask &m, const Tensor &t, const Tensor &f) {
        static_assert(std::is_same_v<Mask, mask_t<Tensor>>);
        Tensor t_ = t, f_ = f;
        Mask m_ = m;
        Shape shape = detail::tensor_broadcast("select_", m_, t_, f_);
        return Tensor(select(m_.m_array, t_.m_array, f_.m_array), shape);
    }

    static Tensor zero_(size_t size) {
        return Tensor(zeros<Array>(size));
    }

    size_t ndim() const { return m_shape.size(); }
    size_t size() const { return m_array.size(); }
    size_t shape(size_t i) const {
        if (i >= m_shape.size())
            drjit_raise("Tensor::shape(%zu): out of bounds!", i);
        return m_shape[i];
    }

    Array &array() { return m_array; }
    const Array &array() const { return m_array; }
    Shape &shape() { return m_shape; }
    const Shape &shape() const { return m_shape; }

    const Value *data() const { return m_array.data(); }
    Value *data() { return m_array.data(); }

protected:
    Tensor(Array &&data, const Shape &shape)
        : m_array(std::move(data)), m_shape(shape) { }

    Tensor(Array &&data, Shape &&shape)
        : m_array(std::move(data)), m_shape(shape) { }

protected:
    Array m_array;
    Shape m_shape;
};

NAMESPACE_END(drjit)
