/*
    drjit/custom.h -- Facilities to implement custom differentiable operations

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#define NB_INTRUSIVE_EXPORT DRJIT_EXTRA_EXPORT

#include <drjit/autodiff.h>
#include <drjit/extra.h>
#include <drjit-core/nanostl.h>

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

/**
 * Base class used to realize custom differentiable operations.
 *
 * Dr.Jit can compute derivatives of builtin operations in both forward and
 * reverse mode. In some cases, it may be useful or even necessary to tell
 * it how a particular operation should be differentiated.
 *
 * To do so, extend this class and and provide callback functions that will be
 * invoked when the AD backend traverses the associated node in the computation
 * graph. This class also provides a convenient way of stashing temporary
 * results during the original function evaluation that can be accessed later
 * on as part of forward or reverse-mode differentiation.
 */

class DRJIT_EXTRA_EXPORT CustomOpBase : public nanobind::intrusive_base {
    friend bool ::ad_custom_op(CustomOpBase*);
    friend bool ::ad_release_one_output(CustomOpBase*);
public:
    CustomOpBase();
    CustomOpBase(const CustomOpBase &) = delete;
    CustomOpBase(CustomOpBase &&) = delete;
    CustomOpBase& operator=(const CustomOpBase &) = delete;
    CustomOpBase& operator=(CustomOpBase &&) = delete;

    virtual ~CustomOpBase();

    /// Forward derivative callback, default implementation raises an exception
    virtual void forward();

    /// Backward derivative callback, default implementation raises an exception
    virtual void backward();

    /// Return a descriptive name (used in GraphViz output)
    virtual const char *name() const;

    /**
     * \brief Register an implicit input or output dependence
     *
     * This function should be called by the \ref eval() implementation when an
     * operation has a differentiable dependence on an *implicit* input (i.e.,
     * one that is not a regular input argument of the operation, such as a
     * private instance field).
     *
     * The function expects the AD index (``index_ad()``) value of the
     * variable.
     *
     * The function returns a boolean result stating whether the index was
     * added. A return value of ``false`` indicates that ``index`` is either
     * zero, or that gradient tracking was previously disabled for the
     * specified variable.
     */
    bool add_index(JitBackend backend, uint32_t index, bool input);

protected:
    /**
     * \brief Called by the AD layer to notify the CustomOp that one of its
     * outputs is no longer referenced. The operation returns ``false`` when
     * all outputs have expired, in which case the CustomOp can be freed.
     */
    bool release_one_output();

protected:
    JitBackend m_backend;
    uint32_t m_outputs_alive;
    uint64_t m_counter_offset;

    vector<uint32_t> m_input_indices;
    vector<uint32_t> m_output_indices;
};

template <typename T>
T ad_scan(CustomOpBase &op, const T &value, bool input) {
    if constexpr (is_diff_v<T>) {
        if constexpr (depth_v<T> > 1) {
            T result;
            if constexpr (T::Size == Dynamic)
                result = empty<T>(value.size());

            for (size_t i = 0; i < value.size(); ++i)
                result.entry(i) = ad_scan(op, value.entry(i), input);

            return result;
        } else {
            uint32_t ad_index = value.index_ad();
            op.add_index(backend_v<T>, ad_index, input);
            return T::borrow(((uint64_t) ad_index) << 32);
        }
    } else if constexpr (is_drjit_struct_v<T>) {
        T result;

        struct_support_t<T>::apply_2(
            value, result,
            [&](auto const &x1, auto &x2) DRJIT_INLINE_LAMBDA {
                x2 = ad_scan(op, x1, input);
            });

        return result;
    } else {
        return value;
    }
}

template <typename T> void new_grad(T &value) {
    if constexpr (is_diff_v<T>) {
        if constexpr (depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                new_grad(value.entry(i));
        } else if constexpr (is_tensor_v<T>) {
            new_grad(value.array());
        } else {
            value.new_grad_();
        }
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(value, [](auto &x) { new_grad(x); });
    } else {
        DRJIT_MARK_USED(value);
    }
}

NAMESPACE_END(detail)

template <typename Output_, typename... Input_>
class CustomOp : public detail::CustomOpBase {
    template <typename Op, typename... Ts>
    friend typename Op::Output custom(const Ts &...);

public:
    using Base     = detail::CustomOpBase;
    using Inputs   = drjit::tuple<Input_...>;
    using Output   = Output_;

    CustomOp(const Input_ &...in)
        : m_inputs(detail::ad_scan(*this, Inputs(in...), true)) { }

private:
    Inputs m_inputs;
    Output m_output;
};

template <typename Op, typename... Inputs>
typename Op::Output custom(const Inputs &...inputs) {
    nanobind::ref<Op> op = new Op(inputs...);

    // Perform the operations
    typename Op::Output output = op->eval(detach(inputs)...);

    // Ensure that the output is registered with the AD layer without depending
    // on previous computation. That dependence is reintroduced later below.
    detail::new_grad(output);

    op->m_output = detail::ad_scan(op, output, false);

    // Tie the operation into the AD graph, or detach if unsuccessful
    if (!ad_custom_op(op.get()))
        disable_grad(output);

    return output;
}

NAMESPACE_END(drjit)
