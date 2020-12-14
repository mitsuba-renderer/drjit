/*
    enoki/custom.h -- Facilities to implement custom differentiable operations

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/autodiff.h>

NAMESPACE_BEGIN(enoki)

template <typename Type_, typename Output_, typename... Input>
struct CustomOp : detail::DiffCallback {
    template <typename C, typename... Ts> friend auto custom(const Ts&... input);
public:
    using Type   = detached_t<Type_>;
    using Output = Output_;
    using Inputs = detail::tuple<Input...>;
    static constexpr bool ClearPrimal = true;

    /**
     * Evaluate the custom function in primal mode. The inputs will be detached
     * from the AD graph, and the output *must* also be detached.
     */
    virtual Output eval(const Input&... input) = 0;

    /// Callback to implement forward-mode derivatives
    virtual void forward() = 0;

    /// Callback to implement reverse-mode derivatives
    virtual void backward() = 0;

    /// Return a descriptive name (used in GraphViz output)
    virtual const char *name() const = 0;

protected:

    /// Check if gradients are enabled for a specific input variable
    template <size_t Index = 0>
    bool grad_enabled_in() const {
        return grad_enabled(m_grad_input->template get<Index>());
    }

    /// Access the gradient associated with the input argument 'Index' (fwd. mode AD)
    template <size_t Index = 0>
    typename Inputs::template type<Index> grad_in() const {
        return grad<false>(m_grad_input->template get<Index>());
    }

    /// Access the primal value associated with the input argument 'Index', requires ClearPrimal=false
    template <size_t Index = 0>
    typename Inputs::template type<Index> value_in() const {
        return detach<false>(m_grad_input->template get<Index>());
    }

    /// Access the gradient associated with the output argument (rev. mode AD)
    Output grad_out() const {
        return grad<false>(m_grad_output);
    }

    /// Accumulate a gradient value into an input argument (rev. mode AD)
    template <size_t Index = 0>
    void set_grad_in(const typename Inputs::template type<Index> &value) {
        accum_grad(m_grad_input->template get<Index>(), value);
    }

    /// Accumulate a gradient value into the output argument (fwd. mode AD)
    void set_grad_out(const Output &value) {
        accum_grad(m_grad_output, value);
    }

    detail::tiny_unique_ptr<Inputs> m_grad_input;
    Output m_grad_output;
};


NAMESPACE_BEGIN(detail)

// Collect indices of variables that are attached to the AD graph
template <typename T>
void diff_vars(const T &value, size_t &counter, uint32_t *out) {
    if constexpr (is_array_v<T>) {
        if constexpr (array_depth_v<T> == 1) {
            if constexpr (is_diff_array_v<T>) {
                if (value.index() > 0) {
                    if (out)
                        out[counter] = value.index();
                    counter++;
                }
            }
        } else {
            for (size_t i = 0; i < value.size(); ++i)
                diff_vars(value.entry(i), counter, out);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(value,
            [&](auto &x) { diff_vars(x, counter, out); }
        );
    }
}

// Clear the primal values associated with an array
template <typename T> T clear_primal(const T &value) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            T result;
            if constexpr (T::Size == Dynamic)
                result = empty<T>(value.size());

            for (size_t i = 0; i < value.size(); ++i)
                result.entry(i) = clear_primal(value.entry(i));

            return result;
        } else {
            return T::create_borrow(value.index(), typename T::Type());
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        T result;

        struct_support_t<T>::apply_2(
            value, result,
            [](auto const &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                x2 = clear_primal(x1);
            });

        return result;
    } else {
        return value;
    }
}

NAMESPACE_END(detail)

template <typename Custom, typename... Input> auto custom(const Input&... input) {
    using Type   = typename Custom::Type;
    using Output = typename Custom::Output;

    detail::tiny_unique_ptr<Custom> custom(new Custom());

    Output output = custom->eval(detach<false>(input)...);

    if (grad_enabled(output))
        enoki_raise("enoki::custom(): the return value of the CustomOp::eval() "
                    "implementation was attached to the AD graph. This is not "
                    "allowed.");

    // Collect the input autodiff variable indices
    size_t diff_vars_in_ctr = 0;
    (detail::diff_vars(input, diff_vars_in_ctr, nullptr), ...);

    if (diff_vars_in_ctr > 0) {
        // Gradients are enabled for at least one input, mark outputs
        enable_grad(output);
        const char *name = custom->name();
        size_t buf_size = strlen(name) + 5;
        char *buf = (char *) alloca(buf_size);
        set_label(output, name);

        if constexpr (Custom::ClearPrimal) {
            // Only retain variable indices
            custom->m_grad_input = new detail::tuple<Input...>(detail::clear_primal(input)...);
            custom->m_grad_output = detail::clear_primal(output);
        } else {
            custom->m_grad_input = new detail::tuple<Input...>(input...);
            custom->m_grad_output = output;
        }

        size_t diff_vars_out_ctr = 0;
        detail::diff_vars(output, diff_vars_out_ctr, nullptr);
        if (diff_vars_out_ctr == 0)
            enoki_raise("enoki::custom(): internal error!");

        detail::tiny_unique_ptr<uint32_t[]> diff_vars_in(new uint32_t[diff_vars_in_ctr]);
        detail::tiny_unique_ptr<uint32_t[]> diff_vars_out(new uint32_t[diff_vars_out_ctr]);

        diff_vars_in_ctr = diff_vars_out_ctr = 0;
        (detail::diff_vars(input, diff_vars_in_ctr, diff_vars_in.get()), ...);
        detail::diff_vars(output, diff_vars_out_ctr, diff_vars_out.get());

        // Create a dummy node in case the branch-in factor is > 1
        uint32_t in_var;
        if (diff_vars_in_ctr > 1) {
            in_var = detail::ad_new<Type>(nullptr, 0, 0, nullptr, (Type *) nullptr);
            snprintf(buf, buf_size, "%s_in", name);
            detail::ad_set_label<Type>(in_var, buf);
            for (uint32_t i = 0; i < diff_vars_in_ctr; ++i)
                detail::ad_add_edge<Type>(diff_vars_in[i], in_var);
        } else {
            in_var = diff_vars_in[0];
            detail::ad_inc_ref<Type>(in_var);
        }

        // Create a dummy node in case the branch-out factor is > 1
        uint32_t out_var;
        if (diff_vars_out_ctr > 1) {
            out_var = detail::ad_new<Type>(nullptr, 0, 0, nullptr, (Type *) nullptr);
            snprintf(buf, buf_size, "%s_out", name);
            detail::ad_set_label<Type>(out_var, buf);
            for (uint32_t i = 0; i < diff_vars_out_ctr; ++i)
                detail::ad_add_edge<Type>(out_var, diff_vars_out[i]);
        } else {
            out_var = diff_vars_out[0];
            detail::ad_inc_ref<Type>(out_var);
        }

        // Connect the two nodes using a custom edge with a callback
        detail::ad_add_edge<Type>(in_var, out_var, custom.release());
        detail::ad_dec_ref<Type>(out_var);
        detail::ad_dec_ref<Type>(in_var);
    }

    return output;
}

NAMESPACE_END(enoki)
