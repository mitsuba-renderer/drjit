/*
    drjit/custom.h -- Facilities to implement custom differentiable operations

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/autodiff.h>
#include <drjit-core/containers.h>

NAMESPACE_BEGIN(drjit)

namespace detail { template <typename T> void clear_diff_vars(T &); };

template <typename DiffType_, typename Output_, typename... Input>
struct CustomOp : detail::DiffCallback {
    template <typename C, typename... Ts> friend auto custom(const Ts&... input);
public:
    using DiffType = DiffType_;
    using Type     = detached_t<DiffType_>;
    using Output   = Output_;
    using Inputs   = dr_tuple<Input...>;

    template <size_t Index>
    using InputType = typename Inputs::template type<Index>;

    static constexpr bool ClearPrimal   = true;

    virtual ~CustomOp() {
        /* Important: reference counts associated with 'm_output' were cleared
           in custom() below to ensure that this edge can be garbage collected.
           We therefore need to clear the variable indices to prevent a second
           reference count decrease from occurring. */
        detail::clear_diff_vars(m_output);
        clear_implicit_dependencies();
    }

    /**
     * Evaluate the custom function in primal mode. The inputs will be detached
     * from the AD graph, and the output *must* also be detached.
     */
    virtual Output eval(const Input&... input) = 0;

    /// Callback to implement forward-mode derivatives
    virtual void forward() = 0;

    /// Callback to implement backward-mode derivatives
    virtual void backward() = 0;

    /// Return a descriptive name (used in GraphViz output)
    virtual const char *name() const = 0;

protected:

    /// Check if gradients are enabled for a specific input variable
    template <size_t Index = 0>
    bool grad_enabled_in() const {
        return grad_enabled(m_inputs->template get<Index>());
    }

    /// Access the gradient associated with the input argument 'Index' (fwd. mode AD)
    template <size_t Index = 0>
    InputType<Index> grad_in() const {
        return grad<false>(m_inputs->template get<Index>());
    }

    /// Access the primal value associated with the input argument 'Index', requires ClearPrimal=false
    template <size_t Index = 0>
    InputType<Index> value_in() const {
        return detach<false>(m_inputs->template get<Index>());
    }

    /// Access the gradient associated with the output argument (backward mode AD)
    Output grad_out() const {
        return grad<false, false>(m_output);
    }

    /// Accumulate a gradient value into an input argument (backward mode AD)
    template <size_t Index = 0>
    void set_grad_in(const InputType<Index> &value) {
        accum_grad(m_inputs->template get<Index>(), value);
    }

    /// Accumulate a gradient value into the output argument (forward mode AD)
    void set_grad_out(const Output &value) {
        accum_grad<false>(m_output, value);
    }

    /**
     * \brief Register an implicit input dependency of the operation on an AD
     * variable
     *
     * This function should be called by the \ref eval() implementation when an
     * operation has a differentiable dependence on an input that is not an
     * input argument (e.g. a private instance variable).
     */
    void add_input_index(uint32_t index) {
        if (!detail::ad_grad_enabled<Type>(index))
            return;
        detail::ad_inc_ref<Type>(index);
        m_implicit_in.push_back(index);
    }

    /// Convenience wrapper around \ref add_input_index
    template <typename T> void add_input(const T &value) {
        if constexpr (is_diff_v<T>) {
            if constexpr (array_depth_v<T> > 1) {
                for (size_t i = 0; i < value.size(); ++i)
                    add_input(value.entry(i));
            } else {
                add_input_index(value.index_ad());
            }
        } else if constexpr (is_drjit_struct_v<T>) {
            struct_support_t<T>::apply_1(value,
                [&](auto &x) { add_input(x); });
        }
    }

    /**
     * \brief Register an implicit output dependency of the operation on an AD
     * variable
     *
     * This function should be called by the \ref eval() implementation when an
     * operation has a differentiable dependence on an output that is not an
     * return value of the operation (e.g. a private instance variable).
     */
    void add_output_index(uint32_t index) {
        if (!detail::ad_grad_enabled<Type>(index))
            return;
        detail::ad_inc_ref<Type>(index);
        m_implicit_out.push_back(index);
    }

    /// Convenience wrapper around \ref add_output_index
    template <typename T> void add_output(const T &value) {
        if constexpr (is_diff_v<T>) {
            if constexpr (array_depth_v<T> > 1) {
                for (size_t i = 0; i < value.size(); ++i)
                    add_output(value.entry(i));
            } else {
                add_output_index(value.index_ad());
            }
        } else if constexpr (is_drjit_struct_v<T>) {
            struct_support_t<T>::apply_1(value,
                [&](auto &x) { add_output(x); });
        }
    }

    /// Release the implicit dependencies registered via add_input/add_output
    void clear_implicit_dependencies() {
        for (size_t i = 0; i < m_implicit_in.size(); ++i)
            detail::ad_dec_ref<Type>(m_implicit_in[i]);
        for (size_t i = 0; i < m_implicit_out.size(); ++i)
            detail::ad_dec_ref<Type>(m_implicit_out[i]);
        m_implicit_in.clear();
        m_implicit_out.clear();
    }
protected:
    dr_unique_ptr<Inputs> m_inputs;
    Output m_output;
    dr_vector<uint32_t> m_implicit_in, m_implicit_out;
};

NAMESPACE_BEGIN(detail)

// Zero out indices of variables that are attached to the AD graph
template <typename T>
void clear_diff_vars(T &value) {
    if constexpr (is_diff_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                clear_diff_vars(value.entry(i));
        } else {
            *value.index_ad_ptr() = 0;
        }
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(value,
            [](auto &x) { clear_diff_vars(x); });
    }
}

// Collect indices of variables that are attached to the AD graph
template <typename T>
void diff_vars(const T &value, size_t &counter, uint32_t *out) {
    if constexpr (is_array_v<T>) {
        if constexpr (array_depth_v<T> == 1) {
            if constexpr (is_diff_v<T>) {
                if (grad_enabled(value)) {
                    if (out)
                        out[counter] = value.index_ad();
                    counter++;
                }
            }
        } else {
            for (size_t i = 0; i < value.size(); ++i)
                diff_vars(value.entry(i), counter, out);
        }
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(value,
            [&](auto &x) { diff_vars(x, counter, out); }
        );
    }
}

// Clear the primal values associated with an array
template <typename T> T clear_primal(const T &value) {
    if constexpr (is_diff_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            T result;
            if constexpr (T::Size == Dynamic)
                result = empty<T>(value.size());

            for (size_t i = 0; i < value.size(); ++i)
                result.entry(i) = clear_primal(value.entry(i));

            return result;
        } else {
            return T::create_borrow(value.index_ad(), typename T::Type());
        }
    } else if constexpr (is_drjit_struct_v<T>) {
        T result;

        struct_support_t<T>::apply_2(
            value, result,
            [](auto const &x1, auto &x2) DRJIT_INLINE_LAMBDA {
                x2 = clear_primal(x1);
            });

        return result;
    } else {
        return value;
    }
}

NAMESPACE_END(detail)

template <typename Custom, typename... Input> auto custom(const Input&... input) {
    using Type     = typename Custom::Type;
    using Output   = typename Custom::Output;

    dr_unique_ptr<Custom> custom(new Custom());

    Output output = custom->eval(detach<false>(input)...);

    if (grad_enabled(output))
        drjit_raise("drjit::custom(): the return value of the CustomOp::eval() "
                    "implementation was attached to the AD graph. This is not "
                    "allowed.");

    // Collect the input autodiff variable indices
    size_t diff_vars_in_ctr = 0;
    (detail::diff_vars(input, diff_vars_in_ctr, nullptr), ...);

    if (diff_vars_in_ctr > 0 || custom->m_implicit_in.size() > 0) {
        uint32_t in_var  = detail::ad_new<Type>(nullptr, 0),
                 out_var = detail::ad_new<Type>(nullptr, 0);

        /* Gradients are enabled for at least one input, or the function
           accesses an instance variable with enabled gradients */
        enable_grad(output);

        if constexpr (Custom::ClearPrimal) {
            // Only retain variable indices
            custom->m_inputs = new dr_tuple<Input...>(detail::clear_primal(input)...);
            custom->m_output = detail::clear_primal(output);
        } else {
            custom->m_inputs = new dr_tuple<Input...>(input...);
            custom->m_output = output;
        }

        size_t diff_vars_out_ctr = 0;
        detail::diff_vars(output, diff_vars_out_ctr, nullptr);
        if (diff_vars_out_ctr + custom->m_implicit_out.size() == 0)
            return output; // Not relevant for AD after all..

        dr_unique_ptr<uint32_t[]> diff_vars_in(
            new uint32_t[diff_vars_in_ctr + custom->m_implicit_in.size()]);
        dr_unique_ptr<uint32_t[]> diff_vars_out(
            new uint32_t[diff_vars_out_ctr + custom->m_implicit_out.size()]);

        diff_vars_out_ctr = 0;
        diff_vars_in_ctr = 0;
        (detail::diff_vars(input, diff_vars_in_ctr, diff_vars_in.get()), ...);
        detail::diff_vars(output, diff_vars_out_ctr, diff_vars_out.get());

        /* Undo the reference count increases that resulted from storage in
           'm_output'. This is important to avoid a reference cycle that would
           prevent the CustomOp from being garbage collected. See also the
           CustomOp destructor. */
        for (size_t i = 0; i < diff_vars_out_ctr; ++i)
            detail::ad_dec_ref<Type>(diff_vars_out[i]);

        // Capture additional dependencies
        for (size_t i = 0; i < custom->m_implicit_in.size(); ++i)
            diff_vars_in[diff_vars_in_ctr++] = custom->m_implicit_in[i];

        for (size_t i = 0; i < custom->m_implicit_out.size(); ++i)
            diff_vars_out[diff_vars_out_ctr++] = custom->m_implicit_out[i];

        const char *name = custom->name();
        size_t buf_size = strlen(name) + 7;
        char *buf = (char *) alloca(buf_size);

        // Create a dummy node in case the branch-in factor is > 1
        if (diff_vars_in_ctr > 1 || diff_vars_in_ctr == 0) {
            snprintf(buf, buf_size, "%s [in]", name);
            detail::ad_set_label<Type>(in_var, buf);
            for (size_t i = 0; i < diff_vars_in_ctr; ++i)
                detail::ad_add_edge<Type>(diff_vars_in[i], in_var);
        } else {
            detail::ad_dec_ref<Type>(in_var);
            in_var = diff_vars_in[0];
            detail::ad_inc_ref<Type>(in_var);
        }

        // Create a dummy node in case the branch-out factor is > 1
        if (diff_vars_out_ctr > 1 || diff_vars_out_ctr == 0) {
            snprintf(buf, buf_size, "%s [out]", name);
            detail::ad_set_label<Type>(out_var, buf);
            for (size_t i = 0; i < diff_vars_out_ctr; ++i)
                detail::ad_add_edge<Type>(out_var, diff_vars_out[i]);
        } else {
            detail::ad_dec_ref<Type>(out_var);
            out_var = diff_vars_out[0];
            detail::ad_inc_ref<Type>(out_var);
        }

        custom->clear_implicit_dependencies();

        // Connect the two nodes using a custom edge with a callback
        detail::ad_add_edge<Type>(in_var, out_var, custom.release());
        detail::ad_dec_ref<Type>(in_var);
        detail::ad_dec_ref<Type>(out_var);
    }

    return output;
}

NAMESPACE_END(drjit)
