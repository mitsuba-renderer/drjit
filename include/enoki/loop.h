/*
    enoki/loop.h -- Infrastructure to execute CUDA&LLVM loops symbolically

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>
#include <enoki-jit/jit.h>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)
template <typename T1, typename...Ts>
struct extract_1 { using type = T1; };
NAMESPACE_END(detail)

template <typename... Args> struct loop {
    using Mask = mask_t<leaf_array_t<typename detail::extract_1<Args...>::type>>;
    static constexpr bool Enabled = is_jit_array_v<Mask>;

    loop(Args&... args) {
        if constexpr (Enabled) {
            // Count the JIT variable IDs of all arguments in 'args'
            (extract(args, false), ...);

            // Allocate storage for these indices
            m_variables = new uint32_t*[m_variable_count];
            m_variables_phi = new uint32_t[m_variable_count];

            // Collect the JIT variable IDs of all arguments in 'args'
            m_variable_count = 0;
            (extract(args, true), ...);

            init();
        }
    }

    ~loop() {
        if constexpr (Enabled) {
            delete[] m_variables;
            delete[] m_variables_phi;

            if (m_counter != 2)
                enoki_raise( // throwing from destructor, will abort the application
                    "enoki::loop::cond() must be called exactly twice! (this "
                    "should happen automatically in the following expression: "
                    "`while(loop.cond(...)) { ... }` )");
        }
    }

    bool cond(const Mask &mask) {
        if constexpr (!Enabled) {
            return (bool) mask;
        } else {
            uint32_t prev;

            if (m_counter == 0) {
                if constexpr (is_llvm_array_v<Mask>)
                    m_mask = mask;

                uint32_t mask_index = 0;
                if (is_diff_array_v<Mask>)
                    mask_index = detach(mask).index();
                else
                    mask_index = mask.index();

                // Reduce loop condition to a single bit
                prev = m_id;
                m_id = jitc_var_new_2(
                    VarType::Bool,
                    "$r0 = call i1 "
                    "@llvm.experimental.vector.reduce.or.v$wi1(<$w x i1> $r1)",
                    1, 0, mask_index, m_id);
                jitc_var_dec_ref_ext(prev);

                // Branch to end of loop if all done
                prev = m_id;
                m_id = jitc_var_new_2(
                    VarType::Invalid,
                    "br $t1 $r1, label %$L2_body, label %$L2_after", 1, 0,
                    m_id, m_loop_id);
                jitc_var_dec_ref_ext(prev);

                // Start the main loop body
                prev = m_id;
                m_id = jitc_var_new_2(
                    VarType::Invalid, "\n$L1_body:", 1, 0, m_loop_id, m_id);
                jitc_var_dec_ref_ext(prev);

                for (size_t i = 0; i < m_variable_count; ++i) {
                    uint32_t *idp = m_variables[i],
                             id = jitc_var_new_3(
                                 jitc_var_type(*m_variables[i]),
                                 "$r0 = phi <$w x $t0> [ $r1, %$L2_header ]", 1,
                                 0, m_variables_phi[i], m_loop_id, m_id);
                    jitc_var_dec_ref_ext(*idp);
                    *idp = id;
                }
            } else if (m_counter == 1) {
                uint32_t mask_index = 0;
                if (is_diff_array_v<Mask>)
                    mask_index = detach(m_mask).index();
                else
                    mask_index = m_mask.index();

                for (size_t i = 0; i < m_variable_count; ++i) {
                    prev = m_id;
                    m_id = jitc_var_new_4(
                        VarType::Invalid,
                        "$r3_end = select <$w x $t1> $r1, <$w x $t2> $r2, <$w x $t3> $r3",
                        1, 0, mask_index, *m_variables[i], m_variables_phi[i], m_id);
                    jitc_var_dec_ref_ext(prev);
                }

                m_mask = Mask();

                prev = m_id;
                m_id = jitc_var_new_2(VarType::Invalid,
                                      "br label %$L1_header\n\n$L1_after:", 1,
                                      0, m_loop_id, m_id);
                jitc_var_dec_ref_ext(prev);

            } else {
                enoki_raise("enoki::loop::cond() was called more than twice!");
            }

            for (size_t i = 0; i < m_variable_count; ++i) {
                uint32_t *idp = m_variables[i],
                         id = jitc_var_new_3(
                             jitc_var_type(*m_variables[i]),
                             "$r0 = phi <$w x $t0> [ $r1, %$L2_header ]", 1,
                             0, m_variables_phi[i], m_loop_id, m_id);
                jitc_var_dec_ref_ext(*idp);
                *idp = id;
            }

            if (m_counter == 1) {
                if (jitc_side_effect_counter() != m_side_effect_counter) {
                    /* There was a side effect somewhere in the loop.
                       Create a dummy variable (also a side effect) that
                       depends on the final branch statement to ensure that the
                       loop is correctly generated. */
                    uint32_t idx =
                        jitc_var_new_1(VarType::Invalid, "", 1, m_cuda, m_id);
                    jitc_var_mark_scatter(idx, 0);
                }

                // Clean up
                jitc_var_dec_ref_ext(m_id);
                jitc_set_cse(m_cse_enabled);
                jitc_set_eval_enabled(m_eval_enabled);
                m_loop_id = m_id = 0;
            }

            return m_counter++ == 0;
        }
    }

    const Mask &mask() const { return m_mask; }

private:
    void init() {
        if constexpr (Enabled) {
            if (m_cuda == m_llvm)
                enoki_raise("enoki::loop(): expected either CUDA or LLVM array "
                            "arguments!");
            else if (m_other)
                enoki_raise("enoki::loop(): mixture of JIT (CUDA/LLVM) and "
                            "non-JIT values specified as loop variables!");

            // Prevent CSE interactions between code inside & outside the loop
            m_cse_enabled = jitc_cse();
            jitc_set_cse(false);

            /// Temporarily disallow any calls to jitc_eval()
            m_eval_enabled = jitc_eval_enabled();
            jitc_set_eval_enabled(false);

            m_side_effect_counter = jitc_side_effect_counter();

            /* Generate a sequence of dummy instructions to ensure that all
               loop variables are evaluated before the loop starts */
            uint32_t prev;
            for (size_t i = 0; i < m_variable_count; ++i) {
                uint32_t id = *m_variables[i];
                if (id == 0)
                    enoki_raise("All variables provided to enoki::loop() must be initialized!");

                if (m_id == 0) {
                    jitc_var_inc_ref_ext(id);
                    m_id = id;
                } else {
                    prev = m_id;
                    m_id = jitc_var_new_2(VarType::Invalid, "", 1, m_cuda, id, m_id);
                    jitc_var_dec_ref_ext(prev);
                }
            }

            // Jump to beginning of loop
            prev = m_id;
            m_loop_id = m_id = jitc_var_new_1(VarType::Invalid,
                 "br label %$L0\n\n$L0:$n"
                 "br label %$L0_header\n\n$L0_header:",
                 1, 0, m_id);
            jitc_var_dec_ref_ext(prev);

            // Create a phi expression per loop variable
            for (size_t i = 0; i < m_variable_count; ++i) {
                uint32_t *idp = m_variables[i];

                prev = m_id;
                m_id = jitc_var_new_3(
                    jitc_var_type(*idp),
                    "$r0 = phi <$w x $t0> [ $r1, %$L2 ], [ $r0_end, %$L2_body ]",
                    1, 0, *idp, m_loop_id, m_id);

                jitc_var_dec_ref_ext(prev);
                jitc_var_dec_ref_ext(*idp);
                jitc_var_inc_ref_ext(m_id);

                *idp = m_variables_phi[i] = m_id;
            }
        }
    }


    /// Extracts JIT variable indices of loop variables
    template <typename T> void extract(T &value, bool store) {
        if constexpr (is_array_v<T>) {
            if constexpr (array_depth_v<T> == 1) {
                if constexpr (is_diff_array_v<T>) {
                    extract(value.detach_(), store);
                } else if constexpr (is_jit_array_v<T>) {
                    if constexpr (is_cuda_array_v<T>)
                        m_cuda = true;
                    else if constexpr (is_llvm_array_v<T>)
                        m_llvm = true;
                    if (store)
                        m_variables[m_variable_count] = value.index_ptr();
                    m_variable_count++;
                } else {
                    m_other = true;
                }
            } else {
                for (size_t i = 0; i < value.size(); ++i)
                    extract(value.entry(i), store);
            }
        } else if constexpr (is_enoki_struct_v<T>) {
            struct_support_t<T>::apply_1(value,
                [&](auto &x) { extract(x, store); }
            );
        } else {
            m_other = true;
        }
    }

private:
    uint32_t **m_variables = nullptr;
    uint32_t *m_variables_phi = nullptr;
    size_t m_variable_count = 0;
    int m_counter = 0;
    bool m_cse_enabled = true;
    bool m_eval_enabled = true;
    bool m_cuda = false;
    bool m_llvm = false;
    bool m_other = false;
    uint32_t m_id = 0;
    uint32_t m_loop_id = 0;
    uint32_t m_side_effect_counter = 0;
    Mask m_mask = true;
};

NAMESPACE_END(enoki)
