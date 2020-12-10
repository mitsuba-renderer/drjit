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

template <typename... Args> struct Loop {
    using Mask = mask_t<leaf_array_t<typename detail::extract_1<Args...>::type>>;
    using UInt32 = uint32_array_t<Mask>;
    static constexpr bool Enabled = is_jit_array_v<Mask>;
    static constexpr bool IsLLVM = is_llvm_array_v<Mask>;
    static constexpr bool IsCUDA = is_cuda_array_v<Mask>;

    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    Loop(Args&... args) {
        if constexpr (Enabled) {
            // Count the JIT variable IDs of all arguments in 'args'
            (extract(args, false), ...);

            // Allocate storage for these indices
            m_vars = new uint32_t*[m_var_count];
            m_vars_phi = new uint32_t[m_var_count];

            // Collect the JIT variable IDs of all arguments in 'args'
            m_var_count = 0;
            (extract(args, true), ...);

            init();
        }
    }

    ~Loop() {
        if constexpr (Enabled) {
            delete[] m_vars;
            delete[] m_vars_phi;

            if (m_counter != 2) {
                jitc_log(::LogLevel::Warn,
                         "enoki::Loop::cond() must be called exactly twice! "
                         "(please make sure that you use the loop object as "
                         "follows: `while (loop.cond(...)) { .. code .. }` )");
                jitc_set_eval_enabled(IsCUDA, m_eval_enabled);
                jitc_var_dec_ref_ext(m_id);
            }
        }
    }

    bool cond(const Mask &mask_) {
        if constexpr (!Enabled) {
            return (bool) mask_;
        } else {
            if (m_counter == 0) {
                Mask mask;
                if constexpr (IsLLVM)
                    mask = mask_ && Mask::active_mask();
                else
                    mask = mask_;

                uint32_t mask_index = 0;
                if constexpr (is_diff_array_v<Mask>)
                    mask_index = detach(mask).index();
                else
                    mask_index = mask.index();

                if constexpr (IsLLVM) {
                    /// ----------- LLVM -----------

                    // Reduce loop condition to a single bit
                    append(jitc_var_new_2(
                        0, VarType::Bool,
                        "$r0 = call i1 "
                        "@llvm.experimental.vector.reduce.or.v$wi1(<$w x i1> $r1)",
                        1, mask_index, m_id));

                    // Branch to end of loop if all done
                    append(jitc_var_new_2(
                        0, VarType::Invalid,
                        "br $t1 $r1, label %$L2_body, label %$L2_post", 1, m_id,
                        m_loop_id));

                    jitc_llvm_active_mask_push(mask_index);
                } else {
                    /// ----------- CUDA -----------
                    // Branch to end of loop if all done
                    append(jitc_var_new_3(1, VarType::Invalid,
                                          "@!$r1 bra $L2_post", 1,
                                          mask_index, m_loop_id, m_id));
                }

                // Start the main loop body
                append(jitc_var_new_2(IsCUDA, VarType::Invalid, "\n$L1_body:", 1,
                                      m_loop_id, m_id));
            } else if (m_counter == 1) {
                uint32_t mask_index = IsCUDA ? 0 : jitc_llvm_active_mask();

                if constexpr (IsLLVM) {
                    // Ensure that the final state of all loop vars. is evaluted by this point
                    for (size_t i = 0; i < m_var_count; ++i)
                        append(jitc_var_new_2(0, VarType::Invalid, "", 1, *m_vars[i], m_id));

                    append(jitc_var_new_2(0, VarType::Invalid,
                                          "br label %$L1_end\n\n$L1_end:", 1,
                                          m_loop_id, m_id));
                }

                // Assign changed variables
                for (size_t i = 0; i < m_var_count; ++i) {
                    if (IsCUDA && m_vars_phi[i] == *m_vars[i])
                        continue;

                    if constexpr (IsLLVM) {
                        append(jitc_var_new_4(
                            0, VarType::Invalid,
                            "$r3_end = select <$w x $t1> $r1, <$w x $t2> $r2, "
                            "<$w x $t3> $r3",
                            1, mask_index, *m_vars[i], m_vars_phi[i], m_id));
                    } else {
                        append(jitc_var_new_3(1, VarType::Invalid,
                                              "mov.$b2 $r2, $r1", 1,
                                              *m_vars[i], m_vars_phi[i], m_id));
                    }
                }

                jitc_var_dec_ref_ext(mask_index);

                append(jitc_var_new_2(IsCUDA, VarType::Invalid,
                                      IsLLVM ? "br label %$L1_phi\n\n$L1_post:"
                                             : "bra $L1_cond$n\n$L1_post:",
                                      1, m_loop_id, m_id));

                if constexpr (IsLLVM)
                    jitc_llvm_active_mask_pop();
            } else {
                enoki_raise("enoki::Loop::cond() was called more than twice!");
            }

            insert_copies();

            if (m_counter == 1) {
                if (jitc_side_effect_counter(IsCUDA) != m_side_effect_counter) {
                    /* There was a side effect somewhere in the loop.
                       Create a dummy variable (also a side effect) that
                       depends on the final branch statement to ensure that the
                       loop is correctly generated. */
                    uint32_t idx =
                        jitc_var_new_1(IsCUDA, VarType::Invalid, "", 1, m_id);
                    jitc_var_mark_scatter(idx, 0);
                }

                // Clean up
                jitc_var_dec_ref_ext(m_id);
                jitc_set_eval_enabled(IsCUDA, m_eval_enabled);
                m_loop_id = m_id = 0;
            }

            return m_counter++ == 0;
        }
    }

    void insert_copies() {
        for (size_t i = 0; i < m_var_count; ++i) {
            uint32_t *idp = m_vars[i];

            uint32_t id = jitc_var_new_2(IsCUDA, jitc_var_type(*idp),
                                         IsLLVM ? "$r0 = select i1 true, <$w x $t1> "
                                                  "$r1, <$w x $t1> zeroinitializer"
                                                : "mov.$b0 $r0, $r1",
                                         1, m_vars_phi[i], m_id);

            jitc_var_dec_ref_ext(*idp);
            *idp = id;
        }
    }

protected:
    Loop() = default;

    /**
       This function generates a stream of wrapper instructions that enforce
       a relative ordering of the instruction stream
     */
    void append(uint32_t id, bool decref = true) {
        jitc_var_dec_ref_ext(m_id);
        m_id = id;
    }

    void init() {
        if constexpr (Enabled) {
            if (m_var_count == 0)
                enoki_raise("enoki::Loop(): no valid loop variables found!");
            else if (m_cuda != IsCUDA || m_llvm != IsLLVM)
                enoki_raise("enoki::Loop(): expected either CUDA or LLVM array "
                            "arguments!");
            else if (m_other)
                enoki_raise("enoki::Loop(): mixture of JIT (CUDA/LLVM) and "
                            "non-JIT values specified as Loop variables!");

            for (size_t i = 0; i < m_var_count; ++i) {
                if (*m_vars[i] == 0)
                    enoki_raise("All variables provided to enoki::Loop() must be initialized!");
            }

            // Temporarily disallow any calls to jitc_eval()
            m_eval_enabled = jitc_eval_enabled(IsCUDA);
            jitc_set_eval_enabled(IsCUDA, false);

            m_side_effect_counter = jitc_side_effect_counter(IsCUDA);

            m_loop_id = m_id = jitc_var_new_0(IsCUDA, VarType::Invalid, "", 1, 1);

            if constexpr (IsLLVM) {
                // Ensure that the initial state of all loop vars. is evaluted by this point
                for (size_t i = 0; i < m_var_count; ++i)
                    append(jitc_var_new_2(0, VarType::Invalid, "", 1, *m_vars[i], m_id));

                /* Insert two dummy basic blocks, used to establish
                   a source in the following set of phi exprs. */
                append(jitc_var_new_2(0, VarType::Invalid,
                                      "br label %$L1_pre\n\n$L1_pre:",
                                      1, m_loop_id, m_id));

                // Create a basic block containing only the phi nodes
                append(jitc_var_new_2(0, VarType::Invalid,
                                      "br label %$L1_phi\n\n$L1_phi:", 1,
                                      m_loop_id, m_id));

                for (size_t i = 0; i < m_var_count; ++i) {
                    uint32_t *idp = m_vars[i];

                    uint32_t id = jitc_var_new_3(
                        0, jitc_var_type(*idp),
                        "$r0 = phi <$w x $t0> [ $r1, %$L2_pre ], "
                        "[ $r0_end, %$L2_end ]",
                        1, *idp, m_loop_id, m_id);

                    m_vars_phi[i] = id;
                    jitc_var_dec_ref_ext(*idp);
                    jitc_var_inc_ref_ext(id);
                    *idp = id;
                    append(id);
                }

                // Next, evalute the branch condition
                append(jitc_var_new_2(
                    0, VarType::Invalid,
                    "br label %$L1_cond\n\n$L1_cond:", 1,
                    m_loop_id, m_id));

            } else {
                for (size_t i = 0; i < m_var_count; ++i) {
                    uint32_t *idp = m_vars[i];

                    uint32_t id = jitc_var_new_3(1, jitc_var_type(*idp),
                                                 "mov.$b0 $r0, $r1", 1, *idp,
                                                 m_loop_id, m_id);

                    m_vars_phi[i] = id;
                    jitc_var_dec_ref_ext(*idp);
                    jitc_var_inc_ref_ext(id);
                    *idp = id;
                    append(id);
                }

                append(jitc_var_new_2(1, VarType::Invalid, "\n$L1_cond:", 1,
                                      m_loop_id, m_id));
            }

            insert_copies();
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
                        m_vars[m_var_count] = value.index_ptr();
                    m_var_count++;
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

protected:
    uint32_t **m_vars = nullptr;
    uint32_t *m_vars_phi = nullptr;
    size_t m_var_count = 0;
    int m_counter = 0;
    bool m_eval_enabled = true;
    bool m_cuda = false;
    bool m_llvm = false;
    bool m_other = false;
    uint32_t m_id = 0;
    uint32_t m_loop_id = 0;
    uint32_t m_side_effect_counter = 0;
};

NAMESPACE_END(enoki)
