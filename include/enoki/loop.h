/*
    enoki/loop.h -- Infrastructure to record CUDA and LLVM loops

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

template <typename Type, typename SFINAE = int> struct Loop;

/// Scalar fallback
template <typename Type> struct Loop<Type, enable_if_t<std::is_scalar_v<Type>>> {
    void init() { }
    template <typename... Args> void put(Args &...) { }
    bool mask() const { return true; }
    bool cond(bool mask) { return mask; }

    Loop() = default;
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    template <typename... Args> Loop(Args&...) { }
};

template <typename Type> struct Loop<Type, enable_if_jit_array_t<Type>> {
    using UInt32 = uint32_array_t<Type>;
    using Mask   = mask_t<UInt32>;

    static constexpr bool IsLLVM = is_llvm_array_v<Type>;
    static constexpr bool IsCUDA = is_cuda_array_v<Type>;

    Loop() = default;
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    template <typename... Args> Loop(Args&... args) {
        put(args...);
        init();
    }

    /// Register JIT variable indices of loop variables
    template <typename Value, typename... Args>
    void put(Value &value, Args &... args) {
        if constexpr (is_array_v<Value>) {
            if constexpr (array_depth_v<Value> == 1) {
                if constexpr (is_diff_array_v<Value>) {
                    put(value.detach_());
                } else if constexpr (is_jit_array_v<Value>) {
                    if (m_initialized)
                        enoki_raise("enoki::Loop::put(): must be called "
                                    "*before* initialization!");
                    if (value.index() == 0)
                        enoki_raise("enoki::Loop::put(): a loop variable (or "
                                    "an element thereof) is unintialized!");

                    m_vars.push_back(value.index_ptr());
                    m_vars_phi.push_back(0);
                }
            } else {
                for (size_t i = 0; i < value.size(); ++i)
                    put(value.entry(i));
            }
        } else if constexpr (is_enoki_struct_v<Value>) {
            struct_support_t<Value>::apply_1(value, [&](auto &x) { put(x); });
        }
        put(args...);
    }

    void put() { }

    void init() {
        if (m_vars.size() == 0)
            return;

        jitc_log(LogLevel::Info,
                 "enoki::Loop(): starting to record loop with %u variables",
                 (uint32_t) m_vars.size());

        for (size_t i = 0; i < m_vars.size(); ++i) {
            if (*m_vars[i] == 0)
                enoki_raise("Variables provided to enoki::Loop() must "
                            "be fully initialized!");
        }

        if (m_initialized)
            enoki_raise("enoki::Loop()::init(): should only be called once!");

        uint32_t flags = jitc_flags();

        // Do nothing if symbolic loops aren't enabled
        if ((flags & (uint32_t) JitFlag::RecordLoops) == 0)
            return;

        m_initialized = true;
        m_flags = flags;

        // Temporarily disallow any calls to jitc_eval()
        jitc_set_flags(m_flags | (uint32_t) JitFlag::RecordingLoop);

        m_side_effect_counter = jitc_side_effect_counter(IsCUDA);

        m_loop_id = m_id = jitc_var_new_0(IsCUDA, VarType::Void, "", 1, 1);

        if constexpr (IsLLVM) {
            // Ensure that the initial state of all loop vars. is evaluted by this point
            for (size_t i = 0; i < m_vars.size(); ++i)
                append(jitc_var_new_2(0, VarType::Void, "", 1, *m_vars[i], m_id));

            /* Insert two dummy basic blocks, used to establish
               a source in the following set of phi exprs. */
            append(jitc_var_new_2(0, VarType::Void,
                                  "br label %$L1_pre\n\n$L1_pre:",
                                  1, m_loop_id, m_id));

            // Create a basic block containing only the phi nodes
            append(jitc_var_new_2(0, VarType::Void,
                                  "br label %$L1_phi\n\n$L1_phi:", 1,
                                  m_loop_id, m_id));

            for (size_t i = 0; i < m_vars.size(); ++i) {
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
                0, VarType::Void,
                "br label %$L1_cond\n\n$L1_cond:", 1,
                m_loop_id, m_id));

        } else {
            for (size_t i = 0; i < m_vars.size(); ++i) {
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

            append(jitc_var_new_2(1, VarType::Void, "\n$L1_cond:", 1,
                                  m_loop_id, m_id));
        }

        insert_copies();
    }

    ~Loop() {
        if (m_initialized && m_counter != 2) {
            jitc_log(::LogLevel::Warn,
                     "enoki::Loop::cond() must be called exactly twice! "
                     "(please make sure that you use the loop object as "
                     "follows: `while (loop.cond(...)) { .. code .. }` )");
            jitc_set_flags(m_flags);
            jitc_var_dec_ref_ext(m_id);
        }
    }

    const Mask &mask() const { return m_mask; }

    bool cond(const Mask &mask) {
        if ((m_flags & (uint32_t) JitFlag::RecordLoops) == 0) {
            jitc_var_schedule(detach(mask).index());
            for (size_t i = 0; i < m_vars.size(); ++i)
                jitc_var_schedule(*m_vars[i]);
            jitc_eval();
            m_mask = mask;
            return any(mask);
        }

        if (!m_initialized)
            enoki_raise("enoki::Loop()::init(): must be called before "
                        "entering the loop!");

        if (m_counter == 0) {
            jitc_log(LogLevel::Info, "enoki::Loop(): begin loop.");

            Mask active_mask;
            if constexpr (IsLLVM)
                active_mask = mask && Mask::active_mask();
            else
                active_mask = mask;

            uint32_t mask_index = detach(active_mask).index();

            if constexpr (IsLLVM) {
                /// ----------- LLVM -----------
                uint32_t intrin = jitc_var_new_intrinsic(0,
                    "declare i1 @llvm.experimental.vector.reduce.or.v$wi1(<$w x i1>)", 1);

                // Reduce loop condition to a single bit
                append(jitc_var_new_3(
                    0, VarType::Bool,
                    "$r0 = call i1 "
                    "@llvm.experimental.vector.reduce.or.v$wi1(<$w x i1> $r1)",
                    1, mask_index, m_id, intrin));

                jitc_var_dec_ref_ext(intrin);

                // Branch to end of loop if all done
                append(jitc_var_new_2(
                    0, VarType::Void,
                    "br $t1 $r1, label %$L2_body, label %$L2_post", 1, m_id,
                    m_loop_id));

                jitc_llvm_active_mask_push(mask_index);
            } else {
                /// ----------- CUDA -----------
                // Branch to end of loop if all done
                append(jitc_var_new_3(1, VarType::Void,
                                      "@!$r1 bra $L2_post", 1,
                                      mask_index, m_loop_id, m_id));
            }

            // Start the main loop body
            append(jitc_var_new_2(IsCUDA, VarType::Void, "\n$L1_body:", 1,
                                  m_loop_id, m_id));
        } else if (m_counter == 1) {
            jitc_log(LogLevel::Info, "enoki::Loop(): end loop.");

            uint32_t mask_index = IsCUDA ? 0 : jitc_llvm_active_mask();

            if constexpr (IsLLVM) {
                // Ensure that the final state of all loop vars. is evaluted by this point
                for (size_t i = 0; i < m_vars.size(); ++i)
                    append(jitc_var_new_2(0, VarType::Void, "", 1, *m_vars[i], m_id));

                append(jitc_var_new_2(0, VarType::Void,
                                      "br label %$L1_end\n\n$L1_end:", 1,
                                      m_loop_id, m_id));
            }

            // Assign changed variables
            for (size_t i = 0; i < m_vars.size(); ++i) {
                if (IsCUDA && m_vars_phi[i] == *m_vars[i])
                    continue;

                if constexpr (IsLLVM) {
                    append(jitc_var_new_4(
                        0, VarType::Void,
                        "$r3_end = select <$w x $t1> $r1, <$w x $t2> $r2, "
                        "<$w x $t3> $r3",
                        1, mask_index, *m_vars[i], m_vars_phi[i], m_id));
                } else {
                    if (*m_vars[i]) {
                        append(jitc_var_new_3(1, VarType::Void,
                                              "mov.$b2 $r2, $r1", 1,
                                              *m_vars[i], m_vars_phi[i], m_id));
                    } else {
                        jitc_log(LogLevel::Warn,
                                 "enoki::Loop(): the %u-th loop variable was "
                                 "overwritten with an uninitialized array! "
                                 "Setting to zero..", (uint32_t) i);
                        append(jitc_var_new_2(1, VarType::Void,
                                              "mov.$b1 $r1, 0", 1,
                                              m_vars_phi[i], m_id));
                    }
                }
            }

            jitc_var_dec_ref_ext(mask_index);

            append(jitc_var_new_2(IsCUDA, VarType::Void,
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
            /* If there was a side effect somewhere in the loop, mark the
               loop itself as a side effect to ensure that it will run. */
            if (jitc_side_effect_counter(IsCUDA) != m_side_effect_counter)
                jitc_var_mark_scatter(m_id, 0);
            else
                jitc_var_dec_ref_ext(m_id);

            jitc_set_flags(m_flags);
            m_loop_id = m_id = 0;
        }

        return m_counter++ == 0;
    }

protected:
    /// Copy all variables from the phi expression in m_vars_phi[i]
    void insert_copies() {
        for (size_t i = 0; i < m_vars.size(); ++i) {
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

    /**
     * \brief Generates a stream of wrapper instructions that enforce a
     * relative ordering of the instruction stream
     */
    void append(uint32_t id) {
        jitc_var_dec_ref_ext(m_id);
        m_id = id;
    }

protected:
    /// *Pointers* to JIT variables used within the loop
    detail::ek_vector<uint32_t*> m_vars;

    /// Scratch space for PHI variable indices that will be created
    detail::ek_vector<uint32_t> m_vars_phi;

    /// Have started executing this loop (& the condition?)
    bool m_initialized = false;

    /// How many times has cond() been called yet?
    int m_counter = 0;

    /// JIT flag backup
    uint32_t m_flags = 0;
    uint32_t m_id = 0;
    uint32_t m_loop_id = 0;
    uint32_t m_side_effect_counter = 0;
    Mask m_mask = true;
};

template <typename... Ts> Loop(Ts &...) -> Loop<leaf_array_t<Ts...>, int>;

NAMESPACE_END(enoki)
