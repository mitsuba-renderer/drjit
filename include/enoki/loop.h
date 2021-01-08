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
#include <enoki-jit/containers.h>

NAMESPACE_BEGIN(enoki)

template <typename Mask, typename SFINAE = int> struct Loop;

/// Scalar fallback
template <typename Mask>
struct Loop<Mask, enable_if_t<std::is_scalar_v<Mask>>> {
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    void init() { }
    template <typename Value> void put(Value &) { }
    bool cond(bool mask) { return mask; }
    template <typename... Args> Loop(const char*, Args&...) { }
};

template <typename Mask>
struct Loop<Mask, enable_if_jit_array_t<Mask>> {
    static constexpr JitBackend Backend = Mask::Backend;

    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    template <typename... Args>
    Loop(const char *name, Args &... args)
        : m_state(0), m_se_offset((uint32_t) -1), m_se_flag(0), m_size(0),
          m_record(jit_flag(JitFlag::LoopRecord)) {

        size_t size = strlen(name) + 1;
        m_name = ek_unique_ptr<char[]>(new char[size]);
        memcpy(m_name.get(), name, size);

        if constexpr (sizeof...(Args) > 0) {
            put(args...);
        }
    }

    ~Loop() {
        // Recover if an error occurred while recording a loop symbolically
        if (m_record && m_se_offset != (uint32_t) -1) {
            jit_side_effects_rollback(Backend, m_se_offset);
            jit_set_flag(JitFlag::PostponeSideEffects, m_se_flag);

            for (size_t i = 0; i < m_index_body.size(); ++i)
                jit_var_dec_ref_ext(m_index_body[i]);
        }

        // Recover if an error occurred while running a wavefront-style loop
        if (!m_record && m_index_out.size() > 0) {
            for (size_t i = 0; i < m_index_out.size(); ++i)
                jit_var_dec_ref_ext(m_index_out[i]);
        }

        if (m_state != 0 && m_state != 3 && m_state != 4)
            jit_log(LogLevel::Warn, "enoki::Loop(): destructed in an inconsistent state.");
    }

    /// Register JIT variable indices of loop variables
    template <typename Value, typename... Args>
    void put(Value &value, Args &... args) {
        if constexpr (is_array_v<Value>) {
            if constexpr (array_depth_v<Value> == 1) {
                if constexpr (is_diff_array_v<Value>) {
                    put(value.detach_());
                } else if constexpr (is_jit_array_v<Value>) {
                    if (m_state)
                        jit_raise("enoki::Loop::put(): must be called "
                                  "*before* initialization!");
                    if (value.index() == 0)
                        jit_raise("enoki::Loop::put(): a loop variable (or "
                                  "an element of a data structure provided "
                                  "as a loop variable) is unintialized!");
                    m_index_p.push_back(value.index_ptr());
                    m_index_in.push_back(value.index());
                    m_invariant.push_back(0);
                    size_t size = value.size();
                    if (m_size != 0 && size != 1 && size != m_size)
                        jit_raise("enoki::Loop::put(): loop variables have "
                                  "inconsistent sizes!");
                    if (size > m_size)
                        m_size = size;
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

    /// Configure the loop variables for recording
    void init() {
        if (m_state)
            jit_raise("Loop(): was already initialized!");

        if (m_record) {
            /* Wrap loop variables using placeholders that represent
               their state just before the loop condition is evaluated */
            m_se_flag = jit_flag(JitFlag::PostponeSideEffects);
            jit_set_flag(JitFlag::PostponeSideEffects, 1);
            m_se_offset = jit_side_effects_scheduled(Backend);
            step();
            m_state = 1;
        }
    }

    bool cond(const Mask &cond) {
        if (m_record)
            return cond_record(cond);
        else
            return cond_wavefront(cond);
    }

protected:
    struct MaskStackHelper {
    public:
        void push(uint32_t index) {
            if (m_armed)
                jit_fail("MaskStackHelper::internal error! (1)");
            jit_var_mask_push(Mask::Backend, index);
            m_armed = true;
        }
        void pop() {
            if (!m_armed)
                jit_fail("MaskStackHelper::internal error! (2)");
            jit_var_mask_pop(Mask::Backend);
            m_armed = false;
        }
        ~MaskStackHelper() {
            if (m_armed)
                pop();
        }
    private:
        bool m_armed = false;
    };

    bool cond_record(const Mask &cond) {
        uint32_t n = (uint32_t) m_index_p.size();
        bool has_invariant;

        switch (m_state) {
            case 0:
                jit_raise("Loop(): must be initialized first!");

            case 1:
                /* The loop condition has been evaluated now.  Wrap loop
                   variables using placeholders once more. They will represent
                   their state at the start of the loop body. */
                m_cond = detach(cond);
                step();
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t index = *m_index_p[i];
                    m_index_body.push_back(index);
                    jit_var_inc_ref_ext(index);
                }
                m_state++;
                if constexpr (Backend == JitBackend::LLVM)
                    m_mask_stack.push(cond.index());
                return true;

            case 2:
            case 3:
                if constexpr (Backend == JitBackend::LLVM)
                    m_mask_stack.pop();
                for (uint32_t i = 0; i < n; ++i)
                    m_index_out.push_back(*m_index_p[i]);

                jit_var_loop(m_name.get(), m_cond.index(),
                             (uint32_t) n, m_index_body.data(),
                             m_index_out.data(), m_se_offset,
                             m_index_out.data(), m_state == 2,
                             m_invariant.data());

                has_invariant = false;
                for (uint32_t i = 0; i < n; ++i)
                    has_invariant |= m_invariant[i];

                if (has_invariant && m_state == 2) {
                    /* Some loop variables don't change while running the loop.
                       This can be exploited by recording the loop a second time
                       while taking this information into account. */
                    jit_side_effects_rollback(Backend, m_se_offset);
                    m_index_out.clear();

                    for (uint32_t i = 0; i < n; ++i) {
                        // Free outputs produced by current iteration
                        uint32_t &index = *m_index_p[i];
                        jit_var_dec_ref_ext(index);

                        if (m_invariant[i]) {
                            uint32_t input = m_index_in[i],
                                    &cur = m_index_body[i];
                            jit_var_inc_ref_ext(input);
                            jit_var_dec_ref_ext(cur);
                            m_index_body[i] = input;
                        }

                        index = m_index_body[i];
                        jit_var_inc_ref_ext(index);
                    }

                    m_state++;
                    if constexpr (Backend == JitBackend::LLVM)
                        m_mask_stack.push(cond.index());
                    return true;
                } else {
                    // No optimization opportunities, stop now.
                    for (uint32_t i = 0; i < n; ++i)
                        jit_var_dec_ref_ext(m_index_body[i]);
                    m_index_body.clear();

                    for (uint32_t i = 0; i < n; ++i) {
                        uint32_t &index = *m_index_p[i];
                        jit_var_dec_ref_ext(index);
                        index = m_index_out[i]; // steal ref
                    }

                    m_index_out.clear();
                    jit_set_flag(JitFlag::PostponeSideEffects, m_se_flag);
                    m_se_offset = (uint32_t) -1;
                    m_cond = Mask();
                    m_state++;
                    return false;
                }

            default:
                jit_raise("Loop(): invalid state!");
        }

        return false;
    }

    // Insert an indirection via placeholder variables
    void step() {
        for (size_t i = 0; i < m_index_p.size(); ++i) {
            uint32_t &index = *m_index_p[i],
                     next = jit_var_new_placeholder(index, 0);
            jit_var_dec_ref_ext(index);
            index = next;
        }
    }

    bool cond_wavefront(const Mask &cond) {
        // Need to mask loop variables for disabled lanes
        if (m_cond.index()) {
            m_mask_stack.pop();
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t i1 = *m_index_p[i], i2 = m_index_out[i];
                *m_index_p[i] = jit_var_new_op_3(JitOp::Select, m_cond.index(), i1, i2);
                jit_var_dec_ref_ext(i1);
                jit_var_dec_ref_ext(i2);
            }
            m_index_out.clear();
            m_cond = Mask();
        }

        // Ensure all loop state is evaluated
        jit_var_schedule(cond.index());
        for (uint32_t i = 0; i < m_index_p.size(); ++i)
            jit_var_schedule(*m_index_p[i]);
        jit_eval();

        // Do we run another iteration?
        if (jit_var_any(cond.index())) {
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t index = *m_index_p[i];
                jit_var_inc_ref_ext(index);
                m_index_out.push_back(index);
            }

            // Mask scatters/gathers/vcalls in the next iteration
            m_cond = cond;
            m_mask_stack.push(cond.index());
            return true;
        } else {
            return false;
        }
    }

protected:
    /// A descriptive name
    ek_unique_ptr<char[]> m_name;

    /// Pointers to loop variable indices
    ek_vector<uint32_t *> m_index_p;

    /// Loop variable indices before entering the loop
    ek_vector<uint32_t> m_index_in;

    /// Loop variable indices at the top of the loop body
    ek_vector<uint32_t> m_index_body;

    /// Loop variable indices after the end of the loop
    ek_vector<uint32_t> m_index_out;

    /// Detects loop-invariant variables to trigger optimizations
    ek_vector<uint8_t> m_invariant;

    /// Stashed mask variable from the previous iteration
    detached_t<Mask> m_cond;

    /// RAII wrapper for the mask stack
    MaskStackHelper m_mask_stack;

    /// Index of the symbolic loop state machine
    uint32_t m_state;

    /// Offset in the side effects queue before the beginning of the loop
    uint32_t m_se_offset;

    /// State of the PostponeSideEffects flag
    int m_se_flag;

    /// Keeps track of the size of loop variables to catch issues
    size_t m_size;

    /// Is the loop being recorded symbolically
    bool m_record;
};

NAMESPACE_END(enoki)
