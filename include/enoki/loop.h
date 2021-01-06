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

template <typename Mask, typename SFINAE = int> struct Loop;

/// Scalar fallback
template <typename Mask>
struct Loop<Mask, enable_if_t<std::is_scalar_v<Mask>>> {
    Loop() = default;
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    void init() { }
    template <typename Value> void put(Value &) { }
    bool cond(bool mask) { return mask; }
    template <typename... Args> Loop(const char *name, Args&...) { }
};

template <typename Mask>
struct Loop<Mask, enable_if_jit_array_t<Mask>> {
    static constexpr JitBackend Backend = Mask::Backend;

    Loop() = default;
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    template <typename... Args>
    Loop(const char *name, Args &... args)
        : m_name(name), m_state(0), m_se_offset((uint32_t) -1),
          m_size(0), m_record(jit_flag(JitFlag::LoopRecord)) {
        if constexpr (sizeof...(Args) > 0) {
            (put(args), ...);
            init();
        }
    }

    ~Loop() {
        if (m_record && m_se_offset != (uint32_t) -1) {
            // An error occurred while recording a loop
            jit_side_effects_rollback(Backend, m_se_offset);
            jit_set_flag(JitFlag::PostponeSideEffects, m_se_flag);
        }

        if (!m_record && m_index_out.size() > 0) {
            // An error occurred while evaluating a loop wavefront-style
            for (uint32_t i = 0; i < m_index_out.size(); ++i)
                jit_var_dec_ref_ext(m_index_out[i]);
            jit_var_mask_pop(Backend);
        }

        if (m_state != 0 && m_state != 3)
            jit_log(LogLevel::Warn, "Loop(): de-allocated in an inconsistent state. "
                          "(Loop.cond() must run exactly twice!)");
    }

    /// Register a loop variable // TODO: nested arrays, structs, etc.
    template <typename Value> void put(Value &value) {
        /// XXX complain when variables are attached
        m_index_p.push_back(value.index_ptr());
        m_index_in.push_back(value.index());
        size_t size = value.size();
        if (m_size != 0 && size != 1 && size != m_size)
            jit_raise("Loop.put(): loop variables have inconsistent sizes!");
        if (size > m_size)
            m_size = size;
    }

    /// Configure the loop variables for recording
    void init() {
        if (m_state)
            jit_raise("Loop(): was already initialized!");

        if (m_record) {
            step();
            m_se_offset = jit_side_effects_scheduled(Backend);
            m_se_flag = jit_flag(JitFlag::PostponeSideEffects);
            jit_set_flag(JitFlag::PostponeSideEffects, 1);
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
    bool cond_wavefront(const Mask &cond) {
        // Need to mask loop variables for disabled lanes
        if (m_cond.index()) {
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t i1 = *m_index_p[i], i2 = m_index_out[i];
                *m_index_p[i] = jit_var_new_op_3(JitOp::Select, m_cond.index(), i1, i2);
                jit_var_dec_ref_ext(i1);
                jit_var_dec_ref_ext(i2);
            }
            jit_var_mask_pop(Backend);
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
            // Mask scatters/gathers/vcalls in the next iteration
            m_cond = cond;
            jit_var_mask_push(Backend, cond.index());

            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t index = *m_index_p[i];
                jit_var_inc_ref_ext(index);
                m_index_out.push_back(index);
            }

            return true;
        } else {
            return false;
        }
    }

    bool cond_record(const Mask &cond) {
        switch (m_state++) {
            case 0:
                jit_raise("Loop(): must be initialized first!");

            case 1:
                m_cond = cond; // detach
                step();
                for (uint32_t i = 0; i < m_index_p.size(); ++i)
                    m_index_body.push_back(*m_index_p[i]);
                return true;

            case 2:
                for (uint32_t i = 0; i < m_index_p.size(); ++i)
                    m_index_out.push_back(*m_index_p[i]);
                jit_var_loop(m_name, m_cond.index(),
                             (uint32_t) m_index_p.size(), m_index_body.data(),
                             m_index_out.data(), m_se_offset,
                             m_index_out.data());
                for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                    uint32_t &index = *m_index_p[i];
                    jit_var_dec_ref_ext(index);
                    index = m_index_out[i];
                }
                jit_set_flag(JitFlag::PostponeSideEffects, m_se_flag);
                return false;

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

protected:
    const char *m_name;
    detail::ek_vector<uint32_t> m_index_in;
    detail::ek_vector<uint32_t> m_index_body;
    detail::ek_vector<uint32_t> m_index_out;
    detail::ek_vector<uint32_t *> m_index_p;
    Mask m_cond; // XXX detached_t<Mask>
    uint32_t m_state;
    uint32_t m_se_offset;
    int m_se_flag;
    size_t m_size;
    bool m_record;
};

NAMESPACE_END(enoki)
