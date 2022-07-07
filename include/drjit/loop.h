/*
    drjit/loop.h -- Infrastructure to record CUDA and LLVM loops

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <drjit-core/containers.h>
#include <drjit-core/state.h>
#include <drjit/array.h>

NAMESPACE_BEGIN(drjit)

/// Scalar fallback, expands into normal C++ loop
template <typename Mask>
struct Loop<Mask, enable_if_t<std::is_scalar_v<Mask>>> {
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    void init() { }
    template <typename... Ts> void put(Ts&...) { }
    bool operator()(bool mask) { return mask; }
    template <typename... Args> Loop(const char*, Args&...) { }
    void set_max_iterations(uint32_t) { }
    void set_eval_stride(uint32_t) { }
};

/// Array case, expands into a symbolic or wavefront-style loop
template <typename Mask>
struct Loop<Mask, enable_if_jit_array_t<Mask>> {
    static constexpr JitBackend Backend = backend_v<Mask>;
    static constexpr bool IsDiff = is_diff_v<Mask>;

    using Float32 = float32_array_t<detached_t<Mask>>;
    using Float64 = float64_array_t<detached_t<Mask>>;

    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    template <typename... Args>
    Loop(const char *name, Args &... args)
        : m_record(jit_flag(JitFlag::LoopRecord)) {

        size_t size = strlen(name) + 1;
        m_name = dr_unique_ptr<char[]>(new char[size]);
        memcpy(m_name.get(), name, size);

        /// Immediately initialize if loop state is specified
        if constexpr (sizeof...(Args) > 0) {
            put(args...);
            init();
        }
    }

    ~Loop() {
        #if !defined(NDEBUG)
            if (m_state != 0 && m_state != 3 && m_state != 4)
                jit_log(
                    ::LogLevel::Warn,
                    "Loop(\"%s\"): destructed in an inconsistent state. An "
                    "exception or disallowed scalar control flow (break, continue) "
                    "likely caused the loop to exit prematurely. Cleaning up..",
                    m_name.get());
        #endif

        jit_var_dec_ref_ext(m_loop_init);
        jit_var_dec_ref_ext(m_loop_cond);

        for (size_t i = 0; i < m_indices_prev.size(); ++i)
            jit_var_dec_ref_ext(m_indices_prev[i]);

        if constexpr (IsDiff) {
            for (size_t i = 0; i < m_indices_ad_prev.size(); ++i) {
                uint32_t index = m_indices_ad_prev[i];

                if (m_ad_float_precision == 32)
                    detail::ad_dec_ref_impl<Float32>(index);
                else if (m_ad_float_precision == 64)
                    detail::ad_dec_ref_impl<Float64>(index);
            }

            if (m_ad_scope) {
                m_ad_scope = false;
                detail::ad_scope_leave<Float64>(false);
                detail::ad_scope_leave<Float32>(false);
            }
        }
    }

    /// Register JIT variable indices of loop variables
    template <typename T, typename... Ts>
    void put(T &value, Ts &... args) {
        if constexpr (is_array_v<T>) {
            if constexpr (array_depth_v<T> == 1) {
                if constexpr (IsDiff && is_diff_v<T> &&
                              std::is_floating_point_v<scalar_t<T>>) {
                    int ad_float_precision = sizeof(scalar_t<T>) * 8;
                    if (m_ad_float_precision == 0)
                        m_ad_float_precision = ad_float_precision;
                    if (m_ad_float_precision != ad_float_precision)
                        jit_raise(
                            "Loop::put(): differentiable loop variables must "
                            "use the same floating point precision! (either "
                            "all single or all double precision)");

                    if (m_record && value.index_ad())
                        jit_raise(
                            "Loop::put(): one of the supplied loop "
                            "variables is attached to the AD graph (i.e. "
                            "grad_enabled(..) is true). However, recorded "
                            "loops cannot be differentiated in their entirety. "
                            "You have two options: either disable loop "
                            "recording via set_flag(JitFlag::LoopRecord, "
                            "false). Alternatively, you could implement the "
                            "adjoint of the loop using dr::CustomOp.");

                    put(value.detach_());
                    m_indices_ad[m_indices_ad.size() - 1] = value.index_ad_ptr();
                } else if constexpr (is_jit_v<T>) {
                    if (m_state)
                        jit_raise("Loop::put(): must be called "
                                  "*before* initialization!");
                    if (value.index() == 0)
                        jit_raise("Loop::put(): a loop variable (or "
                                  "an element of a data structure provided "
                                  "as a loop variable) is uninitialized!");
                    m_indices.push_back(value.index_ptr());
                    m_indices_ad.push_back(nullptr);
                }
            } else {
                for (size_t i = 0; i < value.size(); ++i)
                    put(value.entry(i));
            }
        } else if constexpr (is_drjit_struct_v<T>) {
            struct_support_t<T>::apply_1(value, [&](auto &x) { put(x); });
        } else if constexpr (is_detected_v<loop_put_detector, T>) {
            value->loop_put(*this);
        }
        put(args...);
    }

    template <typename T>
    using loop_put_detector = std::enable_if_t<std::is_same_v<
        bool,
        decltype(std::declval<T>()->loop_put(std::declval<Loop &>()), true)>>;

    void put() { }

    /// Configure the loop variables for recording
    void init() {
        if (m_state)
            jit_raise("Loop(\"%s\"): was already initialized!", m_name.get());

        if constexpr (IsDiff) {
            detail::ad_scope_enter<Float64>(detail::ADScope::Isolate, 0, nullptr);
            detail::ad_scope_enter<Float32>(detail::ADScope::Isolate, 0, nullptr);
            m_ad_scope = true;
        }

        if (!m_record)
            return;

        // Capture JIT state and begin recording session
        m_jit_state.new_scope();

        // Rewrite loop state variables (1)
        m_loop_init = jit_var_loop_init(m_indices.size(), m_indices.data());

        m_state = 1;
        jit_log(::LogLevel::InfoSym,
                "Loop(\"%s\"): --------- begin recording loop ---------", m_name.get());
    }

    /**
     * \brief Set a fixed iteration count
     *
     * Only applies to wavefront-style loops. When the total number of loop
     * iterations is known, call this function to avoid a costly horizontal
     * reduction after each iteration.
     */
    void set_max_iterations(uint32_t iterations) {
        if (m_state > 1)
            jit_raise("Loop(\"%s\"): set_max_iterations() can only be called "
                      "before entering the loop!", m_name.get());

        m_max_iterations = iterations;
    }

    /**
     * \brief Evaluate the loop state variables after every iteration?
     *
     * By default, Dr.Jit-JIT will evaluate all loop state of wave-front-style
     * loops after each iteration. This variable can be used to increase the
     * stride to a higher value (in the extreme case to the maximum loop
     * iteration count, in which case the entire loop will be unrolled into a
     * single kernel).
     */
    void set_eval_stride(uint32_t stride) {
        if (m_state > 1)
            jit_raise("Loop(\"%s\"): set_eval_stride() can only be called "
                      "before entering the loop!", m_name.get());

        m_eval_stride = stride;
    }

    bool operator()(const Mask &cond) {
        if (m_record)
            return cond_record(cond);
        else
            return cond_wavefront(cond);
    }

protected:
    /// State machine to record a loop as-is
    bool cond_record(const Mask &cond) {
        uint32_t rv = 0;

        switch (m_state) {
            case 0:
                jit_raise("Loop(\"%s\"): must be initialized before "
                          "first loop iteration!", m_name.get());
            break;

            case 1:
                // Rewrite loop state variables (2)
                m_loop_cond = jit_var_loop_cond(m_loop_init, cond.index(),
                                                m_indices.size(),
                                                m_indices.data());

                // Backup loop state before loop (for optimization)
                m_indices_prev = dr_vector<uint32_t>(m_indices.size(), 0);
                for (uint32_t i = 0; i < m_indices.size(); ++i) {
                    m_indices_prev[i] = *m_indices[i];
                    jit_var_inc_ref_ext(m_indices_prev[i]);
                }

                // Start recording side effects
                m_jit_state.begin_recording();

                // Mask deactivated SIMD lanes
                if constexpr (Backend == JitBackend::LLVM)
                    m_jit_state.set_mask(cond.index());

                m_state++;

                return true;

            case 2:
            case 3:
                // Rewrite loop state variables (3)
                rv = jit_var_loop(m_name.get(), m_loop_init, m_loop_cond,
                                  m_indices.size(), m_indices_prev.data(),
                                  m_indices.data(), m_jit_state.checkpoint(),
                                  m_state == 2);

                m_state++;

                /* Some loop variables don't change while running the loop.
                   This can be exploited by recording the loop a second time
                   while taking this information into account. */
                if (rv == (uint32_t) -1) {
                    jit_log(::LogLevel::InfoSym,
                            "Loop(\"%s\"): ----- recording loop body *again* ------", m_name.get());
                    return true;
                } else {
                    jit_log(::LogLevel::InfoSym,
                            "Loop(\"%s\"): --------- done recording loop ----------", m_name.get());
                    m_state = 4;

                    for (size_t i = 0; i < m_indices_prev.size(); ++i)
                        jit_var_dec_ref_ext(m_indices_prev[i]);
                    m_indices_prev.clear();

                    m_jit_state.end_recording();
                    m_jit_state.clear_scope();
                    jit_var_mark_side_effect(rv);

                    if constexpr (Backend == JitBackend::LLVM)
                        m_jit_state.clear_mask();

                    if constexpr (IsDiff) {
                        if (m_ad_scope) {
                            m_ad_scope = false;
                            detail::ad_scope_leave<Float64>(true);
                            detail::ad_scope_leave<Float32>(true);
                        }
                    }

                    return false;
                }
                break;

            default:
                jit_raise(m_state == 4 ? "Loop(): attempted to reuse loop "
                                         "object, which is not permitted."
                                       : "Loop(): invalid state!");
        }

        return false;
    }

    /// Unroll a loop using wavefronts
    bool cond_wavefront(const Mask &cond_) {
        Mask cond = cond_;

        if (m_state)
            jit_raise("Loop(): attempted to reuse loop object, which is not "
                      "permitted.");

        // If this is not the first iteration
        if (m_cond.index()) {
            // Clear mask from last iteration
            m_jit_state.clear_mask();

            // Disable lanes that have terminated previously
            cond &= m_cond;

            // Blend with loop state from last iteration based on mask
            for (uint32_t i = 0; i < m_indices.size(); ++i) {
                uint32_t i1 = *m_indices[i], i2 = m_indices_prev[i];
                *m_indices[i] = jit_var_new_op_3(JitOp::Select, m_cond.index(), i1, i2);
                jit_var_dec_ref_ext(i1);
                jit_var_dec_ref_ext(i2);
            }
            m_indices_prev.clear();

            // Likewise, but for AD variables
            if constexpr (IsDiff) {
                for (uint32_t i = 0; i < m_indices_ad.size(); ++i) {
                    if (!m_indices_ad[i])
                        continue;
                    uint32_t i1 = *m_indices_ad[i], i2 = m_indices_ad_prev[i],
                             index_new = 0;

                    if (m_ad_float_precision == 32) {
                        if (i1 > 0 || i2 > 0)
                            index_new = detail::ad_new_select<Float32>(
                                "dr_loop", jit_var_size(*m_indices[i]),
                                detach(m_cond), i1, i2);
                        *m_indices_ad[i] = index_new;
                        detail::ad_dec_ref_impl<Float32>(i1);
                        detail::ad_dec_ref_impl<Float32>(i2);
                    } else if (m_ad_float_precision == 64) {
                        if (i1 > 0 || i2 > 0)
                            index_new = detail::ad_new_select<Float64>(
                                "dr_loop", jit_var_size(*m_indices[i]),
                                detach(m_cond), i1, i2);
                        *m_indices_ad[i] = index_new;
                        detail::ad_dec_ref_impl<Float64>(i1);
                        detail::ad_dec_ref_impl<Float64>(i2);
                    }
                }
                m_indices_ad_prev.clear();
            }
        }

        // Try to compile loop iteration into a single kernel
        size_t size = cond_.size();
        for (uint32_t i = 0; i < m_indices.size(); ++i) {
            jit_var_schedule(*m_indices[i]);
            size_t vsize = jit_var_size(*m_indices[i]);
            if (vsize > size)
                size = vsize;
        }

        jit_var_schedule(cond.index());

        // Should we evaluate the loop & run another iteration?
        bool do_eval = m_iteration % m_eval_stride == 0,
             do_continue = false;

        if (do_eval)
            jit_eval();

        if (m_max_iterations != (uint32_t) -1)
            do_continue = m_iteration++ < m_max_iterations;
        else
            do_continue = jit_var_any(cond.index());

        if (do_continue) {
            for (uint32_t i = 0; i < m_indices.size(); ++i) {
                uint32_t index = *m_indices[i];
                jit_var_inc_ref_ext(index);
                m_indices_prev.push_back(index);
            }

            if constexpr (IsDiff) {
                for (uint32_t i = 0; i < m_indices_ad.size(); ++i) {
                    if (!m_indices_ad[i]) {
                        m_indices_ad_prev.push_back(0);
                        continue;
                    }
                    uint32_t index = *m_indices_ad[i];
                    if (m_ad_float_precision == 64)
                        detail::ad_inc_ref_impl<Float64>(index);
                    else if (m_ad_float_precision == 32)
                        detail::ad_inc_ref_impl<Float32>(index);
                    m_indices_ad_prev.push_back(index);
                }
            }

            // Mask scatters/gathers/vcalls in the next iteration
            m_cond = cond;
            m_cond.resize(size);
            m_jit_state.set_mask(m_cond.index());
            return true;
        } else {
            m_state = 4;

            if constexpr (IsDiff) {
                if (m_ad_scope) {
                    m_ad_scope = false;
                    detail::ad_scope_leave<Float64>(true);
                    detail::ad_scope_leave<Float32>(true);
                }
            }

            return false;
        }
    }

protected:
    /// Is the loop being recorded?
    bool m_record;

    /// RAII wrapper for JIT configuration
    detail::JitState<Backend> m_jit_state;

    /// A descriptive name
    dr_unique_ptr<char[]> m_name;

    /// Pointers to loop variable indices (JIT handles)
    dr_vector<uint32_t *> m_indices;

    /**
     * \brief Temporary index scratch space
     *
     * If m_record = true, this variable guards the contents
     * of m_indices before entering the loop body.
     *
     * In wavefront mode, it represents the loop state
     * of the previous iteration.
     */
    dr_vector<uint32_t> m_indices_prev;

    // --------------- Loop recording ---------------

    /// Variable representing the start of a symbolic loop
    uint32_t m_loop_init = 0;

    /// Variable representing the condition of a symbolic loop
    uint32_t m_loop_cond = 0;

    /// Index of the symbolic loop state machine
    uint32_t m_state = 0;

    // --------------- Wavefront mode ---------------

    /// Pointers to loop variable indices (AD handles)
    dr_vector<uint32_t *> m_indices_ad;

    /// AD variable state of the previous iteration
    dr_vector<uint32_t> m_indices_ad_prev;

    /// Precision of AD floating point variables
    int m_ad_float_precision = 0;
    bool m_ad_scope = false;

    /// In case the iteration count is known
    uint32_t m_iteration = 0;
    uint32_t m_eval_stride = 1;
    uint32_t m_max_iterations = (uint32_t) -1;

    /// Stashed mask variable from the previous iteration
    Mask m_cond;
};

NAMESPACE_END(drjit)
