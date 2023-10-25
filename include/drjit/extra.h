/*
    drjit/extra.h -- List of symbols exported by the drjit-extra shared library

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_MSC_VER)
#  if defined(DRJIT_EXTRA_BUILD)
#    define DRJIT_EXTRA_EXPORT    __declspec(dllexport)
#  else
#    define DRJIT_EXTRA_EXPORT    __declspec(dllimport)
#  endif
#else
#  define DRJIT_EXTRA_EXPORT __attribute__ ((visibility("default")))
#endif

struct UInt32Pair {
    uint32_t first;
    uint32_t second;
};

struct UInt64Pair {
    uint64_t first;
    uint64_t second;
};

#define DR_EXPORT(x)                                                           \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x(uint32_t);                  \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t);

#define DR_EXPORT_2(x)                                                         \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x(uint32_t, uint32_t);        \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t, uint64_t);

#define DR_EXPORT_PAIR(x)                                                      \
    extern DRJIT_EXTRA_EXPORT struct UInt32Pair jit_var_##x(uint32_t);         \
    extern DRJIT_EXTRA_EXPORT struct UInt64Pair ad_var_##x(uint64_t);

#define DR_EXPORT_AD(x)                                                        \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t);

#define DR_EXPORT_AD_2(x)                                                      \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t, uint64_t);

#define DR_EXPORT_AD_3(x)                                                      \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t, uint64_t, uint64_t);

// Unary arithmetic/transcendental operations
DR_EXPORT(exp2)
DR_EXPORT(exp)
DR_EXPORT(log2)
DR_EXPORT(log)
DR_EXPORT(sin)
DR_EXPORT(cos)
DR_EXPORT(tan)
DR_EXPORT(cot)
DR_EXPORT(asin)
DR_EXPORT(acos)
DR_EXPORT(atan)
DR_EXPORT(sinh)
DR_EXPORT(cosh)
DR_EXPORT(tanh)
DR_EXPORT(asinh)
DR_EXPORT(acosh)
DR_EXPORT(atanh)
DR_EXPORT(cbrt)
DR_EXPORT(erf)
DR_EXPORT_2(atan2)
DR_EXPORT_2(ldexp)
DR_EXPORT_PAIR(sincos)
DR_EXPORT_PAIR(sincosh)
DR_EXPORT_PAIR(frexp)

DR_EXPORT_AD(copy)
DR_EXPORT_AD(neg)
DR_EXPORT_AD(abs)
DR_EXPORT_AD(sqrt)
DR_EXPORT_AD(rcp)
DR_EXPORT_AD(rsqrt)

// Binary operations
DR_EXPORT_AD_2(add)
DR_EXPORT_AD_2(sub)
DR_EXPORT_AD_2(mul)
DR_EXPORT_AD_2(div)
DR_EXPORT_AD_2(min)
DR_EXPORT_AD_2(max)

// Ternary operations
DR_EXPORT_AD_3(fma)
DR_EXPORT_AD_3(select)

#undef DR_EXPORT
#undef DR_EXPORT_2
#undef DR_EXPORT_PAIR
#undef DR_EXPORT_AD
#undef DR_EXPORT_AD_2

/// Create a new AD-attached variable for the given JIT variable index
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_new(uint32_t index);

/// Return the gradient value associated with a particular variable
extern DRJIT_EXTRA_EXPORT uint32_t ad_grad(uint64_t index);

/// Check if gradient tracking is enabled for the given variable
extern DRJIT_EXTRA_EXPORT int ad_grad_enabled(uint64_t index);

/// Accumulate into the gradient associated with a given variable
extern DRJIT_EXTRA_EXPORT void ad_accum_grad(uint64_t index, uint32_t value);

/// Clear the gradient of a given variable
extern DRJIT_EXTRA_EXPORT void ad_clear_grad(uint64_t index);

/**
 * \brief Increase the reference count of the given AD variable
 *
 * This function is typically called when an AD variable is copied. It may
 * return a detached variable when an active AD scope disables differentiation
 * of the provided input variable.
 */
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_inc_ref_impl(uint64_t) JIT_NOEXCEPT;

/// Decrease the reference count of the given AD variable
extern DRJIT_EXTRA_EXPORT void ad_var_dec_ref_impl(uint64_t) JIT_NOEXCEPT;

/// Perform a horizontal reduction
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_reduce(JitBackend, VarType,
                                                 JIT_ENUM ReduceOp, uint64_t);

/// Compute an exclusive (exclusive == 1) or inclusive (exclusive == 0) prefix
/// sum
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_prefix_sum(uint64_t index,
                                                     int exclusive);

/// Perform a differentiable gather operation. See jit_var_gather for signature.
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_gather(uint64_t source,
                                                 uint64_t offset, uint64_t mask,
                                                 bool permute);

/// Perform a differentiable scatter operation. See jit_var_scatter for
/// signature.
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_scatter(uint64_t target,
                                                  uint64_t value,
                                                  uint32_t index, uint32_t mask,
                                                  JIT_ENUM ReduceOp reduce_op,
                                                  bool permute);

extern DRJIT_EXTRA_EXPORT uint64_t ad_var_cast(uint64_t, VarType);
extern DRJIT_EXTRA_EXPORT void ad_enqueue(drjit::ADMode, uint64_t);
extern DRJIT_EXTRA_EXPORT void ad_traverse(drjit::ADMode, uint32_t);

/// Label a variable (useful for debugging via graphviz etc.)
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_set_label(uint64_t index,
                                                    const char *label);

/// Return the label associated with a variable
extern DRJIT_EXTRA_EXPORT const char *ad_var_label(uint64_t index);

/// Return a list of variables that are registered with the AD computation grpah
extern DRJIT_EXTRA_EXPORT const char *ad_var_whos();

/// Return GraphViz markup describing registered variables and their connectivity
extern DRJIT_EXTRA_EXPORT const char *ad_var_graphviz();

/// Indicate that the program entered a scope which modifies the AD layer's behavior
extern DRJIT_EXTRA_EXPORT void ad_scope_enter(drjit::ADScope type, size_t size,
                                              const uint64_t *indices);

/// Indicate that the program left a scope which modifies the AD layer's behavior
extern DRJIT_EXTRA_EXPORT void ad_scope_leave(bool process_postponed);

namespace drjit { namespace detail { class CustomOpBase; }};

/// Weave a custom operation into the AD graph
extern DRJIT_EXTRA_EXPORT bool ad_custom_op(drjit::detail::CustomOpBase *);
extern DRJIT_EXTRA_EXPORT bool ad_release_one_output(drjit::detail::CustomOpBase *);

/// VCall implementation: get a list of observed implicit dependencies
extern DRJIT_EXTRA_EXPORT void ad_copy_implicit_deps(drjit::dr_vector<uint32_t>&);

/// Check if a variable represents an implicit dependency on a non-symbolic operand
extern void ad_var_check_implicit(uint64_t index);

typedef void (*ad_vcall_callback)(void *payload, void *self,
                                  const drjit::dr_vector<uint64_t> &args_i,
                                  drjit::dr_vector<uint64_t> &rv_i);
typedef void (*ad_vcall_cleanup)(void*);

/**
 * Perform a differentiable virtual function call
 *
 * The call can be done by index or through the instance registry. In the
 * former case, specify \c callable_count, otherwise specify the \c domain.
 *
 * \param backend
 *     The JIT backend of the operation
 *
 * \param domain
 *     The domain of the virtual function call in the instance registry.
 *     Must be \c nullptr if \c callable_count is provided instead.
 *
 * \param callable_count
 *     The number of callables. Must be zero if \c domain is provided instead.
 *
 * \param name
 *     A descriptive name used in debug message / GraphViz visualizations
 *
 * \param is_getter
 *     Is the function a getter? In this case, an indirect jump is hardly
 *     necessary. We can turn the vectorized query into a simple gather
 *     operation.
 *
 * \param index
 *     Callable index / instance ID of the call.
 *
 * \param mask
 *     Mask that potentially disables some of the calling threads.
 *
 * \param args
 *     Vector of indices to input function arguments.
 *
 * \param rv
 *     Vector of indices to return values. Should be empty initially.
 *
 * \param payload
 *     An opaque pointer that will passed to the \c callback and \c cleanup
 *     routines.
 *
 * \param callback
 *     Callback routine, which \c ad_vcall will invoke a number of times to
 *     record each callable. It is given the \c payload parameter, a \c self
 *     pointer (either a pointer to an instance in the instance registry, or a
 *     callable index encoded as <tt>void*</tt>)
 *
 * \param cleanup
 *     A cleanup routine that deletes storage associated with \c payload.
 *
 * When the function returns \c true, the caller is responsible for calling
 * \c cleanup to destroy the payload. Otherwise, the AD system has taken over
 * ownership and will eventually destroy the payload.
 *
 * The function may raise an exception in the case of a falure, for example by
 * propagating an exception raised from a callable. In this case, the payload has
 * already been destroyed.
 */
extern DRJIT_EXTRA_EXPORT bool
ad_vcall(JitBackend backend, const char *domain, size_t callable_count,
         const char *name, bool is_getter, uint32_t index, uint32_t mask,
         const drjit::dr_vector<uint64_t> &args, drjit::dr_vector<uint64_t> &rv,
         void *payload, ad_vcall_callback callback, ad_vcall_cleanup cleanup,
         bool ad);

#if defined(__cplusplus)
}
#endif

#if defined(__GNUC__)
DRJIT_INLINE uint64_t ad_var_inc_ref(uint64_t index) JIT_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to ad_var_dec_ref */
    if (__builtin_constant_p(index))
        return 0;
    else
        return ad_var_inc_ref_impl(index);
}

DRJIT_INLINE void ad_var_dec_ref(uint64_t index) JIT_NOEXCEPT {
    if (!__builtin_constant_p(index))
        ad_var_dec_ref_impl(index);
}
#else
#define ad_var_dec_ref ad_var_dec_ref_impl
#define ad_var_inc_ref ad_var_inc_ref_impl
#endif