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
extern DRJIT_EXTRA_EXPORT uint32_t ad_grad(uint64_t index, bool null_ok = false);

/// Check if gradient tracking is enabled for the given variable
extern DRJIT_EXTRA_EXPORT int ad_grad_enabled(uint64_t index);

/// Check if gradient tracking is disabled (can't create new AD variables)
extern DRJIT_EXTRA_EXPORT int ad_grad_suspended();

/// Temporarily enforce gradient tracking without creating a new scope
extern DRJIT_EXTRA_EXPORT int ad_set_force_grad(int status);

/// Accumulate into the gradient associated with a given variable
extern DRJIT_EXTRA_EXPORT void ad_accum_grad(uint64_t index, uint32_t value);

/// Clear the gradient of a given variable
extern DRJIT_EXTRA_EXPORT void ad_clear_grad(uint64_t index);

/// Increase the reference count of the given AD variable
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_inc_ref_impl(uint64_t) JIT_NOEXCEPT;

/**
 * \brief Variant of 'ad_var_inc_ref' that conceptually creates a copy
 *
 * This function return a detached variable when an active AD scope disables
 * differentiation of the provided input variable.
 */
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_copy_ref_impl(uint64_t) JIT_NOEXCEPT;

/// Decrease the reference count of the given AD variable
extern DRJIT_EXTRA_EXPORT void ad_var_dec_ref_impl(uint64_t) JIT_NOEXCEPT;

/// Perform a horizontal reduction
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_reduce(JitBackend, VarType,
                                                 JIT_ENUM ReduceOp, uint64_t);

/// Dot product reduction
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_reduce_dot(uint64_t i0, uint64_t i1);

/// Compute an exclusive or inclusive prefix reduction
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_block_prefix_reduce(ReduceOp op,
                                                              uint64_t index,
                                                              uint32_t block_size,
                                                              int exclusive,
                                                              int reverse);

/// Compute the sum of adjacent blocks of size 'block_size'
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_block_reduce(ReduceOp op,
                                                       uint64_t index,
                                                       uint32_t block_size,
                                                       int symbolic);

/// Tile the input array 'count' times
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_tile(uint64_t index, uint32_t count);

/// Perform a differentiable gather operation. See jit_var_gather for signature.
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_gather(uint64_t source,
                                                 uint32_t offset, uint32_t mask,
                                                 JIT_ENUM ReduceMode mode);

/// Gather a contiguous n-dimensional vector
extern DRJIT_EXTRA_EXPORT void ad_var_gather_packet(size_t n, uint64_t source,
                                                    uint32_t offset, uint32_t mask,
                                                    uint64_t *out,
                                                    JIT_ENUM ReduceMode mode);

/// Perform a differentiable scatter operation. See jit_var_scatter for
/// signature.
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_scatter(uint64_t target,
                                                  uint64_t value,
                                                  uint32_t index, uint32_t mask,
                                                  JIT_ENUM ReduceOp reduce_op,
                                                  JIT_ENUM ReduceMode reduce_mode);

/// Gather a contiguous n-dimensional vector (n must be a power of two)
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_scatter_packet(size_t n, uint64_t target,
                                                         const uint64_t *values,
                                                         uint32_t index, uint32_t mask,
                                                         JIT_ENUM ReduceOp reduce_op,
                                                         JIT_ENUM ReduceMode reduce_mode);

/// Create a view of an existing variable that has a smaller size
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_shrink(uint64_t index, size_t size);

extern DRJIT_EXTRA_EXPORT uint64_t ad_var_cast(uint64_t, VarType);
extern DRJIT_EXTRA_EXPORT void ad_enqueue(drjit::ADMode, uint64_t);
extern DRJIT_EXTRA_EXPORT void ad_traverse(drjit::ADMode, uint32_t);

/// Label a variable (useful for debugging via graphviz etc.)
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_set_label(uint64_t index, size_t argc, ...);

/// Return the label associated with a variable
extern DRJIT_EXTRA_EXPORT const char *ad_var_label(uint64_t index);

/// Return a list of variables that are registered with the AD computation grpah
extern DRJIT_EXTRA_EXPORT const char *ad_var_whos();

/// Return GraphViz markup describing registered variables and their connectivity
extern DRJIT_EXTRA_EXPORT const char *ad_var_graphviz();

/// Indicate that the program entered a scope which modifies the AD layer's behavior
extern DRJIT_EXTRA_EXPORT void ad_scope_enter(drjit::ADScope type, size_t size,
                                              const uint64_t *indices, int symbolic);

/// Indicate that the program left a scope which modifies the AD layer's behavior
extern DRJIT_EXTRA_EXPORT void ad_scope_leave(bool process_postponed);

/// Forcefully schedule a variable for evaluation. Returns a new reference
/// with a (potentially) different index. The 'rv' output parameter specifies
/// whether the operation did anything. If so, you should eventually call
/// ``jit_eval()``.
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_schedule_force(uint64_t index, int *rv);

/// Ensure that ``index`` is fully evaluated, and return a pointer to its
/// device memory via ``ptr_out``. This process may require the creation of a
/// new variable, hence the function always returns a new reference (whose
/// index may, however be identical to the input ``index``).
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_data(uint64_t index, void **ptr_out);

/// Mark a variable that constitutes a boundary in an evaluated loop
extern DRJIT_EXTRA_EXPORT void ad_mark_loop_boundary(uint64_t index);

namespace drjit { namespace detail { class CustomOpBase; }};
/// Weave a custom operation into the AD graph
extern DRJIT_EXTRA_EXPORT bool ad_custom_op(drjit::detail::CustomOpBase *);
extern DRJIT_EXTRA_EXPORT bool ad_release_one_output(drjit::detail::CustomOpBase *);

/// Retrieve a list of implicit input/output dependencies observed in the current scope
extern DRJIT_EXTRA_EXPORT void ad_copy_implicit_deps(drjit::vector<uint32_t> &,
                                                     bool input);

/// Kahan-compensated floating point atomic scatter-addition
extern DRJIT_EXTRA_EXPORT void
ad_var_scatter_add_kahan(uint64_t *target_1, uint64_t *target_2, uint64_t value,
                         uint32_t index, uint32_t mask);

/// Check if a variable represents an implicit dependency on a non-symbolic operand
extern void ad_var_check_implicit(uint64_t index);

// Callbacks used by \ref ad_call() below. See the interface for details
typedef void (*ad_call_func)(void *payload, void *self,
                             const drjit::vector<uint64_t> &args_i,
                             drjit::vector<uint64_t> &rv_i);
typedef void (*ad_call_cleanup)(void*);

/**
 * \brief Perform a differentiable virtual function call
 *
 * The following function encapsulates the logic of performing a differentiable
 * function call to a set of callables, either by index or through the instance
 * registry. In the former case, specify \c callable_count, otherwise specify
 * the registry \c domain.
 *
 * This function is used both By Dr.Jit's Python bindings, and the C++ interface.
 *
 * \param backend
 *     The JIT backend of the operation
 *
 * \param domain
 *     The domain of the virtual function call in the instance registry.
 *     Must be \c nullptr if \c callable_count is provided instead.
 *
 * \param symbolic
 *     Set this to \c 0 for evaluated mode, \c 1 for symbolic mode, and \c -1
 *     to select the mode automatically.
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
 *     routines. The caller can use this as it sees fit.
 *
 * \param callback
 *     Callback routine, which \c ad_call will invoke a number of times to
 *     record each callable. It is given the \c payload parameter, a \c self
 *     pointer (either a pointer to an instance in the instance registry, or a
 *     callable index encoded as <tt>void*</tt>), the arguments \c args and a
 *     vector of indices \rv it might fill with return values. The \c self
 *     pointer can be set to \c nullptr, in which case the callback is still
 *     expected to fill \rv with indices to variables of the correct type.
 *
 * \param cleanup
 *     A cleanup routine that deletes storage associated with \c payload.
 *
 * \param ad
 *     Should the operation insert a \c CustomOp into the AD graph to
 *     track derivatives? This only affects symbolic mode.
 *
 * When the function returns \c true, the caller is responsible for calling
 * \c cleanup to destroy the payload. Otherwise, the AD system has taken over
 * ownership and will eventually destroy the payload.
 *
 * The function may raise an exception in the case of a failure, for example by
 * propagating an exception raised from a callable. In this case, the payload has
 * already been destroyed.
 */
extern DRJIT_EXTRA_EXPORT bool
ad_call(JitBackend backend, const char *domain, int symbolic, size_t callable_count,
        const char *name, bool is_getter, uint32_t index, uint32_t mask,
        const drjit::vector<uint64_t> &args, drjit::vector<uint64_t> &rv,
        void *payload, ad_call_func callback, ad_call_cleanup cleanup,
        bool ad);

// Callbacks used by \ref ad_loop() below. See the interface for details
typedef void (*ad_loop_read)(void *payload, drjit::vector<uint64_t> &);
typedef void (*ad_loop_write)(void *payload, const drjit::vector<uint64_t> &, bool restart);
typedef uint32_t (*ad_loop_cond)(void *payload);
typedef void (*ad_loop_body)(void *payload);
typedef void (*ad_loop_delete)(void *payload);

/**
 * \brief Execute a loop with derivative tracking
 *
 * The following function encapsulates the logic of performing a
 * differentiable function of executing a loop that modifies a set of
 * state variables.
 *
 * This function is used both By Dr.Jit's Python bindings, and the C++
 * interface. It has the following parameters:
 *
 * \param backend
 *     The JIT backend of the operation
 *
 * \param symbolic
 *     Set this to \c 0 for evaluated mode, \c 1 for symbolic mode, and \c -1
 *     to select the mode automatically.
 *
 * \param compress
 *     Set this to \c 1 for compress the state of evaluated loops at each
 *     operation, \c 0 to use a simpler masking-based implementation, and \c -1
 *     to select the mode automatically.
 *
 * \param name
 *     A descriptive name used in debug message / GraphViz visualizations
 *
 * \param payload
 *     An opaque pointer that will passed to the various callback routines.
 *     The caller can use this as it sees fit.
 *
 * \param read_cb
 *     Pointer to a callback routine that analyzes the loop state variables and
 *     adds detected JIT/AD variables to the provided index array.
 *
 * \param write_cb
 *     Pointer to a callback routine that updates the loop state variables
 *     by replacing their JIT/AD variable indices with values provided in
 *     the given array.
 *
 * \param cond_cb
 *     Callback routine that evaluates the loop condition and returns the
 *     Jit variable index of the result.
 *
 * \param body_cb
 *     Callback routine that executes one iteration of the loop body.
 *
 * \param delete_cb
 *     A cleanup routine that deletes storage associated with \c payload.
 *
 * \param ad
 *     Should the operation insert a \c CustomOp into the AD graph to
 *     track derivatives? This only affects symbolic mode.
 *
 * When the function returns \c true, the caller is responsible for calling
 * \c cleanup to destroy the payload. Otherwise, the AD system has taken over
 * ownership and will eventually destroy the payload.
 *
 * The function may raise an exception in the case of a failure, for example by
 * propagating an exception raised from a callable. In this case, the payload has
 * already been destroyed.
 */
extern DRJIT_EXTRA_EXPORT bool ad_loop(JitBackend backend, int symbolic, int compress,
                                       long long max_iterations,
                                       const char *name, void *payload,
                                       ad_loop_read read_cb, ad_loop_write write_cb,
                                       ad_loop_cond cond_cb, ad_loop_body body_cb,
                                       ad_loop_delete delete_cb, bool ad);

// Callbacks used by \ref ad_cond() below. See the interface for details
typedef void (*ad_cond_body)(void *payload, bool value,
                             const drjit::vector<uint64_t> &args_i,
                             drjit::vector<uint64_t> &rv_i);
typedef void (*ad_cond_delete)(void *payload);

/**
 * \brief Execute a conditional ("if statement") with derivative tracking
 *
 * The following function encapsulates the logic of a differentiable
 * conditional branch. It is used both By Dr.Jit's Python bindings, and the C++
 * interface. The function has the following parameters:
 *
 * \param backend
 *     The JIT backend of the operation
 *
 * \param symbolic
 *     Set this to \c 0 for evaluated mode, \c 1 for symbolic mode, and \c -1
 *     to select the mode automatically.
 *
 * \param name
 *     A descriptive name used in debug message / GraphViz visualizations
 *
 * \param payload
 *     An opaque pointer that will passed to the various callback routines.
 *     The caller can use this as it sees fit.
 *
 * \param cond
 *     The JIT variable index of the conditional expression.
 *
 * \param args
 *     Input arguments to the conditional statement
 *
 * \param rv
 *     Will be used to store the return value of the conditional statement
 *
 * \param body_cb
 *     Callback to invoke 'true_fn' or 'false_fn'
 *
 * \param delete_cb
 *     A cleanup routine that deletes storage associated with \c payload.
 *
 * \param ad
 *     Should the operation insert a \c CustomOp into the AD graph to
 *     track derivatives? This only affects symbolic mode.
 *
 * When the function returns \c true, the caller is responsible for calling
 * \c cleanup to destroy the payload. Otherwise, the AD system has taken over
 * ownership and will eventually destroy the payload.
 *
 * The function may raise an exception in the case of a failure, for example by
 * propagating an exception raised from a callable. In this case, the payload has
 * already been destroyed.
 */
extern DRJIT_EXTRA_EXPORT bool
ad_cond(JitBackend backend, int symbolic, const char *name, void *payload,
        uint32_t cond, const drjit::vector<uint64_t> &args,
        drjit::vector<uint64_t> &rv, ad_cond_body body_cb,
        ad_cond_delete delete_cb, bool ad);

/// Inform the AD layer that a state variable is temporarily being rewritten
/// by a symbolic operation
extern DRJIT_EXTRA_EXPORT void ad_var_map_put(uint64_t source, uint64_t target);

/// Query the mapping created by ad_var_map() for a given target
extern DRJIT_EXTRA_EXPORT uint64_t ad_var_map_get(uint64_t index);

/// Query/set the status of variable leak warnings
extern DRJIT_EXTRA_EXPORT int ad_leak_warnings();
extern DRJIT_EXTRA_EXPORT void ad_set_leak_warnings(int value);

#if defined(__GNUC__)
DRJIT_INLINE uint64_t ad_var_inc_ref(uint64_t index) JIT_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to ad_var_dec_ref */
    if (__builtin_constant_p(index))
        return 0;
    else
        return ad_var_inc_ref_impl(index);
}

DRJIT_INLINE uint64_t ad_var_copy_ref(uint64_t index) JIT_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to ad_var_dec_ref */
    if (__builtin_constant_p(index))
        return 0;
    else
        return ad_var_copy_ref_impl(index);
}

DRJIT_INLINE void ad_var_dec_ref(uint64_t index) JIT_NOEXCEPT {
    if (!__builtin_constant_p(index))
        ad_var_dec_ref_impl(index);
}
#else
#define ad_var_dec_ref ad_var_dec_ref_impl
#define ad_var_inc_ref ad_var_inc_ref_impl
#define ad_var_copy_ref ad_var_copy_ref_impl
#endif

// Return the AD reference count of a variable (for debugging)
extern DRJIT_EXTRA_EXPORT uint32_t ad_var_ref(uint64_t index);

#if defined(__cplusplus)
}
#endif
