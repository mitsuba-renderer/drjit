/*
    drjit/if_stmt.h -- C++ conditional statement tracing interface

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/autodiff.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

template <size_t... Is, typename Args, typename Mask, typename TrueFn,
          typename FalseFn>
auto if_stmt_impl(std::index_sequence<Is...>, Args &&args, const Mask &cond,
                  TrueFn &&true_fn, FalseFn &&false_fn, const char *name) {
    using namespace std; // for ADL lookup to drjit::get<I> or std::get<I>

    if constexpr (std::is_same_v<Mask, bool>) {
        // This is a scalar conditional statement
        DRJIT_MARK_USED(name);

        if (cond)
            return true_fn(get<Is>(args)...);
        else
            return false_fn(get<Is>(args)...);
    } else if constexpr (is_array_v<Mask> && !is_jit_v<Mask>) {
        // This is a packet-based conditional statement
        DRJIT_MARK_USED(name);

        using Result = decltype(true_fn(get<Is>(args)...));

        Result result;

        if (any(cond))
            result = true_fn(get<Is>(args)...);

        Mask cond2 = !cond;

        if (any(cond2))
            result = select(cond2, false_fn(get<Is>(args)...), result);

        return result;
    } else {
        // This is a JIT-compiled conditional statement

        using Return = std::decay_t<decltype(std::declval<TrueFn>()(
              get<Is>(args)...))>;

        struct IfStmtData {
            std::decay_t<Args> args;
            Mask cond;
            TrueFn true_fn;
            FalseFn false_fn;
            Return rv;
        };

        ad_cond_body body_cb = [](void *p, bool value,
                                  const vector<uint64_t> &args_i,
                                  vector<uint64_t> &rv_i) {
            IfStmtData *isd = (IfStmtData *) p;

            detail::update_indices(isd->args, args_i);

            if (value)
                isd->rv = isd->true_fn(get<Is>(isd->args)...);
            else
                isd->rv = isd->false_fn(get<Is>(isd->args)...);

            detail::collect_indices<true>(isd->rv, rv_i);
        };

        ad_cond_delete delete_cb = [](void *p) { delete (IfStmtData *) p; };

        unique_ptr<IfStmtData> isd(
            new IfStmtData{ std::forward<Args>(args), cond,
                            std::forward<TrueFn>(true_fn),
                            std::forward<FalseFn>(false_fn),
                            Return() });

        detail::index64_vector args_i, rv_i;
        detail::collect_indices<true>(isd->args, args_i);

        bool all_done = ad_cond(Mask::Backend, -1, name, isd.get(), cond.index(),
                                args_i, rv_i, body_cb, delete_cb, true);

        Return rv = std::move(isd->rv);

        if (!all_done)
            isd.release();

        detail::update_indices(rv, rv_i);

        return rv;
    }
}

NAMESPACE_END(detail)

template <typename Args, typename Mask, typename TrueFn, typename FalseFn>
auto if_stmt(Args &&state, const Mask &cond, TrueFn &&true_fn,
             FalseFn &&false_fn, const char *name = nullptr) {
    using ArgsD = std::decay_t<Args>;

    return detail::if_stmt_impl(
        std::make_index_sequence<std::tuple_size<ArgsD>::value>(),
        std::forward<Args>(state), cond, std::forward<TrueFn>(true_fn),
        std::forward<FalseFn>(false_fn), name);
}

NAMESPACE_END(drjit)
