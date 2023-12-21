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
#include <drjit/struct.h> // to traverse std::tuple

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

struct ref_vec : dr_vector<uint64_t> {
    ref_vec() = default;
    ~ref_vec() {
        for (size_t i = 0; i < m_size; ++i)
            ad_var_dec_ref(m_data[i]);
    }
};

template <size_t... Is, typename Args, typename Mask, typename TrueFn,
          typename FalseFn>
auto if_stmt_impl(std::index_sequence<Is...>, Args &&args_, const Mask &cond,
                  TrueFn &&true_fn, FalseFn &&false_fn, const char *name) {
    if constexpr (std::is_same_v<Mask, bool>) {
        DRJIT_MARK_USED(name);

        std::decay_t<Args> args(std::forward<Args>(args_));

        if (cond)
            return true_fn(dr_get<Is>(args)...);
        else
            return false_fn(dr_get<Is>(args)...);
    } else {

        using Return = std::decay_t<decltype(std::declval<TrueFn>()(
              dr_get<Is>(args_)...))>;

        // This is a vectorized conditional statement
        struct IfStmtData {
            std::decay_t<Args> args;
            Mask cond;
            TrueFn true_fn;
            FalseFn false_fn;
            Return rv;
        };

        ad_cond_body body_cb = [](void *p, bool value,
                                  const dr_vector<uint64_t> &args_i,
                                  dr_vector<uint64_t> &rv_i) {
            IfStmtData *isd = (IfStmtData *) p;

            detail::update_indices(isd->args, args_i);

            if (value)
                isd->rv = isd->true_fn(dr_get<Is>(isd->args)...);
            else
                isd->rv = isd->false_fn(dr_get<Is>(isd->args)...);

            detail::collect_indices<true>(isd->rv, rv_i);
        };

        ad_cond_delete delete_cb = [](void *p) { delete (IfStmtData *) p; };

        IfStmtData *isd =
            new IfStmtData{ std::forward<Args>(args_), cond,
                            std::forward<TrueFn>(true_fn),
                            std::forward<FalseFn>(false_fn),
                            Return() };

        ref_vec args_i, rv_i;
        detail::collect_indices<true>(isd->args, args_i);

        bool all_done = ad_cond(Mask::Backend, -1, name, isd, cond.index(),
                                args_i, rv_i, body_cb, delete_cb, true);

        Return rv = std::move(isd->rv);

        detail::update_indices(rv, rv_i);

        if (all_done)
            delete isd;

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
