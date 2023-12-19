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

        using Result = std::decay_t<decltype(std::declval<TrueFn>()(
              dr_get<Is>(args_)...))>;

        // This is a vectorized conditional statement
        struct Payload {
            std::decay_t<Args> args;
            Mask cond;
            TrueFn true_fn;
            FalseFn false_fn;
            Result result;
        };

        ad_cond_read read_cb = [](void *p, dr_vector<uint64_t> &indices) {
            detail::collect_indices<true>(((Payload *) p)->result, indices);
        };

        ad_cond_write write_cb = [](void *p, const dr_vector<uint64_t> &indices) {
            detail::update_indices(((Payload *) p)->result, indices);
        };

        ad_cond_body body_cb = [](void *p, bool value) {
            Payload *payload = (Payload *) p;
            if (value)
                payload->result = payload->true_fn(dr_get<Is>(payload->args)...);
            else
                payload->result = payload->false_fn(dr_get<Is>(payload->args)...);
        };

        ad_cond_delete delete_cb = [](void *p) { delete (Payload *) p; };

        Payload *payload =
            new Payload{ std::forward<Args>(args_), cond,
                         std::forward<TrueFn>(true_fn),
                         std::forward<FalseFn>(false_fn),
                         Result() };

        bool rv = ad_cond(Mask::Backend, -1, name, payload, cond.index(),
                          read_cb, write_cb, body_cb, delete_cb, true);

        Result result = std::move(payload->result);

        if (rv)
            delete payload;

        return result;
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
