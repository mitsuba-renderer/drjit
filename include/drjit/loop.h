/*
    drjit/loop.h -- C++ loop tracing interface

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/autodiff.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

template <typename StateD, size_t... Is, typename State, typename Cond,
          typename Body>
StateD while_loop_impl(std::index_sequence<Is...>, State &&state_, Cond &&cond,
                       Body &&body, const char *name) {
    using Mask = std::decay_t<decltype(cond(dr_get<Is>(state_)...))>;

    if constexpr (std::is_same_v<Mask, bool>) {
        DRJIT_MARK_USED(name);

        StateD state(std::forward<State>(state_));
        // This is a simple scalar loop
        while (cond(dr_get<Is>(state)...))
            body(dr_get<Is>(state)...);
        return state;
    } else {

        // This is a vectorized loop
        struct Payload {
            StateD state;
            Cond cond;
            Body body;
            Mask active;
        };

        ad_loop_read read_cb = [](void *p, dr_vector<uint64_t> &indices) {
            detail::collect_indices<true>(((Payload *) p)->state, indices);
        };

        ad_loop_write write_cb = [](void *p, const dr_vector<uint64_t> &indices) {
            detail::update_indices(((Payload *) p)->state, indices);
        };

        ad_loop_cond cond_cb = [](void *p) -> uint32_t {
            Payload *payload = (Payload *) p;
            payload->active = payload->cond(dr_get<Is>(payload->state)...);
            return payload->active.index();
        };

        ad_loop_body body_cb = [](void *p) {
            Payload *payload = (Payload *) p;
            payload->body(dr_get<Is>(payload->state)...);
        };

        ad_loop_delete delete_cb = [](void *p) { delete (Payload *) p; };

        Payload *payload =
            new Payload{ std::forward<State>(state_), std::forward<Cond>(cond),
                         std::forward<Body>(body), Mask() };

        bool rv = ad_loop(Mask::Backend, -1, name, payload, read_cb, write_cb,
                          cond_cb, body_cb, delete_cb, true);

        StateD state = std::move(payload->state);

        if (rv)
            delete payload;

        return state;
    }
}
NAMESPACE_END(detail)

template <typename State, typename Cond, typename Body, typename... Ts>
std::decay_t<State> while_loop(State &&state, Cond &&cond, Body &&body,
                               const char *name = nullptr) {
    using StateD = std::decay_t<State>;
    return detail::while_loop_impl<StateD>(
        std::make_index_sequence<std::tuple_size<StateD>::value>(),
        std::forward<State>(state), std::forward<Cond>(cond),
        std::forward<Body>(body), name);
}

NAMESPACE_END(drjit)
