/*
    drjit/while_loop.h -- C++ loop tracing interface

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

template <typename StateD, size_t... Is, typename State, typename Cond,
          typename Body>
StateD while_loop_impl(std::index_sequence<Is...>, State &&state_, Cond &&cond,
                       Body &&body, const char *name) {
    using namespace std; // for ADL lookup to drjit::get<I> or std::get<I>

    using Mask = std::decay_t<decltype(cond(get<Is>(state_)...))>;

    if constexpr (std::is_same_v<Mask, bool>) {
        // This is a simple scalar loop
        DRJIT_MARK_USED(name);
        StateD state(std::forward<State>(state_));
        while (cond(get<Is>(state)...))
            body(get<Is>(state)...);

        return state;
    } else if constexpr (is_array_v<Mask> && !is_jit_v<Mask>) {
        // This is a packet-based vectorized loop
        DRJIT_MARK_USED(name);
        StateD state(std::forward<State>(state_));
        Mask active = true;

        while (true) {
            active &= cond(get<Is>(state)...);
            if (none(active))
                break;

            StateD backup(state);
            body(get<Is>(state)...);
            state = select(active, state, backup);
        }

        return state;
    } else {
        // This is a JIT-compiled vectorized loop
        struct Payload {
            StateD state;
            Cond cond;
            Body body;
            Mask active;
        };

        ad_loop_read read_cb = [](void *p, vector<uint64_t> &indices) {
            detail::collect_indices<true>(((Payload *) p)->state, indices);
        };

        ad_loop_write write_cb = [](void *p, const vector<uint64_t> &indices, bool) {
            detail::update_indices(((Payload *) p)->state, indices);
        };

        ad_loop_cond cond_cb = [](void *p) -> uint32_t {
            Payload *payload = (Payload *) p;
            payload->active = payload->cond(get<Is>(payload->state)...);
            return payload->active.index();
        };

        ad_loop_body body_cb = [](void *p) {
            Payload *payload = (Payload *) p;
            payload->body(get<Is>(payload->state)...);
        };

        ad_loop_delete delete_cb = [](void *p) { delete (Payload *) p; };

        unique_ptr<Payload> payload(
            new Payload{ std::forward<State>(state_), std::forward<Cond>(cond),
                         std::forward<Body>(body), Mask() });

        bool all_done = ad_loop(Mask::Backend, -1, -1, 0, name, payload.get(), read_cb,
                                write_cb, cond_cb, body_cb, delete_cb, true);

        StateD state = std::move(payload->state);

        if (!all_done)
            payload.release();

        return state;
    }
}

NAMESPACE_END(detail)

template <typename State, typename Cond, typename Body>
std::decay_t<State> while_loop(State &&state, Cond &&cond, Body &&body,
                               const char *name = nullptr) {
    using StateD = std::decay_t<State>;
    return detail::while_loop_impl<StateD>(
        std::make_index_sequence<std::tuple_size<StateD>::value>(),
        std::forward<State>(state), std::forward<Cond>(cond),
        std::forward<Body>(body), name);
}

NAMESPACE_END(drjit)
