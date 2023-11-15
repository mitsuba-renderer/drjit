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

template <typename Mask, typename State, typename Cond, typename Body>
auto while_loop(State &state, Cond cond, Body body, const char *name = nullptr) {
    if constexpr (std::is_same_v<Mask, bool>) {
        // This is a simple scalar loop

        while (cond(state))
            body(state);
        return state;
    } else {
        // This is a vectorized loop

        struct Payload {
            State state;
            Cond cond;
            Body body;
            Mask active;
        };

        dr_unique_ptr<Payload> payload(new Payload{ state, cond, body, Mask() });

        ad_loop_read read_cb = [](void *p, dr_vector<uint64_t> &indices) {
            detail::collect_indices<true>(((Payload *) p)->state, indices);
        };

        ad_loop_write write_cb = [](void *p, const dr_vector<uint64_t> &indices) {
            detail::update_indices(((Payload *) p)->state, indices);
        };

        ad_loop_cond cond_cb = [](void *p) -> uint32_t {
            Payload *payload = (Payload *) p;
            payload->active = payload->cond(payload->state);
            return payload->active.index();
        };

        ad_loop_body body_cb = [](void *p) {
            Payload *payload = (Payload *) p;
            payload->body(payload->state);
        };

        ad_loop(Mask::Backend, -1, name, payload.get(), read_cb,
                write_cb, cond_cb, body_cb);

        return payload->state;
    }
}

NAMESPACE_END(drjit)
