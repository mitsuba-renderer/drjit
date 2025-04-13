/*
    drjit/freeze.h -- C++ frozen function interface

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/autodiff.h>
#include <drjit/array_router.h>
#include <drjit/jit.h>

NAMESPACE_BEGIN(drjit)

template <bool IncRef, typename Value>
void collect_indices(const Value &value, vector<uint32_t> &indices) {
    traverse_1_fn_ro(value, (void *) &indices,
                     [](void *p, uint64_t index, const char * /*variant*/,
                        const char * /*domain*/) {
                         vector<uint64_t> &indices = *(vector<uint64_t> *) p;
                         if constexpr (IncRef)
                             index = ad_var_inc_ref(index);
                         indices.push_back(index);
                     });
}

template <bool IncRef, typename Value>
void collect_indices(const Value &value, uint32_t n_indices,
                     uint32_t *indices) {
    struct Payload {
        uint32_t n_indices;
        uint32_t *indices;
        uint32_t i = 0;
    };

    Payload p = { n_indices, indices, 0 };

    traverse_1_fn_ro(value, (void *) &p,
                     [](void *payload, uint64_t index, const char * /*variant*/,
                        const char * /*domain*/) {
                         Payload *p = (Payload *) payload;
                         if constexpr (IncRef)
                             index = ad_var_inc_ref(index);
                         p->indices[p->i++] = index;
                     });
}

template <typename Value>
void update_indices(Value &value, uint32_t n_indices, uint32_t *indices) {

    struct Payload {
        uint32_t n_indices;
        uint32_t *indices;
        uint32_t i = 0;
    };

    Payload p = { n_indices, indices, 0 };
    traverse_1_fn_rw(value, (void *) &p,
                     [](void *payload, uint64_t, const char * /*variant*/,
                        const char * /*domain*/) {
                         Payload *p = (Payload *) p;
                         if (p->i >= p->n_indices)
                             jit_fail("More indices available to assign that "
                                      "where provided");
                         return p->indices[p->i++];
                     });
    if (p->i < n_indices)
        jit_fail("Tried to assign more indices than the value could accept");
}

template <JitBackend Backend, typename Func, typename... Args>
auto custom_fn(Func func, Args &&...args) {
    bool recording = jit_flag(JitFlag::FreezingScope);

    if (!recording) {
        return func(args...);
    } else {
        using Output     = typename std::invoke_result<Func, Args...>::type;
        using InputState = std::tuple<typename std::decay<Args>::type...>;

        scoped_set_flag fscope(JitFlag::EnableObjectTraversal, true);

        make_opaque(args...);

        drjit::vector<uint32_t> input_indices;
        drjit::vector<uint32_t> output_indices;

        auto input = std::tuple(args...);

        detail::collect_indices<true>(input, input_indices);

        jit_freeze_pause(Backend);
        auto output = func(args...);
        make_opaque(output);
        jit_freeze_resume(Backend);

        detail::collect_indices<true>(output, output_indices);

        struct Payload{
            Func func;
            uint32_t n_inputs;
            uint32_t n_outputs;
            InputState input;
        };

        Payload p = { func, (uint32_t) input_indices.size(),
                      (uint32_t) output_indices.size(), std::move(input) };

        auto wrapper = [](void *payload, uint32_t *inputs, uint32_t *outputs) {
            Payload *p = (Payload *) payload;

            drjit::vector<uint32_t> input_backup;
            detail::collect_indices<false>(input, input_backup);
            detail::update_indices(p->input, p->n_inputs, inputs);

            auto output = std::apply(p->func, p->input);

            make_opaque(output);
            detail::collect_indices<false>(output, p->n_outputs, outputs);

            detail::update_indices(p->input, input_backup);
        };

        jit_freeze_custom_fn(Backend, wrapper, nullptr, &p,
                             input_indices.size(), input_indices.data(),
                             output_indices.size(), output_indices.data());

        return output;
    }
}

NAMESPACE_END(drjit)
