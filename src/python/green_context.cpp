#include "green_context.h"
#include <drjit-core/jit.h>

PyGreenContext::PyGreenContext(uint32_t sm_count) {
    ctx = jit_cuda_green_context_make(sm_count, &actual, &other_raw);
    requested = sm_count;
    if (!ctx)
        nb::raise("Failed to create CUDA green context.");

    // Note: actual may be larger than requested due to alignment/minimum requirements
    if (actual < requested)
        nb::raise("drjit.green_context(): requested %u SMs but CUDA only provided %u.",
                  requested, actual);
}

PyGreenContext::~PyGreenContext() {
    while (!tokens.empty()) {
        jit_cuda_green_context_leave(tokens.back());
        tokens.pop_back();
    }
    if (ctx) {
        jit_cuda_green_context_release(ctx);
        ctx = nullptr;
    }
}

PyGreenContext *PyGreenContext::enter() {
    if (!ctx)
        nb::raise("GreenContext.enter(): context has been released.");
    void *token = jit_cuda_green_context_enter(ctx);
    tokens.push_back(token);
    return this;
}

void PyGreenContext::exit() {
    if (tokens.empty())
        nb::raise("GreenContext.__exit__(): context is not active.");
    void *token = tokens.back();
    tokens.pop_back();
    jit_cuda_green_context_leave(token);
}

nb::capsule PyGreenContext::remaining_ctx() const {
    if (!other_raw)
        return nb::capsule(nullptr, "CUcontext");
    return nb::capsule(other_raw, "CUcontext");
}
