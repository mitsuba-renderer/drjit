/// PyGreenContext — Python wrapper for CUDA green contexts.
/// Shared between cuda.cpp and test code (queue_ext.cpp).

#pragma once

#include <nanobind/nanobind.h>
#include <vector>

namespace nb = nanobind;

/// Forward declaration of an opaque CUDA green context handle
typedef struct CUDAGreenContext CUDAGreenContext;

struct PyGreenContext {
    explicit PyGreenContext(uint32_t sm_count);
    ~PyGreenContext();

    PyGreenContext *enter();
    void exit();
    nb::capsule remaining_ctx() const;
    uint32_t sm_count() const { return actual; }
    uint32_t requested_sm_count() const { return requested; }

    CUDAGreenContext *ctx = nullptr;
    void *other_raw = nullptr;
    uint32_t actual = 0;
    uint32_t requested = 0;
    std::vector<void *> tokens;
};
