#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_AUTODIFF)
#include "bind.h"
#include <enoki/autodiff.h>
#include <enoki/cuda.h>

void export_cuda_ad(py::module &m) {
    py::module cuda_ad = m.def_submodule("cuda").def_submodule("ad");

    using Guide = ek::DiffArray<ek::CUDAArray<float>>;
    ENOKI_BIND_ARRAY_BASE_1(cuda_ad, Guide, false);
    ENOKI_BIND_ARRAY_BASE_2(false);
    ENOKI_BIND_ARRAY_TYPES(cuda_ad, Guide, false);
}
#endif
