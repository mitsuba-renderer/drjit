#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include <enoki/autodiff.h>
#include <enoki/cuda.h>

void export_cuda_ad(py::module_ &m) {
    py::module_ cuda_ad = m.def_submodule("cuda").def_submodule("ad");

    using Guide = ek::DiffArray<ek::CUDAArray<float>>;
    ENOKI_BIND_ARRAY_BASE(cuda_ad, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(cuda_ad, Guide, false);

    bind_pcg32<Guide>(cuda_ad);
    cuda_ad.attr("Loop") = m.attr("cuda").attr("Loop");
}
#endif
