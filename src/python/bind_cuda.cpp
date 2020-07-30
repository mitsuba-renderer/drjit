#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include "random.h"
#include <enoki/cuda.h>

void export_cuda(py::module &m) {
    py::module cuda = m.def_submodule("cuda");

    using Guide = ek::CUDAArray<float>;
    ENOKI_BIND_ARRAY_BASE(cuda, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(cuda, Guide, false);

    bind_pcg32<Guide>(cuda);
}
#endif
