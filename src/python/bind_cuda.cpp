#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include <enoki/cuda.h>

void export_cuda(py::module &m) {
    py::module cuda = m.def_submodule("cuda");

    using Guide = ek::CUDAArray<float>;
    ENOKI_BIND_ARRAY_BASE_1(cuda, Guide, false);
    ENOKI_BIND_ARRAY_BASE_2(false);
    ENOKI_BIND_ARRAY_TYPES(cuda, Guide, false);
}
#endif
