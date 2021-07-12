#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_CUDA)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>

using Guide = ek::CUDAArray<float>;
using Mask = ek::CUDAArray<bool>;

void export_cuda(py::module_ &m) {
    py::module_ cuda = m.def_submodule("cuda");

    ENOKI_BIND_ARRAY_BASE(cuda, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(cuda, Guide, false);

    bind_pcg32<Guide>(cuda);

    py::class_<ek::Loop<Guide>>(cuda, "LoopBase");

    py::class_<Loop<Guide>, ek::Loop<Guide>> loop(cuda, "Loop");
    loop.def(py::init<const char *, py::handle>(), "name"_a, "vars"_a = py::none())
        .def("put", &Loop<Guide>::put)
        .def("init", &Loop<Guide>::init)
        .def("__call__", &Loop<Guide>::operator());
}
#endif
