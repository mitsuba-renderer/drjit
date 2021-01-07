#if defined(ENOKI_ENABLE_JIT)
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

    py::class_<ek::Loop<Mask>>(cuda, "LoopBase");

    py::class_<Loop<Mask>, ek::Loop<Mask>> loop(cuda, "Loop");
    loop.def(py::init<py::args>())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("cond", &Loop<Mask>::cond)
        .def("mask", &Loop<Mask>::mask);

#if defined(ENOKI_ENABLE_AUTODIFF)
    loop.def("cond", [](Loop<Mask> &g,
                        const ek::DiffArray<ek::CUDAArray<bool>> &mask) {
        return g.cond(ek::detach(mask));
    });
#endif
}
#endif
