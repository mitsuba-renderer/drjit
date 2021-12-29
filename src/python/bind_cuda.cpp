#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_CUDA)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "tensor.h"
#include "texture.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>

using Guide = ek::CUDAArray<float>;
using Mask = ek::CUDAArray<bool>;

void export_cuda(py::module_ &m) {
    py::module_ cuda = m.def_submodule("cuda");

    ENOKI_BIND_ARRAY_BASE(cuda, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(cuda, Guide, false);

    bind_pcg32<Guide>(cuda);

    using Mask = ek::mask_t<Guide>;

    py::class_<ek::Loop<Mask>>(cuda, "LoopBase");

    py::class_<Loop<Mask>, ek::Loop<Mask>> loop(cuda, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "state"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_uniform", &Loop<Mask>::set_uniform)
        .def("set_max_iterations", &Loop<Mask>::set_max_iterations)
        .def("__call__", &Loop<Mask>::operator());

    bind_texture_all<Guide>(cuda);

    ENOKI_BIND_TENSOR_TYPES(cuda);
}
#endif
