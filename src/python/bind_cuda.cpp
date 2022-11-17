#if defined(DRJIT_ENABLE_JIT) && defined(DRJIT_ENABLE_CUDA)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "switch.h"
#include "tensor.h"
#include "texture.h"
#include <drjit/jit.h>
#include <drjit/autodiff.h>

using Guide = dr::CUDAArray<float>;
using Mask = dr::CUDAArray<bool>;

void export_cuda(py::module_ &m) {
    py::module_ cuda = m.def_submodule("cuda");

    DRJIT_BIND_ARRAY_BASE(cuda, Guide, false);
    DRJIT_BIND_ARRAY_TYPES(cuda, Guide, false);

    bind_pcg32<Guide>(cuda);

    using Mask = dr::mask_t<Guide>;

    py::class_<dr::Loop<Mask>>(cuda, "LoopBase");

    py::class_<Loop<Mask>, dr::Loop<Mask>> loop(cuda, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "state"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_max_iterations", &Loop<Mask>::set_max_iterations)
        .def("set_eval_stride", &Loop<Mask>::set_eval_stride)
        .def("__call__", &Loop<Mask>::operator());

    bind_texture_all<Guide>(cuda);

    cuda.def("switch_record_", drjit::detail::switch_record_impl<dr::uint32_array_t<Guide>>);
    cuda.def("switch_reduce_", drjit::detail::switch_reduce_impl<dr::uint32_array_t<Guide>>);

    DRJIT_BIND_TENSOR_TYPES(cuda);
}
#endif
