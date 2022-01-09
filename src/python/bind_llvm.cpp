#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "tensor.h"
#include "texture.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>

using Guide = ek::LLVMArray<float>;
using Mask = ek::LLVMArray<bool>;

void export_llvm(py::module_ &m) {
    py::module_ llvm = m.def_submodule("llvm");

    ENOKI_BIND_ARRAY_BASE(llvm, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(llvm, Guide, false);

    bind_pcg32<Guide>(llvm);

    using Mask = ek::mask_t<Guide>;

    py::class_<ek::Loop<Mask>>(llvm, "LoopBase");

    py::class_<Loop<Mask>, ek::Loop<Mask>> loop(llvm, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "state"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_uniform", &Loop<Mask>::set_uniform)
        .def("set_max_iterations", &Loop<Mask>::set_max_iterations)
        .def("set_eval_stride", &Loop<Mask>::set_eval_stride)
        .def("__call__", &Loop<Mask>::operator());

    bind_texture_all<Guide>(llvm);

    ENOKI_BIND_TENSOR_TYPES(llvm);
}
#endif
