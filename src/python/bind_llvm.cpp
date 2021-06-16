#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>

using Guide = ek::LLVMArray<float>;
using Mask = ek::LLVMArray<bool>;

void export_llvm(py::module_ &m) {
    py::module_ llvm = m.def_submodule("llvm");

    ENOKI_BIND_ARRAY_BASE(llvm, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(llvm, Guide, false);

    bind_pcg32<Guide>(llvm);

    py::class_<ek::Loop<Guide>>(llvm, "LoopBase");

    py::class_<Loop<Guide>, ek::Loop<Guide>> loop(llvm, "Loop");
    loop.def(py::init<const char *, py::handle>(), "name"_a, "vars"_a = py::none())
        .def("put", &Loop<Guide>::put)
        .def("init", &Loop<Guide>::init)
        .def("__call__", &Loop<Guide>::operator());
}
#endif
