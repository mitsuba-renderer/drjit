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

    py::class_<ek::Loop<Mask>>(llvm, "LoopBase");

    py::class_<Loop<Mask>, ek::Loop<Mask>> loop(llvm, "Loop");
    loop.def(py::init<const char *, py::args>())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("cond", &Loop<Mask>::cond);

#if defined(ENOKI_ENABLE_AUTODIFF)
    loop.def("cond", [](Loop<Mask> &g,
                        const ek::DiffArray<ek::LLVMArray<bool>> &mask) {
        return g.cond(ek::detach(mask));
    });
#endif
}
#endif
