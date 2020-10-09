#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include <enoki/llvm.h>

using Guide = ek::LLVMArray<float>;

void export_llvm(py::module_ &m) {
    py::module_ llvm = m.def_submodule("llvm");

    ENOKI_BIND_ARRAY_BASE(llvm, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(llvm, Guide, false);

    bind_pcg32<Guide>(llvm);

    py::class_<Loop<Guide>> loop(llvm, "Loop");
    loop.def(py::init<py::args>())
        .def("cond", &Loop<Guide>::cond);

#if defined(ENOKI_ENABLE_AUTODIFF)
    loop.def("cond", [](Loop<Guide> &g,
                        const ek::DiffArray<ek::LLVMArray<bool>> &mask) {
        return g.cond(ek::detach(mask));
    });
#endif
}
#endif
