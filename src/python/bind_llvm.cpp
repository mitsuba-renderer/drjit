#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include "random.h"
#include <enoki/llvm.h>

void export_llvm(py::module &m) {
    py::module llvm = m.def_submodule("llvm");

    using Guide = ek::LLVMArray<float>;
    ENOKI_BIND_ARRAY_BASE(llvm, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(llvm, Guide, false);

    bind_pcg32<Guide>(llvm);
}
#endif
