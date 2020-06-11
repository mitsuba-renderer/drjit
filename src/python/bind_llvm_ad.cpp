#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include <enoki/autodiff.h>
#include <enoki/llvm.h>

void export_llvm_ad(py::module &m) {
    py::module llvm_ad = m.def_submodule("llvm").def_submodule("ad");

    using Guide = ek::DiffArray<ek::LLVMArray<float>>;
    ENOKI_BIND_ARRAY_BASE_1(llvm_ad, Guide, false);
    ENOKI_BIND_ARRAY_BASE_2(false);
    ENOKI_BIND_ARRAY_TYPES(llvm_ad, Guide, false);

    bind_pcg32<Guide>(llvm_ad);
}
#endif
