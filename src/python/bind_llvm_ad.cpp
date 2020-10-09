#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include <enoki/autodiff.h>
#include <enoki/llvm.h>

void export_llvm_ad(py::module_ &m) {
    py::module_ llvm_ad = m.def_submodule("llvm").def_submodule("ad");

    using Guide = ek::DiffArray<ek::LLVMArray<float>>;
    ENOKI_BIND_ARRAY_BASE(llvm_ad, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(llvm_ad, Guide, false);

    bind_pcg32<Guide>(llvm_ad);
    llvm_ad.attr("Loop") = m.attr("llvm").attr("Loop");
}
#endif
