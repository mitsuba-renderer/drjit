#include "python.h"
#include "random.h"

void bind_llvm_ad(nb::module_ &m) {
    dr::bind_2<dr::DiffArray<dr::LLVMArray<float>>>();
    bind_pcg32<dr::DiffArray<dr::LLVMArray<float>>>(m);
}
