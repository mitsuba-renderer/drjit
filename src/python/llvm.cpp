#include "python.h"
#include "random.h"

void bind_llvm(nb::module_ &m) {
    dr::bind_2<dr::LLVMArray<float>>();
    bind_pcg32<dr::LLVMArray<float>>(m);
}
