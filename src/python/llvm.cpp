#include "python.h"
#include "random.h"

void bind_llvm(nb::module_ &m) {
    dr::bind_all_types<dr::LLVMArray<float>>();
    bind_pcg32<dr::LLVMArray<float>>(m);
}
