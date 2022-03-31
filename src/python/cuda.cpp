#include "python.h"
#include "random.h"

void bind_cuda(nb::module_ &m) {
    dr::bind_all_types<dr::CUDAArray<float>>();
    bind_pcg32<dr::CUDAArray<float>>(m);
}
