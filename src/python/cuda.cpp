#include "python.h"
#include "random.h"

void bind_cuda(nb::module_ &m) {
    dr::bind_2<dr::CUDAArray<float>>();
    bind_pcg32<dr::CUDAArray<float>>(m);
}
