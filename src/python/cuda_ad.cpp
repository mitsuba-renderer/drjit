#include "python.h"
#include "random.h"

void bind_cuda_ad(nb::module_ &m) {
    dr::bind_2<dr::DiffArray<dr::CUDAArray<float>>>();
    bind_pcg32<dr::DiffArray<dr::CUDAArray<float>>>(m);
}
