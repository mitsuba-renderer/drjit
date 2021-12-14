#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_CUDA) && defined(ENOKI_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "tensor.h"
#include "texture.h"
#include <enoki/autodiff.h>
#include <enoki/jit.h>


void export_cuda_ad(py::module_ &m) {
    py::module_ cuda_ad = m.def_submodule("cuda").def_submodule("ad");

    using Guide = ek::DiffArray<ek::CUDAArray<float>>;
    ENOKI_BIND_ARRAY_BASE(cuda_ad, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(cuda_ad, Guide, false);

    bind_pcg32<Guide>(cuda_ad);

    using Mask = ek::mask_t<Guide>;

    py::class_<ek::Loop<Mask>>(cuda_ad, "LoopBase");

    py::class_<Loop<Mask>, ek::Loop<Mask>> loop(cuda_ad, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "vars"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_coherent", &Loop<Mask>::set_coherent)
        .def("__call__", &Loop<Mask>::operator());

    ENOKI_BIND_TENSOR_TYPES(cuda_ad);

    bind_texture_all<Guide>(cuda_ad);

    bind_ad_details(a_f32);
    bind_ad_details(a_f64);
}


#endif
