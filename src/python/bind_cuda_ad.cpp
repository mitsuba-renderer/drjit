#if defined(DRJIT_ENABLE_JIT) && defined(DRJIT_ENABLE_CUDA) && defined(DRJIT_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "switch.h"
#include "tensor.h"
#include "texture.h"
#include <drjit/autodiff.h>
#include <drjit/jit.h>


void export_cuda_ad(py::module_ &m) {
    py::module_ cuda_ad = m.def_submodule("cuda").def_submodule("ad");

    using Guide = dr::DiffArray<dr::CUDAArray<float>>;
    DRJIT_BIND_ARRAY_BASE(cuda_ad, Guide, false);
    DRJIT_BIND_ARRAY_TYPES(cuda_ad, Guide, false);

    bind_pcg32<Guide>(cuda_ad);

    using Mask = dr::mask_t<Guide>;

    py::class_<dr::Loop<Mask>>(cuda_ad, "LoopBase");

    py::class_<Loop<Mask>, dr::Loop<Mask>> loop(cuda_ad, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "state"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_max_iterations", &Loop<Mask>::set_max_iterations)
        .def("set_eval_stride", &Loop<Mask>::set_eval_stride)
        .def("__call__", &Loop<Mask>::operator());

    DRJIT_BIND_TENSOR_TYPES(cuda_ad);

    bind_texture_all<Guide>(cuda_ad);

    cuda_ad.def("switch_record_", drjit::detail::switch_record_impl<dr::uint32_array_t<Guide>>);
    cuda_ad.def("switch_reduce_", drjit::detail::switch_reduce_impl<dr::uint32_array_t<Guide>>);

    bind_ad_details(a_f32);
    bind_ad_details(a_f64);
}


#endif
