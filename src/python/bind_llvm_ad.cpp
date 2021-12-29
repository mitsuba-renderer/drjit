#if defined(ENOKI_ENABLE_JIT) && defined(ENOKI_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "tensor.h"
#include "texture.h"
#include <enoki/autodiff.h>
#include <enoki/jit.h>

void export_llvm_ad(py::module_ &m) {
    py::module_ llvm_ad = m.def_submodule("llvm").def_submodule("ad");

    using Guide = ek::DiffArray<ek::LLVMArray<float>>;
    ENOKI_BIND_ARRAY_BASE(llvm_ad, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(llvm_ad, Guide, false);

    bind_pcg32<Guide>(llvm_ad);

    py::module_ detail = llvm_ad.def_submodule("detail");
    detail.def("ad_add_edge", [](int32_t src_index, int32_t dst_index,
                                 py::handle cb) {
        ek::detail::ad_add_edge<ek::LLVMArray<float>>(
            src_index, dst_index, cb.is_none() ? nullptr : new CustomOp(cb));
    }, "src_index"_a, "dst_index"_a, "cb"_a = py::none());

    using Mask = ek::mask_t<Guide>;

    py::class_<ek::Loop<Mask>>(llvm_ad, "LoopBase");

    py::class_<Loop<Mask>, ek::Loop<Mask>> loop(llvm_ad, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "state"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_uniform", &Loop<Mask>::set_uniform)
        .def("set_max_iterations", &Loop<Mask>::set_max_iterations)
        .def("__call__", &Loop<Mask>::operator());

    ENOKI_BIND_TENSOR_TYPES(llvm_ad);

    bind_texture_all<Guide>(llvm_ad);

    bind_ad_details(a_f32);
    bind_ad_details(a_f64);
}
#endif
