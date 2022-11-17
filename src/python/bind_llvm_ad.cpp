#if defined(DRJIT_ENABLE_JIT) && defined(DRJIT_ENABLE_AUTODIFF)
#include "bind.h"
#include "random.h"
#include "loop.h"
#include "switch.h"
#include "tensor.h"
#include "texture.h"
#include <drjit/autodiff.h>
#include <drjit/jit.h>

void export_llvm_ad(py::module_ &m) {
    py::module_ llvm_ad = m.def_submodule("llvm").def_submodule("ad");

    using Guide = dr::DiffArray<dr::LLVMArray<float>>;
    DRJIT_BIND_ARRAY_BASE(llvm_ad, Guide, false);
    DRJIT_BIND_ARRAY_TYPES(llvm_ad, Guide, false);

    bind_pcg32<Guide>(llvm_ad);

    py::module_ detail = llvm_ad.def_submodule("detail");
    detail.def("ad_add_edge", [](int32_t src_index, int32_t dst_index,
                                 py::handle cb) {
        dr::detail::ad_add_edge<dr::LLVMArray<float>>(
            src_index, dst_index, cb.is_none() ? nullptr : new CustomOp(cb));
    }, "src_index"_a, "dst_index"_a, "cb"_a = py::none());

    using Mask = dr::mask_t<Guide>;

    py::class_<dr::Loop<Mask>>(llvm_ad, "LoopBase");

    py::class_<Loop<Mask>, dr::Loop<Mask>> loop(llvm_ad, "Loop");

    loop.def(py::init<const char *, py::handle>(), "name"_a, "state"_a = py::none())
        .def("put", &Loop<Mask>::put)
        .def("init", &Loop<Mask>::init)
        .def("set_max_iterations", &Loop<Mask>::set_max_iterations)
        .def("set_eval_stride", &Loop<Mask>::set_eval_stride)
        .def("__call__", &Loop<Mask>::operator());

    DRJIT_BIND_TENSOR_TYPES(llvm_ad);

    bind_texture_all<Guide>(llvm_ad);

    llvm_ad.def("switch_record_", drjit::detail::switch_record_impl<dr::uint32_array_t<Guide>>);
    llvm_ad.def("switch_reduce_", drjit::detail::switch_reduce_impl<dr::uint32_array_t<Guide>>);

    bind_ad_details(a_f32);
    bind_ad_details(a_f64);
}
#endif
