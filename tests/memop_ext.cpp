#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/util.h>

namespace nb = nanobind;
namespace dr = drjit;

template <JitBackend Backend> void bind(nb::module_ m) {
    using Float = dr::DiffArray<Backend, float>;
    using UInt32 = dr::uint32_array_t<Float>;
    using Array2f = dr::Array<Float, 2>;
    using Pair = std::pair<Float, Float>;

    m.def("gather", [](Float source, UInt32 index) {
        return dr::gather<Float>(source, index);
    });

    m.def("scatter", [](Float target, Float value, UInt32 index) {
        dr::scatter_reduce(ReduceOp::Add, target, value, index);
        return target;
    });

    m.def("packet_gather", [](Float source, UInt32 index) -> Pair {
        Array2f r = dr::gather<Array2f>(source, index);
        return { r.x(), r.y() };
    });

    m.def("packet_scatter", [](Float target, Float v0, Float v1, UInt32 index) {
        dr::scatter_reduce(ReduceOp::Add, target, Array2f(v0, v1), index);
        return target;
    });

    m.def("packet_scatter_ptr", [](Float v0, Float v1, UInt32 index) -> Pair {
        Float target[4] = { Float(0.f), Float(0.f), Float(0.f), Float(0.f) };
        dr::scatter(target, Array2f(v0, v1), index);
        return { target[0] + target[2], target[1] + target[3] };
    });

    m.def("packet_gather_dynamic", [](Float source, UInt32 index) -> Pair {
        Float out[2];
        dr::gather_packet_dynamic(2, source, index, out, true);
        return { out[0], out[1] };
    });

    m.def("packet_scatter_dynamic", [](Float target, Float v0, Float v1, UInt32 index) {
        Float values[2] = { v0, v1 };
        dr::scatter_reduce_packet_dynamic(ReduceOp::Add, 2, target, values, index, true);
        return target;
    });
}

NB_MODULE(memop_ext, m) {
    nb::module_::import_("drjit");

#if defined(DRJIT_ENABLE_LLVM)
    bind<JitBackend::LLVM>(m.def_submodule("llvm"));
#endif

#if defined(DRJIT_ENABLE_CUDA)
    bind<JitBackend::CUDA>(m.def_submodule("cuda"));
#endif

#if defined(DRJIT_ENABLE_METAL)
    bind<JitBackend::Metal>(m.def_submodule("metal"));
#endif
}
