#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/packet.h>
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

    m.def("packet_scatter_assign", [](Float target, Float v0, Float v1, UInt32 index) {
        dr::scatter<Float>(target, Array2f(v0, v1), index);
        return target;
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

    m.def("packet_scatter_ptr", []() {
        std::array<float, 6> target { };
        dr::scatter(target.data(), dr::Packet<float, 3>(1.f, 2.f, 3.f), int32_t(1));
        return target;
    });

    m.def("nested_packet_scatter_ptr", []() {
        using FloatP = dr::Packet<float, 4>;
        using Int32P = dr::Packet<int32_t, 4>;
        using Vector4fP = dr::Array<FloatP, 4>;

        std::array<float, 16> target { };
        Vector4fP value(
            FloatP(1.f, 2.f, 3.f, 4.f),
            FloatP(5.f, 6.f, 7.f, 8.f),
            FloatP(9.f, 10.f, 11.f, 12.f),
            FloatP(13.f, 14.f, 15.f, 16.f));

        dr::scatter(target.data(), value, Int32P(0, 1, 2, 3));
        return target;
    });

    m.def("deep_nested_packet_scatter_ptr", []() {
        using FloatP = dr::Packet<float, 4>;
        using Int32P = dr::Packet<int32_t, 4>;
        using Vector2fP = dr::Array<FloatP, 2>;
        using Matrix2fP = dr::Array<Vector2fP, 2>;

        std::array<float, 16> target { };
        Matrix2fP value(
            Vector2fP(FloatP(1.f, 5.f, 9.f, 13.f),
                      FloatP(2.f, 6.f, 10.f, 14.f)),
            Vector2fP(FloatP(3.f, 7.f, 11.f, 15.f),
                      FloatP(4.f, 8.f, 12.f, 16.f)));

        dr::scatter(target.data(), value, Int32P(0, 1, 2, 3));
        return target;
    });

    m.def("nested_dynamic_scatter_ptr", []() {
        using FloatD = dr::DynamicArray<float>;
        using UInt32D = dr::DynamicArray<uint32_t>;
        using Vector3fD = dr::Array<FloatD, 3>;

        FloatD x = dr::empty<FloatD>(2),
               y = dr::empty<FloatD>(2),
               z = dr::empty<FloatD>(2);
        x.entry(0) = 1.f; x.entry(1) = 4.f;
        y.entry(0) = 2.f; y.entry(1) = 5.f;
        z.entry(0) = 3.f; z.entry(1) = 6.f;

        UInt32D index = dr::arange<UInt32D>(2);
        std::array<float, 6> target { };
        dr::scatter(target.data(), Vector3fD(x, y, z), index);
        return target;
    });

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
