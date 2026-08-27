#include <nanobind/nanobind.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/packet.h>
#include <drjit/matrix.h>
#include <drjit/transform.h>
#include <drjit/idiv.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;
namespace dr = drjit;

template <typename Float>
Float tile(const Float &source, uint32_t count) {
    return Float::steal(jit_var_tile(source.index(), count));
}

template <typename Float>
Float repeat(const Float &source, uint32_t count) {
    return Float::steal(jit_var_repeat(source.index(), count));
}

template <typename Matrix4, typename Matrix3, typename Quaternion, typename Array>
std::tuple<Matrix3, Quaternion, Array> transform_decompose(Matrix4 m) {
    auto [s, q, t] = dr::transform_decompose(m);
    return std::make_tuple(s, q, t);
}

template <typename Matrix4, typename Matrix3, typename Quaternion, typename Array>
Matrix4 transform_compose(Matrix3 m, Quaternion q, Array tr) {
    return dr::transform_compose<Matrix4>(m, q, tr);
}

template <typename UInt32>
std::pair<uint32_t, uint32_t> divisor_constants(uint32_t d) {
    dr::divisor<uint32_t> div(d);
    return { (uint32_t) div.multiplier, (uint32_t) div.shift };
}

template <typename UInt32>
UInt32 idiv_scalar(const UInt32 &value, uint32_t d) {
    return dr::idiv(value, dr::divisor<uint32_t>(d));
}

template <typename UInt32>
UInt32 idiv_jit(const UInt32 &value, const UInt32 &multiplier, const UInt32 &shift) {
    return dr::idiv(value, dr::divisor<UInt32>(multiplier, shift));
}

template <typename Int32>
Int32 idiv_signed(const Int32 &value, int32_t d) {
    return dr::divisor<Int32>(d)(value);
}

template <typename Matrix4, typename Array>
Matrix4 translate(Array tr) {
    return dr::translate<Matrix4>(tr);
}

template <JitBackend Backend> void bind(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;
    using Matrix4f = dr::Matrix<dr::DiffArray<Backend, float>, 4>;
    using Matrix3f = dr::Matrix<dr::DiffArray<Backend, float>, 3>;
    using Quaternion4f = dr::Quaternion<dr::DiffArray<Backend, float>>;
    using Array3f = dr::Array<dr::DiffArray<Backend, float>, 3>;

    m.def("tile", &tile<Float>);
    m.def("repeat", &repeat<Float>);
    m.def("transform_decompose", &transform_decompose<Matrix4f, Matrix3f, Quaternion4f, Array3f>);
    m.def("transform_compose", &transform_compose<Matrix4f, Matrix3f, Quaternion4f, Array3f>);
    m.def("translate", &translate<Matrix4f, Array3f>);

    using UInt32 = dr::DiffArray<Backend, uint32_t>;
    using Int32 = dr::DiffArray<Backend, int32_t>;
    m.def("divisor_constants", &divisor_constants<UInt32>);
    m.def("idiv_scalar", &idiv_scalar<UInt32>);
    m.def("idiv_jit", &idiv_jit<UInt32>);
    m.def("idiv_signed", &idiv_signed<Int32>);
}

NB_MODULE(py_cpp_consistency_ext, m) {
#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<JitBackend::CUDA>(cuda);
#endif

#if defined(DRJIT_ENABLE_METAL)
    nb::module_ metal = m.def_submodule("metal");
    bind<JitBackend::Metal>(metal);
#endif
}
