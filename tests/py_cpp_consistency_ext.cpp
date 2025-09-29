#include <nanobind/nanobind.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/packet.h>
#include <drjit/matrix.h>
#include <drjit/transform.h>
#include <nanobind/stl/tuple.h>

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
}
