#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/packet.h>
#include <drjit/array_traits.h>

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

template <typename Float>
std::pair<dr::uint32_array_t<Float>, dr::bool_array_t<Float>>
scatter_cas(dr::uint32_array_t<Float> &target,
            const dr::uint32_array_t<Float> &comparison,
            const dr::uint32_array_t<Float> &value,
            const dr::uint32_array_t<Float> &index,
            const dr::mask_t<Float> &mask) {
    return scatter_cas(target, comparison, value, index, mask);
}

template <JitBackend Backend> void bind(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;

    m.def("tile", &tile<Float>);
    m.def("repeat", &repeat<Float>);
    m.def("scatter_cas", &scatter_cas<Float>);
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
