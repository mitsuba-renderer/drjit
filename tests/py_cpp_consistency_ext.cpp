#include <nanobind/nanobind.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/packet.h>

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

template <JitBackend Backend> void bind(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;

    m.def("tile", &tile<Float>);
    m.def("repeat", &repeat<Float>);
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
