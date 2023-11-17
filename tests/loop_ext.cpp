#include <nanobind/stl/pair.h>
#include <drjit/loop.h>
#include <drjit/struct.h>
#include <tuple>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename UInt> std::pair<UInt, UInt> simple_loop() {
    using Bool = dr::mask_t<UInt>;

    UInt i = dr::arange<UInt>(7),
         j = 0;

    std::tie(i, j) = dr::while_loop(
        std::make_tuple(i, j),

        [](const UInt &i, const UInt &) {
            return i < 5;
        },

        [](UInt &i, UInt &j) {
            i += 1;
            j = i + 4;
        }
    );

    return { i, j };
}

template <JitBackend Backend> void bind(nb::module_ &m) {
    using UInt = dr::DiffArray<Backend, uint32_t>;

    m.def("scalar_loop", &simple_loop<uint32_t>);
    m.def("simple_loop", &simple_loop<UInt>);
}

NB_MODULE(loop_ext, m) {
#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<JitBackend::CUDA>(cuda);
#endif
}
