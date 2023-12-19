#include <nanobind/stl/pair.h>
#include <drjit/if_stmt.h>
#include <tuple>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename UInt> UInt simple_cond() {
    using Bool = dr::mask_t<UInt>;

    UInt i = dr::arange<UInt>(10),
         j = 5;

    UInt k = dr::if_stmt(
        std::make_tuple(i, j),

        i < j,

        [](const UInt &i, const UInt &j) {
            return j - i;
        },

        [](const UInt &i, const UInt &j) {
            return i - j;
        }
    );

    return k;
}

template <JitBackend Backend> void bind(nb::module_ &m) {
    using UInt = dr::DiffArray<Backend, uint32_t>;

    m.def("scalar_cond", &simple_cond<uint32_t>);
    m.def("simple_cond", &simple_cond<UInt>);
}

NB_MODULE(if_stmt_ext, m) {
#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<JitBackend::CUDA>(cuda);
#endif
}
