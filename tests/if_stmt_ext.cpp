#include <tuple>
#include <nanobind/stl/pair.h>
#include <drjit/packet.h>
#include <drjit/if_stmt.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename UInt> UInt simple_cond() {
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

template <typename Float> Float my_abs(Float x) {
    return dr::if_stmt(
        dr::make_tuple(x),
        x < 0,
        [](Float x) { return -x; },
        [](Float x) { return  x; }
    );
}


bool packet_cond() {
    using Float = dr::Packet<float, 16>;

    Float x = dr::arange<Float>();

    x = dr::if_stmt(
        dr::make_tuple(x),
        x < 10,
        [](Float x_) { return -x_; },
        [](Float x_) { return  x_; }
    );

    return dr::all(x == Float(0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 10, 11, 12, 13, 14, 15));
}

template <JitBackend Backend> void bind(nb::module_ &m) {
    using UInt = dr::DiffArray<Backend, uint32_t>;
    using Float = dr::DiffArray<Backend, float>;

    m.def("simple_cond", &simple_cond<UInt>);
    m.def("my_abs", &my_abs<Float>);
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

    m.def("scalar_cond", &simple_cond<uint32_t>);
    m.def("packet_cond", &packet_cond);
}
