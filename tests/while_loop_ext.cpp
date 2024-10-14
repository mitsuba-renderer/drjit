#include <nanobind/stl/pair.h>
#include <drjit/while_loop.h>
#include <drjit-core/python.h>
#include <drjit/packet.h>
#include <drjit/random.h>
#include <drjit/traversable_base.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename UInt> drjit::tuple<UInt, UInt> simple_loop() {
    using Bool = dr::mask_t<UInt>;

    UInt i = dr::arange<UInt>(7),
         j = 0;

    drjit::tie(i, j) = dr::while_loop(
        dr::make_tuple(i, j),

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

template <typename T>
struct Sampler {
    Sampler(size_t size) : rng(size) { }

    T next() { return rng.next_float32(); }

    void traverse_1_cb_ro(void *payload,
                          dr::detail::traverse_callback_ro fn) const {
        traverse_1_fn_ro(rng, payload, fn);
    }

    void traverse_1_cb_rw(void *payload,
                          dr::detail::traverse_callback_rw fn) {
        traverse_1_fn_rw(rng, payload, fn);
    }

    dr::PCG32<dr::uint64_array_t<T>> rng;
};

template <typename UInt> UInt loop_with_rng() {
    using Bool = dr::mask_t<UInt>;
    using Sampler = Sampler<dr::float32_array_t<UInt>>;

    auto s1 = dr::make_unique<Sampler>(3);
    auto s2 = dr::make_unique<Sampler>(3);

    UInt i = dr::arange<UInt>(3);

    Sampler *s = s1.get();
    drjit::tie(i, s) = dr::while_loop(
        dr::make_tuple(i, s),

        [](const UInt &i, const Sampler *) {
            return i < 3;
        },

        [](UInt &i, Sampler *s) {
            i += 1;
            s->rng.next_float32();
        }
    );

    return UInt(s1->rng - s2->rng);
}

bool packet_loop() {
    using Float = dr::Packet<float, 16>;
    using RNG = dr::PCG32<Float>;

    RNG a, b, c;
    for (int i = 0; i < 1000; ++i)
        a.next_float32();
    b += 1000;

    return dr::all(a.next_float32() == b.next_float32()) && dr::all((b - c) == 1001);
}

template <JitBackend Backend> void bind(nb::module_ &m) {
    using UInt = dr::DiffArray<Backend, uint32_t>;

    m.def("simple_loop", &simple_loop<UInt>);
    m.def("loop_with_rng", &loop_with_rng<UInt>);
}


NB_MODULE(while_loop_ext, m) {
#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<JitBackend::CUDA>(cuda);
#endif

    m.def("scalar_loop", &simple_loop<uint32_t>);
    m.def("packet_loop", &packet_loop);
}
