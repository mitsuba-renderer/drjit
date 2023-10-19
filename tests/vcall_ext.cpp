#include <nanobind/nanobind.h>
#include <drjit/vcall.h>

namespace nb = nanobind;
namespace dr = drjit;

template <JitBackend Backend>
void bind_simple(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;

    struct Base {
        virtual std::pair<Float, Float> f(Float x, Float y) = 0;
    };

    struct A: Base {
        virtual std::pair<Float, Float> f(Float x, Float y) {
            return { 2 * y, -x };
        }
    };

    struct B: Base {
        virtual std::pair<Float, Float> f(Float x, Float y) {
            return { 3 * y, x };
        }
    };

    nb::class_<Base>(m, "Base")
        .def("f", &Base::f);

    nb::class_<A, Base>(m, "A")
        .def(nb::init<>());

    nb::class_<B, Base>(m, "B")
        .def(nb::init<>());
}

NB_MODULE(vcall_ext, m) {
    nb::module_ llvm = m.def_submodule("llvm");
    nb::module_ cuda = m.def_submodule("cuda");

    bind_simple<JitBackend::LLVM>(llvm);
    bind_simple<JitBackend::CUDA>(cuda);
}
