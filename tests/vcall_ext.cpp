#include <nanobind/nanobind.h>
#define NB_INTRUSIVE_EXPORT NB_IMPORT
#include <nanobind/intrusive/counter.h>
#include <drjit/vcall.h>
#include <drjit/python.h>

namespace nb = nanobind;
namespace dr = drjit;


template <typename Float> struct Base : nb::intrusive_base {
    virtual std::pair<Float, Float> f(Float x, Float y) = 0;

    Base() { jit_registry_put(dr::backend_v<Float>, "Base", this); }

    virtual ~Base() { jit_registry_remove(this); }
};

template <typename Float> struct A : Base<Float> {
    virtual std::pair<Float, Float> f(Float x, Float y) {
        return { 2 * y, -x };
    }
};

template <typename Float> struct B : Base<Float> {
    virtual std::pair<Float, Float> f(Float x, Float y) {
        return { 3 * y, x };
    }
};

DRJIT_VCALL_TEMPLATE_BEGIN(Base)
DRJIT_VCALL_END(Base)

template <JitBackend Backend>
void bind_simple(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;
    using BaseT = Base<Float>;
    using AT = A<Float>;
    using BT = B<Float>;

    nb::class_<BaseT, nb::intrusive_base>(m, "Base")
        .def("f", &BaseT::f);

    nb::class_<AT, BaseT>(m, "A")
        .def(nb::init<>());

    nb::class_<BT, BaseT>(m, "B")
        .def(nb::init<>());

    dr::ArrayBinding b;
    using BaseArray = dr::DiffArray<Backend, BaseT *>;
    dr::bind_array<BaseArray>(b, m, "BasePtr");
}

NB_MODULE(vcall_ext, m) {
    nb::module_ cuda = m.def_submodule("cuda");

#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind_simple<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ llvm = m.def_submodule("llvm");
    bind_simple<JitBackend::CUDA>(llvm);
#endif
}
