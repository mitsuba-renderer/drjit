#include <nanobind/nanobind.h>
#define NB_INTRUSIVE_EXPORT NB_IMPORT
#include <nanobind/intrusive/counter.h>
#include <nanobind/stl/pair.h>
#include <drjit/vcall.h>
#include <drjit/python.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename Float> struct Base : nb::intrusive_base {
    using Mask = dr::mask_t<Float>;

    virtual std::pair<Float, Float> f(Float x, Float y) = 0;
    virtual std::pair<Float, Float> f_masked(const std::pair<Float, Float> &xy, Mask active) = 0;
    virtual Float g(Float, Mask) = 0;
    virtual void dummy() = 0;
    virtual float scalar_getter() = 0;
    virtual Float opaque_getter() = 0;
    virtual Float constant_getter() = 0;
    virtual std::pair<Float, dr::uint32_array_t<Float>> complex_getter() = 0;
    virtual dr::replace_value_t<Float, Base<Float>*> get_self() const = 0;

    Base() {
        if constexpr (dr::is_jit_v<Float>)
            jit_registry_put(dr::backend_v<Float>, "Base", this);
    }

    virtual ~Base() { jit_registry_remove(this); }
};

template <typename Float> struct A : Base<Float> {
    using Mask = dr::mask_t<Float>;

    virtual std::pair<Float, Float> f(Float x, Float y) override {
        return { 2 * y, -x };
    }

    virtual std::pair<Float, Float> f_masked(const std::pair<Float, Float> &xy, Mask active) override {
        if (active.state() != VarState::Literal || active[0] != true)
            throw std::runtime_error("f_masked(): expected the mask to be a literal");
        return f(xy.first, xy.second);
    }

    virtual Float g(Float, Mask) override {
        return value;
    }

    virtual void dummy() override { }
    virtual float scalar_getter() override { return 1.f; }
    virtual Float opaque_getter() override { return opaque; }
    virtual Float constant_getter() override { return 123; }
    virtual std::pair<Float, dr::uint32_array_t<Float>>
    complex_getter() override {
        return { opaque, 5 };
    }
    dr::replace_value_t<Float, Base<Float>*> get_self() const override { return this; }

    Float value;
    Float opaque = dr::opaque<Float>(1.f);
};


template <typename Float> struct B : Base<Float> {
    using Mask = dr::mask_t<Float>;

    virtual std::pair<Float, Float> f(Float x, Float y) override {
        return { 3 * y, x };
    }

    virtual std::pair<Float, Float> f_masked(const std::pair<Float, Float> &xy, Mask active) override {
        if (active.state() != VarState::Literal || active[0] != true)
            throw std::runtime_error("f_masked(): expected the mask to be a literal!");
        return f(xy.first, xy.second);
    }

    virtual Float g(Float x, Mask) override {
        return value*x;
    }

    virtual void dummy() override { }
    virtual float scalar_getter() override { return 2.f; }
    virtual Float opaque_getter() override { return opaque; }
    virtual Float constant_getter() override { return 123; }
    virtual std::pair<Float, dr::uint32_array_t<Float>>
    complex_getter() override {
        return { 2 * opaque, 3 };
    }
    dr::replace_value_t<Float, Base<Float>*> get_self() const override { return this; }

    Float value;
    Float opaque = dr::opaque<Float>(2.f);
};

DRJIT_VCALL_TEMPLATE_BEGIN(Base)
    DRJIT_VCALL_METHOD(f)
    DRJIT_VCALL_METHOD(f_masked)
    DRJIT_VCALL_METHOD(dummy)
    DRJIT_VCALL_METHOD(g)
    DRJIT_VCALL_GETTER(scalar_getter)
    DRJIT_VCALL_GETTER(opaque_getter)
    DRJIT_VCALL_GETTER(complex_getter)
    DRJIT_VCALL_GETTER(constant_getter)
    DRJIT_VCALL_METHOD(get_self)
DRJIT_VCALL_END(Base)

template <JitBackend Backend>
void bind_simple(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;
    using BaseT = Base<Float>;
    using AT = A<Float>;
    using BT = B<Float>;
    using Mask = dr::mask_t<Float>;

    nb::class_<BaseT, nb::intrusive_base>(m, "Base")
        .def("f", &BaseT::f)
        .def("g", &BaseT::g);

    nb::class_<AT, BaseT>(m, "A")
        .def(nb::init<>())
        .def_rw("opaque", &AT::opaque)
        .def_rw("value", &AT::value);

    nb::class_<BT, BaseT>(m, "B")
        .def(nb::init<>())
        .def_rw("opaque", &BT::opaque)
        .def_rw("value", &BT::value);

    dr::ArrayBinding b;
    using BaseArray = dr::DiffArray<Backend, BaseT *>;
    dr::bind_array_t<BaseArray>(b, m, "BasePtr")
        .def("f",
             [](BaseArray &self, Float a, Float b) { return self->f(a, b); })
        .def("f_masked",
             [](BaseArray &self, std::pair<Float, Float> ab, Mask active) {
                 return self->f_masked(ab, active);
             },
             "ab"_a, "mask"_a = true)
        .def("g",
             [](BaseArray &self, Float x, Mask m) { return self->g(x, m); },
             "x"_a, "mask"_a = true)
        .def("dummy", [](BaseArray &self) { return self->dummy(); })
        .def("scalar_getter", [](BaseArray &self, Mask m) {
                return self->scalar_getter(m);
             }, "mask"_a = true)
        .def("opaque_getter", [](BaseArray &self, Mask m) {
                return self->opaque_getter(m);
             }, "mask"_a = true)
        .def("complex_getter", [](BaseArray &self, Mask m) {
                return self->complex_getter(m);
             }, "mask"_a = true)
        .def("constant_getter", [](BaseArray &self, Mask m) {
                return self->constant_getter(m);
             }, "mask"_a = true)
        .def("get_self", [](BaseArray &self) { return self->get_self(); });
}

NB_MODULE(vcall_ext, m) {
#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind_simple<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind_simple<JitBackend::CUDA>(cuda);
#endif
}
