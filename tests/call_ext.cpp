#define NB_INTRUSIVE_EXPORT NB_IMPORT

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/stl/pair.h>
#include <drjit/call.h>
#include <drjit/python.h>
#include <drjit/random.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename T>
struct Sampler {
    Sampler(size_t size) : rng(size) { }

    T next() { return rng.next_float32(); }

    void traverse_1_cb_ro(void *payload, void (*fn)(void *, uint64_t)) const {
        traverse_1_fn_ro(rng, payload, fn);
    }

    void traverse_1_cb_rw(void *payload, uint64_t (*fn)(void *, uint64_t)) {
        traverse_1_fn_rw(rng, payload, fn);
    }

    dr::PCG32<dr::uint64_array_t<T>> rng;
};

template <typename Float> struct Base : nb::intrusive_base {
    using Mask = dr::mask_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;

    virtual std::pair<Float, Float> f(Float x, Float y) = 0;
    virtual std::pair<Float, Float> f_masked(const std::pair<Float, Float> &xy, Mask active) = 0;
    virtual Float g(Float, Mask) = 0;
    virtual void dummy() = 0;
    virtual float scalar_getter() = 0;
    virtual Float opaque_getter() = 0;
    virtual Float constant_getter() = 0;
    virtual std::pair<Float, dr::uint32_array_t<Float>> complex_getter() = 0;
    virtual dr::replace_value_t<Float, Base<Float>*> get_self() const = 0;
    virtual std::pair<Sampler<Float> *, Float> sample(Sampler<Float> *) = 0;
    virtual dr::Array<Float, 4> gather_packet(UInt32 i) const = 0;
    virtual void scatter_packet(UInt32, dr::Array<Float, 4>) = 0;
    virtual void scatter_add_packet(UInt32, dr::Array<Float, 4>) = 0;

    Base() {
        if constexpr (dr::is_jit_v<Float>)
            jit_registry_put(dr::backend_v<Float>, "Base", this);
    }


    virtual ~Base() { jit_registry_remove(this); }
};

template <typename Float> struct A : Base<Float> {
    using Mask = dr::mask_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;

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

    virtual std::pair<Sampler<Float> *, Float> sample(Sampler<Float> *s) override {
        return { s, s->next() };
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

    dr::Array<Float, 4> gather_packet(UInt32 i) const override {
        return dr::gather<dr::Array<Float, 4>>(value, i);
    }
    void scatter_packet(UInt32 i, dr::Array<Float, 4> arg) override {
        dr::scatter(value, arg, i);
    }
    void scatter_add_packet(UInt32 i, dr::Array<Float, 4> arg) override {
        dr::scatter_add(value, arg, i);
    }

    Float value;
    Float opaque = dr::opaque<Float>(1.f);
};

template <typename Float> struct B : Base<Float> {
    using Mask = dr::mask_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;

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

    virtual std::pair<Sampler<Float> *, Float> sample(Sampler<Float> *s) override {
        return { s, 0 };
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
    dr::Array<Float, 4> gather_packet(UInt32) const override {
        return 0;
    }
    void scatter_packet(UInt32, dr::Array<Float, 4>) override { }
    void scatter_add_packet(UInt32, dr::Array<Float, 4>) override { }


    Float value;
    Float opaque = dr::opaque<Float>(2.f);
};

DRJIT_CALL_TEMPLATE_BEGIN(Base)
    DRJIT_CALL_METHOD(f)
    DRJIT_CALL_METHOD(f_masked)
    DRJIT_CALL_METHOD(dummy)
    DRJIT_CALL_METHOD(g)
    DRJIT_CALL_METHOD(sample)
    DRJIT_CALL_METHOD(gather_packet)
    DRJIT_CALL_METHOD(scatter_packet)
    DRJIT_CALL_METHOD(scatter_add_packet)
    DRJIT_CALL_GETTER(scalar_getter)
    DRJIT_CALL_GETTER(opaque_getter)
    DRJIT_CALL_GETTER(complex_getter)
    DRJIT_CALL_GETTER(constant_getter)
    DRJIT_CALL_METHOD(get_self)
DRJIT_CALL_END(Base)


template <JitBackend Backend>
void bind(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;
    using UInt32 = dr::uint32_array_t<Float>;
    using BaseT = Base<Float>;
    using AT = A<Float>;
    using BT = B<Float>;
    using Mask = dr::mask_t<Float>;
    using Sampler = ::Sampler<Float>;

    auto sampler = nb::class_<Sampler>(m, "Sampler")
        .def(nb::init<size_t>())
        .def("next", &Sampler::next)
        .def_rw("rng", &Sampler::rng);

    bind_traverse(sampler);

    nb::class_<BaseT, nb::intrusive_base>(m, "Base")
        .def("f", &BaseT::f)
        .def("f_masked", &BaseT::f_masked)
        .def("g", &BaseT::g)
        .def("sample", &BaseT::sample);

    nb::class_<AT, BaseT>(m, "A")
        .def(nb::init<>())
        .def_rw("opaque", &AT::opaque)
        .def_rw("value", &AT::value);

    nb::class_<BT, BaseT>(m, "B")
        .def(nb::init<>())
        .def_rw("opaque", &BT::opaque)
        .def_rw("value", &BT::value);

    using BaseArray = dr::DiffArray<Backend, BaseT *>;
    m.def("dispatch_f", [](BaseArray &self, Float a, Float b) {
        return dr::dispatch(
            self, [](BaseT *inst, Float a_, Float b_) { return inst->f(a_, b_); }, a, b);
    });

    dr::ArrayBinding b;
    auto base_ptr = dr::bind_array_t<BaseArray>(b, m, "BasePtr")
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
        .def("sample", [](BaseArray &self, Sampler *sampler) {
                return self->sample(sampler);
             }, "sampler"_a)
        .def("get_self", [](BaseArray &self) { return self->get_self(); })
        .def("gather_packet", [](BaseArray &self, UInt32 i) { return self->gather_packet(i); })
        .def("scatter_packet", [](BaseArray &self, UInt32 i, dr::Array<Float, 4> arg) { self->scatter_packet(i, arg); })
        .def("scatter_add_packet", [](BaseArray &self, UInt32 i, dr::Array<Float, 4> arg) { self->scatter_add_packet(i, arg); });
}

NB_MODULE(call_ext, m) {
    nb::module_::import_("drjit");

#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<JitBackend::CUDA>(cuda);
#endif
}
