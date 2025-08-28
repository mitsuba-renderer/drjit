#define NB_INTRUSIVE_EXPORT NB_IMPORT

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/stl/pair.h>
#include <drjit/call.h>
#include <drjit/python.h>
#include <drjit/random.h>
#include <drjit/traversable_base.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename T>
struct Sampler : dr::TraversableBase {
    Sampler() : rng(1) {}
    Sampler(size_t size) : rng(size) { }

    T next() { return rng.next_float32(); }

    dr::PCG32<dr::uint64_array_t<T>> rng;

    DR_TRAVERSE_CB(dr::TraversableBase, rng);
};

template <typename Float> struct Base : drjit::TraversableBase {
    using Mask = dr::mask_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;

    virtual std::pair<Float, Float> f(Float x, Float y) = 0;
    virtual std::pair<Float, Float> f_masked(const std::pair<Float, Float> &xy, Mask active) = 0;
    virtual Float g(Float, Mask) = 0;
    virtual Float h(Float) = 0;
    virtual Float nested(Float x, UInt32 s) = 0;
    /// Nested vcall, using a member variable as a pointer.
    virtual Float nested_self(Float x) = 0;
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
        if constexpr (dr::is_jit_v<Float>){
            drjit::registry_put(Variant, Domain, this);
        }
    }

    virtual ~Base() override { jit_registry_remove(this); }

    static constexpr const char *Variant =
        Float::Backend == JitBackend::CUDA ? "cuda" : "llvm";
    static constexpr const char *Domain = "Base";

    DR_TRAVERSE_CB(drjit::TraversableBase)
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

    virtual Float h(Float x) override{
        return value + x;
    }

    virtual Float nested(Float x, UInt32 /*s*/) override {
        return x + dr::gather<Float>(value, UInt32(0));
    }

    virtual Float nested_self(Float x) override {
        return x + dr::gather<Float>(value, UInt32(0));
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

    /// Additional interface that will be exposed for calls to `A`
    uint32_t a_get_property() const { return scalar_property; }
    Float a_gather_extra_value(UInt32 idx, Mask active) const {
        return dr::gather<Float>(extra_value, idx, active);
    }


    uint32_t scalar_property;
    Float value, extra_value;
    Float opaque = dr::opaque<Float>(1.f);

    DR_TRAVERSE_CB(Base<Float>, value, opaque)
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

    virtual Float h(Float x) override{
        return value - x;
    }

    virtual Float nested(Float x, UInt32 s_param) override {
        using BaseArray = dr::replace_value_t<Float, Base<Float>*>;
        BaseArray self = dr::reinterpret_array<BaseArray>(s_param);
        return self->nested(x, s_param);
    }

    virtual Float nested_self(Float x) override {
        using BaseArray = dr::replace_value_t<Float, Base<Float>*>;
        BaseArray self = dr::reinterpret_array<BaseArray>(this->s);
        return self->nested(x, this->s);
    }

    virtual std::pair<Sampler<Float> *, Float> sample(Sampler<Float> *sampler) override {
        return { sampler, 0 };
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
    UInt32 s;

    DR_TRAVERSE_CB(Base<Float>, value, opaque)
};

template <typename Float> constexpr const char *get_variant() {
    return Float::Backend == JitBackend::CUDA ? "cuda" : "llvm";
}

DRJIT_CALL_TEMPLATE_BEGIN(Base)
    DRJIT_CALL_METHOD(f)
    DRJIT_CALL_METHOD(f_masked)
    DRJIT_CALL_METHOD(dummy)
    DRJIT_CALL_METHOD(g)
    DRJIT_CALL_METHOD(h)
    DRJIT_CALL_METHOD(nested)
    DRJIT_CALL_METHOD(nested_self)
    DRJIT_CALL_METHOD(sample)
    DRJIT_CALL_METHOD(gather_packet)
    DRJIT_CALL_METHOD(scatter_packet)
    DRJIT_CALL_METHOD(scatter_add_packet)
    DRJIT_CALL_GETTER(scalar_getter)
    DRJIT_CALL_GETTER(opaque_getter)
    DRJIT_CALL_GETTER(complex_getter)
    DRJIT_CALL_GETTER(constant_getter)
    DRJIT_CALL_METHOD(get_self)
public:
    static constexpr const char *variant_() { return get_variant<Ts...>(); }
DRJIT_CALL_END()


DRJIT_CALL_TEMPLATE_INHERITED_BEGIN(A, Base)
    DRJIT_CALL_METHOD(a_gather_extra_value)
    DRJIT_CALL_GETTER(a_get_property)
DRJIT_CALL_END()


template <JitBackend Backend>
void bind(nb::module_ &m) {
    using Float = dr::DiffArray<Backend, float>;
    using UInt32 = dr::uint32_array_t<Float>;
    using BaseT = Base<Float>;
    using AT = A<Float>;
    using BT = B<Float>;
    using Mask = dr::mask_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;
    using Sampler = ::Sampler<Float>;

    auto sampler = nb::class_<Sampler>(m, "Sampler")
        .def(nb::init<>())
        .def(nb::init<size_t>())
        .def("next", &Sampler::next)
        .def_rw("rng", &Sampler::rng);

    bind_traverse(sampler);

    auto base_cls = nb::class_<BaseT, nb::intrusive_base>(m, "Base")
        .def("f", &BaseT::f)
        .def("f_masked", &BaseT::f_masked)
        .def("g", &BaseT::g)
        .def("nested", &BaseT::nested)
        .def("nested_self", &BaseT::nested_self)
        .def("sample", &BaseT::sample);
    bind_traverse(base_cls);

    auto a_cls = nb::class_<AT, BaseT>(m, "A")
        .def(nb::init<>())
        .def("a_get_property", &AT::a_get_property)
        .def("a_gather_extra_value", &AT::a_gather_extra_value)
        .def_rw("opaque", &AT::opaque)
        .def_rw("value", &AT::value)
        .def_rw("extra_value", &AT::extra_value)
        .def_rw("scalar_property", &AT::scalar_property);
    bind_traverse(a_cls);

    auto b_cls = nb::class_<BT, BaseT>(m, "B")
        .def(nb::init<>())
        .def_rw("opaque", &BT::opaque)
        .def_rw("value", &BT::value)
        .def_rw("s", &BT::s);
    bind_traverse(b_cls);

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
        .def("h", [](BaseArray &self, Float x) { return self->h(x); }, "x"_a)
        .def("nested",
             [](BaseArray &self, Float x, UInt32 s) { return self->nested(x, s); },
             "x"_a, "s"_a)
        .def("nested_self",
             [](BaseArray &self, Float x) { return self->nested_self(x); },
             "x"_a)
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


    dr::ArrayBinding a_ptr_b;
    using APtr = dr::DiffArray<Backend, AT *>;
    auto a_ptr = dr::bind_array_t<APtr>(a_ptr_b, m, "APtr")
        .def("a_get_property", [](APtr &self, Mask m) {
                return self->a_get_property(m);
             }, "mask"_a = true)
        .def("a_gather_extra_value", [](APtr &self, const UInt32 &idx, const Mask &m) {
                return self->a_gather_extra_value(idx, m);
             }, "idx"_a, "mask"_a);
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
