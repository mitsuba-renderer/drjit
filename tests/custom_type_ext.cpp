#include <drjit/autodiff.h>
#include <drjit/packet.h>
#include <drjit/python.h>
#include <drjit/traversable_base.h>
#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <vector>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <typename Value_>
struct Color : dr::StaticArrayImpl<Value_, 3, false, Color<Value_>> {
    using Base = dr::StaticArrayImpl<Value_, 3, false, Color<Value_>>;

    /// Helper alias used to implement type promotion rules
    template <typename T> using ReplaceValue = Color<T>;

    using ArrayType = Color;
    using MaskType = dr::Mask<Value_, 3>;

    decltype(auto) r() const { return Base::x(); }
    decltype(auto) r() { return Base::x(); }

    decltype(auto) g() const { return Base::y(); }
    decltype(auto) g() { return Base::y(); }

    decltype(auto) b() const { return Base::z(); }
    decltype(auto) b() { return Base::z(); }

    DRJIT_ARRAY_IMPORT(Color, Base)
};


template <typename Value>
struct CustomHolder {
public:
    CustomHolder() {}
    CustomHolder(const Value &v) : m_value(v) {}
    Value &value() { return m_value; }
    bool schedule_force_() { return dr::detail::schedule_force(m_value); }

private:
    Value m_value;
};

class Object : public drjit::TraversableBase {
    DR_TRAVERSE_CB(drjit::TraversableBase);
};

template <typename Value>
class CustomBase : public Object{
    Value m_base_value;

public:
    CustomBase(const Value &base_value) : Object(), m_base_value(base_value) {}

    Value &base_value() { return m_base_value; }
    virtual Value &value() = 0;

    DR_TRAVERSE_CB(Object, m_base_value);
};

template <typename Value>
class PyCustomBase : public CustomBase<Value>{
public:
    using Base = CustomBase<Value>;
    NB_TRAMPOLINE(Base, 1);

    PyCustomBase(const Value &base_value) : Base(base_value) {}

    Value &value() override { NB_OVERRIDE_PURE(value); }

    DR_TRAMPOLINE_TRAVERSE_CB(Base);
};

template <typename Value>
class CustomA: public CustomBase<Value>{
public:
    using Base = CustomBase<Value>;

    CustomA(const Value &value, const Value &base_value) : Base(base_value), m_value(value) {}

    Value &value() override { return m_value; }

private:
    Value m_value;

    DR_TRAVERSE_CB(Base, m_value);
};

template<typename Value>
class Nested: Object{
    using Base = Object;

    std::vector<std::pair<nb::ref<Object>, size_t>> m_nested;

public:
    Nested(nb::ref<Object> a, nb::ref<Object> b) {
        m_nested.push_back(std::make_pair(a, 0));
        m_nested.push_back(std::make_pair(b, 1));
    }

    DR_TRAVERSE_CB(Base, m_nested);
};

template <JitBackend Backend> void bind(nb::module_ &m) {
    dr::ArrayBinding b;
    using Float = dr::DiffArray<Backend, float>;
    using Color3f = Color<Float>;

    dr::bind_array_t<Color3f>(b, m, "Color3f")
        .def_prop_rw("r",
            [](Color3f &c) -> Float & { return c.r(); },
            [](Color3f &c, Float &value) { c.r() = value; })
        .def_prop_rw("g",
            [](Color3f &c) -> Float & { return c.g(); },
            [](Color3f &c, Float &value) { c.g() = value; })
        .def_prop_rw("b",
            [](Color3f &c) -> Float & { return c.b(); },
            [](Color3f &c, Float &value) { c.b() = value; });

    using CustomFloatHolder = CustomHolder<Float>;
    nb::class_<CustomFloatHolder>(m, "CustomFloatHolder")
        .def(nb::init<Float>())
        .def("value", &CustomFloatHolder::value, nanobind::rv_policy::reference);

    using CustomBase   = CustomBase<Float>;
    using PyCustomBase = PyCustomBase<Float>;
    using CustomA      = CustomA<Float>;
    using Nested       = Nested<Float>;

    auto object = nb::class_<Object>(
        m, "Object",
        nb::intrusive_ptr<Object>(
            [](Object *o, PyObject *po) noexcept { o->set_self_py(po); }));

    auto base =
        nb::class_<CustomBase, Object, PyCustomBase>(m, "CustomBase")
            .def(nb::init<Float>())
            .def("value", nb::overload_cast<>(&CustomBase::value))
            .def("base_value", nb::overload_cast<>(&CustomBase::base_value));

    drjit::bind_traverse(base);

    auto a = nb::class_<CustomA, CustomBase>(m, "CustomA")
                 .def(nb::init<Float, Float>());

    drjit::bind_traverse(a);

    auto nested = nb::class_<Nested>(m, "Nested")
                      .def(nb::init<nb::ref<Object>, nb::ref<Object>>());

    drjit::bind_traverse(nested);

    m.def("cpp_make_opaque",
          [](CustomFloatHolder &holder) { dr::make_opaque(holder); }
    );
}

NB_MODULE(custom_type_ext, m) {
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });

#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<JitBackend::LLVM>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<JitBackend::CUDA>(cuda);
#endif

    // Tests: DRJIT_STRUCT, traversal mechanism, array/struct stringification
    m.def("struct_to_string", []{
        using Float = dr::Packet<float, 4>;
        using Array3f = dr::Array<Float, 3>;

        struct Ray {
            Float time;
            Array3f o, d;
            bool has_ray_differentials;

            DRJIT_STRUCT(Ray, time, o, d, has_ray_differentials)
        };

        Ray x = dr::zeros<Ray>();

        x.has_ray_differentials = true;
        x.o.y()[2] = 3;
        return dr::string(x);
    });
}
