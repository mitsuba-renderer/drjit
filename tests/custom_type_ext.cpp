#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/packet.h>

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
}

NB_MODULE(custom_type_ext, m) {
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
