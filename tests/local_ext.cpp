#define NB_INTRUSIVE_EXPORT NB_IMPORT

#include <nanobind/nanobind.h>
#include <drjit/local.h>

namespace nb = nanobind;
namespace dr = drjit;

using nb::literals::operator""_a;

template <typename Float, typename Local>
auto bind_local(nb::module_ &m, const dr::string& name) {
    auto c =  nb::class_<Local>(m, name.c_str())
        .def(nb::init<>())
        .def(nb::init<typename Local::Value>())
        .def("__len__", &Local::size)
        .def("read", &Local::read, "index"_a, "active"_a = true)
        .def("write", &Local::write, "offset"_a, "value"_a, "active"_a = true);

    if constexpr (dr::is_jit_v<Float>)
        c = c.def("resize", &Local::resize);

    return c;
}

template <typename Float>
void bind(nb::module_ &m) {
    using UInt32 = dr::uint32_array_t<Float>;
    using Local10 = dr::Local<Float, 10>;
    using LocalDyn = dr::Local<Float, dr::Dynamic>;

    bind_local<Float, Local10>(m, "Local10");

    if constexpr (dr::is_jit_v<Float>)
        bind_local<Float, LocalDyn>(m, "LocalDyn");

    struct MyStruct
    {
        Float value;
        UInt32 priority;
        DRJIT_STRUCT(MyStruct, value, priority)

        MyStruct(int i) : value(i), priority(i) {}
    };

    auto mystruct = nb::class_<MyStruct>(m, "MyStruct")
        .def(nb::init<>())
        .def(nb::init<int>())
        .def_rw("value", &MyStruct::value)
        .def_rw("priority", &MyStruct::priority);

    nb::handle u32;
    if constexpr (dr::is_array_v<UInt32>)
        u32 = nb::type<UInt32>();
    else
        u32 = nb::handle((PyObject *) &PyLong_Type);

    nb::handle f32;
    if constexpr (dr::is_array_v<UInt32>)
        f32 = nb::type<Float>();
    else
        f32 = nb::handle((PyObject *) &PyFloat_Type);

    nb::dict fields;
    fields["value"] = f32;
    fields["priority"] = u32;
    mystruct.attr("DRJIT_STRUCT") = fields;

    using LocalStruct10 = dr::Local<MyStruct, 10, UInt32>;
    bind_local<Float, LocalStruct10>(m, "LocalStruct10");
}

NB_MODULE(local_ext, m) {
    nb::module_::import_("drjit");

#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<dr::LLVMArray<float>>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<dr::CUDAArray<float>>(cuda);
#endif

    nb::module_ scalar = m.def_submodule("scalar");
    bind<float>(scalar);
}
