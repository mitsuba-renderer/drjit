#define NB_INTRUSIVE_EXPORT NB_IMPORT

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <drjit/local.h>
#include <drjit/while_loop.h>
#include <drjit-core/nanostl.h>

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

    m.def(("test_" + name + "_loop").c_str(), []() {
        auto initial = Local();
        auto counter = int32_t(0);

        if constexpr (dr::is_jit_v<Float>)
            initial.resize(10);

        dr::tie(initial, counter) = dr::while_loop(
            dr::make_tuple(initial, counter),
            [](const Local& l, const int32_t& i) {
                DRJIT_MARK_USED(l);
                return i < 5;
            },
            [](Local& l, int32_t& i) {
                l.write(i, dr::full<typename Local::Value>(i));
                auto written = l.read(i);
                DRJIT_MARK_USED(written);
                i += 1;
            }
        );
        for(unsigned int i = 0; i < 5; ++i) {
            auto value = initial.read(i);
            if(dr::any(value != dr::full<typename Local::Value>(i))) {
                jit_raise("Index %d doesn't match %s", i, dr::string(value).c_str());
            }
        }
    });

    m.def(("test_" + name + "_loop_struct").c_str(), []() {
        auto initial = Local();
        auto counter = int32_t(0);

        if constexpr (dr::is_jit_v<Float>)
            initial.resize(10);


        struct LoopState {
            Local l;
            int32_t counter;

            DRJIT_STRUCT(LoopState, l, counter)
        } ls = { initial, counter };

        dr::tie(ls) = dr::while_loop(
            dr::make_tuple(ls),
            [](const LoopState &ls) { return ls.counter < 5; },
            [](LoopState &ls) {
                ls.l.write(ls.counter, dr::full<typename Local::Value>(ls.counter));
                auto written = ls.l.read(ls.counter);
                DRJIT_MARK_USED(written);
                ls.counter += 1;
            }
        );
        for(unsigned int i = 0; i < 5; ++i) {
            auto value = ls.l.read(i);
            if(dr::any(value != dr::full<typename Local::Value>(i))) {
                jit_raise("Index %d doesn't match %s", i, dr::string(value).c_str());
            }
        }
    });

    return c;
}

template <typename Float>
void bind(nb::module_ &m) {
    using UInt32 = dr::uint32_array_t<Float>;
    using Bool = dr::bool_array_t<Float>;

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

        Bool operator!=(const MyStruct& other) const {
            return priority != other.priority;
        }

        MyStruct(int i) : value(i), priority(i) {}
    };

    auto mystruct = nb::class_<MyStruct>(m, "MyStruct")
        .def(nb::init<>())
        .def(nb::init<int>())
        .def_rw("value", &MyStruct::value)
        .def_rw("priority", &MyStruct::priority)
        .def(nb::self != nb::self);

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
