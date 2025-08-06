#define NB_INTRUSIVE_EXPORT NB_IMPORT

#include <nanobind/nanobind.h>
#include <drjit/local.h>

namespace nb = nanobind;
namespace dr = drjit;

template <typename Float>
void bind(nb::module_ &m) {
    using UInt32 = dr::uint32_array_t<Float>;

    m.def("lookup", [](UInt32 offset, Float value, UInt32 offset2) {
        dr::Local<Float, 10> local(3.f);
        local.write(offset, value);
        return local.read(offset2);
    });
}

NB_MODULE(local_ext, m) {
    nb::module_::import_("drjit");

#if defined(DRJIT_ENABLE_LLVM)
    nb::module_ llvm = m.def_submodule("llvm");
    bind<dr::LLVMArray<float>>(llvm);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    nb::module_ cuda = m.def_submodule("cuda");
    bind<dr::CUDAArray<float>>(llvm);
#endif

    nb::module_ scalar = m.def_submodule("scalar");
    bind<float>(scalar);
}
