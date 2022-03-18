#include "python.h"
#include <drjit/autodiff.h>

static void log_callback(LogLevel /* level */, const char *msg) {
    /* Try to print to the Python console if possible, but *never*
       acquire the GIL and risk deadlock over this. */
    if (nb::safe())
        nb::print(msg);
    else
        fprintf(stderr, "%s\n", msg);
}

extern void bind_arraybase(nb::module_ m);
extern void bind_ops(nb::module_ m);

NB_MODULE(drjit_ext, m_) {
#if defined(DRJIT_ENABLE_JIT)
    jit_set_log_level_stderr(LogLevel::Disable);
    jit_set_log_level_callback(LogLevel::Warn, log_callback);
    jit_init_async((uint32_t) JitBackend::CUDA |
                   (uint32_t) JitBackend::LLVM);
#endif

    (void) m_;
    nb::module_ m = nb::module_::import_("drjit"),
                scalar = m.def_submodule("scalar"),
                cuda = m.def_submodule("cuda"),
                cuda_ad = cuda.def_submodule("ad"),
                llvm = m.def_submodule("llvm"),
                llvm_ad = llvm.def_submodule("ad");

    bind_arraybase(m);
    bind_ops(m);

    dr::bind_2<float>();
    dr::bind_2<dr::LLVMArray<float>>();
    dr::bind_2<dr::CUDAArray<float>>();
    dr::bind_2<dr::DiffArray<dr::LLVMArray<float>>>();
    dr::bind_2<dr::DiffArray<dr::CUDAArray<float>>>();

    m.def("shape", &shape);

    // Type aliases
    scalar.attr("Int32") = nb::handle(&PyLong_Type);
    scalar.attr("UInt32") = nb::handle(&PyLong_Type);
    scalar.attr("Int64") = nb::handle(&PyLong_Type);
    scalar.attr("UInt64") = nb::handle(&PyLong_Type);
    scalar.attr("Float") = nb::handle(&PyFloat_Type);
    scalar.attr("Float32") = nb::handle(&PyFloat_Type);
    scalar.attr("Float64") = nb::handle(&PyFloat_Type);
    llvm.attr("Int32") = llvm.attr("Int");
    llvm.attr("UInt32") = llvm.attr("UInt");
    llvm.attr("Float32") = llvm.attr("Float");
}
