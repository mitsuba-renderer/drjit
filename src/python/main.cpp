#include "python.h"

static void log_callback(LogLevel /* level */, const char *msg) {
    /* Try to print to the Python console if possible, but *never*
       acquire the GIL and risk deadlock over this. */

    if (_Py_IsFinalizing()) {
        fprintf(stderr, "%s\n", msg);
    } else if (PyGILState_Check()) {
        nb::print(msg);
    } else {
        char *msg_copy = NB_STRDUP(msg);
        int rv = Py_AddPendingCall(
            [](void *p) -> int {
                nb::print((char *) p);
                free(p);
                return 0;
            }, msg_copy);
        if (rv)
            nb::detail::fail("log_callback(): Py_AddPendingCall(): failed!");
    }
}

NB_MODULE(drjit_ext, m_) {
#if defined(DRJIT_ENABLE_JIT)
    jit_set_log_level_stderr(LogLevel::Disable);
    jit_set_log_level_callback(LogLevel::Warn, log_callback);
    jit_init((uint32_t) JitBackend::CUDA |
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
    bind_traits(m);

    bind_scalar(scalar);
    bind_cuda(cuda);
    bind_cuda_ad(cuda_ad);
    bind_llvm(llvm);
    bind_llvm_ad(cuda_ad);

    m.def("shape", &shape);

    // Type aliases
    scalar.attr("Int") = nb::handle(&PyLong_Type);
    scalar.attr("UInt") = nb::handle(&PyLong_Type);
    scalar.attr("Int64") = nb::handle(&PyLong_Type);
    scalar.attr("UInt64") = nb::handle(&PyLong_Type);
    scalar.attr("Float") = nb::handle(&PyFloat_Type);
    scalar.attr("Float64") = nb::handle(&PyFloat_Type);
}
