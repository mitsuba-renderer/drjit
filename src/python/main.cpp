/*
    main.cpp -- Entry point of the Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "bind.h"
#include "base.h"
#include "shape.h"
#include "log.h"
#include "traits.h"
#include "scalar.h"
#include "llvm.h"
#include "reduce.h"
#include "eval.h"
#include "iter.h"
#include "init.h"
#include "memop.h"
#include "slice.h"
#include "dlpack.h"
#include "autodiff.h"

NB_MODULE(drjit_ext, m_) {
    (void) m_;
    nb::module_ m = nb::module_::import_("drjit");
    m.doc() = "A Just-In-Time-Compiler for Differentiable Rendering";

    export_log(m, nanobind_module_def_drjit_ext);

    uint32_t backends = 0;

#if defined(DRJIT_ENABLE_LLVM)
    backends |= (uint32_t) JitBackend::LLVM;

    nb::module_ llvm = m.def_submodule("llvm"),
                llvm_ad = llvm.def_submodule("ad");
#endif

#if defined(DRJIT_ENABLE_CUDA)
    backends |= (uint32_t) JitBackend::CUDA;

    nb::module_ cuda = m.def_submodule("cuda"),
                cuda_ad = cuda.def_submodule("ad");
#endif

    nb::enum_<JitBackend>(m, "JitBackend")
        .value("CUDA", JitBackend::CUDA)
        .value("LLVM", JitBackend::LLVM);

    m.def("has_backend", &jit_has_backend);

    m.def("whos_str", &jit_var_whos);
    m.def("whos", []() { nb::print(jit_var_whos()); });

    jit_init(backends);

    nb::module_ detail = m.attr("detail"),
                scalar = m.def_submodule("scalar");

    export_bind(detail);
    export_base(m);
    export_init(m);
    export_shape(m);
    export_traits(m);
    export_iter(detail);
    export_reduce(m);
    export_eval(m);
    export_memop(m);
    export_slice(m);
    export_dlpack(m);
    export_autodiff(m);

    export_scalar();

#if defined(DRJIT_ENABLE_LLVM)
    export_llvm();
    export_llvm_ad();
#endif
}
