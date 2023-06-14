#include "bind.h"
#include "base.h"

NB_MODULE(drjit_ext, m_) {
    (void) m_;
    nb::module_ m = nb::module_::import_("drjit");

    nb::module_ detail = m.def_submodule("detail"),
                scalar = m.def_submodule("scalar"),
                cuda = m.def_submodule("cuda"),
                cuda_ad = cuda.def_submodule("ad"),
                llvm = m.def_submodule("llvm"),
                llvm_ad = llvm.def_submodule("ad");

    export_bind(detail);
    export_base(m);

    using T = drjit::Array<bool, 3>;
    ArrayBinding b;
    dr::bind_init<T>(b);
    dr::bind_base<T>(b);
    bind(b);

    using T2 = drjit::Array<float, 3>;
    dr::bind_init<T2>(b);
    dr::bind_base<T2>(b);
    bind(b);
}
