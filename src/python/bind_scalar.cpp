#include "bind.h"

void export_scalar(py::module &m) {
    py::module scalar = m.def_submodule("scalar");
    ENOKI_BIND_ARRAY_TYPES(scalar, float, true);

    bind_full(d_i32, true); bind_full(d_u32, true); bind_full(d_i64, true);
    bind_full(d_u64, true); bind_full(d_f32, true); bind_full(d_f64, true);
    bind_full(d_b, true);
}
