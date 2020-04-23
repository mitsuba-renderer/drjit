#include "bind.h"

void export_scalar(py::module &m) {
    py::module scalar = m.def_submodule("scalar");
    ENOKI_BIND_ARRAY_TYPES(scalar, float, false);

    bind_full(d_i32); bind_full(d_u32); bind_full(d_i64);
    bind_full(d_u64); bind_full(d_f32); bind_full(d_f64);
    bind_full(d_b);
}
