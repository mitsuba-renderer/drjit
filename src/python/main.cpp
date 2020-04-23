#include "common.h"

extern void export_route_basics(py::module &m);
extern void export_route_math(py::module &m);
extern void export_type_traits(py::module &m);
extern void export_constants(py::module &m);
extern void export_scalar(py::module &m);
extern void export_packet(py::module &m);

PYBIND11_MODULE(enoki, m) {
    export_route_basics(m);
    export_route_math(m);
    export_type_traits(m);
    export_constants(m);
    export_scalar(m);
    export_packet(m);
}
