#include "bind.h"

void export_packet(py::module &m) {
    py::module packet = m.def_submodule("packet");

    using Guide = ek::Packet<float>;
    ENOKI_BIND_ARRAY_BASE_1(packet, Guide, true);
    ENOKI_BIND_ARRAY_BASE_2(true);
    ENOKI_BIND_ARRAY_TYPES(packet, Guide, true);
}
