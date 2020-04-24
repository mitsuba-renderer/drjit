#include "bind.h"

void export_packet(py::module &m) {
    py::module packet = m.def_submodule("packet");

    using Guide = ek::Packet<float>;
    ENOKI_BIND_ARRAY_BASE_1(packet, Guide, false);
    ENOKI_BIND_ARRAY_BASE_2(false);
    ENOKI_BIND_ARRAY_TYPES(packet, Guide, false);
}
