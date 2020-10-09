#include "bind.h"
#include "random.h"

void export_packet(py::module_ &m) {
    py::module_ packet = m.def_submodule("packet");

    using Guide = ek::Packet<float>;
    ENOKI_BIND_ARRAY_BASE(packet, Guide, false);
    ENOKI_BIND_ARRAY_TYPES(packet, Guide, false);

    bind_pcg32<Guide>(packet);
}
