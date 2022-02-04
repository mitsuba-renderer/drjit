#if defined(DRJIT_ENABLE_PYTHON_PACKET)
#include "bind.h"
#include "random.h"

void export_packet(py::module_ &m) {
    py::module_ packet = m.def_submodule("packet");

    using Guide = dr::Packet<float>;
    DRJIT_BIND_ARRAY_BASE(packet, Guide, false);
    DRJIT_BIND_ARRAY_TYPES(packet, Guide, false);

    bind_pcg32<Guide>(packet);
}
#endif
