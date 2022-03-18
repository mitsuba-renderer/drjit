#if defined(DRJIT_ENABLE_PYTHON_PACKET)
#include "bind.h"
#include "random.h"

void export_packet(nb::module_ &m) {
    nb::module_ packet = m.def_submodule("packet");

    using Guide = dr::Packet<float>;
    DRJIT_BIND_ARRAY_BASE(packet, Guide, false);
    DRJIT_BIND_ARRAY_TYPES(packet, Guide, false);

    bind_pcg32<Guide>(packet);
}
#endif
