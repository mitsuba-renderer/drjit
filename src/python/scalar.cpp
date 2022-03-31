#include "python.h"
#include "random.h"

void bind_scalar(nb::module_ &m) {
    dr::bind_all_types<float>();
    bind_pcg32<float>(m);
}
