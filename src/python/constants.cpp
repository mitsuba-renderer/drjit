#include "common.h"
#include <enoki/packet.h>

void export_constants(py::module &m) {
    m.attr("E")             = ek::E<double>;

    m.attr("Pi")            = ek::Pi<double>;
    m.attr("InvPi")         = ek::InvPi<double>;
    m.attr("SqrtPi")        = ek::SqrtPi<double>;
    m.attr("InvSqrtPi")     = ek::InvSqrtPi<double>;

    m.attr("TwoPi")         = ek::TwoPi<double>;
    m.attr("InvTwoPi")      = ek::InvTwoPi<double>;
    m.attr("SqrtTwoPi")     = ek::SqrtTwoPi<double>;
    m.attr("InvSqrtTwoPi")  = ek::InvSqrtTwoPi<double>;

    m.attr("FourPi")        = ek::FourPi<double>;
    m.attr("InvFourPi")     = ek::InvFourPi<double>;
    m.attr("SqrtFourPi")    = ek::SqrtFourPi<double>;
    m.attr("InvSqrtFourPi") = ek::InvSqrtFourPi<double>;

    m.attr("SqrtTwo")       = ek::SqrtTwo<double>;
    m.attr("InvSqrtTwo")    = ek::InvSqrtTwo<double>;

    m.attr("NaN")           = ek::NaN<double>;
    m.attr("Infinity")      = ek::Infinity<double>;

    m.def("Epsilon", [](const py::object &o) -> double {
        return var_type(o) == VarType::Float32 ? (double) ek::Epsilon<float>
                                               : ek::Epsilon<double>;
    });

    m.def("OneMinusEpsilon", [](const py::object &o) -> double {
        return var_type(o) == VarType::Float32
                   ? (double) ek::OneMinusEpsilon<float>
                   : ek::OneMinusEpsilon<double>;
    });

    m.def("RecipOverflow", [](const py::object &o) -> double {
        return var_type(o) == VarType::Float32
                   ? (double) ek::RecipOverflow<float>
                   : ek::RecipOverflow<double>;
    });

    m.attr("has_x86_32") = has_x86_32;
    m.attr("has_x86_64") = has_x86_64;
    m.attr("has_x86") = has_x86;
    m.attr("has_arm_32") = has_arm_32;
    m.attr("has_arm_64") = has_arm_64;
    m.attr("has_arm") = has_arm;
    m.attr("has_sse42") = has_sse42;
    m.attr("has_fma") = has_fma;
    m.attr("has_f16c") = has_f16c;
    m.attr("has_avx") = has_avx;
    m.attr("has_avx2") = has_avx2;
    m.attr("has_avx512") = has_avx512;
    m.attr("has_neon") = has_neon;
    m.attr("has_vectorization") = has_vectorization;
}
