/*
    drjit/sh.h -- Real spherical harmonics evaluation routines

    The generated code is based on the paper `Efficient Spherical Harmonic
    Evaluation, Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2,
    84-90, 2013 by Peter-Pike Sloan

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>

NAMESPACE_BEGIN(drjit)

template <typename Vector3f>
void sh_eval(const Vector3f &d, size_t order, value_t<Vector3f> *out) {
    switch (order) {
        case 0: sh_eval_0(d, out); break;
        case 1: sh_eval_1(d, out); break;
        case 2: sh_eval_2(d, out); break;
        case 3: sh_eval_3(d, out); break;
        case 4: sh_eval_4(d, out); break;
        case 5: sh_eval_5(d, out); break;
        case 6: sh_eval_6(d, out); break;
        case 7: sh_eval_7(d, out); break;
        case 8: sh_eval_8(d, out); break;
        case 9: sh_eval_9(d, out); break;
        default: throw drjit::Exception("sh_eval(): order too high!");
    }
}

template <typename Vector3f>
void sh_eval_0(const Vector3f &, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    out[0] = Value(Scalar(0.28209479177387814));
}

template <typename Vector3f>
void sh_eval_1(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z();
    Value c0, s0, tmp_a;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
}

template <typename Vector3f>
void sh_eval_2(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_c = Scalar(0.546274215296039478);
    out[8] = tmp_c * c1;
    out[4] = tmp_c * s1;
}

template <typename Vector3f>
void sh_eval_3(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_c = Scalar(-0.590043589926643519);
    out[15] = tmp_c * c0;
    out[9] = tmp_c * s0;
}

template <typename Vector3f>
void sh_eval_4(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    out[20] = fmadd(z * Scalar(1.98431348329844304), out[12], out[6] * Scalar(-1.00623058987490532));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-4.6833258049010249), Scalar(2.00713963067186763));
    out[21] = tmp_a * c0;
    out[19] = tmp_a * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(3.31161143515146028), Scalar(-0.473087347878779985));
    out[22] = tmp_c * c1;
    out[18] = tmp_c * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.590043589926643519);
    out[15] = tmp_a * c0;
    out[9] = tmp_a * s0;
    tmp_b = z * Scalar(-1.77013076977993067);
    out[23] = tmp_b * c0;
    out[17] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_c = Scalar(0.625835735449176256);
    out[24] = tmp_c * c1;
    out[16] = tmp_c * s1;
}

template <typename Vector3f>
void sh_eval_5(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    out[20] = fmadd(z * Scalar(1.98431348329844304), out[12], out[6] * Scalar(-1.00623058987490532));
    out[30] = fmadd(z * Scalar(1.98997487421323993), out[20], out[12] * Scalar(-1.00285307284481395));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-4.6833258049010249), Scalar(2.00713963067186763));
    out[21] = tmp_a * c0;
    out[19] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.03100960115899021), tmp_a, tmp_c * Scalar(-0.991031208965114985));
    out[31] = tmp_b * c0;
    out[29] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(3.31161143515146028), Scalar(-0.473087347878779985));
    out[22] = tmp_c * c1;
    out[18] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(7.19030517745998665), Scalar(-2.39676839248666207));
    out[32] = tmp_a * c1;
    out[28] = tmp_a * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.590043589926643519);
    out[15] = tmp_a * c0;
    out[9] = tmp_a * s0;
    tmp_b = z * Scalar(-1.77013076977993067);
    out[23] = tmp_b * c0;
    out[17] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-4.40314469491725369), Scalar(0.48923829943525049));
    out[33] = tmp_c * c0;
    out[27] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.625835735449176256);
    out[24] = tmp_a * c1;
    out[16] = tmp_a * s1;
    tmp_b = z * Scalar(2.07566231488104114);
    out[34] = tmp_b * c1;
    out[26] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_c = Scalar(-0.656382056840170258);
    out[35] = tmp_c * c0;
    out[25] = tmp_c * s0;
}

template <typename Vector3f>
void sh_eval_6(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    out[20] = fmadd(z * Scalar(1.98431348329844304), out[12], out[6] * Scalar(-1.00623058987490532));
    out[30] = fmadd(z * Scalar(1.98997487421323993), out[20], out[12] * Scalar(-1.00285307284481395));
    out[42] = fmadd(z * Scalar(1.99304345718356646), out[30], out[20] * Scalar(-1.00154202096221923));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-4.6833258049010249), Scalar(2.00713963067186763));
    out[21] = tmp_a * c0;
    out[19] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.03100960115899021), tmp_a, tmp_c * Scalar(-0.991031208965114985));
    out[31] = tmp_b * c0;
    out[29] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.02131498923702768), tmp_b, tmp_a * Scalar(-0.995226703056238504));
    out[43] = tmp_c * c0;
    out[41] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(3.31161143515146028), Scalar(-0.473087347878779985));
    out[22] = tmp_c * c1;
    out[18] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(7.19030517745998665), Scalar(-2.39676839248666207));
    out[32] = tmp_a * c1;
    out[28] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.11394181566096995), tmp_a, tmp_c * Scalar(-0.973610120462326756));
    out[44] = tmp_b * c1;
    out[40] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.590043589926643519);
    out[15] = tmp_a * c0;
    out[9] = tmp_a * s0;
    tmp_b = z * Scalar(-1.77013076977993067);
    out[23] = tmp_b * c0;
    out[17] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-4.40314469491725369), Scalar(0.48923829943525049));
    out[33] = tmp_c * c0;
    out[27] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-10.1332578546641603), Scalar(2.76361577854477058));
    out[45] = tmp_a * c0;
    out[39] = tmp_a * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.625835735449176256);
    out[24] = tmp_a * c1;
    out[16] = tmp_a * s1;
    tmp_b = z * Scalar(2.07566231488104114);
    out[34] = tmp_b * c1;
    out[26] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(5.55021390801596581), Scalar(-0.504564900728724064));
    out[46] = tmp_c * c1;
    out[38] = tmp_c * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.656382056840170258);
    out[35] = tmp_a * c0;
    out[25] = tmp_a * s0;
    tmp_b = z * Scalar(-2.3666191622317525);
    out[47] = tmp_b * c0;
    out[37] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_c = Scalar(0.683184105191914415);
    out[48] = tmp_c * c1;
    out[36] = tmp_c * s1;
}

template <typename Vector3f>
void sh_eval_7(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    out[20] = fmadd(z * Scalar(1.98431348329844304), out[12], out[6] * Scalar(-1.00623058987490532));
    out[30] = fmadd(z * Scalar(1.98997487421323993), out[20], out[12] * Scalar(-1.00285307284481395));
    out[42] = fmadd(z * Scalar(1.99304345718356646), out[30], out[20] * Scalar(-1.00154202096221923));
    out[56] = fmadd(z * Scalar(1.99489143482413467), out[42], out[30] * Scalar(-1.00092721392195827));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-4.6833258049010249), Scalar(2.00713963067186763));
    out[21] = tmp_a * c0;
    out[19] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.03100960115899021), tmp_a, tmp_c * Scalar(-0.991031208965114985));
    out[31] = tmp_b * c0;
    out[29] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.02131498923702768), tmp_b, tmp_a * Scalar(-0.995226703056238504));
    out[43] = tmp_c * c0;
    out[41] = tmp_c * s0;
    tmp_a = fmadd(z * Scalar(2.01556443707463773), tmp_c, tmp_b * Scalar(-0.99715504402183186));
    out[57] = tmp_a * c0;
    out[55] = tmp_a * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(3.31161143515146028), Scalar(-0.473087347878779985));
    out[22] = tmp_c * c1;
    out[18] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(7.19030517745998665), Scalar(-2.39676839248666207));
    out[32] = tmp_a * c1;
    out[28] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.11394181566096995), tmp_a, tmp_c * Scalar(-0.973610120462326756));
    out[44] = tmp_b * c1;
    out[40] = tmp_b * s1;
    tmp_c = fmadd(z * Scalar(2.08166599946613307), tmp_b, tmp_a * Scalar(-0.984731927834661791));
    out[58] = tmp_c * c1;
    out[54] = tmp_c * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.590043589926643519);
    out[15] = tmp_a * c0;
    out[9] = tmp_a * s0;
    tmp_b = z * Scalar(-1.77013076977993067);
    out[23] = tmp_b * c0;
    out[17] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-4.40314469491725369), Scalar(0.48923829943525049));
    out[33] = tmp_c * c0;
    out[27] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-10.1332578546641603), Scalar(2.76361577854477058));
    out[45] = tmp_a * c0;
    out[39] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.20794021658196149), tmp_a, tmp_c * Scalar(-0.95940322360024699));
    out[59] = tmp_b * c0;
    out[53] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.625835735449176256);
    out[24] = tmp_a * c1;
    out[16] = tmp_a * s1;
    tmp_b = z * Scalar(2.07566231488104114);
    out[34] = tmp_b * c1;
    out[26] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(5.55021390801596581), Scalar(-0.504564900728724064));
    out[46] = tmp_c * c1;
    out[38] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(13.4918050467267694), Scalar(-3.11349347232156193));
    out[60] = tmp_a * c1;
    out[52] = tmp_a * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.656382056840170258);
    out[35] = tmp_a * c0;
    out[25] = tmp_a * s0;
    tmp_b = z * Scalar(-2.3666191622317525);
    out[47] = tmp_b * c0;
    out[37] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-6.7459025233633847), Scalar(0.518915578720260395));
    out[61] = tmp_c * c0;
    out[51] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.683184105191914415);
    out[48] = tmp_a * c1;
    out[36] = tmp_a * s1;
    tmp_b = z * Scalar(2.64596066180190048);
    out[62] = tmp_b * c1;
    out[50] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_c = Scalar(-0.707162732524596271);
    out[63] = tmp_c * c0;
    out[49] = tmp_c * s0;
}

template <typename Vector3f>
void sh_eval_8(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    out[20] = fmadd(z * Scalar(1.98431348329844304), out[12], out[6] * Scalar(-1.00623058987490532));
    out[30] = fmadd(z * Scalar(1.98997487421323993), out[20], out[12] * Scalar(-1.00285307284481395));
    out[42] = fmadd(z * Scalar(1.99304345718356646), out[30], out[20] * Scalar(-1.00154202096221923));
    out[56] = fmadd(z * Scalar(1.99489143482413467), out[42], out[30] * Scalar(-1.00092721392195827));
    out[72] = fmadd(z * Scalar(1.9960899278339137), out[56], out[42] * Scalar(-1.00060078106951478));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-4.6833258049010249), Scalar(2.00713963067186763));
    out[21] = tmp_a * c0;
    out[19] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.03100960115899021), tmp_a, tmp_c * Scalar(-0.991031208965114985));
    out[31] = tmp_b * c0;
    out[29] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.02131498923702768), tmp_b, tmp_a * Scalar(-0.995226703056238504));
    out[43] = tmp_c * c0;
    out[41] = tmp_c * s0;
    tmp_a = fmadd(z * Scalar(2.01556443707463773), tmp_c, tmp_b * Scalar(-0.99715504402183186));
    out[57] = tmp_a * c0;
    out[55] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.01186954040739119), tmp_a, tmp_c * Scalar(-0.998166817890174474));
    out[73] = tmp_b * c0;
    out[71] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(3.31161143515146028), Scalar(-0.473087347878779985));
    out[22] = tmp_c * c1;
    out[18] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(7.19030517745998665), Scalar(-2.39676839248666207));
    out[32] = tmp_a * c1;
    out[28] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.11394181566096995), tmp_a, tmp_c * Scalar(-0.973610120462326756));
    out[44] = tmp_b * c1;
    out[40] = tmp_b * s1;
    tmp_c = fmadd(z * Scalar(2.08166599946613307), tmp_b, tmp_a * Scalar(-0.984731927834661791));
    out[58] = tmp_c * c1;
    out[54] = tmp_c * s1;
    tmp_a = fmadd(z * Scalar(2.06155281280883029), tmp_c, tmp_b * Scalar(-0.990337937660287326));
    out[74] = tmp_a * c1;
    out[70] = tmp_a * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.590043589926643519);
    out[15] = tmp_a * c0;
    out[9] = tmp_a * s0;
    tmp_b = z * Scalar(-1.77013076977993067);
    out[23] = tmp_b * c0;
    out[17] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-4.40314469491725369), Scalar(0.48923829943525049));
    out[33] = tmp_c * c0;
    out[27] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-10.1332578546641603), Scalar(2.76361577854477058));
    out[45] = tmp_a * c0;
    out[39] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.20794021658196149), tmp_a, tmp_c * Scalar(-0.95940322360024699));
    out[59] = tmp_b * c0;
    out[53] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.15322168769582012), tmp_b, tmp_a * Scalar(-0.975217386560017774));
    out[75] = tmp_c * c0;
    out[69] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.625835735449176256);
    out[24] = tmp_a * c1;
    out[16] = tmp_a * s1;
    tmp_b = z * Scalar(2.07566231488104114);
    out[34] = tmp_b * c1;
    out[26] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(5.55021390801596581), Scalar(-0.504564900728724064));
    out[46] = tmp_c * c1;
    out[38] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(13.4918050467267694), Scalar(-3.11349347232156193));
    out[60] = tmp_a * c1;
    out[52] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.30488611432322132), tmp_a, tmp_c * Scalar(-0.948176387355465389));
    out[76] = tmp_b * c1;
    out[68] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.656382056840170258);
    out[35] = tmp_a * c0;
    out[25] = tmp_a * s0;
    tmp_b = z * Scalar(-2.3666191622317525);
    out[47] = tmp_b * c0;
    out[37] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-6.7459025233633847), Scalar(0.518915578720260395));
    out[61] = tmp_c * c0;
    out[51] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-17.2495531104905417), Scalar(3.44991062209810817));
    out[77] = tmp_a * c0;
    out[67] = tmp_a * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.683184105191914415);
    out[48] = tmp_a * c1;
    out[36] = tmp_a * s1;
    tmp_b = z * Scalar(2.64596066180190048);
    out[62] = tmp_b * c1;
    out[50] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(7.98499149089313942), Scalar(-0.532332766059542606));
    out[78] = tmp_c * c1;
    out[66] = tmp_c * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.707162732524596271);
    out[63] = tmp_a * c0;
    out[49] = tmp_a * s0;
    tmp_b = z * Scalar(-2.91570664069931995);
    out[79] = tmp_b * c0;
    out[65] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_c = Scalar(0.728926660174829988);
    out[80] = tmp_c * c1;
    out[64] = tmp_c * s1;
}

template <typename Vector3f>
void sh_eval_9(const Vector3f &d, value_t<Vector3f> *out) {
    static_assert(array_size_v<Vector3f> == 3, "The parameter 'd' should be a 3D vector.");

    using Value = value_t<Vector3f>;
    using Scalar = scalar_t<Value>;

    Value x = d.x(), y = d.y(), z = d.z(), z2 = z * z;
    Value c0, c1, s0, s1, tmp_a, tmp_b, tmp_c;

    out[0] = Value(Scalar(0.28209479177387814));
    out[2] = z * Scalar(0.488602511902919923);
    out[6] = fmadd(z2, Scalar(0.94617469575756008), Scalar(-0.315391565252520045));
    out[12] = z * fmadd(z2, Scalar(1.865881662950577), Scalar(-1.1195289977703462));
    out[20] = fmadd(z * Scalar(1.98431348329844304), out[12], out[6] * Scalar(-1.00623058987490532));
    out[30] = fmadd(z * Scalar(1.98997487421323993), out[20], out[12] * Scalar(-1.00285307284481395));
    out[42] = fmadd(z * Scalar(1.99304345718356646), out[30], out[20] * Scalar(-1.00154202096221923));
    out[56] = fmadd(z * Scalar(1.99489143482413467), out[42], out[30] * Scalar(-1.00092721392195827));
    out[72] = fmadd(z * Scalar(1.9960899278339137), out[56], out[42] * Scalar(-1.00060078106951478));
    out[90] = fmadd(z * Scalar(1.99691119506793657), out[72], out[56] * Scalar(-1.0004114379931337));
    c0 = x;
    s0 = y;

    tmp_a = Scalar(-0.488602511902919978);
    out[3] = tmp_a * c0;
    out[1] = tmp_a * s0;
    tmp_b = z * Scalar(-1.09254843059207896);
    out[7] = tmp_b * c0;
    out[5] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-2.28522899732232876), Scalar(0.457045799464465774));
    out[13] = tmp_c * c0;
    out[11] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-4.6833258049010249), Scalar(2.00713963067186763));
    out[21] = tmp_a * c0;
    out[19] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.03100960115899021), tmp_a, tmp_c * Scalar(-0.991031208965114985));
    out[31] = tmp_b * c0;
    out[29] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.02131498923702768), tmp_b, tmp_a * Scalar(-0.995226703056238504));
    out[43] = tmp_c * c0;
    out[41] = tmp_c * s0;
    tmp_a = fmadd(z * Scalar(2.01556443707463773), tmp_c, tmp_b * Scalar(-0.99715504402183186));
    out[57] = tmp_a * c0;
    out[55] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.01186954040739119), tmp_a, tmp_c * Scalar(-0.998166817890174474));
    out[73] = tmp_b * c0;
    out[71] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.00935312974101166), tmp_b, tmp_a * Scalar(-0.998749217771908837));
    out[91] = tmp_c * c0;
    out[89] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.546274215296039478);
    out[8] = tmp_a * c1;
    out[4] = tmp_a * s1;
    tmp_b = z * Scalar(1.44530572132027735);
    out[14] = tmp_b * c1;
    out[10] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(3.31161143515146028), Scalar(-0.473087347878779985));
    out[22] = tmp_c * c1;
    out[18] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(7.19030517745998665), Scalar(-2.39676839248666207));
    out[32] = tmp_a * c1;
    out[28] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.11394181566096995), tmp_a, tmp_c * Scalar(-0.973610120462326756));
    out[44] = tmp_b * c1;
    out[40] = tmp_b * s1;
    tmp_c = fmadd(z * Scalar(2.08166599946613307), tmp_b, tmp_a * Scalar(-0.984731927834661791));
    out[58] = tmp_c * c1;
    out[54] = tmp_c * s1;
    tmp_a = fmadd(z * Scalar(2.06155281280883029), tmp_c, tmp_b * Scalar(-0.990337937660287326));
    out[74] = tmp_a * c1;
    out[70] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.04812235835781919), tmp_a, tmp_c * Scalar(-0.993485272670404207));
    out[92] = tmp_b * c1;
    out[88] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.590043589926643519);
    out[15] = tmp_a * c0;
    out[9] = tmp_a * s0;
    tmp_b = z * Scalar(-1.77013076977993067);
    out[23] = tmp_b * c0;
    out[17] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-4.40314469491725369), Scalar(0.48923829943525049));
    out[33] = tmp_c * c0;
    out[27] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-10.1332578546641603), Scalar(2.76361577854477058));
    out[45] = tmp_a * c0;
    out[39] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.20794021658196149), tmp_a, tmp_c * Scalar(-0.95940322360024699));
    out[59] = tmp_b * c0;
    out[53] = tmp_b * s0;
    tmp_c = fmadd(z * Scalar(2.15322168769582012), tmp_b, tmp_a * Scalar(-0.975217386560017774));
    out[75] = tmp_c * c0;
    out[69] = tmp_c * s0;
    tmp_a = fmadd(z * Scalar(2.11804417118980526), tmp_c, tmp_b * Scalar(-0.983662844979209416));
    out[93] = tmp_a * c0;
    out[87] = tmp_a * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.625835735449176256);
    out[24] = tmp_a * c1;
    out[16] = tmp_a * s1;
    tmp_b = z * Scalar(2.07566231488104114);
    out[34] = tmp_b * c1;
    out[26] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(5.55021390801596581), Scalar(-0.504564900728724064));
    out[46] = tmp_c * c1;
    out[38] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(13.4918050467267694), Scalar(-3.11349347232156193));
    out[60] = tmp_a * c1;
    out[52] = tmp_a * s1;
    tmp_b = fmadd(z * Scalar(2.30488611432322132), tmp_a, tmp_c * Scalar(-0.948176387355465389));
    out[76] = tmp_b * c1;
    out[68] = tmp_b * s1;
    tmp_c = fmadd(z * Scalar(2.22917715070623501), tmp_b, tmp_a * Scalar(-0.967152839723182112));
    out[94] = tmp_c * c1;
    out[86] = tmp_c * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.656382056840170258);
    out[35] = tmp_a * c0;
    out[25] = tmp_a * s0;
    tmp_b = z * Scalar(-2.3666191622317525);
    out[47] = tmp_b * c0;
    out[37] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-6.7459025233633847), Scalar(0.518915578720260395));
    out[61] = tmp_c * c0;
    out[51] = tmp_c * s0;
    tmp_a = z * fmadd(z2, Scalar(-17.2495531104905417), Scalar(3.44991062209810817));
    out[77] = tmp_a * c0;
    out[67] = tmp_a * s0;
    tmp_b = fmadd(z * Scalar(2.40163634692206163), tmp_a, tmp_c * Scalar(-0.939224604204370817));
    out[95] = tmp_b * c0;
    out[85] = tmp_b * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.683184105191914415);
    out[48] = tmp_a * c1;
    out[36] = tmp_a * s1;
    tmp_b = z * Scalar(2.64596066180190048);
    out[62] = tmp_b * c1;
    out[50] = tmp_b * s1;
    tmp_c = fmadd(z2, Scalar(7.98499149089313942), Scalar(-0.532332766059542606));
    out[78] = tmp_c * c1;
    out[66] = tmp_c * s1;
    tmp_a = z * fmadd(z2, Scalar(21.3928901909086377), Scalar(-3.77521591604270101));
    out[96] = tmp_a * c1;
    out[84] = tmp_a * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_a = Scalar(-0.707162732524596271);
    out[63] = tmp_a * c0;
    out[49] = tmp_a * s0;
    tmp_b = z * Scalar(-2.91570664069931995);
    out[79] = tmp_b * c0;
    out[65] = tmp_b * s0;
    tmp_c = fmadd(z2, Scalar(-9.26339318284890467), Scalar(0.544905481344053255));
    out[97] = tmp_c * c0;
    out[83] = tmp_c * s0;
    c1 = fmsub(x, c0, y * s0);
    s1 = fmadd(x, s0, y * c0);

    tmp_a = Scalar(0.728926660174829988);
    out[80] = tmp_a * c1;
    out[64] = tmp_a * s1;
    tmp_b = z * Scalar(3.17731764895469793);
    out[98] = tmp_b * c1;
    out[82] = tmp_b * s1;
    c0 = fmsub(x, c1, y * s1);
    s0 = fmadd(x, s1, y * c1);

    tmp_c = Scalar(-0.74890095185318839);
    out[99] = tmp_c * c0;
    out[81] = tmp_c * s0;
}

NAMESPACE_END(drjit)
