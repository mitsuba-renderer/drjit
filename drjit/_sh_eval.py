#  drjit/_sh_eval.py -- Real spherical harmonics evaluation routines
#
#  The generated code is based on the paper `Efficient Spherical Harmonic
#  Evaluation, Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2,
#  84-90, 2013 by Peter-Pike Sloan

import drjit
from typing import TypeVar

SelfT, SelfCpT = TypeVar("SelfT"), TypeVar("SelfCpT")
ValT, ValCpT = TypeVar("ValT"), TypeVar("ValCpT")
RedT, PlainT, MaskT = TypeVar("RedT"), TypeVar("PlainT"), TypeVar("MaskT")

def sh_eval_0(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    Float = type(d.x)
    r[0] = Float(0.28209479177387814)

def sh_eval_1(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    x, y, z = d
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0

def sh_eval_2(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_c = 0.546274215296039478
    r[8] = tmp_c * c1
    r[4] = tmp_c * s1

def sh_eval_3(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_c = -0.590043589926643519
    r[15] = tmp_c * c0
    r[9] = tmp_c * s0

def sh_eval_4(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    r[20] = fma(z * 1.98431348329844304, r[12], r[6] * -1.00623058987490532)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    tmp_a = z * fma(z2, -4.6833258049010249, 2.00713963067186763)
    r[21] = tmp_a * c0
    r[19] = tmp_a * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    tmp_c = fma(z2, 3.31161143515146028, -0.473087347878779985)
    r[22] = tmp_c * c1
    r[18] = tmp_c * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.590043589926643519
    r[15] = tmp_a * c0
    r[9] = tmp_a * s0
    tmp_b = z * -1.77013076977993067
    r[23] = tmp_b * c0
    r[17] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_c = 0.625835735449176256
    r[24] = tmp_c * c1
    r[16] = tmp_c * s1

def sh_eval_5(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    r[20] = fma(z * 1.98431348329844304, r[12], r[6] * -1.00623058987490532)
    r[30] = fma(z * 1.98997487421323993, r[20], r[12] * -1.00285307284481395)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    tmp_a = z * fma(z2, -4.6833258049010249, 2.00713963067186763)
    r[21] = tmp_a * c0
    r[19] = tmp_a * s0
    tmp_b = fma(z * 2.03100960115899021, tmp_a, tmp_c * -0.991031208965114985)
    r[31] = tmp_b * c0
    r[29] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    tmp_c = fma(z2, 3.31161143515146028, -0.473087347878779985)
    r[22] = tmp_c * c1
    r[18] = tmp_c * s1
    tmp_a = z * fma(z2, 7.19030517745998665, -2.39676839248666207)
    r[32] = tmp_a * c1
    r[28] = tmp_a * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.590043589926643519
    r[15] = tmp_a * c0
    r[9] = tmp_a * s0
    tmp_b = z * -1.77013076977993067
    r[23] = tmp_b * c0
    r[17] = tmp_b * s0
    tmp_c = fma(z2, -4.40314469491725369, 0.48923829943525049)
    r[33] = tmp_c * c0
    r[27] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.625835735449176256
    r[24] = tmp_a * c1
    r[16] = tmp_a * s1
    tmp_b = z * 2.07566231488104114
    r[34] = tmp_b * c1
    r[26] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_c = -0.656382056840170258
    r[35] = tmp_c * c0
    r[25] = tmp_c * s0

def sh_eval_6(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    r[20] = fma(z * 1.98431348329844304, r[12], r[6] * -1.00623058987490532)
    r[30] = fma(z * 1.98997487421323993, r[20], r[12] * -1.00285307284481395)
    r[42] = fma(z * 1.99304345718356646, r[30], r[20] * -1.00154202096221923)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    tmp_a = z * fma(z2, -4.6833258049010249, 2.00713963067186763)
    r[21] = tmp_a * c0
    r[19] = tmp_a * s0
    tmp_b = fma(z * 2.03100960115899021, tmp_a, tmp_c * -0.991031208965114985)
    r[31] = tmp_b * c0
    r[29] = tmp_b * s0
    tmp_c = fma(z * 2.02131498923702768, tmp_b, tmp_a * -0.995226703056238504)
    r[43] = tmp_c * c0
    r[41] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    tmp_c = fma(z2, 3.31161143515146028, -0.473087347878779985)
    r[22] = tmp_c * c1
    r[18] = tmp_c * s1
    tmp_a = z * fma(z2, 7.19030517745998665, -2.39676839248666207)
    r[32] = tmp_a * c1
    r[28] = tmp_a * s1
    tmp_b = fma(z * 2.11394181566096995, tmp_a, tmp_c * -0.973610120462326756)
    r[44] = tmp_b * c1
    r[40] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.590043589926643519
    r[15] = tmp_a * c0
    r[9] = tmp_a * s0
    tmp_b = z * -1.77013076977993067
    r[23] = tmp_b * c0
    r[17] = tmp_b * s0
    tmp_c = fma(z2, -4.40314469491725369, 0.48923829943525049)
    r[33] = tmp_c * c0
    r[27] = tmp_c * s0
    tmp_a = z * fma(z2, -10.1332578546641603, 2.76361577854477058)
    r[45] = tmp_a * c0
    r[39] = tmp_a * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.625835735449176256
    r[24] = tmp_a * c1
    r[16] = tmp_a * s1
    tmp_b = z * 2.07566231488104114
    r[34] = tmp_b * c1
    r[26] = tmp_b * s1
    tmp_c = fma(z2, 5.55021390801596581, -0.504564900728724064)
    r[46] = tmp_c * c1
    r[38] = tmp_c * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.656382056840170258
    r[35] = tmp_a * c0
    r[25] = tmp_a * s0
    tmp_b = z * -2.3666191622317525
    r[47] = tmp_b * c0
    r[37] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_c = 0.683184105191914415
    r[48] = tmp_c * c1
    r[36] = tmp_c * s1

def sh_eval_7(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    r[20] = fma(z * 1.98431348329844304, r[12], r[6] * -1.00623058987490532)
    r[30] = fma(z * 1.98997487421323993, r[20], r[12] * -1.00285307284481395)
    r[42] = fma(z * 1.99304345718356646, r[30], r[20] * -1.00154202096221923)
    r[56] = fma(z * 1.99489143482413467, r[42], r[30] * -1.00092721392195827)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    tmp_a = z * fma(z2, -4.6833258049010249, 2.00713963067186763)
    r[21] = tmp_a * c0
    r[19] = tmp_a * s0
    tmp_b = fma(z * 2.03100960115899021, tmp_a, tmp_c * -0.991031208965114985)
    r[31] = tmp_b * c0
    r[29] = tmp_b * s0
    tmp_c = fma(z * 2.02131498923702768, tmp_b, tmp_a * -0.995226703056238504)
    r[43] = tmp_c * c0
    r[41] = tmp_c * s0
    tmp_a = fma(z * 2.01556443707463773, tmp_c, tmp_b * -0.99715504402183186)
    r[57] = tmp_a * c0
    r[55] = tmp_a * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    tmp_c = fma(z2, 3.31161143515146028, -0.473087347878779985)
    r[22] = tmp_c * c1
    r[18] = tmp_c * s1
    tmp_a = z * fma(z2, 7.19030517745998665, -2.39676839248666207)
    r[32] = tmp_a * c1
    r[28] = tmp_a * s1
    tmp_b = fma(z * 2.11394181566096995, tmp_a, tmp_c * -0.973610120462326756)
    r[44] = tmp_b * c1
    r[40] = tmp_b * s1
    tmp_c = fma(z * 2.08166599946613307, tmp_b, tmp_a * -0.984731927834661791)
    r[58] = tmp_c * c1
    r[54] = tmp_c * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.590043589926643519
    r[15] = tmp_a * c0
    r[9] = tmp_a * s0
    tmp_b = z * -1.77013076977993067
    r[23] = tmp_b * c0
    r[17] = tmp_b * s0
    tmp_c = fma(z2, -4.40314469491725369, 0.48923829943525049)
    r[33] = tmp_c * c0
    r[27] = tmp_c * s0
    tmp_a = z * fma(z2, -10.1332578546641603, 2.76361577854477058)
    r[45] = tmp_a * c0
    r[39] = tmp_a * s0
    tmp_b = fma(z * 2.20794021658196149, tmp_a, tmp_c * -0.95940322360024699)
    r[59] = tmp_b * c0
    r[53] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.625835735449176256
    r[24] = tmp_a * c1
    r[16] = tmp_a * s1
    tmp_b = z * 2.07566231488104114
    r[34] = tmp_b * c1
    r[26] = tmp_b * s1
    tmp_c = fma(z2, 5.55021390801596581, -0.504564900728724064)
    r[46] = tmp_c * c1
    r[38] = tmp_c * s1
    tmp_a = z * fma(z2, 13.4918050467267694, -3.11349347232156193)
    r[60] = tmp_a * c1
    r[52] = tmp_a * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.656382056840170258
    r[35] = tmp_a * c0
    r[25] = tmp_a * s0
    tmp_b = z * -2.3666191622317525
    r[47] = tmp_b * c0
    r[37] = tmp_b * s0
    tmp_c = fma(z2, -6.7459025233633847, 0.518915578720260395)
    r[61] = tmp_c * c0
    r[51] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.683184105191914415
    r[48] = tmp_a * c1
    r[36] = tmp_a * s1
    tmp_b = z * 2.64596066180190048
    r[62] = tmp_b * c1
    r[50] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_c = -0.707162732524596271
    r[63] = tmp_c * c0
    r[49] = tmp_c * s0

def sh_eval_8(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    r[20] = fma(z * 1.98431348329844304, r[12], r[6] * -1.00623058987490532)
    r[30] = fma(z * 1.98997487421323993, r[20], r[12] * -1.00285307284481395)
    r[42] = fma(z * 1.99304345718356646, r[30], r[20] * -1.00154202096221923)
    r[56] = fma(z * 1.99489143482413467, r[42], r[30] * -1.00092721392195827)
    r[72] = fma(z * 1.9960899278339137, r[56], r[42] * -1.00060078106951478)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    tmp_a = z * fma(z2, -4.6833258049010249, 2.00713963067186763)
    r[21] = tmp_a * c0
    r[19] = tmp_a * s0
    tmp_b = fma(z * 2.03100960115899021, tmp_a, tmp_c * -0.991031208965114985)
    r[31] = tmp_b * c0
    r[29] = tmp_b * s0
    tmp_c = fma(z * 2.02131498923702768, tmp_b, tmp_a * -0.995226703056238504)
    r[43] = tmp_c * c0
    r[41] = tmp_c * s0
    tmp_a = fma(z * 2.01556443707463773, tmp_c, tmp_b * -0.99715504402183186)
    r[57] = tmp_a * c0
    r[55] = tmp_a * s0
    tmp_b = fma(z * 2.01186954040739119, tmp_a, tmp_c * -0.998166817890174474)
    r[73] = tmp_b * c0
    r[71] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    tmp_c = fma(z2, 3.31161143515146028, -0.473087347878779985)
    r[22] = tmp_c * c1
    r[18] = tmp_c * s1
    tmp_a = z * fma(z2, 7.19030517745998665, -2.39676839248666207)
    r[32] = tmp_a * c1
    r[28] = tmp_a * s1
    tmp_b = fma(z * 2.11394181566096995, tmp_a, tmp_c * -0.973610120462326756)
    r[44] = tmp_b * c1
    r[40] = tmp_b * s1
    tmp_c = fma(z * 2.08166599946613307, tmp_b, tmp_a * -0.984731927834661791)
    r[58] = tmp_c * c1
    r[54] = tmp_c * s1
    tmp_a = fma(z * 2.06155281280883029, tmp_c, tmp_b * -0.990337937660287326)
    r[74] = tmp_a * c1
    r[70] = tmp_a * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.590043589926643519
    r[15] = tmp_a * c0
    r[9] = tmp_a * s0
    tmp_b = z * -1.77013076977993067
    r[23] = tmp_b * c0
    r[17] = tmp_b * s0
    tmp_c = fma(z2, -4.40314469491725369, 0.48923829943525049)
    r[33] = tmp_c * c0
    r[27] = tmp_c * s0
    tmp_a = z * fma(z2, -10.1332578546641603, 2.76361577854477058)
    r[45] = tmp_a * c0
    r[39] = tmp_a * s0
    tmp_b = fma(z * 2.20794021658196149, tmp_a, tmp_c * -0.95940322360024699)
    r[59] = tmp_b * c0
    r[53] = tmp_b * s0
    tmp_c = fma(z * 2.15322168769582012, tmp_b, tmp_a * -0.975217386560017774)
    r[75] = tmp_c * c0
    r[69] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.625835735449176256
    r[24] = tmp_a * c1
    r[16] = tmp_a * s1
    tmp_b = z * 2.07566231488104114
    r[34] = tmp_b * c1
    r[26] = tmp_b * s1
    tmp_c = fma(z2, 5.55021390801596581, -0.504564900728724064)
    r[46] = tmp_c * c1
    r[38] = tmp_c * s1
    tmp_a = z * fma(z2, 13.4918050467267694, -3.11349347232156193)
    r[60] = tmp_a * c1
    r[52] = tmp_a * s1
    tmp_b = fma(z * 2.30488611432322132, tmp_a, tmp_c * -0.948176387355465389)
    r[76] = tmp_b * c1
    r[68] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.656382056840170258
    r[35] = tmp_a * c0
    r[25] = tmp_a * s0
    tmp_b = z * -2.3666191622317525
    r[47] = tmp_b * c0
    r[37] = tmp_b * s0
    tmp_c = fma(z2, -6.7459025233633847, 0.518915578720260395)
    r[61] = tmp_c * c0
    r[51] = tmp_c * s0
    tmp_a = z * fma(z2, -17.2495531104905417, 3.44991062209810817)
    r[77] = tmp_a * c0
    r[67] = tmp_a * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.683184105191914415
    r[48] = tmp_a * c1
    r[36] = tmp_a * s1
    tmp_b = z * 2.64596066180190048
    r[62] = tmp_b * c1
    r[50] = tmp_b * s1
    tmp_c = fma(z2, 7.98499149089313942, -0.532332766059542606)
    r[78] = tmp_c * c1
    r[66] = tmp_c * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.707162732524596271
    r[63] = tmp_a * c0
    r[49] = tmp_a * s0
    tmp_b = z * -2.91570664069931995
    r[79] = tmp_b * c0
    r[65] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_c = 0.728926660174829988
    r[80] = tmp_c * c1
    r[64] = tmp_c * s1

def sh_eval_9(d: drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], r: list[ValT]) -> None:
    from drjit import fma
    x, y, z = d
    z2 = z*z
    Float = type(x)

    r[0] = Float(0.28209479177387814)
    r[2] = z * 0.488602511902919923
    r[6] = fma(z2, 0.94617469575756008, -0.315391565252520045)
    r[12] = z * fma(z2, 1.865881662950577, -1.1195289977703462)
    r[20] = fma(z * 1.98431348329844304, r[12], r[6] * -1.00623058987490532)
    r[30] = fma(z * 1.98997487421323993, r[20], r[12] * -1.00285307284481395)
    r[42] = fma(z * 1.99304345718356646, r[30], r[20] * -1.00154202096221923)
    r[56] = fma(z * 1.99489143482413467, r[42], r[30] * -1.00092721392195827)
    r[72] = fma(z * 1.9960899278339137, r[56], r[42] * -1.00060078106951478)
    r[90] = fma(z * 1.99691119506793657, r[72], r[56] * -1.0004114379931337)
    c0 = x
    s0 = y

    tmp_a = -0.488602511902919978
    r[3] = tmp_a * c0
    r[1] = tmp_a * s0
    tmp_b = z * -1.09254843059207896
    r[7] = tmp_b * c0
    r[5] = tmp_b * s0
    tmp_c = fma(z2, -2.28522899732232876, 0.457045799464465774)
    r[13] = tmp_c * c0
    r[11] = tmp_c * s0
    tmp_a = z * fma(z2, -4.6833258049010249, 2.00713963067186763)
    r[21] = tmp_a * c0
    r[19] = tmp_a * s0
    tmp_b = fma(z * 2.03100960115899021, tmp_a, tmp_c * -0.991031208965114985)
    r[31] = tmp_b * c0
    r[29] = tmp_b * s0
    tmp_c = fma(z * 2.02131498923702768, tmp_b, tmp_a * -0.995226703056238504)
    r[43] = tmp_c * c0
    r[41] = tmp_c * s0
    tmp_a = fma(z * 2.01556443707463773, tmp_c, tmp_b * -0.99715504402183186)
    r[57] = tmp_a * c0
    r[55] = tmp_a * s0
    tmp_b = fma(z * 2.01186954040739119, tmp_a, tmp_c * -0.998166817890174474)
    r[73] = tmp_b * c0
    r[71] = tmp_b * s0
    tmp_c = fma(z * 2.00935312974101166, tmp_b, tmp_a * -0.998749217771908837)
    r[91] = tmp_c * c0
    r[89] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.546274215296039478
    r[8] = tmp_a * c1
    r[4] = tmp_a * s1
    tmp_b = z * 1.44530572132027735
    r[14] = tmp_b * c1
    r[10] = tmp_b * s1
    tmp_c = fma(z2, 3.31161143515146028, -0.473087347878779985)
    r[22] = tmp_c * c1
    r[18] = tmp_c * s1
    tmp_a = z * fma(z2, 7.19030517745998665, -2.39676839248666207)
    r[32] = tmp_a * c1
    r[28] = tmp_a * s1
    tmp_b = fma(z * 2.11394181566096995, tmp_a, tmp_c * -0.973610120462326756)
    r[44] = tmp_b * c1
    r[40] = tmp_b * s1
    tmp_c = fma(z * 2.08166599946613307, tmp_b, tmp_a * -0.984731927834661791)
    r[58] = tmp_c * c1
    r[54] = tmp_c * s1
    tmp_a = fma(z * 2.06155281280883029, tmp_c, tmp_b * -0.990337937660287326)
    r[74] = tmp_a * c1
    r[70] = tmp_a * s1
    tmp_b = fma(z * 2.04812235835781919, tmp_a, tmp_c * -0.993485272670404207)
    r[92] = tmp_b * c1
    r[88] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.590043589926643519
    r[15] = tmp_a * c0
    r[9] = tmp_a * s0
    tmp_b = z * -1.77013076977993067
    r[23] = tmp_b * c0
    r[17] = tmp_b * s0
    tmp_c = fma(z2, -4.40314469491725369, 0.48923829943525049)
    r[33] = tmp_c * c0
    r[27] = tmp_c * s0
    tmp_a = z * fma(z2, -10.1332578546641603, 2.76361577854477058)
    r[45] = tmp_a * c0
    r[39] = tmp_a * s0
    tmp_b = fma(z * 2.20794021658196149, tmp_a, tmp_c * -0.95940322360024699)
    r[59] = tmp_b * c0
    r[53] = tmp_b * s0
    tmp_c = fma(z * 2.15322168769582012, tmp_b, tmp_a * -0.975217386560017774)
    r[75] = tmp_c * c0
    r[69] = tmp_c * s0
    tmp_a = fma(z * 2.11804417118980526, tmp_c, tmp_b * -0.983662844979209416)
    r[93] = tmp_a * c0
    r[87] = tmp_a * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.625835735449176256
    r[24] = tmp_a * c1
    r[16] = tmp_a * s1
    tmp_b = z * 2.07566231488104114
    r[34] = tmp_b * c1
    r[26] = tmp_b * s1
    tmp_c = fma(z2, 5.55021390801596581, -0.504564900728724064)
    r[46] = tmp_c * c1
    r[38] = tmp_c * s1
    tmp_a = z * fma(z2, 13.4918050467267694, -3.11349347232156193)
    r[60] = tmp_a * c1
    r[52] = tmp_a * s1
    tmp_b = fma(z * 2.30488611432322132, tmp_a, tmp_c * -0.948176387355465389)
    r[76] = tmp_b * c1
    r[68] = tmp_b * s1
    tmp_c = fma(z * 2.22917715070623501, tmp_b, tmp_a * -0.967152839723182112)
    r[94] = tmp_c * c1
    r[86] = tmp_c * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.656382056840170258
    r[35] = tmp_a * c0
    r[25] = tmp_a * s0
    tmp_b = z * -2.3666191622317525
    r[47] = tmp_b * c0
    r[37] = tmp_b * s0
    tmp_c = fma(z2, -6.7459025233633847, 0.518915578720260395)
    r[61] = tmp_c * c0
    r[51] = tmp_c * s0
    tmp_a = z * fma(z2, -17.2495531104905417, 3.44991062209810817)
    r[77] = tmp_a * c0
    r[67] = tmp_a * s0
    tmp_b = fma(z * 2.40163634692206163, tmp_a, tmp_c * -0.939224604204370817)
    r[95] = tmp_b * c0
    r[85] = tmp_b * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.683184105191914415
    r[48] = tmp_a * c1
    r[36] = tmp_a * s1
    tmp_b = z * 2.64596066180190048
    r[62] = tmp_b * c1
    r[50] = tmp_b * s1
    tmp_c = fma(z2, 7.98499149089313942, -0.532332766059542606)
    r[78] = tmp_c * c1
    r[66] = tmp_c * s1
    tmp_a = z * fma(z2, 21.3928901909086377, -3.77521591604270101)
    r[96] = tmp_a * c1
    r[84] = tmp_a * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_a = -0.707162732524596271
    r[63] = tmp_a * c0
    r[49] = tmp_a * s0
    tmp_b = z * -2.91570664069931995
    r[79] = tmp_b * c0
    r[65] = tmp_b * s0
    tmp_c = fma(z2, -9.26339318284890467, 0.544905481344053255)
    r[97] = tmp_c * c0
    r[83] = tmp_c * s0
    c1 = fma(x, c0, -y * s0)
    s1 = fma(x, s0, y * c0)

    tmp_a = 0.728926660174829988
    r[80] = tmp_a * c1
    r[64] = tmp_a * s1
    tmp_b = z * 3.17731764895469793
    r[98] = tmp_b * c1
    r[82] = tmp_b * s1
    c0 = fma(x, c1, -y * s1)
    s0 = fma(x, s1, y * c1)

    tmp_c = -0.74890095185318839
    r[99] = tmp_c * c0
    r[81] = tmp_c * s0
