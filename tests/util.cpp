#include "test.h"
#include <drjit/custom.h>
#include <drjit/dynamic.h>
#include <drjit/jit.h>
#include <drjit/util.h>


namespace dr = drjit;

using Float = dr::DiffArray<dr::LLVMArray<float>>;
using UInt32 = dr::DiffArray<dr::LLVMArray<uint32_t>>;

template <typename T>
void test_meshgrid2d(bool index_xy, const T &x, const T &y, size_t expected_size,
                     const T &expected_x, const T &expected_y) {
    auto [xx, yy] = dr::meshgrid(x, y, index_xy);

    assert(xx.size() == (x.size() == 1 ? 1 : expected_size));
    assert(yy.size() == (y.size() == 1 ? 1 : expected_size));
    assert(dr::all(dr::eq(xx, expected_x)));
    assert(dr::all(dr::eq(yy, expected_y)));
}

DRJIT_TEST(test01_meshgrid_2d) {
    jit_init((uint32_t) JitBackend::LLVM);
    // Results checked against `np.meshgrid`.
    for (bool indexing : {true, false}) {
        test_meshgrid2d(indexing, Float(0.f), Float(100.f), 1,
                        Float(0.f), Float(100.f));
        test_meshgrid2d(indexing, Float(0.f), Float(100.f, 150.f, 200.f), 3,
                        Float(0), Float(100, 150, 200));
    }

    test_meshgrid2d(true, Float(0.f, 1.f), Float(1.f, 2.f, 3.f, 4.f), 8,
                    Float(0, 1, 0, 1, 0, 1, 0, 1),
                    Float(1, 1, 2, 2, 3, 3, 4, 4));
    test_meshgrid2d(false, Float(0.f, 1.f), Float(1.f, 2.f, 3.f, 4.f), 8,
                    Float(0, 0, 0, 0, 1, 1, 1, 1),
                    Float(1, 2, 3, 4, 1, 2, 3, 4));
    test_meshgrid2d(true, UInt32(0, 1), UInt32(1, 2, 3, 4), 8,
                    UInt32(0, 1, 0, 1, 0, 1, 0, 1),
                    UInt32(1, 1, 2, 2, 3, 3, 4, 4));
    test_meshgrid2d(false, UInt32(0, 1), UInt32(1, 2, 3, 4), 8,
                    UInt32(0, 0, 0, 0, 1, 1, 1, 1),
                    UInt32(1, 2, 3, 4, 1, 2, 3, 4));
}


template <typename T>
void test_meshgrid3d(bool index_xy, const T &x, const T &y, const T &z, size_t expected_size,
                     const T &expected_x, const T &expected_y, const T &expected_z) {
    auto [xx, yy, zz] = dr::meshgrid(x, y, z, index_xy);
    dr::eval(xx, yy, zz);
    std::cout << index_xy << ", xx: " << to_string(xx) << std::endl;
    std::cout << index_xy << ", yy: " << to_string(yy) << std::endl;
    std::cout << index_xy << ", zz: " << to_string(zz) << std::endl;

    assert(xx.size() == expected_size || (x.size() == 1 && xx.size() == 1));
    assert(yy.size() == expected_size || (y.size() == 1 && yy.size() == 1));
    assert(zz.size() == expected_size || (z.size() == 1 && zz.size() == 1));
    assert(dr::all(dr::eq(xx, expected_x)));
    assert(dr::all(dr::eq(yy, expected_y)));
    assert(dr::all(dr::eq(zz, expected_z)));
}

DRJIT_TEST(test02_meshgrid_3d) {
    jit_init((uint32_t) JitBackend::LLVM);
    // Results checked against `np.meshgrid`.
    for (bool indexing : {true, false}) {
        test_meshgrid3d(indexing, Float(0.f), Float(1.f), Float(2.f), 1,
                        Float(0), Float(1), Float(2));
        test_meshgrid3d(indexing, UInt32(0), UInt32(1), UInt32(2, 3, 4), 3,
                        UInt32(0),
                        UInt32(1),
                        UInt32(2, 3, 4));
        test_meshgrid3d(indexing, UInt32(0), UInt32(1, 2, 3), UInt32(4), 3,
                        UInt32(0),
                        UInt32(1, 2, 3),
                        UInt32(4));
        test_meshgrid3d(indexing, UInt32(0, 1, 2), UInt32(3), UInt32(4), 3,
                        UInt32(0, 1, 2),
                        UInt32(3),
                        UInt32(4));
    }

    test_meshgrid3d(true, UInt32(0, 1), UInt32(2, 3), UInt32(4, 5), 8,
                    UInt32(0, 0, 1, 1, 0, 0, 1, 1),
                    UInt32(2, 2, 2, 2, 3, 3, 3, 3),
                    UInt32(4, 5, 4, 5, 4, 5, 4, 5));
    test_meshgrid3d(false, UInt32(0, 1), UInt32(2, 3), UInt32(4, 5), 8,
                    UInt32(0, 0, 0, 0, 1, 1, 1, 1),
                    UInt32(2, 2, 3, 3, 2, 2, 3, 3),
                    UInt32(4, 5, 4, 5, 4, 5, 4, 5));

    test_meshgrid3d(true, UInt32(0, 1), UInt32(2, 3, 4), UInt32(5), 6,
                    UInt32(0, 1, 0, 1, 0, 1),
                    UInt32(2, 2, 3, 3, 4, 4),
                    UInt32(5, 5, 5, 5, 5, 5));
    test_meshgrid3d(false, UInt32(0, 1), UInt32(2, 3, 4), UInt32(5), 6,
                    UInt32(0, 0, 0, 1, 1, 1),
                    UInt32(2, 3, 4, 2, 3, 4),
                    UInt32(5, 5, 5, 5, 5, 5));

    test_meshgrid3d(true, UInt32(0), UInt32(1, 2, 3), UInt32(4, 5), 6,
                    UInt32(0, 0, 0, 0, 0, 0),
                    UInt32(1, 1, 2, 2, 3, 3),
                    UInt32(4, 5, 4, 5, 4, 5));
    test_meshgrid3d(false, UInt32(0), UInt32(1, 2, 3), UInt32(4, 5), 6,
                    UInt32(0, 0, 0, 0, 0, 0),
                    UInt32(1, 1, 2, 2, 3, 3),
                    UInt32(4, 5, 4, 5, 4, 5));

    test_meshgrid3d(true, Float(0, 1, 2, 3), Float(4, 5), Float(6, 7, 8), 24,
                    Float(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3),
                    Float(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5),
                    Float(6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8));
    test_meshgrid3d(false, Float(0, 1, 2, 3), Float(4, 5), Float(6, 7, 8), 24,
                    Float(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3),
                    Float(4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5),
                    Float(6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8));
}
