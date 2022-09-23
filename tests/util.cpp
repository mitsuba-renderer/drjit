#include "test.h"
#include <drjit/custom.h>
#include <drjit/dynamic.h>
#include <drjit/jit.h>
#include <drjit/util.h>


namespace dr = drjit;

using Float = dr::DiffArray<dr::LLVMArray<float>>;
using UInt32 = dr::DiffArray<dr::LLVMArray<uint32_t>>;

template <typename T>
void test_meshgrid2d(const T &x, const T &y, size_t expected_size,
                     const char *expected_x, const char *expected_y) {
    auto [xx, yy] = dr::meshgrid(x, y);

    assert(xx.size() == (x.size() == 1 ? 1 : expected_size));
    assert(yy.size() == (y.size() == 1 ? 1 : expected_size));
    assert(to_string(xx) == expected_x);
    assert(to_string(yy) == expected_y);
}

DRJIT_TEST(test01_meshgrid_2d) {
    jit_init((uint32_t) JitBackend::LLVM);
    // Results checked against `np.meshgrid`.
    test_meshgrid2d(Float(0.f), Float(100.f), 1,
                    "[0]", "[100]");
    test_meshgrid2d(Float(0.f), Float(100.f, 150.f, 200.f), 3,
                    "[0]", "[100, 150, 200]");
    test_meshgrid2d(Float(0.f, 1.f), Float(1.f, 2.f, 3.f, 4.f), 8,
                   "[0, 1, 0, 1, 0, 1, 0, 1]",
                   "[1, 1, 2, 2, 3, 3, 4, 4]");

    test_meshgrid2d(UInt32(0, 1), UInt32(1, 2, 3, 4), 8,
                   "[0, 1, 0, 1, 0, 1, 0, 1]",
                   "[1, 1, 2, 2, 3, 3, 4, 4]");
}


template <typename T>
void test_meshgrid3d(const T &x, const T &y, const T &z, size_t expected_size,
                     const char *expected_x, const char *expected_y, const char *expected_z) {
    auto [xx, yy, zz] = dr::meshgrid(x, y, z);

    assert(xx.size() == expected_size || (x.size() == 1 && xx.size() == 1));
    assert(yy.size() == expected_size || (y.size() == 1 && yy.size() == 1));
    assert(zz.size() == expected_size || (z.size() == 1 && zz.size() == 1));
    assert(to_string(xx) == expected_x);
    assert(to_string(yy) == expected_y);
    assert(to_string(zz) == expected_z);
}

DRJIT_TEST(test02_meshgrid_3d) {
    jit_init((uint32_t) JitBackend::LLVM);
    // Results checked against `np.meshgrid`.
    test_meshgrid3d(Float(0.f), Float(1.f), Float(2.f), 1,
                   "[0]", "[1]", "[2]");
    test_meshgrid3d(UInt32(0), UInt32(1), UInt32(2, 3, 4), 3,
                   "[0]",
                   "[1]",
                   "[2, 3, 4]");
    test_meshgrid3d(UInt32(0), UInt32(1, 2, 3), UInt32(4), 3,
                   "[0]",
                   "[1, 2, 3]",
                   "[4]");
    test_meshgrid3d(UInt32(0, 1, 2), UInt32(3), UInt32(4), 3,
                   "[0, 1, 2]",
                   "[3]",
                   "[4]");

    test_meshgrid3d(UInt32(0, 1), UInt32(2, 3), UInt32(4, 5), 8,
                   "[0, 0, 1, 1, 0, 0, 1, 1]",
                   "[2, 2, 2, 2, 3, 3, 3, 3]",
                   "[4, 5, 4, 5, 4, 5, 4, 5]");
    test_meshgrid3d(UInt32(0, 1), UInt32(2, 3, 4), UInt32(5), 6,
                   "[0, 1, 0, 1, 0, 1]",
                   "[2, 2, 3, 3, 4, 4]",
                   "[5, 5, 5, 5, 5, 5]");
    test_meshgrid3d(UInt32(0), UInt32(1, 2, 3), UInt32(4, 5), 6,
                   "[0, 0, 0, 0, 0, 0]",
                   "[1, 1, 2, 2, 3, 3]",
                   "[4, 5, 4, 5, 4, 5]");
    test_meshgrid3d(Float(0, 1, 2), Float(3, 4), Float(5, 6, 7), 18,
                    "[0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]",
                    "[3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]",
                    "[5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7]");
}
