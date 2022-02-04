/*
    tests/memory.cpp -- tests for load/store/gather/scatter, etc.

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#if defined(_MSC_VER)
#  include <windows.h>
#endif

DRJIT_TEST_ALL(test01_load) {
    alignas(alignof(T)) Value mem[Size];
    for (size_t i = 0; i < Size; ++i)
        mem[i] = (Value) i;
    assert(load_aligned<T>(mem) == arange<T>());
}

DRJIT_TEST_ALL(test02_load) {
    Value mem[Size];
    for (size_t i = 0; i < Size; ++i)
        mem[i] = (Value) i;
    assert(load<T>(mem) == arange<T>());
}

DRJIT_TEST_ALL(test03_store) {
    alignas(alignof(T)) Value mem[Size];
    store_aligned(mem, arange<T>());
    for (size_t i = 0; i < Size; ++i)
        assert(mem[i] == (Value) i);
}

DRJIT_TEST_ALL(test04_store) {
    Value mem[Size];
    store(mem, arange<T>());
    for (size_t i = 0; i < Size; ++i)
        assert(mem[i] == (Value) i);
}

DRJIT_TEST_ALL(test05_gather) {
    alignas(alignof(T)) Value mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Value) i;
    }

    auto id32 = load<Array<uint32_t, Size>>(indices32);
    auto id64 = load<Array<uint64_t, Size>>(indices64);

#if defined(_MSC_VER)
    /// MSVC doesn't seem to correctly track data dependencies involving gathers
    MemoryBarrier();
#endif

    store_aligned(dst, gather<T>(mem, id32));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Value(Size - 1 - i));
    memset(dst, 0, sizeof(Value) * Size);

#if defined(_MSC_VER)
    /// MSVC doesn't seem to correctly track data dependencies involving gathers
    MemoryBarrier();
#endif

    store_aligned(dst, gather<T>(mem, id64));

    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Value(Size - 1 - i));
    memset(dst, 0, sizeof(Value) * Size);
}

DRJIT_TEST_ALL(test06_gather_pointers) {
    using Scalar   = scalar_t<T>;
    using ValuePtr = replace_scalar_t<T, const Scalar *>;
    using UInt32P  = replace_scalar_t<T, uint32_t>;
    Scalar v1 = 1337, v2 = 42;

    // v1, v2, v1, v2, ...
    std::vector<Scalar *> pointers(Size, nullptr);
    for (size_t i = 0; i < Size; ++i)
        pointers[i] = (i % 2) == 0 ? &v1 : &v2;

    UInt32P indices;
    constexpr uint32_t middle = (Size+1) / 2;
    if constexpr (is_array_v<T> && Size > 1) {
        // v1, v1, ..., v2, v2, ...
        for (uint32_t i = 0; i < middle; ++i)
            indices[i] = i * 2;
        for (uint32_t i = middle; i < Size; ++i)
            indices[i] = (i - middle) * 2 + 1;
    } else {
        indices = 0;
    }

    auto gathered = gather<ValuePtr>(pointers.data(), indices);
    static_assert(std::is_same_v<decltype(gathered), ValuePtr>);

    if constexpr (is_array_v<T> && Size > 1)
        for (size_t i = 0; i < Size; ++i)
            assert(gathered[i] == (i < middle ? &v1 : &v2));
    else
        assert(all(gathered == &v1));
}

DRJIT_TEST_ALL(test07_gather_mask) {
    alignas(alignof(T)) Value mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Value) i;
    }

    auto id32 = load<Array<uint32_t, Size>>(indices32);
    auto id64 = load<Array<uint64_t, Size>>(indices64);
    auto idx = arange<uint_array_t<T>>();
    auto even_mask = mask_t<T>(eq(sl<1>(sr<1>(idx)), idx));

#if defined(_MSC_VER)
    /// MSVC doesn't seem to correctly track data dependencies involving gathers
    MemoryBarrier();
#endif

    memset(dst, 0, sizeof(Value) * Size);
    store_aligned(dst, gather<T>(mem, id32, even_mask));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == ((i % 2 == 0) ? Value(Size - 1 - i) : 0));
    memset(dst, 0, sizeof(Value) * Size);

#if defined(_MSC_VER)
    /// MSVC doesn't seem to correctly track data dependencies involving gathers
    MemoryBarrier();
#endif

    store_aligned(dst, gather<T>(mem, id64, even_mask));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == ((i % 2 == 0) ? Value(Size - 1 - i) : 0));
}

DRJIT_TEST_ALL(test08_scatter) {
    alignas(alignof(T)) Value mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Value) i;
    }

    auto id32 = load<Array<uint32_t, Size>>(indices32);
    auto id64 = load<Array<uint64_t, Size>>(indices64);

    memset(dst, 0, sizeof(Value) * Size);
    scatter(dst, load_aligned<T>(mem), id32);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Value(Size - 1 - i));
    memset(dst, 0, sizeof(Value) * Size);

    scatter(dst, load_aligned<T>(mem), id64);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Value(Size - 1 - i));
    memset(dst, 0, sizeof(Value) * Size);
}

DRJIT_TEST_ALL(test09_scatter_mask) {
    alignas(alignof(T)) Value mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Value) i;
    }
    auto id32 = load<Array<uint32_t, Size>>(indices32);
    auto id64 = load<Array<uint64_t, Size>>(indices64);

    auto idx = arange<uint_array_t<T>>();
    auto even_mask = mask_t<T>(eq(sl<1>(sr<1>(idx)), idx));

    memset(dst, 0, sizeof(Value) * Size);
    scatter(dst, load_aligned<T>(mem), id32, even_mask);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == (((Size-1-i) % 2 == 0) ? Value(Size - 1 - i) : 0));
    memset(dst, 0, sizeof(Value) * Size);

    scatter(dst, load_aligned<T>(mem), id64, even_mask);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == (((Size-1-i) % 2 == 0) ? Value(Size - 1 - i) : 0));
}
