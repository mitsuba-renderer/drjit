/*
    detail.h -- Bindings for miscellaneous implementation details

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <drjit-core/hash.h>
#include <tsl/robin_map.h>
#include "common.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4324) // structure was padded due to alignment specifier (in TSL robin_map)
#endif

/// Helper data structure to track copies performed by \ref copy(), \ref update_indices()
struct CopyMap {
public:
    ~CopyMap() {
        clear();
    }

    void put(nb::handle k, nb::handle v) {
        k.inc_ref();
        v.inc_ref();
        map.insert({ k, v });
    }

    nb::handle get(nb::handle k) {
        auto it = map.find(k);
        if (it == map.end())
            return nb::handle();
        return it->second;
    }

    void clear() {
        for (auto [k, v] : map) {
            k.dec_ref();
            v.dec_ref();
        }
        map.clear();
    }

    struct handle_hash {
        size_t operator()(nb::handle h) const { return PointerHasher()(h.ptr()); }
    };

    struct handle_eq {
        size_t operator()(nb::handle h1, nb::handle h2) const { return h1.is(h2); }
    };

    tsl::robin_map<nb::handle, nb::handle, handle_hash, handle_eq> map;
};

/// RAII helper to temporarily stash the reference count of a Dr.Jit variable
struct StashRef {
    StashRef(uint32_t index) : handle(jit_var_stash_ref(index)) { }
    ~StashRef() { jit_var_unstash_ref(handle); }
    StashRef(StashRef &&w) : handle(w.handle) { w.handle = 0; }
    StashRef(const StashRef &) = delete;
    uint64_t handle;
};

// See misc.cpp for documentation of these functions
extern nb::object copy(nb::handle h, CopyMap *copy_map = nullptr);
extern nb::object uncopy(nb::handle h, CopyMap &copy_map);
extern void collect_indices(nb::handle, dr::dr_vector<uint64_t> &,
                            bool inc_ref = false);
extern nb::object update_indices(nb::handle, const dr::dr_vector<uint64_t> &,
                                 CopyMap *copy_map = nullptr,
                                 bool preserve_dirty = false);
extern void check_compatibility(nb::handle, nb::handle, const char *name);
extern void stash_ref(nb::handle h, std::vector<StashRef> &);

extern nb::object reset(nb::handle h);
extern void enable_py_tracing();
extern void disable_py_tracing();

extern void export_detail(nb::module_ &);
