#pragma once

#include "common.h"

enum ApplyMode {
    Normal,
    RichCompare,
    Select
};

/// Perform an arithmetic operation (e.g. addition, FMA), potentially via recursion
template <ApplyMode Mode, typename Func, typename... Args, size_t... Is>
PyObject *apply(ArrayOp op, Func func, std::index_sequence<Is...>,
                Args... args) noexcept;

/// Callback operator for 'traverse' below
struct TraverseCallback {
    virtual void operator()(nb::handle h) = 0;
};

/// Invoke the given callback on leaf elements of the pytree 'h'
extern void traverse(const char *op, TraverseCallback &callback, nb::handle h);
