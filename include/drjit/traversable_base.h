#pragma once

#include "fwd.h"
#include <drjit-core/jit.h>
#include <drjit-core/macros.h>
#include <drjit/array_traverse.h>
#include <drjit/map.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

NAMESPACE_BEGIN(drjit)

struct NB_INTRUSIVE_EXPORT TraversableBase : public nanobind::intrusive_base {
};

NAMESPACE_END(drjit)
