
#pragma once


#include "nanobind/intrusive/counter.h"

NAMESPACE_BEGIN(drjit)
    
struct TraversableBase: nanobind::intrusive_base{
    virtual void traverse_1_cb_ro(void *payload, void (*fn)(void *, uint64_t)) const = 0;
    virtual void traverse_1_cb_rw(void *payload, uint64_t (*fn)(void *, uint64_t)) = 0;
};

NAMESPACE_END(drjit)
