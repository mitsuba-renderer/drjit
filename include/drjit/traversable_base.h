
#pragma once

#include "array_traverse.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/intrusive/ref.h"
#include <type_traits>
#include <vector>

NAMESPACE_BEGIN(drjit)

struct TraversableBase : nanobind::intrusive_base {
    virtual void traverse_1_cb_ro(void *, void (*)(void *, uint64_t)) const = 0;
    virtual void traverse_1_cb_rw(void *, uint64_t (*)(void *, uint64_t))   = 0;
};

template <typename T> struct is_ref_t<nanobind::ref<T>> : std::true_type {};
template <typename T> struct is_iterable_t<std::vector<T>> : std::true_type {};

#define DR_TRAVERSE_MEMBER_RO(member)                                          \
    drjit::traverse_1_fn_ro(member, payload, fn);
#define DR_TRAVERSE_MEMBER_RW(member)                                          \
    drjit::traverse_1_fn_rw(member, payload, fn);

#define DR_TRAVERSE_CB_RO(Base, ...)                                           \
    void traverse_1_cb_ro(void *payload, void (*fn)(void *, uint64_t))         \
        const override {                                                       \
        if constexpr (!std::is_same_v<Base, drjit::TraversableBase>)           \
            Base::traverse_1_cb_ro(payload, fn);                               \
        DRJIT_MAP(DR_TRAVERSE_MEMBER_RO, __VA_ARGS__)                          \
    }

#define DR_TRAVERSE_CB_RW(Base, ...)                                           \
    void traverse_1_cb_rw(void *payload, uint64_t (*fn)(void *, uint64_t))     \
        override {                                                             \
        if constexpr (!std::is_same_v<Base, drjit::TraversableBase>)           \
            Base::traverse_1_cb_rw(payload, fn);                               \
        DRJIT_MAP(DR_TRAVERSE_MEMBER_RW, __VA_ARGS__)                          \
    }

#define DR_TRAVERSE_CB(Base, ...)                                              \
public:                                                                        \
    DR_TRAVERSE_CB_RO(Base, __VA_ARGS__)                                       \
    DR_TRAVERSE_CB_RW(Base, __VA_ARGS__)

NAMESPACE_END(drjit)
