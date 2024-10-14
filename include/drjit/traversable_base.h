#pragma once

#include "array_traverse.h"
#include "drjit-core/macros.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/intrusive/ref.h"
#include <drjit-core/jit.h>
#include <drjit/map.h>

NAMESPACE_BEGIN(drjit)

/// Interface for traversing C++ objects.
struct TraversableBase : nanobind::intrusive_base {
    virtual void traverse_1_cb_ro(void *, void (*)(void *, uint64_t)) const = 0;
    virtual void traverse_1_cb_rw(void *, uint64_t (*)(void *, uint64_t))   = 0;
};

/// Macro for generating call to traverse_1_fn_ro for a class member
#define DR_TRAVERSE_MEMBER_RO(member)                                          \
    drjit::log_member_open(false, #member);                                    \
    drjit::traverse_1_fn_ro(member, payload, fn);                              \
    drjit::log_member_close();
/// Macro for generating call to traverse_1_fn_rw for a class member
#define DR_TRAVERSE_MEMBER_RW(member)                                          \
    drjit::log_member_open(true, #member);                                     \
    drjit::traverse_1_fn_rw(member, payload, fn);                              \
    drjit::log_member_close();

inline void log_member_open(bool rw, const char *member) {
    jit_log(LogLevel::Debug, "%s%s{", rw ? "rw " : "ro ", member);
}

inline void log_member_close() { jit_log(LogLevel::Debug, "}"); }

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

/// Macro to generate traverse_1_cb_ro and traverse_1_cb_rw methods for each
/// member in the list.
#define DR_TRAVERSE_CB(Base, ...)                                              \
public:                                                                        \
    DR_TRAVERSE_CB_RO(Base, __VA_ARGS__)                                       \
    DR_TRAVERSE_CB_RW(Base, __VA_ARGS__)

#define DR_TRAMPOLINE_TRAVERSE_CB(Base)                                        \
public:                                                                        \
    void traverse_1_cb_ro(void *payload, void (*fn)(void *, uint64_t))         \
        const override {                                                       \
        if constexpr (!std ::is_same_v<Base, drjit ::TraversableBase>)         \
            Base ::traverse_1_cb_ro(payload, fn);                              \
        drjit::traverse_py_cb_ro(this, payload, fn);                           \
    }                                                                          \
    void traverse_1_cb_rw(void *payload, uint64_t (*fn)(void *, uint64_t))     \
        override {                                                             \
        if constexpr (!std ::is_same_v<Base, drjit ::TraversableBase>)         \
            Base ::traverse_1_cb_rw(payload, fn);                              \
        drjit::traverse_py_cb_rw(this, payload, fn);                           \
    }

#if defined(_MSC_VER)
#define DRJIT_EXPORT __declspec(dllexport)
#else
#define DRJIT_EXPORT __attribute__((visibility("default")))
#endif

NAMESPACE_END(drjit)
