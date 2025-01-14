#pragma once

#include <drjit/fwd.h>
#include <drjit/array_traverse.h>
#include <drjit-core/macros.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <drjit-core/jit.h>
#include <drjit/map.h>

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)
/**
 * The callback used to traverse all jit arrays of a C++ object such as a
 * Mitsuba scene.
 *
 * \param payload:
 *     To wrap closures, a payload can be provided to the ``traverse_1_cb_ro``
 *     function, that is passed to the callback.
 *
 * \param index:
 *     A non-owning index of the traversed jit array.
 *
 * \param variant:
 *     If a ``JitArray`` has the attribute ``IsClass`` it is referring to a drjit
 *     class. When such a variable is traversed, the ``variant`` and ``domain``
 *     string of its ``CallSupport`` is provided to the callback using this
 *     argument. Otherwise the string is equal to "".
 *
 * \param domain:
 *     The domain of the ``CallSupport`` when traversing a class variable.
 */
using traverse_callback_ro = void (*)(void *payload, uint64_t index,
                                      const char *variant, const char *domain);
/**
 * The callback used to traverse and modify all jit arrays of a C++ object such
 * as a Mitsuba scene.
 *
 * \param payload:
 *     To wrap closures, a payload can be provided to the ``traverse_1_cb_ro``
 *     function, that is passed to the callback.
 *
 * \param index:
 *     A non-owning index of the traversed jit array.
 *
 * \return
 *     The new index of the traversed variable. This index is borrowed, and
 *     should therefore be non-owning.
 */
using traverse_callback_rw = uint64_t (*)(void *payload, uint64_t index);
NAMESPACE_END(detail)

/**
 * Interface for traversing C++ objects.
 *
 * This interface should be inherited by any class that can be added to the
 * registry. We try to ensure this by wrapping the function ``jit_registry_put``
 * in the function ``drjit::registry_put`` that takes a ``TraversableBase`` for
 * the pointer argument.
 */
struct DRJIT_EXPORT TraversableBase : public nanobind::intrusive_base {
    /**
     * Traverse all jit arrays in this c++ object. For every jit variable, the
     * callback should be called, with the provided payload pointer.
     *
     * \param payload:
     *    A pointer to a payload struct. The callback ``cb`` is called with this
     *    pointer.
     * \param cb:
     *    A function pointer, that is called with the ``payload`` pointer, the
     *    index of the jit variable, and optionally the domain and variant of a
     *    ``Class`` variable.
     */
    virtual void traverse_1_cb_ro(void *payload,
                                  detail::traverse_callback_ro cb) const = 0;
    /**
     * Traverse all jit arrays in this c++ object, and assign the output of the
     * callback to them. For every jit variable, the callback should be called,
     * with the provided payload pointer.
     *
     * \param payload:
     *    A pointer to a payload struct. The callback ``cb`` is called with this
     *    pointer.
     * \param cb:
     *    A function pointer, that is called with the ``payload`` pointer, the
     *    index of the jit variable, and optionally the domain and variant of a
     *    ``Class`` variable. The resulting index of calling this function
     *    pointer will be assigned to the traversed variable. The return value
     *    of the is borrowed from when overwriting assigning the traversed
     *    variable.
     */
    virtual void traverse_1_cb_rw(void *payload,
                                  detail::traverse_callback_rw cb) = 0;
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
    void traverse_1_cb_ro(void *payload,                                       \
                          drjit::detail::traverse_callback_ro fn)              \
        const override {                                                       \
        if constexpr (!std::is_same_v<Base, drjit::TraversableBase>)           \
            Base::traverse_1_cb_ro(payload, fn);                               \
        DRJIT_MAP(DR_TRAVERSE_MEMBER_RO, __VA_ARGS__)                          \
    }

#define DR_TRAVERSE_CB_RW(Base, ...)                                           \
    void traverse_1_cb_rw(void *payload,                                       \
                          drjit::detail::traverse_callback_rw fn) override {   \
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
    void traverse_1_cb_ro(void *payload,                                       \
                          drjit::detail::traverse_callback_ro fn)              \
        const override {                                                       \
        if constexpr (!std ::is_same_v<Base, drjit ::TraversableBase>)         \
            Base ::traverse_1_cb_ro(payload, fn);                              \
        drjit::traverse_py_cb_ro(this, payload, fn);                           \
    }                                                                          \
    void traverse_1_cb_rw(void *payload,                                       \
                          drjit::detail::traverse_callback_rw fn) override {   \
        if constexpr (!std ::is_same_v<Base, drjit ::TraversableBase>)         \
            Base ::traverse_1_cb_rw(payload, fn);                              \
        drjit::traverse_py_cb_rw(this, payload, fn);                           \
    }

inline uint32_t registry_put(const char *variant, const char *domain,
                             TraversableBase *ptr) {
    return jit_registry_put(variant, domain, (void *) ptr);
}

NAMESPACE_END(drjit)
