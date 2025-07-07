#pragma once

#include "fwd.h"
#include <drjit-core/jit.h>
#include <drjit-core/macros.h>
#include <drjit/array_traverse.h>
#include <drjit/map.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)
/**
 * \brief The callback used to traverse all JIT arrays of a C++ object.
 *
 * \param payload:
 *     To wrap closures, a payload can be provided to the ``traverse_1_cb_ro``
 *     function, that is passed to the callback.
 *
 * \param index:
 *     A non-owning index of the traversed JIT array.
 *
 * \param variant:
 *     If a ``JitArray`` has the attribute ``IsClass`` it is referring to a
 *     drjit class. When such a variable is traversed, the ``variant`` and
 *     ``domain`` string of its ``CallSupport`` is provided to the callback
 *     using this argument. Otherwise the string is equal to "".
 *
 * \param domain:
 *     The domain of the ``CallSupport`` when traversing a class variable.
 */
using traverse_callback_ro = void (*)(void *payload, uint64_t index,
                                      const char *variant, const char *domain);
/**
 * \brief The callback used to traverse and modify all JIT arrays of a C++ object.
 *
 * \param payload:
 *     To wrap closures, a payload can be provided to the ``traverse_1_cb_ro``
 *     function, that is passed to the callback.
 *
 * \param index:
 *     A non-owning index of the traversed JIT array.
 *
 * \param variant:
 *     If a ``JitArray`` has the attribute ``IsClass`` it is referring to a
 *     drjit class. When such a variable is traversed, the ``variant`` and
 *     ``domain`` string of its ``CallSupport`` is provided to the callback
 *     using this argument. Otherwise the string is equal to "".
 *
 * \param domain:
 *     The domain of the ``CallSupport`` when traversing a class variable.
 *
 * \return
 *     The new index of the traversed variable. This index is borrowed, and
 *     should therefore be non-owning.
 */
using traverse_callback_rw = uint64_t (*)(void *payload, uint64_t index,
                                          const char *variant,
                                          const char *domain);

inline void log_member_open(bool rw, const char *member) {
    DRJIT_MARK_USED(rw);
    DRJIT_MARK_USED(member);
#ifndef NDEBUG
    jit_log(LogLevel::Debug, "%s%s{", rw ? "rw " : "ro ", member);
#endif
}

inline void log_member_close() {
#ifndef NDEBUG
    jit_log(LogLevel::Debug, "}");
#endif
}

NAMESPACE_END(detail)

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4275) // non dll-interface class 'nanobind::intrusive_base' used as base
#endif

/**
 * \brief Interface for traversing C++ objects.
 *
 * This interface should be inherited by any class that can be added to the
 * registry. We try to ensure this by wrapping the function ``jit_registry_put``
 * in the function ``drjit::registry_put`` that takes a ``TraversableBase`` for
 * the pointer argument.
 */
struct DRJIT_EXTRA_EXPORT TraversableBase : public nanobind::intrusive_base {
    /**
     * \brief Traverse all JIT arrays in this c++ object. For every jit
     *     variable, the callback should be called, with the provided payload
     *     pointer.
     *
     * \param payload:
     *    A pointer to a payload struct. The callback ``cb`` is called with this
     *    pointer.
     *
     * \param cb:
     *    A function pointer, that is called with the ``payload`` pointer, the
     *    index of the jit variable, and optionally the domain and variant of a
     *    ``Class`` variable.
     */
    virtual void traverse_1_cb_ro(void *payload,
                                  detail::traverse_callback_ro cb) const = 0;

    /**
     * \brief Traverse all JIT arrays in this c++ object, and assign the output of the
     *     callback to them. For every jit variable, the callback should be called,
     *     with the provided payload pointer.
     *
     * \param payload:
     *    A pointer to a payload struct. The callback ``cb`` is called with this
     *    pointer.
     *
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

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

/**
 * \brief Macro for generating call to \c traverse_1_fn_ro for a class member.
 *
 * This is only a utility macro, for the DR_TRAVERSE_CB_RO macro. It can only be
 * used in a context, where the ``payload`` and ``fn`` variables are present.
 */
#define DR_TRAVERSE_MEMBER_RO(member)                                          \
    drjit::detail::log_member_open(false, #member);                            \
    drjit::traverse_1_fn_ro(member, payload, fn);                              \
    drjit::detail::log_member_close();

/**
 * \brief Macro for generating call to \c traverse_1_fn_rw for a class member.
 *
 * This is only a utility macro, for the DR_TRAVERSE_CB_RW macro. It can only be
 * used in a context, where the ``payload`` and ``fn`` variables are present.
 */
#define DR_TRAVERSE_MEMBER_RW(member)                                          \
    drjit::detail::log_member_open(true, #member);                             \
    drjit::traverse_1_fn_rw(member, payload, fn);                              \
    drjit::detail::log_member_close();

/**
 * \brief Macro, generating the implementation of the ``traverse_1_cb_ro``
 *     method.
 *
 * The first argument should be the base class, from which the current class
 * inherits. The other arguments should list all members of that class, which
 * are supposed to be read only traversable.
 */
#define DR_TRAVERSE_CB_RO(Base, ...)                                           \
    void traverse_1_cb_ro(void *payload,                                       \
                          drjit::detail::traverse_callback_ro fn)              \
        const override {                                                       \
        static_assert(                                                         \
            std::is_base_of<drjit::TraversableBase,                            \
                            std::remove_pointer_t<decltype(this)>>::value);    \
        if constexpr (!std::is_same_v<Base, drjit::TraversableBase>)           \
            Base::traverse_1_cb_ro(payload, fn);                               \
        DRJIT_MAP(DR_TRAVERSE_MEMBER_RO, __VA_ARGS__)                          \
    }

/**
 * \breif Macro, generating the implementation of the ``traverse_1_cb_rw``
 *     method.
 *
 * The first argument should be the base class, from which the current class
 * inherits. The other arguments should list all members of that class, which
 * are supposed to be read and write traversable.
 */
#define DR_TRAVERSE_CB_RW(Base, ...)                                           \
    void traverse_1_cb_rw(void *payload,                                       \
                          drjit::detail::traverse_callback_rw fn) override {   \
        static_assert(                                                         \
            std::is_base_of<drjit::TraversableBase,                            \
                            std::remove_pointer_t<decltype(this)>>::value);    \
        if constexpr (!std::is_same_v<Base, drjit::TraversableBase>)           \
            Base::traverse_1_cb_rw(payload, fn);                               \
        DRJIT_MAP(DR_TRAVERSE_MEMBER_RW, __VA_ARGS__)                          \
    }

/**
 * \brief Macro, generating the both the implementations of the
 *     ``traverse_1_cb_ro`` and ``traverse_1_cb_rw`` methods.
 *
 * The first argument should be the base class, from which the current class
 * inherits. The other arguments should list all members of that class, which
 * are supposed to be read and write traversable.
 */
#define DR_TRAVERSE_CB(Base, ...)                                              \
public:                                                                        \
    DR_TRAVERSE_CB_RO(Base, __VA_ARGS__)                                       \
    DR_TRAVERSE_CB_RW(Base, __VA_ARGS__)

/**
 * \brief Macro, generating the implementations of ``traverse_1_cb_ro`` and
 *     ``traverse_1_cb_rw`` of a nanobind trampoline class.
 *
 * This macro should only be instantiated on trampoline classes, that serve as
 * the base class for derived types in Python. Adding this macro to a trampoline
 * class, allows for the automatic traversal of all python members in any
 * derived python class.
 */
#define DR_TRAMPOLINE_TRAVERSE_CB(Base)                                        \
public:                                                                        \
    void traverse_1_cb_ro(void *payload,                                       \
                          drjit::detail::traverse_callback_ro fn)              \
        const override {                                                       \
        DRJIT_MARK_USED(payload);                                              \
        DRJIT_MARK_USED(fn);                                                   \
        if constexpr (!std ::is_same_v<Base, drjit ::TraversableBase>)         \
            Base::traverse_1_cb_ro(payload, fn);                               \
        drjit::traverse_py_cb_ro(this, payload, fn);                           \
    }                                                                          \
    void traverse_1_cb_rw(void *payload,                                       \
                          drjit::detail::traverse_callback_rw fn) override {   \
        DRJIT_MARK_USED(payload);                                              \
        DRJIT_MARK_USED(fn);                                                   \
        if constexpr (!std ::is_same_v<Base, drjit ::TraversableBase>)         \
            Base::traverse_1_cb_rw(payload, fn);                               \
        drjit::traverse_py_cb_rw(this, payload, fn);                           \
    }

/**
 * \brief Register a \c TraversableBase pointer with Dr.Jit's pointer registry
 *
 * This should be used instead of \c jit_registry_put, as it enforces the
 * pointers to be of type \c TraversableBase, allowing for traversal of registry
 * bound pointers.
 *
 * Dr.Jit provides a central registry that maps registered pointer values to
 * low-valued 32-bit IDs. The main application is efficient virtual function
 * dispatch via \ref jit_var_call(), through the registry could be used for
 * other applications as well.
 *
 * This function registers the specified pointer \c ptr with the registry,
 * returning the associated ID value, which is guaranteed to be unique within
 * the specified domain identified by the \c (variant, domain) strings.
 * The domain is normally an identifier that is associated with the "flavor"
 * of the pointer (e.g. instances of a particular class), and which ensures
 * that the returned ID values are as low as possible.
 *
 * Caution: for reasons of efficiency, the \c domain parameter is assumed to a
 * static constant that will remain alive. The RTTI identifier
 * <tt>typeid(MyClass).name()<tt> is a reasonable choice that satisfies this
 * requirement.
 *
 * Raises an exception when ``ptr`` is ``nullptr``, or when it has already been
 * registered with *any* domain.
 */
inline uint32_t registry_put(const char *variant, const char *domain,
                             TraversableBase *ptr) {
    return jit_registry_put(variant, domain, (void *) ptr);
}

NAMESPACE_END(drjit)
