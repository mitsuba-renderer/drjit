#pragma once

#include "fwd.h"
#include <drjit-core/jit.h>
#include <drjit-core/macros.h>
#include <drjit/array_traverse.h>
#include <drjit/extra.h>
#include <drjit/map.h>

// The intrusive reference counting implementation lives in drjit-extra
#if !defined(NB_INTRUSIVE_EXPORT)
#  define NB_INTRUSIVE_EXPORT DRJIT_EXTRA_EXPORT
#endif

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)
NAMESPACE_END(detail)

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4275) // non dll-interface class 'nanobind::intrusive_base' used as base
#endif

/**
 * \brief Interface for traversing C++ objects.
 *
 * This interface exposes members (Dr.Jit arrays, child objects) of classes
 * using a callback for recursive traversal. The function freezing feature
 * (``@dr.freeze``) requires this functionality to peek into otherwise opaque
 * C++ instances and observe how their contents change.
 */
struct DRJIT_EXTRA_EXPORT TraversableBase : public nanobind::intrusive_base {
    /**
     * \brief Invoke the provided \ref TraverseVisitor callback for all tracked
     * Dr.Jit arrays and TraversableBase-derived instances reachable from
     * \c this.
     *
     * See the \ref TraverseVisitor class for details on the protocol.
     */
    virtual void traverse_cb(void *payload, const TraverseVisitor &cb) = 0;
};

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

/// Helper macro for DR_TRAVERSE_CB, which names the member it traverses
#define DR_TRAVERSE_MEMBER(member)                                             \
    drjit::traverse_fn(member, payload, cb, #member);

/**
 * \brief Macro generating the implementation of the ``traverse_cb`` method
 *
 * The first argument is the base class of the current class. The remaining
 * arguments list the members to traverse. Trampoline classes of nanobind
 * bindings list no members: the Dr.Jit arrays stored in the ``__dict__`` of a
 * derived Python object are discovered by the Python-side traversal.
 */
#define DR_TRAVERSE_CB(Base, ...)                                              \
public:                                                                        \
    void traverse_cb(void *payload, const drjit::TraverseVisitor &cb)          \
        override {                                                             \
        static_assert(                                                         \
            std::is_base_of<drjit::TraversableBase,                            \
                            std::remove_pointer_t<decltype(this)>>::value);    \
        DRJIT_MARK_USED(payload);                                              \
        DRJIT_MARK_USED(cb);                                                   \
        if constexpr (!std::is_same_v<Base, drjit::TraversableBase>)           \
            Base::traverse_cb(payload, cb);                                    \
        DRJIT_MAP(DR_TRAVERSE_MEMBER, __VA_ARGS__)                             \
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
