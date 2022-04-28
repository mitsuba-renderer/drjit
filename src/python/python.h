#pragma once

#include <drjit/python.h>
#include <drjit/autodiff.h>
#include "docstr.h"

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

using meta = dr::detail::array_metadata;
using supp = dr::detail::array_supplement;

extern nb::handle array_base;
extern nb::handle array_module;

/// Is 'h' a Dr.Jit array type?
inline bool is_drjit_type(nb::handle h) {
    return PyType_IsSubtype((PyTypeObject *) h.ptr(),
                            (PyTypeObject *) array_base.ptr());
}

/// Is 'type(h)' a Dr.Jit array type?
inline bool is_drjit_array(nb::handle h) { return is_drjit_type(h.type()); }

// Return sequence protocol access methods for the given type
nb::detail::tuple<lenfunc, ssizeargfunc, ssizeobjargproc>
get_sq(nb::handle tp, const char *name, void *check);

extern Py_ssize_t len(PyObject *o) noexcept;
extern nb::object shape(nb::handle_of<dr::ArrayBase> h) noexcept;


/**
 * \brief Given a list of Dr.Jit arrays and scalars, determine the flavor and
 * shape of the result array and broadcast/convert everything into this form.
 *
 * \param op
 *    Name of the operation for error messages
 *
 * \param o
 *    Array of input operands of size 'n'
 *
 * \param n
 *    Number of operands
 *
 * \param select
 *    Should be 'true' if this is a drjit.select() operation, in which case the
 *    first operand will be promoted to a mask array
 */
extern bool promote(const char *op, PyObject **o, size_t n, bool select = false);


// Entry points of various parts of the bindings
extern void bind_array_builtin(nb::module_ m);
extern void bind_array_math(nb::module_ m);
extern void bind_ops(nb::module_ m);
extern void bind_traits(nb::module_ m);
extern void bind_scalar(nb::module_ &m);
extern void bind_cuda(nb::module_ &m);
extern void bind_cuda_ad(nb::module_ &m);
extern void bind_llvm(nb::module_ &m);
extern void bind_llvm_ad(nb::module_ &m);

