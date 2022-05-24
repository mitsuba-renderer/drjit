#pragma once

#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <nanobind/tensor.h>
#include <vector>
#include "docstr.h"

#define DRJIT_DYNAMIC 0xFF

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

extern bool is_float_v(nb::handle h);
extern nb::handle detached_t(nb::handle h);
extern nb::handle leaf_array_t(nb::handle h);
extern nb::handle expr_t(nb::handle h0, nb::handle h1);

// Return sequence protocol access methods for the given type
nb::detail::tuple<lenfunc, ssizeargfunc, ssizeobjargproc>
get_sq(nb::handle tp, const char *name, void *check);

extern Py_ssize_t len(PyObject *o) noexcept;
extern nb::object shape(nb::handle_t<dr::ArrayBase> h) noexcept;


/// Check if the given metadata record is valid
bool meta_check(meta m);

/// Compute the metadata type of an operation combining 'a' and 'b'
extern meta meta_promote(meta a, meta b);
extern meta meta_from_builtin(PyObject *o) noexcept;

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

inline bool operator==(const dr::detail::array_metadata &a,
                       const dr::detail::array_metadata &b) {
    if (a.is_vector != b.is_vector || a.is_complex != b.is_complex ||
        a.is_quaternion != b.is_quaternion || a.is_matrix != b.is_matrix ||
        a.is_tensor != b.is_tensor || a.is_diff != b.is_diff ||
        a.is_llvm != b.is_llvm || a.is_cuda != b.is_cuda ||
        a.is_sequence != b.is_sequence || a.is_valid != b.is_valid ||
        a.type != b.type || a.ndim != b.ndim)
        return false;

    for (int i = 0; i < a.ndim; ++i) {
        if (a.shape[i] != b.shape[i])
            return false;
    }

    return true;
}

inline bool operator!=(const dr::detail::array_metadata &a,
                       const dr::detail::array_metadata &b) {
    return !operator==(a, b);
}

extern std::pair<nb::tuple, nb::object>
slice_index(const nb::type_object &dtype, const nb::tuple &shape,
            const nb::tuple &indices);

extern nb::object gather(nb::type_object dtype, nb::object source,
                         nb::object index, nb::object active);

extern nb::object ravel(nb::handle_t<dr::ArrayBase> h, char order,
                        std::vector<size_t> *shape_out = nullptr,
                        std::vector<int64_t> *strides_out = nullptr);

extern nb::object unravel(const nb::type_object_t<dr::ArrayBase> &dtype,
                          nb::handle_t<dr::ArrayBase> array, char order);

extern nb::handle reinterpret_array_t(nb::handle h, VarType vt);

extern nb::dlpack::dtype dlpack_dtype(VarType vt);

extern void meta_print(meta m);

// Entry points of various parts of the bindings
extern void bind_array_builtin(nb::module_ m);
extern void bind_array_math(nb::module_ m);
extern void bind_array_misc(nb::module_ m);
extern void bind_array_autodiff(nb::module_ m);
extern void bind_traits(nb::module_ m);
extern void bind_tensor(nb::module_ m);

extern void bind_scalar(nb::module_ &m);
extern void bind_cuda(nb::module_ &m);
extern void bind_cuda_ad(nb::module_ &m);
extern void bind_llvm(nb::module_ &m);
extern void bind_llvm_ad(nb::module_ &m);

