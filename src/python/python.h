#pragma once

#include <drjit/python.h>
#include <drjit/autodiff.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

using meta = dr::detail::array_metadata;
using ops = dr::detail::array_ops;
using supp = dr::detail::array_supplement;

extern nb::handle array_base;
extern nb::handle array_module;

inline bool is_drjit_type(nb::handle h) {
    return PyType_IsSubtype((PyTypeObject *) h.ptr(),
                            (PyTypeObject *) array_base.ptr());
}

inline bool is_drjit_array(nb::handle h) { return is_drjit_type(h.type()); }

extern Py_ssize_t len(PyObject *o) noexcept;
extern nb::object shape(nb::handle_of<dr::ArrayBase> h) noexcept;

extern void bind_arraybase(nb::module_ m);
extern void bind_ops(nb::module_ m);
extern void bind_traits(nb::module_ m);
extern void bind_scalar(nb::module_ &m);
extern void bind_cuda(nb::module_ &m);
extern void bind_cuda_ad(nb::module_ &m);
extern void bind_llvm(nb::module_ &m);
extern void bind_llvm_ad(nb::module_ &m);

