#pragma once
#include "common.h"

extern void export_slice(nb::module_&);

extern std::pair<nb::tuple, nb::object>
slice_index(const nb::type_object_t<dr::ArrayBase> &dtype,
            const nb::tuple &shape, const nb::tuple &indices);

extern PyObject *mp_subscript(PyObject *self, PyObject *key);
