/*
    dlpack.h -- Data exchange with other tensor frameworks

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "dlpack.h"
#include "base.h"
#include "memop.h"
#include <nanobind/ndarray.h>

nb::dlpack::dtype dlpack_dtype(VarType vt) {
    switch (vt) {
        case VarType::Bool:    return nb::dtype<bool>(); break;
        case VarType::UInt8:   return nb::dtype<uint8_t>(); break;
        case VarType::Int8:    return nb::dtype<int8_t>(); break;
        case VarType::UInt16:  return nb::dtype<uint16_t>(); break;
        case VarType::Int16:   return nb::dtype<int16_t>(); break;
        case VarType::UInt32:  return nb::dtype<uint32_t>(); break;
        case VarType::Int32:   return nb::dtype<int32_t>(); break;
        case VarType::UInt64:  return nb::dtype<uint64_t>(); break;
        case VarType::Int64:   return nb::dtype<int64_t>(); break;
        case VarType::Float32: return nb::dtype<float>(); break;
        case VarType::Float64: return nb::dtype<double>(); break;
        default:
            nb::detail::raise_type_error("Type is incompatible with DLPack.");
    }
}

static nb::ndarray<> dlpack(nb::handle_t<ArrayBase> h, bool force_cpu) {
    const ArraySupplement &s = supp(h.type());
    bool is_dynamic = false;

    if (s.is_tensor) {
        is_dynamic = true;
    } else {
        for (int i = 0; i < s.ndim; ++i)
            is_dynamic |= s.shape[i] == DRJIT_DYNAMIC;
    }

    nb::dlpack::dtype dtype = dlpack_dtype((VarType) s.type);

    dr_vector<size_t> shape;
    dr_vector<int64_t> strides;

    int32_t device_id = 0, device_type = nb::device::cpu::value;

    if (is_dynamic) {
        nb::object flat = ravel(h, 'C', &shape, &strides);
        const ArraySupplement &s2 = supp(flat.type());

        void *ptr;
        if (s2.index) {
            uint32_t index = s2.index(inst_ptr(flat));

            JitBackend backend = (JitBackend) s2.backend;

            if (force_cpu && backend == JitBackend::CUDA) {
                uint32_t index_new =
                    jit_var_migrate(index, AllocType::Host);

                nb::object tmp = nb::inst_alloc(flat.type());
                s2.init_index(index_new, inst_ptr(tmp));
                nb::inst_mark_ready(tmp);
                flat = std::move(tmp);
            }

            jit_var_eval(index);
            ptr = jit_var_ptr(index);

            if (backend == JitBackend::CUDA) {
                device_type = nb::device::cuda::value;
                device_id = jit_var_device(index);
            } else {
                jit_sync_thread();
            }
        } else {
            ptr = s2.data(inst_ptr(h));
        }

        return {
            ptr,
            shape.size(),
            shape.data(),
            flat,
            strides.data(),
            dtype,
            device_type,
            device_id
        };
    } else {
        void *ptr = s.data(inst_ptr(h));

        shape.resize(s.ndim);
        strides.resize(s.ndim);

        int64_t stride = 1;
        for (int i = s.ndim - 1; ; --i) {
            shape[i] = s.shape[i];
            strides[i] = stride;
            stride *= s.shape[i];

            // Special case: array containing 3D SIMD arrays which are 4D-aligned
            if (i == s.ndim - 1 && s.talign == 16 && s.shape[i] == 3)
                stride++;

            if (i == 0)
                break;
        }

        return {
            ptr,
            s.ndim,
            shape.data(),
            h,
            strides.data(),
            dtype,
            device_type,
            device_id
        };
    }
}

void export_dlpack(nb::module_ &) {
    nb::class_<ArrayBase> ab = nb::borrow<nb::class_<ArrayBase>>(array_base);

    ab.def("__dlpack__",
           [](nb::handle_t<ArrayBase> h) {
               return dlpack(h, false);
           }, doc_dlpack)
      .def("__array__",
           [](nb::handle_t<ArrayBase> h) {
               return nb::ndarray<nb::numpy>(dlpack(h, true).handle());
           }, doc_array);
}
