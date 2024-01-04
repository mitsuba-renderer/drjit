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
#include <drjit-core/half.h>

nb::dlpack::dtype dlpack_dtype(VarType vt) {
    using half = drjit::half;
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
        case VarType::Float16: return nb::dtype<half>(); break;
        case VarType::Float32: return nb::dtype<float>(); break;
        case VarType::Float64: return nb::dtype<double>(); break;
        default:
            nb::raise_type_error("Type is incompatible with DLPack.");
    }
}

using JitVar = drjit::JitArray<JitBackend::None, void>;

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
    void *ptr;
    nb::object owner;

    if (is_dynamic) {
        owner = ravel(h, s.is_complex ? 'F' : 'C', &shape, &strides);
        const ArraySupplement &s2 = supp(owner.type());

        if (s2.index) {
            uint32_t index = (uint32_t) s2.index(inst_ptr(owner));
            JitBackend backend = (JitBackend) s2.backend;

            JitVar value = JitVar::borrow(index);
            if (force_cpu && backend == JitBackend::CUDA)
                value = JitVar::steal(jit_var_migrate(value.index(), AllocType::Host));

            value = JitVar::steal(jit_var_data(value.index(), &ptr));

            if (value.index() != index) {
                nb::object tmp = nb::inst_alloc(owner.type());
                s2.init_index(value.index(), inst_ptr(tmp));
                nb::inst_mark_ready(tmp);
                owner = std::move(tmp);
            }

            if (backend == JitBackend::CUDA && !force_cpu) {
                device_type = nb::device::cuda::value;
                device_id = jit_var_device(index);
            } else {
                jit_sync_thread();
            }
        } else {
            ptr = s2.data(inst_ptr(h));
        }
    } else {
        owner = nb::borrow(h);
        ptr = s.data(inst_ptr(h));

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
    }

    if (s.is_complex) {
        dtype.code = (uint8_t) nb::dlpack::dtype_code::Complex;
        dtype.bits *= 2;

        for (size_t i = 1; i < shape.size(); ++i) {
            shape[i - 1] = shape[i];
            strides[i - 1] = strides[i] / 2;
        }

        shape.resize(shape.size() - 1);
        strides.resize(strides.size() - 1);
    }

    return {
        ptr,
        shape.size(),
        shape.data(),
        owner,
        strides.data(),
        dtype,
        device_type,
        device_id
    };
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
