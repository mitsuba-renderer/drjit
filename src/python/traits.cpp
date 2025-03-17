/*
    traits.cpp -- implementation of Dr.Jit type traits such as
    is_array_v, uint32_array_t, etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "traits.h"
#include "base.h"
#include "meta.h"

nb::handle scalar_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    while (is_drjit_type(tp))
        tp = supp(tp).value;
    return tp;
}

nb::handle leaf_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        do {
            nb::handle tp2 = supp(tp).value;
            if (!is_drjit_type(tp2))
                break;
            tp = tp2;
        } while (true);

        return supp(tp).array;
    }
    return tp;
}

static size_t itemsize_v(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp))
        return jit_type_size((VarType) supp(tp).type);
    throw nb::type_error("Unsupported input type!");
}

nb::object reinterpret_array_t(nb::handle h, VarType vt) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        ArrayMeta m = supp(tp);
        m.type = (uint16_t) vt;
        tp = meta_get_type(m);
    } else {
        if (vt == VarType::Bool)
            tp = &PyBool_Type;
        else if (vt == VarType::Float16 ||
                 vt == VarType::Float32 ||
                 vt == VarType::Float64)
            tp = &PyFloat_Type;
        else
            tp = &PyLong_Type;
    }
    return borrow(tp);
}

nb::object expr_t(nb::handle h0, nb::handle h1) {
    nb::handle tp0 = h0.is_type() ? h0 : h0.type(),
               tp1 = h1.is_type() ? h1 : h1.type();

    if (tp0.is(tp1))
        return nb::borrow(tp0);

    ArrayMeta m0 = meta_get_general(h0),
              m1 = meta_get_general(h1),
              m  = meta_promote(m0, m1);

    if ((VarType) m.type == VarType::BaseFloat)
        m.type = (uint32_t) VarType::Float32;

    if (!meta_check(m))
        nb::raise_type_error(
            "drjit.expr_t(): incompatible types \"%s\" and \"%s\"",
            nb::type_name(tp0).c_str(), nb::type_name(tp1).c_str());

    return nb::borrow(meta_get_type(m));
}

nb::type_object value_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    return nb::borrow<nb::type_object>(
        is_drjit_type(tp) ? supp(tp).value : tp);
}

nb::object expr_t(nb::args args) {
    nb::object result = nb::none();
    size_t ctr = 0;

    for (nb::handle h : args) {
        if (ctr++ == 0)
            result = nb::borrow(args[0]);
        else
            result = expr_t(result, h);
    }

    return result;
}

bool is_special_v(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        const ArrayMeta &m = supp(tp);
        return m.is_complex || m.is_quaternion || m.is_matrix;
    }
    return false;
}

bool is_matrix_v(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    return is_drjit_type(tp) ? supp(tp).is_matrix : false;
}

bool is_complex_v(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    return is_drjit_type(tp) ? supp(tp).is_complex : false;
}

bool is_quaternion_v(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    return is_drjit_type(tp) ? supp(tp).is_quaternion : false;
}


nb::object tensor_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        ArrayMeta m2 { }, m = supp(tp);
        m2.type = m.type;
        m2.backend = m.backend;
        m2.is_diff = m.is_diff;
        m2.is_tensor = true;
        m2.is_valid = true;
        return nb::borrow(meta_get_type(m2));
    }
    return nb::none();
}

nb::object matrix_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        ArrayMeta m = supp(tp);
        if (m.is_matrix || m.is_vector) {
            m.is_vector = false;
            m.is_matrix = true;
            // Make sure corresponding Matrix binding is available
            nb::handle t = meta_get_type(m, false);
            if (t.is_valid())
                return nb::borrow(t);
        }
    }
    return nb::none();
}

void export_traits(nb::module_ &m) {
    m.attr("Dynamic") = -1;

    m.def("value_t", value_t , doc_value_t);

    m.def("mask_t",
          [](nb::handle h) -> nb::handle {
              nb::handle tp = h.is_type() ? h : h.type();
              return is_drjit_type(tp) ? supp(tp).mask
                                       : (PyObject *) &PyBool_Type;
          }, doc_mask_t);

    m.def("array_t",
          [](nb::handle h) -> nb::handle {
              nb::handle tp = h.is_type() ? h : h.type();
              return is_drjit_type(tp) ? supp(tp).array : tp.ptr();
          }, doc_array_t);

    m.def("scalar_t", scalar_t, doc_scalar_t);
    m.def("leaf_t", leaf_t, doc_leaf_t);

    m.def("is_array_v",
          [](nb::handle h) -> bool {
              return is_drjit_type(h.is_type() ? h : h.type());
          }, doc_is_array_v, nb::arg().none());

    m.def("size_v",
          [](nb::handle h) -> Py_ssize_t {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  Py_ssize_t shape =
                      supp(tp).shape[0];
                  if (shape == DRJIT_DYNAMIC)
                      shape = -1;
                  return shape;
              } else {
                  return 1;
              }
          }, doc_size_v);

    m.def("is_jit_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  JitBackend backend =
                      (JitBackend) supp(tp).backend;
                  return backend != JitBackend::None;
              }
              return false;
          }, doc_is_jit_v);

    m.def("type_v",
          [](nb::handle h) {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp))
                  return (VarType) supp(tp).type;
              return VarType::Void;
          }, doc_type_v);

    m.def("replace_type_t",
          [](nb::handle h, VarType vt) {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  ArrayMeta m = supp(tp);
                  m.type = (uint16_t) vt;
                  tp = meta_get_type(m);
              }
              return nb::borrow(tp);
          }, doc_replace_type_t);

    m.def("backend_v",
          [](nb::handle h) {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp))
                  return (JitBackend) supp(tp).backend;
              return JitBackend::None;
          }, doc_backend_v);

    m.def("is_dynamic_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  const ArraySupplement &s = supp(tp);
                  if (s.is_tensor)
                      return true;
                  for (int i = 0; i < s.ndim; ++i) {
                      if (s.shape[i] == DRJIT_DYNAMIC)
                          return true;
                  }
              }
              return false;
          }, doc_is_dynamic_v);

    m.def("is_tensor_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              return is_drjit_type(tp) ? supp(tp).is_tensor : false;
          }, doc_is_tensor_v);

    m.def("is_complex_v", &is_complex_v, doc_is_complex_v);
    m.def("is_quaternion_v", &is_quaternion_v, doc_is_quaternion_v);
    m.def("is_matrix_v", is_matrix_v, doc_is_matrix_v);

    m.def("is_special_v", is_special_v, doc_is_special_v);

    m.def("is_vector_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              return is_drjit_type(tp) ? supp(tp).is_vector : false;
          }, doc_is_vector_v);

    m.def("depth_v",
          [](nb::handle h) -> size_t {
              nb::handle tp = h.is_type() ? h : h.type();
              return is_drjit_type(tp) ? supp(tp).ndim : 0;
          }, doc_depth_v);

    m.def("is_mask_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              return is_drjit_type(tp) ? (VarType) supp(tp).type == VarType::Bool
                                       : tp.is(&PyBool_Type);
          }, doc_is_mask_v);

    m.def("is_float_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  VarType vt = (VarType) supp(tp).type;
                  return vt == VarType::Float16 || vt == VarType::Float32 ||
                         vt == VarType::Float64;
              } else {
                  return tp.is(&PyFloat_Type);
              }
          }, doc_is_float_v);

    m.def("is_half_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  VarType vt = (VarType) supp(tp).type;
                  return vt == VarType::Float16;
              } else {
                  return tp.is(&PyFloat_Type);
              }
          }, doc_is_half_v);

    m.def("is_arithmetic_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  VarType vt = (VarType) supp(tp).type;
                  return vt != VarType::Bool && vt != VarType::Pointer;
              } else {
                  return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type);
              }
          }, doc_is_arithmetic_v);

    m.def("is_integral_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  VarType vt = (VarType) supp(tp).type;
                  return vt == VarType::Int8 || vt == VarType::Int16 ||
                         vt == VarType::Int32 || vt == VarType::Int64 ||
                         vt == VarType::UInt8 || vt == VarType::UInt16 ||
                         vt == VarType::UInt32 || vt == VarType::UInt64;
              } else {
                  return tp.is(&PyLong_Type);
              }
          }, doc_is_integral_v);

    m.def("is_signed_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  VarType vt = (VarType) supp(tp).type;
                  return vt == VarType::Int8 || vt == VarType::Int16 ||
                         vt == VarType::Int32 || vt == VarType::Int64 ||
                         vt == VarType::Float16 || vt == VarType::Float32 ||
                         vt == VarType::Float64;
              } else {
                  return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type);
              }
          }, doc_is_signed_v);

    m.def("is_unsigned_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  VarType vt = (VarType) supp(tp).type;
                  return vt == VarType::UInt8 || vt == VarType::UInt16 ||
                         vt == VarType::UInt32 || vt == VarType::UInt64 ||
                         vt == VarType::Bool;
              } else {
                  return tp.is(&PyBool_Type);
              }
          }, doc_is_unsigned_v);

    m.def("is_diff_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp))
                  return supp(tp).is_diff;
              else
                  return false;
          }, doc_is_diff_v);

    m.def("itemsize_v", &itemsize_v, doc_itemsize_v);

    m.def("tensor_t", &tensor_t, doc_tensor_t);

    m.def("matrix_t", &matrix_t, doc_matrix_t);

    m.def("reinterpret_array_t",
          [](nb::handle h, VarType vt) { return reinterpret_array_t(h, vt); },
          doc_reinterpret_array_t);

    m.def("uint32_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::UInt32); },
          doc_uint32_array_t);

    m.def("int32_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::Int32); },
          doc_int32_array_t);

    m.def("uint64_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::UInt64); },
          doc_uint64_array_t);

    m.def("int64_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::Int64); },
          doc_int64_array_t);

    m.def("float16_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::Float16); },
          doc_float16_array_t);

    m.def("float32_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::Float32); },
          doc_float32_array_t);

    m.def("float64_array_t",
          [](nb::handle h) { return reinterpret_array_t(h, VarType::Float64); },
          doc_float64_array_t);

    m.def("uint_array_t",
          [](nb::handle h) {
              VarType vt;
              switch (itemsize_v(h)) {
                  case 1: vt = VarType::UInt8; break;
                  case 2: vt = VarType::UInt16; break;
                  case 4: vt = VarType::UInt32; break;
                  case 8: vt = VarType::UInt64; break;
                  default: throw nb::type_error("Unsupported input type!");
              }
              return reinterpret_array_t(h, vt);
          }, doc_uint_array_t);

    m.def("int_array_t",
          [](nb::handle h) {
              VarType vt;
              switch (itemsize_v(h)) {
                  case 1: vt = VarType::Int8; break;
                  case 2: vt = VarType::Int16; break;
                  case 4: vt = VarType::Int32; break;
                  case 8: vt = VarType::Int64; break;
                  default: throw nb::type_error("Unsupported input type!");
              }
              return reinterpret_array_t(h, vt);
          }, doc_int_array_t);

    m.def("float_array_t",
          [](nb::handle h) {
              VarType vt;
              switch (itemsize_v(h)) {
                  case 2: vt = VarType::Float16; break;
                  case 4: vt = VarType::Float32; break;
                  case 8: vt = VarType::Float64; break;
                  default: throw nb::type_error("Unsupported input type!");
              }
              return reinterpret_array_t(h, vt);
          }, doc_float_array_t);

    m.def("is_struct_v",
          [](nb::handle h) -> bool {
              nb::handle tp = h.is_type() ? h : h.type();
              return nb::hasattr(tp, "DRJIT_STRUCT");
          }, doc_is_struct_v);

    m.def("diff_array_t",
          [](nb::handle h) {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  ArrayMeta m = supp(tp);
                  m.is_diff = true;
                  tp = meta_get_type(m);
              }
              return tp;
          }, doc_diff_array_t);

    m.def("detached_t",
          [](nb::handle h) {
              nb::handle tp = h.is_type() ? h : h.type();
              if (is_drjit_type(tp)) {
                  ArrayMeta m = supp(tp);
                  m.is_diff = false;
                  tp = meta_get_type(m);
              }
              return tp;
          }, doc_detached_t);

    m.def("expr_t", nb::overload_cast<nb::args>(expr_t), doc_expr_t);
}
