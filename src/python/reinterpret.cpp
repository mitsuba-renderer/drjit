#include "common.h"

py::object reinterpret_scalar(const py::object &source,
                              VarType vt_source,
                              VarType vt_target) {
    if (var_type_size[(int) vt_source] != var_type_size[(int) vt_target])
        dr::drjit_raise("reinterpret_scalar(): input/output size mismatch!");

    uint64_t temp = 0;

    switch (vt_source) {
        case VarType::UInt8:    temp = (uint64_t) py::cast<uint8_t>(source); break;
        case VarType::Int8:     temp = (uint64_t) (uint8_t) py::cast<int8_t>(source); break;
        case VarType::UInt16:   temp = (uint64_t) py::cast<uint16_t>(source); break;
        case VarType::Int16:    temp = (uint64_t) (uint16_t) py::cast<int16_t>(source); break;
        case VarType::UInt32:   temp = (uint64_t) py::cast<uint32_t>(source); break;
        case VarType::Int32:    temp = (uint64_t) (uint32_t) py::cast<int32_t>(source); break;
        case VarType::UInt64:   temp = (uint64_t) py::cast<uint64_t>(source); break;
        case VarType::Int64:    temp = (uint64_t) py::cast<int64_t>(source); break;
        case VarType::Float32:  temp = (uint64_t) dr::memcpy_cast<uint32_t>(py::cast<float>(source)); break;
        case VarType::Float64:  temp = (uint64_t) dr::memcpy_cast<uint64_t>(py::cast<double>(source)); break;
        default: throw py::type_error("reinterpret_scalar(): unsupported input type!");
    }

    switch (vt_target) {
        case VarType::UInt8:   return py::cast((uint8_t) temp);
        case VarType::Int8:    return py::cast((int8_t) temp);
        case VarType::UInt16:  return py::cast((uint16_t) temp);
        case VarType::Int16:   return py::cast((int16_t) temp);
        case VarType::UInt32:  return py::cast((uint32_t) temp);
        case VarType::Int32:   return py::cast((int32_t) temp);
        case VarType::UInt64:  return py::cast((uint64_t) temp);
        case VarType::Int64:   return py::cast((int64_t) temp);
        case VarType::Float32: return py::cast(dr::memcpy_cast<float>((uint32_t) temp));
        case VarType::Float64: return py::cast(dr::memcpy_cast<double>((uint64_t) temp));
        default: throw py::type_error("reinterpret_scalar(): unsupported output type!");
    }
}
