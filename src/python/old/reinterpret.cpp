#include "common.h"

nb::object reinterpret_scalar(const nb::object &source,
                              VarType vt_source,
                              VarType vt_target) {
    if (var_type_size[(int) vt_source] != var_type_size[(int) vt_target])
        dr::drjit_raise("reinterpret_scalar(): input/output size mismatch!");

    uint64_t temp = 0;

    switch (vt_source) {
        case VarType::UInt8:    temp = (uint64_t) nb::cast<uint8_t>(source); break;
        case VarType::Int8:     temp = (uint64_t) (uint8_t) nb::cast<int8_t>(source); break;
        case VarType::UInt16:   temp = (uint64_t) nb::cast<uint16_t>(source); break;
        case VarType::Int16:    temp = (uint64_t) (uint16_t) nb::cast<int16_t>(source); break;
        case VarType::UInt32:   temp = (uint64_t) nb::cast<uint32_t>(source); break;
        case VarType::Int32:    temp = (uint64_t) (uint32_t) nb::cast<int32_t>(source); break;
        case VarType::UInt64:   temp = (uint64_t) nb::cast<uint64_t>(source); break;
        case VarType::Int64:    temp = (uint64_t) nb::cast<int64_t>(source); break;
        case VarType::Float32:  temp = (uint64_t) dr::memcpy_cast<uint32_t>(nb::cast<float>(source)); break;
        case VarType::Float64:  temp = (uint64_t) dr::memcpy_cast<uint64_t>(nb::cast<double>(source)); break;
        default: throw nb::type_error("reinterpret_scalar(): unsupported input type!");
    }

    switch (vt_target) {
        case VarType::UInt8:   return nb::cast((uint8_t) temp);
        case VarType::Int8:    return nb::cast((int8_t) temp);
        case VarType::UInt16:  return nb::cast((uint16_t) temp);
        case VarType::Int16:   return nb::cast((int16_t) temp);
        case VarType::UInt32:  return nb::cast((uint32_t) temp);
        case VarType::Int32:   return nb::cast((int32_t) temp);
        case VarType::UInt64:  return nb::cast((uint64_t) temp);
        case VarType::Int64:   return nb::cast((int64_t) temp);
        case VarType::Float32: return nb::cast(dr::memcpy_cast<float>((uint32_t) temp));
        case VarType::Float64: return nb::cast(dr::memcpy_cast<double>((uint64_t) temp));
        default: throw nb::type_error("reinterpret_scalar(): unsupported output type!");
    }
}
