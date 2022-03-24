#include <drjit/python.h>
#include "../ext/nanobind/src/buffer.h"

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

namespace nb = nanobind;

static nb::detail::Buffer buffer;

static const char *type_name[] = {
    "Void", "Bool", "Int8", "UInt8", "Int16", "UInt16", "Int", "UInt",
    "Int64", "UInt64", "Pointer", "Float16", "Float", "Float64"
};

static const char *type_suffix[] = {
    "v", "b", "i8", "u8", "i16", "u16", "i", "u",
    "i64", "u64", "p", "f16", "f", "f64"
};

const char *array_name(array_metadata meta) {
    buffer.clear();

    int depth = 1;
    for (size_t i = 1; i < 4; ++i) {
        if (!meta.shape[i])
            break;
        depth++;
    }

    if (meta.is_llvm || meta.is_cuda)
        depth--;

    const char *suffix = nullptr;

    if (depth == 0) {
        buffer.put_dstr(type_name[meta.type]);
    } else {
        const char *prefix = "Array";
        if (meta.is_complex)
            prefix = "Complex";
        else if (meta.is_quaternion)
            prefix = "Quaternion";
        else if (meta.is_matrix)
            prefix = "Matrix";
        else if (meta.is_tensor)
            prefix = "Tensor";
        buffer.put_dstr(prefix);
        suffix = type_suffix[meta.type];
    }

    for (int i = 0; i < depth; ++i) {
        if (meta.shape[i] == 0xFF)
            buffer.put('X');
        else
            buffer.put_uint32(meta.shape[i]);
    }

    if (suffix)
        buffer.put_dstr(suffix);

    return buffer.get();
}

static nb::handle submodules[6];
static nb::handle array_base;

void prepare_imports() {
    submodules[0] = nb::module_::import_("drjit");
    submodules[1] = nb::module_::import_("drjit.scalar");
    submodules[2] = nb::module_::import_("drjit.cuda");
    submodules[3] = nb::module_::import_("drjit.cuda.ad");
    submodules[4] = nb::module_::import_("drjit.llvm");
    submodules[5] = nb::module_::import_("drjit.llvm.ad");
    array_base = submodules[0].attr("ArrayBase");
}

static nb::handle array_module(array_metadata meta) {
    int index = 1;
    if (meta.is_llvm || meta.is_cuda) {
        index = meta.is_llvm ? 4 : 2;
        index += (int) meta.is_diff;
    }

    if (!array_base.is_valid())
        prepare_imports();
    return submodules[index];
}

const nb::handle array_get(array_metadata meta) {
    return array_module(meta).attr(array_name(meta));
}

extern nb::handle bind(const char *name, array_supplement &supp,
                       const std::type_info *type,
                       const std::type_info *value_type,
                       void (*copy)(void *, const void *),
                       void (*move)(void *, void *) noexcept,
                       void (*destruct)(void *) noexcept,
                       void (*type_callback)(PyTypeObject *) noexcept) noexcept {
    if (!name)
        name = array_name(supp.meta);

    nb::detail::type_data d;
    d.flags = (uint32_t) nb::detail::type_flags::has_scope |
              (uint32_t) nb::detail::type_flags::has_base_py |
              (uint32_t) nb::detail::type_flags::has_type_callback |
              (uint32_t) nb::detail::type_flags::is_destructible |
              (uint32_t) nb::detail::type_flags::is_copy_constructible |
              (uint32_t) nb::detail::type_flags::is_move_constructible;

    if (move) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_move;
        d.move = move;
    }

    if (copy) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_copy;
        d.copy = copy;
    }

    if (destruct) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_destruct;
        d.destruct = destruct;
    }

    d.align = supp.meta.talign;
    d.size = supp.meta.tsize_rel * supp.meta.talign;
    d.supplement = sizeof(array_supplement);
    d.name = name;
    d.type = type;
    d.type_callback = type_callback;

    d.scope = array_module(supp.meta).ptr();
    d.base_py = (PyTypeObject *) array_base.ptr();

    nb::handle h = nb::detail::nb_type_new(&d);

    VarType vt = (VarType) supp.meta.type;
    bool is_mask     = vt == VarType::Bool;
    bool is_float    = vt == VarType::Float16 || vt == VarType::Float32 ||
                       vt == VarType::Float64;

    bool is_integral = vt == VarType::Int8 || vt == VarType::UInt8 ||
                       vt == VarType::Int16 || vt == VarType::UInt16 ||
                       vt == VarType::Int32 || vt == VarType::UInt32 ||
                       vt == VarType::Int64 || vt == VarType::UInt64;

    h.attr("IsMask") = is_mask;
    h.attr("IsFloat") = is_float;
    h.attr("IsIntegral") = is_integral;
    h.attr("IsArithmetic") = is_float || is_integral;
    h.attr("IsVector") = (bool) supp.meta.is_vector;
    h.attr("IsComplex") = (bool) supp.meta.is_complex;
    h.attr("IsQuaternion") = (bool) supp.meta.is_quaternion;
    h.attr("IsMatrix") = (bool) supp.meta.is_matrix;
    h.attr("IsTensor") = (bool) supp.meta.is_tensor;

    int depth = 1;
    for (size_t i = 1; i < 4; ++i) {
        uint8_t value = supp.meta.shape[i];
        if (!value)
            break;
        depth++;
    }
    h.attr("Depth") = depth;

    nb::tuple shape = nb::steal<nb::tuple>(PyTuple_New(depth));
    for (int i = 0; i < depth; ++i) {
        uint8_t value = supp.meta.shape[i];
        Py_ssize_t value_sz = value == 0xFF ? -1 : value;
        PyTuple_SET_ITEM(shape.ptr(), i, PyLong_FromSize_t(value_sz));
    }

    h.attr("Shape") = shape;
    h.attr("Size") = shape[0];

    nb::handle value_type_py;
    if (!value_type) {
        if (is_mask)
            value_type_py = (PyObject *) &PyBool_Type;
        else if (is_float)
            value_type_py = (PyObject *) &PyFloat_Type;
        else
            value_type_py = (PyObject *) &PyLong_Type;
    } else {
        value_type_py = nb::detail::nb_type_lookup(value_type);
        if (!value_type_py.is_valid())
            nb::detail::fail("bind(): element type not found!");
    }

    auto pred = [](PyTypeObject *tp, PyObject *o,
                   nb::detail::cleanup_list *) -> bool {
        detail::array_supplement &s =
            nb::type_supplement<detail::array_supplement>(tp);

        if (PySequence_Check(o)) {
            Py_ssize_t size = s.meta.shape[0], len = PySequence_Length(o);
            if (len == -1)
                PyErr_Clear();
            return size == 0xFF || len == size;
        } else if (s.value == Py_TYPE(o)) {
            return true;
        } else {
            VarType v = (VarType) s.meta.type;
            return (v == VarType::Float16 || v == VarType::Float32 ||
                    v == VarType::Float64) &&
                   Py_TYPE(o) == &PyLong_Type;
        }
    };
    nb::detail::implicitly_convertible(pred, type);

    nb::handle mask_type_py;
    if (is_mask) {
        mask_type_py = h.ptr();
    } else {
        array_metadata mask_meta = supp.meta;
        mask_meta.type = (uint16_t) VarType::Bool;
        mask_meta.is_vector = mask_meta.is_complex = mask_meta.is_quaternion =
            mask_meta.is_matrix = false;
        mask_type_py = array_get(mask_meta);
    }

    h.attr("Value") = value_type_py;
    h.attr("Mask") = mask_type_py;
    supp.value = (PyTypeObject *) value_type_py.ptr();
    supp.mask = (PyTypeObject *) mask_type_py.ptr();

    nb::type_supplement<detail::array_supplement>(h.ptr()) = supp;

    return h.ptr();
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
