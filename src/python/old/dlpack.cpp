#include "common.h"
#include <nanobind/functional.h>

struct DLManagedTensor {
    void *data;
    int device_type;
    int device_id;
    int ndim;
    int dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
    PyObject *manager_ctx;
    void (*deleter)(DLManagedTensor *);
};

static_assert(sizeof(DLManagedTensor) == 64, "Alignment issue?");


static void cleanup(DLManagedTensor *mt) {
    delete[] mt->shape;
    delete[] mt->strides;
    Py_DECREF(mt->manager_ctx);
    delete mt;
}

static int64_t *convert_tuple(const nb::tuple &t) {
    size_t size = t.size();
    if (size == 0)
        return nullptr;

    std::unique_ptr<int64_t[]> result(new int64_t[size]);
    for (size_t i = 0; i < t.size(); ++i)
        result[i] = nb::cast<int64_t>(t[i]);
    return result.release();
}

static nb::object convert_tuple(int64_t *index, int ndim) {
    if (index == nullptr)
        return nb::none();

    nb::tuple t(ndim);

    for (int i = 0; i < ndim; ++i)
        PyTuple_SET_ITEM(t.ptr(), i, PyLong_FromSsize_t((nb::ssize_t) index[i]));

    return std::move(t);
}

static int convert_dtype(VarType type) {
    int code;
    if (var_type_is_float[(int) type])
        code = 2;
    else
        code = var_type_is_unsigned[(int) type] ? 1 : 0;

    return code | ((var_type_size[(int) type] * 8) << 8) | (1 << 16);
}

static VarType convert_dtype(int dtype) {
    for (int i = 0; i < (int) VarType::Count; ++i) {
        if (convert_dtype((VarType) i) == dtype)
            return (VarType) i;
    }

    throw std::runtime_error("Unsupported dtype!");
}

nb::capsule to_dlpack(const nb::object &owner, uint64_t data, VarType type,
                      int device, const nb::tuple &shape,
                      const nb::tuple &strides) {
    DLManagedTensor* t = new DLManagedTensor();
    t->data = (void *) data;
    t->ndim = (int) shape.size();
    t->device_type = device == -1 ? 1 : 2;
    t->device_id = device == -1 ? 0 : device;
    t->shape = convert_tuple(shape);
    t->strides = convert_tuple(strides);
    t->dtype = convert_dtype(type);
    t->byte_offset = 0;
    t->deleter = cleanup;
    t->manager_ctx = owner.ptr();
    Py_INCREF(t->manager_ctx);

    nb::capsule capsule(t, "dltensor", [](PyObject *o) {
        DLManagedTensor *mt = reinterpret_cast<DLManagedTensor *>(
            PyCapsule_GetPointer(o, "dltensor"));
        if (mt)
            cleanup(mt);
        else
            PyErr_Clear();
    });

    return capsule;
}

nb::dict from_dlpack(const nb::capsule &o) {
    const char *name = PyCapsule_GetName(o.ptr());
    if (strcmp(name, "dltensor") != 0)
        throw std::runtime_error("DLTensor capsule was already consumed!");

    DLManagedTensor *t = (DLManagedTensor *) PyCapsule_GetPointer(o.ptr(), name);

    std::function<void(const nb::capsule &)> consume = [](const nb::capsule &o) {
        PyCapsule_SetName(o.ptr(), "used_dltensor");
        PyCapsule_SetDestructor(o.ptr(), nullptr);
    };

    std::function<void(void)> release = [t]() {
        t->deleter(t);
    };

    nb::dict d;
    d["data"] = nb::cast((uintptr_t) t->data + t->byte_offset);
    d["shape"] = convert_tuple(t->shape, t->ndim);
    d["strides"] = convert_tuple(t->strides, t->ndim);
    d["dtype"] = convert_dtype(t->dtype);
    d["device_type"] = nb::cast(t->device_type);
    d["consume"] = nb::cast(consume);
    d["release"] = nb::cast(release);

    return d;
}
