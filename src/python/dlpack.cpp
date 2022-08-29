#include "common.h"
#include <pybind11/functional.h>

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
    py::gil_scoped_acquire acquire;
    delete[] mt->shape;
    delete[] mt->strides;
    Py_DECREF(mt->manager_ctx);
    delete mt;
}

static int64_t *convert_tuple(const py::tuple &t) {
    size_t size = t.size();
    if (size == 0)
        return nullptr;

    std::unique_ptr<int64_t[]> result(new int64_t[size]);
    for (size_t i = 0; i < t.size(); ++i)
        result[i] = py::cast<int64_t>(t[i]);
    return result.release();
}

static py::object convert_tuple(int64_t *index, int ndim) {
    if (index == nullptr)
        return py::none();

    py::tuple t(ndim);

    for (int i = 0; i < ndim; ++i)
        PyTuple_SET_ITEM(t.ptr(), i, PyLong_FromSsize_t((py::ssize_t) index[i]));

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

py::capsule to_dlpack(const py::object &owner, uint64_t data, VarType type,
                      int device, const py::tuple &shape,
                      const py::tuple &strides) {
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

    py::capsule capsule(t, "dltensor", [](PyObject *o) {
        DLManagedTensor *mt = reinterpret_cast<DLManagedTensor *>(
            PyCapsule_GetPointer(o, "dltensor"));
        if (mt)
            cleanup(mt);
        else
            PyErr_Clear();
    });

    return capsule;
}

py::dict from_dlpack(const py::capsule &o) {
    const char *name = PyCapsule_GetName(o.ptr());
    if (strcmp(name, "dltensor") != 0)
        throw std::runtime_error("DLTensor capsule was already consumed!");

    DLManagedTensor *t = (DLManagedTensor *) PyCapsule_GetPointer(o.ptr(), name);

    std::function<void(const py::capsule &)> consume = [](const py::capsule &o) {
        PyCapsule_SetName(o.ptr(), "used_dltensor");
        PyCapsule_SetDestructor(o.ptr(), nullptr);
    };

    std::function<void(void)> release = [t]() {
        t->deleter(t);
    };

    py::dict d;
    d["data"] = py::cast((uintptr_t) t->data + t->byte_offset);
    d["shape"] = convert_tuple(t->shape, t->ndim);
    d["strides"] = convert_tuple(t->strides, t->ndim);
    d["dtype"] = convert_dtype(t->dtype);
    d["device_type"] = py::cast(t->device_type);
    d["consume"] = py::cast(consume);
    d["release"] = py::cast(release);

    return d;
}
