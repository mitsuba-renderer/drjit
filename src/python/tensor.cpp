#include "python.h"
#include <vector>
#include <iostream>

/// Holds metadata about slicing component
struct Component {
    Py_ssize_t start, step, slice_size, size;
    nb::object object;

    Component(Py_ssize_t start, Py_ssize_t step, Py_ssize_t slice_size,
              Py_ssize_t size)
        : start(start), step(step), slice_size(slice_size), size(size) { }

    Component(nb::handle h, Py_ssize_t slice_size, Py_ssize_t size)
        : start(0), step(1), slice_size(slice_size), size(size),
          object(nb::borrow(h)) { }
};

std::pair<nb::tuple, nb::object> slice_index(const nb::type_object &dtype,
                                             const nb::tuple &shape,
                                             const nb::tuple &indices) {
    bool dtype_fail = true;
    if (is_drjit_type(dtype)) {
        meta m = nb::type_supplement<supp>(dtype).meta;

        if (m.ndim == 1 && m.shape[0] == 0xFF &&
            (VarType) m.type == VarType::UInt32)
            dtype_fail = false;
    }

    if (dtype_fail)
        throw nb::type_error("slice_index(): dtype must be a dynamically "
                             "sized unsigned 32 bit Dr.Jit array!");

    size_t shape_offset = 0;
    size_t size_out = 1;
    nb::list shape_out;

    // Preallocate memory for computed slicing components
    size_t shape_len = nb::len(shape);
    std::vector<Component> components;
    components.reserve(shape_len);

    for (nb::handle h : indices) {
        if (h.is_none()) {
            shape_out.append(1);
            continue;
        }

        if (shape_offset >= shape_len)
            nb::detail::raise("too many indices");

        Py_ssize_t size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
        nb::handle tp = h.type();

        if (tp.is(&PyLong_Type)) {
            Py_ssize_t v = nb::cast<Py_ssize_t>(h);
            if (v < 0)
                v += size;

            if (v < 0 || v >= size)
                nb::detail::raise(
                    "index %zd is out of bounds for axis %zu with size %zd", v,
                    components.size(), size);

            components.emplace_back(v, 1, 1, size);
            continue;
        } else if (tp.is(&PySlice_Type)) {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(h.ptr(), &start, &stop, &step) < 0)
                nb::detail::raise_python_error();
            Py_ssize_t slice_size =
                PySlice_AdjustIndices(size, &start, &stop, step);
            components.emplace_back(start, step, slice_size, size);
            shape_out.append(slice_size);
            size_out *= slice_size;
            continue;
        } else if (is_drjit_type(tp)) {
            meta m = nb::type_supplement<supp>(tp).meta;

            if (m.ndim == 1 && m.shape[0] == 0xFF) {
                VarType vt = (VarType) m.type;
                nb::object o = nb::borrow(h);

                size_t slice_size = nb::len(h);
                if (vt == VarType::Int8 || vt == VarType::Int16 ||
                    vt == VarType::Int32 || vt == VarType::Int64) {
                    o[o < nb::cast(0)] += nb::cast(size);
                }

                if (!o.type().is(dtype))
                    o = dtype(o);

                components.emplace_back(o, slice_size, size);
                shape_out.append(slice_size);
                size_out *= slice_size;
                continue;
            }
        }

        nb::detail::raise("unsupported type \"%s\" in slice",
                          ((PyTypeObject *) tp.ptr())->tp_name);
    }

    nb::object fma    = array_module.attr("fma"),
               arange = array_module.attr("arange"),
               gather = array_module.attr("gather");

    nb::object index = arange(dtype, size_out),
               index_out = dtype(0);

    size_out = 1;

    for (auto it = components.rbegin(); it != components.rend(); ++it) {
        const Component &c = *it;
        nb::object index_next, index_rem;

        if (it + 1 != components.rend()) {
            index_next = index.floor_div(dtype(c.slice_size));
            index_rem = fma(index_next, dtype(uint32_t(-c.slice_size)), index);
        } else {
            index_rem = index;
        }

        nb::object index_val;
        if (c.object.is_none())
            index_val = fma(index_rem, dtype(uint32_t(c.step * size_out)),
                            dtype(uint32_t(c.start * size_out)));
        else
            index_val =
                gather(dtype, c.object, index_rem) * dtype(uint32_t(size_out));

        index_out += index_val;

        index = std::move(index_next);
        size_out *= c.size;
    }

    return { nb::steal<nb::tuple>(PySequence_Tuple(shape_out.ptr())),
             index_out };
}

void bind_tensor(nb::module_ m) {
    m.def("slice_index", &slice_index, doc_slice_index, "dtype"_a, "shape"_a,
          "indices"_a);
}

#if 0

void slice_index(const nb::type_object &dtype, const nb::tuple &shape,
                 const nb::tuple &indices) {



        if (tp.is(&PyEllipsis_Type)) {
            if (ellipsis)
                nb::detail::raise("slice_tensor(): multiple ellipses (...) are not allowed!");
            for (size_t i = 0, remain = nb::len(shape) - (nb::len(indices) - none_count); i < remain; ++i) {
                if (shape_offset >= nb::len(shape))
                    nb::fail("slice_index(): internal error!");
                size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
                components.emplace_back(0, size, 1, size);
            }
            ellipsis = true;
            continue;
        }
    }

    // # Implicit ellipsis
    // for j in range(len(shape) - shape_offset):
    //     components.append((0, shape[shape_offset], 1))
    //     shape_offset += 1

}

def slice_tensor(shape, indices, uint32):
    """
    """
        elif isinstance(v, Sequence):
            components.append(uint32([v2 if v2 >= 0 else v2 + size for v2 in v]))
        else:


    if size_out > 0:
        size_out = 1
        index_out = uint32(0)
        shape_offset = len(shape)-1

        for i in reversed(range(len(components))):
            comp = components[i]
            if comp is None:
                continue
            size = len(comp if isinstance(comp, uint32) else range(*comp))
            index_next = index_tmp // size
            index_rem = index_tmp - index_next * size

            if isinstance(comp, uint32):
                index_val = drjit.gather(uint32, comp, index_rem)
            else:
                if comp[0] >= 0 and comp[2] >= 0:
                    index_val = comp[0] + comp[2] * index_rem
                else:
                    index_val = uint32(comp[0] + comp[2] * drjit.int32_array_t(index_rem))

            index_out += index_val * size_out
            index_tmp = index_next
            size_out *= shape[shape_offset]
            shape_offset -= 1

    return shape_out, index_out
#endif


// def tensor_getitem(tensor, slice_arg):
//     tensor_t = type(tensor)
//     shape, index = slice_tensor(tensor.shape, slice_arg, tensor_t.Index)
//     return tensor_t(drjit.gather(tensor_t.Array, tensor.array, index), shape)
//
//
// def tensor_setitem(tensor, slice_arg, value):
//     tensor_t = type(tensor)
//     shape, index = slice_tensor(tensor.shape, slice_arg, tensor_t.Index)
//     drjit.scatter(target=tensor.array, value=value, index=index)
