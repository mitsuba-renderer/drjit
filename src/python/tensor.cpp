struct Component {
    Py_ssize_t start, end, step, size;
    nb::object object;

    Component(Py_ssize_t start, Py_ssize_t end, Py_ssize_t step, Py_ssize_t size)
        : start(start), end(end), step(step), size(size) { }

    Component(nb::handle h) : object(nb::borrow(h)) {
        if (h.is_none())
            size = 1;
        else
            size = nb::len(h);
    }
};

void slice_index(const nb::type_object &dtype, const nb::tuple &shape,
                 const nb::tuple &indices) {
    size_t none_count = 0, shape_offset = 0;
    bool ellipsis = false;

    if (!is_drjit_type(dtype))
        nb::detail::raise("slice_index(): dtype must be a Dr.Jit array!");

    meta dtype_meta = get_meta(dtype);
    if (dtype_meta.ndim != 1 || dtype_meta.shape[0] != 0xFF ||
        (VarType) dtype_meta.type != VarType::UInt32)
        nb::detail::raise("slice_index(): dtype must be a dynamically sized "
                          "unsigned 32 bit Dr.Jit array!");

    std::vector<Component> components;
    components.reserve(shape.size());

    for (nb::handle h : shape) {
        if (h.is_none())
            none_count++;
    }

    for (nb::handle h : indices) {
        if (h.is_none()) {
            components.emplace_back(h);
            continue;
        }

        if (shape_offset >= nb::len(shape))
            nb::raise("slice_index(): too many indices specified!");

        Py_ssize_t size = nb::cast<Py_ssize_t>(shape[shape_offset]);
        nb::handle tp = h.type();

        if (tp.is(&PyLong_Type)) {
            // Simple integer index, handle wrap-around
            Py_ssize_t v = nb::cast<Py_ssize_t>(h);
            if (v < 0)
                v += size;

            if (v < 0 || v >= size)
                nb::detail::raise("slice_tensor(): index %zd for dimension "
                                  "%zu is out of range (size = %zd)!" %
                                  (v, nb::len(components), size));

            components.emplace_back(v, v + 1, 1, 1);
            ++shape_offset;
            continue;
        }

        if (tp.is(&PySlice_Type)) {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(h.ptr(), &start, &stop, &step) < 0)
                nb::detail::raise_python_error();
            Py_ssize_t size = PySlice_AdjustIndices(size, &start, &stop, step);
            components.emplace_back(start, stop, step, size);
            ++shape_offset;
            continue;
        }

        if (is_drjit_type(tp)) {
            const supp &s = get_supp(tp);
            if (s.ndim == 1 && s.shape[0] == 0xFF && s.value == &PyLong_Type) {
                nb::object o = borrow(h);
                VarType vt = (VarType) s.meta.type;

                if (vt == VarType::Int8 || vt == VarType::Int16 ||
                    vt == VarType::Int32 || vt == VarType::Int64)
                    o[o < 0] += size;

                components.append(o);
                ++shape_offset;
                continue;
            }
        }

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

        nb::detail::raise(
            "slice_tensor(): type '%s' cannot be used to index into a tensor!",
            ((PyTypeObject *) tp.ptr())->tp_name);
    }

    // # Implicit ellipsis
    // for j in range(len(shape) - shape_offset):
    //     components.append((0, shape[shape_offset], 1))
    //     shape_offset += 1

    Py_ssize_t size_out = 1;
    nb::tuple shape_out = steal<nb::tuple>(PyTuple_New(components.size()));

    for (const Component &c : components) {
        size_out *= c.size; // XXX oops
        PyTuple_SET_ITEM(shape_out, i, PyLong_From_Ssize_t(c.size));
    }
}

#if 0
def slice_tensor(shape, indices, uint32):
    """
    """
        elif isinstance(v, Sequence):
            components.append(uint32([v2 if v2 >= 0 else v2 + size for v2 in v]))
        else:

    # Compute total index size
    size_out = 1
    shape_out = []
    for comp in components:
        if comp is None:
            shape_out.append(1)
        else:
            size = len(comp if isinstance(comp, uint32) else range(*comp))
            if size != 1:
                shape_out.append(size)
                size_out *= shape_out[-1]
    shape_out = tuple(shape_out)

    index_tmp = drjit.arange(uint32, size_out)
    index_out = uint32()

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
