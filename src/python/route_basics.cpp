#include "route.h"

const char *var_type_name[(int) VarType::Count] = {
    "Invalid", "Int8",   "UInt8",   "Int16",   "UInt16",  "Int32", "UInt32",
    "Int64",   "UInt64", "Float16", "Float32", "Float64", "Bool",  "Pointer"
};

const char *var_type_suffix[(int) VarType::Count] = {
    "???", "i8",  "u8",  "i16", "u16", "i32", "u32",
    "i64", "u64", "f16", "f32", "f64", "b", "p"
};

const uint32_t var_type_size[(int) VarType::Count] {
    (uint32_t) -1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8, 1, 8
};

size_t coeff_evals = 0;

static py::object array_route_empty(py::handle h, size_t);
static py::object array_route_min(py::object a0, py::object a1);
static py::object array_route_max(py::object a0, py::object a1);
static py::object array_route_fmadd(py::object a0, py::object a1, py::object a2);

static char array_name_tmp[64];
const char *array_name(VarType vtype, size_t depth, size_t size, bool scalar_mode) {
    if (depth == 0 || (!scalar_mode && depth == 1))
        return var_type_name[(int) vtype];

    if (size == ek::Dynamic)
        snprintf(array_name_tmp, sizeof(array_name_tmp), "ArrayX%s",
                 var_type_suffix[(int) vtype]);
    else
        snprintf(array_name_tmp, sizeof(array_name_tmp), "Array%zu%s", size,
                 var_type_suffix[(int) vtype]);

    return array_name_tmp;
}

static py::list array_shape(py::object o) {
    py::list l;
    ssize_t len;
    while ((len = PyObject_Length(o.ptr())) != -1) {
        l.append(len);
        if (len > 0)
            o = o[py::int_(0)];
        else
            o = array_route_empty(o.attr("Value"), 0);
    }
    PyErr_Clear();
    return l;
}

static int array_ndim(py::object o) {
    int ndim = 0;
    while (PyObject_Length(o.ptr()) != -1) {
        o = o[py::int_(0)];
        ndim++;
    }
    PyErr_Clear();
    return ndim;
}

static bool array_ragged_impl(const py::object &o, const py::list &shape,
                              size_t i, size_t ndim) {
    size_t size = py::cast<size_t>(shape[i]);

    if (py::len(o) != size)
        return true;

    if (i + 1 != ndim) {
        for (uint32_t j = 0; j < size; ++j) {
            if (array_ragged_impl(o[py::int_(j)], shape, i + 1, ndim))
                return true;
        }
    }

    return false;
}

static bool array_ragged(const py::object &o) {
    py::list shape = array_shape(o);
    size_t ndim = shape.size();
    if (ndim == 0)
        return false;
    return array_ragged_impl(o, shape, 0, ndim);
}

static size_t print_threshold = 20;

void array_repr_impl(py::object o, const py::list &shape, size_t i,
                     size_t ndim, py::tuple &tuple, py::str &result) {
    if (i == ndim) {
        result = result + py::repr(o[tuple]);
    } else {
        uint32_t k = ndim - i - 1;
        size_t size = py::cast<size_t>(shape[k]);
        result = result + py::str("[");
        for (uint32_t j = 0; j < size; ++j) {
            if (size > print_threshold && j == 5) {
                result = result + py::str(".. " + std::to_string(size - 10) + " skipped ..");
                j = size - 6;
            } else {
                tuple[k] = py::cast(j);
                array_repr_impl(o, shape, i + 1, ndim, tuple, result);
            }

            if (j + 1 < size) {
                if (k == 0)
                    result = result + py::str(", ");
                else
                    result = result + py::str(",\n") +
                             (py::str(" ") * py::int_(i + 1));
            }
        }
        result = result + py::str("]");
    }
}

static py::str array_repr(py::object o) {
    py::list shape = array_shape(o);
    size_t ndim = shape.size();
    py::str result;
    if (ndim == 0) {
        result = "[]";
    } else if (array_ragged_impl(o, shape, 0, ndim)) {
        result = "[ragged array]";
    } else {
        result = "";
        py::tuple tuple(ndim);
        array_repr_impl(o, shape, 0, ndim, tuple, result);
    }
    return result;
}

static py::object array_getitem(const py::object &o, size_t index) {
    ssize_t len = PyObject_Length(o.ptr());
    if (len == -1)
        PyErr_Clear();
    if ((ssize_t) index < len) {
        coeff_evals++;
        return o.attr("coeff")(py::int_(index));
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "__getitem__(): index %zu exceeds the array bounds %zu", index,
                 (size_t) len);
        throw py::index_error(msg);
    }
}

void array_setitem(const py::object &o, size_t index, const py::object &value) {
    ssize_t len = PyObject_Length(o.ptr());
    if (len == -1)
        PyErr_Clear();
    if ((ssize_t) index < len) {
        coeff_evals++;
        o.attr("set_coeff")(py::int_(index), value);
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "__setitem__(): index %zu exceeds the array bounds %zu", index,
                 (size_t) len);
        throw py::index_error(msg);
    }
}

static py::object array_getitem_tuple(py::object o, const py::tuple &t) {
    size_t tuple_len = t.size();
    for (size_t i = 0; i < tuple_len; ++i) {
        ssize_t len = PyObject_Length(o.ptr());
        if (len == -1)
            PyErr_Clear();
        size_t index = py::cast<size_t>(t[i]);
        if ((ssize_t) index >= len)
            throw py::index_error();
        o = o.attr("coeff")(py::int_(index));
        coeff_evals++;
    }
    return o;
}

static void array_setitem_tuple(py::object o, const py::tuple &t, const py::object &value) {
    size_t tuple_len = t.size();
    for (size_t i = 0; i < tuple_len; ++i) {
        ssize_t len = PyObject_Length(o.ptr());
        if (len == -1)
            PyErr_Clear();
        size_t index = py::cast<size_t>(t[i]);
        if ((ssize_t) index >= len)
            throw py::index_error();
        if (i + 1 < tuple_len)
            o = o.attr("coeff")(py::int_(index));
        else
            o.attr("set_coeff")(py::int_(index), value);
        coeff_evals++;
    }
}

static py::object array_route_all(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_mask("all", var_type(a));
    return a.attr("all_")();
}

static py::object array_route_any(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_mask("any", var_type(a));
    return a.attr("any_")();
}

static py::object array_route_none(const py::object &a) {
    return ~array_route_any(a);
}

static py::object array_generic_all(const py::object &a) {
    size_t size = py::len(a);
    if (size == 0)
        ek::enoki_raise("all(): zero-sized array!");

    py::object value = a[py::int_(0)];
    for (size_t i = 1; i < size; ++i)
        value = value & a[py::int_(i)];
    return value;
}

static py::object array_generic_any(const py::object &a) {
    size_t size = py::len(a);
    if (size == 0)
        ek::enoki_raise("any(): zero-sized array!");

    py::object value = a[py::int_(0)];
    for (size_t i = 1; i < size; ++i)
        value = value | a[py::int_(i)];
    return value;
}

static py::object array_route_hsum(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hsum", var_type(a));
    return a.attr("hsum_")();
}

static py::object array_route_hsum_async(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hsum_async", var_type(a));
    if (py::hasattr(a, "hsum_async_"))
        return a.attr("hsum_async_")();
    else
        return a.get_type()(a.attr("hsum_")());
}

static py::object array_route_hprod(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hprod", var_type(a));
    return a.attr("hprod_")();
}

static py::object array_route_hprod_async(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hprod_async", var_type(a));
    if (py::hasattr(a, "hprod_async_"))
        return a.attr("hprod_async_")();
    else
        return a.get_type()(a.attr("hprod_")());
}

static py::object array_route_hmin(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hmin", var_type(a));
    return a.attr("hmin_")();
}

static py::object array_route_hmin_async(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hmin_async", var_type(a));
    if (py::hasattr(a, "hmin_async_"))
        return a.attr("hmin_async_")();
    else
        return a.get_type()(a.attr("hmin_")());
}

static py::object array_route_hmax(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hmax", var_type(a));
    return a.attr("hmax_")();
}

static py::object array_route_hmax_async(const py::object &a) {
    if (!var_is_enoki_type(a))
        return a;
    test_arithmetic("hmax_async", var_type(a));
    if (py::hasattr(a, "hmax_async_"))
        return a.attr("hmax_async_")();
    else
        return a.get_type()(a.attr("hmax_")());
}

static py::object array_route_dot(py::object &a, py::object &b) {
    var_promote("dot", a, b, true, test_float);
    return a.attr("dot_")(b);
}

static py::object array_route_dot_async(py::object &a, py::object &b) {
    var_promote("dot_async", a, b, true, test_float);
    if (py::hasattr(a, "dot_async_"))
        return a.attr("dot_async_")(b);
    else
        return a.get_type()(a.attr("dot_")(b));
}

static py::object array_route_hmean(const py::object &a) {
    return array_route_hsum(a) / py::float_((double) py::len(a));
}

static py::object array_route_hmean_async(const py::object &a) {
    return array_route_hsum_async(a) / py::float_((double) py::len(a));
}

static py::object array_generic_hsum(const py::object &a) {
    size_t size = py::len(a);
    if (size == 0)
        ek::enoki_raise("hsum(): zero-sized array!");

    py::object value = a[py::int_(0)];
    for (size_t i = 1; i < size; ++i)
        value = value + a[py::int_(i)];
    return value;
}

static py::object array_generic_hprod(const py::object &a) {
    size_t size = py::len(a);
    if (size == 0)
        ek::enoki_raise("hprod(): zero-sized array!");

    py::object value = a[py::int_(0)];
    for (size_t i = 1; i < size; ++i)
        value = value * a[py::int_(i)];
    return value;
}

static py::object array_generic_hmin(const py::object &a) {
    size_t size = py::len(a);
    if (size == 0)
        ek::enoki_raise("hmin(): zero-sized array!");

    py::object value = a[py::int_(0)];
    for (size_t i = 1; i < size; ++i)
        value = array_route_min(value, a[py::int_(i)]);
    return value;
}

static py::object array_generic_hmax(const py::object &a) {
    size_t size = py::len(a);
    if (size == 0)
        ek::enoki_raise("hmax(): zero-sized array!");

    py::object value = a[py::int_(0)];
    for (size_t i = 1; i < size; ++i)
        value = array_route_max(value, a[py::int_(i)]);
    return value;
}

static py::object array_generic_dot(const py::object &a0, const py::object &a1) {
    size_t s0 = py::len(a0), s1 = py::len(a1), sr = std::max(s0, s1), si;
    array_check("dot", a0, a1, s0, s1, sr, si);
    if (sr == 0)
        ek::enoki_raise("dot(): zero-sized array!");

    py::int_ z(0);
    py::object value = a0[z] * a1[z];
    bool fp = jitc_is_floating_point(var_type(a0));
    for (size_t i_ = 1; i_ < sr; ++i_) {
        py::int_ i(i_);
        if (fp)
            value = array_route_fmadd(a0[s0 > 1 ? i : z], a1[s1 > 1 ? i : z], value);
        else
            value = value + a0[s0 > 1 ? i : z] * a1[s1 > 1 ? i : z];
    }
    return value;
}

static py::object array_route_hsum_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_hsum(a);
    return a;
}

static py::object array_route_hprod_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_hprod(a);
    return a;
}

static py::object array_route_hmin_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_hmin(a);
    return a;
}

static py::object array_route_hmax_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_hmax(a);
    return a;
}

static py::object array_route_hmean_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_hmean(a);
    return a;
}

static py::object array_route_all_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_all(a);
    return a;
}

static py::object array_route_any_nested(py::object a) {
    while (var_is_enoki(a))
        a = array_route_any(a);
    return a;
}

static py::object array_route_none_nested(const py::object &a) {
    return ~array_route_any_nested(a);
}

#define ENOKI_PY_REDUCTION_DEFAULT(name)                                       \
    static py::object array_route_##name##_or(const py::object &defval,        \
                                              const py::object &value) {       \
        if (var_is_enoki_jit(value))                                           \
            return defval;                                                     \
        else                                                                   \
            return array_route_##name(value);                                  \
    }

ENOKI_PY_REDUCTION_DEFAULT(any)
ENOKI_PY_REDUCTION_DEFAULT(any_nested)
ENOKI_PY_REDUCTION_DEFAULT(all)
ENOKI_PY_REDUCTION_DEFAULT(all_nested)
ENOKI_PY_REDUCTION_DEFAULT(none)
ENOKI_PY_REDUCTION_DEFAULT(none_nested)


ENOKI_PY_UNARY_OPERATION(neg, -a0[i0])
ENOKI_PY_UNARY_OPERATION(not, ~a0[i0])
ENOKI_PY_ROUTE_UNARY(neg, test_arithmetic)
ENOKI_PY_ROUTE_UNARY(not, test_any)

ENOKI_PY_BINARY_OPERATION(add, a0[i0] + a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(iadd, a0[i0] += a1[i1])
ENOKI_PY_ROUTE_BINARY(add, test_arithmetic)
ENOKI_PY_ROUTE_BINARY(iadd, test_arithmetic)

ENOKI_PY_BINARY_OPERATION(sub, a0[i0] - a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(isub, a0[i0] -= a1[i1])
ENOKI_PY_ROUTE_BINARY(sub, test_arithmetic)
ENOKI_PY_ROUTE_BINARY(isub, test_arithmetic)

ENOKI_PY_BINARY_OPERATION(mul, a0[i0] * a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(imul, a0[i0] *= a1[i1])
ENOKI_PY_ROUTE_BINARY(mul, test_arithmetic)
ENOKI_PY_ROUTE_BINARY(imul, test_arithmetic)

ENOKI_PY_BINARY_OPERATION(truediv, a0[i0] / a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(itruediv, a0[i0] /= a1[i1])

ENOKI_PY_BINARY_OPERATION(floordiv, py::reinterpret_steal<py::object>(
    PyNumber_FloorDivide(a0[i0].ptr(), a1[i1].ptr())))

ENOKI_PY_BINARY_OPERATION_COMPOUND(ifloordiv, py::reinterpret_steal<py::object>(
    PyNumber_InPlaceFloorDivide(a0[i0].ptr(), a1[i1].ptr())))

static py::object array_route_truediv(py::object &a0, py::object &a1) {
    PyTypeObject *t0 = (PyTypeObject *) a1.get_type().ptr();
    if (t0 == &PyFloat_Type || t0 == &PyLong_Type) {
        py::float_ one(1.f);
        if (!a1.equal(one)) {
            a1 = one / a1;
            return array_route_mul(a0, a1);
        } else {
            return a0;
        }
    }

    var_promote("truediv", a0, a1, true, test_float_truediv);

    return a0.attr("truediv_")(a1);
}

static py::object array_route_itruediv(py::object &a0, py::object &a1) {
    PyTypeObject *t0 = (PyTypeObject *) a1.get_type().ptr();
    if (t0 == &PyFloat_Type || t0 == &PyLong_Type) {
        py::float_ one(1.f);
        if (!a1.equal(one)) {
            a1 = one / a1;
            return array_route_imul(a0, a1);
        } else {
            return a0;
        }
    }

    var_promote("itruediv", a0, a1, true, test_float_truediv);

    return a0.attr("itruediv_")(a1);
}

static py::object array_route_floordiv(py::object &a0, py::object &a1) {
    PyTypeObject *t0 = (PyTypeObject *) a1.get_type().ptr();
    if (t0 == &PyLong_Type) {
        py::int_ one(1), m_one(-1);
        if (a1.equal(one)) {
            return a0;
        } else if (a1.equal(m_one)) {
            return -a0;
        } else {
            // divisor<uint32_t> div(py::cast<uint32_t>(a1));
            // py::object q = array_route_mulhi(py::cast(div.multiplier), a0);
            // py::object t = py_route_shift(a0 - q, py::int_(1)) + q;
            // return py_route_shift(t, py::int_(div.shift));
        }
    }

    var_promote("floordiv", a0, a1, true, test_integral_floordiv);
    return a0.attr("floordiv_")(a1);
}

static py::object array_route_ifloordiv(py::object &a0, py::object &a1) {
    var_promote("ifloordiv", a0, a1, true, test_integral_floordiv);
    return a0.attr("ifloordiv_")(a1);
}

ENOKI_PY_BINARY_OPERATION(mod, a0[i0] % a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(imod, a0[i0] %= a1[i1])
ENOKI_PY_ROUTE_BINARY(mod, test_integral)
ENOKI_PY_ROUTE_BINARY(imod, test_integral)

ENOKI_PY_BINARY_OPERATION(sl, a0[i0] << a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(isl, a0[i0] <<= a1[i1])
ENOKI_PY_ROUTE_BINARY(sl, test_integral)
ENOKI_PY_ROUTE_BINARY(isl, test_integral)

ENOKI_PY_BINARY_OPERATION(sr, a0[i0] >> a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND(isr, a0[i0] >>= a1[i1])
ENOKI_PY_ROUTE_BINARY(sr, test_integral)
ENOKI_PY_ROUTE_BINARY(isr, test_integral)

ENOKI_PY_BINARY_OPERATION_BITOP(and, a0[i0] & a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND_BITOP(iand, a0[i0] &= a1[i1])
ENOKI_PY_ROUTE_BINARY_BITOP(and, test_any)
ENOKI_PY_ROUTE_BINARY_BITOP(iand, test_any)

ENOKI_PY_BINARY_OPERATION_BITOP(or, a0[i0] | a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND_BITOP(ior, a0[i0] |= a1[i1])
ENOKI_PY_ROUTE_BINARY_BITOP(or, test_any)
ENOKI_PY_ROUTE_BINARY_BITOP(ior, test_any)

ENOKI_PY_BINARY_OPERATION_BITOP(xor, a0[i0] ^ a1[i1])
ENOKI_PY_BINARY_OPERATION_COMPOUND_BITOP(ixor, a0[i0] ^= a1[i1])
ENOKI_PY_ROUTE_BINARY_BITOP(xor, test_any)
ENOKI_PY_ROUTE_BINARY_BITOP(ixor, test_any)

py::object py_compare(py::handle a, py::handle b, int mode) {
    py::object p = py::reinterpret_steal<py::object>(
        PyObject_RichCompare(a.ptr(), b.ptr(), mode));
    if (!p.ptr())
        throw py::error_already_set();
    return p;
}

py::object var_mask(const py::object &base) {
    return base.attr("ReplaceScalar")(VarType::Bool);
}

ENOKI_PY_BINARY_OPERATION_MASK(lt, py_compare(a0[i0], a1[i1], Py_LT))
ENOKI_PY_BINARY_OPERATION_MASK(le, py_compare(a0[i0], a1[i1], Py_LE))
ENOKI_PY_BINARY_OPERATION_MASK(gt, py_compare(a0[i0], a1[i1], Py_GT))
ENOKI_PY_BINARY_OPERATION_MASK(ge, py_compare(a0[i0], a1[i1], Py_GE))

ENOKI_PY_ROUTE_BINARY(lt, test_arithmetic)
ENOKI_PY_ROUTE_BINARY(le, test_arithmetic)
ENOKI_PY_ROUTE_BINARY(gt, test_arithmetic)
ENOKI_PY_ROUTE_BINARY(ge, test_arithmetic)

py::object array_route_select(py::object a0, py::object a1, py::object a2) {
    if (!var_promote_select(a0, a1, a2))
        return py::cast<bool>(a0) ? a1 : a2;
    return a1.attr("select_")(a0, a1, a2);
}

static py::object array_generic_select(
    const py::object &a0, const py::object &a1, const py::object &a2) {
    size_t s0 = py::len(a0), s1 = py::len(a1), s2 = py::len(a2),
           sr = std::max({ s0, s1, s2 }), si;
    array_check_select("select", a0, a1, a2, s0, s1, s2, si);
    py::object ar = a1.attr("empty_")(si);
    py::int_ z(0);
    for (size_t i_ = 0; i_ < sr; ++i_) {
        py::int_ i(i_);
        py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z,
                   i2 = s2 > 1 ? i : z;
        ar[i] = array_route_select(a0[i0], a1[i1], a2[i2]);
    }
    return ar;
}

static py::object array_route_empty(py::handle h, size_t size = 1) {
    if (var_is_enoki_type(h))
        return h.attr("empty_")(size);
    else
        return h(0);
}

py::object array_route_zero(py::handle h, size_t size = 1) {
    if (var_is_enoki_type(h))
        return h.attr("zero_")(size);
    else
        return h(0);
}

static py::object array_route_full(const py::object &o, const py::object &value, size_t size) {
    if (var_is_enoki_type(o))
        return o.attr("full_")(value, size);
    else
        return o(value);
}

static py::object array_route_linspace(const py::object &o,
                                       const py::object &min,
                                       const py::object &max, size_t size) {
    if (var_is_enoki_type(o))
        return o.attr("linspace_")(min, max, size);
    else
        return min;
}

static py::object array_route_arange_1(const py::object &o, size_t size) {
    if (var_is_enoki_type(o))
        return o.attr("arange_")(0, size, 1);
    else
        return py::int_(0);
}

static py::object array_route_arange_2(const py::object &o, ssize_t start, ssize_t end, ssize_t step) {
    if (var_is_enoki_type(o))
        return o.attr("arange_")(start, end, step);
    else
        return py::int_(start);
}

static py::object array_route_eq(py::object a0, py::object a1) {
    if (var_promote("eq", a0, a1, false, test_any))
        return a0.attr("eq_")(a1);
    else
        return py::cast(a0.equal(a1));
}

static py::object array_route_neq(py::object a0, py::object a1) {
    if (var_promote("neq", a0, a1, false, test_any))
        return a0.attr("neq_")(a1);
    else
        return py::cast(a0.not_equal(a1));
}

static py::object array_eq_reduce(py::object a0, py::object a1) {
    return array_route_all_nested(array_route_eq(a0, a1));
}

static py::object array_neq_reduce(py::object a0, py::object a1) {
    return array_route_any_nested(array_route_neq(a0, a1));
}

ENOKI_PY_BINARY_OPERATION_MASK(eq, array_route_eq(a0[i0], a1[i1]))
ENOKI_PY_BINARY_OPERATION_MASK(neq, array_route_neq(a0[i0], a1[i1]))

py::object array_route_abs(const py::object &a) {
    if (var_is_enoki(a)) {
        test_arithmetic("abs", var_type(a));
        return a.attr("abs_")();
    } else {
        return py::reinterpret_steal<py::object>(PyNumber_Absolute(a.ptr()));
    }
}
ENOKI_PY_ROUTE_UNARY_FLOAT(sqrt, ek::sqrt(a0d))
ENOKI_PY_ROUTE_UNARY_FLOAT(rcp, ek::rcp(a0d))
ENOKI_PY_ROUTE_UNARY_FLOAT(rsqrt, ek::rsqrt(a0d))

ENOKI_PY_UNARY_OPERATION(abs, array_route_abs(a0[i0]))
ENOKI_PY_UNARY_OPERATION(sqrt, array_route_sqrt(a0[i0]))
ENOKI_PY_UNARY_OPERATION(rcp, array_route_rcp(a0[i0]))
ENOKI_PY_UNARY_OPERATION(rsqrt, array_route_rsqrt(a0[i0]))

static py::object array_route_sqr(const py::object &a) {
    return a * a;
}

ENOKI_PY_ROUTE_TERNARY_FLOAT(fmadd, ek::fmadd(a0d, a1d, a2d))
ENOKI_PY_ROUTE_TERNARY_FLOAT(fmsub, ek::fmsub(a0d, a1d, a2d))
ENOKI_PY_ROUTE_TERNARY_FLOAT(fnmadd, ek::fnmadd(a0d, a1d, a2d))
ENOKI_PY_ROUTE_TERNARY_FLOAT(fnmsub, ek::fnmsub(a0d, a1d, a2d))

ENOKI_PY_TERNARY_OPERATION(fmadd, array_route_fmadd(a0[i0], a1[i1], a2[i2]))
ENOKI_PY_TERNARY_OPERATION(fmsub, array_route_fmsub(a0[i0], a1[i1], a2[i2]))
ENOKI_PY_TERNARY_OPERATION(fnmadd, array_route_fnmadd(a0[i0], a1[i1], a2[i2]))
ENOKI_PY_TERNARY_OPERATION(fnmsub, array_route_fnmsub(a0[i0], a1[i1], a2[i2]))


static py::object array_route_max(py::object a0, py::object a1) {
    if (var_promote("max", a0, a1, false, test_arithmetic))
        return a0.attr("max_")(a1);
    else
        return a0 >= a1 ? a0 : a1;
}

static py::object array_route_min(py::object a0, py::object a1) {
    if (var_promote("min", a0, a1, false, test_arithmetic))
        return a0.attr("min_")(a1);
    else
        return a0 < a1 ? a0 : a1;
}

ENOKI_PY_BINARY_OPERATION(min, array_route_min(a0[i0], a1[i1]))
ENOKI_PY_BINARY_OPERATION(max, array_route_max(a0[i0], a1[i1]))

static py::object array_route_reinterpret(const py::object &target_type, const py::object &source,
                                          VarType vt_target, VarType vt_source) {
    if (!PyType_Check(target_type.ptr()))
        ek::enoki_raise("array_reinterpret(): requires a type argument!");
    if (var_is_enoki_type(target_type)) {
        return target_type.attr("reinterpret_array_")(source);
    } else {
        if (vt_source == VarType::Invalid || vt_target == VarType::Invalid)
            ek::enoki_raise("reinterpret_array(): missing VarType information for scalar cast!");
        else if (var_type_size[(int) vt_source] != var_type_size[(int) vt_target])
            ek::enoki_raise("reinterpret_array(): input/output size mismatch!");

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
            case VarType::Float32:  temp = (uint64_t) ek::memcpy_cast<uint32_t>(py::cast<float>(source)); break;
            case VarType::Float64:  temp = (uint64_t) ek::memcpy_cast<uint64_t>(py::cast<double>(source)); break;
            default: throw py::type_error("reinterpret_array(): unsupported input type!");
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
            case VarType::Float32: return py::cast(ek::memcpy_cast<float>((uint32_t) temp));
            case VarType::Float64: return py::cast(ek::memcpy_cast<double>((uint64_t) temp));
            default: throw py::type_error("reinterpret_array(): unsupported output type!");
        }
    }
}

void export_route_basics(py::module &m) {
    py::enum_<VarType>(m, "VarType")
        .value("Invalid", VarType::Invalid)
        .value("Int8", VarType::Int8)
        .value("UInt8", VarType::UInt8)
        .value("Int16", VarType::Int16)
        .value("UInt16", VarType::UInt16)
        .value("Int32", VarType::Int32)
        .value("UInt32", VarType::UInt32)
        .value("Int64", VarType::Int64)
        .value("UInt64", VarType::UInt64)
        .value("Float16", VarType::Float16)
        .value("Float32", VarType::Float32)
        .value("Float64", VarType::Float64)
        .def_property_readonly(
            "Size", [](VarType v) { return var_type_size[(int) v]; });

    py::class_<ek::detail::reinterpret_flag>(m, "reinterpret_flag");

    auto base = py::class_<ek::ArrayBase>(m, "ArrayBase")
        .def_property("x",
            [](const py::object &o) {
                return array_getitem(o, 0);
            },
            [](const py::object &o, const py::object &v) {
                return array_setitem(o, 0, v);
            })
        .def_property("y",
            [](const py::object &o) {
                return array_getitem(o, 1);
            },
            [](const py::object &o, const py::object &v) {
                return array_setitem(o, 1, v);
            })
        .def_property("z",
            [](const py::object &o) {
                return array_getitem(o, 2);
            },
            [](const py::object &o, const py::object &v) {
                return array_setitem(o, 2, v);
            })
        .def_property("w",
            [](const py::object &o) {
                return array_getitem(o, 3);
            },
            [](const py::object &o, const py::object &v) {
                return array_setitem(o, 3, v);
            })
        .def("__getitem__", &array_getitem)
        .def("__getitem__", &array_getitem_tuple)
        .def("__setitem__", &array_setitem)
        .def("__setitem__", &array_setitem_tuple)
        .def("__neg__", &array_route_neg)
        .def("neg_", &array_generic_neg)
        .def("__invert__", &array_route_not)
        .def("not_", &array_generic_not)
        .def("__add__", &array_route_add)
        .def("__radd__", &array_route_add)
        .def("__iadd__", &array_route_iadd)
        .def("add_", &array_generic_add)
        .def("iadd_", &array_generic_iadd)
        .def("__sub__", &array_route_sub)
        .def("__rsub__", &array_route_sub)
        .def("__isub__", &array_route_isub)
        .def("sub_", &array_generic_sub)
        .def("isub_", &array_generic_isub)
        .def("__mul__", &array_route_mul)
        .def("__rmul__", &array_route_mul)
        .def("__imul__", &array_route_imul)
        .def("mul_", &array_generic_mul)
        .def("imul_", &array_generic_imul)
        .def("__truediv__", &array_route_truediv)
        .def("__rtruediv__", &array_route_truediv)
        .def("__itruediv__", &array_route_itruediv)
        .def("truediv_", &array_generic_truediv)
        .def("itruediv_", &array_generic_itruediv)
        .def("__floordiv__", &array_route_floordiv)
        .def("__rfloordiv__", &array_route_floordiv)
        .def("__ifloordiv__", &array_route_ifloordiv)
        .def("floordiv_", &array_generic_floordiv)
        .def("ifloordiv_", &array_generic_ifloordiv)
        .def("__mod__", &array_route_mod)
        .def("__rmod__", &array_route_mod)
        .def("__imod__", &array_route_imod)
        .def("mod_", &array_generic_mod)
        .def("imod_", &array_generic_imod)
        .def("__lshift__", &array_route_sl)
        .def("__rlshift__", &array_route_sl)
        .def("__ilshift__", &array_route_isl)
        .def("sl_", &array_generic_sl)
        .def("isl_", &array_generic_isl)
        .def("__rshift__", &array_route_sr)
        .def("__rrshift__", &array_route_sr)
        .def("__irshift__", &array_route_isr)
        .def("sr_", &array_generic_sr)
        .def("isr_", &array_generic_isr)
        .def("__and__", &array_route_and)
        .def("__rand__", &array_route_and)
        .def("__iand__", &array_route_iand)
        .def("and_", &array_generic_and)
        .def("iand_", &array_generic_iand)
        .def("__or__", &array_route_or)
        .def("__ror__", &array_route_or)
        .def("__ior__", &array_route_ior)
        .def("or_", &array_generic_or)
        .def("ior_", &array_generic_ior)
        .def("__xor__", &array_route_xor)
        .def("__rxor__", &array_route_xor)
        .def("__ixor__", &array_route_ixor)
        .def("xor_", &array_generic_xor)
        .def("ixor_", &array_generic_ixor)
        .def("__lt__", &array_route_lt)
        .def("lt_", &array_generic_lt)
        .def("__le__", &array_route_le)
        .def("le_", &array_generic_le)
        .def("__gt__", &array_route_gt)
        .def("gt_", &array_generic_gt)
        .def("__ge__", &array_route_ge)
        .def("ge_", &array_generic_ge)
        .def("eq_", &array_generic_eq)
        .def("neq_", &array_generic_neq)
        .def("__eq__", &array_eq_reduce)
        .def("__neq__", &array_neq_reduce)
        .def("min_", &array_generic_min)
        .def("max_", &array_generic_max)
        .def("all_", &array_generic_all)
        .def("any_", &array_generic_any)
        .def("hsum_", &array_generic_hsum)
        .def("hprod_", &array_generic_hprod)
        .def("hmin_", &array_generic_hmin)
        .def("hmax_", &array_generic_hmax)
        .def("dot_", &array_generic_dot)
        .def("__abs__", &array_route_abs)
        .def("abs_", &array_generic_abs)
        .def("sqrt_", &array_generic_sqrt)
        .def("rsqrt_", &array_generic_rsqrt)
        .def("rcp_", &array_generic_rcp)
        .def("fmadd_", &array_generic_fmadd)
        .def("fmsub_", &array_generic_fmsub)
        .def("fnmadd_", &array_generic_fnmadd)
        .def("fnmsub_", &array_generic_fnmsub)
        .def_static("select_", &array_generic_select)
        .def("__bool__",
            [](const py::object &o) {
                throw std::runtime_error(
                    "To convert an Enoki array into a boolean value, use a mask reduction "
                    "operation such as enoki.all(), enoki.any(), enoki.none(). Special "
                    "variants (enoki.all_nested(), etc.) are available for nested arrays.");
            })
        .def("__repr__", &array_repr);

    base.attr("ReplaceScalar") = classmethod([](const py::object &cls,
                                                VarType v) -> py::object {
        std::string module = (py::str) cls.attr("__module__");
        return py::module::import(module.c_str())
            .attr(array_name(v, py::cast<size_t>(cls.attr("Depth")),
                             py::cast<size_t>(cls.attr("Size")),
                             module.find("scalar") != std::string::npos));
    });

    base.attr("reinterpret_array_") = classmethod([](const py::object &target,
                                                     const py::object &array) -> py::object {
        py::object target_type = target.attr("Value");
        VarType vt_source = py::cast<VarType>(array.attr("Type"));
        VarType vt_target = py::cast<VarType>(target.attr("Type"));

        size_t size = py::len(array), size_init;
        array_check("reinterpret_array", array, size, size_init);
        py::object result = target.attr("empty_")(size_init);
        for (size_t i_ = 0; i_ < size; ++i_) {
            py::int_ i0(i_);
            result[i0] = array_route_reinterpret(target_type, array[i0],
                                                 vt_target, vt_source);
        }
        return result;
    });

    base.attr("zero_") = classmethod([](const py::object &target,
                                        size_t size) -> py::object {
        py::object target_type = target.attr("Value");
        size_t size_init;
        array_check("zero", target, size, size_init);
        py::object result = target.attr("empty_")(size_init);
        size_t size_cur = py::len(result);
        for (size_t i_ = 0; i_ < size_cur; ++i_) {
            py::int_ i0(i_);
            result[i0] = array_route_zero(target_type, size);
        }
        return result;
    });

    base.attr("full_") = classmethod([](const py::object &target,
                                        const py::object &value,
                                        size_t size) -> py::object {
        py::object target_type = target.attr("Value");
        size_t size_init;
        array_check("full", target, size, size_init);
        py::object result = target.attr("empty_")(size_init);
        size_t size_cur = py::len(result);
        for (size_t i_ = 0; i_ < size_cur; ++i_) {
            py::int_ i0(i_);
            result[i0] = array_route_full(target_type, value, size);
        }
        return result;
    });

    base.attr("arange_") = classmethod([](const py::object &target,
                                          ssize_t start, ssize_t stop,
                                          ssize_t step) -> py::object {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
        py::object target_type = target.attr("Value");
        size_t size_init;
        array_check("arange", target, size, size_init);
        py::object result = target.attr("empty_")(size_init);
        size_t size_cur = py::len(result);
        for (size_t i_ = 0; i_ < size_cur; ++i_) {
            py::int_ i0(i_);
            result[i0] = target_type(start + (ssize_t) i_ * step);
        }
        return result;
    });

    base.attr("linspace_") = classmethod([](const py::object &target,
                                            py::object min, py::object max,
                                            size_t size) -> py::object {
        size_t size_init;
        py::object target_type = target.attr("Value");
        array_check("linspace", target, size, size_init);
        py::object result = target.attr("empty_")(size_init);
        size_t size_cur = py::len(result);
        py::object step;
        if (size_cur <= 1)
            step = target_type(0);
        else
            step = (max - min) / target_type(size_cur - 1);
        for (size_t i_ = 0; i_ < size_cur; ++i_) {
            py::int_ i0(i_);
            result[i0] = min + i0 * step;
        }
        return result;
    });

    m.def("shape", &array_shape);
    m.def("ndim", &array_ndim);
    m.def("ragged", &array_ragged);

    m.def("eq", &array_route_eq);
    m.def("neq", &array_route_neq);

    m.def("sqr", &array_route_sqr);
    m.def("abs", &array_route_abs);
    m.def("sqrt", &array_route_sqrt);
    m.def("rsqrt", &array_route_rsqrt);
    m.def("rcp", &array_route_rcp);

    m.def("min", &array_route_min);
    m.def("max", &array_route_max);

    m.def("fmadd", &array_route_fmadd);
    m.def("fmsub", &array_route_fmsub);
    m.def("fnmadd", &array_route_fnmadd);
    m.def("fnmsub", &array_route_fnmsub);

    m.def("hsum", &array_route_hsum);
    m.def("hprod", &array_route_hprod);
    m.def("hmin", &array_route_hmin);
    m.def("hmax", &array_route_hmax);
    m.def("hmean", &array_route_hmean);
    m.def("dot", &array_route_dot);

    m.def("hsum_async", &array_route_hsum_async);
    m.def("hprod_async", &array_route_hprod_async);
    m.def("hmin_async", &array_route_hmin_async);
    m.def("hmax_async", &array_route_hmax_async);
    m.def("hmean_async", &array_route_hmean_async);
    m.def("dot_async", &array_route_dot_async);

    m.def("hsum_nested", &array_route_hsum_nested);
    m.def("hprod_nested", &array_route_hprod_nested);
    m.def("hmin_nested", &array_route_hmin_nested);
    m.def("hmax_nested", &array_route_hmax_nested);
    m.def("hmean_nested", &array_route_hmean_nested);

    m.def("all", &array_route_all);
    m.def("any", &array_route_any);
    m.def("none", &array_route_none);
    m.def("all_or", &array_route_all_or);
    m.def("any_or", &array_route_any_or);
    m.def("none_or", &array_route_none_or);

    m.def("all_nested", &array_route_all_nested);
    m.def("any_nested", &array_route_any_nested);
    m.def("none_nested", &array_route_none_nested);
    m.def("all_nested_or", &array_route_all_nested_or);
    m.def("any_nested_or", &array_route_any_nested_or);
    m.def("none_nested_or", &array_route_none_nested_or);

    m.def("select", &array_route_select);

    m.def("empty", &array_route_empty, "type"_a, "size"_a = 1);
    m.def("zero", &array_route_zero, "type"_a, "size"_a = 1);
    m.def("full", &array_route_full, "type"_a, "value"_a, "size"_a = 1);
    m.def("linspace", &array_route_linspace, "type"_a, "min"_a, "max"_a, "size"_a = 1);
    m.def("arange", &array_route_arange_1, "type"_a, "size"_a = 1);
    m.def("arange", &array_route_arange_2, "type"_a, "start"_a, "end"_a, "step"_a = 1);

    m.def("reinterpret_array", &array_route_reinterpret, "target_type"_a,
          "source"_a, "vt_target"_a = VarType::Invalid,
          "vt_source"_a = VarType::Invalid);

    m.def("_coeff_evals", []() { return coeff_evals; });

    m.def("print_threshold", []() { return print_threshold; });
    m.def("set_print_threshold",
          [](size_t size) { print_threshold = std::max(size, (size_t) 11); });

    py::register_exception<enoki::Exception>(m, "Exception");
}
