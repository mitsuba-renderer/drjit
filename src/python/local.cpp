/*
    local.cpp -- Python bindings for Dr.Jit-Core variable arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "local.h"
#include "apply.h"
#include "base.h"
#include "init.h"
#include "meta.h"

/// Abstract callback declaration used by traverse() below
struct Callback {
    virtual nb::object process(nb::handle tp, nb::handle v1, nb::handle v2) = 0;
};

/// Forward declaration
static nb::object traverse(nb::handle tp, nb::handle v1, nb::handle v2,
                           bool ret, Callback *cb);

Local::Local(nb::handle dtype, size_t length, nb::handle value)
    : m_dtype(nb::borrow(dtype)), m_length(length),
      m_value(value.is_none() ? nb::object() : nb::borrow(value)) {

    /// Allocate variable arrays for the input PyTree
    struct LocalCallback : Callback {
        size_t length;
        dr::vector<uint32_t> arrays;
        JitBackend backend = JitBackend::None;

        LocalCallback(size_t length) : length(length) {}

        nb::object process(nb::handle tp, nb::handle v1, nb::handle) override {
            const ArraySupplement &s = supp(tp);
            backend = (JitBackend) s.backend;
            uint32_t result;
            if (v1.is_valid()) {
                uint32_t i1 = (uint32_t) s.index(inst_ptr(v1));
                size_t size = jit_var_size(i1);
                uint32_t i2 = jit_array_create((JitBackend) s.backend,
                                               (VarType) s.type, size, length);
                result = jit_array_init(i2, i1);
                jit_var_dec_ref(i2);
            } else {
                result = jit_array_create((JitBackend) s.backend,
                                          (VarType) s.type, 1, length);
            }
            arrays.push_back(result);
            return nb::object();
        }
    };

    LocalCallback cb(m_length);

    traverse(m_dtype, m_value, nb::handle(), false, &cb);
    m_arrays = std::move(cb.arrays);
    m_backend = cb.backend;

    if (m_backend == JitBackend::None)
        nb::raise_type_error("dr.local(%s): type does not contain any Jit-tracked arrays!",
                             nb::type_name(dtype).c_str());

    ArrayMeta m{};
    m.backend = (uint32_t) m_backend;
    m.ndim = 1;
    m.type = (uint32_t) VarType::UInt32;
    m_index_tp = meta_get_type(m);
    m.type = (uint32_t) VarType::Bool;
    m_mask_tp = meta_get_type(m);
}

Local::Local(const Local &l) : m_dtype(l.m_dtype), m_length(l.m_length), m_value(l.m_value), m_backend(l.m_backend), m_index_tp(l.m_index_tp), m_mask_tp(l.m_mask_tp) {
    m_arrays.reserve(l.m_arrays.size());
    for (uint32_t index: l.m_arrays) {
        jit_var_inc_ref(index);
        m_arrays.push_back(index);
    }
}

Local::~Local() {
    for (uint32_t index : m_arrays)
        jit_var_dec_ref(index);
}

nb::object Local::read(nb::handle index_, nb::handle mask_) const {
    nb::object index =
        index_.type().is(m_index_tp) ? nb::borrow(index_) : m_index_tp(index_);
    nb::object mask =
        mask_.type().is(m_mask_tp) ? nb::borrow(mask_) : m_mask_tp(mask_);

    /// Read from the variable arrays
    struct GetItemCallback : Callback {
        const dr::vector<uint32_t> &arrays;
        uint32_t index, mask, ctr;
        GetItemCallback(const dr::vector<uint32_t> &arrays, uint32_t index,
                        uint32_t mask)
            : arrays(arrays), index(index), mask(mask), ctr(0) {}

        nb::object process(nb::handle tp, nb::handle, nb::handle) override {
            const ArraySupplement &s = supp(tp);
            if (ctr >= arrays.size())
                nb::raise("Local.read(): internal error, ran out of "
                          "variable arrays!");
            uint32_t result = jit_array_read(arrays[ctr++], index, mask);
            nb::object result_o = nb::inst_alloc(tp);
            s.init_index(result, inst_ptr(result_o));
            nb::inst_mark_ready(result_o);
            jit_var_dec_ref(result);
            return result_o;
        }
    };

    GetItemCallback cb(m_arrays,
                       (uint32_t) supp(m_index_tp).index(inst_ptr(index)),
                       (uint32_t) supp(m_mask_tp).index(inst_ptr(mask)));

    nb::object result = traverse(m_dtype, m_value, nb::handle(), true, &cb);

    if (cb.ctr != m_arrays.size())
        nb::raise("Local.read(): internal error, did not access all variable "
                  "arrays!");

    return result;
}

void Local::write(nb::handle index_, nb::handle value_, nb::handle mask_) {
    nb::object index =
        index_.type().is(m_index_tp) ? nb::borrow(index_) : m_index_tp(index_);
    nb::object mask =
        mask_.type().is(m_mask_tp) ? nb::borrow(mask_) : m_mask_tp(mask_);
    nb::object value =
        value_.type().is(m_dtype) ? nb::borrow(value_) : m_dtype(value_);

    /// Write to the variable arrays
    struct SetItemCallback : Callback {
        dr::vector<uint32_t> &arrays;
        uint32_t index, mask, ctr;
        SetItemCallback(dr::vector<uint32_t> &arrays, uint32_t index,
                        uint32_t mask)
            : arrays(arrays), index(index), mask(mask), ctr(0) {}

        nb::object process(nb::handle tp, nb::handle,
                           nb::handle value) override {
            const ArraySupplement &s = supp(tp);
            if (ctr >= arrays.size())
                nb::raise("Local.write(): internal error, ran out of "
                          "variable arrays!");
            uint64_t value_i = s.index(inst_ptr(value));
            if (value_i >> 32)
                nb::raise("Local memory writes are not differentiable. You "
                          "must use 'drjit.detach()' to disable gradient "
                          "tracking of the written value.");

            uint32_t result =
                jit_array_write(arrays[ctr], index, (uint32_t) value_i, mask);
            jit_var_dec_ref(arrays[ctr]);
            arrays[ctr++] = result;
            return nb::object();
        }
    };

    SetItemCallback cb(m_arrays,
                       (uint32_t) supp(m_index_tp).index(inst_ptr(index)),
                       (uint32_t) supp(m_mask_tp).index(inst_ptr(mask)));

    traverse(m_dtype, m_value, value, false, &cb);

    if (cb.ctr != m_arrays.size())
        nb::raise("Local.write(): internal error, did not access all variable "
                  "arrays!");
}

nb::str Local::repr() const {
    return nb::str("Local[{}, {}]").format(nb::type_name(m_dtype), m_length);
}

static void raise_size_mismatch_error(nb::handle tp, size_t size1,
                                      size_t size2) {
    nb::raise("drjit.Local: dynamically sized array of type '%s' has an "
              "inconsistent size (%zu vs %zu)",
              nb::type_name(tp).c_str(), size1, size2);
}

static void raise_key_mismatch_error(nb::handle keys_1, nb::handle keys_2) {
    nb::raise("drjit.Local: dictionary has inconsistnt keys (%s vs %s)",
              nb::str(keys_1).c_str(), nb::str(keys_2).c_str());
}

static void raise_dynamic_size_error(nb::handle tp) {
    nb::raise("drjit.Local(): 'dtype' contains a dynamically sized field "
              "of type '%s'. You must specify a default (via the 'value' "
              "argument) to specify its size",
              nb::type_name(tp).c_str());
}

/// Walk through a trio of PyTrees (type, value1, value2) and invoke a
/// callback at leaf variables. We're not using the existing traversal
/// helpers from 'apply.h' since the needs here are slightly different (in
/// particular, the 'tp' argument is a type, not a value)
static nb::object traverse(nb::handle tp, nb::handle v1, nb::handle v2,
                           bool ret, Callback *cb) {
    if (!tp.is_type()) {
        tp = extract_type(nb::borrow(tp));
        if (!tp.is_type() && v1.is_valid())
            tp = v1.type();
    }
    if (!tp.is_type())
        nb::raise("drjit.Local: encountered '%s' while traversing 'dtype', "
                  "which is not a valid Python type object",
                  nb::str(tp).c_str());

    if (v1.is_valid() && !v1.type().is(tp))
        nb::raise("drjit.Local: detected an inconsistency while traversing "
                  "'dtype' and 'value' (expected type '%s', got '%s')",
                  nb::type_name(tp).c_str(), nb::inst_name(v1).c_str());
    if (v2.is_valid() && !v2.type().is(tp))
        nb::raise("drjit.Local: detected an inconsistency while traversing "
                  "'dtype' and 'value' (expected type '%s', got '%s')",
                  nb::type_name(tp).c_str(), nb::inst_name(v2).c_str());

    nb::object result;

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.backend == (uint16_t) JitBackend::None) {
            /* Do nothing */
        } else if (s.is_tensor) {
            nb::raise("drjit.Local: tensor types are not permitted");
        } else if (s.ndim == 1) {
            result = cb->process(tp, v1, v2);
        } else if (s.ndim > 1) {
            size_t size = s.shape[0];
            ArrayBase *result_p = nullptr;

            if (ret) {
                result = nb::inst_alloc_zero(tp);
                result_p = inst_ptr(result);
            }

            if (size == DRJIT_DYNAMIC) {
                if (v1.is_valid()) {
                    size = s.len(inst_ptr(v1));
                    if (v2.is_valid()) {
                        size_t size2 = s.len(inst_ptr(v2));
                        if (size != size2)
                            raise_size_mismatch_error(tp, size, size2);
                    }
                    if (ret)
                        s.init(size, result_p);
                } else {
                    raise_dynamic_size_error(tp);
                }
            }

            for (size_t i = 0; i < size; ++i) {
                nb::object v1_i = v1.is_valid() ? nb::steal(s.item(v1.ptr(), i))
                                                : nb::object();
                nb::object v2_i = v2.is_valid() ? nb::steal(s.item(v2.ptr(), i))
                                                : nb::object();
                nb::object r_i = traverse(s.value, v1_i, v2_i, ret, cb);
                if (ret)
                    s.set_item(result.ptr(), i, r_i.ptr());
            }
        }
    } else if (tp.is(&PyTuple_Type) || tp.is(&PyList_Type)) {
        if (!v1.is_valid())
            raise_dynamic_size_error(tp);
        size_t size = nb::len(v1);
        if (v2.is_valid()) {
            size_t size2 = nb::len(v2);
            if (size != size2)
                raise_size_mismatch_error(tp, size, size2);
        }

        if (ret)
            result = nb::list();

        for (size_t i = 0; i < size; ++i) {
            nb::object v1_i = v1[i];
            nb::object v2_i = v2.is_valid() ? v2[i] : nb::object();
            nb::object r_i = traverse(v1_i.type(), v1_i, v2_i, ret, cb);
            if (ret)
                nb::borrow<nb::list>(result).append(r_i);
        }

        if (ret && tp.is(&PyTuple_Type))
            result = nb::tuple(result);
    } else if (tp.is(&PyDict_Type)) {
        if (!v1.is_valid())
            raise_dynamic_size_error(tp);

        nb::object keys = nb::borrow<nb::dict>(v1).keys();
        if (v2.is_valid()) {
            nb::object keys_2 = nb::borrow<nb::dict>(v2).keys();
            if (!keys.equal(keys_2))
                raise_key_mismatch_error(keys, keys_2);
        }

        if (ret)
            result = nb::dict();

        for (nb::handle k : keys) {
            nb::object v1_k = v1[k];
            nb::object v2_k = v2.is_valid() ? v2[k] : nb::object();
            nb::object r_k = traverse(v1_k.type(), v1_k, v2_k, ret, cb);

            if (ret)
                result[k] = r_k;
        }
    } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
        if (ret)
            result = tp();
        for (auto [k, v] : ds) {
            nb::object v1_k = v1.is_valid() ? nb::getattr(v1, k) : nb::object();
            nb::object v2_k = v2.is_valid() ? nb::getattr(v2, k) : nb::object();
            nb::object r_k = traverse(v, v1_k, v2_k, ret, cb);
            if (ret)
                nb::setattr(result, k, r_k);
        }
    } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
        if (ret)
            result = nb::dict();
        nb::dict d;
        for (auto field : df) {
            nb::object k = field.attr(DR_STR(name));
            nb::object v = field.attr(DR_STR(type));
            nb::object v1_k = v1.is_valid() ? nb::getattr(v1, k) : nb::object();
            nb::object v2_k = v2.is_valid() ? nb::getattr(v2, k) : nb::object();
            nb::object r_k = traverse(v, v1_k, v2_k, ret, cb);
            if (ret)
                result[k] = r_k;
        }
        if (ret)
            result = tp(**result);
    } else if (tp.is(nb::type<Local>())) {
        nb::raise("drjit.Local: 'dtype' may not contain nested "
                  "'drjit.Local' instances.");
    } else if (ret) {
        result = nb::none();
    }

    return result;
}

nb::handle local_type;

void export_local(nb::module_ &m) {
    local_type = nb::class_<Local>(m, "Local", nb::is_generic(),
                                   nb::sig("class Local(typing.Generic[T])"), doc_Local)
        .def(nb::init<Local>(), doc_Local_Local)
        .def("__len__", &Local::len, doc_Local___len__)
        .def("__getitem__",
             [](Local &a, nb::object index) {
                 return a.read(std::move(index), a.mask_type()(true));
             },
             nb::sig("def __getitem__(self, arg: int | AnyArray, /) -> T"),
             doc_Local___getitem__)
        .def("__setitem__",
             [](Local &a, nb::object index, nb::object value) {
                 a.write(std::move(index), value, a.mask_type()(true));
             },
             nb::sig("def __setitem__(self, arg0: int | AnyArray, arg1: T, /) -> None"),
             doc_Local___setitem__)
        .def("read", &Local::read, "index"_a, "active"_a = true,
             nb::sig("def read(self, index: int | AnyArray, active: bool | "
                     "AnyArray = True) -> T"), doc_Local_read)
        .def("write", &Local::write, "index"_a, "value"_a, "active"_a = true,
             nb::sig("def write(self, index: int | AnyArray, value: T, active: "
                     "bool | AnyArray = True) -> None"), doc_Local_write)
        .def("__repr__", &Local::repr);

    m.def(
        "alloc_local",
        [](nb::type_object h, size_t size, nb::handle value) {
            return new Local(h, size, value);
        },
        "dtype"_a, "size"_a, "value"_a = nb::none(),
        nb::rv_policy::take_ownership,
        nb::sig("def alloc_local(dtype: type[T], size: int, value: T | None = None) "
                "-> Local[T]"),
        doc_alloc_local
    );
}
