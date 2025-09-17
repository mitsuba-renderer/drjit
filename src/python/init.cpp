/*
    init.cpp -- Implementation of <Dr.Jit array>.__init__() and
    other initializion routines like dr.zero(), dr.empty(), etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <drjit-core/half.h>
#include <nanobind/ndarray.h>
#include "../ext/nanobind/src/buffer.h"
#include "drjit/python.h"
#include "meta.h"
#include "base.h"
#include "memop.h"
#include "shape.h"
#include "dlpack.h"
#include "init.h"
#include "coop_vec.h"

/// Forward declaration
static bool array_init_from_seq(PyObject *self, const ArraySupplement &s, PyObject *seq);

/// Convenience function to skip costly examinations of common types
static bool is_builtin(PyTypeObject *tp) {
    return tp == &PyLong_Type || tp == &PyFloat_Type || tp == &PyBool_Type;
}

/// Constructor for all dr.ArrayBase subclasses (except tensors)
int tp_init_array(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
    PyTypeObject *self_tp = Py_TYPE(self);
    const ArraySupplement &s = supp(self_tp);
    Py_ssize_t argc = NB_TUPLE_GET_SIZE(args);
    ArraySupplement::SetItem set_item = s.set_item;
    bool do_flip_axes = false;

    try {
        if (kwds) {
            PyObject *flip_axes = PyDict_GetItemString(kwds, "flip_axes");
            if (!flip_axes || PyDict_Size(kwds) == 0)
                raise_if(kwds, "Unknown keyword argument.");
            do_flip_axes = flip_axes == Py_True;
        }

        if (argc == 0) {
            // Default initialization, e.g., ``Array3f()``
            nb::detail::nb_inst_zero(self);
            return 0;
        } else if (argc > 1) {
            // Initialize from argument list, e.g., ``Array3f(1, 2, 3)``
            raise_if(!array_init_from_seq(self, s, args),
                     "Could not initialize array from argument list.");
            return 0;
        } else {
            // Initialize from a single element, e.g., ``Array3f(other_array)``
            // or ``Array3f(1.0)``
            PyObject *arg = NB_TUPLE_GET_ITEM(args, 0);
            PyTypeObject *arg_tp = Py_TYPE(arg);
            bool try_sequence_import = true,
                 is_drjit_tensor = false;
            bool arg_is_drjit = is_drjit_type(arg_tp);

            // Initialization from another Dr.Jit array
            if (arg_is_drjit) {
                const ArraySupplement &s_arg = supp(arg_tp);
                // Copy-constructor
                if (arg_tp == self_tp) {
                    nb::detail::nb_inst_copy(self, arg);
                    return 0;
                } else if (s_arg.is_tensor) {
                    is_drjit_tensor = true;
                } else {
                    ArrayMeta m_self = s,
                              m_arg  = s_arg;

                    // Convert AD <-> non-AD 1D types without casting
                    ArrayMeta m_temp = s_arg;
                    m_temp.is_diff = m_self.is_diff;
                    if (m_temp == m_self && m_self.ndim == 1 && s_arg.index) {
                        uint32_t index = (uint32_t) s_arg.index(inst_ptr(arg));
                        s.init_index(index, inst_ptr(self));
                        nb::inst_mark_ready(self);
                        return 0;
                    }

                    // Convert AD <-> non-AD 1D types with casting
                    m_temp = s_arg;
                    if ((m_temp.is_diff != m_self.is_diff) &&
                        m_self.ndim == 1 && s_arg.index) {
                        m_temp.is_diff = m_self.is_diff;

                        nb::handle arg_casted_t = meta_get_type(m_temp);
                        nb::object arg_casted = arg_casted_t(nb::handle(arg));
                        nb::object init_args = nb::make_tuple(arg_casted);

                        return tp_init_array(self, init_args.ptr(), kwds);
                    }

                    // Potentially do a cast
                    m_temp = s_arg;
                    m_temp.type = s.type;
                    if (m_temp == m_self && s.cast && s.cast != DRJIT_OP_NOT_IMPLEMENTED) {
                        s.cast(inst_ptr(arg), (VarType) s_arg.type, false, inst_ptr(self));
                        nb::inst_mark_ready(self);
                        return 0;
                    }

                    if (NB_UNLIKELY(s_arg.is_matrix && s.is_matrix && s.shape[0] != s_arg.shape[0])) {
                        // Convert between matrices of different sizes
                        nb::inst_zero(self);
                        for (size_t i = 0; i < (size_t) s.shape[0]; ++i) {
                            for (size_t j = 0; j < (size_t) s.shape[0]; ++j) {
                                if (i < (size_t) s_arg.shape[0] && j < (size_t) s_arg.shape[0])
                                    nb::handle(nb::handle(self)[i])[j] = nb::handle(nb::handle(arg)[i])[j];
                                else
                                    nb::handle(nb::handle(self)[i])[j] = nb::float_(double(i == j));
                            }
                        }
                        return 0;
                    }

                    // Potentially load from the CPU
                    m_temp = s;
                    m_temp.backend = (uint64_t) JitBackend::None;
                    m_temp.is_vector = true;
                    m_temp.is_diff = false;

                    if (m_temp == m_arg && s.init_data && s_arg.data) {
                        ArrayBase *arg_p = inst_ptr(arg);
                        size_t len = s_arg.len(arg_p);
                        void *data = s_arg.data(arg_p);
                        s.init_data(len, data, inst_ptr(self));
                        nb::inst_mark_ready(self);
                        return 0;
                    }

                    // Disallow inefficient element-by-element imports of dynamic arrays
                    if (s.ndim     == 1 && s.shape[0]     == DRJIT_DYNAMIC &&
                        s_arg.ndim == 1 && s_arg.shape[0] == DRJIT_DYNAMIC) {
                        try_sequence_import = false;
                    } else {
                        // Always broadcast when the element type is one of the sub-elements
                        // or its AD/non-AD counterpart
                        PyTypeObject *cur_tp = (PyTypeObject *) s.value;
                        while (cur_tp) {
                            ArrayMeta m_curr =  supp(cur_tp);
                            m_curr.is_diff = m_arg.is_diff;
                            m_curr.talign = m_arg.talign;
                            if (m_curr == m_arg) {
                                try_sequence_import = false;
                                break;
                            }
                            if (!is_drjit_type(cur_tp))
                                break;
                            cur_tp = (PyTypeObject *) supp(cur_tp).value;
                        }
                    }
                }
            }

            // Try to construct from an instance created by another
            // array programming framework or a Dr.Jit tensor
            nb::object converted_complex_scalar;
            if (is_drjit_tensor || (!arg_is_drjit && !is_builtin(arg_tp) && nb::ndarray_check(arg))) {
                // For scalar types we want to rely on broadcasting below
                if (is_drjit_tensor || meta_get(arg).ndim) {
                    // Import flattened array in C-style ordering
                    nb::object flattened;

                    if (s.is_complex)
                        do_flip_axes = true;

                    if (is_drjit_tensor) {
                        const ArraySupplement &as = supp(arg_tp);
                        const dr::vector<size_t> &shape = as.tensor_shape(inst_ptr(arg));
                        if (shape.size() != s.ndim)
                            nb::raise("dimensionality mismatch (target has %u, "
                                      "source has %zu dimensions)",
                                      s.ndim, shape.size());
                        for (uint32_t d = 0; d < s.ndim; ++d) {
                            if (s.shape[d] == DRJIT_DYNAMIC)
                                continue;
                            size_t source_shape =
                                do_flip_axes ? shape[shape.size() - 1 - d]
                                             : shape[d];
                            if (s.shape[d] != source_shape)
                            nb::raise("mismatched shape (axis %u has size %u in target type, %zu in source tensor)",
                                      d, s.shape[d], source_shape);
                        }
                        flattened = nb::steal(as.tensor_array(arg));
                    } else {
                        flattened = import_ndarray(s, arg);
                    }

                    nb::object unraveled = unravel(
                        nb::borrow<nb::type_object_t<dr::ArrayBase>>(self_tp),
                        flattened, do_flip_axes ? 'F' : 'C');

                    nb::inst_move(self, unraveled);
                    return 0;
                } else if (s.is_complex && nb::hasattr(arg, "__complex__")) { /* complex scalar */
                    converted_complex_scalar =
                        nb::handle(arg).attr("__complex__")();
                    arg = converted_complex_scalar.ptr();
                    arg_tp = Py_TYPE(arg);
                }

                try_sequence_import = false;
            }

            // Try to construct from a sequence/iterable type
            if (try_sequence_import && array_init_from_seq(self, s, arg))
                return 0;

            if (s.is_complex && arg_tp == &PyComplex_Type) {
                nb::object t = nb::make_tuple(PyComplex_RealAsDouble(arg),
                                              PyComplex_ImagAsDouble(arg));

                if (array_init_from_seq(self, s, t.ptr()))
                    return 0;
            }

            // No sequence/iterable type, try broadcasting
            Py_ssize_t size = s.shape[0];
            raise_if(size == 0,
                     "Input has the wrong size (expected 0 elements, got 1).");

            nb::object element;
            PyTypeObject *value_tp = (PyTypeObject *) s.value;

            if (s.is_matrix)
                value_tp = (PyTypeObject *) supp(value_tp).value;

            if (arg_tp == value_tp || PyType_IsSubtype(arg_tp, value_tp) ||
                    (s.is_class && arg == Py_None)) {
                element = nb::borrow(arg);
            } else {
                PyObject *args2[2] = { nullptr, arg };
                element = nb::steal(
                    NB_VECTORCALL((PyObject *) value_tp, args2 + 1,
                                  1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));
                if (NB_UNLIKELY(!element.is_valid())) {
                    nb::error_scope scope;
                    nb::raise("Broadcast from type '%s' to type '%s' failed.%s",
                              nb::type_name(arg_tp).c_str(),
                              nb::type_name(value_tp).c_str(),
                              try_sequence_import
                                  ? ""
                                  : " Refused to perform an inefficient "
                                    "element-by-element copy.");
                }
            }

            if (size == DRJIT_DYNAMIC) {
                if (s.init_const) {
                    s.init_const(1, false, element.ptr(), inst_ptr(self));
                    nb::inst_mark_ready(self);
                    return 0;
                }

                size = 1;
                s.init(1, inst_ptr(self));
                nb::inst_mark_ready(self);
            } else {
                nb::inst_zero(self);
            }

            if (s.is_complex) {
                nb::float_ zero(0.0);
                raise_if(set_item(self, 0, element.ptr()) ||
                         set_item(self, 1, zero.ptr()),
                         "Item assignment failed.");
            } else if (s.is_quaternion) {
                nb::float_ zero(0.0);
                raise_if(set_item(self, 0, zero.ptr()) ||
                         set_item(self, 1, zero.ptr()) ||
                         set_item(self, 2, zero.ptr()) ||
                         set_item(self, 3, element.ptr()),
                         "Item assignment failed.");
            } else if (s.is_matrix) {
                nb::float_ zero(0.0);

                for (Py_ssize_t i = 0; i < size; ++i) {
                    nb::object col = nb::steal(s.item(self, i));
                    for (Py_ssize_t j = 0; j < size; ++j)
                        col[j] = (i == j) ? element : zero;
                }
            } else {
                for (Py_ssize_t i = 0; i < size; ++i)
                    raise_if(set_item(self, i, element.ptr()),
                             "Item assignment failed.");
            }

            return 0;
        }
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        nb::chain_error(PyExc_TypeError, "%U.__init__(): %s", tp_name.ptr(), e.what());
        return -1;
    }
}

static bool array_init_from_seq(PyObject *self, const ArraySupplement &s, PyObject *seq) {
    ssizeargfunc sq_item = nullptr;
    lenfunc sq_length = nullptr;

    PyTypeObject *tp = Py_TYPE(seq);
#if defined(Py_LIMITED_API)
    sq_length = (lenfunc) nb::type_get_slot(tp, Py_sq_length);
    sq_item = (ssizeargfunc) nb::type_get_slot(tp, Py_sq_item);
#else
    PySequenceMethods *sm = tp->tp_as_sequence;
    if (sm) {
        sq_length = sm->sq_length;
        sq_item = sm->sq_item;
    }
#endif

    if (!sq_length || !sq_item) {
        // Special case for general iterable types. Handled recursively
        getiterfunc tp_iter;

#if defined(Py_LIMITED_API)
        tp_iter = (getiterfunc) nb::type_get_slot(tp, Py_tp_iter);
#else
        tp_iter = tp->tp_iter;
#endif

        if (tp_iter) {
            nb::object seq2 = nb::steal(PySequence_List(seq));
            raise_if(!seq2.is_valid(),
                     "Could not convert iterable into a sequence.");
            return array_init_from_seq(self, s, seq2.ptr());
        }

        return false;
    }

    Py_ssize_t size = sq_length(seq);
    raise_if(size < 0, "Unable to determine the size of the given sequence.");
    ArraySupplement::SetItem set_item = s.set_item;

    if (s.is_matrix && size == s.shape[0] * s.shape[1]) {
        nb::inst_zero(self);
        for (Py_ssize_t i = 0; i < s.shape[0]; ++i) {
            for (Py_ssize_t j = 0; j < s.shape[1]; ++j) {
                nb::object o = nb::steal(sq_item(seq, i*s.shape[1]+j));
                raise_if(!o.is_valid(),
                         "Item retrieval failed.");
                nb::handle self_h = self;
                self_h[i][j] = o;
            }
        }
        return true;
    }

    bool is_dynamic = s.shape[0] == DRJIT_DYNAMIC;
    raise_if(!is_dynamic && s.shape[0] != size,
             "Input has the wrong size (expected %u elements, got %zd).",
             (unsigned) s.shape[0], size);

    if (size == 1 && s.init_const) {
        nb::object o = nb::steal(sq_item(seq, 0));
        raise_if(!o.is_valid(), "Item retrieval failed.");
        s.init_const((size_t) size, false, o.ptr(), inst_ptr(self));
        nb::inst_mark_ready(self);
        return true;
    }

    if (s.ndim == 1 && s.init_data) {
        bool fail = false;

        #define FROM_SEQ_IMPL(T)                                           \
        {                                                                  \
            nb::detail::make_caster<T> caster;                             \
            T *p = (T *) storage.get();                                    \
            for (Py_ssize_t i = 0; i < size; ++i) {                        \
                nb::object o = nb::steal(sq_item(seq, i));                 \
                if (NB_UNLIKELY(!o.is_valid() ||                           \
                    !caster.from_python(o,                                 \
                                        (uint8_t) nb::detail::cast_flags:: \
                                            convert, nullptr))) {          \
                    fail = true;                                           \
                    break;                                                 \
                }                                                          \
                p[i] = caster.value;                                       \
            }                                                              \
        }

        if (!s.is_class) {
            size_t byte_size = jit_type_size((VarType) s.type) * (size_t) size;
            dr::unique_ptr<uint8_t[]> storage(new uint8_t[byte_size]);
            switch ((VarType) s.type) {
                case VarType::Bool:    FROM_SEQ_IMPL(bool);     break;
                case VarType::Float16: FROM_SEQ_IMPL(dr::half); break;
                case VarType::Float32: FROM_SEQ_IMPL(float);    break;
                case VarType::Float64: FROM_SEQ_IMPL(double);   break;
                case VarType::Int8:    FROM_SEQ_IMPL(int8_t);  break;
                case VarType::UInt8:   FROM_SEQ_IMPL(uint8_t); break;
                case VarType::Int32:   FROM_SEQ_IMPL(int32_t);  break;
                case VarType::UInt32:  FROM_SEQ_IMPL(uint32_t); break;
                case VarType::Int64:   FROM_SEQ_IMPL(int64_t);  break;
                case VarType::UInt64:  FROM_SEQ_IMPL(uint64_t); break;
                default: fail = true;
            }
            raise_if(fail, "Could not construct from sequence (invalid type in input).");
            s.init_data((size_t) size, storage.get(), inst_ptr(self));
        } else {
            const std::type_info &cpp_type = nb::type_info(s.value);
            dr::unique_ptr<void*[]> storage((new void*[size]));

            for (Py_ssize_t i = 0; i < size; ++i) {
                nb::object o = nb::steal(sq_item(seq, i));

                void *ptr = nullptr;
                if (!nb::detail::nb_type_get(&cpp_type, o.ptr(), 0, nullptr, &ptr)) {
                    fail = true;
                    break;
                }
                storage[i] = ptr;
            }

            raise_if(fail, "Could not construct from sequence (invalid type in input).");
            s.init_data((size_t) size, storage.get(), inst_ptr(self));
        }


        nb::inst_mark_ready(self);

        return true;
    }

    if (is_dynamic) {
        s.init((size_t) size, inst_ptr(self));
        nb::inst_mark_ready(self);
    } else {
        nb::inst_zero(self);
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
        nb::object o = nb::steal(sq_item(seq, i));
        raise_if(!o.is_valid(),
                 "Item retrieval failed.");
        raise_if(set_item(self, i, o.ptr()),
                 "Item assignment failed.");
    }

    return true;
}

// Forward declaration
static void ndarray_keep_alive(JitBackend backend, uint32_t index,
                               nb::detail::ndarray_handle *p);

nb::object import_ndarray(ArrayMeta m, PyObject *arg, vector<size_t> *shape_out,
                          bool force_ad) {
    int64_t shape[4];
    nb::detail::ndarray_config conf { };
    conf.order = 'C';
    conf.ro = true;

    if ((VarType) m.type != VarType::Void)
        conf.dtype = drjit_type_to_dlpack((VarType) m.type);

    if (m.ndim) {
        conf.ndim = m.ndim;
        conf.shape = shape;
        for (size_t i = 0; i < m.ndim; ++i) {
            shape[i] = m.shape[i];
            if (shape[i] == DRJIT_DYNAMIC)
                shape[i] = -1;
        }
    }

    if (m.is_complex) {
        for (int32_t i = 1; i < conf.ndim; ++i)
            shape[i - 1] = shape[i];
        conf.dtype.code = (uint8_t) nb::dlpack::dtype_code::Complex;
        conf.dtype.bits *= 2;
        conf.ndim -= 1;
    }

    nb::detail::ndarray_handle *th = nb::detail::ndarray_import(
        arg, &conf, (uint8_t) nb::detail::cast_flags::convert, nullptr);

    if (!th && m.ndim > 1 && m.shape[m.ndim - 1] == DRJIT_DYNAMIC) {
        // Try conversion of scalar to vectorized representation
        conf.ndim--;
        th = nb::detail::ndarray_import(
            arg, &conf, (uint8_t) nb::detail::cast_flags::convert, nullptr);
        if (!th)
            conf.ndim++;
    }

    if (!th) {
        nb::str arg_name = nb::inst_name(arg);
        nb::detail::Buffer buf(256);

        buf.fmt("Unable to initialize from an array of type '%s'. The input "
                "should have the following configuration for this to succeed: ",
                arg_name.c_str());

        if (conf.shape) {
            buf.fmt("ndim=%u, shape=(", conf.ndim);

            for (int32_t i = 0; i < conf.ndim; ++i) {
                if (shape[i] == -1)
                    buf.put('*');
                else
                    buf.put_uint32((uint32_t) shape[i]);
                if (i + 1 < conf.ndim)
                    buf.put(", ");
            }
            buf.put("), ");
        }

        if (conf.dtype != nb::dlpack::dtype()) {
            buf.put("dtype=");
            nb::dlpack::dtype_code code = (nb::dlpack::dtype_code) conf.dtype.code;
            switch (code) {
                case nb::dlpack::dtype_code::Bool: buf.put("bool"); break;
                case nb::dlpack::dtype_code::Int: buf.put("int"); break;
                case nb::dlpack::dtype_code::UInt: buf.put("uint"); break;
                case nb::dlpack::dtype_code::Float: buf.put("float"); break;
                case nb::dlpack::dtype_code::Bfloat: buf.put("bfloat"); break;
                case nb::dlpack::dtype_code::Complex: buf.put("complex"); break;
            }

            if (code != nb::dlpack::dtype_code::Bool)
                buf.put_uint32(conf.dtype.bits);
            buf.put(", order='C'.");
        }

        throw nb::type_error(buf.get());
    }

    nb::ndarray<> ndarr(th);
    size_t size = 1, ndim = ndarr.ndim();
    if (shape_out)
        shape_out->resize(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        size_t cur_size = ndarr.shape(i);
        size *= cur_size;
        if (shape_out)
            shape_out->operator[](i) = cur_size;
    }

    if (m == ArrayMeta{}) {
        switch (ndarr.device_type()) {
            case nb::device::cuda::value:
                m.backend = (uint8_t) JitBackend::CUDA;
                break;

            case nb::device::cpu::value:
                m.backend = (uint8_t) JitBackend::LLVM;
                break;

            default:
                nb::raise("import_ndarray(): unsupported device type!");
        }

        m.type = (uint16_t) dlpack_type_to_drjit(ndarr.dtype());
    }

    if (m.is_complex) {
        if (shape_out) {
            shape_out->resize(shape_out->size() + 1);
            for (size_t i = 1; i < ndim; ++i)
                shape_out->operator[](i) = shape_out->operator[](i - 1);
            shape_out->operator[](0) = 0;
        }
        ndim += 1;
        size *= 2;
    }

    ArrayMeta temp_meta { };
    temp_meta.backend = m.backend;
    temp_meta.is_diff = m.is_diff | force_ad;
    temp_meta.type = m.type;
    temp_meta.ndim = 1;
    temp_meta.shape[0] = DRJIT_DYNAMIC;
    nb::handle temp_t = meta_get_type(temp_meta);
    nb::object temp = nb::inst_alloc(temp_t);
    JitBackend backend = (JitBackend) m.backend;
    VarType vt = (VarType) m.type;

    if (backend != JitBackend::None) {
        int32_t device_type = backend == JitBackend::CUDA
                                  ? nb::device::cuda::value
                                  : nb::device::cpu::value;

        uint32_t index;

        if (device_type == ndarr.device_type()) {
            index = jit_var_mem_map(backend, vt, ndarr.data(), size, 0);
            // Hold a reference to the ndarray while Dr.Jit is using it
            ndarray_keep_alive(backend, index, th);
        } else {
            AllocType at;
            switch (ndarr.device_type()) {
                case nb::device::cuda::value: at = AllocType::Device; break;
                case nb::device::cpu::value:  at = AllocType::Host; break;
                default: nb::raise("Unsupported source device!");
            }

            index = jit_var_mem_copy(backend, at, vt, ndarr.data(), size);
        }

        supp(temp_t).init_index(index, inst_ptr(temp));
        jit_var_dec_ref(index);
    } else {
        if (ndarr.device_type() != nb::device::cpu::value)
            nb::raise("Unsupported source device!");

        supp(temp_t).init_data(size, ndarr.data(), inst_ptr(temp));
    }

    nb::inst_mark_ready(temp);
    return temp;
}

// The ndarray release sequence implemented by the following callbacks is
// paranoid but needed in some case. When Dr.Jit wants to release an array, it
// might still be accessed by concurrently running code (this is particularly
// relevant for LLVM mode, where both host and "device" share the same address
// space). Decreasing the reference count is therefore done by enqueueuing a
// host function that is called after Dr.Jit is guaranteed to have finished
// accessing the array. But this presents another problem: decreasing the
// DLPack reference count might involve CPython API calls that require holding
// the GIL, which is not a nice requirement for things running in the
// CUDA-internal message queue thread or nanothread worker due to a danger of
// deadlocks. We therefore manage the array cleanup calls in a separate thread.
// This thread will *eventually* decrease the reference count of the array.
// This is similar to Py_AddPendingCall, but avoids the issue where
// Py_AddPendingCall is not always serviced, see also
// https://github.com/python/cpython/issues/95820.

extern int disable_gc_scope;

using CleanupCallback = int(*)(void*);
static std::mutex python_cleanup_queue_mutex;
static std::condition_variable python_cleanup_queue_cond;
static bool python_cleanup_thread_stop = false;
static std::vector<std::pair<CleanupCallback, void*>> python_cleanup_queue;
static std::thread python_cleanup_thread;

void python_cleanup_thread_main() {
    while (true) {
        std::unique_lock lock(python_cleanup_queue_mutex);
        python_cleanup_queue_cond.wait(lock, [] {
            return !python_cleanup_queue.empty() || python_cleanup_thread_stop;
        });

        std::vector<std::pair<CleanupCallback, void*>> calls_to_execute;
        calls_to_execute.swap(python_cleanup_queue);
        lock.unlock();

        nb::gil_scoped_acquire guard;
        for (const auto& item : calls_to_execute)
            item.first(item.second);
        if (python_cleanup_thread_stop)
            break;
    }
}

void python_cleanup_thread_static_initialization() {
    python_cleanup_thread = std::thread(&python_cleanup_thread_main);
}

void python_cleanup_thread_static_shutdown() {
    {
        std::scoped_lock lock(python_cleanup_queue_mutex);
        python_cleanup_thread_stop = true;
    }
    nb::gil_scoped_release guard;
    python_cleanup_queue_cond.notify_one();
    python_cleanup_thread.join();
}

void enqueue_python_cleanup_call(CleanupCallback callback, void* data_ptr) {
    std::scoped_lock lock(python_cleanup_queue_mutex);
    python_cleanup_queue.emplace_back(callback, data_ptr);
    python_cleanup_queue_cond.notify_one();
}

static int ndarray_free_cb_3(void *p) {
    if (disable_gc_scope) {
        // Don't service pending calls while in the logger critical section.
        // That's because we can have arbitrary Dr.Jit/Dr.Jit-Extra functions
        // on the call stack. Re-entering Python to then free nd-arrays,
        // which can call code in Dr.Jit/Dr.Jit-extra, is a recipe for deadlocks.
        enqueue_python_cleanup_call(ndarray_free_cb_3, p);
    } else {
        nb::detail::ndarray_dec_ref((nb::detail::ndarray_handle *) p);
    }
    return 0;
}

int drjit_py_is_alive = 1;

static void ndarray_free_cb_2(void *p) {
    if (nb::is_alive() && drjit_py_is_alive)
        enqueue_python_cleanup_call(ndarray_free_cb_3, p);
}

static void ndarray_free_cb(uint32_t, int free, void *p) {
    if (!free)
        return;

    // Decode packed pointer + backend ID created in ndarray_keep_alive
    uintptr_t msg = (uintptr_t) p, mask = 3;
    JitBackend backend = (JitBackend) (msg & mask);
    void *p2 = (void *) (msg & ~mask);

    // Don't run the next step if Dr.Jit has already shut down
    if (nb::is_alive() && jit_has_backend(backend) && drjit_py_is_alive)
        jit_enqueue_host_func(backend, ndarray_free_cb_2, p2);
}

static void ndarray_keep_alive(JitBackend backend, uint32_t index, nb::detail::ndarray_handle *p) {
    if (!index)
        return;

    if ((int) backend > 3)
        jit_raise("ndarray_keep_alive(): internal error, backend index too large.");

    nb::detail::ndarray_inc_ref(p);

    // Pack pointer + backend ID and send to ndarray_free_cb (asynchronously)
    uintptr_t msg = (uintptr_t) p;
    msg |= (uintptr_t) backend;
    jit_var_set_callback(index, ndarray_free_cb, (void *) msg);
}

nb::object full_alt(nb::type_object dtype, nb::handle value, size_t size);
nb::object empty_alt(nb::type_object dtype, size_t size);

nb::object view_to_tensor(nb::handle h, dr::vector<size_t> &shape) {
    MatrixView &view = nb::cast<MatrixView &>(nb::handle(h));
    if (view.transpose)
        nb::raise("The view is transposed. Conversion into tensor format still "
                  "needs to be implemented.");

    if (view.descr.layout != MatrixLayout::RowMajor)
        nb::raise("This tensor is in an inference/training-optimal layout. To "
                  "convert it back into tensor form, you must unpack it into a "
                  "row-major representation via drjit.nn.unpack().");

    if (view.descr.stride != view.descr.cols)
        nb::raise("Unsupported row stride: expected stride %u, found %u.",
                  view.descr.cols, view.descr.stride);

    shape.push_back(view.descr.rows);
    shape.push_back(view.descr.cols);

    return view.buffer[nb::slice(view.descr.offset,
                                 view.descr.offset + view.descr.size, 1u)];
}

int tp_init_tensor(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
    PyTypeObject *self_tp = Py_TYPE(self);

    try {
        PyObject *array = nullptr, *shape = nullptr, *flip_axes = nullptr;
        const char *kwlist[4] = { "array", "shape", "flip_axes", nullptr };
        raise_if(!PyArg_ParseTupleAndKeywords(
                     args, kwds, "|OO!O!", (char **) kwlist, &array,
                     &PyTuple_Type, &shape, &PyBool_Type, &flip_axes),
                 "Invalid tensor constructor arguments.");

        const ArraySupplement &s = supp(self_tp);
        bool do_flip_axes = flip_axes == Py_True;

        PyTypeObject *array_tp = array ? Py_TYPE(array) : nullptr;
        raise_if(do_flip_axes && (shape || !array_tp ||
                                  (!is_drjit_type(array_tp) &&
                                   !nb::handle(array_tp).is(coop_vector_type)) ||
                                  array_tp == self_tp),
                 "flip_axes=True requires that 'shape' is not specified, and "
                 "that the input is a nested Dr.Jit array type (e.g. "
                 "drjit.cuda.Array3f).");

        if (!shape && !array) {
            nb::detail::nb_inst_zero(self);
            s.tensor_shape(inst_ptr(self)).push_back(0);
            return 0;
        }

        raise_if(!array, "Input array must be specified.");

        // Same type -> copy constructor
        if (array_tp == self_tp) {
            if (shape)
                nb::raise(
                    "use 'Tensor(x.array, shape)' or 'drjit.reshape(Tensor, x, "
                    "shape)' to reshape a tensor");
            if (do_flip_axes)
                nb::raise("The flip_axes argument is only supported when "
                          "constructing tensors from N-D arrays or cooperative "
                          "vectors");
            nb::detail::nb_inst_copy(self, array);
            return 0;
        }

        nb::detail::nb_inst_zero(self);
        vector<size_t> &shape_vec = s.tensor_shape(inst_ptr(self));

        nb::object args_2;
        if (!shape) {
            nb::object flat;
            if (!is_drjit_type(array_tp) && !is_builtin(array_tp) && nb::ndarray_check(array)) {
                // Try to construct from an instance created by another
                // array programming framework
                flat = import_ndarray(s, array, &shape_vec);
            } else if (nb::isinstance<MatrixView>(nb::handle(array))) {
                flat = view_to_tensor(array, shape_vec);
            } else {
                // Infer the shape of an arbitrary data structure & flatten it
                VarType vt = (VarType) s.type;
                char order = do_flip_axes ? 'F' : 'C';
                flat = ravel(array, order, &shape_vec, nullptr, &vt);
                if (do_flip_axes)
                    std::reverse(shape_vec.begin(), shape_vec.end());
            }
            args_2 = nb::make_tuple(flat);
        } else {
            // Shape is given, require flat input
            args_2 = nb::make_tuple(nb::handle(array));
            shape_vec.resize((size_t) NB_TUPLE_GET_SIZE(shape));

            for (size_t i = 0; i < shape_vec.size(); ++i) {
                PyObject *o = NB_TUPLE_GET_ITEM(shape, (Py_ssize_t) i);
                size_t rv = PyLong_AsSize_t(o);
                raise_if(rv == (size_t) -1, "Invalid shape tuple.");
                shape_vec[i] = rv;
            }
        }

        nb::object self_array = nb::steal(s.tensor_array(self));
        int rv = tp_init_array(self_array.ptr(), args_2.ptr(), nullptr);
        auto [ready, destruct] = nb::inst_state(self_array);
        (void) destruct;
        nb::inst_set_state(self_array, ready, false);
        raise_if(rv, "Tensor storage initialization failed.");

        // Double-check that the size makes sense
        size_t size_exp = 1, size = nb::len(self_array);
        for (size_t i = 0; i < shape_vec.size(); ++i)
            size_exp *= shape_vec[i];

        raise_if(size != size_exp,
                 "Input array has the wrong number of entries (got %zu, "
                 "expected %zu).", size, size_exp);

        return 0;
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(self_tp);
        e.restore();
        nb::chain_error(PyExc_TypeError, "%U.__init__(): internal error.", tp_name.ptr());
        return -1;
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        nb::chain_error(PyExc_TypeError, "%U.__init__(): %s", tp_name.ptr(), e.what());
        return -1;
    }
}

// Forward declaration
nb::object full(const char *name, nb::handle dtype, nb::handle value,
                const dr::vector<size_t> &shape, bool opaque) {
    return full(name, dtype, value, shape.size(), shape.data(), opaque);
}

nb::object full(const char *name, nb::handle dtype, nb::handle value,
                size_t size, bool opaque) {
    dr::vector<size_t> shape;

    if (is_drjit_type(dtype)) {
        const ArraySupplement &s = supp(dtype);

        if (s.is_tensor) {
            shape.resize(1);
            shape[0] = size;
        } else {
            shape.resize(s.ndim);

            for (size_t i = 0; i < s.ndim; ++i) {
                size_t k = s.shape[i];
                if (k == DRJIT_DYNAMIC)
                    k = (i == (size_t) s.ndim - 1) ? size : 1;
                shape[i] = k;
            }
        }
    } else {
        shape.resize(1);
        shape[0] = size;
    }

    return full(name, dtype, value, shape, opaque);
}

nb::object full(const char *name, nb::handle dtype, nb::handle value,
                size_t ndim, const size_t *shape, bool opaque) {
    try {
        if (is_drjit_type(dtype)) {
            const ArraySupplement &s = supp(dtype);

            if (s.is_tensor) {
                size_t size = 1;
                for (size_t i = 0; i < ndim; ++i)
                    size *= shape[i];

                nb::tuple shape_tuple =
                    cast_shape(vector<size_t>(shape, shape + ndim));

                return dtype(full(name, s.array, value, 1, &size, opaque), shape_tuple);
            }

            bool fail = s.ndim != ndim;
            if (!fail) {
                for (size_t i = 0; i < ndim; ++i)
                    fail |= s.shape[i] != DRJIT_DYNAMIC && s.shape[i] != shape[i];
            }

            raise_if(fail, "the provided \"shape\" and \"dtype\" parameters are incompatible.");

            nb::object result = nb::inst_alloc(dtype);

            if (s.init_const && value.is_valid()) {
                if ((VarType) s.type == VarType::Bool && value.type().is(&PyLong_Type))
                    value = nb::cast<int>(value) ? Py_True : Py_False;

                s.init_const(shape[0], opaque, value.ptr(), inst_ptr(result));
                nb::inst_mark_ready(result);
                return result;
            }

            if (s.shape[0] == DRJIT_DYNAMIC) {
                s.init(shape[0], inst_ptr(result));
                nb::inst_mark_ready(result);
            } else {
                nb::inst_zero(result);
            }

            if (!value.is_valid() && ndim == 1) {
                return result;
            } else {
                ArraySupplement::SetItem set_item = s.set_item;
                nb::object o;
                for (size_t i = 0; i < shape[0]; ++i) {
                    nb::object v = nb::borrow(value);
                    if ((s.is_complex && i == 1) || (s.is_quaternion && i != 3))
                        v = nb::int_(0);
                    if (i == 0 || !value.is_valid() || opaque || s.is_complex || s.is_quaternion)
                        o = full(name, s.value, v, ndim - 1, shape + 1, opaque);
                    set_item(result.ptr(), i, o.ptr());
                }
            }

            return result;
        } else if (dtype.is(&PyLong_Type) || dtype.is(&PyFloat_Type) || dtype.is(&PyBool_Type)) {
            if (value.is_valid())
                return dtype(value);
            else
                return dtype(0);
        } else {
            if (nb::dict ds = get_drjit_struct(dtype); ds.is_valid()) {
                nb::object result = dtype();

                if (value.is(nb::int_(0))) {
                    nb::object custom_zero = nb::getattr(result, "zero_", nb::handle());
                    if (custom_zero.is_valid()) {
                        custom_zero(shape[0]);
                        return result;
                    }
                }

                for (auto [k, v] : ds) {
                    raise_if(!v.is_type(), "DRJIT_STRUCT type annotation invalid");

                    nb::object entry;
                    if (is_drjit_type(v) && ndim == 1)
                        entry = full(name, v, value, shape[0], opaque);
                    else
                        entry = full(name, v, value, ndim, shape, opaque);

                    nb::setattr(result, k, entry);
                }

                return result;
            } else if (nb::object df = get_dataclass_fields(dtype); df.is_valid()) {
                nb::object result = nb::dict();

                for (auto field : df) {
                    nb::object k = field.attr(DR_STR(name)),
                               v = field.attr(DR_STR(type));

                    if (!v.is_type())
                        v = extract_type(v);

                    raise_if(!v.is_type(), "dataclass type annotations invalid");

                    nb::object entry;
                    if (is_drjit_type(v) && ndim == 1)
                        entry = full(name, v, value, shape[0], opaque);
                    else
                        entry = full(name, v, value, ndim, shape, opaque);

                    result[k] = entry;
                }
                return dtype(**result);
            }

            if (!value.is_valid() || value.is(nb::int_(0)))
                return dtype();

            nb::raise("unsupported dtype");
        }
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(dtype);
        nb::raise_from(
            e, PyExc_RuntimeError,
            "drjit.%s(<%U>): could not construct output (see the error above).",
            name, tp_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(dtype);
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.%s(<%U>): could not construct output: %s.", name,
                        tp_name.ptr(), e.what());
        nb::raise_python_error();
    }
}

nb::object arange(const nb::type_object_t<ArrayBase> &dtype,
                  Py_ssize_t start, Py_ssize_t end, Py_ssize_t step) {
    const ArraySupplement &s = supp(dtype);

    if (s.is_tensor) {
        return dtype(arange(
            nb::borrow<nb::type_object_t<ArrayBase>>(s.array),
            start, end, step
        ));
    }

    if (s.ndim != 1 || s.shape[0] != DRJIT_DYNAMIC)
        throw nb::type_error("drjit.arange(): unsupported dtype -- must "
                             "be a dynamically sized 1D array.");

    VarType vt = (VarType) s.type;
    if (vt == VarType::Bool || vt == VarType::Pointer)
        throw nb::type_error("drjit.arange(): unsupported dtype -- must "
                             "be an arithmetic type.");

    Py_ssize_t size = (end - start + step - (step > 0 ? 1 : -1)) / step;
    ArrayMeta meta = s;
    meta.type = (uint16_t) VarType::UInt32;

    nb::handle counter_tp = meta_get_type(meta);
    const ArraySupplement &counter_s = supp(counter_tp);

    if (!counter_s.init_counter)
        throw nb::type_error("drjit.arange(): unsupported dtype.");

    if (size == 0)
        return dtype();
    else if (size < 0)
        nb::raise("drjit.arange(): size cannot be negative.");

    nb::object result = nb::inst_alloc(counter_tp);
    counter_s.init_counter((size_t) size, inst_ptr(result));
    nb::inst_mark_ready(result);

    if (start == 0 && step == 1)
        return dtype(result);
    else
        return fma(dtype(result), dtype(step), dtype(start));
}

nb::object linspace(const nb::type_object_t<ArrayBase> &dtype,
                    double start, double stop, size_t size, bool endpoint) {
    const ArraySupplement &s = supp(dtype);

    if (s.is_tensor) {
        return dtype(linspace(
            nb::borrow<nb::type_object_t<ArrayBase>>(s.array),
            start, stop, size, endpoint
        ));
    }

    if (s.ndim != 1 || s.shape[0] != DRJIT_DYNAMIC)
        throw nb::type_error("drjit.linspace(): unsupported dtype -- must "
                             "be a dynamically sized 1D array.");

    VarType vt = (VarType) s.type;
    if (vt != VarType::Float16 && vt != VarType::Float32 && vt != VarType::Float64)
        throw nb::type_error("drjit.linspace(): unsupported dtype -- must "
                             "be an floating point type.");

    ArrayMeta meta = s;
    meta.type = (uint16_t) VarType::UInt32;

    nb::handle counter_tp = meta_get_type(meta);
    const ArraySupplement &counter_s = supp(counter_tp);

    if (!counter_s.init_counter)
        throw nb::type_error("drjit.linspace(): unsupported dtype.");

    if (size == 0)
        return dtype();

    nb::object counter = nb::inst_alloc(counter_tp);
    counter_s.init_counter((size_t) size, inst_ptr(counter));
    nb::inst_mark_ready(counter);

    nb::handle dtype_c = dtype;
    if ((VarType) s.type == VarType::Float16) {
        ArrayMeta m = s;
        m.type = (uint16_t) VarType::Float32;
        dtype_c = meta_get_type(m);
    }

    double step = (stop - start) / (size - ((endpoint && size > 0) ? 1 : 0));
    nb::object result = fma(dtype_c(counter), dtype_c(step), dtype_c(start));

    if (!dtype_c.is(dtype))
        result = dtype(result);

    return result;
}

/// Extract types from typing.Optional[T], typing.Union[T, None], etc.
nb::object extract_type(nb::object tp) {
    try {
        nb::object args = nb::module_::import_("typing").attr("get_args")(tp);
        size_t len = nb::len(args);
        nb::handle nt = nb::none().type();
        if (len == 1)
            tp = args[0];
        else if (len == 2 && args[1].is(nt))
            tp = args[0];
        else if (len == 2 && args[0].is(nt))
            tp = args[1];
    } catch (...) { }
    return tp;
}

void export_init(nb::module_ &m) {
    m.def("empty",
          [](nb::type_object dtype, size_t size) {
              return full("empty", dtype, nb::handle(), size);
          }, "dtype"_a, "shape"_a = 1, doc_empty)
     .def("empty",
          [](nb::type_object dtype, dr::vector<size_t> shape) {
              return full("empty", dtype, nb::handle(), shape);
          }, "dtype"_a, "shape"_a)
     .def("zeros",
          [](nb::type_object dtype, size_t size) {
              return full("zeros", dtype, nb::int_(0), size);
          }, "dtype"_a, "shape"_a = 1, doc_zeros)
     .def("zeros",
          [](nb::type_object dtype, dr::vector<size_t> shape) {
              return full("zeros", dtype, nb::int_(0), shape);
          }, "dtype"_a, "shape"_a)
     .def("ones",
          [](nb::type_object dtype, size_t size) {
              return full("ones", dtype, nb::int_(1), size);
          }, "dtype"_a, "shape"_a = 1, doc_ones)
     .def("ones",
          [](nb::type_object dtype, dr::vector<size_t> shape) {
              return full("ones", dtype, nb::int_(1), shape);
          }, "dtype"_a, "shape"_a)
     .def("full",
          [](nb::type_object dtype, nb::handle value, size_t size) {
              return full("full", dtype, value, size);
          }, "dtype"_a, "value"_a, "shape"_a = 1, doc_full)
     .def("full",
          [](nb::type_object dtype, nb::handle value, dr::vector<size_t> shape) {
              return full("full", dtype, value, shape);
          }, "dtype"_a, "value"_a, "shape"_a)
     .def("opaque",
          [](nb::type_object dtype, nb::handle value, size_t size) {
              return full("opaque", dtype, value, size, true);
          }, "dtype"_a, "value"_a, "shape"_a = 1, doc_opaque)
     .def("opaque",
          [](nb::type_object dtype, nb::handle value, dr::vector<size_t> shape) {
              return full("opaque", dtype, value, shape, true);
          }, "dtype"_a, "value"_a, "shape"_a)
     .def("arange",
          [](const nb::type_object_t<ArrayBase> &dtype, Py_ssize_t size) {
              return arange(dtype, 0, size, 1);
          }, "dtype"_a, "size"_a, doc_arange,
          nb::sig("def arange(dtype: type[T], size: int) -> T"))
     .def("arange",
          [](const nb::type_object_t<ArrayBase> &dtype,
             Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step) {
              return arange(dtype, start, stop, step);
        }, "dtype"_a, "start"_a, "stop"_a, "step"_a = 1,
        nb::sig("def arange(dtype: type[T], start: int, stop: int, step: int = 1) -> T"))
     .def("linspace",
          [](const nb::type_object_t<ArrayBase> &dtype, double start,
             double stop, size_t num, bool endpoint) {
              return linspace(dtype, start, stop, num, endpoint);
          }, "dtype"_a, "start"_a, "stop"_a, "num"_a,
             "endpoint"_a = true, doc_linspace,
        nb::sig("def linspace(dtype: type[T], start: float, stop: float, num: int, endpoint: bool = True) -> T"));
}
