/*
    print.cpp -- implementation of drjit.format(), drjit.print(),
    and ArrayBase.__repr__().

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "print.h"
#include "base.h"
#include "init.h"
#include "memop.h"
#include "meta.h"
#include "shape.h"
#include "eval.h"
#include "apply.h"
#include "reduce.h"
#include <nanobind/stl/string.h>
#include <drjit/autodiff.h>
#include <memory>
#include <algorithm>

#include "../ext/nanobind/src/buffer.h"

using Buffer = nanobind::detail::Buffer;

/// Convert a Dr.Jit array into a human-readable representation. Used by
/// drjit.print(), drjit.format(), and drjit.ArrayBase.__repr__()
static void repr_array(Buffer &buffer, nb::handle h, size_t indent,
                       size_t threshold, const dr_vector<size_t> &shape,
                       size_t depth, nb::list &index) {
    const ArraySupplement &s = supp(h.type());

    size_t ndim = shape.size();

    // On vectorized types, iterate over the last dimension first
    size_t i = depth;
    if ((JitBackend) s.backend != JitBackend::None && !s.is_tensor) {
        if (depth == 0)
            i = ndim - 1;
        else
            i -= 1;
    }

    bool last_dim = depth == ndim - 1;

    // Reverse the dimensions of non-tensor shapes for convenience
    size_t size = shape.empty() ? 0 : shape[i];

    if ((s.is_complex || s.is_quaternion) && last_dim) {
        // Special handling for complex numbers and quaternions
        bool prev = false;

        for (size_t j = 0; j < size; ++j) {
            index[i] = nb::cast(j);
            nb::object o = h[nb::tuple(index)];

            double d = nb::cast<double>(o);
            if (d == 0)
                continue;

            if (prev || d < 0)
                buffer.put(d < -1 ? '-' : '+');
            buffer.fmt("%g", fabs(d));
            prev = true;

            if (s.is_complex && j == 1)
                buffer.put('j');
            else if (s.is_quaternion && j < 3)
                buffer.put("ijk"[j]);
        }
        if (!prev)
            buffer.put("0");
    } else if (s.is_tensor && ndim == 0) {
        // Special handling for 0D tensors
        nb::object o = nb::steal(s.tensor_array(h.ptr()))[0];

        if (PyFloat_CheckExact(o.ptr()))
            buffer.fmt("%g", nb::cast<double>(o));
        else
            buffer.put_dstr(nb::str(o).c_str());
    } else {
        buffer.put('[');
        for (size_t j = 0; j < size; ++j) {
            index[i] = nb::cast(j);
            size_t edge_items = 3;

            if (size > threshold && j == edge_items) {
                size_t j2 = size - edge_items - 1;
                buffer.fmt(".. %zu skipped ..", (size_t) (j2 - j + 1));
                j = j2;
            } else if (!last_dim) {
                repr_array(buffer, h, indent + 1, threshold, shape, depth + 1, index);
            } else {
                nb::object o = h[nb::tuple(index)];

                if (s.is_tensor)
                    o = nb::steal(s.tensor_array(o.ptr()))[0];

                if (PyFloat_CheckExact(o.ptr())) {
                    double d = nb::cast<double>(o);
                    buffer.fmt("%g", d);
                } else {
                    buffer.put_dstr(nb::str(o).c_str());
                }
            }

            if (j + 1 < size) {
                if (last_dim) {
                    buffer.put(", ");
                } else {
                    buffer.put(",\n");
                    buffer.put(' ', indent + 1);
                }
            }
        }
        buffer.put(']');
    }
}

PyObject *tp_repr(PyObject *self) noexcept {
    try {
        dr_vector<size_t> shape;
        Buffer buffer(128);
        if (!shape_impl(self, shape)) {
            buffer.put("[ragged array]");
        } else {
            nb::object zero = nb::int_(0);
            nb::list index;
            for (size_t i = 0; i < shape.size(); ++i)
                index.append(zero);
            if (!index.is_valid())
                nb::raise_python_error();
            schedule(self);
            repr_array(buffer, self, 0, 20, shape, 0, index);
        }
        return PyUnicode_FromString(buffer.get());
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::inst_name(self);
        e.restore();
        nb::chain_error(PyExc_RuntimeError, "%U.__repr__(): internal error.",
                        tp_name.ptr());
        return nullptr;
    } catch (const std::exception &e) {
        nb::str tp_name = nb::inst_name(self);
        nb::chain_error(PyExc_RuntimeError, "%U.__repr__(): %s", tp_name.ptr(),
                        e.what());
        return nullptr;
    }
}

/// Convert a PyTree into a human-readable representation
void repr_general(Buffer &buffer, nb::handle h, size_t indent_, size_t threshold) {
    nb::handle tp = h.type();
    size_t indent = indent_ + 2;

    if (is_drjit_type(tp)) {
        dr_vector<size_t> shape;
        if (!shape_impl(h, shape)) {
            buffer.put("[ragged array]");
        } else {
            nb::object zero = nb::int_(0);
            nb::list index;
            for (size_t i = 0; i < shape.size(); ++i)
                index.append(zero);
            if (!index.is_valid())
                nb::raise_python_error();
            repr_array(buffer, h, indent_, threshold, shape, 0, index);
        }
    } else if (tp.is(&PyUnicode_Type)) {
        if (indent > 2)
            buffer.put("'");
        buffer.put_dstr(nb::borrow<nb::str>(h).c_str());
        if (indent > 2)
            buffer.put("'");
    } else if (tp.is(&PyFloat_Type)) {
        buffer.fmt("%g", nb::cast<double>(h));
    } else if (tp.is(&PyTuple_Type)) {
        nb::tuple t = nb::borrow<nb::tuple>(h);
        size_t size = nb::len(t);

        if (size == 0) {
            buffer.put("()");
            return;
        }

        buffer.put("(\n");
        buffer.put(' ', indent);
        for (size_t i = 0; i < size; ++i) {
            repr_general(buffer, t[i], indent, threshold);
            if (i + 1 < size)
                buffer.put(',');
            buffer.put('\n');
            buffer.put(' ', indent);
        }
        buffer.rewind(2);
        buffer.put(')');
    } else if (tp.is(&PyList_Type)) {
        nb::list l = nb::borrow<nb::list>(h);
        size_t size = nb::len(l);
        if (size == 0) {
            buffer.put("[]");
            return;
        }

        buffer.put("[\n");
        buffer.put(' ', indent);
        for (size_t i = 0; i < size; ++i) {
            repr_general(buffer, l[i], indent, threshold);
            if (i + 1 < size)
                buffer.put(',');
            buffer.put('\n');
            buffer.put(' ', indent);
        }
        buffer.rewind(2);
        buffer.put(']');
    } else if (tp.is(&PyDict_Type)) {
        size_t i = 0, size = nb::len(h);
        if (size == 0) {
            buffer.put("{}");
            return;
        }

        buffer.put("{\n");
        buffer.put(' ', indent);

        for (auto [k, v] : nb::borrow<nb::dict>(h)) {
            repr_general(buffer, k, indent, threshold);
            buffer.put(": ");
            repr_general(buffer, v, indent, threshold);
            if (++i < size)
                buffer.put(",");
            buffer.put('\n');
            buffer.put(' ', indent);
        }
        buffer.rewind(2);
        buffer.put('}');
    } else {
        nb::object dstruct = nb::getattr(tp, "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
            buffer.put_dstr(nb::str((nb::object) tp.attr("__name__")).c_str());
            buffer.put("[\n");
            buffer.put(' ', indent);
            size_t i = 0, size = nb::len(dstruct);
            for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                buffer.put_dstr(nb::str(k).c_str());
                buffer.put('=');
                repr_general(buffer, nb::getattr(h, k), indent, threshold);
                if (++i < size)
                    buffer.put(",");
                buffer.put('\n');
                buffer.put(' ', indent);
            }
            buffer.rewind(2);
            buffer.put(']');
        } else {
            buffer.put_dstr(nb::repr(h).c_str());
        }
    }
}

// Forward declaration
static nb::object format_impl(const char *name, const std::string &fmt,
                              nb::handle file, nb::args args,
                              nb::kwargs kwargs, bool internal = false);

struct DelayedPrint {
    std::string fmt;
    nb::object file;
    nb::object counter;
    size_t limit;
    nb::args args;
    nb::kwargs kwargs;
    bool success = false;

    static void callback(uint32_t, int free, void *p) {
        nb::gil_scoped_acquire guard;
        try {
            DelayedPrint *d = (DelayedPrint *) p;
            if (free) {
                delete d;
                return;
            }
            nb::object tid_o = d->kwargs["thread_id"];
            nb::handle tp = tid_o.type();
            const ArraySupplement &s = supp(tp);

            size_t ctr_size = nb::cast<size_t>(d->counter[0]),
                   max_size = nb::len(tid_o),
                   size     = std::min(max_size, ctr_size);

            void *ptr = nullptr;
            uint32_t tid_index =
                jit_var_data((uint32_t) s.index(inst_ptr(tid_o)), &ptr);

            std::unique_ptr<uint32_t[]> ids(new uint32_t[size]);
            jit_memcpy((JitBackend) s.backend, ids.get(), ptr,
                       size * sizeof(uint32_t));
            jit_var_dec_ref(tid_index);

            std::unique_ptr<uint32_t[]> perm(new uint32_t[size]);
            for (size_t i = 0; i < size; ++i)
                perm[i] = (uint32_t) i;

            std::stable_sort(
                perm.get(), perm.get() + size,
                [&ids](uint32_t i0, uint32_t i1) { return ids[i0] < ids[i1]; });

            uint32_t perm_index =
                jit_var_mem_copy((JitBackend) s.backend, AllocType::Host,
                                 VarType::UInt32, perm.get(), size);

            nb::object perm_o = nb::inst_alloc(tp);
            s.init_index(perm_index, inst_ptr(perm_o));
            jit_var_dec_ref(perm_index);
            nb::inst_mark_ready(perm_o);

            struct PermuteData : TransformCallback {
                nb::object perm_o;
                PermuteData(nb::object perm_o) : perm_o(perm_o) { }

                void operator()(nb::handle h1, nb::handle h2) override {
                    nb::object o =
                        gather(nb::borrow<nb::type_object>(h1.type()),
                               nb::borrow(h1), perm_o, nb::cast(true), false);
                    nb::inst_replace_move(h2, o);
                }
            };

            PermuteData rd(perm_o);
            d->args = nb::borrow<nb::args>(transform("drjit.print", rd, d->args));
            d->kwargs = nb::borrow<nb::kwargs>(transform("drjit.print", rd, d->kwargs));
            d->kwargs["limit"] = nb::cast(d->limit);

            format_impl("drjit.print", d->fmt, d->file, d->args, d->kwargs, true);

            if (ctr_size > max_size) {
                PyErr_WarnFormat(
                    PyExc_RuntimeWarning, 1,
                    "dr.print(): symbolic print statement only captured "
                    "%zu of %zu available outputs. The above is a "
                    "nondeterministic sample, in which entries are in the "
                    "right order but not necessarily contiguous. Specify "
                    "`limit=..` to capture more information and/or add the "
                    "special format field `{thread_id}` show the thread "
                    "ID/array index associated with each entry of the captured "
                    "output.", max_size, ctr_size);
            }
        } catch (nb::python_error &e) {
            e.restore();
            nb::chain_error(PyExc_RuntimeError,
                           "drjit.print(): encountered an exception (see above).");
        } catch (const std::exception &e) {
            nb::chain_error(PyExc_RuntimeError, "drjit.print(): %s", e.what());
        }
    }
};

// Central formatting routine used by drjit.print() and drjit.format()
static nb::object format_impl(const char *name, const std::string &fmt,
                              nb::handle file, nb::args args,
                              nb::kwargs kwargs, bool internal) {
    try {
        // Check if the input contains symbolic variables, and whether they are compatible in that case
        struct Examine : TraverseCallback {
            JitBackend backend = JitBackend::None;
            size_t size = 1;
            std::string error;

            void operator()(nb::handle h) override {
                const ArraySupplement &s = supp(h.type());
                if (!s.index)
                    return;

                uint32_t index = (uint32_t) s.index(inst_ptr(h));
                if (!index)
                    return;

                // The following errors specifically apply to symbolic mode, so don't raise them just yet
                VarInfo info = jit_set_backend(index);
                if (info.size != size && size != 1 && info.size != 1)
                    error = "arguments have incompatible sizes (" +
                            std::to_string(size) + " vs " +
                            std::to_string(info.size) + ")";
                else
                    size = std::max(size, info.size);

                if (backend == JitBackend::None)
                    backend = info.backend;
                else if (backend != info.backend)
                    error = "arguments have incompatible backends";
            }
        };

        Examine examine;
        traverse(name, examine, args);
        traverse(name, examine, kwargs);

        if (kwargs.contains("thread_id") && !internal)
            nb::raise("the 'thread_id' keyword argument is reserved and should "
                      "not be specified.");

        bool symbolic = jit_flag(JitFlag::SymbolicScope);
        if (kwargs.contains("mode")) {
            const char *mode = nb::cast<const char *>(kwargs["mode"]);
            if (strcmp(mode, "auto") == 0) {
                /* Nothing */
            } if (strcmp(mode, "evaluate") == 0) {
                symbolic = false;
            } else if (strcmp(mode, "symbolic") == 0) {
                symbolic = true;
            } else {
                nb::raise("'mode' parameter must be one of \"auto\", "
                          "\"evaluate\", or \"symbolic\".");
            }
            nb::del(kwargs["mode"]);
        }

        size_t limit;
        if (kwargs.contains("limit")) {
            Py_ssize_t limit_ = nb::cast<Py_ssize_t>(kwargs["limit"]);
            if (limit_ < 0)
                limit = (size_t) SIZE_MAX;
            else
                limit = (size_t) limit_;
            nb::del(kwargs["limit"]);
        } else {
            limit = 20;
        }

        if (examine.backend == JitBackend::None)
            symbolic = false;

        if (symbolic) {
            if (!examine.error.empty())
                nb::raise("%s", examine.error.c_str());

            nb::handle index_tp, mask_tp;
            {
                ArrayMeta m{};
                m.backend = (uint16_t) examine.backend;
                m.type = (uint16_t) VarType::UInt32;
                m.ndim = 1;
                m.shape[0] = DRJIT_DYNAMIC;
                index_tp = meta_get_type(m);
                m.type = (uint16_t) VarType::Bool;
                mask_tp = meta_get_type(m);
            }

            dr::suspend_grad suspend_guard;
            nb::object active = nb::inst_alloc(mask_tp);

            uint32_t mask_1 = jit_var_bool(examine.backend, true),
                     mask_2 = jit_var_mask_apply(mask_1, (uint32_t) examine.size);

            supp(active.type()).init_index(mask_2, inst_ptr(active));
            nb::inst_mark_ready(active);
            jit_var_dec_ref(mask_1);
            jit_var_dec_ref(mask_2);

            if (kwargs.contains("active")) {
                active &= kwargs["active"];
                nb::del(kwargs["active"]);
            }

            nb::object counter = index_tp(0),
                       slot = scatter_inc(counter, index_tp(0), active),
                       limit_o = nb::int_(limit);

            kwargs["thread_id"] =
                arange(nb::borrow<nb::type_object_t<dr::ArrayBase>>(index_tp),
                       0, examine.size, 1);

            active &= nb::steal(
                PyObject_RichCompare(slot.ptr(), limit_o.ptr(), Py_LT));

            struct CaptureData : TransformCallback {
                size_t limit;
                nb::object active;
                nb::object slot;
                CaptureData(size_t limit, nb::object active, nb::object slot)
                    : limit(limit), active(active), slot(slot) { }

                void operator()(nb::handle h1, nb::handle h2) override {
                    nb::handle tp = h1.type();
                    const ArraySupplement &s = supp(tp);
                    if (!s.index)
                        return;

                    nb::object o = full("empty", tp, nb::handle(), limit);
                    scatter(o, nb::borrow(h1), slot, active);

                    nb::inst_replace_move(h2, o);
                }
            };

            CaptureData capture(limit, active, slot);
            jit_var_set_callback((uint32_t) supp(index_tp).index(inst_ptr(slot)),
                &DelayedPrint::callback,
                new DelayedPrint{
                    fmt, nb::borrow(file),
                    std::move(counter),
                    limit,
                    nb::borrow<nb::args>(transform(name, capture, args)),
                    nb::borrow<nb::kwargs>(transform(name, capture, kwargs)) });

            return nb::none();
        }

        schedule(args);
        schedule(kwargs);
        Buffer buffer(128);

        if (kwargs.contains("active")) {
            nb::object active = kwargs["active"];
            nb::del(kwargs["active"]);
            nb::handle active_tp = active.type();

            if (active_tp.is(&PyBool_Type)) {
                if (nb::cast<bool>(active) == false)
                    return nb::none();
            } else if (is_drjit_type(active.type())) {
                // Try to reduce the input
                try {
                    nb::object indices = ::compress(active);
                    if (nb::len(indices) == 0)
                        return nb::none();
                    nb::list args2;
                    nb::dict kwargs2;
                    for (nb::handle h: args) {
                        nb::object v = nb::borrow(h);
                        try {
                            v = ::slice(v, indices);
                        } catch (...) { }
                        args2.append(v);
                    }
                    for (nb::handle kv: kwargs.items()) {
                        nb::object k = kv[0], v = kv[1];
                        try {
                            v = ::slice(v, indices);
                        } catch (...) { }
                        kwargs2[k] = v;
                    }
                    schedule(args2);
                    schedule(kwargs2);
                    args = nb::borrow<nb::args>(nb::tuple(args2));
                    kwargs = nb::borrow<nb::kwargs>(std::move(kwargs2));
                } catch (...) { }
            } else {
                nb::raise("The 'active' argument type is unsupported.");
            }
        }

        const char *start = fmt.c_str(), *p = start;
        size_t args_pos = 0, nargs = args.size();
        bool implicit_numbering = false,
             explicit_numbering = false;

        while (*p) {
            char c = *p++;

            if (c == '{') {
                // Check for an escaped brace
                if (*p == '{') {
                    buffer.put(*p++);
                    continue;
                }

                // Find the other end of the brace expression
                const size_t pos = (size_t) (p - start - 1);
                const char *p2 = p;

                while (*p2 && *p2 != '}')
                    p2++;
                if (!*p2)
                    nb::raise(
                        "unmatched brace in format string (at position %zu).",
                        pos);

                bool equal_sign = p2[-1] == '=';

                if (equal_sign) {
                    buffer.put(p, p2-p);
                    p2--;
                }

                nb::object value;
                if (p == p2) {
                    if (explicit_numbering)
                        nb::raise("cannot switch from explicit to implicit "
                                  "field numbering.");
                    implicit_numbering = true;

                    if (args_pos >= nargs)
                        nb::raise(
                            "missing positional argument %zu referenced by "
                            "format string (at position %zu).",
                            args_pos, pos);
                    value = args[args_pos++];
                } else {
                    nb::str key(p, p2 - p);
                    if (kwargs.contains(key)) {
                        value = kwargs[key];
                    } else {
                        size_t index = (size_t) -1;
                        try {
                            nb::int_ key_i(key);
                            index = nb::cast<size_t>(key_i);
                        } catch (...) { }

                        if (index != (size_t) -1) {
                            if (implicit_numbering)
                                nb::raise("cannot switch from implicit to "
                                          "explicit field numbering.");
                            explicit_numbering = true;

                            if (index < args.size())
                                value = args[index];
                            else
                                nb::raise("missing positional argument %zu "
                                          "referenced by format string (at "
                                          "position %zu).", index, pos);
                        } else {
                            nb::raise(
                                "missing keyword argument \"%s\" referenced by "
                                "format string (at position %zu).",
                                key.c_str(), pos);
                        }
                    }
                }
                repr_general(buffer, value, 0, limit);
                if (equal_sign)
                    p2++;
                p = p2 + 1;
            } else if (c == '}') {
                if (*p++ == '}')
                    buffer.put('}');
                else
                    nb::raise("single \"}\" encountered in format string.");
            } else {
                buffer.put(c);
            }
        }

        nb::str str(buffer.get(), buffer.size());
        if (file.is_valid()) {
            file.attr("write")(str);
            return nb::none();
        } else {
            return str;
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "%s(): encountered an exception (see above).", name);
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "%s(): %s", name, e.what());
        nb::raise_python_error();
    }
}

void print_impl(const std::string &fmt, nb::args args, nb::kwargs kwargs) {
    nb::object file;
    if (kwargs.contains("file")) {
        file = kwargs["file"];
        nb::del(kwargs["file"]);
    } else {
        file = nb::module_::import_("sys").attr("stdout");
    }

    std::string end;
    if (kwargs.contains("end")) {
        end = nb::cast<const char *>(kwargs["end"]);
        nb::del(kwargs["end"]);
    } else {
        end = "\n";
    }

    format_impl("drjit.print", fmt + end, file, args, kwargs);
}

void export_print(nb::module_ &m) {
    m.def("format",
          [](const std::string &fmt, nb::args args, nb::kwargs kwargs) {
              return format_impl("drjit.format", fmt, nb::handle(), args, kwargs);
          },
          "fmt"_a.noconvert(), "args"_a, "kwargs"_a, nb::raw_doc(doc_format))
     .def("format",
          [](nb::handle value, nb::kwargs kwargs) {
              return format_impl("drjit.format", "{}", nb::handle(),
                                 nb::borrow<nb::args>(nb::make_tuple(value)), kwargs);
          })
     .def("print", &print_impl, "fmt"_a.noconvert(), "args"_a, "kwargs"_a, nb::raw_doc(doc_print))
     .def("print",
          [](nb::args args, nb::kwargs kwargs) { print_impl("{}", args, kwargs); },
          "args"_a, "kwargs"_a);
}
