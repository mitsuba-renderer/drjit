/*
    tracker.cpp -- Helper class to track variables representing arguments and
    return values in symbolic operations such as dr.while_loop, dr.if_stmt, etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "tracker.h"
#include "base.h"

using index64_vector = dr::detail::index64_vector;
using VariableGroup = VariableTracker::VariableGroup;

/* See tracker.h and docstrings.rst for more complete documentation of
   the functionality implemented below */

/// Temporary data structure used during the traversal
struct TraverseContext {
    /// A vector of indices passed to or returned from the traversal
    dr::vector<uint64_t> &indices;

    /// Which set of variables are targeted by the traversal?
    VariableGroup group;

    /// Are we writing or reading variable state?
    bool write;

    /// Is this the first time we're traversing this set of variables?
    bool first_time;

    /// Label of the variable currently being traversed
    dr::string label;

    /// Stack to avoid infinite recursion
    dr::vector<nb::handle> stack;

    /// Temporary index into 'm_variables[0/1]' during traversal
    size_t var_index = 0;

    /// Temporary index into 'm_ids[0/1]' during traversal
    size_t offset = 0;

    TraverseContext(dr::vector<uint64_t> &indices, VariableGroup group,
                    bool write, bool first_time)
        : indices(indices), group(group), write(write),
          first_time(first_time) { }
};

// Temporarily push a value onto the stack
struct StackGuard {
    StackGuard(TraverseContext &ctx, nb::handle h) : stack(ctx.stack) {
        for (nb::handle h2 : stack) {
            if (h.is(h2))
                nb::raise("detected a cycle in field %s. This is not permitted.",
                          ctx.label.c_str());
        }

        stack.push_back(h);
    }

    ~StackGuard() { stack.pop_back(); }

    dr::vector<nb::handle> &stack;
};

// Temporarily append a suffix to the variable label
struct ScopedAppendLabel {
    template <typename...Ts> ScopedAppendLabel(TraverseContext &ctx, Ts&&... args) : s(ctx.label) {
        length = s.length();
        s.put(std::forward<Ts>(args)...);
    }

    ~ScopedAppendLabel() { s.resize(length); }

    dr::string &s;
    size_t length;
};

void VariableTracker::set_labels(VariableGroup group, dr::vector<dr::string> &&labels) {
    m_labels[(int) group] = std::move(labels);
}

const dr::vector<dr::string> &VariableTracker::labels(VariableGroup group) const {
    return m_labels[(int) group];
}

void VariableTracker::read(VariableGroup group, nb::handle state, dr::vector<uint64_t> &indices) {
    TraverseContext ctx(
        /* indices = */ indices,
        /* group = */ group,
        /* write = */ false,
        /* first_time = */ m_variables[(int) group].empty());
    traverse(ctx, state);
}

/// Traverse the PyTree ``state`` and write variable indices.
void VariableTracker::write(VariableGroup group, nb::handle state, const dr::vector<uint64_t> &indices) {
    TraverseContext ctx(
        /* indices = */ const_cast<dr::vector<uint64_t>&>(indices),
        /* group = */ group,
        /* write = */ true,
        /* first_time = */ m_variables[(int) group].empty()
    );
    traverse(ctx, state);
}

/// \brief Clear the internal state of the tracker (except for the labels).
void VariableTracker::clear() {
    m_key_map.clear();
    clear(VariableGroup::Outputs);
    clear(VariableGroup::Inputs);
}

/// \brief Clear the internal state of the tracker for a specific group (except for the labels).
void VariableTracker::clear(VariableGroup group) {
    m_variables[(int) group].clear();

    for(auto it = m_key_map.begin(); it != m_key_map.end(); ++it)
        it.value()[(int) group] = 0;
}

/// Reset all variable groups to their initial state
void VariableTracker::reset() {
    reset(VariableGroup::Outputs);
    reset(VariableGroup::Inputs);
}

/// Reset a specific variable group to its initial state
void VariableTracker::reset(VariableGroup group) {
    for (Variable &v : m_variables[(int) group]) {
        if (!v.index_orig)
            continue;
        const ArraySupplement &s = supp(v.value.type());
        s.reset_index(v.index_orig, inst_ptr(v.value_orig));
        if (!v.value_orig.is(v.value))
            s.reset_index(v.index_orig, inst_ptr(v.value));
    }
    clear(group);
}

/// Finalize input/output variables following the symbolic operation
void VariableTracker::finalize() {
    finalize(VariableGroup::Outputs);
    finalize(VariableGroup::Inputs);
}

// Finalize a specific variable group following a symbolic operation.
void VariableTracker::finalize(VariableGroup group) {
    for (Variable &v : m_variables[(int) group]) {
        if (!v.index_orig || v.value.is(v.value_orig))
            continue;

        const ArraySupplement &s = supp(v.value_orig.type());

        uint64_t index =
            v.mutated ? s.index(inst_ptr(v.value)) : v.index_orig;

        s.reset_index(index, inst_ptr(v.value_orig));
    }
}

void VariableTracker::check_size(size_t size) {
    check_size(VariableGroup::Inputs, size);
    check_size(VariableGroup::Outputs, size);
}

void VariableTracker::check_size(VariableGroup group, size_t size) {
    for (Variable &v : m_variables[(int) group]) {
        if (v.index == v.index_orig)
            continue;

        size_t size_2 = jit_var_size(v.index);
        if (size != size_2 && size != 1 && size_2 != 1 && !jit_var_is_dirty(v.index))
            nb::raise("this operation processes arrays of size %zu, while "
                      "state variable '%s' has an incompatible size %zu",
                      size, v.label.c_str(), size_2);
    }
}

/// Implementation detail of ``read()`` and ``write()``
void VariableTracker::traverse(TraverseContext &ctx, nb::handle state) {
    dr::vector<dr::string> &labels = m_labels[ctx.group];

    if (!labels.empty()) {
        size_t state_len = nb::len(state);
        if (labels.size() != state_len)
            nb::raise("the 'state' and 'labels' arguments have an inconsistent size");

        for (size_t i = 0; i < state_len; ++i) {
            ctx.label = labels[i];
            traverse_impl(ctx, state[i]);
        }

    } else {
        ctx.label = ctx.group == Inputs ? "arg" : "rv";
        traverse_impl(ctx, state);
    }

    if (ctx.offset != ctx.indices.size())
        nb::raise("internal error, did not consume all variable indices");
}

/**
 * Traverse a PyTree and either read or update the encountered Jit-compiled
 * variables. The function performs many checks to detect and report
 * inconsistencies with a useful error message that identifies variables by
 * name.
 */
void VariableTracker::traverse_impl(TraverseContext &ctx, nb::handle h) {
    StackGuard stack_guard(ctx, h);

    nb::handle tp = h.type();
    dr::vector<Variable> &vars = m_variables[ctx.group];
    Variable *v = nullptr;
    nb::object prev_value;

    if (ctx.first_time) {
        // When traversing a PyTree for the first time, ensure that the entries
        // are unique and consistent between outputs and outputs (if both are
        // present).
        std::array<size_t, VariableGroup::Count> &pos = m_key_map[ctx.label];

        if (pos[ctx.group])
            nb::raise("state variable '%s' occurs more than once",
                      ctx.label.c_str());
        pos[ctx.group] = vars.size() + 1;

        vars.emplace_back(ctx.label, nb::borrow(h));
        v = &vars.back();

        size_t index_2 = pos[1 - ctx.group];
        if (index_2) {
            const Variable &v2 = m_variables[1-ctx.group][index_2 - 1];
            if (!v2.value.type().is(tp))
                nb::raise(
                    "the type of state variable '%s' changed from '%s' to "
                    "'%s', which is not permitted",
                    ctx.label.c_str(), nb::inst_name(v2.value).c_str(),
                    nb::type_name(tp).c_str());
        }
    } else {
        // When re-traversing a PyTree subsequently, ensure that the
        // names and types of the state variables remain consistent.

        if (ctx.var_index >= vars.size())
            nb::raise(
                "the number of state variables and their structure must "
                "remain fixed while tracing symbolic operations. Aborting "
                "because a new state variable of '%s' of type '%s' was "
                "added during the operation, which is not permitted",
                ctx.label.c_str(), nb::type_name(tp).c_str());

        v = &vars[ctx.var_index++];

        if (v->label != ctx.label)
            nb::raise(
                "the number of state variables and their structure must "
                "remain fixed while tracing symbolic operations. Aborting "
                "because state variable of '%s' of type '%s' cannot be "
                "found anymore. Instead, another variable '%s' of type "
                "'%s' was found in its place, which is not permitted",
                v->label.c_str(), nb::inst_name(v->value).c_str(),
                ctx.label.c_str(), nb::type_name(tp).c_str());

        if (!v->value.type().is(tp))
            nb::raise(
                "the type of state variable '%s' changed from '%s' to "
                "'%s', which is not permitted",
                ctx.label.c_str(), nb::inst_name(v->value).c_str(),
                nb::type_name(tp).c_str());

        prev_value = nb::borrow(v->value);
        v->value = nb::borrow(h);
    }

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if ((JitBackend) s.backend == JitBackend::None)
            return;

        if (s.is_tensor) {
            ScopedAppendLabel guard(ctx, ".array");
            traverse_impl(ctx, nb::steal(s.tensor_array(h.ptr())));
        } else if (s.ndim > 1) {
            Py_ssize_t len = s.shape[0];
            if (len == DRJIT_DYNAMIC)
                len = s.len(inst_ptr(h));

            for (Py_ssize_t i = 0; i < len; ++i) {
                ScopedAppendLabel guard(ctx, "[", i, "]");
                traverse_impl(ctx, nb::steal(s.item(h.ptr(), i)));
            }
        } else if (s.index) {
            uint64_t idx = s.index(inst_ptr(h));

            if (!idx && ctx.first_time && ctx.write) {
                // This cases arises when revisiting a state variables as part
                // of an AD traversal. This state state will have been passed
                // through drjit.detail.reset(), which creates a structural copy
                // with uninitialized Dr.Jit arrays.

                if (ctx.offset >= ctx.indices.size())
                    nb::raise("internal error at state variable '%s': ran "
                              "out of indices", ctx.label.c_str());

                idx = ctx.indices[ctx.offset];
                s.reset_index(idx, inst_ptr(h));
            }

            //printf("%s '%s' (%p) %llu\n", ctx.write ? "WRITE" : "READ ", v->label.c_str(), h.ptr(), idx);
            if (!idx)
                nb::raise("state variable '%s' of type '%s' is uninitialized",
                          ctx.label.c_str(), nb::inst_name(h).c_str());

            VarInfo vi = jit_set_backend((uint32_t) idx);

            if (ctx.first_time) {
                v->index_orig = ad_var_inc_ref(idx);
                v->index = ad_var_inc_ref(idx);
                v->size = vi.size;
            } else {
                if (vi.size != v->size && vi.size != 1 && v->size != 1 && m_check_size)
                    nb::raise("the size of state variable '%s' of "
                              "type '%s' changed from %zu to %zu. These "
                              "sizes aren't compatible, and such a change "
                              "is therefore not permitted",
                              ctx.label.c_str(), nb::inst_name(h).c_str(),
                              v->size, vi.size);

                v->mutated |= !ctx.write && h.is(prev_value) && idx != v->index;
                ad_var_dec_ref(v->index);
                v->index = ad_var_inc_ref(idx);
                v->size = vi.size;
            }

            if (!ctx.write) {
                ad_var_inc_ref(idx);
                ctx.indices.push_back(idx);
                ctx.offset++;
                // printf("-> %llu\n", idx);
            } else {
                if (ctx.offset >= ctx.indices.size())
                    nb::raise("internal error at state variable '%s': ran "
                              "out of indices", ctx.label.c_str());

                uint64_t idx_new = ctx.indices[ctx.offset++];
                VarInfo vi_new = jit_set_backend((uint32_t) idx_new);

                if (vi_new.size != vi.size && vi_new.size != 1 && vi.size != 1 && m_check_size)
                    nb::raise("the symbolic operation tried to change the "
                              "size of state variable '%s' of type "
                              "'%s' from %zu to %zu. Aborting because "
                              "these sizes aren't compatible",
                              ctx.label.c_str(), nb::inst_name(h).c_str(),
                              vi.size, vi_new.size);

                if (vi.type != vi_new.type)
                    nb::raise("internal error: the JIT variable type of "
                              "state variable '%s' (of type '%s') changed "
                              "from '%s' to '%s'",
                              ctx.label.c_str(), nb::inst_name(h).c_str(),
                              jit_type_name(vi.type), jit_type_name(vi_new.type));

                if (idx != idx_new) {
                    s.reset_index(idx_new, inst_ptr(h));
                    // printf("-> %llu\n", idx_new);
                }
            }
        }
    } else if (tp.is(&PyList_Type)) {
        size_t ctr = 0;
        for (nb::handle v: nb::borrow<nb::list>(h)) {
            ScopedAppendLabel guard(ctx, "[", ctr++, "]");
            traverse_impl(ctx, v);
        }
    } else if (tp.is(&PyTuple_Type)) {
        size_t ctr = 0;
        for (nb::handle v: nb::borrow<nb::tuple>(h)) {
            ScopedAppendLabel guard(ctx, "[", ctr++, "]");
            traverse_impl(ctx, v);
        }
    } else if (tp.is(&PyDict_Type)) {
        for (nb::handle kv: nb::borrow<nb::dict>(h).items()) {
            ScopedAppendLabel guard(ctx, "[", nb::repr(kv[0]).c_str(), "]");
            traverse_impl(ctx, kv[1]);
        }
    } else {
        nb::object dstruct = nb::getattr(tp, "DRJIT_STRUCT", nb::handle());
        nb::object traverse_cb = nb::getattr(
            h, ctx.write ? "_traverse_1_cb_rw" : "_traverse_1_cb_ro",
            nb::handle());

        if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
            for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                ScopedAppendLabel guard(ctx, ".", nb::str(k).c_str());
                traverse_impl(ctx, nb::getattr(h, k));
            }
        } else if (traverse_cb.is_valid()) {
            ScopedAppendLabel guard(ctx, "._traverse_cb()");
            m_tmp_ctx = &ctx;
            nb::object self = nb::cast(this, nb::rv_policy::reference);
            nb::object self_meth = self.attr(ctx.write ? "_traverse_write" : "_traverse_read");
            traverse_cb(self_meth);
            m_tmp_ctx = nullptr;
        }
    }
}

uint64_t VariableTracker::_traverse_write(uint64_t idx) {
    if (!m_tmp_ctx)
        nb::raise("The _traverse_rw() function is an internal API and should "
                  "not be called in user code.");

    TraverseContext &ctx = *m_tmp_ctx;
    if (ctx.offset >= ctx.indices.size())
        nb::raise("internal error after state variable '%s': ran "
                  "out of indices", ctx.label.c_str());

    uint64_t idx_new = ctx.indices[ctx.offset++];

    if (!idx_new)
        nb::raise("internal error after state variable "
                  "'%s': uninitialized variable",
                  ctx.label.c_str());

    VarInfo vi = jit_set_backend(idx),
            vi_new = jit_set_backend(idx_new);

    if (vi.size != vi_new.size && vi.size != 1 && vi_new.size != 1 &&
        m_check_size)
        nb::raise(
            "the size of an unnamed state variable "
            "inside '%s' changed from %zu to %zu. "
            "These sizes aren't compatible, and such a "
            "change is therefore not permitted",
            ctx.label.c_str(), vi.size, vi_new.size);

    if (vi.type != vi_new.type)
        nb::raise(
            "the type of an unnamed state variable "
            "inside '%s' changed from '%s' to '%s', "
            "which is not permitted",
            ctx.label.c_str(), jit_type_name(vi.type),
            jit_type_name(vi_new.type));

    return idx_new;
}

void VariableTracker::_traverse_read(uint64_t index) {
    if (!m_tmp_ctx)
        nb::raise("The _traverse_ro() function is an internal API and should "
                  "not be called in user code.");

    TraverseContext &ctx = *m_tmp_ctx;
    ad_var_inc_ref(index);
    ctx.indices.push_back(index);
    ctx.offset++;
}


void export_tracker(nb::module_ &m) {
    // Reference-counted index vector. This class stores references to Dr.Jit
    // variables and generally behaves like a ``list[int]``. The main difference
    // is that it holds references to the elements so that they cannot expire.

    // The main purpose of this class is to represent the inputs and outputs of
    // :py:func:`drjit.detail.VariableTracker.read` and
    // :py:func:`drjit.detail.VariableTracker.write`.

    nb::class_<index64_vector>(m, "IndexVector", doc_detail_IndexVector)
        .def(nb::init<>())
        .def("append", &index64_vector::push_back_borrow)
        .def("clear", &index64_vector::release)
        .def("__len__", &index64_vector::size)
        .def("__getitem__",
             [](index64_vector &v, size_t i) {
                 if (i >= v.size())
                     throw nb::index_error();
                 return v[i];
             })
        .def("__setitem__", [](index64_vector &v, size_t i, uint64_t value) {
            if (i >= v.size())
                throw nb::index_error();
            ad_var_inc_ref(value);
            ad_var_dec_ref(v[i]);
            v[i] = value;
        });

    auto vt =
        nb::class_<VariableTracker>(m, "VariableTracker",
                                    doc_detail_VariableTracker)
            .def(nb::init<>())
            .def("clear",
                 nb::overload_cast<VariableTracker::VariableGroup>(
                     &VariableTracker::clear),
                 doc_detail_VariableTracker_clear)
            .def("clear", nb::overload_cast<>(&VariableTracker::clear),
                 doc_detail_VariableTracker_clear_2)
            .def("reset",
                 nb::overload_cast<VariableTracker::VariableGroup>(
                     &VariableTracker::reset),
                 doc_detail_VariableTracker_reset)
            .def("reset", nb::overload_cast<>(&VariableTracker::reset),
                 doc_detail_VariableTracker_reset_2)
            .def("finalize",
                 nb::overload_cast<VariableTracker::VariableGroup>(
                     &VariableTracker::finalize),
                 doc_detail_VariableTracker_finalize)
            .def("finalize", nb::overload_cast<>(&VariableTracker::finalize),
                 doc_detail_VariableTracker_finalize_2)
            .def("check_size",
                 nb::overload_cast<VariableTracker::VariableGroup, size_t>(
                     &VariableTracker::check_size),
                 doc_detail_VariableTracker_check_size)
            .def("check_size", nb::overload_cast<size_t>(&VariableTracker::check_size),
                 doc_detail_VariableTracker_check_size_2)
            .def("labels", &VariableTracker::labels, "group"_a,
                 doc_detail_VariableTracker_labels)
            .def("set_labels", &VariableTracker::set_labels, "group"_a,
                 "labels"_a, doc_detail_VariableTracker_set_labels)
            .def("write", &VariableTracker::write, "group"_a, "state"_a,
                 "indices"_a, doc_detail_VariableTracker_write)
            .def("read",
                 [](VariableTracker &vt, VariableTracker::VariableGroup group,
                    nb::handle state) {
                     index64_vector indices;
                     vt.read(group, state, indices);
                     return indices;
                 },
                 "group"_a, "state"_a, doc_detail_VariableTracker_read)
            .def("_traverse_read", &VariableTracker::_traverse_read)
            .def("_traverse_write", &VariableTracker::_traverse_write);

    nb::enum_<VariableTracker::VariableGroup>(vt, "VariableGroup")
        .value("Inputs", VariableTracker::VariableGroup::Inputs)
        .value("Outputs", VariableTracker::VariableGroup::Outputs)
        .export_values();
}
