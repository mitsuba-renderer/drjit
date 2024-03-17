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
#include <tsl/robin_map.h>
#include <drjit/autodiff.h>
#include <utility>
#include <string_view>
#include <array>

using dr::detail::index64_vector;

/**
 * Helper class for tracking state variables during control flow operations.
 *
 * This class reads and updates state variables as part of control flow
 * operations such as ``dr.while_loop()`` and ``dr.if_stmt()``. It checks
 * that each variable remains consistent across this multi-step process.
 *
 * Consistency here means that:
 *
 * - The tree structure of the :ref:`PyTree <pytrees>` PyTree is preserved
 *   across calls to :py:func:`read()`` and :py:func:`write()``.
 *
 * - The type of very PyTree element is similarly preserved.
 *
 * - The sizes of Dr.Jit arrays in the PyTree remain compatible across calls to
 *   :py:func:`read()` and :py:func:`write()`. The sizes of two arrays ``a``
 *   and ``b`` are considered compatible if ``a+b`` is well-defined (this may
 *   require broadcasting).
 *
 * In the case of an inconsistency, the implementation generates an error
 * message that identifies the problematic variable by name. This requires the
 * assignment of variable labels via the :py:func:`set_labels()` function.
 *
 * Variables are tracked in two groups: inputs and outputs. In the case of an
 * operation like :py:func:`drjit.while_loop()` where inputs and outputs
 * coincide, only the latter group is actually used. If a variable occurs both
 * as an input and as an output (as identified via its label), then its
 * consistency is also checked across groups.
 */
struct VariableTracker {
private:
    // Forward declarations
    struct Variable;
    struct TraverseContext;

public:
    // Identifies variables targeted by ``read()`` and ``write()`` operations
    enum VariableGroup {
        Inputs = 0,
        Outputs = 1,
        Count
    };

    VariableTracker() = default;

    /*
     * \brief Label the variables of the given group. This is important to
     * obtain actionable error messages in case of inconsistencies.
     */
    void set_labels(VariableGroup group, dr::vector<dr::string> &&labels) {
        m_labels[(int) group] = std::move(labels);
    }

    /// Return the current set of labels for the given group
    const dr::vector<dr::string> &labels(VariableGroup group) const {
        return m_labels[(int) group];
    }

    /**
     * \brief Traverse the PyTree ``state`` and read variable indices.
     *
     * The implementation performs the consistency checks mentioned in the class
     * description and appends the indices of encountered Dr.Jit arrays to the
     * reference-counted output vector ``indices``.
     */
    void read(VariableGroup group, nb::handle state, index64_vector &indices) {
        TraverseContext ctx(
            /* indices = */ indices,
            /* group = */ group,
            /* write = */ false,
            /* first_time = */ m_variables[(int) group].empty());
        traverse(ctx, state);
    }

    /**
     * \brief Traverse the PyTree ``state`` and write variable indices.
     *
     * The implementation performs the consistency checks mentioned in the class
     * description and appends the indices of encountered Dr.Jit arrays to the
     * reference-counted output vector ``indices``.
     */
    void write(VariableGroup group, nb::handle state, const index64_vector &indices) {
        TraverseContext ctx(
            /* indices = */ const_cast<index64_vector&>(indices),
            /* group = */ group,
            /* write = */ true,
            /* first_time = */ m_variables[(int) group].empty()
        );
        traverse(ctx, state);
    }

    /// \brief Clear the internal state of the tracker (except for the labels).
    void clear() {
        clear(VariableGroup::Outputs);
        clear(VariableGroup::Inputs);
    }

    /// \brief Clear the internal state of the tracker for a specific group (except for the labels).
    void clear(VariableGroup group) {
        m_variables[(int) group].clear();
    }

    /// Reset all variable groups to their initial state
    void reset() {
        reset(VariableGroup::Outputs);
        reset(VariableGroup::Inputs);
    }

    /// Reset a specific variable group to its initial state
    void reset(VariableGroup group) {
        for (Variable &v : m_variables[(int) group]) {
            nb::handle h = v.value_orig;
            if (v.index_orig)
                supp(h.type()).reset_index(v.index_orig, inst_ptr(h));
        }
        clear(group);
    }

    /// Finalize input/output variables following the symbolic operation
    void finalize() {
        finalize(VariableGroup::Outputs);
        finalize(VariableGroup::Inputs);
    }

    /**
     * Finalize a specific variable group following a symbolic operation.
     *
     * If a variable was consistently *replaced* by another variable without
     * mutating the original variable's contents, then this operation will
     * restore the original variable to its initial value.
     *
     * This needed so that an operation such as
     *
     * .. code-block:: python
     *
     *    @dr.syntax
     *    def f(x):
     *       if x < 0:
     *           # Associate the name 'x' with a new variable but don't change
     *           # the original variable.
     *           x = x + 1 return x
     *
     * does not mutate the caller-provided ``x``, which would be surprising and
     * bug-prone.
     *
     * On the other hand, ``x`` *will* be mutated in the next snippet since an
     * in-place update was performed within the ``if`` statement.
     *
     * .. code-block:: python
     *
     *    @dr.syntax
     *    def f(x):
     *       if x < 0:
     *           x += 1 # Replace the contents of 'x' with 'x+1'
     *
     * Some situations are less clearly defined. Consider the following
     * conditional, where one branch mutates and the other one replaces a
     * variable.
     *
     * .. code-block:: python
     *
     *    @dr.syntax
     *    def f(x):
     *       if x < 0:
     *           x += 1
     *       else:
     *           x = x + 2
     *
     * In this case, finalization will consider ``x`` to be mutated and
     * rewrite the caller-provided ``x`` so that it reflects the outcome
     * of both branches.
     */
    void finalize(VariableGroup group) {
        for (Variable &v : m_variables[(int) group]) {
            if (!v.index_orig)
                continue;

            const ArraySupplement &s = supp(v.value_orig.type());

            uint64_t index =
                v.mutated ? s.index(inst_ptr(v.value)) : v.index_orig;
            s.reset_index(index, inst_ptr(v.value_orig));
        }
    }

private:
    /// Implementation detail of ``read()`` and ``write()``
    void traverse(TraverseContext &ctx, nb::handle state) {
        if (ctx.first_time && ctx.write)
            nb::raise("traverse(): variables must be read before they can be written.");

        dr::vector<dr::string> &labels = m_labels[ctx.group];

        if (!labels.empty()) {
            size_t state_len = nb::len(state);
            if (labels.size() != state_len)
                nb::raise("the 'state' and 'labels' arguments have an inconsistent size.");

            for (size_t i = 0; i < state_len; ++i) {
                ctx.label = labels[i];
                traverse_impl(ctx, state[i]);
            }

        } else {
            ctx.label = ctx.group == Inputs ? "arg" : "rv";
            traverse_impl(ctx, state);
        }
    }

    /**
     * Traverse a PyTree and either read or update the encountered Jit-compiled
     * variables. The function performs many checks to detect and report
     * inconsistencies with a useful error message that identifies variables by
     * name.
     */
    void traverse_impl(TraverseContext &ctx, nb::handle h) {
        StackGuard stack_guard(ctx, h);

        nb::handle tp = h.type();
        dr::vector<Variable> &vars = m_variables[ctx.group];
        Variable *v = nullptr;
        nb::object prev_value;

        if (!ctx.write && ctx.first_time) {
            // When reading a PyTree for the first time, ensure that the entries are unique
            // and consistent between outputs and outputs (if both are present).
            std::array<size_t, VariableGroup::Count> &pos = m_key_map[ctx.label.c_str()];

            if (pos[ctx.group])
                nb::raise("state variable '%s' occurs more than once",
                          ctx.label.c_str());
            pos[ctx.group] = vars.size();

            vars.emplace_back(ctx.label, nb::borrow(h));
            v = &vars.back();

            size_t index_2 = pos[1 - ctx.group];
            if (index_2) {
                const Variable &v2 = m_variables[1-ctx.group][index_2];
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
                uint64_t i1 = s.index(inst_ptr(h));
                VarInfo vi1 = jit_set_backend((uint32_t) i1);

                if (vi1.size)
                    nb::raise("state variable '%s' of type '%s' is uninitialized",
                              ctx.label.c_str(), nb::inst_name(h).c_str());

                if (ctx.first_time) {
                    v->index_orig = ad_var_inc_ref(i1);
                } else {
                    uint64_t i2 = s.index(inst_ptr(prev_value));
                    size_t s2 = jit_var_size((uint32_t) i2);

                    if (vi1.size != s2 && vi1.size != 1 && s2 != 1)
                        nb::raise("the size of loop state variable '%s' of "
                                  "type '%s' changed from %zu to %zu. These "
                                  "sizes aren't compatible, and such a change "
                                  "is therefore not permitted",
                                  ctx.label.c_str(), nb::inst_name(h).c_str(),
                                  vi1.size, s2);

                    v->mutated |= !ctx.write && h.is(prev_value) && i1 != i2;
                }

                if (!ctx.write) {
                    ctx.indices.push_back_borrow(i1);
                    ctx.offset++;
                } else {
                    if (ctx.offset >= ctx.indices.size())
                        nb::raise("internal error at state variable '%s': ran "
                                  "out of indices",
                                  ctx.label.c_str());

                    uint64_t i3 = ctx.indices[ctx.offset++];
                    VarInfo vi3 = jit_set_backend((uint32_t) i3);

                    if (vi3.size != vi1.size && vi3.size != 1 && vi1.size != 1)
                        nb::raise("the symbolic operation tried to change the "
                                  "size of loop state variable '%s' of type "
                                  "'%s' from %zu to %zu. Aborting because "
                                  "these sizes aren't compatible",
                                  ctx.label.c_str(), nb::inst_name(h).c_str(),
                                  vi1.size, vi3.size);

                    if (vi1.type != vi3.type)
                        nb::raise("internal error: the JIT variable type of loop "
                                  "state variable '%s' (of type '%s') changed "
                                  "from '%s' to '%s'",
                                  ctx.label.c_str(), nb::inst_name(h).c_str(),
                                  jit_type_name(vi1.type), jit_type_name(vi3.type));

                    supp(tp).reset_index(i3, inst_ptr(h));
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
                tp, ctx.write ? "_traverse_1_cb_rw" : "_traverse_1_cb_ro",
                nb::handle());

            if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
                for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                    ScopedAppendLabel guard(ctx, ".", nb::str(k).c_str());
                    traverse_impl(ctx, nb::getattr(h, k));
                }
            } else if (traverse_cb.is_valid()) {
                ScopedAppendLabel guard(ctx, "._traverse_cb()");
                if (ctx.write) {
                    traverse_cb(
                        nb::cpp_function([&ctx](uint64_t i1) -> uint64_t {
                            if (ctx.offset >= ctx.indices.size())
                                nb::raise("internal error at state variable '%s': ran "
                                          "out of indices",
                                          ctx.label.c_str());
                                        nb::raise("internal error, ran out of indices (2)");

                            uint64_t i2 = ctx.indices[ctx.offset++];

                            VarInfo vi1 = jit_set_backend(i1),
                                    vi2 = jit_set_backend(i2);

                            if (vi1.size != vi2.size && vi1.size != 1 && vi2.size != 1)
                                nb::raise(
                                    "the size of an unnamed state variable "
                                    "inside '%s' changed from %zu to %zu. "
                                    "These sizes aren't compatible, and such a "
                                    "change is therefore not permitted",
                                    ctx.label.c_str(), vi1.size, vi2.size);

                            if (vi1.type != vi2.type)
                                nb::raise(
                                    "the type of an unnamed state variable "
                                    "inside '%s' changed from '%s' to '%s', "
                                    "which is not permitted",
                                    ctx.label.c_str(), jit_type_name(vi1.type),
                                    jit_type_name(vi2.type));

                            return i2;
                        })
                    );
                } else {
                    traverse_cb(
                        nb::cpp_function([&ctx](uint64_t index) {
                            ctx.indices.push_back_borrow(index);
                            ctx.offset++;
                        })
                    );
                }
            }
        }
    }
private:
    VariableTracker(const VariableTracker &) = delete;
    VariableTracker(VariableTracker &&) = delete;

    // State of a single variable before and during an operation
    struct Variable {
        dr::string label;
        nb::object value_orig;
        nb::object value;
        uint64_t index_orig;
        bool mutated;
        ~Variable() { ad_var_dec_ref(index_orig); }

        Variable(const dr::string &label, const nb::object &value)
            : label(label), value_orig(value), value(value), index_orig(0), mutated(false) { }
        Variable(Variable &&v) noexcept : label(std::move(v.label)), value_orig(std::move(v.value_orig)),
                                          value(std::move(v.value)), index_orig(v.index_orig), mutated(v.mutated) {
            v.index_orig = 0;
        }
        Variable(const Variable &v) = delete;
    };

    /// Temporary data structure used during the traversal
    struct TraverseContext {
        /// A vector of indices passed to or returned from the traversal
        index64_vector &indices;

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

        TraverseContext(index64_vector &indices, VariableGroup group,
                        bool write, bool first_time)
            : indices(indices), group(group), write(write),
              first_time(first_time) {}
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


private:
    // Labels identifying top-lvel elements in 'm_variables'
    dr::vector<dr::string> m_labels[VariableGroup::Count];

    /// A list of input (if separately tracked) and output variables
    dr::vector<Variable> m_variables[VariableGroup::Count];

    /// A mapping from variable identifiers to their indices in 'm_variables'
    tsl::robin_map<std::string_view, std::array<size_t, VariableGroup::Count>> m_key_map;
};

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
                 "group"_a, "state"_a, doc_detail_VariableTracker_read);

    nb::enum_<VariableTracker::VariableGroup>(vt, "VariableGroup")
        .value("Inputs", VariableTracker::VariableGroup::Inputs)
        .value("Outputs", VariableTracker::VariableGroup::Outputs)
        .export_values();
}
