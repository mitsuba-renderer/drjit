/*
    tracker.h -- Helper class to track variables representing arguments and
    return values in symbolic operations such as dr.while_loop, dr.if_stmt, etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include <drjit/autodiff.h>
#include <string_view>
#include <tsl/robin_map.h>
#include <array>

struct TraverseContext;

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
    void set_labels(VariableGroup group, dr::vector<dr::string> &&labels);

    /// Return the current set of labels for the given group
    const dr::vector<dr::string> &labels(VariableGroup group) const;

    /// Should VariableTracker ensure that variables maintain compatible sizes?
    bool check_size() const { return m_check_size; }

    /// Should VariableTracker ensure that variables maintain compatible sizes?
    void set_check_size(bool value) { m_check_size = value; }

    /**
     * \brief Traverse the PyTree ``state`` and read variable indices.
     *
     * The implementation performs the consistency checks mentioned in the class
     * description and appends the indices of encountered Dr.Jit arrays to the
     * reference-counted output vector ``indices``.
     */
    void read(VariableGroup group, nb::handle state, dr::vector<uint64_t> &indices);

    /**
     * \brief Traverse the PyTree ``state`` and write variable indices.
     *
     * The implementation performs the consistency checks mentioned in the class
     * description and appends the indices of encountered Dr.Jit arrays to the
     * reference-counted output vector ``indices``.
     */
    void write(VariableGroup group, nb::handle state, const dr::vector<uint64_t> &indices);

    /// \brief Clear the internal state of the tracker (except for the labels).
    void clear();

    /// \brief Clear the internal state of the tracker for a specific group (except for the labels).
    void clear(VariableGroup group);

    /// Reset all variable groups to their initial state
    void reset();

    /// Reset a specific variable group to its initial state
    void reset(VariableGroup group);

    /// Check that all modified variables are compatible with size ``size``.
    void check_size(size_t size);

    /// Check that modified variables in ``group`` are compatible with size ``size``.
    void check_size(VariableGroup group, size_t size);

    /// Finalize input/output variables following the symbolic operation
    void finalize();

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
    void finalize(VariableGroup group);


    // Internal API for type-erased traversal
    uint64_t _traverse_write(uint64_t idx);
    void _traverse_read(uint64_t index);

private:
    VariableTracker(const VariableTracker &) = delete;
    VariableTracker(VariableTracker &&) = delete;

    // State of a single variable before and during an operation
    struct Variable {
        dr::string label;
        nb::object value_orig;
        nb::object value;
        uint64_t index_orig;
        uint64_t index;
        size_t size;
        bool mutated;

        ~Variable() {
            ad_var_dec_ref(index_orig);
            ad_var_dec_ref(index);
        }

        Variable(const dr::string &label, const nb::object &value)
            : label(label), value_orig(value), value(value), index_orig(0),
              index(0), size(0), mutated(false) { }

        Variable(Variable &&v) noexcept
            : label(std::move(v.label)), value_orig(std::move(v.value_orig)),
              value(std::move(v.value)), index_orig(v.index_orig),
              index(v.index_orig), size(v.size), mutated(v.mutated) {
            v.index_orig = 0;
            v.index = 0;
            v.size = 0;
        }

        Variable(const Variable &v) = delete;
    };

    struct StringHash {
        size_t operator()(const dr::string &s) const {
            return std::hash<std::string_view>()(std::string_view(s.c_str(), s.size()));
        }
    };

    /// Implementation detail of ``read()`` and ``write()``
    void traverse(TraverseContext &ctx, nb::handle state);

    /**
     * Traverse a PyTree and either read or update the encountered Jit-compiled
     * variables. The function performs many checks to detect and report
     * inconsistencies with a useful error message that identifies variables by
     * name.
     */
    void traverse_impl(TraverseContext &ctx, nb::handle h);

private:
    // Labels identifying top-lvel elements in 'm_variables'
    dr::vector<dr::string> m_labels[VariableGroup::Count];

    /// A list of input (if separately tracked) and output variables
    dr::vector<Variable> m_variables[VariableGroup::Count];

    /// A mapping from variable identifiers to their indices in 'm_variables'
    tsl::robin_map<dr::string, std::array<size_t, VariableGroup::Count>, StringHash> m_key_map;

    /// Perform extra checks to ensure that the size of variables remains compatible?
    bool m_check_size = true;

    /// Temporary stratch space to hold a pointer to an active TraverseContext
    TraverseContext *m_tmp_ctx;
};

extern void export_tracker(nb::module_ &);
