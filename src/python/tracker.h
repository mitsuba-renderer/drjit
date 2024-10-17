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

/**
 * Helper class for tracking state variables during control flow operations.
 *
 * This class reads and writes state variables as part of control flow
 * operations such as :py:func:`dr.while_loop() <while_loop>` and
 * :py:func:`dr.if_stmt() <if_stmt>`. It checks that each variable remains
 * consistent across this multi-step process.
 *
 * Consistency here means that:
 *
 * - The tree structure of the :ref:`PyTree <pytrees>` PyTree is preserved
 *   across calls to :py:func:`read()`` and :py:func:`write()``.
 *
 * - The type of every PyTree element is similarly preserved.
 *
 * - The sizes of Dr.Jit arrays in the PyTree remain compatible across calls to
 *   :py:func:`read()` and :py:func:`write()`. The sizes of two arrays ``a``
 *   and ``b`` are considered compatible if ``a+b`` is well-defined (it's okay
 *   if this involves an intermediate broadcasting step.)
 *
 * In the case of an inconsistency, the implementation generates an error
 * message that identifies the problematic variable by name.
 */
struct VariableTracker {
public:
    struct Context;

    /*
     * \brief Create a new variable tracker
     *
     * The constructor accepts two parameters:
     *
     * - ``strict``: Certain types of Python objects (e.g. custom Python classes
     *   without ``DRJIT_STRUCT`` field, scalar Python numeric types) are not
     *   traversed by the variable tracker. If ``strict`` mode is enabled, any
     *   inconsistency here will cause the implementation to immediately give up
     *   with an error message. This is not always desired, hence this behavior
     *   is configurable.
     *
     * - ``check_size``: If set to ``true``, the tracker will ensure that
     *   variables remain size-compatible. The one case in Dr.Jit where this is
     *   not desired are evaluated loops with compression enabled (i.e.,
     *   inactive elements are pruned, which causes the array size to
     *   progressively shrink).
     */
    VariableTracker(bool strict = true, bool check_size = true);

    /// Free all state, decrease reference counts, etc.
    ~VariableTracker();

    /**
     * \brief Traverse a PyTree and read its variable indices.
     *
     * This function recursively traverses the PyTree ``state`` and appends the
     * indices of encountered Dr.Jit arrays to the reference-counted output
     * vector ``indices``. It performs numerous consistency checks during this
     * process to ensure that variables remain consistent over time.
     *
     * The ``labels`` argument optionally identifies the top-level variable
     * names tracked by this instance. This is recommended to obtain actionable
     * error messages in the case of inconsistencies. Otherwise,
     * ``default_label`` is prefixed to variable names.
     */
    void read(nb::handle state, dr::vector<uint64_t> &indices,
              const dr::vector<dr::string> &labels = {},
              const char *default_label = "state");

    /**
     * \brief Traverse a PyTree and write its variable indices.
     *
     * This function recursively traverses the PyTree ``state`` and updates the
     * encountered Dr.Jit arrays with indices from the ``indices`` argument.
     * It performs numerous consistency checks during this
     * process to ensure that variables remain consistent over time.
     *
     * When ``preserve_dirty`` is set to ``true``, the function leaves
     * dirty arrays (i.e., ones with pending side effects) unchanged.
     *
     * The ``labels`` argument optionally identifies the top-level variable
     * names tracked by this instance. This is recommended to obtain actionable
     * error messages in the case of inconsistencies. Otherwise,
     * ``default_label`` is prefixed to variable names.
     */
    void write(nb::handle state, const dr::vector<uint64_t> &indices,
               bool preserve_dirty = false,
               const dr::vector<dr::string> &labels = {},
               const char *default_label = "state");

    /// \brief Clear all variable state stored by the variable tracker
    void clear();

    /**
     * \brief Undo all changes and restore tracked variables to their original
     * state
     *
     * This function traverses the original PyTree captured by the first call to
     * \ref read() it undoes any structural changes (e.g., variable aliasing due
     * to assignments), if detected.
     *
     * If ``indices`` is provided, the function uses them to reinitialize any
     * encountered Dr.Jit arrays with a different set of IDs. Dirty arrays, if
     * found, will be left as-is if \ref preserve_dirty is specified.
     */
    nb::object restore(const dr::vector<dr::string> &labels = {},
                       const char *default_label = "state",
                       const dr::vector<uint64_t> *indices = nullptr,
                       bool preserve_dirty = false);

    /**
     * \brief Create a new copy of the PyTree representing the final
     * version of the PyTree following a symbolic operation.
     *
     * This function returns a PyTree representing the latest state. This PyTree
     * is created lazily, and it references the original one whenever values
     * were unchanged. This function also propagates in-place updates when
     * they are detected.
     *
     * If ``indices`` is provided, the function uses them to reinitialize any
     * encountered Dr.Jit arrays with a different set of IDs.
     */
    nb::object rebuild(
       const dr::vector<dr::string> &labels = {},
       const char *default_label = "state",
       const dr::vector<uint64_t> *indices = nullptr);

    /// Check that the PyTree is compatible with size ``size``.
    void verify_size(size_t size);

private:
    VariableTracker(const VariableTracker &) = delete;
    VariableTracker(VariableTracker &&) = delete;

private:
    struct Impl;

    Impl *m_impl;
};

extern void export_tracker(nb::module_ &);
