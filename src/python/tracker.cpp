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
#include "shape.h"
#include <string_view>
#include <drjit/autodiff.h>
#include <tsl/robin_map.h>

// #define DEBUG_TRACKER

using index64_vector = dr::detail::index64_vector;

/**
 * This struct represents the state of Python object encountered during PyTree
 * traversal. It could store a Dr.Jit array, a Python list/dict, etc.
 *
 * The VariableMap data structure declared later maps state variable names
 * to ``Variable`` instances.
 */
struct Variable {
    /// The Python object encountered during the first PyTree traversal
    nb::object value_orig;

    /// The Python object encountered during the last PyTree traversal
    nb::object value;

    /// For Dr.Jit arrays: the array ID encountered during the first PyTree traversal
    uint64_t index_orig;

    /// For Dr.Jit arrays: the array ID encountered during the last PyTree traversal
    uint64_t index;

    /// The size of the array/list/dict/..
    size_t size;

    /// Set to true when an in-place mutation was detected
    bool mutated;

    ~Variable() {
        ad_var_dec_ref(index_orig);
        ad_var_dec_ref(index);
    }

    /// Initialize from an existing object. The array indices must be set separately
    Variable(nb::handle value)
        : value_orig(nb::borrow(value)), value(nb::borrow(value)), index_orig(0),
          index(0), size(0), mutated(false) { }

    /// Move constructor
    Variable(Variable &&v) noexcept
        : value_orig(std::move(v.value_orig)),
          value(std::move(v.value)), index_orig(v.index_orig),
          index(v.index), size(v.size), mutated(v.mutated) {
        v.index_orig = 0;
        v.index = 0;
        v.size = 0;
        v.mutated = 0;
    }

    /// Move assignment
    Variable &operator=(Variable &&v) noexcept {
        value_orig = std::move(v.value_orig);
        value = std::move(v.value);
        uint64_t old0 = index_orig, old1 = index;
        index_orig = v.index_orig;
        index = v.index;
        v.index = 0;
        v.index_orig = 0;
        ad_var_dec_ref(old0);
        ad_var_dec_ref(old1);
        size = v.size;
        mutated = v.mutated;
        v.size = 0;
        v.mutated = 0;
        return *this;
    }

    // Variables can be move-constructed/assigned, but that's it.
    Variable(const Variable &v) = delete;
    Variable &operator=(const Variable &) = delete;
};

/// Hash function for ``dr::string`` based on the builtin STL string hasher
struct StringHash {
    size_t operator()(const dr::string &s) const {
        return std::hash<std::string_view>()(std::string_view(s.data(), s.size()));
    }
};

/// Associative data structure mapping from names strings to ``Variable`` instances
using VariableMap =
    tsl::robin_map<dr::string, Variable, StringHash,
                   std::equal_to<dr::string>,
                   std::allocator<std::pair<dr::string, Variable>>, true>;

// Implementation details of VariableTracker. Implemented via PIMPL
// so that callers don't inherit the STL header file dependencies required
// by the VariableMap declaration above
struct VariableTracker::Impl {
    Impl(bool strict, bool check_size)
        : strict(strict), check_size(check_size) { }

    /// Pointer to a hash table storing the variable state
    VariableMap state;

    /// Enable extra-strict consistency checks?
    bool strict;

    /// Perform extra checks to ensure that the size of variables remains compatible?
    bool check_size;

    /// Implementation detail of ``read()`` and ``write()``
    void traverse(Context &ctx, nb::handle state,
                  const dr::vector<dr::string> &labels,
                  const char *default_label);

    /**
     * Traverse a PyTree and either read or update the encountered Jit-compiled
     * variables. The function performs many checks to detect and report
     * inconsistencies with a useful error message that identifies variables by
     * name.
     *
     * The function returns 'true' when changes in the subtree were detected.
     */
    bool traverse(Context &ctx, nb::handle h);

    /// Undo all changes and restore tracked variables to their original state
    nb::object restore(dr::string &label);

    /// Rebuild the final state of a PyTree following an operation
    std::pair<nb::object, bool> rebuild(dr::string &label);
};


/**
 * \brief Temporary data structure used during the traversal
 *
 * This data structure tracks various quantities that are temporarily needed
 * while traversing a PyTree. It is instantiated by the
 * ``VariableTracker::read()`` and ``VariableTracker::write()`` functions.
 */
struct VariableTracker::Context {
    /// A vector of indices passed to or returned from the traversal
    dr::vector<uint64_t> &indices;

    /// Are we writing or reading variable state?
    bool write;

    /// Preserve dirty variables when writing variable indices?
    bool preserve_dirty;

    /// Enable strict size checks?
    bool check_size;

    /// Label of the variable currently being traversed
    dr::string label;

    /// A stack to avoid infinite recursion
    dr::vector<nb::handle> stack;

    /// Temporary index into 'indices' during traversal
    size_t index_offset;

    Context(dr::vector<uint64_t> &indices, bool write, bool preserve_dirty,
            bool check_size)
        : indices(indices), write(write), preserve_dirty(preserve_dirty),
          check_size(check_size), index_offset(0) { }

    // Internal API for type-erased traversal
    uint64_t _traverse_write(uint64_t idx);
    void _traverse_read(uint64_t index);
};

// Temporarily push a value onto the stack
struct StackGuard {
    StackGuard(VariableTracker::Context &ctx, nb::handle h) : stack(ctx.stack) {
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
    template <typename...Ts> ScopedAppendLabel(dr::string &s, Ts&&... args) : s(s) {
        length = s.length();
        s.put(std::forward<Ts>(args)...);
    }

    template <typename...Ts> ScopedAppendLabel(VariableTracker::Context &ctx, Ts&&... args) : s(ctx.label) {
        length = s.length();
        s.put(std::forward<Ts>(args)...);
    }

    ~ScopedAppendLabel() { s.resize(length); }

    dr::string &s;
    size_t length;
};

VariableTracker::VariableTracker(bool strict, bool check_size)
    : m_impl(new Impl(strict, check_size)) {
}

VariableTracker::~VariableTracker() {
    delete m_impl;
}

void VariableTracker::read(nb::handle state, dr::vector<uint64_t> &indices,
                           const dr::vector<dr::string> &labels,
                           const char *default_label) {
    Context ctx(
        /* indices = */ indices,
        /* write = */ false,
        /* preserve_dirty = */ false,
        /* check_size = */ m_impl->check_size
    );
    m_impl->traverse(ctx, state, labels, default_label);
}

/// Traverse the PyTree ``state`` and write variable indices.
void VariableTracker::write(nb::handle state,
                            const dr::vector<uint64_t> &indices,
                            bool preserve_dirty,
                            const dr::vector<dr::string> &labels,
                            const char *default_label) {
    Context ctx(
        /* indices = */ const_cast<dr::vector<uint64_t>&>(indices),
        /* write = */ true,
        /* preserve_dirty = */ preserve_dirty,
        /* check_size = */ m_impl->check_size
    );
    m_impl->traverse(ctx, state, labels, default_label);
}


/// Implementation detail of ``read()`` and ``write()``
void VariableTracker::Impl::traverse(Context &ctx, nb::handle state,
                                     const dr::vector<dr::string> &labels,
                                     const char *default_label) {
    if (labels.empty()) {
        ctx.label = default_label;
        traverse(ctx, state);
    } else {
        if (!state.type().is(&PyTuple_Type))
            nb::raise("VariableTracker::traverse(): must specify state "
                      "variables using a tuple if 'labels' is provided");

        size_t state_len = nb::len(state);
        if (labels.size() != state_len)
            nb::raise("the variable state and labels have an inconsistent size (%zu vs %zu)",
                      state_len, labels.size());

        for (size_t i = 0; i < state_len; ++i) {
            ctx.label = labels[i];
            traverse(ctx, state[i]);
        }
    }

    if (ctx.index_offset != ctx.indices.size())
        nb::raise("internal error, only consumed %zu/%zu variable indices",
                  ctx.index_offset, ctx.indices.size());
}

static size_t size_valid(Variable *v, const dr::string &label, nb::handle h, size_t size) {
    if (v->size == 0)
        v->size = size;
    else if (v->size != size)
        nb::raise(
            "the size of state variable '%s' of type '%s' changed "
            "from %zu to %zu.",
            label.c_str(), nb::inst_name(h).c_str(), v->size, size);
    return size;
}

/**
 * Traverse a PyTree and either read or update the encountered Jit-compiled
 * variables. The function performs numerous checks to detect and report
 * inconsistencies with a useful error message that identifies variables by
 * name.
 */
bool VariableTracker::Impl::traverse(Context &ctx, nb::handle h) {
    StackGuard stack_guard(ctx, h);
    nb::handle tp = h.type();

    // Have we already encountered this variable before?
    bool new_variable = false;
    VariableMap::iterator it;
    std::tie(it, new_variable) = state.try_emplace(ctx.label, h);
    Variable *v = &it.value();

    // Update the Python object
    nb::object prev = v->value;
    v->value = nb::borrow(h);

    // Ensure that the variable type remains consistent
    if (!prev.type().is(tp))
        nb::raise("the type of state variable '%s' changed from '%s' to '%s', "
                  "which is not permitted",
                  ctx.label.c_str(), nb::inst_name(v->value).c_str(),
                  nb::type_name(tp).c_str());

    uint32_t changed = false;

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if ((JitBackend) s.backend == JitBackend::None)
            return false;

        if (s.is_tensor) {
            ScopedAppendLabel guard(ctx, ".array");
            changed = traverse(ctx, nb::steal(s.tensor_array(h.ptr())));
        } else if (s.ndim > 1) {
            size_t size = s.shape[0];
            if (size == DRJIT_DYNAMIC)
                size = s.len(inst_ptr(h));
            size_valid(v, ctx.label, h, size);

            for (size_t i = 0; i < size; ++i) {
                ScopedAppendLabel guard(ctx, "[", i, "]");
                changed |= traverse(ctx, nb::steal(s.item(h.ptr(), (Py_ssize_t) i)));
            }
        } else if (s.index) {
            uint64_t idx = s.index(inst_ptr(h));

            if (!idx && new_variable && ctx.write) {
                // This case arises when revisiting a state variables as part
                // of an AD traversal. This state state will have been passed
                // through drjit.detail.reset(), which creates a structural copy
                // with uninitialized Dr.Jit arrays.

                if (ctx.index_offset >= ctx.indices.size())
                    nb::raise("internal error at state variable '%s': ran "
                              "out of indices", ctx.label.c_str());

                idx = ctx.indices[ctx.index_offset];
                s.reset_index(idx, inst_ptr(h));
            }

            #if defined(DEBUG_TRACKER)
                printf("%s '%s' (%p) a%u r%u (size=%zu)\n",
                       ctx.write ? "write " : "read ",
                       ctx.label.c_str(), h.ptr(), uint32_t(idx << 32),
                       (uint32_t) idx,
                       jit_var_size((uint32_t) idx));
            #endif

            if (!idx)
                nb::raise("state variable '%s' of type '%s' is uninitialized",
                          ctx.label.c_str(), nb::inst_name(h).c_str());

            VarInfo vi = jit_set_backend((uint32_t) idx);

            if (new_variable) {
                if (!v->index_orig)
                    v->index_orig = ad_var_inc_ref(idx);
                v->index = ad_var_inc_ref(idx);
                v->size = vi.size;
            } else {
                if (vi.size != v->size && vi.size != 1 && v->size != 1 && check_size)
                    nb::raise(
                        "the size of state variable '%s' of type '%s' changed "
                        "from %zu to %zu. These sizes aren't compatible, and "
                        "such a change is therefore not permitted",
                        ctx.label.c_str(), nb::inst_name(h).c_str(), v->size,
                        vi.size);

                changed = idx != v->index;
                if (changed) {
                    uint64_t old = v->index;
                    v->index = ad_var_inc_ref(idx);
                    ad_var_dec_ref(old);
                    v->size = vi.size;
                }
            }

            if (!ctx.write) {
                if (idx != v->index) {
                    uint64_t old = v->index;
                    v->index = ad_var_inc_ref(idx);
                    ad_var_dec_ref(old);
                }
                ctx.indices.push_back(ad_var_inc_ref(idx));
                ctx.index_offset++;
                #if defined(DEBUG_TRACKER)
                    printf("-> read: a%u r%u\n", (uint32_t) (idx << 32), (uint32_t) idx);
                #endif
            } else {
                if (ctx.index_offset >= ctx.indices.size())
                    nb::raise("internal error at state variable '%s': ran "
                              "out of indices", ctx.label.c_str());

                uint64_t idx_new = ctx.indices[ctx.index_offset++];
                VarInfo vi_new = jit_set_backend((uint32_t) idx_new);

                if (vi_new.size != vi.size && vi_new.size != 1 &&
                    vi.size != 1 && check_size)
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

                if (idx != idx_new &&
                    !(ctx.preserve_dirty && jit_var_is_dirty((uint32_t) idx))) {
                    #if defined(DEBUG_TRACKER)
                        printf("-> write: a%u r%u\n", (uint32_t) (idx_new << 32), (uint32_t) idx_new);
                    #endif
                    s.reset_index(idx_new, inst_ptr(h));
                    uint64_t old = v->index;
                    v->index = ad_var_inc_ref(idx_new);
                    ad_var_dec_ref(old);
                }
            }
        }
    } else if (tp.is(&PyTuple_Type)) {
        nb::tuple t = nb::borrow<nb::tuple>(h);
        size_t size = size_valid(v, ctx.label, h, nb::len(t));
        for (size_t i = 0; i < size; ++i) {
            ScopedAppendLabel guard(ctx, "[", i, "]");
            changed |= traverse(ctx, t[i]);
        }
    } else if (tp.is(&PyList_Type)) {
        nb::list l = nb::borrow<nb::list>(h);
        size_t size = size_valid(v, ctx.label, h, nb::len(l));
        for (size_t i = 0; i < size; ++i) {
            ScopedAppendLabel guard(ctx, "[", i, "]");
            changed |= traverse(ctx, l[i]);
        }
    } else if (tp.is(&PyDict_Type)) {
        nb::dict d = nb::borrow<nb::dict>(h);
        size_valid(v, ctx.label, h, nb::len(d));
        for (nb::handle kv: d.items()) {
            ScopedAppendLabel guard(ctx, "[", nb::repr(kv[0]).c_str(), "]");
            changed |= traverse(ctx, kv[1]);
        }
    } else {
        nb::object traverse_cb = nb::getattr(
            h, ctx.write ? DR_STR(_traverse_1_cb_rw) : DR_STR(_traverse_1_cb_ro),
            nb::handle());
        nb::object dcls = nb::getattr(tp, DR_STR(__dataclass_fields__), nb::handle());

        if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            for (auto [k, v] : ds) {
                ScopedAppendLabel guard(ctx, ".", nb::str(k).c_str());
                changed |= traverse(ctx, nb::getattr(h, k));
            }
        } else if (traverse_cb.is_valid()) {
            ScopedAppendLabel guard(ctx, "._traverse_cb()");
            traverse_cb(
                nb::cast(ctx, nb::rv_policy::reference)
                    .attr(ctx.write ? DR_STR(_traverse_write) : DR_STR(_traverse_read)));
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                ScopedAppendLabel guard(ctx, ".", nb::str(k).c_str());
                changed |= traverse(ctx, nb::getattr(h, k));
            }
        } else if (strict && !new_variable && !h.is(prev)) {
            bool is_same = false;
            try {
                is_same = h.equal(prev);
            } catch (...) { }

            if (!is_same) {
                dr::string s0, s1;
                try { s0 = nb::str(prev).c_str(); } catch (...) { }
                try { s1 = nb::str(h).c_str(); } catch (...) { }
                nb::raise(
                    "the non-array state variable '%s' of type '%s' changed "
                    "from '%s' to '%s'. You can annotate the loop/conditional "
                    "with ``strict=False`` to disable this check.",
                    ctx.label.c_str(), nb::type_name(tp).c_str(), s0.c_str(),
                    s1.c_str());
            }
        }
    }

    // Detect changes performed via in-place mutation
    if (!ctx.write && changed && h.is(prev)) {
        it = state.find(ctx.label);
        if (it == state.end())
            nb::raise("VariableTracker::traverse(): internal error -- cannot "
                      "find mutated variable anymore.");
        it.value().mutated = true;
    }

    return changed;
}

uint64_t VariableTracker::Context::_traverse_write(uint64_t idx) {
    if (index_offset >= indices.size())
        nb::raise("internal error after state variable '%s': ran "
                  "out of indices", label.c_str());

    uint64_t idx_new = indices[index_offset++];

    if (!idx_new)
        nb::raise("internal error after state variable "
                  "'%s': uninitialized variable",
                  label.c_str());

    VarInfo vi = jit_set_backend(idx),
            vi_new = jit_set_backend(idx_new);

    if (vi.size != vi_new.size && vi.size != 1 && vi_new.size != 1 &&
        check_size)
        nb::raise(
            "the size of an unnamed state variable "
            "inside '%s' changed from %zu to %zu. "
            "These sizes aren't compatible, and such a "
            "change is therefore not permitted",
            label.c_str(), vi.size, vi_new.size);

    if (vi.type != vi_new.type)
        nb::raise(
            "the type of an unnamed state variable "
            "inside '%s' changed from '%s' to '%s', "
            "which is not permitted",
            label.c_str(), jit_type_name(vi.type),
            jit_type_name(vi_new.type));

    return idx_new;
}

void VariableTracker::Context::_traverse_read(uint64_t index) {
    ad_var_inc_ref(index);
    indices.push_back(index);
    index_offset++;
}

void VariableTracker::clear() {
    m_impl->state.clear();
}

void VariableTracker::verify_size(size_t size) {
    for (auto &kv : m_impl->state) {
        const Variable &v = kv.second;

        // Check if the variable was unchanged by the loop
        if (v.index == v.index_orig ||
            strcmp(jit_var_kind_name(v.index), "loop_phi") == 0)
            continue;

        size_t size_2 = jit_var_size(v.index);
        if (size != size_2 && size != 1 && size_2 != 1 && !jit_var_is_dirty(v.index))
            nb::raise("this operation processes arrays of size %zu, while "
                      "state variable '%s' has an incompatible size %zu",
                      size, kv.first.c_str(), size_2);
    }
}

nb::object VariableTracker::restore(const dr::vector<dr::string> &labels,
                                    const char *default_label) {

    dr::string label;
    if (labels.empty()) {
        label = default_label;
        return m_impl->restore(label);
    } else {
        nb::object result = nb::steal(PyTuple_New(labels.size()));
        for (size_t i = 0; i < labels.size(); ++i) {
            label = labels[i];
            NB_TUPLE_SET_ITEM(result.ptr(), i, m_impl->restore(label).release().ptr());
        }
        return result;
    }
}

nb::object VariableTracker::rebuild(const dr::vector<dr::string> &labels,
                                    const char *default_label) {

    dr::string label;
    if (labels.empty()) {
        label = default_label;
        return m_impl->rebuild(label).first;
    } else {
        nb::object result = nb::steal(PyTuple_New(labels.size()));
        for (size_t i = 0; i < labels.size(); ++i) {
            label = labels[i];
            NB_TUPLE_SET_ITEM(result.ptr(), i, m_impl->rebuild(label).first.release().ptr());
        }
        return result;
    }
}

nb::object VariableTracker::Impl::restore(dr::string &label) {
    VariableMap::iterator it = state.find(label);
    if (it == state.end())
        nb::raise("VariableTracker::restore(): could not find variable "
                  "named \"%s\"", label.c_str());

    Variable *v = &it.value();
    nb::object value = v->value_orig;
    nb::handle tp = value.type();

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if ((JitBackend) s.backend == JitBackend::None)
            return value;

        if (s.is_tensor) {
            ScopedAppendLabel guard(label, ".array");
            nb::inst_replace_copy(
                nb::steal(s.tensor_array(value.ptr())),
                restore(label));
        } else if (s.ndim > 1) {
            size_t size = size_valid(v, label, value, nb::len(value));
            for (size_t i = 0; i < size; ++i) {
                ScopedAppendLabel guard(label, "[", i, "]");
                nb::inst_replace_copy(
                    nb::steal(s.item(value.ptr(), (Py_ssize_t) i)),
                    restore(label));
            }
        } else if (s.index) {
            s.reset_index(v->index_orig, inst_ptr(value));
        }
    } else if (tp.is(&PyTuple_Type)) {
        size_valid(v, label, value, nb::len(value));
        for (size_t i = 0; i < v->size; ++i) {
            ScopedAppendLabel guard(label, "[", i, "]");
            (void) restore(label);
        }
    } else if (tp.is(&PyList_Type)) {
        nb::list l = nb::borrow<nb::list>(value);
        size_t size = size_valid(v, label, value, nb::len(l));
        for (size_t i = 0; i < size; ++i) {
            ScopedAppendLabel guard(label, "[", i, "]");
            l[i] = restore(label);
        }
    } else if (tp.is(&PyDict_Type)) {
        nb::dict d = nb::borrow<nb::dict>(value);
        size_t size = size_valid(v, label, value, nb::len(d));
        for (nb::handle k: d.keys()) {
            ScopedAppendLabel guard(label, "[", nb::repr(k).c_str(), "]");
            d[k] = restore(label);
        }
    } else {
        if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            for (auto [k, v] : ds) {
                ScopedAppendLabel guard(label, ".", nb::str(k).c_str());
                nb::setattr(value, k, restore(label));
            }
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                ScopedAppendLabel guard(label, ".", nb::str(k).c_str());
                nb::setattr(value, k, restore(label));
            }
        }
    }

    return value;
}

std::pair<nb::object, bool> VariableTracker::Impl::rebuild(dr::string &label) {
    VariableMap::iterator it = state.find(label);
    if (it == state.end())
        nb::raise("VariableTracker::rebuild(): could not find variable "
                  "named \"%s\"", label.c_str());

    Variable *v = &it.value();
    nb::object value = v->value_orig;
    nb::handle tp = value.type();
    bool new_object = false, mutate = v->mutated;

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if ((JitBackend) s.backend == JitBackend::None)
            return { value, false };

        if (s.is_tensor) {
            ScopedAppendLabel guard(label, ".array");
            auto [o, n] = rebuild(label);
            if (n) {
                if (mutate) {
                    jit_raise(
                        "VariableTracker::rebuild(): internal error involving "
                        "a tensor, this case should never arise");
                } else {
                    value = tp(o, ::shape(value));
                    new_object = true;
                }
            }
        } else if (s.ndim > 1) {
            size_t size = size_valid(v, label, value, nb::len(value));
            nb::list tmp;
            for (size_t i = 0; i < size; ++i) {
                ScopedAppendLabel guard(label, "[", i, "]");
                auto [o, n] = rebuild(label);
                tmp.append(o);
                new_object |= n;
            }

            if (new_object) {
                if (mutate) {
                    for (size_t i = 0; i < size; ++i)
                        value[i] = tmp[i];
                    new_object = false;
                } else {
                    value = tp(tmp);
                }
            }
        } else if (s.index) {
            ArrayBase *ptr = inst_ptr(value);
            if (v->index == s.index(ptr)) {
                // unchanged
            } else if (mutate) {
                s.reset_index(v->index, ptr);
            } else {
                value = inst_alloc(tp);
                s.init_index(v->index, inst_ptr(value));
                nb::inst_mark_ready(value);
                new_object = true;
            }
        }
    } else if (tp.is(&PyTuple_Type) || tp.is(&PyList_Type)) {
        size_t size = size_valid(v, label, value, nb::len(value));
        nb::list tmp;
        for (size_t i = 0; i < size; ++i) {
            ScopedAppendLabel guard(label, "[", i, "]");
            auto [o, n] = rebuild(label);
            tmp.append(o);
            new_object |= n;
        }
        if (new_object) {
            bool is_list = tp.is(&PyList_Type);
            if (mutate && is_list) {
                for (size_t i = 0; i < size; ++i)
                    value[i] = tmp[i];
                new_object = false;
            } else {
                value = is_list ? tmp : (nb::object) nb::tuple(tmp);
            }
        }
    } else if (tp.is(&PyDict_Type)) {
        nb::dict tmp, value_d = nb::borrow<nb::dict>(value);
        size_t size = size_valid(v, label, value, nb::len(value_d));
        for (nb::handle k: value_d.keys()) {
            ScopedAppendLabel guard(label, "[", nb::repr(k).c_str(), "]");
            auto [o, n] = rebuild(label);
            tmp[k] = o;
            new_object |= n;
        }
        if (new_object) {
            if (mutate) {
                value_d.update(tmp);
                new_object = false;
            } else {
                value = tmp;
            }
        }
    } else {
        if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            nb::object tmp = tp();
            for (auto [k, v] : ds) {
                ScopedAppendLabel guard(label, ".", nb::str(k).c_str());
                auto [o, n] = rebuild(label);
                nb::setattr(tmp, k, o);
                new_object |= n;
            }
            if (new_object) {
                if (mutate) {
                    for (nb::handle k : ds.keys())
                        nb::setattr(value, k, nb::getattr(tmp, k));
                    new_object = false;
                } else {
                    value = tmp;
                }
            }
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            nb::object tmp = tp();
            for (auto field : df) {
                nb::object k = field.attr(DR_STR(name));
                ScopedAppendLabel guard(label, ".", nb::str(k).c_str());
                auto [o, n] = rebuild(label);
                nb::setattr(tmp, k, o);
                new_object |= n;
            }
            if (new_object) {
                if (mutate) {
                    for (auto field : df) {
                        nb::object k = field.attr(DR_STR(name));
                        nb::setattr(value, k, nb::getattr(tmp, k));
                    }
                    new_object = false;
                } else {
                    value = tmp;
                }
            }
        } else if (!value.is(v->value)) {
            value = v->value;
            new_object = true;
        }
    }

    return { value, new_object };
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

    auto trk = nb::class_<VariableTracker>(m, "VariableTracker",
                                           doc_detail_VariableTracker)
        .def(nb::init<bool, bool>(),
             "strict"_a = true, "check_size"_a = true,
             doc_detail_VariableTracker_VariableTracker)
        .def("write",
             [](VariableTracker &vt, nb::handle state,
                const index64_vector &indices, bool preserve_dirty,
                const dr::vector<dr::string> &labels,
                const char *default_label) {
                 vt.write(state, indices, preserve_dirty, labels, default_label);
             },
             "state"_a, "indices"_a, "preserve_dirty"_a = false,
             "labels"_a = nb::tuple(),
             "default_label"_a = "state", doc_detail_VariableTracker_write)
        .def("read",
             [](VariableTracker &vt, nb::handle state,
                const dr::vector<dr::string> &labels,
                const char *default_label) {
                 index64_vector indices;
                 vt.read(state, indices, labels, default_label);
                 return indices;
             },
             "state"_a, "labels"_a = nb::tuple(), "default_label"_a = "state",
             doc_detail_VariableTracker_read)
        .def("verify_size", &VariableTracker::verify_size)
        .def("clear", &VariableTracker::clear,
             doc_detail_VariableTracker_clear)
        .def("restore", &VariableTracker::restore,
             "labels"_a = nb::tuple(),
             "default_label"_a = "state",
             doc_detail_VariableTracker_restore)
        .def("rebuild", &VariableTracker::rebuild,
             "labels"_a = nb::tuple(),
             "default_label"_a = "state",
             doc_detail_VariableTracker_rebuild);

    nb::class_<VariableTracker::Context>(trk, "Context")
        .def("_traverse_read", &VariableTracker::Context::_traverse_read)
        .def("_traverse_write", &VariableTracker::Context::_traverse_write);
}
