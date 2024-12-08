/*
    freeze.h -- Bindings for drjit.freeze()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include "drjit/autodiff.h"

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <drjit-core/hash.h>

struct FrozenFunction;

namespace detail {

using index64_vector = drjit::detail::index64_vector;

enum class LayoutFlag : uint32_t {
    SingletonArray = (1 << 0),
    Unaligned      = (1 << 1),
    GradEnabled    = (1 << 2),
    Postponed      = (1 << 3),
    Registry       = (1 << 4),
};

/// Stores information about python objects, such as their type, their number of
/// sub-elements or their field keys. This can be used to reconstruct a PyTree
/// from a flattened variable array.
struct Layout {
    /// Nanobind type of the container/variable
    nb::type_object type;
    /// Number of members in this container.
    /// Can be used to traverse the layout without knowing the type.
    uint32_t num = 0;
    /// Optional field identifiers of the container
    /// for example: keys in dictionary
    std::vector<nb::object> fields;
    /// Optional drjit type of the variable
    VarType vt = VarType::Void;
    /// Optional evaluation state of the variable
    VarState vs = VarState::Invalid;
    uint32_t flags = 0;
    /// The literal data
    uint64_t literal = 0;
    /// The index in the flat_variables array of this variable.
    /// This can be used to determine aliasing.
    uint32_t index = 0;
    /// We have to track the condition, where two variables have the same size
    /// during recording but don't when replaying.
    /// Therefore we de-duplicate the size.
    uint32_t size_index = 0;

    /// If a non drjit type is passed as function arguments or result, we simply
    /// cache it here.
    /// TODO: possibly do the same for literals?
    nb::object py_object = nb::none();

    bool operator==(const Layout &rhs) const;
};


// Additional context required when traversing the inputs
struct TraverseContext {
    /// Set of postponed ad nodes, used to mark inputs to functions.
    const tsl::robin_set<uint32_t, UInt32Hasher> *postponed = nullptr;
};

/**
 * A flattened representation of the PyTree.
 */
struct FlatVariables {

    // Index, used to iterate over the variables/layouts when constructing
    // python objects
    uint32_t layout_index = 0;

    /// The flattened and de-duplicated variable indices of the input/output to
    /// a frozen function
    std::vector<uint32_t> variables;
    /// Mapping from drjit variable index to index in flat variables
    tsl::robin_map<uint32_t, uint32_t, UInt32Hasher> index_to_slot;

    /// We have to track the condition, where two variables have the same size
    /// during recording but don't when replaying.
    /// Therefore we construct equivalence classes of sizes.
    /// This vector represents the different sizes, encountered during
    /// traversal. The algorithm used to "add" a size is the same as for adding
    /// a variable index.
    std::vector<uint32_t> sizes;
    /// Mapping from the size to its index in the ``sizes`` vector.
    tsl::robin_map<uint32_t, uint32_t, UInt32Hasher> size_to_slot;

    /// This saves information about the type, size and fields of pytree
    /// objects. The information is stored in DFS order.
    std::vector<Layout> layout;
    JitBackend backend = JitBackend::None;

    // Whether variables should be borrowed, instead of stealing them
    bool borrow = true;

    FlatVariables() {}
    FlatVariables(bool borrow) : borrow(borrow) {}

    void clear() {
        this->layout_index = 0;
        this->variables.clear();
        this->index_to_slot.clear();
        this->layout.clear();
        this->backend = JitBackend::None;
    }
    void release() {
        for (uint32_t &index : this->variables) {
            jit_var_dec_ref(index);
        }
    }

    /**
     * Adds a variable to the flattened array, deduplicating it.
     * This allows for checking for aliasing conditions, as aliasing inputs map
     * to the same flat variable index.
     */
    uint32_t add_variable_index(uint32_t variable_index);

    /**
     * This function returns an index into the ``sizes`` vector, representing an
     * equivalence class for the variable size. It uses a HashMap and vector to
     * deduplicate sizes.
     *
     * This is necessary, to catch cases, where two variables had the same size
     * when freezing a function and two different sizes when replaying.
     * In that case one kernel would be recorded, that evaluates both variables.
     * However, when replaying two kernels would have to be launched since the
     * now differently sized variables cannot be evaluated by the same kernel.
     */
    uint32_t add_size(uint32_t size);

    /**
     * Traverse the variable referenced by a jit index and add it to the flat
     * variables. An optional type python type can be supplied if it is known.
     */
    void traverse_jit_index(uint32_t index, TraverseContext &ctx,
                            nb::handle tp = nb::none());
    /**
     * Add an ad variable by it's index. Both the value and gradient are added
     * to the flattened variables. If the ad index has been marked as postponed
     * in the \c TraverseContext.postponed field, we mark the resulting layout
     * with that flag. This will cause the gradient edges to be propagated when
     * assigning to the input. The function takes an optional python-type if
     * it is known.
     */
    void traverse_ad_index(uint64_t index, TraverseContext &ctx,
                           nb::handle tp = nb::none());

    /**
     * Wrapper aground traverse_ad_index for a python handle.
     */
    void traverse_ad_var(nb::handle h, TraverseContext &ctx);

    /**
     * Traverse a c++ tree using it's `traverse_1_cb_ro` callback.
     */
    void traverse_cb(const drjit::TraversableBase *traversable,
                     TraverseContext &ctx, nb::object type = nb::none());

    /**
     * Traverses a PyTree in DFS order, and records it's layout in the
     * `layout` vector.
     *
     * When hitting a drjit primitive type, it calls the
     * `traverse_dr_var` method, which will add their indices to the
     * `flat_variables` vector. The collect method will also record metadata
     * about the drjit variable in the layout. Therefore, the layout can be used
     * as an identifier to the recording of the frozen function.
     */
    void traverse(nb::handle h, TraverseContext &ctx);

    /**
     * First traverses the PyTree, then the registry. This ensures that
     * additional data to vcalls is tracked correctly.
     */
    void traverse_with_registry(nb::handle h, TraverseContext &ctx);

    /**
     * Construct a variable, given it's layout.
     * This is the counterpart to `traverse_jit_index`.
     */
    uint32_t construct_jit_index(const Layout &layout);

    /**
     * Construct/assign the variable index given a layout.
     * This corresponds to `traverse_ad_index`>
     *
     * This function is also used for assignment to ad-variables.
     * If a `prev_index` is provided, and it is an ad-variable the gradient and
     * value of the flat variables will be applied to the ad variable,
     * preserving the ad_idnex.
     *
     * It returns an owning reference.
     */
    uint64_t construct_ad_index(const Layout &layout, uint32_t shrink = 0,
                                uint64_t prev_index = 0);

    /**
     * Construct an ad variable given it's layout.
     * This corresponds to `traverse_ad_var`
     */
    nb::object construct_ad_var(const Layout &layout, uint32_t shrink = 0);

    /**
     * This is the counterpart to the traverse method, used to construct the
     * output of a frozen function. Given a layout vector and flat_variables, it
     * re-constructs the PyTree.
     */
    nb::object construct();

    /**
     * Assigns an ad variable.
     * Corresponds to `traverse_ad_var`.
     * This uses `construct_ad_index` to either construct a new ad variable or
     * assign the value and gradient to an already existing one.
     */
    void assign_ad_var(Layout &layout, nb::handle dst);

    /**
     * Helper function, used to assign a callback variable.
     *
     * \param tmp
     *     This vector is populated with the indices to variables that have been
     *     constructed. It is required to release the references, since the
     *     references created by `construct_ad_index` are owning and they are
     *     borrowed after the callback returns.
     */
    uint64_t assign_cb_internal(uint64_t index, index64_vector &tmp);

    /**
     * Assigns variables using it's `traverse_cb_rw` callback.
     * This corresponds to `traverse_cb`.
     */
    void assign_cb(drjit::TraversableBase *traversable);

    /**
     * Assigns the flattened variables to an already existing PyTree.
     * This is used when input variables have changed.
     */
    void assign(nb::handle dst);

    /**
     * First assigns the registry and then the PyTree.
     * Corresponds to `traverse_with_registry`.
     */
    void assign_with_registry(nb::handle dst);
};

struct RecordingKey {
    std::vector<Layout> layout;
    uint32_t flags;

    RecordingKey() {}
    RecordingKey(std::vector<Layout> layout, uint32_t flags)
        : layout(std::move(layout)), flags(flags) {}

    bool operator==(const RecordingKey &rhs) const {
        return this->layout == rhs.layout && this->flags == rhs.flags;
    }
};

struct RecordingKeyHasher {
    size_t operator()(const RecordingKey &key) const;
};

struct FunctionRecording {
    Recording *recording = nullptr;
    FlatVariables out_variables;

    FunctionRecording() : out_variables(false) {}
    FunctionRecording(const FunctionRecording &)            = delete;
    FunctionRecording &operator=(const FunctionRecording &) = delete;
    FunctionRecording(FunctionRecording &&)                 = default;
    FunctionRecording &operator=(FunctionRecording &&)      = default;

    ~FunctionRecording() {
        if (this->recording) {
            jit_freeze_destroy(this->recording);
        }
        this->recording = nullptr;
    }

    void clear() {
        if (this->recording) {
            jit_freeze_destroy(this->recording);
        }
        this->recording     = nullptr;
        this->out_variables = FlatVariables(false);
    }

    /*
     * Record a function, given it's python input and flattened input.
     */
    nb::object record(nb::callable func, FrozenFunction *frozen_func,
                      nb::list input, const FlatVariables &in_variables);
    /*
     * Replays the recording.
     *
     * This constructs the output and re-assigns the input.
     */
    nb::object replay(nb::callable func, FrozenFunction *frozen_func,
                      nb::list input, const FlatVariables &in_variables);
};

using RecordingMap =
    tsl::robin_map<RecordingKey, std::unique_ptr<FunctionRecording>,
                   RecordingKeyHasher>;

} // namespace detail

struct FrozenFunction {
    nb::callable func;

    detail::RecordingMap recordings;
    detail::RecordingKey prev_key;
    uint32_t recording_counter = 0;

    FrozenFunction(nb::callable func) : func(func) {}
    ~FrozenFunction() {}

    FrozenFunction(const FrozenFunction &)            = delete;
    FrozenFunction &operator=(const FrozenFunction &) = delete;
    FrozenFunction(FrozenFunction &&)                 = default;
    FrozenFunction &operator=(FrozenFunction &&)      = default;

    uint32_t saved_recordings() { return this->recordings.size(); }

    void clear();

    nb::object operator()(nb::args args, nb::kwargs kwargs);
};

extern FrozenFunction freeze(nb::callable);
extern void export_freeze(nb::module_ &);
