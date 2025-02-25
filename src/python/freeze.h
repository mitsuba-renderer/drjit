/*
    freeze.h -- Bindings for drjit.freeze()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include <drjit-core/hash.h>
#include <drjit-core/jit.h>
#include <drjit/autodiff.h>
#include <string>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <drjit/fwd.h>

#include "../ext/nanobind/src/buffer.h"

struct FrozenFunction;

namespace detail {

using Buffer = nanobind::detail::Buffer;

using index64_vector = drjit::detail::index64_vector;
using index32_vector = drjit::detail::index32_vector;

enum class LayoutFlag : uint32_t {
    /// Whether this variable has size 1
    SingletonArray = (1 << 0),
    /// Whether this variable is unaligned in memory
    Unaligned = (1 << 1),
    /// Whether this layout represents a literal variable
    Literal = (1 << 2),
    /// Whether this layout represents an undefined variable (they behave
    /// similarly to literals)
    Undefined = (1 << 3),
    /// Whether this variable has gradients enabled
    GradEnabled = (1 << 4),
    /// Did this variable have gradient edges attached when recording, that
    /// where postponed by the ``isolate_grad`` function?
    Postponed = (1 << 5),
    /// Does this node represent a JIT Index?
    JitIndex = (1 << 6),
    /// This layout node is a recursive reference to another node.
    RecursiveRef = (1 << 7),
};

/// Stores information about python objects, such as their type, their number of
/// sub-elements or their field keys. This can be used to reconstruct a PyTree
/// from a flattened variable array.
struct Layout {

    /// The literal data
    uint64_t literal = 0;

    /// Optional field identifiers of the container
    /// for example: keys in dictionary
    drjit::vector<nb::object> fields;

    /// Number of members in this container.
    /// Can be used to traverse the layout without knowing the type.
    uint32_t num = 0;
    /// Either store the index if this is an opaque variable or the size of
    /// the variable if this is a Literal or Undefined variable. This will be
    /// hashed as part of the key.
    union {
        /// The index in the flat_variables array of this variable.
        /// This can be used to determine aliasing.
        uint32_t index = 0;
        /// If this node is representing a literal or undefined variable, the
        /// size of is stored here instead.
        uint32_t literal_size;
    };

    /// Flags, storing information about variables and literals.
    uint32_t flags : 8; // LayoutFlag

    /// Optional drjit type of the variable
    uint32_t vt : 4; // VarType

    /// Variable index of literal. Instead of constructing a literal every time,
    /// we keep a reference to it.
    uint32_t literal_index = 0;

    /// If a non drjit type is passed as function arguments or result, we simply
    /// cache it here.
    /// TODO: possibly do the same for literals?
    nb::object py_object;

    /// Nanobind type of the container/variable
    nb::type_object type;

    bool operator==(const Layout &rhs) const;
    bool operator!=(const Layout &rhs) const { return !(*this == rhs); }

    Layout()
        : literal(0), fields(), num(0), index(0), flags(0), vt(0),
          literal_index(0), py_object(), type() {};

    Layout(const Layout &)            = delete;
    Layout &operator=(const Layout &) = delete;

    Layout(Layout &&)            = default;
    Layout &operator=(Layout &&) = default;
};

/**
 * \brief Stores information about opaque variables.
 *
 * When traversing a PyTree, literal variables are stored directly and
 * non-literal variables are first scheduled and their indices deduplicated and
 * added to the ``FlatVariables::variables`` field. After calling ``jit_eval``,
 * information about variables can be recorded using
 * ``FlatVariables::record_jit_variables``. This struct stores that information
 * per deduplicated variable.
 */
struct VarLayout {
    /// Optional drjit type of the variable
    VarType vt = VarType::Void;
    /// Optional evaluation state of the variable
    VarState vs = VarState::Invalid;
    /// Flags, storing information about variables
    uint32_t flags = 0;
    /// We have to track the condition, where two variables have the same size
    /// during recording but don't when replaying.
    /// Therefore we de-duplicate the size.
    uint32_t size_index = 0;

    VarLayout() = default;

    VarLayout(const VarLayout &)            = delete;
    VarLayout &operator=(const VarLayout &) = delete;

    VarLayout(VarLayout &&)            = default;
    VarLayout &operator=(VarLayout &&) = default;

    bool operator==(const VarLayout &rhs) const;
    bool operator!=(const VarLayout &rhs) const { return !(*this == rhs); }
};

// Additional context required when traversing the inputs
struct TraverseContext {
    /// Set of postponed ad nodes, used to mark inputs to functions.
    const tsl::robin_set<uint32_t, UInt32Hasher> *postponed = nullptr;
    tsl::robin_map<const void *, nb::object, PointerHasher> visited;
    index32_vector free_list;
    /// If this flag is set to ``true``, the PyTree will not be deduplicated
    /// during traversal. Refcycles will still be prevented, but some objects
    /// might be traversed multiple times.
    bool deduplicate_pytree = true;
    Buffer path;

    TraverseContext() : path(1024) {}
};

/**
 * \brief A flattened representation of the PyTree.
 *
 * This struct stores a flattened representation of a PyTree as well a
 * representation of it. It can therefore be used to either construct the PyTree
 * as well as assign the variables to an existing PyTree. Furthermore, this
 * struct can also be used as a key to the ``RecordingMap``, determining which
 * recording should be used given an input to a frozen function.
 * Information about the PyTree is stored in DFS Encoding. Every node of the
 * tree is represented by a ``Layout`` element in the ``layout`` vector.
 */
struct FlatVariables {

    uint32_t flags = 0;

    /// The flattened and de-duplicated variable indices of the input/output to
    /// a frozen function
    drjit::vector<uint32_t> variables;
    /// Mapping from drjit jit index to index in flat variables. Used to
    /// deduplicate jit indices.
    tsl::robin_map<uint32_t, uint32_t, UInt32Hasher> index_to_slot;

    /// We have to track the condition, where two variables have the same size
    /// during recording but don't when replaying.
    /// Therefore we construct equivalence classes of sizes.
    /// This vector represents the different sizes, encountered during
    /// traversal. The algorithm used to "add" a size is the same as for adding
    /// a variable index.
    drjit::vector<uint32_t> sizes;
    /// Mapping from the size to its index in the ``sizes`` vector. This is used
    /// to construct size equivalence classes (i.e. deduplicating sizes).
    tsl::robin_map<uint32_t, uint32_t, UInt32Hasher> size_to_slot;

    /// This saves information about the type, size and fields of pytree
    /// objects. The information is stored in DFS order.
    drjit::vector<Layout> layout;
    /// Stores information about non-literal jit variables.
    drjit::vector<VarLayout> var_layout;
    /// The collective backend for all input variables. It can be used to ensure
    /// that all variables have the same backend.
    JitBackend backend = JitBackend::None;
    /// The variant, if any, used to traverse the registry.
    std::string variant;
    /// All domains (deduplicated), encountered while traversing the PyTree and
    /// its C++ objects. This can be used to traverse the registry. We use a
    /// vector instead of a hash set, since we expect the number of domains not
    /// to exceed 100.
    drjit::vector<std::string> domains;

    // Index, used to iterate over the variables/layouts when constructing
    // python objects
    uint32_t layout_index = 0;

    uint32_t recursion_level = 0;

    struct recursion_guard {
        FlatVariables *flat_variables;
        recursion_guard(FlatVariables *flat_variables)
            : flat_variables(flat_variables) {
            if (flat_variables->recursion_level >= 50) {
                PyErr_SetString(PyExc_RecursionError,
                                "runaway recursion detected");
                nb::raise_python_error();
            }
            // NOTE: the recursion_level has to be incremented after potentially
            // throwing an exception, as throwing an exception in the
            // constructor prevents the destructor from being called.
            flat_variables->recursion_level++;
        }
        ~recursion_guard() { flat_variables->recursion_level--; }
    };

    /**
     * Describes how many elements have to be pre-allocated for the ``layout``,
     * ``index_to_slot`` and ``size_to_slot`` containers.
     */
    struct Heuristic {
        size_t layout        = 0;
        size_t index_to_slot = 0;
        size_t size_to_slot  = 0;

        Heuristic max(Heuristic rhs) {
            return Heuristic{
                std::max(layout, rhs.layout),
                std::max(index_to_slot, rhs.index_to_slot),
                std::max(size_to_slot, rhs.size_to_slot),
            };
        }
    };

    FlatVariables() {}
    FlatVariables(Heuristic heuristic) {
        layout.reserve(heuristic.layout);
        index_to_slot.reserve(heuristic.index_to_slot);
        size_to_slot.reserve(heuristic.size_to_slot);
    }

    FlatVariables(const FlatVariables &)            = delete;
    FlatVariables &operator=(const FlatVariables &) = delete;

    FlatVariables(FlatVariables &&)            = default;
    FlatVariables &operator=(FlatVariables &&) = default;

    ~FlatVariables();

    void clear() {
        this->layout_index = 0;
        this->variables.clear();
        this->index_to_slot.clear();
        this->layout.clear();
        this->backend = JitBackend::None;
    }
    /// Borrow all variables held by this struct.
    void borrow();
    /// Release all variables held by this struct.
    void release();

    /**
     * Generates a mask of variables that should be made opaque in the next
     * iteration. This should only be called if \c compatible_auto_opaque
     * returns true for the corresponding \c FlatVariables pair.
     *
     * Returns true if new variables have been discovered that should be made
     * opaque, otherwise returns false.
     */
    bool fill_opaque_mask(FlatVariables &prev,
                          drjit::vector<bool> &opaque_mask);

    /**
     * Schedule variables that have been collected when traversing the PyTree.
     *
     * This function iterates over all ``Layout`` nodes that represent JIT
     * indices and either calls ``jit_var_schedule`` or
     * ``jit_var_schedule_force`` on them, depending on whether
     * ``schedule_force`` is true or the boolean in the ``opaque_mask``
     * corresponding to that variable is true.
     *
     * \param schedule_force
     *     Overrides the use of \c opaque_mask and makes all variables opaque
     *
     * \param opaque_mask
     *     A pointer to a compatible boolean array, indicating if some of the
     *     variables should be made opaque. Can be \c nullptr, in which case it
     *     will be ignored.
     */
    void schedule_jit_variables(bool schedule_force,
                                const drjit::vector<bool> *opaque_mask);

    /**
     * \brief Records information about jit variables, that have been traversed.
     *
     * After traversing the PyTree, collecting non-literal indices in
     * ``variables`` and evaluating the collected indices, we can collect
     * information about the underlying variables. This information is used in
     * the key of the ``RecordingMap`` to determine which recording should be
     * replayed or if the function has to be re-traced. This function iterates
     * over the collected indices and collects that information.
     */
    void record_jit_variables();

    /**
     * Returns a struct representing heuristics to pre-allocate memory for the
     * layout, of the flat variables.
     */
    Heuristic heuristic() {
        return Heuristic{
            layout.size(),
            index_to_slot.size(),
            size_to_slot.size(),
        };
    };

    /**
     * \brief Add a variant domain pair to be traversed using the registry.
     *
     * When traversing a jit variable, that references a pointer to a class,
     * such as a BSDF or Shape in Mitsuba, we have to traverse all objects
     * registered with that variant-domain pair in the registry. This function
     * adds the variant-domain pair, deduplicating the domain. Whether a
     * variable references a class is represented by it's ``IsClass`` const
     * attribute. If the domain is an empty string (""), this function skips
     * adding the variant-domain pair.
     */
    void add_domain(const char *variant, const char *domain);

    /**
     * Adds a jit index to the flattened array, deduplicating it.
     * This allows to check for aliasing conditions, where two variables
     * actually refer to the same index. The function should only be called for
     * scheduled non-literal variable indices.
     */
    uint32_t add_jit_index(uint32_t variable_index);

    /**
     * This function returns an index into the ``sizes`` vector, representing an
     * equivalence class of variable sizes. It uses a HashMap and vector to
     * deduplicate sizes.
     *
     * This is necessary, to catch cases, where two variables had the same size
     * when recording a function and two different sizes when replaying.
     * In that case one kernel would be recorded, that evaluates both variables.
     * However, when replaying two kernels would have to be launched since the
     * now differently sized variables cannot be evaluated by the same kernel.
     */
    uint32_t add_size(uint32_t size);

    /**
     * Traverse the variable referenced by a jit index and add it to the flat
     * variables. An optional type python type can be supplied if it is known.
     * Depending on the ``TraverseContext::schedule_force`` the underlying
     * variable is either scheduled (``jit_var_schedule``) or force scheduled
     * (``jit_var_schedule_force``). If the variable after evaluation is a
     * literal, it is directly recorded in the ``layout`` otherwise, it is added
     * to the ``variables`` array, allowing the variables to be used when
     * recording the frozen function.
     */
    void traverse_jit_index(uint32_t index, TraverseContext &ctx,
                            nb::handle tp = nullptr);
    /**
     * Add an ad variable by it's index. Both the value and gradient are added
     * to the flattened variables. If the ad index has been marked as postponed
     * in the \c TraverseContext.postponed field, we mark the resulting layout
     * with that flag. This will cause the gradient edges to be propagated when
     * assigning to the input. The function takes an optional python-type if
     * it is known.
     */
    void traverse_ad_index(uint64_t index, TraverseContext &ctx,
                           nb::handle tp = nullptr);

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
     *
     * Optionally, the index of a variable can be provided that will be
     * overwritten with the result of this function. In that case, the function
     * will check for compatible variable types.
     */
    uint32_t construct_jit_index(uint32_t prev_index = 0);

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
    uint64_t construct_ad_index(uint32_t shrink = 0, uint64_t prev_index = 0);

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
    void assign(nb::handle dst, TraverseContext &ctx);

    /**
     * First assigns the registry and then the PyTree.
     * Corresponds to `traverse_with_registry`.
     */
    void assign_with_registry(nb::handle dst, TraverseContext &ctx);

    bool operator==(const FlatVariables &rhs) const {
        return this->layout == rhs.layout &&
               this->var_layout == rhs.var_layout && this->flags == rhs.flags;
    }
};

/// Helper struct to hash input variables
struct FlatVariablesHasher {
    size_t operator()(const std::shared_ptr<FlatVariables> &key) const;
};

/// Helper struct to compare input variables
struct FlatVariablesEqual {
    using is_transparent = void;
    bool operator()(const std::shared_ptr<FlatVariables> &lhs,
                    const std::shared_ptr<FlatVariables> &rhs) const {
        return *lhs.get() == *rhs.get();
    }
};

/**
 * \brief A recording of a frozen function, recorded with a certain layout of
 * input variables.
 */
struct FunctionRecording {
    uint32_t last_used   = 0;
    Recording *recording = nullptr;
    FlatVariables out_variables;

    FunctionRecording() : out_variables() {}
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
        this->out_variables = FlatVariables();
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

using RecordingMap = tsl::robin_map<std::shared_ptr<FlatVariables>,
                                    std::unique_ptr<FunctionRecording>,
                                    FlatVariablesHasher, FlatVariablesEqual>;

} // namespace detail

struct FrozenFunction {
    nb::callable func;

    detail::RecordingMap recordings;
    std::shared_ptr<detail::FlatVariables> prev_key;
    drjit::vector<bool> opaque_mask;

    uint32_t recording_counter    = 0;
    uint32_t call_counter         = 0;
    int max_cache_size            = -1;
    uint32_t warn_recording_count = 10;
    JitBackend default_backend    = JitBackend::None;
    bool auto_opaque              = true;

    detail::FlatVariables::Heuristic in_heuristics;

    FrozenFunction(nb::callable func, int max_cache_size = -1,
                   uint32_t warn_recording_count = 10,
                   JitBackend backend            = JitBackend::None,
                   bool auto_opaque              = false)
        : func(func), max_cache_size(max_cache_size),
          warn_recording_count(warn_recording_count), default_backend(backend),
          auto_opaque(auto_opaque) {}
    ~FrozenFunction() {}

    FrozenFunction(const FrozenFunction &)            = delete;
    FrozenFunction &operator=(const FrozenFunction &) = delete;
    FrozenFunction(FrozenFunction &&)                 = default;
    FrozenFunction &operator=(FrozenFunction &&)      = default;

    uint32_t n_cached_recordings() { return this->recordings.size(); }

    void clear();

    nb::object operator()(nb::args args, nb::kwargs kwargs);
};

extern void export_freeze(nb::module_ &);
