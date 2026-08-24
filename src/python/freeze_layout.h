/*
    freeze_layout.h -- Infrastructure to describe inputs and outputs of frozen
    functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.

    Overview
    --------

    A ``FrozenFunction`` (see freeze.h) records the kernels launched by a
    Python callable and replays them on later calls. Replay requires the input
    to be structurally compatible with that of the original recording, meaning
    that it must have the same PyTree structure, Python types, literal values,
    and size relationships between variables. Throughout this file, the *root*
    denotes the input of a call as a whole: the tuple ``(args, kwargs,
    closure, state)`` that ``FrozenFunction::operator()`` assembles.

    This file defines the class ``Layout``, which encodes all of these
    properties, along with the operations handling the two main code paths:

    1. Fast path
    ------------

    Ideally, a frozen function repeatedly receives the same flavor of inputs.
    The ``verify_layout()`` function exploits this to avoid building a new
    ``Layout``. It walks the current input in lockstep with the nodes of the
    most recently used ``Layout``, both to validate compatibility and to
    collect the argument list needed for the subsequent kernel launches in a
    ``SlotBindings`` instance. Its working memory (``VerifierScratch``) and
    the bindings are owned by the frozen function and reused across calls,
    which minimizes dynamic memory allocation and Python reference counting
    when one recording is frequently reused.

    2. Slow path
    ------------

    Otherwise, the implementation calls ``build_input_layout()`` to build a
    new ``Layout`` from scratch. The ``LayoutBuilder`` walks the PyTree and
    appends one ``Node`` per visited object, in DFS order.

    Each node links to a ``TypeInfo`` in ``Layout::types``, which caches the
    classification of a Python type (``TypeClass``: array, tensor, nested
    array, tuple, list, dict, struct, dataclass, C++ object, opaque value),
    the ``ArraySupplement`` of Dr.Jit types, and the declared field names of
    structs and dataclasses. Types are generally used repeatedly, hence this
    classification happens only once per type object.

    The information most relevant to the backend are the "leaf" arrays tracked
    by Dr.Jit-Core. These are classified into

    - ``Leaf``: JIT-backed Dr.Jit Python array types with ``ndim==1``
    - ``BareLeaf``: leaf arrays discovered while traversing C++ classes

    Both can potentially carry gradients, which are referenced by the same node.

    The LayoutBuilder deduplicates the evaluated leaves into "slots", numbered
    in order of first appearance. It also schedules unevaluated leaves for
    a joint evaluation done at the end of the walk. Slots do not store sizes
    and are instead grouped into equivalence classes (``Slot::size_class``).
    A recording made with one set of input sizes generalizes to other sizes,
    as long the slots within each size class still agree.

    The remaining payloads live in side tables of the ``Layout``:

    - ``grads``: gradients of the leaves whose gradients are enabled
    - ``literals``: values and sizes of literal and undefined leaves, which
      are not slots and must match exactly
    - ``names``: dictionary keys and field names
    - ``shapes``: tensor shapes
    - ``opaques``: opaque Python values (anything that is not traversed)
    - ``cpp_types``: the dynamic C++ types of objects

    Finally, the layout is compared to cached recordings via ``layout_equal()``.
    This comparison is bitwise and passes over stored ``Node``, ``Slot``, and
    ``Literal`` entries, hence it is important that these data structures do
    not contain undefined padding bytes. If a match is found, it can be
    replayed using the generated ``SlotBindings``.

    Example: for the input ``(x, [y, y], obj)``, where ``x`` is an evaluated
    array, ``y`` a literal, and ``obj`` a C++ object holding one array and a
    child object that holds one array, the layout consists of these nodes:

    ```
      0  Tuple      size=3
      1    Leaf     slot 0                    x
      2    List     size=2
      3      Leaf   literal                   y
      4      Leaf   literal                   y (same value, not a slot)
      5    Object   size=2, cpp_types[0]      obj
      6      BareLeaf slot 1                  array held by obj
      7      Object size=1, cpp_types[1]      child
      8        BareLeaf slot 2                array held by child
    ```

    3. C++ objects and the registry
    -------------------------------

    C++ subclasses of ``drjit::TraversableBase`` provide a callback that makes
    their contents visible to function freezing. Such an object produces an
    ``Object`` node whose children are the members reported by the callback:
    a ``BareLeaf`` per array and an ``Object`` subtree per child object.
    ``Node::ref`` pins the polymorphic C++ type of every ``Object`` in
    ``Layout::cpp_types``, and ``Layout::node_names`` records the name of the
    member that each of them was reached through, which ``node_path()`` uses
    to refer to them.

    C++ types can further be subclassed in Python, where instances may hold
    additional Dr.Jit arrays in their attributes. The attribute dictionary of
    such an instance (see ``traversable_dict()`` in common.h) follows the C++
    members as one more ``Dict`` child, and the Python type is recorded in
    the type table. A plain nanobind wrapper contributes nothing, so binding
    a C++ object to Python does not change its description.

    Objects and mutable containers are tracked by identity (C++ or Python
    pointer). A repeat encounter becomes a ``RecursiveRef`` node that points
    to the first visit, which deduplicates shared objects and terminates
    reference cycles.

    Dr.Jit pointer arrays (e.g., ``JitArray<T*>``) refer to objects through
    the registry, and the callable may reach these objects implicitly. The
    builder therefore appends a ``Registry`` node after the root, whose
    children are the live registry entries of every domain associated with
    pointer arrays seen in the input. Each entry is described like any other
    object (or by a ``RecursiveRef``).

    4. Recording: the result and modified inputs
    --------------------------------------------

    A replay must reproduce the function result and any inputs modified in
    place (e.g., ``x += 1`` on an argument). ``build_output_layout()``
    describes both in one ``Layout`` holding two consecutive DFS trees: the
    result before ``Layout::input_begin``, and the post-call input (including
    the registry) from there on.

    ``plan_outputs()`` compares the input part of this layout against the
    input layout position by position. The function may assign new arrays to
    input leaves, but any other change (container entries, dictionary keys,
    tensor shapes, Python values, ...) cannot be replayed. In that case
    ``plan_outputs()`` throws ``InputChanged`` and the caller records the
    function once more, which handles callables that initialize part of
    their input on the first call.

    Leaves whose variable changed, and AD leaves with ``Postponed`` gradient
    edges, are flagged ``Dirty`` and their ancestors ``DirtySubtree``. The
    result slots and the dirty input leaves receive output positions
    (``slot_output``, ``n_outputs``) and become outputs of the backend
    recording. Inputs that the function only reads cost nothing on replay.

    5. Replay: constructing the result and updating the input
    ---------------------------------------------------------

    ``jit_freeze_replay()`` returns one variable per output position.
    ``construct_output()`` rebuilds the result from the output layout, and
    ``update_input()`` walks the input part in lockstep with the live input
    and assigns the dirty leaves, skipping subtrees without ``DirtySubtree``.
    Both also run once right after recording so that errors surface early.

    6. Auto-opaque literals
    -----------------------

    A literal whose value differs between calls would force a new recording
    every time. With ``auto_opaque`` enabled, the builder compares each
    literal against the previous recording's layout (``prev``) at the same
    node position and makes it opaque (``jit_var_schedule_force()``) when
    the value changed. The slot is then flagged ``ForceOpaque``. This flag
    is also inherited from ``prev``, so forced positions accumulate across
    recordings, and the verifier makes literals at such positions opaque as
    well. The opaque variable is written back into the Python array or C++
    object right away, which is safe even if the walk fails later since it
    holds the same contents as the literal.
*/

#pragma once

#include "common.h"
#include <drjit-core/hash.h>
#include <drjit-core/jit.h>
#include <drjit/autodiff.h>
#include <drjit/python.h>
#include <drjit/traversable_base.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <tsl/robin_set.h>
#include <typeinfo>

/// Classification of a Python type in a layout
enum class TypeClass : uint8_t {
    /// JIT-backed Dr.Jit array variable with ``ndim == 1``
    Leaf,

    /// JIT variable without a Python object, emitted by a C++ object's
    /// traversal callback
    BareLeaf,

    /// Dr.Jit tensor
    Tensor,

    /// Dr.Jit array with ``ndim > 1``, or one that is not backed by the JIT
    /// compiler. Its entries are described individually, which means that a
    /// scalar array is captured by value just like a Python ``float``.
    Nested,

    /// Python sequence types
    Tuple,
    List,
    Dict,

    /// Class with a ``DRJIT_STRUCT`` annotation
    Struct,

    /// A standard Python ``dataclass``
    Dataclass,

    /// A C++ object deriving from ``drjit::TraversableBase``
    Object,

    /// Any other Python object, captured by value
    Opaque,

    /// The registry pseudo-node appended after the root of the input
    Registry
};

/// Per-node flags
enum class NodeFlag : uint8_t {
    /// The node refers to an object that appeared earlier (``ref`` is the
    /// index of the node where it was first visited)
    RecursiveRef = 1 << 0,

    /// The variable is a literal or undefined (``ref`` indexes
    /// ``Layout::literals``)
    Literal      = 1 << 1,

    /// Leaf with enabled gradients, whose gradient is described by the
    /// entry of ``Layout::grads`` that ``Node::grad`` refers to
    GradEnabled  = 1 << 2,

    /// Evaluated leaf whose input may arrive as a literal that must be made
    /// opaque (see the ``auto_opaque`` feature)
    ForceOpaque  = 1 << 3,

    /// AD leaf of the input whose gradient edges were postponed by the
    /// isolated gradient scope of the frozen function. Writing it back
    /// enqueues it for backward propagation.
    Postponed    = 1 << 4,

    /// Leaf of the input that the function modified and that must be
    /// written back after a replay
    Dirty        = 1 << 5,

    /// Node whose subtree contains a dirty leaf
    DirtySubtree = 1 << 6
};

/// The Layout builder generates one Node per PyTree element in DFS order.
/// Subtrees always form a contiguous range delimited by ``Node::next``. The
/// gradient of an AD leaf is a further ``BareLeaf`` node in ``Layout::grads``
/// that only uses the fields describing a variable (``flags``, ``vt``,
/// ``ref``).
struct Node {
    /// Index into ``Layout::types``
    uint16_t type_id = 0;

    /// Combination of ``NodeFlag`` values
    uint8_t flags = 0;

    /// Variable type of leaves (see \ref VarType)
    uint8_t vt = 0;

    /// Index of the next sibling, i.e. the end of this node's subtree
    uint32_t next = 0;

    union {
        /// Number of children (leaves have none)
        uint32_t size = 0;

        /// Index of the gradient in ``Layout::grads`` (``GradEnabled`` leaves)
        uint32_t grad;
    };

    /// Payload, whose meaning depends on the node:
    ///
    /// - evaluated leaf: slot index
    /// - literal/undefined leaf: index into ``Layout::literals``
    /// - dict: offset of the keys in ``Layout::names``
    /// - tensor: offset of the shape in ``Layout::shapes``
    /// - opaque value: index into ``Layout::opaques``
    /// - object: index into ``Layout::cpp_types``
    /// - recursive reference: index of the node first describing the object
    uint32_t ref = 0;

    bool has(NodeFlag f) const { return (flags & (uint8_t) f) != 0; }
    void set(NodeFlag f) { flags |= (uint8_t) f; }
};

static_assert(sizeof(Node) == 16);

/// Information about a Python type referenced by the layout
struct TypeInfo {
    /// The Python type, invalid for the fixed entries without a Python type
    /// (``BareLeaf``, ``Registry``, and C++ objects without a Python side)
    nb::object tp;

    /// Classification of the type, which selects the code path of the walkers
    TypeClass cls = TypeClass::Opaque;

    /// Array supplement of Dr.Jit types
    const ArraySupplement *supp = nullptr;

    /// Offset of the declared field names of Struct/Dataclass types in
    /// ``Layout::names``
    uint32_t name_ref = 0;

    /// Number of declared field names
    uint32_t size = 0;
};

/// Information about a deduplicated input variable
struct Slot {
    /// Variable type (see \ref VarType)
    uint8_t vt = 0;

    /// Set when the variable has exactly one entry
    bool singleton = false;

    /// Set when the variable's storage is unaligned
    bool unaligned = false;

    /// Explicit padding, so that slots can be compared bytewise
    uint8_t unused = 0;

    /// Index of the size equivalence class
    uint32_t size_class = 0;
};

static_assert(sizeof(Slot) == 8);

/// Information about a literal or undefined leaf
struct Literal {
    /// Literal value (zero for undefined variables)
    uint64_t value = 0;

    /// Number of entries
    uint32_t size = 0;

    /// Set for undefined variables (a full word keeps the struct padding-free)
    uint32_t undefined = 0;
};

static_assert(sizeof(Literal) == 16);

/// Layout descriptor of a frozen function's input or output
struct Layout {
    /// Nodes of the PyTree in DFS order
    drjit::vector<Node> nodes;

    /// Name of the C++ member that each node was reached through, parallel to
    /// ``nodes`` and empty for everything else. The entries are the
    /// compile-time strings of \ref DR_TRAVERSE_CB and are used by
    /// ``node_path()``, hence they take no part in ``layout_equal()``.
    drjit::vector<const char *> node_names;

    /// Classification of the Python types referenced by ``nodes``
    drjit::vector<TypeInfo> types;

    /// Deduplicated list of leaf arrays
    drjit::vector<Slot> slots;

    /// Gradients of the leaves flagged ``GradEnabled``
    drjit::vector<Node> grads;

    /// Values and sizes of literal and undefined leaves
    drjit::vector<Literal> literals;

    /// Tensor shapes, each stored as the rank followed by all entries except
    /// the first (which follows from the size of the underlying array)
    drjit::vector<uint32_t> shapes;

    /// Dictionary keys and field names
    drjit::vector<nb::object> names;

    /// Opaque values captured by equality
    drjit::vector<nb::object> opaques;

    /// Dynamic C++ types of the objects in the input
    drjit::vector<const std::type_info *> cpp_types;

    /// Number of size equivalence classes (see ``Slot::size_class``)
    uint32_t n_size_classes = 0;

    /// JIT flags in effect when the recording was made
    uint32_t jit_flags = 0;

    /// Backend of the input variables
    JitBackend backend = JitBackend::None;

    /// Variant and domains of class arrays, traversed via the registry
    std::string variant;
    drjit::vector<std::string> domains;

    /// Names of the positional parameters of the frozen function, which
    /// ``node_path()`` uses to refer to an input by name
    nb::tuple arg_names;

    /// Node index of the root of the input. This is zero for an input
    /// layout. An output layout describes the function result first, and the
    /// input follows at this index.
    uint32_t input_begin = 0;

    // =====================================================================
    //  Fields used by the output layout of a recording only
    // =====================================================================

    /// Number of variables that the recording returns on replay
    uint32_t n_outputs = 0;

    /// Maps each slot to its position among the recording's outputs, or
    /// ``NoOutput`` for an unmodified input that is not returned
    drjit::vector<uint32_t> slot_output;

    /// Entry of ``slot_output`` for slots that a replay does not produce
    static constexpr uint32_t NoOutput = (uint32_t) -1;

    Layout();
    Layout(const Layout &) = delete;
    Layout &operator=(const Layout &) = delete;
    Layout(Layout &&) = default;
    Layout &operator=(Layout &&) = default;

    /// Visit the Python references held by this layout (GC support)
    int tp_traverse(visitproc visit, void *arg) const;

    /// Render a human-readable path to a node for diagnostics
    std::string node_path(uint32_t node) const;
};

/// Variables bound to the slots of a layout. Both the builder and the
/// verifier produce this structure, and recording and replay consume it.
struct SlotBindings {
    /// JIT index bound to each slot
    drjit::vector<uint32_t> indices;

    /// Owning references that keep the bindings alive until the call is
    /// done: inputs held while recording, gradients returned by ``ad_grad()``
    /// and variables created by making literals opaque
    drjit::detail::index32_vector owned;

    /// Attribute values read during the walk. A property may have produced
    /// an array that nothing else references, so they are kept alive until
    /// the bound variables have been used.
    drjit::vector<nb::object> keep_alive;

    /// AD variables of the input whose gradients are enabled (owning
    /// references), each paired with the gradient it held before the call
    drjit::vector<std::pair<uint64_t, uint32_t>> grads;

    /// Reset the gradients of the AD variables to their state before the
    /// call, which undoes the accumulation performed by an aborted recording
    void restore_grads();

    void release();
    ~SlotBindings() { release(); }
};

/// Working memory of \ref verify_layout(), reused across calls
struct VerifierScratch {
    /// Identity of the object observed at each node, used to verify
    /// recursive references
    drjit::vector<const void *> node_obj;

    /// JIT variable observed in the input for each slot. It differs from the
    /// bound variable when a literal was made opaque.
    drjit::vector<uint32_t> slot_source;

    /// Size observed for each size class, or ``Unbound``
    drjit::vector<uint32_t> class_size;

    /// Slots bound to unevaluated variables. Their state and alignment are
    /// checked after ``jit_eval()``.
    drjit::vector<uint32_t> scheduled;

    /// Registry pointers of the current call
    drjit::vector<void *> registry_ptrs;

    /// Marks slots and size classes that the walk has not bound yet
    static constexpr uint32_t Unbound = (uint32_t) -1;
};

/**
 * \brief Build the layout of a frozen function input
 *
 * This is the slow path described in the overview. It schedules and evaluates
 * unevaluated leaves, deduplicates them into slots, and makes literals opaque
 * where the auto-opaque feature asks for it.
 *
 * \param root
 *     The input PyTree, i.e., the tuple ``(args, kwargs, closure, state)``
 *
 * \param arg_names
 *     Names of the positional parameters of the function, used to refer to
 *     an input by name in diagnostics
 *
 * \param backend
 *     Backend to assume when the input contains no JIT variables
 *
 * \param prev
 *     Layout of the previous recording, or null. A literal at a position
 *     whose node is flagged ``ForceOpaque`` in ``prev``, or where ``prev``
 *     recorded a different literal, is made opaque.
 *
 * \param force_all
 *     Make every literal opaque, regardless of ``prev``
 *
 * \param bindings
 *     Released on entry, then receives the variable bound to each slot
 */
extern std::shared_ptr<Layout>
build_input_layout(nb::handle root, nb::tuple arg_names, JitBackend backend,
                   const Layout *prev, bool force_all, SlotBindings &bindings);

/// Compare two layouts as cache keys
extern bool layout_equal(const Layout &a, const Layout &b);

/// Describe the first difference between two layouts, naming the input that
/// carries it (an empty string when they agree)
extern std::string layout_diff(const Layout &cur, const Layout &prev);

/**
 * \brief Build the output layout of a recording
 *
 * The returned layout describes the function result, followed by the input as
 * it looks after the call (see section 4 of the overview). Unevaluated
 * variables are scheduled and evaluated.
 *
 * \param output
 *     The value returned by the recorded function
 *
 * \param input
 *     The input PyTree that was passed to ``build_input_layout()``
 *
 * \param backend
 *     Backend of the recording
 *
 * \param postponed
 *     AD indices whose gradient edges were postponed by the isolated gradient
 *     scope of the call. Their leaves are flagged ``Postponed``.
 *
 * \param bindings
 *     Receives the variable bound to each slot
 */
extern Layout
build_output_layout(nb::handle output, nb::handle input, nb::tuple arg_names,
                    JitBackend backend,
                    const tsl::robin_set<uint32_t, UInt32Hasher> &postponed,
                    SlotBindings &bindings);

/**
 * \brief Determine which input leaves the recorded function modified
 *
 * Compares the input part of ``out`` against ``in``, flags modified leaves
 * ``Dirty`` and their ancestors ``DirtySubtree``, and assigns output
 * positions to the slots that a replay must produce (``slot_output``,
 * ``n_outputs``). Returns the variables that the backend recording must
 * return, in output order.
 *
 * Throws ``InputChanged`` when the callable changed anything but the
 * variables of its input (see section 4 of the overview). The caller records
 * the function once more in that case, since the change is usually a
 * one-time initialization that the next call no longer performs.
 *
 * \param out
 *     Output layout of the recording, as returned by ``build_output_layout()``
 *
 * \param out_bindings
 *     Variable bound to each slot of ``out``
 *
 * \param in
 *     Input layout of the recording
 *
 * \param in_bindings
 *     Variable bound to each slot of ``in``
 */
extern drjit::vector<uint32_t>
plan_outputs(Layout &out, const uint32_t *out_bindings, const Layout &in,
             const uint32_t *in_bindings);

/// Thrown by ``plan_outputs()`` when the function changed the structure of
/// its input or the Python values it holds
struct InputChanged : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * \brief Construct the result of a frozen function from its output layout
 *
 * ``values`` holds the variables returned by the replay (or recorded), in
 * output order. The returned references are borrowed by the constructed
 * objects.
 */
extern nb::object construct_output(const Layout &out, const uint32_t *values);

/**
 * \brief Update the input PyTree with the leaves that the function modified
 *
 * Walks the input part of the output layout in lockstep with the live input
 * and assigns the variables of dirty leaves from ``values``. Clean subtrees
 * are skipped.
 */
extern void update_input(const Layout &out, nb::handle input,
                         const uint32_t *values);

/**
 * \brief Verify an input against a layout and bind its slots
 *
 * Returns ``true`` if the input matches, in which case ``bindings.indices``
 * holds the JIT index of every slot. Variables that were not yet evaluated
 * are scheduled and evaluated as part of the check. On a mismatch the
 * bindings are released and ``false`` is returned. A mismatch at the level
 * of the recording (a different recording must be looked up or made) is not
 * an error.
 */
extern bool verify_layout(const Layout &layout, nb::handle root,
                          VerifierScratch &scratch, SlotBindings &bindings);

// =====================================================================
//  Helpers shared by the walkers (freeze_layout.cpp, freeze_output.cpp,
//  freeze_verify.cpp)
// =====================================================================

/// Fixed entries of the type table for nodes without a Python type
static constexpr uint16_t BareLeafType = 0, ObjectType = 1, RegistryType = 2;

/// Objects whose identity is tracked to detect aliasing and cycles
inline bool is_tracked(TypeClass cls) {
    switch (cls) {
        case TypeClass::List:
        case TypeClass::Dict:
        case TypeClass::Struct:
        case TypeClass::Dataclass:
        case TypeClass::Object:
            return true;
        default:
            return false;
    }
}

/// Does this node describe a variable?
inline bool is_leaf(const Layout &s, const Node &n) {
    TypeClass cls = s.types[n.type_id].cls;
    return cls == TypeClass::Leaf || cls == TypeClass::BareLeaf;
}

/// Dynamic type of a C++ object
inline const std::type_info *cpp_type(const drjit::TraversableBase *obj) {
    return &typeid(*obj);
}

inline bool same_cpp_type(const std::type_info *a, const std::type_info *b) {
    return a == b || *a == *b;
}

/// C++ object wrapped by the Python object ``h`` (a ``TraversableBase`` subtype)
inline drjit::TraversableBase *object_ptr(nb::handle h) {
    drjit::TraversableBase *obj = traversable_ptr(h);
    if (!obj)
        nb::raise("the C++ object of type %s is not initialized (was the base "
                  "class constructor called?)", nb::inst_name(h).c_str());
    return obj;
}

/// Identity under which the object at a node is tracked: the C++ pointer of
/// objects (shared with the C++ reached path), the PyObject pointer otherwise
inline const void *node_identity(nb::handle h, TypeClass cls) {
    if (cls == TypeClass::Object)
        return object_ptr(h);
    return h.ptr();
}

/// Compare everything two nodes record about their objects and return the
/// reason for a difference, or an empty string. ``a_base`` is subtracted from
/// recursive references of ``a``. The state of the variable at a leaf (type,
/// literal value, gradients, size class) is only compared when
/// ``leaf_state`` is set.
extern std::string node_difference(const Layout &a, uint32_t j,
                                   const Layout &b, uint32_t k,
                                   uint32_t a_base = 0,
                                   bool leaf_state = false);

/// Collect the registry pointers of all domains referenced by a layout
extern void registry_pointers(const Layout &s, drjit::vector<void *> &pointers);

/// Run the traversal callback of a C++ object and return the number of
/// reported members. ``on_var(index, name, variant, domain)`` returns an
/// owning replacement index (kept alive until the walk returns) or zero.
template <typename Var, typename Child>
uint32_t traverse_members(drjit::TraversableBase *obj, Var &&on_var,
                          Child &&on_child) {
    struct Payload {
        Var &on_var;
        Child &on_child;
        drjit::detail::index64_vector owned;
        uint32_t count;
    } p { on_var, on_child, {}, 0 };

    for_each_member(
        obj, drjit::TraverseRole::Freeze,
        [&p](uint64_t index, const char *name, const char *variant,
             const char *domain) -> uint64_t {
            if (!index)
                return index;
            p.count++;
            uint64_t result = p.on_var(index, name, variant, domain);
            if (!result)
                return index;
            p.owned.push_back_steal(result);
            return result;
        },
        [&p](drjit::TraversableBase *child, const char *name) {
            p.count++;
            p.on_child(child, name);
        });

    return p.count;
}

/// Replace the gradient of the AD variable ``index`` by ``grad`` (resized if literal)
extern void attach_grad(uint64_t index, uint32_t grad);

/// Owning AD index combining ``ad_index`` with a new value (and gradient)
inline uint64_t make_ad_index(uint32_t ad_index, uint32_t value,
                              bool grad_enabled, uint32_t grad) {
    uint64_t index = ((uint64_t) ad_index << 32) | (uint64_t) value;
    ad_var_inc_ref(index);
    if (grad_enabled)
        attach_grad(index, grad);
    return index;
}

/// Shared recursion limit of the walkers
struct recursion_guard {
    uint32_t &depth;
    recursion_guard(uint32_t &depth) : depth(depth) {
        if (depth >= 50) {
            PyErr_SetString(PyExc_RecursionError,
                            "runaway recursion detected");
            nb::raise_python_error();
        }
        depth++;
    }
    ~recursion_guard() { depth--; }
};
