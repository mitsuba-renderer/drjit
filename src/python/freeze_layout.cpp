/*
    freeze_layout.cpp -- Infrastructure to describe inputs and outputs of
    frozen functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "freeze_layout.h"
#include "base.h"
#include <cstring>
#include <tsl/robin_map.h>

// =========================================================================
//  Shared helpers
// =========================================================================

static const char *backend_name(JitBackend backend) {
    switch (backend) {
        case JitBackend::CUDA:  return "CUDA";
        case JitBackend::LLVM:  return "LLVM";
        case JitBackend::Metal: return "Metal";
        default:                return "scalar";
    }
}

/// Names of the entries of the root tuple ``(args, kwargs, closure, state)``.
/// An argument, a keyword argument or a captured variable is named directly,
/// so these labels only appear when a path stops at this level, or when it
/// leads into ``state``, whose entries have no names of their own.
static const char *root_label(uint32_t index) {
    static const char *labels[] = { "args", "kwargs", "closure", "state" };
    return index < 4 ? labels[index] : "";
}

/// The walk of ``node_path()`` is not inside a named entry of the root
static constexpr uint32_t RootEntryNone = (uint32_t) -1;

void registry_pointers(const Layout &s, drjit::vector<void *> &pointers) {
    pointers.clear();
    for (const std::string &domain : s.domains) {
        uint32_t bound =
            jit_registry_id_bound(s.variant.c_str(), domain.c_str());
        size_t offset = pointers.size();
        pointers.resize(offset + bound, nullptr);
        jit_registry_get_pointers(s.variant.c_str(), domain.c_str(),
                                  &pointers[offset]);
    }
}

void attach_grad(uint64_t index, uint32_t grad) {
    ad_clear_grad(index);
    uint32_t size = (uint32_t) jit_var_size((uint32_t) index);
    if ((VarState) jit_var_state(grad) == VarState::Literal &&
        jit_var_size(grad) != size) {
        uint32_t resized = jit_var_resize(grad, size);
        ad_accum_grad(index, resized);
        jit_var_dec_ref(resized);
    } else {
        ad_accum_grad(index, grad);
    }
}

// =========================================================================
//  Layout
// =========================================================================

Layout::Layout() {
    types.resize(3);
    types[BareLeafType].cls = TypeClass::BareLeaf;
    types[ObjectType].cls   = TypeClass::Object;
    types[RegistryType].cls = TypeClass::Registry;
}

int Layout::tp_traverse(visitproc visit, void *arg) const {
    Py_VISIT(arg_names.ptr());
    for (const TypeInfo &ti : types)
        Py_VISIT(ti.tp.ptr());
    for (const nb::object &o : names)
        Py_VISIT(o.ptr());
    for (const nb::object &o : opaques)
        Py_VISIT(o.ptr());
    return 0;
}

std::string Layout::node_path(uint32_t node) const {
    std::string result;
    uint32_t n_nodes = (uint32_t) nodes.size();

    // A node that is still open (only while the layout is being built)
    // extends to the end of the node array
    auto end_of = [&](uint32_t i) { return nodes[i].next ? nodes[i].next : n_nodes; };

    // The result and the input of an output layout form separate trees
    uint32_t cur = node >= input_begin ? input_begin : 0;
    bool root = cur == input_begin;

    // The registry node follows the root
    if (node >= end_of(cur)) {
        result = "registry";
        cur = end_of(cur);
        root = false;
    }

    // Entry of the root that the walk descended into, while its child is
    // still to be named
    uint32_t entry = RootEntryNone;

    while (cur != node) {
        const Node &n = nodes[cur];
        const TypeInfo &ti = types[n.type_id];

        // Find the child whose subtree contains the node
        uint32_t child = cur + 1, ordinal = 0;
        while (child < end_of(cur) && !(node >= child && node < end_of(child))) {
            child = end_of(child);
            ordinal++;
        }
        if (child >= end_of(cur))
            break;

        // The member of a C++ object that reported this child, if any
        const char *name = node_names[child];

        // An argument, a keyword argument or a captured variable is named
        // rather than addressed through the entry of the root that holds it
        if (root && ordinal < 3 && child != node) {
            entry = ordinal;
        } else if (root) {
            result += root_label(ordinal);
        } else if (entry != RootEntryNone) {
            if (entry == 0 && ordinal < arg_names.size())
                result += nb::str(arg_names[ordinal]).c_str();
            else if (entry == 0)
                result += "args[" + std::to_string(ordinal) + "]";
            else
                result += nb::str(names[n.ref + ordinal]).c_str();
            entry = RootEntryNone;
        } else if (*name) {
            // A member expanding into several entries (a container, a nested
            // array) reports the same name for each, hence the extra index
            uint32_t index = 0, count = 0;
            for (uint32_t i = cur + 1; i < end_of(cur); i = end_of(i)) {
                if (strcmp(node_names[i], name) != 0)
                    continue;
                if (i < child)
                    index++;
                count++;
            }
            result += "." + std::string(name);
            if (count > 1)
                result += "[" + std::to_string(index) + "]";
        } else if (ti.cls == TypeClass::Dict) {
            result += "[\"";
            result += nb::str(names[n.ref + ordinal]).c_str();
            result += "\"]";
        } else if (ti.cls == TypeClass::Struct || ti.cls == TypeClass::Dataclass) {
            result += ".";
            result += nb::str(names[ti.name_ref + ordinal]).c_str();
        } else {
            result += "[" + std::to_string(ordinal) + "]";
        }

        cur = child;
        root = false;
    }

    return result.empty() ? "the root" : result;
}

// =========================================================================
//  LayoutBuilder
// =========================================================================

namespace {

/**
 * Builds the ``Layout`` of an input or output by walking a PyTree in DFS
 * order, which is the slow path of a call (see freeze_layout.h). Its methods
 * come in three families:
 *
 * - ``visit*()`` appends one node per object and descends into its children.
 *   ``visit()`` dispatches on the ``TypeClass`` of a Python object, while
 *   ``visit_cpp()`` handles objects reached through the traversal callback of
 *   another object or through the registry.
 *
 * - ``bind_var()`` and ``bind_leaf()`` describe the variable of a leaf. They
 *   schedule it for evaluation, record literals as such, and deduplicate
 *   everything else into a slot of ``bindings``.
 *
 * - ``intern_*()`` add an entry to one of the tables that nodes refer to
 *   (Python types, C++ types) and return its index.
 *
 * Once the caller has evaluated the scheduled variables, ``finish()``
 * completes the description of the slots.
 */
struct LayoutBuilder {
    Layout &s;
    SlotBindings &bindings;

    /// Layout of the previous recording
    const Layout *prev;

    /// Make every literal opaque instead of recording its value, which is
    /// what happens when the auto-opaque feature is turned off
    bool force_all;

    /// AD leaves of the input whose gradient edges were postponed by the
    /// isolated gradient scope
    const tsl::robin_set<uint32_t, UInt32Hasher> *postponed = nullptr;

    /// Maps the identity of tracked objects (see ``node_identity()``) to the
    /// node where they were first visited
    tsl::robin_map<const void *, uint32_t, PointerHasher> visited;

    /// Maps the JIT index of an evaluated leaf to its slot
    tsl::robin_map<uint32_t, uint32_t, UInt32Hasher> slot_of_index;

    /// Maps a Python type object to its entry in the type table
    tsl::robin_map<PyObject *, uint16_t, PointerHasher> type_id_of;

    /// Leaves whose literal changed since the previous recording
    drjit::vector<uint32_t> changed_literals;

    uint32_t depth = 0;

    LayoutBuilder(Layout &s, SlotBindings &bindings, const Layout *prev,
                  bool force_all)
        : s(s), bindings(bindings), prev(prev), force_all(force_all) { }

    /// Append a node with the given type table entry and return its index
    uint32_t new_node(uint16_t type_id, const char *name = "") {
        uint32_t index = (uint32_t) s.nodes.size();
        s.nodes.emplace_back().type_id = type_id;
        s.node_names.push_back(name);
        return index;
    }

    /// Mark the end of a node's subtree
    void close_node(uint32_t node) {
        s.nodes[node].next = (uint32_t) s.nodes.size();
    }

    /// Append a dictionary key or field name to the name table
    void add_name(nb::handle name) {
        s.names.push_back(nb::borrow(name));
    }

    /**
     * Classify the type of the Python object ``h``. ``fields`` receives the
     * names of the fields declared by a ``DRJIT_STRUCT`` annotation or a
     * dataclass, if detected.
     */
    static void classify_type(nb::handle h, TypeInfo &ti,
                              drjit::vector<nb::object> &fields) {
        nb::handle tph = h.type();
        PyTypeObject *tp = (PyTypeObject *) tph.ptr();
        ti.tp = nb::borrow(tph);

        // The type flags identify subclasses of the builtin containers
        unsigned long flags = PyType_GetFlags(tp);

        if (is_builtin_scalar(tph)) {
            ti.cls = TypeClass::Opaque;
        } else if (is_drjit_type(tph)) {
            const ArraySupplement &supp_ = supp(tph);
            ti.supp = &supp_;
            if (supp_.is_tensor)
                ti.cls = TypeClass::Tensor;
            else if (supp_.ndim > 1 ||
                     (JitBackend) supp_.backend == JitBackend::None)
                ti.cls = TypeClass::Nested;
            else
                ti.cls = TypeClass::Leaf;
        } else if (flags & Py_TPFLAGS_TUPLE_SUBCLASS) {
            ti.cls = TypeClass::Tuple;
        } else if (flags & Py_TPFLAGS_LIST_SUBCLASS) {
            ti.cls = TypeClass::List;
        } else if (flags & Py_TPFLAGS_DICT_SUBCLASS) {
            ti.cls = TypeClass::Dict;
        } else if (PyType_IsSubtype(tp, (PyTypeObject *) traversable_base_type.ptr())) {
            ti.cls = TypeClass::Object;
        } else if (nb::dict ds = get_drjit_struct(tph); ds.is_valid()) {
            ti.cls = TypeClass::Struct;
            for (nb::handle k : ds.keys())
                fields.push_back(nb::borrow(k));
        } else if (nb::dict df = dataclass_field_dict(tph); df.is_valid()) {
            ti.cls = TypeClass::Dataclass;
            for (auto [k, field] : df)
                if (is_dataclass_field(field))
                    fields.push_back(nb::borrow(k));
        } else {
            raise_if_unbound_traversable(tph);
            ti.cls = TypeClass::Opaque;
        }
    }

    /// Return the index of the type of ``h`` in the type table, adding it if needed
    uint16_t intern_type(nb::handle h) {
        PyObject *tp = (PyObject *) Py_TYPE(h.ptr());
        auto [it, inserted] = type_id_of.try_emplace(tp, (uint16_t) s.types.size());
        if (!inserted)
            return it->second;

        if (s.types.size() >= 0xffff)
            nb::raise("detected >=65535 distinct Python types in the input, "
                      "which is beyond the limit.");

        TypeInfo ti;
        drjit::vector<nb::object> fields;
        classify_type(h, ti, fields);
        ti.name_ref = (uint32_t) s.names.size();
        ti.size     = (uint32_t) fields.size();
        for (nb::handle k : fields)
            add_name(k);

        s.types.push_back(std::move(ti));
        return (uint16_t) (s.types.size() - 1);
    }

    /// Return the index of a C++ type in ``cpp_types``, adding it if needed
    uint32_t intern_cpp_type(const std::type_info *type) {
        for (size_t i = 0; i < s.cpp_types.size(); ++i)
            if (same_cpp_type(s.cpp_types[i], type))
                return (uint32_t) i;
        s.cpp_types.push_back(type);
        return (uint32_t) (s.cpp_types.size() - 1);
    }

    /// Return the index of a C++ type that may be subclassed in Python
    uint16_t intern_object_type(const drjit::TraversableBase *obj) {
        nb::handle self = obj->self_py();
        if (self.is_valid() && NB_CALL(nb_inst_python_derived)(self.ptr()))
            return intern_type(self);
        return ObjectType;
    }

    /// Record the variant and domain of a class pointer array
    void add_domain(const char *variant, const char *domain) {
        if (!domain || !variant || domain[0] == '\0')
            return;

        if (s.domains.empty()) {
            s.variant = variant;
        } else if (s.variant != variant) {
            jit_raise("variant mismatch! All arguments to a frozen function "
                      "have to have the same variant. Variant %s of a previous "
                      "argument does not match variant %s of this argument.",
                      s.variant.c_str(), variant);
        }

        for (const std::string &d : s.domains)
            if (d == domain)
                return;
        s.domains.push_back(domain);
    }

    /// Did the previous recording see a different literal ``pv``? Such a literal
    /// is made opaque ("auto-opaque"). ``node`` identifies the leaf for diagnostics.
    bool literal_changed(const Node *pv, uint32_t node, const Literal &l) {
        if (!pv || !pv->has(NodeFlag::Literal))
            return false;

        const Literal &lp = prev->literals[pv->ref];
        if (l.value == lp.value && l.size == lp.size)
            return false;

        changed_literals.push_back(node);
        return true;
    }

    /// Report the literals that ``literal_changed()`` found. This runs once
    /// the layout is complete, since rendering a path walks the layout.
    void log_changed_literals() {
        if (changed_literals.empty() || !log_enabled(LogLevel::Info))
            return;
        jit_log(LogLevel::Info,
                "drjit.freeze(): the literal values below changed between "
                "calls and are made opaque, which requires a new recording. "
                "Making them opaque beforehand avoids this overhead.");
        for (uint32_t node : changed_literals)
            jit_log(LogLevel::Info, " - %s", s.node_path(node).c_str());
    }

    /**
     * Bind one JIT variable of a leaf. ``n`` is the node of the leaf for its
     * value, and its entry in ``Layout::grads`` for its gradient; ``pv``
     * describes the same variable in the previous recording, or is null.
     * Literals at forced positions are made opaque, and unevaluated variables
     * are scheduled. ``index`` is updated to the variable that ends up bound
     * to the slot, and ``forced`` is set when it differs from the input
     * variable.
     */
    void bind_var(Node &n, const Node *pv, uint32_t node, uint32_t &index,
                  bool &forced) {
        VarInfo info = jit_var_info(index);

        if (info.type == VarType::Pointer)
            jit_raise("pointer inputs are not supported!");

        if (s.backend == JitBackend::None)
            s.backend = info.backend;
        else if (s.backend != info.backend)
            jit_raise("backend mismatch error (backend of this variable %s "
                      "does not match backend of others %s)!",
                      backend_name(info.backend), backend_name(s.backend));

        n.vt = (uint8_t) info.type;

        bool force = force_all || (pv && pv->has(NodeFlag::ForceOpaque));

        if (info.state == VarState::Literal || info.state == VarState::Undefined) {
            Literal l { info.state == VarState::Literal ? info.literal : 0,
                        (uint32_t) info.size,
                        info.state == VarState::Undefined };
            if (!force && !literal_changed(pv, node, l)) {
                n.set(NodeFlag::Literal);
                n.ref = (uint32_t) s.literals.size();
                s.literals.push_back(l);
                return;
            }
            force = true;
            int rv = 0;
            index = jit_var_schedule_force(index, &rv);
            bindings.owned.push_back(index);
            forced = true;
        } else if (info.state != VarState::Evaluated) {
            jit_var_schedule(index);
        }

        if (force)
            n.set(NodeFlag::ForceOpaque);

        // Deduplicate the variable into a slot
        auto [it, inserted] = slot_of_index.try_emplace(
            index, (uint32_t) bindings.indices.size());
        if (inserted)
            bindings.indices.push_back(index);
        n.ref = it->second;
    }

    /**
     * Bind a leaf node holding a (possibly AD-attached) variable. When a
     * literal was made opaque, the function returns an owning combined index
     * that the caller must store in place of ``index``, and zero otherwise.
     */
    uint64_t bind_leaf(uint32_t node, uint64_t index) {
        uint32_t ad_index = (uint32_t) (index >> 32);
        bool grad_enabled = ad_index != 0 && ad_grad_enabled(index);

        // Description of this leaf in the previous recording, if any
        const Node *pn =
            (prev && node < prev->nodes.size()) ? &prev->nodes[node] : nullptr;

        uint32_t value = (uint32_t) index;
        bool value_forced = false;
        bind_var(s.nodes[node], pn, node, value, value_forced);

        uint32_t grad = 0;
        bool grad_forced = false;
        if (grad_enabled) {
            Node &n = s.nodes[node];
            n.set(NodeFlag::GradEnabled);
            n.grad = (uint32_t) s.grads.size();
            if (postponed && postponed->contains(ad_index))
                n.set(NodeFlag::Postponed);

            const Node *pg = nullptr;
            if (pn && pn->has(NodeFlag::GradEnabled))
                pg = &prev->grads[pn->grad];

            grad = ad_grad(index);
            bindings.owned.push_back(grad);
            bind_var(s.grads.emplace_back(), pg, node, grad, grad_forced);
        }

        close_node(node);

        if (value_forced || grad_forced)
            return make_ad_index(ad_index, value, grad_enabled, grad);
        return 0;
    }

    /// Describe the leaf held by the Python array ``h``
    void visit_leaf(uint32_t node, nb::handle h, const ArraySupplement *supp_) {
        if (supp_->is_class) {
            nb::str variant = nb::borrow<nb::str>(nb::getattr(h, "Variant")),
                    domain  = nb::borrow<nb::str>(nb::getattr(h, "Domain"));
            add_domain(variant.c_str(), domain.c_str());
        }

        uint64_t forced = bind_leaf(node, supp_->index(inst_ptr(h)));
        if (forced) {
            supp_->reset_index(forced, inst_ptr(h));
            ad_var_dec_ref(forced);
        }
    }

    /// Describe a declared field, keeping the value alive in case a property
    /// produced it
    void visit_field(nb::handle h, nb::handle name) {
        nb::object value = nb::getattr(h, name);
        visit(value);
        bindings.keep_alive.push_back(std::move(value));
    }

    /// Describe the members of a C++ object and its Python attributes
    void visit_object(uint32_t node, drjit::TraversableBase *obj) {
        s.nodes[node].ref = intern_cpp_type(cpp_type(obj));

        uint32_t size = traverse_members(
            obj,
            [&](uint64_t index, const char *name, const char *variant,
                const char *domain) -> uint64_t {
                add_domain(variant, domain);
                return bind_leaf(new_node(BareLeafType, name), index);
            },
            [&](drjit::TraversableBase *child, const char *name) {
                visit_cpp(child, name);
            });

        if (nb::dict d = traversable_dict(obj); d.is_valid()) {
            size++;
            visit(d);
        }

        s.nodes[node].size = size;
    }

    /// Describe a C++ object reached through another object or the registry
    void visit_cpp(drjit::TraversableBase *obj, const char *name = "") {
        recursion_guard guard(depth);
        uint32_t node = new_node(ObjectType, name);

        auto [it, inserted] = visited.try_emplace(obj, node);
        if (!inserted) {
            s.nodes[node].set(NodeFlag::RecursiveRef);
            s.nodes[node].ref = it->second;
            close_node(node);
            return;
        }

        s.nodes[node].type_id = intern_object_type(obj);
        visit_object(node, obj);
        close_node(node);
    }

    /// Describe the Python object ``h``
    void visit(nb::handle h) {
        recursion_guard guard(depth);

        uint16_t type_id = intern_type(h);
        uint32_t node    = new_node(type_id);
        TypeClass cls    = s.types[type_id].cls;

        if (is_tracked(cls)) {
            auto [it, inserted] = visited.try_emplace(node_identity(h, cls), node);
            if (!inserted) {
                s.nodes[node].set(NodeFlag::RecursiveRef);
                s.nodes[node].ref = it->second;
                close_node(node);
                return;
            }
        }

        switch (cls) {
            case TypeClass::Leaf:
                visit_leaf(node, h, s.types[type_id].supp);
                return;

            case TypeClass::Tensor: {
                const ArraySupplement *supp_ = s.types[type_id].supp;
                const dr::vector<size_t> &shape = supp_->tensor_shape(inst_ptr(h));
                s.nodes[node].ref = (uint32_t) s.shapes.size();
                s.shapes.push_back((uint32_t) shape.size());
                for (size_t i = 1; i < shape.size(); ++i)
                    s.shapes.push_back((uint32_t) shape[i]);
                s.nodes[node].size = 1;
                visit(nb::steal(supp_->tensor_array(h.ptr())));
                break;
            }

            case TypeClass::Nested: {
                const ArraySupplement *supp_ = s.types[type_id].supp;
                Py_ssize_t len = supp_->shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = (Py_ssize_t) supp_->len(inst_ptr(h));
                s.nodes[node].size = (uint32_t) len;
                for (Py_ssize_t i = 0; i < len; ++i)
                    visit(nb::steal(supp_->item(h.ptr(), i)));
                break;
            }

            case TypeClass::Tuple: {
                Py_ssize_t len = NB_TUPLE_GET_SIZE(h.ptr());
                s.nodes[node].size = (uint32_t) len;
                for (Py_ssize_t i = 0; i < len; ++i)
                    visit(NB_TUPLE_GET_ITEM(h.ptr(), i));
                break;
            }

            case TypeClass::List: {
                Py_ssize_t len = NB_LIST_GET_SIZE(h.ptr());
                s.nodes[node].size = (uint32_t) len;
                for (Py_ssize_t i = 0; i < len; ++i)
                    visit(NB_LIST_GET_ITEM(h.ptr(), i));
                break;
            }

            case TypeClass::Dict: {
                nb::dict dict = nb::borrow<nb::dict>(h);
                s.nodes[node].size = (uint32_t) dict.size();
                s.nodes[node].ref = (uint32_t) s.names.size();
                for (auto [k, v] : dict)
                    add_name(k);
                for (auto [k, v] : dict)
                    visit(v);
                break;
            }

            case TypeClass::Struct:
            case TypeClass::Dataclass: {
                // The type table may be reallocated by the recursive visits
                uint32_t name_ref = s.types[type_id].name_ref,
                         size     = s.types[type_id].size;
                s.nodes[node].size = size;
                for (uint32_t i = 0; i < size; ++i)
                    visit_field(h, s.names[name_ref + i]);
                break;
            }

            case TypeClass::Object:
                visit_object(node, object_ptr(h));
                break;

            case TypeClass::Opaque:
                s.nodes[node].ref = (uint32_t) s.opaques.size();
                s.opaques.push_back(nb::borrow(h));
                break;

            default:
                nb::raise("internal error, unexpected type class");
        }

        close_node(node);
    }

    /// Describe the registry entries of the domains seen so far
    void visit_registry() {
        if (s.domains.empty())
            return;

        uint32_t node = new_node(RegistryType);

        drjit::vector<void *> pointers;
        registry_pointers(s, pointers);

        uint32_t count = 0;
        for (void *ptr : pointers) {
            if (!ptr)
                continue;
            count++;
            visit_cpp((drjit::TraversableBase *) ptr);
        }

        s.nodes[node].size = count;
        close_node(node);
    }

    /// Describe the slots. Must be called after evaluation.
    void finish() {
        uint32_t n_slots = (uint32_t) bindings.indices.size();
        s.slots.resize(n_slots);

        // Numbers the distinct sizes in order of first appearance
        tsl::robin_map<uint32_t, uint32_t, UInt32Hasher> class_of_size;

        for (uint32_t slot = 0; slot < n_slots; ++slot) {
            uint32_t index = bindings.indices[slot];

            // Hold a reference to every input while the function is recorded
            // so that in-place operations (e.g. scatters) copy the input first
            bindings.owned.push_back_borrow(index);

            VarInfo info = jit_var_info(index);
            if (info.state != VarState::Evaluated)
                jit_raise("internal error, variable r%u is in an unexpected "
                          "state (%u) after evaluation.", index, (uint32_t) info.state);

            auto [it, inserted] = class_of_size.try_emplace(
                (uint32_t) info.size, (uint32_t) class_of_size.size());

            Slot &si      = s.slots[slot];
            si.vt         = (uint8_t) info.type;
            si.singleton  = info.size == 1;
            si.unaligned  = info.unaligned;
            si.size_class = it->second;
        }

        s.n_size_classes = (uint32_t) class_of_size.size();
    }
};

/// Path of the innermost open node that was interrupted by an error
static std::string open_node_path(const Layout &s) {
    for (size_t i = s.nodes.size(); i > 0; --i)
        if (s.nodes[i - 1].next == 0)
            return s.node_path((uint32_t) (i - 1));
    return "the root";
}

/// Run a traversal of ``walk`` with the builder, evaluate and finish the layout
template <typename Walk>
static void run_builder(LayoutBuilder &b, const char *what, Walk &&walk) {
    try {
        walk();
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "drjit.freeze(): error encountered while traversing the "
                       "%s at %s (see above).",
                       what, open_node_path(b.s).c_str());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.freeze(): error encountered while traversing the "
                        "%s at %s: %s",
                        what, open_node_path(b.s).c_str(), e.what());
        nb::raise_python_error();
    }

    b.log_changed_literals();

    // Evaluate the scheduled variables. This also flushes pending side effects
    // (e.g. scatters), which must not leak into or out of a recording.
    {
        state_unlock_guard guard;
        nb::gil_scoped_release guard2;
        jit_eval();
    }

    b.finish();
}

} // namespace

std::shared_ptr<Layout>
build_input_layout(nb::handle root, nb::tuple arg_names, JitBackend backend,
                   const Layout *prev, bool force_all, SlotBindings &bindings) {
    ProfilerPhase profile("build_input_layout()");
    bindings.release();
    auto layout = std::make_shared<Layout>();
    Layout &s   = *layout;
    s.jit_flags = jit_flags();
    s.backend   = backend;
    s.arg_names = std::move(arg_names);

    LayoutBuilder b(s, bindings, prev, force_all);
    run_builder(b, "input", [&] {
        b.visit(root);
        b.visit_registry();
    });

    return layout;
}

void SlotBindings::release() {
    owned.release();
    indices.clear();
    keep_alive.clear();
}

// =========================================================================
//  Key comparison and diagnostics
// =========================================================================

bool layout_equal(const Layout &a, const Layout &b) {
    if (a.jit_flags != b.jit_flags ||
        a.backend != b.backend || a.n_size_classes != b.n_size_classes ||
        a.nodes.size() != b.nodes.size() || a.types.size() != b.types.size() ||
        a.slots.size() != b.slots.size() || a.grads.size() != b.grads.size() ||
        a.literals.size() != b.literals.size() ||
        a.shapes.size() != b.shapes.size() ||
        a.names.size() != b.names.size() ||
        a.opaques.size() != b.opaques.size() || a.variant != b.variant ||
        a.domains.size() != b.domains.size() ||
        a.cpp_types.size() != b.cpp_types.size())
        return false;

    if (memcmp(a.nodes.data(), b.nodes.data(), a.nodes.size() * sizeof(Node)) != 0 ||
        memcmp(a.slots.data(), b.slots.data(), a.slots.size() * sizeof(Slot)) != 0 ||
        memcmp(a.grads.data(), b.grads.data(), a.grads.size() * sizeof(Node)) != 0 ||
        memcmp(a.literals.data(), b.literals.data(), a.literals.size() * sizeof(Literal)) != 0 ||
        memcmp(a.shapes.data(), b.shapes.data(), a.shapes.size() * sizeof(uint32_t)) != 0)
        return false;

    for (size_t i = 0; i < a.types.size(); ++i)
        if (!a.types[i].tp.is(b.types[i].tp) ||
            a.types[i].cls != b.types[i].cls)
            return false;

    for (size_t i = 0; i < a.cpp_types.size(); ++i)
        if (!same_cpp_type(a.cpp_types[i], b.cpp_types[i]))
            return false;

    for (size_t i = 0; i < a.domains.size(); ++i)
        if (a.domains[i] != b.domains[i])
            return false;

    for (size_t i = 0; i < a.names.size(); ++i)
        if (!py_equal(a.names[i], b.names[i]))
            return false;

    for (size_t i = 0; i < a.opaques.size(); ++i)
        if (!py_equal(a.opaques[i], b.opaques[i]))
            return false;

    return true;
}

static std::string py_type_name(nb::handle tp) {
    return tp.is_valid() ? nb::type_name(tp).c_str() : "(none)";
}

/// Compare the type, literal value and slot properties of one variable held by
/// a leaf, and return the reason for a difference, or ``nullptr``. Both nodes
/// must have the same flags, since ``ref`` is only interpreted based on ``va``.
static const char *var_difference(const Layout &a, const Node &va,
                                  const Layout &b, const Node &vb) {
    if (va.vt != vb.vt)
        return "the variable type changed";
    if (va.has(NodeFlag::Literal))
        return memcmp(&a.literals[va.ref], &b.literals[vb.ref],
                      sizeof(Literal)) != 0
                   ? "the literal value changed" : nullptr;
    const Slot &sa = a.slots[va.ref], &sb = b.slots[vb.ref];
    if (sa.singleton != sb.singleton || sa.size_class != sb.size_class ||
        sa.unaligned != sb.unaligned)
        return "the variable size, size class or alignment changed";
    return nullptr;
}

std::string node_difference(const Layout &a, uint32_t j, const Layout &b,
                            uint32_t k, uint32_t a_base, bool leaf_state) {
    const Node &na = a.nodes[j], &nb_ = b.nodes[k];
    const TypeInfo &ta = a.types[na.type_id], &tb = b.types[nb_.type_id];

    if (!ta.tp.is(tb.tp) || ta.cls != tb.cls)
        return "the type changed from " + py_type_name(tb.tp) + " to " +
               py_type_name(ta.tp);

    bool ref_a = na.has(NodeFlag::RecursiveRef),
         ref_b = nb_.has(NodeFlag::RecursiveRef);
    if (ref_a != ref_b || (ref_a && na.ref - a_base != nb_.ref))
        return "the object is shared with a different part of the input";
    if (ref_a)
        return "";

    switch (ta.cls) {
        case TypeClass::Leaf:
        case TypeClass::BareLeaf: {
            if (!leaf_state)
                return "";
            if (na.flags != nb_.flags)
                return "the literal/evaluated/gradient state changed";
            if (const char *r = var_difference(a, na, b, nb_))
                return r;
            if (na.has(NodeFlag::GradEnabled)) {
                const Node &ga = a.grads[na.grad], &gb = b.grads[nb_.grad];
                if (ga.flags != gb.flags)
                    return "the literal/evaluated state of a gradient changed";
                if (const char *r = var_difference(a, ga, b, gb))
                    return r;
            }
            return "";
        }

        case TypeClass::Dict:
            if (na.size == nb_.size) {
                for (uint32_t i = 0; i < na.size; ++i)
                    if (!py_equal(a.names[na.ref + i], b.names[nb_.ref + i]))
                        return "the dictionary keys changed";
            }
            break;

        case TypeClass::Tensor: {
            const uint32_t *sa = a.shapes.data() + na.ref,
                           *sb = b.shapes.data() + nb_.ref;
            if (sa[0] != sb[0] || memcmp(sa, sb, sa[0] * sizeof(uint32_t)) != 0)
                return "the tensor shape changed";
            break;
        }

        case TypeClass::Object:
            if (!same_cpp_type(a.cpp_types[na.ref], b.cpp_types[nb_.ref]))
                return std::string("the C++ type changed from ") +
                       b.cpp_types[nb_.ref]->name() + " to " +
                       a.cpp_types[na.ref]->name();
            break;

        case TypeClass::Opaque:
            if (!py_equal(a.opaques[na.ref], b.opaques[nb_.ref]))
                return "the value changed from " +
                       std::string(nb::str(b.opaques[nb_.ref]).c_str()) +
                       " to " + nb::str(a.opaques[na.ref]).c_str();
            break;

        default:
            break;
    }

    if (na.size != nb_.size)
        return "the number of entries changed from " +
               std::to_string(nb_.size) + " to " + std::to_string(na.size);

    return "";
}

std::string layout_diff(const Layout &cur, const Layout &prev) {
    if (cur.nodes.size() != prev.nodes.size())
        return "the input is structured differently (" +
               std::to_string(cur.nodes.size()) + " vs " +
               std::to_string(prev.nodes.size()) + " nodes)";

    for (uint32_t i = 0; i < (uint32_t) cur.nodes.size(); ++i) {
        std::string diff = node_difference(cur, i, prev, i, 0, true);
        if (!diff.empty())
            return "'" + cur.node_path(i) + "' (" + diff + ")";
    }

    return "";
}

// =========================================================================
//  Output layout containing both the result and potentially updated inputs
// =========================================================================

Layout build_output_layout(
    nb::handle output, nb::handle input, nb::tuple arg_names,
    JitBackend backend, const tsl::robin_set<uint32_t, UInt32Hasher> &postponed,
    SlotBindings &bindings) {
    ProfilerPhase profile("build_output_layout()");

    Layout s;
    s.jit_flags = jit_flags();
    s.backend   = backend;
    s.arg_names = std::move(arg_names);

    LayoutBuilder b(s, bindings, nullptr, false);
    b.postponed = &postponed;

    run_builder(b, "output", [&] {
        s.input_begin = (uint32_t) -1;
        b.visit(output);
        s.input_begin = (uint32_t) s.nodes.size();
        b.visited.clear();
        b.visit(input);
        b.visit_registry();
    });

    return s;
}
