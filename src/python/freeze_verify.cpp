/*
    freeze_verify.cpp -- quickly check an input for compatibility

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "freeze_layout.h"

namespace {

/// Thrown by the verifier when the input does not match the layout. The
/// reason is phrased so that it can be quoted in a sentence, and ``node`` is
/// ``NoNode`` when the difference concerns the input as a whole.
struct Mismatch {
    uint32_t node;
    const char *reason;

    static constexpr uint32_t NoNode = (uint32_t) -1;
};

[[noreturn]] NB_NOINLINE static void mismatch(uint32_t node, const char *reason) {
    throw Mismatch { node, reason };
}

/**
 * Walks a PyTree in lockstep with the nodes of an existing ``Layout``, which
 * is the fast path of a call (see freeze_layout.h). The ``visit*()`` methods
 * mirror those of the ``LayoutBuilder``: rather than appending a node, each
 * one compares an object against the node at the cursor ``cur`` and raises
 * ``Mismatch`` when they disagree. ``bind_var()`` and ``bind_leaf()`` likewise
 * check the variable of a leaf and bind it to the slot that the node records,
 * so that a successful walk leaves the arguments of the replay in
 * ``bindings``.
 */
struct LayoutVerifier {
    const Layout &s;
    VerifierScratch &scratch;
    SlotBindings &bindings;

    /// Cursor into ``s.nodes`` that advances in lockstep with the input
    uint32_t cur = 0;

    LayoutVerifier(const Layout &s, VerifierScratch &scratch,
                   SlotBindings &bindings)
        : s(s), scratch(scratch), bindings(bindings) { }

    /// Check the leafe node ``n`` against variable index ``index`` and bind its
    /// slot. Opaquing replaces ``index`` and sets ``forced=true``.
    void bind_var(const Node &n, uint32_t node, uint32_t &index, bool &forced) {
        VarInfo info = jit_var_info(index);

        if ((uint8_t) info.type != n.vt)
            mismatch(node, "the variable type changed");
        if (info.backend != s.backend)
            mismatch(node, "the backend changed");

        if (n.has(NodeFlag::Literal)) {
            const Literal &l = s.literals[n.ref];
            bool undefined = info.state == VarState::Undefined;
            if (info.size != l.size || (uint32_t) undefined != l.undefined ||
                (!undefined && (info.state != VarState::Literal ||
                                info.literal != l.value)))
                mismatch(node, "the literal value changed");
            return;
        }

        uint32_t source = index;
        uint32_t slot = n.ref;
        uint32_t &bound = bindings.indices[slot];
        if (bound != VerifierScratch::Unbound) {
            // The input variable must be the one seen at the first occurrence
            if (scratch.slot_source[slot] != source)
                mismatch(node,
                         "two inputs no longer refer to the same variable");
            // The first occurrence may have been made opaque
            if (bound != source) {
                index  = bound;
                forced = true;
            }
            return;
        }

        switch (info.state) {
            case VarState::Evaluated:
                break;

            case VarState::Literal:
            case VarState::Undefined: {
                if (!n.has(NodeFlag::ForceOpaque))
                    mismatch(node, "a literal arrived where an evaluated "
                                   "variable was recorded");
                int rv = 0;
                uint32_t opaque = jit_var_schedule_force(index, &rv);
                bindings.owned.push_back(opaque);
                index  = opaque;
                forced = true;
                info   = jit_var_info(index);
                break;
            }

            default:
                jit_var_schedule(index);
                break;
        }

        const Slot &si = s.slots[slot];
        if ((info.size == 1) != si.singleton)
            mismatch(node, "the variable size changed to or from 1");

        uint32_t &class_size = scratch.class_size[si.size_class];
        if (class_size == VerifierScratch::Unbound)
            class_size = (uint32_t) info.size;
        else if (class_size != info.size)
            mismatch(node, "the variable sizes are related differently");

        if (info.state == VarState::Evaluated) {
            if (info.unaligned != si.unaligned)
                mismatch(node, "the variable alignment changed");
        } else {
            scratch.scheduled.push_back(slot);
        }

        bound = index;
        scratch.slot_source[slot] = source;
    }

    /**
     * Verify a leaf holding a (possibly AD-attached) variable and bind its
     * slot. When a literal was made opaque, the function returns an owning
     * combined index that the caller must store in place of ``index``, and
     * zero otherwise.
     */
    uint64_t bind_leaf(uint32_t node, uint64_t index) {
        const Node &n = s.nodes[node];
        uint32_t ad_index = (uint32_t) (index >> 32);
        bool grad_enabled = ad_index != 0 && ad_grad_enabled(index);

        if (grad_enabled != n.has(NodeFlag::GradEnabled))
            mismatch(node, "the gradient state changed");

        uint32_t value = (uint32_t) index;
        bool value_forced = false;
        bind_var(n, node, value, value_forced);

        uint32_t grad = 0;
        bool grad_forced = false;
        if (grad_enabled) {
            grad = ad_grad(index);
            bindings.owned.push_back(grad);
            bind_var(s.grads[n.grad], node, grad, grad_forced);
        }

        if (value_forced || grad_forced)
            return make_ad_index(ad_index, value, grad_enabled, grad);
        return 0;
    }

    /// Verify the leaf held by the Python array ``h``
    void visit_leaf(uint32_t node, nb::handle h, const ArraySupplement *supp_) {
        uint64_t forced = bind_leaf(node, supp_->index(inst_ptr(h)));
        if (forced) {
            supp_->reset_index(forced, inst_ptr(h));
            ad_var_dec_ref(forced);
        }
    }

    /// Verify a declared field, keeping the value alive in case a property
    /// produced it
    void visit_field(uint32_t node, nb::handle h, nb::handle name) {
        nb::object value = nb::getattr(h, name, nb::handle());
        if (!value.is_valid())
            mismatch(node, "an attribute is missing");
        visit(value);
        bindings.keep_alive.push_back(std::move(value));
    }

    /// Verify the members and Python attributes of a C++ object against its
    /// node. The caller has checked the type table entry.
    void visit_object(uint32_t node, drjit::TraversableBase *obj) {
        const Node &n = s.nodes[node];

        if (!same_cpp_type(s.cpp_types[n.ref], cpp_type(obj)))
            mismatch(node, "the C++ type changed");

        traverse_members(
            obj,
            [&](uint64_t index, const char *, const char *,
                const char *) -> uint64_t {
                uint32_t ln = cur++;
                if (ln >= n.next || s.nodes[ln].type_id != BareLeafType)
                    mismatch(node, "the number of members changed");
                return bind_leaf(ln, index);
            },
            [&](drjit::TraversableBase *child, const char *) {
                if (cur >= n.next)
                    mismatch(node, "the number of members changed");
                visit_cpp(child);
            });

        if (nb::dict d = traversable_dict(obj); d.is_valid()) {
            if (cur >= n.next)
                mismatch(node, "the number of members changed");
            visit(d);
        }

        if (cur != n.next)
            mismatch(node, "the number of members changed");
    }

    /// Verify a C++ object reached through another object or the registry
    void visit_cpp(drjit::TraversableBase *obj) {
        uint32_t node = cur++;
        const Node &n = s.nodes[node];
        const TypeInfo &ti = s.types[n.type_id];

        if (ti.cls != TypeClass::Object)
            mismatch(node, "a C++ object was expected here");

        if (n.has(NodeFlag::RecursiveRef)) {
            if (scratch.node_obj[n.ref] != obj)
                mismatch(node,
                         "the object is shared with a different part of the "
                         "input");
            return;
        }
        scratch.node_obj[node] = obj;

        // A recorded Python type means that the counterpart was a Python
        // subclass instance; a plain wrapper may come and go
        nb::handle self = obj->self_py();
        if (ti.tp.is_valid() &&
            (!self.is_valid() || (PyObject *) Py_TYPE(self.ptr()) != ti.tp.ptr()))
            mismatch(node, "the Python type of the object changed");

        visit_object(node, obj);
    }

    /// Verify the Python object ``h`` against the node at the cursor
    void visit(nb::handle h) {
        uint32_t node = cur++;
        const Node &n = s.nodes[node];
        const TypeInfo &ti = s.types[n.type_id];

        if ((PyObject *) Py_TYPE(h.ptr()) != ti.tp.ptr())
            mismatch(node, "the Python type changed");

        if (n.has(NodeFlag::RecursiveRef)) {
            if (scratch.node_obj[n.ref] != node_identity(h, ti.cls))
                mismatch(node,
                         "the object is shared with a different part of the "
                         "input");
            return;
        }

        if (is_tracked(ti.cls))
            scratch.node_obj[node] = node_identity(h, ti.cls);

        switch (ti.cls) {
            case TypeClass::Leaf:
                visit_leaf(node, h, ti.supp);
                break;

            case TypeClass::Tensor: {
                const dr::vector<size_t> &shape = ti.supp->tensor_shape(inst_ptr(h));
                const uint32_t *rec = s.shapes.data() + n.ref;
                if (rec[0] != (uint32_t) shape.size())
                    mismatch(node, "the tensor shape changed");
                for (uint32_t i = 1; i < rec[0]; ++i)
                    if (rec[i] != shape[i])
                        mismatch(node, "the tensor shape changed");
                visit(nb::steal(ti.supp->tensor_array(h.ptr())));
                break;
            }

            case TypeClass::Nested: {
                Py_ssize_t len = ti.supp->shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = (Py_ssize_t) ti.supp->len(inst_ptr(h));
                if ((uint32_t) len != n.size)
                    mismatch(node, "the number of entries changed");
                for (Py_ssize_t i = 0; i < len; ++i)
                    visit(nb::steal(ti.supp->item(h.ptr(), i)));
                break;
            }

            case TypeClass::Tuple: {
                Py_ssize_t len = NB_TUPLE_GET_SIZE(h.ptr());
                if ((uint32_t) len != n.size)
                    mismatch(node, "the number of entries changed");
                for (Py_ssize_t i = 0; i < len; ++i)
                    visit(NB_TUPLE_GET_ITEM(h.ptr(), i));
                break;
            }

            case TypeClass::List: {
                Py_ssize_t len = NB_LIST_GET_SIZE(h.ptr());
                if ((uint32_t) len != n.size)
                    mismatch(node, "the number of entries changed");
                for (Py_ssize_t i = 0; i < len; ++i)
                    visit(NB_LIST_GET_ITEM(h.ptr(), i));
                break;
            }

            case TypeClass::Dict: {
                nb::dict dict = nb::borrow<nb::dict>(h);
                if ((uint32_t) dict.size() != n.size)
                    mismatch(node, "the number of entries changed");
                uint32_t i = 0;
                for (auto [k, v] : dict) {
                    if (!py_equal(k, s.names[n.ref + i++]))
                        mismatch(node, "the dictionary keys changed");
                    visit(v);
                }
                break;
            }

            case TypeClass::Struct:
            case TypeClass::Dataclass:
                for (uint32_t i = 0; i < ti.size; ++i)
                    visit_field(node, h, s.names[ti.name_ref + i]);
                break;

            case TypeClass::Object:
                visit_object(node, object_ptr(h));
                break;

            case TypeClass::Opaque:
                if (!py_equal(h, s.opaques[n.ref]))
                    mismatch(node, "the value changed");
                break;

            default:
                mismatch(node, "the recorded node type is not supported");
        }
    }

    /// Verify the registry entries of the recorded domains
    void visit_registry() {
        if (s.domains.empty())
            return;

        uint32_t node = cur++;
        const Node &n = s.nodes[node];

        drjit::vector<void *> &pointers = scratch.registry_ptrs;
        registry_pointers(s, pointers);

        uint32_t count = 0;
        for (void *ptr : pointers) {
            if (!ptr)
                continue;
            if (++count > n.size)
                mismatch(node, "the number of registry entries changed");
            visit_cpp((drjit::TraversableBase *) ptr);
        }

        if (count != n.size)
            mismatch(node, "the number of registry entries changed");
    }
};

} // namespace

bool verify_layout(const Layout &s, nb::handle root,
                   VerifierScratch &scratch, SlotBindings &bindings) {
    ProfilerPhase profile("verify_layout()");
    try {
        if (s.jit_flags != jit_flags())
            mismatch(Mismatch::NoNode, "the JIT flags changed");

        scratch.node_obj.resize(s.nodes.size());
        scratch.slot_source.resize(s.slots.size());
        scratch.class_size.clear();
        scratch.class_size.resize(s.n_size_classes, VerifierScratch::Unbound);
        scratch.scheduled.clear();
        bindings.indices.clear();
        bindings.indices.resize(s.slots.size(), VerifierScratch::Unbound);

        LayoutVerifier v(s, scratch, bindings);
        v.visit(root);
        v.visit_registry();
        if (v.cur != s.nodes.size())
            mismatch(v.cur, "the input has fewer entries than the recording");

        // Flush queued evaluations and pending side effects before replay
        {
            state_unlock_guard guard;
            nb::gil_scoped_release guard2;
            jit_eval();
        }

        for (uint32_t slot : scratch.scheduled) {
            VarInfo info = jit_var_info(bindings.indices[slot]);
            if (info.state != VarState::Evaluated ||
                info.unaligned != s.slots[slot].unaligned)
                mismatch(Mismatch::NoNode,
                         "a variable is in an unexpected state after "
                         "evaluation");
        }
    } catch (const Mismatch &m) {
        if (log_enabled(LogLevel::Info)) {
            if (m.node == Mismatch::NoNode)
                jit_log(LogLevel::Info,
                        "drjit.freeze(): the input is incompatible with the "
                        "most recently used recording (%s).", m.reason);
            else
                jit_log(LogLevel::Info,
                        "drjit.freeze(): the input is incompatible with the "
                        "most recently used recording ('%s': %s).",
                        s.node_path(m.node).c_str(), m.reason);
        }
        bindings.release();
        return false;
    }

    return true;
}
