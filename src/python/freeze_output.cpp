/*
    freeze_output.cpp -- Output planning, result construction and input
    write-back of frozen functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "freeze_layout.h"

/// Is one variable held by a leaf still bound to the same value?
static bool var_equal(const Layout &out, const Node &a, const uint32_t *out_bindings,
                      const Layout &in,  const Node &b, const uint32_t *in_bindings) {
    bool literal = a.has(NodeFlag::Literal);
    if (a.vt != b.vt || literal != b.has(NodeFlag::Literal))
        return false;
    if (literal)
        return memcmp(&out.literals[a.ref], &in.literals[b.ref], sizeof(Literal)) == 0;
    return out_bindings[a.ref] == in_bindings[b.ref];
}

drjit::vector<uint32_t> plan_outputs(Layout &out, const uint32_t *out_bindings,
                                     const Layout &in,
                                     const uint32_t *in_bindings) {
    ProfilerPhase profile("plan_outputs()");

    uint32_t begin = out.input_begin, end = (uint32_t) out.nodes.size(),
             n_in = (uint32_t) in.nodes.size();

    // Walk the input part of the output layout and the input layout in
    // lockstep, flagging the leaves whose variable changed. A leaf may gain or
    // lose its gradient (an AD leaf is written back as a whole, so value and
    // gradient share the flag). Everything else must match.
    uint32_t j = begin, k = 0;
    for (; j < end && k < n_in; ++j, ++k) {
        Node &a = out.nodes[j];
        const Node &b = in.nodes[k];

        std::string diff = node_difference(out, j, in, k, begin);
        if (!diff.empty()) {
            if (out.types[a.type_id].cls == TypeClass::Registry)
                nb::raise("the function registered or unregistered instances "
                          "of a class that its input refers to through "
                          "pointer arrays, which a replay cannot reproduce.");
            nb::raise("the function changed its input at %s (%s). A frozen "
                      "function may assign new arrays to its input, but it "
                      "must not change the structure of the input or the "
                      "Python values it holds, since a replay cannot "
                      "reproduce such a change.",
                      out.node_path(j).c_str(), diff.c_str());
        }

        if (!is_leaf(out, a))
            continue;

        bool grad_a = a.has(NodeFlag::GradEnabled),
             grad_b = b.has(NodeFlag::GradEnabled);
        if (a.has(NodeFlag::Postponed) || grad_a != grad_b ||
            !var_equal(out, a, out_bindings, in, b, in_bindings) ||
            (grad_a && grad_b &&
             !var_equal(out, out.grads[a.grad], out_bindings, in,
                        in.grads[b.grad], in_bindings)))
            a.set(NodeFlag::Dirty);
    }

    if (j != end || k != n_in)
        nb::raise("freeze(): internal error, the input layouts are not "
                  "isomorphic.");

    // Propagate to the ancestors
    drjit::vector<uint32_t> stack;
    for (uint32_t j = begin; j < end; ++j) {
        while (!stack.empty() && out.nodes[stack.back()].next <= j)
            stack.pop_back();
        if (out.nodes[j].has(NodeFlag::Dirty)) {
            out.nodes[j].set(NodeFlag::DirtySubtree);
            for (uint32_t a : stack)
                out.nodes[a].set(NodeFlag::DirtySubtree);
        }
        if (out.nodes[j].next > j + 1)
            stack.push_back(j);
    }

    // Assign output positions: every slot of the result, and every dirty slot
    // of the input
    out.slot_output.clear();
    out.slot_output.resize(out.slots.size(), Layout::NoOutput);
    out.n_outputs = 0;
    drjit::vector<uint32_t> outputs;

    auto assign = [&](const Node &v) {
        if (v.has(NodeFlag::Literal))
            return;
        uint32_t slot = v.ref;
        if (out.slot_output[slot] == Layout::NoOutput) {
            out.slot_output[slot] = out.n_outputs++;
            outputs.push_back(out_bindings[slot]);
        }
    };

    for (uint32_t j = 0; j < end; ++j) {
        const Node &n = out.nodes[j];
        if (!is_leaf(out, n))
            continue;
        if (j >= begin && !n.has(NodeFlag::Dirty))
            continue;
        assign(n);
        if (n.has(NodeFlag::GradEnabled))
            assign(out.grads[n.grad]);
    }

    return outputs;
}

namespace {

/// Shared leaf logic of the output constructor and the input write-back
struct OutputWalker {
    const Layout &s;
    const uint32_t *values;

    /// Cursor into ``s.nodes``
    uint32_t cur = 0;

    OutputWalker(const Layout &s, const uint32_t *values, uint32_t cur)
        : s(s), values(values), cur(cur) { }

    /// Owning reference to one variable held by a leaf. Literal and undefined
    /// variables are recreated from the layout, the others are outputs of
    /// the replay.
    uint32_t var_value(const Node &v) {
        if (v.has(NodeFlag::Literal)) {
            const Literal &l = s.literals[v.ref];
            if (l.undefined)
                return jit_var_undefined(s.backend, (VarType) v.vt, l.size);
            return jit_var_literal(s.backend, (VarType) v.vt, &l.value, l.size);
        }

        uint32_t pos = s.slot_output[v.ref];
        if (pos == Layout::NoOutput)
            jit_raise("freeze(): internal error, no output recorded for "
                      "slot %u.", v.ref);
        return jit_var_inc_ref(values[pos]);
    }

    /**
     * Owning combined index for the leaf at ``node``. ``prev`` is the current
     * index of the array that receives the value, or zero when constructing a
     * new one.
     */
    uint64_t leaf_index(uint32_t node, uint64_t prev) {
        const Node &n = s.nodes[node];
        uint32_t value = var_value(n);

        if (!n.has(NodeFlag::GradEnabled))
            return value;

        uint32_t grad = var_value(s.grads[n.grad]);
        uint32_t ad_index = (uint32_t) (prev >> 32);
        uint64_t index;
        if (ad_index) {
            index = ((uint64_t) ad_index << 32) | value;
            ad_var_inc_ref(index);
        } else {
            index = ad_var_new(value);
        }
        jit_var_dec_ref(value);

        attach_grad(index, grad);
        jit_var_dec_ref(grad);

        if (ad_index && n.has(NodeFlag::Postponed))
            ad_enqueue(drjit::ADMode::Backward, index);

        return index;
    }
};

/// Constructs the result of a frozen function
struct OutputConstructor : OutputWalker {
    /// Objects constructed so far, to resolve recursive references
    drjit::vector<nb::object> node_obj;

    OutputConstructor(const Layout &s, const uint32_t *values)
        : OutputWalker(s, values, 0) {
        node_obj.resize(s.input_begin);
    }

    nb::object visit() {
        uint32_t node = cur++;
        const Node &n = s.nodes[node];
        const TypeInfo &ti = s.types[n.type_id];
        nb::handle tp = ti.tp;

        if (n.has(NodeFlag::RecursiveRef)) {
            if (!node_obj[n.ref].is_valid())
                nb::raise("cannot construct an output that refers to an "
                          "object which is not constructible.");
            return node_obj[n.ref];
        }

        nb::object result;
        switch (ti.cls) {
            case TypeClass::Leaf: {
                uint64_t index = leaf_index(node, 0);
                result = nb::inst_alloc(tp);
                ti.supp->init_index(index, inst_ptr(result));
                nb::inst_mark_ready(result);
                ad_var_dec_ref(index);
                break;
            }

            case TypeClass::Tensor: {
                nb::object array = visit();
                const uint32_t *rec = s.shapes.data() + n.ref;
                uint32_t rank = rec[0];
                nb::tuple_builder shape(rank);
                if (rank > 0) {
                    // The first dimension follows from the size of the array
                    size_t inner = 1;
                    for (uint32_t i = 1; i < rank; ++i)
                        inner *= rec[i];
                    size_t width = nb::len(array);
                    shape.put(nb::int_(inner > 0 ? width / inner : 0));
                    for (uint32_t i = 1; i < rank; ++i)
                        shape.put(nb::int_(rec[i]));
                }
                result = tp(array, shape.commit());
                break;
            }

            case TypeClass::Nested: {
                result = nb::inst_alloc_zero(tp);
                dr::ArrayBase *p = inst_ptr(result);
                if (ti.supp->shape[0] == DRJIT_DYNAMIC)
                    ti.supp->init(n.size, p);
                for (uint32_t i = 0; i < n.size; ++i)
                    result[i] = visit();
                nb::inst_mark_ready(result);
                break;
            }

            case TypeClass::Tuple: {
                nb::tuple_builder tb(n.size);
                for (uint32_t i = 0; i < n.size; ++i)
                    tb.put(visit());
                nb::tuple t = tb.commit();
                result = tp.is(&PyTuple_Type) ? std::move(t) : tp(*t);
                break;
            }

            case TypeClass::List: {
                nb::list_builder lb(n.size);
                for (uint32_t i = 0; i < n.size; ++i)
                    lb.put(visit());
                nb::list l = lb.commit();
                result = tp.is(&PyList_Type) ? std::move(l) : tp(l);
                break;
            }

            case TypeClass::Dict: {
                result = tp.is(&PyDict_Type) ? nb::dict() : tp();
                for (uint32_t i = 0; i < n.size; ++i) {
                    nb::handle key = s.names[n.ref + i];
                    result[key] = visit();
                }
                break;
            }

            case TypeClass::Struct: {
                result = tp();
                for (uint32_t i = 0; i < ti.size; ++i) {
                    nb::handle name = s.names[ti.name_ref + i];
                    nb::setattr(result, name, visit());
                }
                break;
            }

            case TypeClass::Dataclass: {
                nb::dict kwargs;
                for (uint32_t i = 0; i < ti.size; ++i) {
                    nb::handle name = s.names[ti.name_ref + i];
                    kwargs[name] = visit();
                }
                result = tp(**kwargs);
                break;
            }

            case TypeClass::Opaque:
                result = nb::borrow(s.opaques[n.ref]);
                break;

            default:
                nb::raise("the output contains a value of type %s that "
                          "cannot be constructed when replaying.",
                          tp.is_valid() ? nb::type_name(tp).c_str()
                                        : "(C++ object)");
        }

        if (is_tracked(ti.cls))
            node_obj[node] = result;

        return result;
    }
};

// Writes the dirty leaves of the output back into an input with known layout.
struct InputUpdater : OutputWalker {
    uint32_t depth = 0;

    InputUpdater(const Layout &s, const uint32_t *values)
        : OutputWalker(s, values, s.input_begin) { }

    void assign_leaf(uint32_t node, nb::handle h, const ArraySupplement *supp_) {
        uint64_t prev  = supp_->index(inst_ptr(h));
        uint64_t index = leaf_index(node, prev);
        supp_->reset_index(index, inst_ptr(h));
        ad_var_dec_ref(index);
    }

    void visit_object(uint32_t node, drjit::TraversableBase *obj) {
        const Node &n = s.nodes[node];

        traverse_members(
            obj,
            [&](uint64_t index, const char *, const char *,
                const char *) -> uint64_t {
                uint32_t ln = cur++;
                if (s.nodes[ln].has(NodeFlag::Dirty))
                    return leaf_index(ln, index);
                return 0;
            },
            [&](drjit::TraversableBase *child, const char *) {
                visit_cpp(child);
            });

        if (nb::dict d = traversable_dict(obj); d.is_valid())
            visit(d);

        if (cur != n.next)
            jit_raise("freeze(): internal error, the members of a C++ object "
                      "changed while the input was written back.");
    }

    void visit_cpp(drjit::TraversableBase *obj) {
        recursion_guard guard(depth);
        uint32_t node = cur++;
        const Node &n = s.nodes[node];
        if (!n.has(NodeFlag::DirtySubtree)) {
            cur = n.next;
            return;
        }
        visit_object(node, obj);
    }

    /// Write the dirty leaves below ``h`` back
    void visit(nb::handle h) {
        recursion_guard guard(depth);
        uint32_t node = cur++;
        const Node &n = s.nodes[node];
        const TypeInfo &ti = s.types[n.type_id];

        if (!n.has(NodeFlag::DirtySubtree)) {
            cur = n.next;
            return;
        }

        switch (ti.cls) {
            case TypeClass::Leaf:
                assign_leaf(node, h, ti.supp);
                break;

            case TypeClass::Tensor:
                visit(nb::steal(ti.supp->tensor_array(h.ptr())));
                break;

            case TypeClass::Nested:
                for (uint32_t i = 0; i < n.size; ++i)
                    visit(nb::steal(ti.supp->item(h.ptr(), (Py_ssize_t) i)));
                break;

            case TypeClass::Tuple:
                for (uint32_t i = 0; i < n.size; ++i)
                    visit(NB_TUPLE_GET_ITEM(h.ptr(), (Py_ssize_t) i));
                break;

            case TypeClass::List:
                for (uint32_t i = 0; i < n.size; ++i)
                    visit(NB_LIST_GET_ITEM(h.ptr(), (Py_ssize_t) i));
                break;

            case TypeClass::Dict:
                for (auto [k, v] : nb::borrow<nb::dict>(h))
                    visit(v);
                break;

            case TypeClass::Struct:
            case TypeClass::Dataclass:
                for (uint32_t i = 0; i < ti.size; ++i)
                    visit(nb::getattr(h, s.names[ti.name_ref + i]));
                break;

            case TypeClass::Object:
                visit_object(node, object_ptr(h));
                break;

            default:
                nb::raise("freeze(): internal error, unexpected node while "
                          "writing the input back.");
        }
    }

    void visit_registry() {
        if (s.domains.empty())
            return;

        uint32_t node = cur++;
        const Node &n = s.nodes[node];
        if (!n.has(NodeFlag::DirtySubtree)) {
            cur = n.next;
            return;
        }

        drjit::vector<void *> pointers;
        registry_pointers(s, pointers);
        for (void *ptr : pointers)
            if (ptr)
                visit_cpp((drjit::TraversableBase *) ptr);
    }
};

} // namespace

nb::object construct_output(const Layout &out, const uint32_t *values) {
    ProfilerPhase profile("construct_output()");
    OutputConstructor c(out, values);
    return c.visit();
}

void update_input(const Layout &out, nb::handle input,
                  const uint32_t *values) {
    ProfilerPhase profile("update_input()");
    InputUpdater u(out, values);
    u.visit(input);
    u.visit_registry();
}
