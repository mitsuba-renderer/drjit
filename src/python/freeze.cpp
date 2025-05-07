#include "freeze.h"
#include "apply.h"
#include "autodiff.h"
#include "base.h"
#include "common.h"
#include "reduce.h"
#include "listobject.h"
#include "object.h"
#include "pyerrors.h"
#include "shape.h"
#include "tupleobject.h"
#include <bitset>
#include <drjit-core/hash.h>
#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/autodiff.h>
#include <drjit/extra.h>
#include <drjit/fwd.h>
#include <drjit/traversable_base.h>
#include <ios>
#include <nanobind/nanobind.h>
#include <ostream>
#include <sstream>
#include <vector>

/**
 * \brief Helper struct to profile and log frozen functions.
 */
struct ProfilerPhase {
    std::string m_message;
    ProfilerPhase(const char *message) : m_message(message) {
        jit_log(LogLevel::Debug, "profiler start: %s", message);
#if defined(DRJIT_ENABLE_NVTX)
        jit_profile_range_push(message);
#endif
    }

    ProfilerPhase(const drjit::TraversableBase *traversable) {
        int status;
        char message[1024] = {0};
        const char *name = typeid(*traversable).name();
        snprintf(message, 1024, "traverse_cb %s", name);

        jit_log(LogLevel::Debug, "profiler start: %s", message);
        jit_profile_range_push(message);
        m_message = message;
    }

    ~ProfilerPhase() {
#if defined(DRJIT_ENABLE_NVTX)
        jit_profile_range_pop();
#endif
        jit_log(LogLevel::Debug, "profiler end: %s", m_message.c_str());
    }
};

struct ADScopeContext {
    bool process_postponed;
    ADScopeContext(drjit::ADScope type, size_t size, const uint64_t *indices,
                   int symbolic, bool process_postponed)
        : process_postponed(process_postponed) {
        ad_scope_enter(type, size, indices, symbolic);
    }
    ~ADScopeContext() { ad_scope_leave(process_postponed); }
};

struct scoped_set_flag {
    uint32_t backup;
    scoped_set_flag(JitFlag flag, bool enabled) : backup(jit_flags()) {
        uint32_t flags = backup;
        if (enabled)
            flags |= (uint32_t)flag;
        else
            flags &= ~(uint32_t) flag;

        jit_set_flags(flags);
    }

    ~scoped_set_flag() {
        jit_set_flags(backup);
    }
};

struct state_lock_guard {
    state_lock_guard() { jit_state_lock(); }
    ~state_lock_guard() { jit_state_unlock(); }
};

using namespace detail;

bool Layout::operator==(const Layout &rhs) const {
    if (((bool) this->type != (bool) rhs.type) || !(this->type.equal(rhs.type)))
        return false;

    if (this->num != rhs.num)
        return false;

    if (this->fields.size() != rhs.fields.size())
        return false;

    for (uint32_t i = 0; i < this->fields.size(); ++i) {
        if (!(this->fields[i].equal(rhs.fields[i])))
            return false;
    }

    if (this->index != rhs.index)
        return false;

    if (this->flags != rhs.flags)
        return false;

    if (this->literal != rhs.literal)
        return false;

    if (this->vt != rhs.vt)
        return false;

    if (((bool) this->py_object != (bool) rhs.py_object) ||
        !this->py_object.equal(rhs.py_object))
        return false;

    return true;
}

bool VarLayout::operator==(const VarLayout &rhs) const {
    if (this->vt != rhs.vt)
        return false;

    if (this->vs != rhs.vs)
        return false;

    if (this->flags != rhs.flags)
        return false;

    if (this->size_index != rhs.size_index)
        return false;

    return true;
}

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
void FlatVariables::add_domain(const char *variant, const char *domain) {
    // Since it is not possible to pass nullptr strings to nanobind functions we
    // assume, that a valid domain indicates a valid variant. If the variant is
    // empty at the end of traversal, we know that no Class variable was
    // traversed, and registry traversal is not necessary.
    if (domain && variant && strcmp(domain, "") != 0) {
        jit_log(LogLevel::Debug, "variant=%s, domain=%s", variant, domain);

        if (domains.empty()){
            this->variant = variant;
        }
        else if (this->variant != variant)
            jit_raise("traverse(): Variant missmatch! All arguments to a "
                      "frozen function have to have the same variant. "
                      "Variant %s of a previos argument does not match "
                      "variant %s of this argument.",
                      this->variant.c_str(), variant);

        bool contains = false;
        for (std::string &d : domains) {
            if (d == domain) {
                contains = true;
                break;
            }
        }
        if (!contains)
            domains.push_back(domain);
    }
}

/**
 * Adds a jit index to the flattened array, deduplicating it.
 * This allows to check for aliasing conditions, where two variables
 * actually refer to the same index. The function should only be called for
 * scheduled non-literal variable indices.
 */
uint32_t FlatVariables::add_jit_index(uint32_t index) {
    uint32_t next_slot = this->variables.size();
    auto result        = this->index_to_slot.try_emplace(index, next_slot);
    auto it            = result.first;
    bool inserted = result.second;

    if (inserted) {
        this->variables.push_back(index);
        // Borrow the variable
        jit_var_inc_ref(index);
        this->var_layout.emplace_back();
        return next_slot;
    } else {
        return it.value();
    }
}

/**
 * Compares two \c FlatVariables to determine if it is possible to create an
 * opaque_mask. This is the case, if the only change between the PyTrees
 * consists of changes in literal values.
 */
bool compatible_auto_opaque(FlatVariables &cur, FlatVariables &prev){
    // NOTE: We only test the size of the layout, as a full test is somewhat
    // expensive, and the worst case is that we make too many variables opaque,
    // which does not impact correctness. If this causes problems, more
    // extensive tests might have to be reintroduced.
    if (cur.layout.size() != prev.layout.size()) {
        return false;
    }
    return true;
}

bool FlatVariables::fill_opaque_mask(FlatVariables &prev,
                                   drjit::vector<bool> &opaque_mask) {
    // If we notice that only a literal has changed, we can set the
    // corresponding bit in the mask, indicating that this literal should be
    // made opaque next time.
    uint32_t opaque_cunter = 0;
    bool new_opaques = false;
    for (uint32_t i = 0; i < this->layout.size(); i++) {
        Layout &layout      = this->layout[i];
        Layout &prev_layout = prev.layout[i];

        if (layout.flags & (uint32_t) LayoutFlag::Literal &&
            prev_layout.flags & (uint32_t) LayoutFlag::Literal &&
            (layout.literal != prev_layout.literal ||
             layout.index != prev_layout.index)) {
            opaque_mask[i] = true;
            new_opaques = true;
        }
        if (opaque_mask[i])
            opaque_cunter++;
    }

    jit_log(LogLevel::Debug,
            "compare_opaque(): %u variables will be made opaque",
            opaque_cunter);

    return new_opaques;
}

void FlatVariables::schedule_jit_variables(bool schedule_force,
                                           drjit::vector<bool> *opaque_mask) {
    ProfilerPhase profiler("schedule_jit_variables");
    for (uint32_t i = layout_index; i < layout.size(); i++) {
        Layout &layout = this->layout[i];

        if (!(layout.flags & (uint32_t) LayoutFlag::JitIndex))
            continue;

        uint32_t index = layout.index;

        int rv = 0;
        // We have to force scheduling of undefined variables, in order to
        // handle variables initialized with ``empty``.
        if (schedule_force ||
            (opaque_mask && (*opaque_mask)[i - layout_index]) ||
            jit_var_state(index) == VarState::Undefined) {
            // Returns owning reference
            index = jit_var_schedule_force(index, &rv);
        } else {
            // Schedule and create owning reference
            rv = jit_var_schedule(index);
            jit_var_inc_ref(index);
        }

        VarInfo info = jit_var_info(index);
        if (backend == info.backend || this->backend == JitBackend::None) {
            backend = info.backend;
        } else {
            jit_raise("freeze(): backend missmatch error (backend of this "
                      "variable %s does not match backend of others %s)!",
                      info.backend == JitBackend::CUDA ? "CUDA" : "LLVM",
                      backend == JitBackend::CUDA ? "CUDA" : "LLVM");
        }

        if (info.state == VarState::Literal) {
            // Special case, where the variable is a literal. This should not
            // occur, as all literals are made opaque in beforehand, however it
            // is nice to have a fallback.
            layout.literal = info.literal;
            // Store size in index variable, as this is not used for literals
            layout.index = info.size;
            layout.vt    = info.type;
            layout.literal_index = index;

            layout.flags |= (uint32_t) LayoutFlag::Literal;
        } else {
            layout.index = this->add_jit_index(index);
            layout.vt    = info.type;
            jit_var_dec_ref(index);
        }
    }
    layout_index = layout.size();
}

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
void FlatVariables::record_jit_variables() {
    ProfilerPhase profiler("record_jit_variables");
    assert(variables.size() == var_layout.size());
    for (uint32_t i = 0; i < var_layout.size(); i++){
        uint32_t index = variables[i];
        VarLayout &layout = var_layout[i];

        VarInfo info = jit_var_info(index);
        if (info.type == VarType::Pointer) {
            // We do not support pointers as inputs. It might be possible with
            // some extra handling, but they are never used directly.
            jit_raise("Pointer inputs not supported!");
        }

        layout.vs = info.state;
        layout.vt = info.type;
        layout.size_index = this->add_size(info.size);

        if (info.state == VarState::Evaluated) {
            // Special case, handling evaluated/opaque variables.

            layout.flags |=
                (info.size == 1 ? (uint32_t) LayoutFlag::SingletonArray : 0);
            layout.flags |= (info.unaligned ? (uint32_t) LayoutFlag::Unaligned : 0);

        } else {
            jit_raise("collect(): found variable %u in unsupported state %u!",
                      index, (uint32_t) info.state);
        }
    }
}

/**
 * This function returns an index of an equivalence class for the variable
 * size in the flattened variables.
 * It uses a hashmap and vector to deduplicate sizes.
 *
 * This is necessary, to catch cases, where two variables had the same size
 * when freezing a function and two different sizes when replaying.
 * In that case one kernel would be recorded, that evaluates both variables.
 * However, when replaying two kernels would have to be launched since the
 * now differently sized variables cannot be evaluated by the same kernel.
 */
uint32_t FlatVariables::add_size(uint32_t size) {
    uint32_t next_slot = this->sizes.size();
    auto result        = this->size_to_slot.try_emplace(size, next_slot);
    auto it            = result.first;
    bool inserted      = result.second;

    if (inserted) {
        this->sizes.push_back(size);
        return next_slot;
    } else {
        return it.value();
    }
}

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
void FlatVariables::traverse_jit_index(uint32_t index, TraverseContext &ctx,
                                       nb::handle tp) {
    (void) ctx;
    Layout &layout = this->layout.emplace_back();

    if (tp)
        layout.type = nb::borrow<nb::type_object>(tp);

    layout.flags |= (uint32_t) LayoutFlag::JitIndex;
    layout.index = index;
    layout.vt = jit_var_type(index);
}

/**
 * Construct a variable, given it's layout.
 * This is the counterpart to `traverse_jit_index`.
 *
 * Optionally, the index of a variable can be provided that will be
 * overwritten with the result of this function. In that case, the function
 * will check for compatible variable types.
 */
uint32_t FlatVariables::construct_jit_index(uint32_t prev_index) {
    Layout &layout = this->layout[layout_index++];

    uint32_t index;
    VarType vt;
    if (layout.flags & (uint32_t) LayoutFlag::Literal) {
        index = layout.literal_index;
        jit_var_inc_ref(index);
        vt    = layout.vt;
    } else {
        VarLayout &var_layout = this->var_layout[layout.index];
        index = this->variables[layout.index];
        jit_log(LogLevel::Debug, "    uses output[%u] = r%u", layout.index,
                index);

        jit_var_inc_ref(index);
        vt = var_layout.vt;
    }

    if (prev_index) {
        if (vt != (VarType) jit_var_type(prev_index))
            jit_fail("VarType missmatch %u != %u while assigning (r%u) "
                     "-> (r%u)!",
                     (uint32_t) vt, (uint32_t) jit_var_type(prev_index),
                     (uint32_t) prev_index, (uint32_t) index);
    }
    return index;
}

/**
 * Add an ad variable by it's index. Both the value and gradient are added
 * to the flattened variables. If the ad index has been marked as postponed
 * in the \c TraverseContext.postponed field, we mark the resulting layout
 * with that flag. This will cause the gradient edges to be propagated when
 * assigning to the input. The function takes an optional python-type if
 * it is known.
 */
void FlatVariables::traverse_ad_index(uint64_t index, TraverseContext &ctx,
                                      nb::handle tp) {
    // NOTE: instead of emplacing a Layout representing the ad variable always,
    // we only do so if the gradients have been enabled. We use this format,
    // since most variables will not be ad enabled. The layout therefore has to
    // be peeked in ``construct_ad_index`` before deciding if an ad or jit
    // index should be constructed/assigned.
    int grad_enabled = ad_grad_enabled(index);
    if (grad_enabled) {
        Layout &layout = this->layout.emplace_back();
        uint32_t ad_index = (uint32_t) (index >> 32);

        if (tp)
            layout.type = nb::borrow<nb::type_object>(tp);
        layout.num = 2;
        // layout.vt  = jit_var_type(index);

        // Set flags
        layout.flags |= (uint32_t) LayoutFlag::GradEnabled;
        // If the edge with this node as it's target has been postponed by
        // the isolate gradient scope, it has been enqueued and we mark the
        // ad variable as such.
        if (ctx.postponed && ctx.postponed->contains(ad_index)) {
            layout.flags |= (uint32_t) LayoutFlag::Postponed;
        }

        traverse_jit_index((uint32_t) index, ctx, tp);
        uint32_t grad = ad_grad(index);
        traverse_jit_index(grad, ctx, tp);
        ctx.free_list.push_back_steal(grad);
    } else {
        traverse_jit_index(index, ctx, tp);
    }
}

/**
 * Construct/assign the variable index given a layout.
 * This corresponds to `traverse_ad_index`.
 *
 * This function is also used for assignment to ad-variables.
 * If a `prev_index` is provided, and it is an ad-variable the gradient and
 * value of the flat variables will be applied to the ad variable,
 * preserving the ad_idnex.
 *
 * It returns an owning reference.
 */
uint64_t FlatVariables::construct_ad_index(uint32_t shrink,
                                           uint64_t prev_index) {
    Layout &layout = this->layout[this->layout_index];

    uint64_t index;
    if ((layout.flags & (uint32_t) LayoutFlag::GradEnabled) != 0) {
        Layout &layout = this->layout[this->layout_index++];
        bool postponed = (layout.flags & (uint32_t) LayoutFlag::Postponed);

        uint32_t val = construct_jit_index(prev_index);
        uint32_t grad = construct_jit_index(prev_index);

        // Resize the gradient if it is a literal
        if ((VarState) jit_var_state(grad) == VarState::Literal) {
            uint32_t new_grad = jit_var_resize(grad, jit_var_size(val));
            jit_var_dec_ref(grad);
            grad = new_grad;
        }

        // If the prev_index variable is provided we assign the new value
        // and gradient to the ad variable of that index instead of creating
        // a new one.
        uint32_t ad_index = (uint32_t) (prev_index >> 32);
        if (ad_index) {
            index = (((uint64_t) ad_index) << 32) | ((uint64_t) val);
            ad_var_inc_ref(index);
        } else
            index = ad_var_new(val);

        jit_log(LogLevel::Debug, " -> ad_var r%zu", index);
        jit_var_dec_ref(val);

        // Equivalent to set_grad
        ad_clear_grad(index);
        ad_accum_grad(index, grad);
        jit_var_dec_ref(grad);

        // Variables, that have been postponed by the isolate gradient scope
        // will be enqueued, which propagates their gradeint to previous
        // functions.
        if (ad_index && postponed) {
            ad_enqueue(drjit::ADMode::Backward, index);
        }
    } else {
        index = construct_jit_index(prev_index);
    }

    if (shrink > 0)
        index = ad_var_shrink(index, shrink);

    return index;
}

/**
 * Wrapper aground traverse_ad_index for a python handle.
 */
void FlatVariables::traverse_ad_var(nb::handle h, TraverseContext &ctx) {
    auto s = supp(h.type());

    if (s.is_class) {
        auto variant = nb::borrow<nb::str>(nb::getattr(h, "Variant"));
        auto domain  = nb::borrow<nb::str>(nb::getattr(h, "Domain"));
        add_domain(variant.c_str(), domain.c_str());
    }

    raise_if(s.index == nullptr, "freeze(): ArraySupplement index function "
                                 "pointer is nullptr.");

    uint64_t index = s.index(inst_ptr(h));

    this->traverse_ad_index(index, ctx, h.type());
}

/**
 * Construct an ad variable given it's layout.
 * This corresponds to `traverse_ad_var`
 */
nb::object FlatVariables::construct_ad_var(const Layout &layout,
                                           uint32_t shrink) {
    uint64_t index = construct_ad_index(shrink);

    auto result              = nb::inst_alloc_zero(layout.type);
    const ArraySupplement &s = supp(result.type());
    s.init_index(index, inst_ptr(result));
    nb::inst_mark_ready(result);

    // We have to release the reference, since assignment will borrow from
    // it.
    ad_var_dec_ref(index);

    return result;
}

/**
 * Assigns an ad variable.
 * Corresponds to `traverse_ad_var`.
 * This uses `construct_ad_index` to either construct a new ad variable or
 * assign the value and gradient to an already existing one.
 */
void FlatVariables::assign_ad_var(Layout &layout, nb::handle dst) {
    const ArraySupplement &s = supp(layout.type);

    uint64_t index;
    if (s.index) {
        // ``construct_ad_index`` is used for assignment
        index = construct_ad_index(0, s.index(inst_ptr(dst)));
    } else
        index = construct_ad_index();

    s.reset_index(index, inst_ptr(dst));
    jit_log(LogLevel::Debug, "index=%zu, grad_enabled=%u, ad_grad_enabled=%u",
            index, grad_enabled(dst), ad_grad_enabled(index));

    // Release reference, since ``construct_ad_index`` returns owning
    // reference and ``s.reset_index`` borrows from it.
    ad_var_dec_ref(index);
}

/**
 * Traverse a c++ tree using it's `traverse_1_cb_ro` callback.
 */
void FlatVariables::traverse_cb(const drjit::TraversableBase *traversable,
                                TraverseContext &ctx, nb::object type) {
    // ProfilerPhase profiler(traversable);

    uint32_t layout_index = this->layout.size();
    Layout &layout        = this->layout.emplace_back();
    layout.type           = nb::borrow<nb::type_object>(type);

    struct Payload{
        TraverseContext &ctx;
        FlatVariables *flat_variables = nullptr;
        uint32_t num_fields = 0;
    };

    Payload p{ctx, this, 0};

    traversable->traverse_1_cb_ro((void*) & p,
        [](void *p, uint64_t index, const char *variant, const char *domain) {
            if (!index)
                return;
            Payload *payload = (Payload *)p;
            payload->flat_variables->add_domain(variant, domain);
            payload->flat_variables->traverse_ad_index(index, payload->ctx);
            payload->num_fields++;
        });

    this->layout[layout_index].num = p.num_fields;
}

/**
 * Helper function, used to assign a callback variable.
 *
 * \param tmp
 *     This vector is populated with the indices to variables that have been
 *     constructed. It is required to release the references, since the
 *     references created by `construct_ad_index` are owning and they are
 *     borrowed after the callback returns.
 */
uint64_t FlatVariables::assign_cb_internal(uint64_t index,
                                           index64_vector &tmp) {
    if (!index)
        return index;

    uint64_t new_index = this->construct_ad_index(0, index);

    tmp.push_back_steal(new_index);
    return new_index;
}

/**
 * Assigns variables using it's `traverse_cb_rw` callback.
 * This corresponds to `traverse_cb`.
 */
void FlatVariables::assign_cb(drjit::TraversableBase *traversable) {
    Layout &layout = this->layout[layout_index++];


    struct Payload{
        FlatVariables *flat_variables =  nullptr;
        Layout &layout;
        index64_vector tmp;
        uint32_t field_counter = 0;
    };
    Payload p{ this, layout, index64_vector(), 0 };
    traversable->traverse_1_cb_rw(
        (void *) &p, [](void *p, uint64_t index, const char *, const char *) {
            if (!index)
                return index;
            Payload *payload = (Payload *) p;
            if (payload->field_counter >= payload->layout.num)
                jit_raise("While traversing an object "
                          "for assigning inputs, the number of variables to "
                          "assign (>%u) did not match the number of variables "
                          "traversed when recording (%u)!",
                          payload->field_counter, payload->layout.num);
            payload->field_counter++;
            return payload->flat_variables->assign_cb_internal(index, payload->tmp);
        });

    if (p.field_counter != layout.num)
        jit_raise("While traversing and object for assigning inputs, the "
                  "number of variables to assign did not match the number "
                  "of variables traversed when recording!");
}

/**
 * Helper struct, to construct path strings to variables.
 */
struct scoped_path {
    std::string &path;
    size_t size;
    scoped_path(std::string &path, const char *suffix, bool dict = false)
        : path(path), size(path.size()) {
        if (dict) {
            path += "[\"";
            path += suffix;
            path += "\"]";
        } else {
            path += ".";
            path += suffix;
        }
    }
    scoped_path(std::string &path, uint32_t suffix)
        : path(path), size(path.size()) {
        path += "[";
        path += std::to_string(suffix);
        path += "]";
    }
    ~scoped_path() { path.resize(size); }
};

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
void FlatVariables::traverse(nb::handle h, TraverseContext &ctx) {
    recursion_guard guard(this);

    scoped_set_flag traverse_scope(JitFlag::EnableObjectTraversal, true);

    ProfilerPhase profiler("traverse");
    nb::handle tp = h.type();

    auto tp_name = nb::type_name(tp).c_str();
    jit_log(LogLevel::Debug, "FlatVariables::traverse(): %s {", tp_name);

    try {
        uint32_t layout_index = this->layout.size();
        Layout &layout = this->layout.emplace_back();
        layout.type = nb::borrow<nb::type_object>(tp);
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);
            if (s.is_tensor) {
                nb::handle array = s.tensor_array(h.ptr());

                auto full_shape = nb::borrow<nb::tuple>(shape(h));

                // Instead of adding the whole shape of a tensor to the key, we
                // only add the inner part, not containing dimension 0. When
                // indexing into a tensor, this is the only dimension that is
                // not used in the index calculation. When constructing a tensor
                // this dimension is reconstructed from the width of the
                // underlying array.

                nb::list inner_shape;
                if (full_shape.size() > 0)
                    for (uint32_t i = 1; i < full_shape.size(); i++) {
                        inner_shape.append(full_shape[i]);
                    }

                layout.py_object = nb::tuple(inner_shape);

                traverse(nb::steal(array), ctx);
            } else if (s.ndim != 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(h));

                layout.num = len;

                for (Py_ssize_t i = 0; i < len; ++i){
                    scoped_path ps(ctx.path, i);
                    traverse(nb::steal(s.item(h.ptr(), i)), ctx);
                }
            } else {
                layout.num = 1;
                traverse_ad_var(h, ctx);
            }
        } else if (tp.is(&PyTuple_Type)) {
            nb::tuple tuple = nb::borrow<nb::tuple>(h);

            layout.num = tuple.size();

            for (uint32_t i = 0; i < tuple.size(); i++) {
                scoped_path ps(ctx.path, i);
                auto h2 = tuple[i];
                traverse(h2, ctx);
            }
        } else if (tp.is(&PyList_Type)) {
            nb::list list = nb::borrow<nb::list>(h);

            layout.num = list.size();

            for (uint32_t i = 0; i < list.size(); i++) {
                scoped_path ps(ctx.path, i);
                auto h2 = list[i];
                traverse(h2, ctx);
            }
        } else if (tp.is(&PyDict_Type)) {
            nb::dict dict = nb::borrow<nb::dict>(h);

            layout.num = dict.size();
            layout.fields.reserve(layout.num);
            for (auto k : dict.keys()) {
                layout.fields.push_back(nb::borrow(k));
            }

            for (auto [k, v] : dict) {
                scoped_path ps(ctx.path, nb::str(k).c_str(), true);
                traverse(v, ctx);
            }
        } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {

            layout.num = ds.size();
            layout.fields.reserve(layout.num);
            for (auto k : ds.keys()) {
                layout.fields.push_back(nb::borrow(k));
            }

            for (auto [k, v] : ds) {
                scoped_path ps(ctx.path, nb::str(k).c_str());
                traverse(nb::getattr(h, k), ctx);
            }
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {

            for (auto field : df) {
                nb::object k = field.attr(DR_STR(name));
                layout.fields.push_back(nb::borrow(k));
            }
            layout.num = layout.fields.size();

            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                scoped_path ps(ctx.path, nb::str(k).c_str());
                traverse(nb::getattr(h, k), ctx);
            }
        } else if (auto traversable = get_traversable_base(h); traversable) {
            traverse_cb(traversable, ctx, nb::borrow<nb::type_object>(tp));
        } else if (auto cb = get_traverse_cb_ro(tp); cb.is_valid()) {
            ProfilerPhase profiler("traverse cb");

            uint32_t num_fields = 0;

            // Traverse the opaque C++ object
            cb(h, nb::cpp_function([&](uint64_t index, const char *variant,
                                       const char *domain) {
                   if (!index)
                       return;
                   add_domain(variant, domain);
                   num_fields++;
                   this->traverse_ad_index(index, ctx, nb::none());
                   return;
               }));

            // Update layout number of fields
            this->layout[layout_index].num = num_fields;
        } else {
            jit_log(LogLevel::Info,
                    "traverse(): You passed a value of type %s to a frozen "
                    "function, it could not be converted to a Dr.Jit type. "
                    "Changing this value in future calls to the frozen "
                    "function will cause it to be re-traced. The value is "
                    "located at %s.",
                    nb::str(tp).c_str(), ctx.path.c_str());

            layout.py_object = nb::borrow<nb::object>(h);
        }
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "FlatVariables::traverse(): error encountered while "
            "processing an argument of type '%U' at location %s (see above).",
            ctx.path.c_str(), nb::type_name(tp).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(
            PyExc_RuntimeError,
            "FlatVariables::traverse(): error encountered "
            "while processing an argument of type '%U' at location %s: %s",
            ctx.path.c_str(), nb::type_name(tp).ptr(), e.what());
        nb::raise_python_error();
    }

    jit_log(LogLevel::Debug, "}");
}

/**
 * This is the counterpart to the ``traverse`` method, used to construct the
 * output of a frozen function. Given a layout vector and flat_variables, it
 * re-constructs the PyTree.
 */
nb::object FlatVariables::construct() {
    recursion_guard guard(this);

    if (this->layout.size() == 0) {
        return nb::none();
    }

    const Layout &layout = this->layout[layout_index++];

    auto tp_name = nb::type_name(layout.type).c_str();
    jit_log(LogLevel::Debug, "FlatVariables::construct(): %s {", tp_name);

    if (layout.type.is(nb::none().type())) {
        return nb::none();
    }
    try {
        if (is_drjit_type(layout.type)) {
            const ArraySupplement &s = supp(layout.type);
            if (s.is_tensor) {
                nb::object array = construct();

                // Reconstruct the full shape from the inner part, stored in the
                // layout and the width of the underlying array.
                auto inner_shape  = nb::borrow<nb::tuple>(layout.py_object);
                auto first_dim    = prod(shape(array), nb::none())
                                    .floor_div(prod(inner_shape, nb::none()));

                nb::list full_shape;
                full_shape.append(first_dim);
                for (uint32_t i = 0; i < inner_shape.size(); i++) {
                    full_shape.append(inner_shape[i]);
                }

                nb::object tensor = layout.type(array, nb::tuple(full_shape));
                return tensor;
            } else if (s.ndim != 1) {
                auto result      = nb::inst_alloc_zero(layout.type);
                dr::ArrayBase *p = inst_ptr(result);
                size_t size      = s.shape[0];
                if (size == DRJIT_DYNAMIC) {
                    size = s.len(p);
                    s.init(size, p);
                }
                for (size_t i = 0; i < size; ++i) {
                    result[i] = construct();
                }
                nb::inst_mark_ready(result);
                return result;
            } else {
                return construct_ad_var(layout);
            }
        } else if (layout.type.is(&PyTuple_Type)) {
            nb::list list;
            for (uint32_t i = 0; i < layout.num; ++i) {
                list.append(construct());
            }
            return nb::tuple(list);
        } else if (layout.type.is(&PyList_Type)) {
            nb::list list;
            for (uint32_t i = 0; i < layout.num; ++i) {
                list.append(construct());
            }
            return std::move(list);
        } else if (layout.type.is(&PyDict_Type)) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return std::move(dict);
        } else if (nb::dict ds = get_drjit_struct(layout.type); ds.is_valid()) {
            nb::object tmp = layout.type();
            // TODO: validation against `ds`
            for (auto k : layout.fields) {
                nb::setattr(tmp, k, construct());
            }
            return tmp;
        } else if (nb::object df = get_dataclass_fields(layout.type);
                   df.is_valid()) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return layout.type(**dict);
        } else if (layout.py_object) {
            return layout.py_object;
        } else {
            nb::raise("Tried to construct a variable of type %s that is not "
                      "constructable!",
                      nb::type_name(layout.type).c_str());
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "FlatVariables::construct(): error encountered while "
                       "processing an argument of type '%U' (see above).",
                       nb::type_name(layout.type).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "FlatVariables::construct(): error encountered "
                        "while processing an argument of type '%U': %s",
                        nb::type_name(layout.type).ptr(), e.what());
        nb::raise_python_error();
    }

    jit_log(LogLevel::Debug, "}");
}

/**
 * Assigns the flattened variables to an already existing PyTree.
 * This is used when input variables have changed.
 */
void FlatVariables::assign(nb::handle dst, TraverseContext &ctx) {
    recursion_guard guard(this);
    scoped_set_flag traverse_scope(JitFlag::EnableObjectTraversal, true);

    nb::handle tp  = dst.type();
    Layout &layout = this->layout[layout_index++];

    jit_log(LogLevel::Debug, "FlatVariables::assign(): %s with %s {",
            nb::type_name(tp).c_str(), nb::type_name(layout.type).c_str());

    if (!layout.type.equal(tp))
        jit_raise("Type missmatch! Type of the object at location %s when "
                  "recording (%s) does not match type of object that is "
                  "assigned (%s).",
                  ctx.path.c_str(), nb::type_name(tp).c_str(),
                  nb::type_name(layout.type).c_str());

    try {
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);

            if (s.is_tensor) {
                nb::handle array = s.tensor_array(dst.ptr());
                assign(nb::steal(array), ctx);
            } else if (s.ndim != 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(dst));

                for (Py_ssize_t i = 0; i < len; ++i) {
                    scoped_path ps(ctx.path, i);
                    assign(dst[i], ctx);
                }
            } else {
                assign_ad_var(layout, dst);
            }
        } else if (tp.is(&PyTuple_Type)) {
            nb::tuple tuple = nb::borrow<nb::tuple>(dst);
            raise_if(
                tuple.size() != layout.num,
                "The number of objects in this tuple changed from %u to %u "
                "while recording the function.",
                layout.num, (uint32_t) tuple.size());

            for (uint32_t i = 0; i < tuple.size(); i++){
                scoped_path ps(ctx.path, i);
                auto h2 = tuple[i];
                assign(h2, ctx);
            }
        } else if (tp.is(&PyList_Type)) {
            nb::list list = nb::borrow<nb::list>(dst);
            raise_if(
                list.size() != layout.num,
                "The number of objects in a list at %s changed from %u to %u "
                "while recording the function.",
                ctx.path.c_str(), layout.num, (uint32_t) list.size());

            for (uint32_t i = 0; i < list.size(); i++) {
                scoped_path ps(ctx.path, i);
                auto h2 = list[i];
                assign(h2, ctx);
            }
        } else if (tp.is(&PyDict_Type)) {
            nb::dict dict = nb::borrow<nb::dict>(dst);
            for (auto &k : layout.fields) {
                scoped_path ps(ctx.path, nb::str(k).c_str(), true);
                if (dict.contains(&k))
                    assign(dict[k], ctx);
                else
                    dst[k] = construct();
            }
        } else if (nb::dict ds = get_drjit_struct(dst); ds.is_valid()) {
            for (auto &k : layout.fields) {
                scoped_path ps(ctx.path, nb::str(k).c_str());
                if (nb::hasattr(dst, k))
                    assign(nb::getattr(dst, k), ctx);
                else
                    nb::setattr(dst, k, construct());
            }
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            for (auto k : layout.fields) {
                scoped_path ps(ctx.path, nb::str(k).c_str());
                if (nb::hasattr(dst, k))
                    assign(nb::getattr(dst, k), ctx);
                else
                    nb::setattr(dst, k, construct());
            }
        } else if (auto traversable = get_traversable_base(dst); traversable) {
            assign_cb(traversable);
        } else if (nb::object cb = get_traverse_cb_rw(tp); cb.is_valid()) {
            index64_vector tmp;
            uint32_t num_fields = 0;

            cb(dst, nb::cpp_function([&](uint64_t index, const char *,
                                         const char *) {
                   if (!index)
                       return index;
                   jit_log(LogLevel::Debug,
                           "assign(): traverse_cb[%u] was a%u r%u", num_fields,
                           (uint32_t) (index >> 32), (uint32_t) index);
                   num_fields++;
                   if (num_fields > layout.num)
                       jit_raise(
                           "While traversing the object of type %s at location "
                           "%s for assigning inputs, the number of variables "
                           "to assign (>%u) did not match the number of "
                           "variables traversed when recording(%u)!",
                           ctx.path.c_str(), nb::str(tp).c_str(), num_fields,
                           layout.num);
                   return assign_cb_internal(index, tmp);
               }));
            if (num_fields != layout.num)
                jit_raise(
                    "While traversing the object of type %s at location %s "
                    "for assigning inputs, the number of variables "
                    "to assign did not match the number of variables "
                    "traversed when recording!",
                    ctx.path.c_str(), nb::str(tp).c_str());
        } else {
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "FlatVariables::assign(): error encountered while "
                       "processing an argument at %s "
                       "of type '%U' (see above).",
                       ctx.path.c_str(), nb::type_name(tp).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "FlatVariables::assign(): error encountered "
                        "while processing an argument at %s "
                        "of type '%U': %s",
                        ctx.path.c_str(), nb::type_name(tp).ptr(), e.what());
        nb::raise_python_error();
    }

    jit_log(LogLevel::Debug, "}");
}

/**
 * First traverses the PyTree, then the registry. This ensures that
 * additional data to vcalls is tracked correctly.
 */
void FlatVariables::traverse_with_registry(nb::handle h, TraverseContext &ctx) {
    scoped_set_flag traverse_scope(JitFlag::EnableObjectTraversal, true);

    // Traverse the handle
    traverse(h, ctx);

    // Traverse the registry (if a class variable was traversed)
    if (!domains.empty()) {
        ProfilerPhase profiler("traverse_registry");
        uint32_t layout_index = this->layout.size();
        Layout &layout        = this->layout.emplace_back();
        layout.type           = nb::borrow<nb::type_object>(nb::none());

        uint32_t num_fields = 0;

        jit_log(LogLevel::Debug, "registry{");

        drjit::vector<void *> registry_pointers;
        for (std::string &domain : domains) {
            uint32_t registry_bound =
                jit_registry_id_bound(variant.c_str(), domain.c_str());
            uint32_t offset = registry_pointers.size();
            registry_pointers.resize(registry_pointers.size() + registry_bound, nullptr);
            jit_registry_get_pointers(variant.c_str(), domain.c_str(),
                                      &registry_pointers[offset]);
        }

        jit_log(LogLevel::Debug, "registry_bound=%u", registry_pointers.size());
        jit_log(LogLevel::Debug, "layout_index=%u", this->layout.size());
        for (void *ptr : registry_pointers) {
            jit_log(LogLevel::Debug, "ptr=%p", ptr);
            if (!ptr)
                continue;

            // WARN: very unsafe cast!
            // We assume, that any object added to the registry inherits from
            // TraversableBase. This is ensured by the signature of the
            // ``drjit::registry_put`` function.
            auto traversable = (drjit::TraversableBase *) ptr;
            auto self        = traversable->self_py();

            if (self)
                traverse(self, ctx);
            else
                traverse_cb(traversable, ctx);

            num_fields++;
        }
        jit_log(LogLevel::Debug, "}");

        this->layout[layout_index].num = num_fields;
    }
}

/**
 * First assigns the registry and then the PyTree.
 * Corresponds to `traverse_with_registry`.
 */
void FlatVariables::assign_with_registry(nb::handle dst, TraverseContext &ctx) {
    scoped_set_flag traverse_scope(JitFlag::EnableObjectTraversal, true);

    // Assign the handle
    assign(dst, ctx);

    // Assign registry (if a class variable was traversed)
    if (!domains.empty()) {
        Layout &layout      = this->layout[layout_index++];
        uint32_t num_fields = 0;

        jit_log(LogLevel::Debug, "registry{");

        drjit::vector<void *> registry_pointers;
        for (std::string &domain : domains) {
            uint32_t registry_bound =
                jit_registry_id_bound(variant.c_str(), domain.c_str());
            uint32_t offset = registry_pointers.size();
            registry_pointers.resize(registry_pointers.size() + registry_bound, nullptr);
            jit_registry_get_pointers(variant.c_str(), domain.c_str(),
                                      &registry_pointers[offset]);
        }

        jit_log(LogLevel::Debug, "registry_bound=%u", registry_pointers.size());
        jit_log(LogLevel::Debug, "layout_index=%u", this->layout_index);
        for (void *ptr : registry_pointers) {
            jit_log(LogLevel::Debug, "ptr=%p", ptr);
            if (!ptr)
                continue;

            // WARN: very unsafe cast!
            // We assume, that any object added to the registry inherits from
            // TraversableBase. This is ensured by the signature of the
            // ``drjit::registry_put`` function.
            auto traversable = (drjit::TraversableBase *) ptr;
            auto self        = traversable->self_py();

            if (self)
                assign(self, ctx);
            else
                assign_cb(traversable);

            num_fields++;
        }
        jit_log(LogLevel::Debug, "}");
    }
}

FlatVariables::~FlatVariables() {
    state_lock_guard guard;
    for (uint32_t i = 0; i < layout.size(); ++i) {
        Layout &l = layout[i];
        if (l.flags & (uint32_t) LayoutFlag::Literal && l.literal_index) {
            jit_var_dec_ref(l.literal_index);
        }
    }
}

void FlatVariables::borrow() {
    state_lock_guard guard;
    for (uint32_t &index : this->variables)
        jit_var_inc_ref(index);
}

void FlatVariables::release() {
    state_lock_guard guard;
    for (uint32_t &index : this->variables)
        jit_var_dec_ref(index);
}

bool log_diff_variable(LogLevel level, const FlatVariables &curr,
                       const FlatVariables &prev, std::string &path,
                       uint32_t slot) {
    const VarLayout &curr_l = curr.var_layout[slot];
    const VarLayout &prev_l = prev.var_layout[slot];

    if(curr_l.vt != prev_l.vt){
        jit_log(level, "%s: The variable type changed from %u to %u.",
                path.c_str(), prev_l.vt, curr_l.vt);
        return false;
    }
    if(curr_l.size_index != prev_l.size_index){
        jit_log(level,
                "%s: The size equivalence class of the variable changed from "
                "%u to %u.",
                path.c_str(), prev_l.size_index, curr_l.size_index);
        return false;
    }

    return true;
}

/**
 * Log the difference of the layout nodes at ``index`` for the two FlatVariables.
 */
bool log_diff(LogLevel level, const FlatVariables &curr,
              const FlatVariables &prev, uint32_t &index, std::string &path) {

    const Layout &curr_l = curr.layout[index];
    const Layout &prev_l = prev.layout[index];
    index++;

    if (curr_l.flags != prev_l.flags) {
        jit_log(level, "%s: The flags of this node changed from 0x%lx to 0x%lx",
                path.c_str(), prev_l.flags, curr_l.flags);
        return false;
    }

    if (curr_l.index != prev_l.index) {
        jit_log(level,
                "%s: The index into the array of deduplicated variables "
                "changed from s%u to s%u. This can occur if two variables "
                "referred to the same JIT index, but do no longer.",
                path.c_str(), prev_l.index, curr_l.index);
        return false;
    }

    if (curr_l.flags & (uint32_t) LayoutFlag::JitIndex &&
        !(curr_l.flags & (uint32_t) LayoutFlag::Literal)) {
        uint32_t slot = curr_l.index;
        if (!log_diff_variable(level, curr, prev, path, slot))
            return false;
    }

    if (((bool) curr_l.type != (bool) prev_l.type) ||
        !(curr_l.type.equal(prev_l.type))) {
        jit_log(level, "%s: The type of this node changed from %s to %s",
                path.c_str(), nb::str(prev_l.type).c_str(),
                nb::str(curr_l.type).c_str());
        return false;
    }

    if (curr_l.literal != prev_l.literal) {
        jit_log(level,
                "%s: The literal value of this variable changed from 0x%llx to "
                "0x%llx",
                path.c_str(), prev_l.literal, curr_l.literal);
        return false;
    }

    if (((bool) curr_l.py_object != (bool) prev_l.py_object) ||
        !curr_l.py_object.equal(prev_l.py_object)) {
        jit_log(level, "%s: The object changed from %s to %s", path.c_str(),
                nb::str(prev_l.py_object).c_str(),
                nb::str(curr_l.py_object).c_str());
        return false;
    }

    if (curr_l.num != prev_l.num){
        jit_log(level,
                "%s: The number of elements of this container changed from %u "
                "to %u",
                path.c_str(), prev_l.num, curr_l.num);
        return false;
    }

    if (curr_l.fields.size() != prev_l.fields.size()){
        jit_log(level,
                "%s: The number of elements of this container changed from %u "
                "to %u",
                path.c_str(), prev_l.fields.size(), curr_l.fields.size());
        return false;
    }

    for (uint32_t i = 0; i < curr_l.fields.size(); ++i) {
        if (!(curr_l.fields[i].equal(prev_l.fields[i]))) {
            jit_log(level, "%s: The %ith key changed from \"%s\" to \"%s\"",
                    path.c_str(), i, nb::str(curr_l.fields[i]).c_str(),
                    nb::str(prev_l.fields[i]).c_str());
        }
    }

    if (curr_l.fields.size() > 0){
        for (uint32_t i = 0; i < curr_l.fields.size(); i++){
            auto &field = curr_l.fields[i];

            uint32_t path_size = path.size();

            if(curr_l.type.is(&PyDict_Type)){
                path += "[\"";
                path += nb::str(field).c_str();
                path += "\"]";
            }else{
                path += ".";
                path += nb::str(field).c_str();
            }

            log_diff(level, curr, prev, index, path);

            path.resize(path_size);
        }
    }else{
        for (uint32_t i = 0; i < curr_l.num; i++) {
            uint32_t path_size = path.size();
            path += "[";
            path += std::to_string(i);
            path += "]";

            log_diff(level, curr, prev, index, path);

            path.resize(path_size);
        }
    }

    return true;
}

/**
 * Log the difference of the two FlatVariables.
 */
bool log_diff(LogLevel level, const FlatVariables &curr,
              const FlatVariables &prev) {
    if (curr.flags != prev.flags) {
        jit_log(level, "The flags of the input changed from %lx to %lx",
                prev.flags, curr.flags);
        return false;
    }
    if (curr.layout.size() != prev.layout.size()) {
        jit_log(level,
                "The number of elements in the input changed from %u to %u.",
                prev.layout.size(), curr.layout.size());
        return false;
    }
    if (curr.var_layout.size() != prev.var_layout.size()) {
        jit_log(level,
                "The number of opaque variables in the input changed from %u "
                "to %u.",
                prev.var_layout.size(), curr.var_layout.size());
        return false;
    }

    uint32_t index = 0;
    std::string path;
    log_diff(level, curr, prev, index, path);

    return true;
}
inline void hash_combine(uint64_t &seed, uint64_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const uint64_t mult = 0x9ddfea08eb382d69ull;
    uint64_t a          = (value ^ seed) * mult;
    a ^= (a >> 47);
    uint64_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

size_t
FlatVariablesHasher::operator()(const std::shared_ptr<FlatVariables> &key) const {
    ProfilerPhase profiler("hash");
    // Hash the layout

    // TODO: Maybe we can use xxh by first collecting in vector<uint64_t>?

    uint64_t hash = (uint64_t) (key->layout.size() << 32) |
                    (uint64_t) (key->var_layout.size() << 2);

    for (const Layout &layout : key->layout) {
        // if layout.fields is not 0 then layout.num == layout.fields.size()
        // therefore we can omit layout.fields.size().
        hash_combine(hash,
                     ((uint64_t) layout.num << 32) | ((uint64_t) layout.index));
        hash_combine(hash,
                     ((uint64_t) layout.flags << 32) | ((uint64_t) layout.vt));
        if (layout.flags & (uint32_t) LayoutFlag::JitIndex)
            hash_combine(hash, layout.literal);

        uint32_t type_hash = 0;
        if (layout.type)
            type_hash = nb::hash(layout.type);

        uint32_t object_hash = 0;
        if (layout.py_object) {
            PyObject *ptr = layout.py_object.ptr();
            Py_hash_t rv = PyObject_Hash(ptr);

            // Try to hash the object, and otherwise fallback to ``id()``
            if (rv == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                object_hash = (uintptr_t) ptr;
            } else {
                object_hash = rv;
            }
        }
        if (type_hash && object_hash)
            hash_combine(hash, ((uint64_t) type_hash << 32) |
                                   ((uint64_t) (uint32_t) object_hash));
        for (auto &field : layout.fields)
             hash_combine(hash, nb::hash(field.ptr()));
    }

    for (const VarLayout &layout : key->var_layout) {
        // layout.vt: 4
        // layout.vs: 4
        // layout.flags: 6
        hash_combine(hash, ((uint64_t) layout.size_index << 32) |
                               ((uint64_t) layout.flags << 8) |
                               ((uint64_t) layout.vs << 4) |
                               ((uint64_t) layout.vt));
    }

    return hash;
}

/*
 * Record a function, given it's python input and flattened input.
 */
nb::object FunctionRecording::record(nb::callable func,
                                     FrozenFunction *frozen_func,
                                     nb::list input,
                                     const FlatVariables &in_variables) {
    ProfilerPhase profiler("record");
    JitBackend backend = in_variables.backend;

    frozen_func->recording_counter++;
    if (frozen_func->recording_counter > frozen_func->warn_recording_count &&
        frozen_func->recordings.size() >= 1) {
        jit_log(
            LogLevel::Warn,
            "The frozen function has been recorded %u times, this indicates a "
            "problem with how the frozen function is being called. For "
            "example, calling it with changing python values such as an "
            "index. For more information about which variables changed set the "
            "log level to ``LogLevel::Debug``.",
            frozen_func->recording_counter);
        log_diff(LogLevel::Debug, in_variables, *frozen_func->prev_key);
    }

    jit_log(LogLevel::Info,
            "Recording (n_inputs=%u):", in_variables.variables.size());
    jit_freeze_start(backend, in_variables.variables.data(),
                     in_variables.variables.size());

    // Record the function
    nb::object output;
    {
        ProfilerPhase profiler("function");
        output = func(*input[0], **input[1]);
    }

    // Collect nodes, that have been postponed by the `Isolate` scope in a
    // hash set.
    // These are the targets of postponed edges, as the isolate gradient
    // scope only handles backward mode differentiation.
    // If they are, then we have to enqueue them when replaying the
    // recording.
    tsl::robin_set<uint32_t, UInt32Hasher> postponed;
    {
        drjit::vector<uint32_t> postponed_vec;
        ad_scope_postponed(&postponed_vec);
        for (uint32_t index : postponed_vec)
            postponed.insert(index);
    }

    jit_log(LogLevel::Info, "Traversing output");
    {
        ProfilerPhase profiler("traverse output");
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);

        jit_log(LogLevel::Info, "traverse output");

        TraverseContext ctx;
        ctx.postponed = &postponed;

        out_variables.traverse(output, ctx);
        out_variables.schedule_jit_variables(false, nullptr);

        out_variables.traverse_with_registry(input, ctx);
        out_variables.schedule_jit_variables(false, nullptr);

        out_variables.layout_index = 0;

        { // Evaluate the variables, scheduled when traversing
            nb::gil_scoped_release guard;
            jit_eval();
        }

        out_variables.record_jit_variables();
    }

    jit_freeze_pause(backend);

    if ((out_variables.variables.size() > 0 &&
         in_variables.variables.size() > 0) &&
        out_variables.backend != backend) {
        Recording *recording = jit_freeze_stop(backend, nullptr, 0);
        jit_freeze_destroy(recording);

        nb::raise(
            "freeze(): backend missmatch error (backend %u of "
            "output variables did not match backend %u of input variables)",
            (uint32_t) out_variables.backend, (uint32_t) backend);
    }

    // Exceptions, thrown by the recording functions will be recorded and
    // re-thrown when calling ``jit_freeze_stop``. Since the output variables
    // are borrowed, we have to release them in that case, and catch these
    // exceptions.
    try {
        recording = jit_freeze_stop(backend, out_variables.variables.data(),
                                    out_variables.variables.size());
    } catch (nb::python_error &e) {
        out_variables.release();
        nb::raise_from(e, PyExc_RuntimeError,
                       "record(): error encountered while recording a function "
                       "(see above).");
    } catch (const std::exception &e) {
        out_variables.release();
        nb::chain_error(PyExc_RuntimeError, "record(): %s", e.what());
        nb::raise_python_error();
    }

    jit_log(LogLevel::Info, "Recording done (n_outputs=%u)",
            out_variables.variables.size());

    // For catching input assignment mismatches, we assign the input and
    // output
    {
        state_lock_guard guard;
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);

        out_variables.layout_index = 0;
        jit_log(LogLevel::Debug, "Construct:");
        output = nb::borrow<nb::object>(out_variables.construct());
        // NOTE: temporarily disable this to not enqueue twice
        TraverseContext ctx;
        out_variables.assign(input, ctx);
        out_variables.layout_index = 0;
    }

    // Traversal takes owning references, so here we need to release them.
    out_variables.release();

    return output;
}
/*
 * Replays the recording.
 *
 * This constructs the output and re-assigns the input.
 */
nb::object FunctionRecording::replay(nb::callable func,
                                     FrozenFunction *frozen_func,
                                     nb::list input,
                                     const FlatVariables &in_variables) {
    ProfilerPhase profiler("replay");

    jit_log(LogLevel::Info, "Replaying:");
    int dryrun_success;
    {
        ProfilerPhase profiler("dry run");
        dryrun_success =
            jit_freeze_dry_run(recording, in_variables.variables.data());
    }
    if (!dryrun_success) {
        // Dry run has failed. Re-record the function.
        jit_log(LogLevel::Info, "Dry run failed! re-recording");
        this->clear();
        try {
            return this->record(func, frozen_func, input, in_variables);
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_RuntimeError,
                           "replay(): error encountered while re-recording a "
                           "function (see above).");
        } catch (const std::exception &e) {
            jit_freeze_abort(in_variables.backend);

            nb::chain_error(PyExc_RuntimeError, "record(): %s", e.what());
            nb::raise_python_error();
        }
    } else {
        ProfilerPhase profiler("jit replay");
        nb::gil_scoped_release guard;
        jit_freeze_replay(recording, in_variables.variables.data(),
                          out_variables.variables.data());
    }
    jit_log(LogLevel::Info, "Replaying done:");

    // Construct Output variables
    nb::object output;
    {
        state_lock_guard guard;
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);
        out_variables.layout_index = 0;
        {
            ProfilerPhase profiler("construct output");
            output = nb::borrow<nb::object>(out_variables.construct());
        }
        {
            ProfilerPhase profiler("assign input");
            TraverseContext ctx;
            out_variables.assign_with_registry(input, ctx);
        }
    }

    // out_variables is assigned by ``jit_record_replay``, which transfers
    // ownership to this array. Therefore, we have to drop the variables
    // afterwards.
    out_variables.release();

    return output;
}

nb::object FrozenFunction::operator()(nb::args args, nb::kwargs kwargs) {
    nb::object result;
    {
        // Enter Isolate grad scope, so that gradients are not propagated
        // outside of the function scope.
        ADScopeContext ad_scope(drjit::ADScope::Isolate, 0, nullptr, -1, true);

        // Kernel freezing can be enabled or disabled with the
        // ``JitFlag::KernelFreezing``. Alternatively, when calling a frozen
        // function from another one, we simply record the inner function.
        if (!jit_flag(JitFlag::KernelFreezing) ||
            jit_flag(JitFlag::FreezingScope) || max_cache_size == 0) {
            ProfilerPhase profiler("function");
            return func(*args, **kwargs);
        }

        call_counter++;

        nb::list input;
        input.append(args);
        input.append(kwargs);

        auto in_variables =
            std::make_shared<FlatVariables>(FlatVariables(in_heuristics));
        in_variables->backend = this->default_backend;
        // Evaluate and traverse input variables (args and kwargs)
        // Repeat this a max of 2 times if the number of variables that should
        // be made opaque changed.
        for (uint32_t i = 0; i < 2; i++) {
            state_lock_guard guard;
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, 0,
                                    true);


            // Traverse input variables
            ProfilerPhase profiler("traverse input");
            jit_log(LogLevel::Info, "freeze(): Traversing input");

            TraverseContext ctx;
            in_variables->traverse_with_registry(input, ctx);

            // If this is the first time the frozen function has been called or
            // the layout is not compatible with the previous one, we clear the
            // opaque_mask.
            bool auto_opaque = false;
            if (prev_key) {
                auto_opaque = compatible_auto_opaque(*in_variables, *prev_key);
                if (!auto_opaque) {
                    // The mask is reset if they are not compatible
                    opaque_mask.resize(in_variables->layout.size());
                    for (uint32_t i = 0; i < opaque_mask.size(); i++)
                        opaque_mask[i] = false;
                    jit_log(LogLevel::Debug, "auto-opaque incompatible");
                }
            } else
                opaque_mask.resize(in_variables->layout.size(), false);

            in_variables->schedule_jit_variables(!this->auto_opaque,
                                                 &opaque_mask);

            in_variables->layout_index = 0;

            { // Evaluate the variables, scheduled when traversing
                ProfilerPhase profiler("eval");
                nb::gil_scoped_release guard;
                jit_eval();
            }

            in_variables->record_jit_variables();
            bool new_opaques = false;
            if (prev_key && auto_opaque)
                new_opaques =
                    in_variables->fill_opaque_mask(*prev_key, opaque_mask);

            if (new_opaques) {
                // If new variables have been discovered that should be made
                // opaque, we repeat traversal of the input to make them opaque.
                // This reduces the number of variants that are saved by one.
                jit_log(LogLevel::Warn,
                        "While traversing the frozen function input, new "
                        "literal variables have been discovered which changed "
                        "from one call to another. These will be made opaque, "
                        "and the input will be traversed again. This will "
                        "incurr some overhead. To prevent this, make those "
                        "variables opaque in beforehand. Below, a list of "
                        "variables that changed will be shown.");
                if (prev_key)
                    log_diff(LogLevel::Warn, *in_variables, *prev_key);
                in_variables->release();
                in_variables = std::make_shared<FlatVariables>(
                    FlatVariables(in_heuristics));
            } else {
                break;
            }
        }

        in_heuristics = in_heuristics.max(in_variables->heuristic());

        raise_if(in_variables->backend == JitBackend::None,
                 "freeze(): Cannot infer backend without providing input "
                 "variable to frozen function!");

        auto it = this->recordings.find(in_variables);

        // Evict the least recently used recording if the cache is "full"
        if (max_cache_size > 0 &&
            recordings.size() >= (uint32_t) max_cache_size &&
            it == this->recordings.end()) {

            uint32_t lru_last_used        = UINT32_MAX;
            RecordingMap::iterator lru_it = recordings.begin();

            for (auto it = recordings.begin(); it != recordings.end(); it++) {
                auto &recording = it.value();
                if (recording->last_used < lru_last_used) {
                    lru_last_used = recording->last_used;
                    lru_it        = it;
                }
            }
            recordings.erase(lru_it);

            it = this->recordings.find(in_variables);
        }

        if (it == this->recordings.end()) {
            {
                // TODO: single traverse
                ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, 0,
                                        true);
                TraverseContext ctx;
                in_variables->assign_with_registry(input, ctx);
            }

            // FunctionRecording recording;
            auto recording = std::make_unique<FunctionRecording>();
            recording->last_used = call_counter - 1;

            try {
                result = recording->record(func, this, input, *in_variables);
            } catch (nb::python_error &e) {
                in_variables->release();
                jit_freeze_abort(in_variables->backend);
                nb::raise_from(
                    e, PyExc_RuntimeError,
                    "record(): error encountered while recording a frozen "
                    "function (see above).");
            } catch (const std::exception &e) {
                in_variables->release();
                jit_freeze_abort(in_variables->backend);

                nb::chain_error(PyExc_RuntimeError, "record(): %s", e.what());
                nb::raise_python_error();
            };

            in_variables->release();

            this->prev_key = in_variables;
            this->recordings.insert(
                { std::move(in_variables), std::move(recording) });

        } else {
            FunctionRecording *recording = it.value().get();

            recording->last_used = call_counter - 1;

            {
                result = recording->replay(func, this, input, *in_variables);
            }

            // Drop references to variables
            in_variables->release();
        }
    }
    ad_traverse(drjit::ADMode::Backward,
                (uint32_t) drjit::ADFlag::ClearVertices);
    return result;
}

void FrozenFunction::clear() {
    recordings.clear();
    prev_key          = std::make_shared<FlatVariables>(FlatVariables());
    recording_counter = 0;
    call_counter      = 0;
}

/**
 * This function inspects the content of the frozen function to detect reference
 * cycles, that could lead to memory or type leaks. It can be called by the
 * garbage collector by adding it to the ``type_slots`` of the
 * ``FrozenFunction`` definition.
 */
int frozen_function_tp_traverse(PyObject *self, visitproc visit, void *arg) {
    FrozenFunction *f = nb::inst_ptr<FrozenFunction>(self);

    nb::handle func = nb::find(f->func);
    Py_VISIT(func.ptr());

    for (auto &it : f->recordings) {
        for (auto &layout : it.first->layout) {
            nb::handle type = nb::find(layout.type);
            Py_VISIT(type.ptr());
            nb::handle object = nb::find(layout.py_object);
            Py_VISIT(object.ptr());
        }
        for (auto &layout : it.second->out_variables.layout) {
            nb::handle type = nb::find(layout.type);
            Py_VISIT(type.ptr());
            nb::handle object = nb::find(layout.py_object);
            Py_VISIT(object.ptr());
        }
    }

    return 0;
}

/**
 * This function releases the internal function of the ``FrozenFunction``
 * object. It is used by the garbage collector to "break" potential reference
 * cycles, resulting from the frozen function being referenced in the closure of
 * the wrapped variable.
 */
int frozen_function_clear(PyObject *self) {
    FrozenFunction *f = nb::inst_ptr<FrozenFunction>(self);

    f->func.release();

    return 0;
}

// Slot data structure referencing the above two functions
static PyType_Slot slots[] = { { Py_tp_traverse,
                                 (void *) frozen_function_tp_traverse },
                               { Py_tp_clear, (void *) frozen_function_clear },
                               { 0, nullptr } };

void export_freeze(nb::module_ & /*m*/) {

    nb::module_ d = nb::module_::import_("drjit.detail");
    auto traversable_base =
        nb::class_<drjit::TraversableBase>(d, "TraversableBase");
    nb::class_<FrozenFunction>(d, "FrozenFunction", nb::type_slots(slots))
        .def(nb::init<nb::callable, int, uint32_t, JitBackend, bool>())
        .def_prop_ro(
            "n_cached_recordings",
            [](FrozenFunction &self) { return self.n_cached_recordings(); })
        .def_ro("n_recordings", &FrozenFunction::recording_counter)
        .def("clear", &FrozenFunction::clear)
        .def("__call__", &FrozenFunction::operator());
}
