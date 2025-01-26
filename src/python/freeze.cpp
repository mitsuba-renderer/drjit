#include "freeze.h"
#include "apply.h"
#include "autodiff.h"
#include "base.h"
#include "common.h"
#include <drjit-core/hash.h>
#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/autodiff.h>
#include <drjit/extra.h>
#include <drjit/fwd.h>
#include <drjit/traversable_base.h>
#include "listobject.h"
#include "nanobind/nanobind.h"
#include "object.h"
#include "pyerrors.h"
#include "shape.h"
#include "tupleobject.h"
#include <bitset>
#include <cxxabi.h>
#include <ios>
#include <ostream>
#include <vector>

struct ProfilerPhase {
    std::string m_message;
    ProfilerPhase(const char *message) : m_message(message) {
        jit_log(LogLevel::Debug, "profiler start: %s", message);
        jit_profile_range_push(message);
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
        jit_profile_range_pop();
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
    if (this->vt != rhs.vt)
        return false;

    if (this->vs != rhs.vs)
        return false;

    if (this->flags != rhs.flags)
        return false;

    if (this->index != rhs.index)
        return false;

    if (this->size_index != rhs.size_index)
        return false;

    if (this->literal != rhs.literal)
        return false;

    if (((bool) this->py_object != (bool) rhs.py_object) ||
        !this->py_object.equal(rhs.py_object))
        return false;

    return true;
}

static void log_layouts(const std::vector<Layout> &layouts, std::ostream &os,
                        uint32_t &index, std::string &padding) {
    const Layout &layout = layouts[index++];

    auto tp_name = nb::type_name(layout.type).c_str();
    os << padding << "type = " << tp_name << std::endl;
    os << padding << "num: " << layout.num << std::endl;
    os << padding << "vt: " << (uint32_t) layout.vt << std::endl;
    os << padding << "vs: " << (uint32_t) layout.vs << std::endl;
    os << padding << "flats: " << std::bitset<8>(layout.flags) << std::endl;
    os << padding << "literal: " << std::hex << layout.literal << std::endl;
    os << padding << "index: " << layout.index << std::endl;
    os << padding << "size_index: " << layout.size_index << std::endl;
    os << padding << "py_object: " << nb::str(layout.py_object).c_str()
       << std::endl;

    if (layout.fields.size() == 0)
        for (uint32_t i = 0; i < layout.num; i++) {
            os << padding << "Layout[" << std::endl;
            padding.append("    ");

            log_layouts(layouts, os, index, padding);

            padding.resize(padding.length() - 4);
            os << padding << "]" << std::endl;
        }
    else
        for (const auto &field : layout.fields) {
            os << padding << nb::str(field).c_str() << ": Layout[" << std::endl;
            padding.append("    ");

            log_layouts(layouts, os, index, padding);

            padding.resize(padding.length() - 4);
            os << padding << "]" << std::endl;
        }
}

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
 * Adds a variable to the flattened array, deduplicating it.
 * This allows for checking for aliasing conditions, as aliasing inputs map
 * to the same flat variable index.
 */
uint32_t FlatVariables::add_variable_index(uint32_t variable_index) {
    uint32_t next_slot = this->variables.size();
    auto result   = this->index_to_slot.try_emplace(variable_index, next_slot);
    auto it       = result.first;
    bool inserted = result.second;

    if (inserted) {
        this->variables.push_back(variable_index);
        return next_slot;
    } else {
        return it.value();
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
 */
void FlatVariables::traverse_jit_index(uint32_t index, TraverseContext &ctx,
                                       nb::handle tp) {
    Layout &layout = this->layout.emplace_back();

    VarInfo info           = jit_set_backend(index);
    JitBackend var_backend = info.backend;
    VarType vt             = info.type;
    uint32_t var_size      = info.size;
    VarState vs = info.state;
    bool unaligned = info.unaligned;

    if (backend == var_backend || this->backend == JitBackend::None) {
        backend = var_backend;
    } else {
        jit_raise("freeze(): backend missmatch error (backend of this "
                  "variable %s does not match backend of others %s)!",
                  var_backend == JitBackend::CUDA ? "CUDA" : "LLVM",
                  backend == JitBackend::CUDA ? "CUDA" : "LLVM");
    }

    if (vt == VarType::Pointer) {
        // We do not support pointers as inputs. It might be possible with
        // some extra handling, but they are never used directly.
        jit_raise("Pointer inputs not supported!");
    }

    if (tp)
        layout.type = nb::borrow<nb::type_object>(tp);
    layout.vs         = vs;
    layout.vt         = vt;
    layout.size_index = this->add_size(var_size);

    if (vs == VarState::Literal) {
        // Special case, where the variable is a literal. This should not
        // occur, as all literals are made opaque in beforehand, however it
        // is nice to have a fallback.
        layout.literal = info.literal;
        // Store size in index variable, as this is not used for literals
        layout.index = var_size;
    } else if (vs == VarState::Evaluated) {
        // Special case, handling evaluated/opaque variables.

        layout.index = this->add_variable_index(index);

        layout.flags |=
            (var_size == 1 ? (uint32_t) LayoutFlag::SingletonArray : 0);
        layout.flags |= (unaligned ? (uint32_t) LayoutFlag::Unaligned : 0);

    } else {
        jit_raise("collect(): found variable %u in unsupported state %u!",
                  index, (uint32_t) vs);
    }
}

/**
 * Construct a variable, given it's layout.
 * This is the counterpart to `traverse_jit_index`.
 */
uint32_t FlatVariables::construct_jit_index(uint32_t prev_index) {
    Layout &layout = this->layout[layout_index++];

    uint32_t index;
    if (layout.vs == VarState::Literal) {
        index = jit_var_literal(this->backend, layout.vt, &layout.literal,
                                layout.index);

    } else {
        index = this->variables[layout.index];
        jit_log(LogLevel::Debug, "    uses output[%u] = r%u", layout.index,
                index);

        jit_var_inc_ref(index);
    }

    if (prev_index) {
        if (layout.vt != (VarType) jit_var_type(prev_index))
            jit_fail("VarType missmatch %u != %u while assigning (r%u) "
                     "-> (r%u)!",
                     (uint32_t) layout.vt, (uint32_t) jit_var_type(prev_index),
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
    // be peeked in ``construct_ad_index`` before descending if an ad or jit
    // index should be constructed/assigned.
    int grad_enabled = ad_grad_enabled(index);
    if (grad_enabled) {
        Layout &layout = this->layout.emplace_back();
        uint32_t ad_index = (uint32_t) (index >> 32);

        if (tp)
            layout.type = nb::borrow<nb::type_object>(tp);
        layout.num = 2;
        layout.vt  = jit_var_type(index);

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
        jit_var_dec_ref(grad);
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

        uint32_t val = construct_jit_index();
        uint32_t grad = construct_jit_index();

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

        if (prev_index) {
            if (layout.vt != (VarType) jit_var_type(prev_index))
                jit_fail(
                    "VarType missmatch %u != %u while assigning (a%u, r%u) "
                    "-> (a%u, r%u)!",
                    (uint32_t) layout.vt, (uint32_t) jit_var_type(prev_index),
                    (uint32_t) (prev_index >> 32), (uint32_t) prev_index,
                    (uint32_t) (index >> 32), (uint32_t) index);
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
    ProfilerPhase profiler(traversable);

    uint32_t layout_index = this->layout.size();
    Layout &layout        = this->layout.emplace_back();
    layout.type           = nb::borrow<nb::type_object>(type);

    uint32_t num_fileds = 0;

    struct Payload {
        FlatVariables *flat_vars;
        uint32_t num_fields;
        TraverseContext *ctx;
        std::vector<uint64_t> indices;
    };
    Payload payload{ this, 0, &ctx };
    traversable->traverse_1_cb_ro(
        (void *) &payload,
        [](void *p, uint64_t index, const char *variant, const char *domain) {
            if (!index)
                return;
            Payload *payload = (Payload *) p;
            payload->flat_vars->add_domain(variant, domain);
            payload->num_fields++;
            payload->indices.push_back(index);
        });

    {
        ProfilerPhase p("indices");
        for (uint64_t index : payload.indices) {
            this->traverse_ad_index(index, ctx);
        }
    }

    this->layout[layout_index].num = payload.num_fields;
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

    struct Payload {
        FlatVariables *flat_vars;
        index64_vector tmp;
        uint32_t num_fields;
        uint32_t field_counter;
    };
    Payload payload{ this, index64_vector(), (uint32_t) layout.num, 0 };
    traversable->traverse_1_cb_rw(
        (void *) &payload, [](void *p, uint64_t index) {
            if (!index)
                return index;
            Payload *payload = (Payload *) p;
            jit_log(LogLevel::Debug, "    field_counter=%u",
                    payload->field_counter);
            if (payload->field_counter >= payload->num_fields)
                jit_raise("While traversing an object "
                          "for assigning inputs, the number of variables to "
                          "assign (>%u) did not match the number of variables "
                          "traversed when recording (%u)!",
                          payload->field_counter, payload->num_fields);
            payload->field_counter++;

            return payload->flat_vars->assign_cb_internal(index, payload->tmp);
        });
    if (payload.field_counter != layout.num)
        jit_raise("While traversing and object for assigning inputs, the "
                  "number of variables to assign did not match the number "
                  "of variables traversed when recording!");
}

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

                layout.py_object = shape(h);
                layout.literal   = width(array);

                traverse(nb::steal(array), ctx);
            } else if (s.ndim != 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(h));

                layout.num  = len;

                for (Py_ssize_t i = 0; i < len; ++i)
                    traverse(nb::steal(s.item(h.ptr(), i)), ctx);
            } else {
                traverse_ad_var(h, ctx);
            }
        } else if (tp.is(&PyTuple_Type)) {
            nb::tuple tuple = nb::borrow<nb::tuple>(h);

            layout.num  = tuple.size();

            for (nb::handle h2 : tuple) {
                traverse(h2, ctx);
            }
        } else if (tp.is(&PyList_Type)) {
            nb::list list = nb::borrow<nb::list>(h);

            layout.num  = list.size();

            for (nb::handle h2 : list) {
                traverse(h2, ctx);
            }
        } else if (tp.is(&PyDict_Type)) {
            nb::dict dict = nb::borrow<nb::dict>(h);

            layout.num  = dict.size();
            layout.fields.reserve(layout.num);
            for (auto k : dict.keys()) {
                layout.fields.push_back(nb::borrow(k));
            }

            for (auto [k, v] : dict) {
                traverse(v, ctx);
            }
        } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {

            layout.num  = ds.size();
            layout.fields.reserve(layout.num);
            for (auto k : ds.keys()) {
                layout.fields.push_back(nb::borrow(k));
            }

            for (auto [k, v] : ds) {
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
                traverse(nb::getattr(h, k), ctx);
            }
        } else if (auto traversable = get_traversable_base(h); traversable) {
            traverse_cb(traversable, ctx, nb::borrow<nb::type_object>(tp));
        } else if (auto cb = get_traverse_cb_ro(tp); cb.is_valid()) {
            ProfilerPhase profiler("traverse cb");

            uint32_t num_fields = 0;
            std::vector<uint64_t> indices;
            indices.reserve(4096);

            // Traverse the opaque C++ object
            cb(h, nb::cpp_function([&](uint64_t index, const char *variant,
                                       const char *domain) {
                   if (!index)
                       return;
                   add_domain(variant, domain);
                   // jit_log(LogLevel::Debug, "traverse(): traverse_cb[%u] =
                   // a%u r%u",
                   //         num_fields, (uint32_t) (index >> 32), (uint32_t)
                   //         index);
                   num_fields++;
                   indices.push_back(index);
                   return;
               }));

            {
                ProfilerPhase p("indices");

                for (uint64_t index : indices) {
                    this->traverse_ad_index(index, ctx, nb::none());
                }
            }

            // Update layout number of fields
            this->layout[layout_index].num = num_fields;
        } else if (tp.is(&_PyNone_Type)) {
        } else {
            jit_log(LogLevel::Warn,
                    "traverse(): You passed a value to a frozen function, "
                    "that could not be converted to Dr.Jit types. This is "
                    "not recommended and the value will be cached.",
                    nb::type_name(tp).c_str());

            layout.py_object = nb::borrow<nb::object>(h);
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "FlatVariables::traverse(): error encountered while "
                       "processing an argument of type '%U' (see above).",
                       nb::type_name(tp).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "FlatVariables::traverse(): error encountered "
                        "while processing an argument of type '%U': %s",
                        nb::type_name(tp).ptr(), e.what());
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
                nb::object tensor = layout.type(array, layout.py_object);
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
            return list;
        } else if (layout.type.is(&PyDict_Type)) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return dict;
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
void FlatVariables::assign(nb::handle dst) {
    nb::handle tp  = dst.type();
    Layout &layout = this->layout[layout_index++];

    jit_log(LogLevel::Debug, "FlatVariables::assign(): %s with %s {",
            nb::type_name(tp).c_str(), nb::type_name(layout.type).c_str());

    if (!layout.type.equal(tp))
        jit_fail(
            "Type missmatch! Type of the object when recording (%s) does not "
            "match type of object that is assigned (%s).",
            nb::type_name(tp).c_str(), nb::type_name(layout.type).c_str());

    try {
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);

            if (s.is_tensor) {
                nb::handle array = s.tensor_array(dst.ptr());
                assign(nb::steal(array));
            } else if (s.ndim != 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(dst));

                for (Py_ssize_t i = 0; i < len; ++i)
                    assign(dst[i]);
            } else {
                assign_ad_var(layout, dst);
            }
        } else if (tp.is(&PyTuple_Type)) {
            nb::tuple tuple = nb::borrow<nb::tuple>(dst);
            raise_if(tuple.size() != layout.num, "");

            for (nb::handle h2 : tuple)
                assign(h2);
        } else if (tp.is(&PyList_Type)) {
            nb::list list = nb::borrow<nb::list>(dst);
            raise_if(list.size() != layout.num, "");

            for (nb::handle h2 : list)
                assign(h2);
        } else if (tp.is(&PyDict_Type)) {
            nb::dict dict = nb::borrow<nb::dict>(dst);
            for (auto &k : layout.fields) {
                if (dict.contains(&k))
                    assign(dict[k]);
                else
                    dst[k] = construct();
            }
        } else if (nb::dict ds = get_drjit_struct(dst); ds.is_valid()) {
            for (auto &k : layout.fields) {
                if (nb::hasattr(dst, k))
                    assign(nb::getattr(dst, k));
                else
                    nb::setattr(dst, k, construct());
            }
        } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()) {
            for (auto k : layout.fields) {
                if (nb::hasattr(dst, k))
                    assign(nb::getattr(dst, k));
                else
                    nb::setattr(dst, k, construct());
            }
        } else if (auto traversable = get_traversable_base(dst); traversable) {
            assign_cb(traversable);
        } else if (nb::object cb = get_traverse_cb_rw(tp); cb.is_valid()) {
            index64_vector tmp;
            uint32_t num_fields = 0;

            cb(dst, nb::cpp_function([&](uint64_t index) {
                   if (!index)
                       return index;
                   jit_log(LogLevel::Debug,
                           "assign(): traverse_cb[%u] was a%u r%u", num_fields,
                           (uint32_t) (index >> 32), (uint32_t) index);
                   num_fields++;
                   if (num_fields > layout.num)
                       jit_raise(
                           "While traversing the object of type %s "
                           "for assigning inputs, the number of variables "
                           "to assign (>%u) did not match the number of "
                           "variables traversed when recording(%u)!",
                           nb::str(tp).c_str(), num_fields, layout.num);
                   return assign_cb_internal(index, tmp);
               }));
            if (num_fields != layout.num)
                jit_raise("While traversing the object of type %s "
                          "for assigning inputs, the number of variables "
                          "to assign did not match the number of variables "
                          "traversed when recording!",
                          nb::str(tp).c_str());
        } else {
        }
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "FlatVariables::assign(): error encountered while "
                       "processing an argument "
                       "of type '%U' (see above).",
                       nb::type_name(tp).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError,
                        "FlatVariables::assign(): error encountered "
                        "while processing an argument "
                        "of type '%U': %s",
                        nb::type_name(tp).ptr(), e.what());
        nb::raise_python_error();
    }

    jit_log(LogLevel::Debug, "}");
}

/**
 * First traverses the PyTree, then the registry. This ensures that
 * additional data to vcalls is tracked correctly.
 */
void FlatVariables::traverse_with_registry(nb::handle h, TraverseContext &ctx) {

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

        std::vector<void *> registry_pointers;
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
void FlatVariables::assign_with_registry(nb::handle dst) {

    // Assign the handle
    assign(dst);

    // Assign registry (if a class variable was traversed)
    if (!domains.empty()) {
        Layout &layout      = this->layout[layout_index++];
        uint32_t num_fields = 0;

        jit_log(LogLevel::Debug, "registry{");

        std::vector<void *> registry_pointers;
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
                assign(self);

            assign_cb(traversable);
            num_fields++;
        }
        jit_log(LogLevel::Debug, "}");
    }
}

std::ostream &operator<<(std::ostream &os, const FlatVariables &r) {
    std::string offset = "    ";

    os << "FlatVariables[" << std::endl;

    std::string padding("    ");
    uint32_t index = 0;

    os << padding << "variables = [";
    for (uint64_t index : r.variables) {
        os << "r%u, ";
    }
    os << "]" << std::endl;

    os << padding << "sizes = [";
    for (uint64_t index : r.variables) {
        os << "%u, ";
    }
    os << "]" << std::endl;

    os << padding << "Layout[" << std::endl;

    padding.append("    ");
    log_layouts(r.layout, os, index, padding);
    padding.resize(padding.length() - 4);

    os << padding << "]" << std::endl;

    os << "]" << std::endl;
    return os;
}

void traverse_traversable(drjit::TraversableBase *traversable,
                          TraverseCallback &cb, bool rw = false) {
    struct Payload {
        TraverseCallback &cb;
    };
    Payload payload{ cb };
    if (rw) {
        traversable->traverse_1_cb_rw(
            (void *) &payload, [](void *p, uint64_t index) {
                Payload *payload = (Payload *) p;

                uint64_t new_index = payload->cb(index);
                return new_index;
            });
    } else {
        traversable->traverse_1_cb_ro(
            (void *) &payload,
            [](void *p, uint64_t index, const char *, const char *) {
                Payload *payload = (Payload *) p;
                payload->cb(index);
            });
    }
}

/**
 * Traverses the PyTree and all registry domains referenced in it.
 */
static void traverse_with_registry(const char *op, TraverseCallback &tc,
                                   nb::handle h, bool rw = false) {

    // Determine domains, that have to be traversed, by traversing in read only mode

    struct DomainTraverseCallback : TraverseCallback {

        std::string variant;
        std::vector<std::string> domains;

        void add_domain(const char *variant, const char *domain) {
            // Since it is not possible to pass nullptr strings to nanobind
            // functions we
            // assume, that a valid domain indicates a valid variant. If the
            // variant is emtpy at the end of traversal, we know that no Class
            // variable was traversed, and registry traversal is not necessary.
            if (domain && variant && strcmp(domain, "") != 0) {
                jit_log(LogLevel::Debug, "variant=%s, domain=%s", variant,
                        domain);

                if (domains.empty()) {
                    this->variant = variant;
                } else if (this->variant != variant)
                    jit_raise(
                        "traverse(): Variant missmatch! All arguments to a "
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

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_class && s.index) {
                auto variant = nb::borrow<nb::str>(nb::getattr(h, "Variant"));
                auto domain  = nb::borrow<nb::str>(nb::getattr(h, "Domain"));
                if (s.index)
                    operator()(s.index(inst_ptr(h)), variant.c_str(),
                               domain.c_str());

            } else if (s.index)
                operator()(s.index(inst_ptr(h)), nullptr, nullptr);
        }
        uint64_t operator()(uint64_t, const char *variant,
                            const char *domain) override {
            add_domain(variant, domain);
            return 0;
        }
    };
    DomainTraverseCallback domain_callback;

    traverse("domain traverse", domain_callback, h);

    // Traverse the registry
    if (!domain_callback.domains.empty()) {
        std::vector<void *> registry_pointers;
        for (std::string &domain : domain_callback.domains) {
            uint32_t registry_bound =
                jit_registry_id_bound(domain_callback.variant.c_str(), domain.c_str());
            uint32_t offset = registry_pointers.size();
            registry_pointers.resize(registry_pointers.size() + registry_bound, nullptr);
            jit_registry_get_pointers(domain_callback.variant.c_str(), domain.c_str(),
                                      &registry_pointers[offset]);
        }

        for (void *ptr : registry_pointers) {
            if (!ptr)
                continue;

            // WARN: very unsafe cast!
            // We assume, that any object added to the registry inherits from
            // TraversableBase. This is ensured by the signature of the
            // ``drjit::registry_put`` function.
            auto traversable = (drjit::TraversableBase *) ptr;
            auto self        = traversable->self_py();

            if (self)
                traverse(op, tc, self, rw);

            traverse_traversable(traversable, tc, rw);
        }
    }

    traverse(op, tc, h, rw);
}

/**
 * Schedules all variables in this PyTree, including the ones in C++ objects
 * traversable through the `traverse_1_cb_rw` methods. It uses
 * ``jit_var_schedule_force`` to force evaluation of literals. This function is
 * called before traversing the inputs and outputs of a frozen function. Inputs
 * and outputs have to be scheduled, since we use pointers to track variables,
 * so all variables have to be evaluated.
 *
 * \param eval
 *     If this boolean is set to ``true``, ``jit_eval`` is called if variables
 *     have been scheduled. If it is set to ``false``, we only schedule the
 *     variables.
 *
 * \param registry
 *     Boolean, indicating whether we should schedule the registry as well.
 */
static void deep_make_opaque(nb::handle h, bool eval = true,
                             bool registry = false) {
    jit_log(LogLevel::Debug, "make_opaque");

    struct ScheduleForceCallback : TraverseCallback {
        bool result = false;
        // NOTE: this is a really common pattern throughout my code, which could
        // be resolved by making the ``traverse_cb_rw`` steal the index and not
        // borrow it.
        index64_vector release_list;

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.index)
                s.reset_index(operator()(s.index(inst_ptr(h))), inst_ptr(h));
        }

        uint64_t operator()(uint64_t index, const char *variant = nullptr,
                            const char *domain = nullptr) override {
            if (!index)
                return index;
            uint64_t new_index;
            jit_log(LogLevel::Debug, "    schedule a%u, r%u",
                    (uint32_t) (index >> 32), (uint32_t) index);
            if (ad_grad_enabled(index)) {

                uint32_t grad = ad_grad(index);

                int rv    = 0;
                new_index = ad_var_schedule_force(index, &rv);
                if (rv) {
                    jit_log(LogLevel::Debug,
                            "   scheduled ad-variable a%u, r%u -> a%u, r%u",
                            (uint32_t) (index >> 32), (uint32_t) index,
                            (uint32_t) (new_index >> 32), (uint32_t) new_index);
                    jit_log(LogLevel::Debug, "    state=%u",
                            jit_var_state(new_index));
                    result = true;
                }

                rv                = 0;
                uint32_t new_grad = jit_var_schedule_force(grad, &rv);
                jit_var_dec_ref(grad);
                if (rv) {
                    jit_log(LogLevel::Debug,
                            "    scheduled gradient r%u -> r%u", grad,
                            new_grad);
                    jit_log(LogLevel::Debug, "    state=%u",
                            jit_var_state(new_grad));
                    result = true;
                }

                ad_clear_grad(new_index);
                ad_accum_grad(new_index, new_grad);
                jit_var_dec_ref(new_grad);
            } else {
                int rv    = 0;
                new_index = ad_var_schedule_force(index, &rv);
                if (rv) {
                    jit_log(LogLevel::Debug,
                            "   scheduled variable r%u, label=%s -> r%u",
                            (uint32_t) index, jit_var_label(index),
                            (uint32_t) new_index);
                    result = true;
                }
            }

            jit_log(LogLevel::Debug, "    return a%u, r%u",
                    (uint32_t) (new_index >> 32), (uint32_t) new_index);

            release_list.push_back_steal(new_index);
            return new_index;
        }
    };

    ScheduleForceCallback op;
    if (registry)
        traverse_with_registry("schedule_force", op, h, true);
    else
        traverse("schedule_force", op, h, true);

    if (op.result && eval) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

/**
 * Similarly to ``deep_make_opaque`` this function schedules all variables in
 * the PyTree, including the C++ objects. However, it uses ``jitc_var_schedule``
 * instead of ``jitc_ar_schedule_force``, not evaluating literals. This function
 * is used to evaluate the output of frozen functions.
 *
 * \param eval
 *     If this boolean is set to ``true``, ``jit_eval`` is called if variables
 *     have been scheduled. If it is set to ``false``, we only schedule the
 *     variables.
 *
 * \param registry
 *     Boolean, indicating whether we should schedule the registry as well.
 */
static void deep_eval(nb::handle h, bool eval = true) {
    jit_log(LogLevel::Debug, "deep eval");

    struct ScheduleCallback : TraverseCallback {
        bool result = false;
        // NOTE: this is a really common pattern throughout my code, which could
        // be resolved by making the ``traverse_cb_rw`` steal the index and not
        // borrow it.
        index64_vector release_list;

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.index)
                s.reset_index(operator()(s.index(inst_ptr(h)), nullptr,
                                         nullptr),
                              inst_ptr(h));
        }

        uint64_t operator()(uint64_t index, const char *variant,
                            const char *domain) override {
            if (ad_grad_enabled(index)) {
                int rv = 0;

                if (jit_var_schedule(index)) {
                    jit_log(LogLevel::Debug,
                            "   scheduled ad-variable a%u, r%u, label=%s",
                            (uint32_t) (index >> 32), (uint32_t) index,
                            jit_var_label(index));
                    result = true;
                }

                uint32_t grad = ad_grad(index);
                if (jit_var_schedule(grad)) {
                    jit_log(LogLevel::Debug,
                            "    scheduled gradient r%u, label=%s", grad,
                            jit_var_label(grad));
                    result = true;
                }
                jit_var_dec_ref(grad);

            } else {
                int rv = jit_var_schedule(index);
                if (rv) {
                    jit_log(LogLevel::Debug,
                            "   scheduled variable r%u, label=%s",
                            (uint32_t) index, jit_var_label(index));
                    result = true;
                }
            }
            ad_var_inc_ref(index);

            jit_log(LogLevel::Debug, "    scheduled a%u r%u",
                    (uint32_t) (index >> 32), (uint32_t) index);

            release_list.push_back_steal(index);
            return index;
        }
    };

    ScheduleCallback op;
    traverse("deep_eval", op, h, true);

    if (op.result && eval) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

std::ostream &operator<<(std::ostream &os, const RecordingKey &r) {
    std::string offset = "    ";

    os << "RecordingKey[" << std::endl;
    os << "    flags = " << r.flags << std::endl;

    std::string padding("    ");
    uint32_t index = 0;

    os << padding << "Layout[" << std::endl;

    padding.append("    ");
    log_layouts(r.layout, os, index, padding);
    padding.resize(padding.length() - 4);

    os << padding << "]" << std::endl;

    os << "]" << std::endl;
    return os;
}

inline void hash_combine(size_t &seed, size_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const size_t mult = 0x9ddfea08eb382d69ull;
    size_t a          = (value ^ seed) * mult;
    a ^= (a >> 47);
    size_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

size_t
RecordingKeyHasher::operator()(const std::shared_ptr<RecordingKey> &key) const {
    ProfilerPhase profiler("hash");
    // Hash the layout
    // NOTE: string hashing seems to be less efficient
    size_t hash = key->layout.size();
    for (const Layout &layout : key->layout) {
        hash_combine(hash, layout.num);
        hash_combine(hash, layout.fields.size());
        hash_combine(hash, (size_t) layout.vt);
        hash_combine(hash, (size_t) layout.vs);
        hash_combine(hash, (size_t) layout.flags);
        hash_combine(hash, (size_t) layout.literal);
        hash_combine(hash, (size_t) layout.index);
        hash_combine(hash, (size_t) layout.size_index);
        if (layout.type)
            hash_combine(hash, nb::hash(layout.type));
        if (layout.py_object)
            hash_combine(hash, nb::hash(layout.py_object));
        for (auto &field : layout.fields) {
            hash_combine(hash, nb::hash(field));
        }
    }

    hash_combine(hash, (size_t) key->flags);

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

    // output.append(result);
    // output.append(input);

    // Eval the input and output and it's gradients.
    jit_log(LogLevel::Debug, "Evaluating output:");
    {
        ProfilerPhase profiler("evaluate input + output");
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);
        {
            ProfilerPhase profiler("schedule input");
            deep_make_opaque(input, false, true);
        }
        {
            ProfilerPhase profiler("schedule output");
            deep_eval(output, false);
        }
        {
            nb::gil_scoped_release guard;
            jit_eval();
        }
    }

    // Pause recording before traversal as to not accidentally record
    // unwanted operations.
    jit_freeze_pause(backend);

    // TODO: validate, that gradients wheren't enabled for inputs inside the
    // frozen function.

    // Collect nodes, that have been postponed by the `Isolate` scope in a
    // hash set.
    // These are the targets of postponed edges, as the isolate gradient
    // scope only handles backward mode differentiation.
    // If they are, then we have to enqueue them when replaying the
    // recording.
    tsl::robin_set<uint32_t, UInt32Hasher> postponed;
    {
        drjit::vector<uint32_t> postponed_vec;
        ad_scope_postponed(postponed_vec);
        for (uint32_t index : postponed_vec)
            postponed.insert(index);
    }

    jit_log(LogLevel::Info, "Traversing output");
    {
        ProfilerPhase profiler("traverse output");
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);

        TraverseContext ctx;
        ctx.postponed = &postponed;
        out_variables.traverse(output, ctx);
        out_variables.traverse_with_registry(input, ctx);
    }

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

    recording = jit_freeze_stop(backend, out_variables.variables.data(),
                                out_variables.variables.size());

    jit_log(LogLevel::Info, "Recording done (n_outputs=%u)",
            out_variables.variables.size());

    // For catching input assignment mismatches, we assign the input and
    // output
    {
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);

        out_variables.layout_index = 0;
        jit_log(LogLevel::Debug, "Construct:");
        output = nb::borrow<nb::object>(out_variables.construct());
        // NOTE: temporarily disable this to not enqueue twice
        // jit_log(LogLevel::Debug, "Assign:");
        // out_variables.assign(input);
        out_variables.layout_index = 0;
    }

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
        jit_log(LogLevel::Warn, "re-recording");
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
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);
        out_variables.layout_index = 0;
        {
            ProfilerPhase profiler("construct output");
            output = nb::borrow<nb::object>(out_variables.construct());
        }
        {
            ProfilerPhase profiler("assign input");
            out_variables.assign_with_registry(input);
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
            jit_flag(JitFlag::FreezingScope)) {
            ProfilerPhase profiler("function");
            return func(*args, **kwargs);
        }

        nb::list input;
        input.append(args);
        input.append(kwargs);

        FlatVariables in_variables;
        // Evaluate and traverse input variables (args and kwargs)
        {
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, 0,
                                    true);
            // Evaluate input variables, forcing evaluation of undefined
            // variables
            {
                ProfilerPhase profiler("evaluate input");
                deep_make_opaque(input, true, true);
            }
            {
                nb::gil_scoped_release guard;
                jit_eval();
            }

            // Traverse input variables
            ProfilerPhase profiler("traverse input");
            jit_log(LogLevel::Debug, "freeze(): Traversing input.");
            TraverseContext ctx;
            in_variables.traverse_with_registry(input, ctx);
            // In order to prevent issues with scattering, we borrow all input
            // variables, incrementing their refcount.
            in_variables.borrow();
        }

        // for (uint32_t i = 0; i < 10000; i++){
        //     FlatVariables vars;
        //     TraverseContext ctx;
        //     vars.traverse_with_registry(input, ctx);
        //     vars.borrow();
        //     vars.release();
        //     // jit_log(LogLevel::Warn, "vars.layout.size()=%u",
        //     vars.layout.size());
        // }

        raise_if(in_variables.backend == JitBackend::None,
                 "freeze(): Cannot infer backend without providing input "
                 "variable to frozen function!");

        uint32_t flags = jit_flags();
        auto key       = std::make_shared<RecordingKey>(RecordingKey(std::move(in_variables.layout), flags));
        auto it        = this->recordings.find(key);

        if (it == this->recordings.end()) {
#ifndef NDEBUG
            if (this->recordings.size() >= 1) {
                jit_log(LogLevel::Info,
                        "Function input missmatch! Function will be retraced.");

                std::ostringstream repr;
                repr << key;

                std::ostringstream repr_prev;
                repr_prev << prev_key;

                jit_log(LogLevel::Debug, "new key: %s", repr.str().c_str());
                jit_log(LogLevel::Debug, "old key: %s",
                        repr_prev.str().c_str());
            }
#endif
            // FunctionRecording recording;
            auto recording = std::make_unique<FunctionRecording>();

            try {
                result = recording->record(func, this, input, in_variables);
            } catch (nb::python_error &e) {
                in_variables.release();
                jit_freeze_abort(in_variables.backend);
                nb::raise_from(
                    e, PyExc_RuntimeError,
                    "record(): error encountered while recording a frozen"
                    "function (see above).");
            } catch (const std::exception &e) {
                in_variables.release();
                jit_freeze_abort(in_variables.backend);

                nb::chain_error(PyExc_RuntimeError, "record(): %s", e.what());
                nb::raise_python_error();
            };

            in_variables.release();

            this->prev_key = key;
            this->recordings.insert({ std::move(key), std::move(recording) });

        } else {
            FunctionRecording *recording = it.value().get();

            {
                result = recording->replay(func, this, input, in_variables);
            }

            // Drop references to variables
            in_variables.release();
        }
    }
    ad_traverse(drjit::ADMode::Backward,
                (uint32_t) drjit::ADFlag::ClearVertices);
    return result;
}

void FrozenFunction::clear() {
    recordings.clear();
    prev_key          = std::make_shared<RecordingKey>(RecordingKey());
    recording_counter = 0;
}

FrozenFunction freeze(nb::callable func) { return FrozenFunction(func); }

void export_freeze(nb::module_ &m) {
    // m.def("freeze", &freeze, doc_freeze);
    nb::module_ d = nb::module_::import_("drjit.detail");
    nb::class_<FrozenFunction>(d, "FrozenFunction")
        .def(nb::init<nb::callable>())
        // .def("__get__",
        //      [](nb::object self, nb::object instance, nb::object) {
        //          if (instance.is_none()) {
        //              return self;
        //          } else {
        //              return nb::cpp_function(
        //                  [self, instance](nb::args args, nb::kwargs kwargs) {
        //                      return self(instance, *args, **kwargs);
        //                  },
        //                  nb::rv_policy::move);
        //          }
        //      })
        .def_prop_ro(
            "n_cached_recordings",
            [](FrozenFunction &self) { return self.saved_recordings(); })
        .def_ro("n_recordings", &FrozenFunction::recording_counter)
        .def("clear", &FrozenFunction::clear)
        .def("__call__", &FrozenFunction::operator());
}
