/**
 * This file contains the frontend part of function freezing (``@dr.freeze()``).
 *
 * It provides Python bindings of the ``FrozenFunction`` class, which is not
 * used directly but rather via a higher-level wrapper in `__init__.py`.
 * A ``FrozenFunction`` represents a function annotated with `@dr.freeze()`. A
 * call to its `operator()` proceeds as follows:
 *
 * 1. If the wrapped callable was previously recorded, it walks through the
 *    ``Layout`` representing the prior input configuration. While doing so, it
 *    checks the current inputs for compatibility and binds detected variables.
 *    If the input is deemed compatible, it launches the kernel graph directly.
 *
 * 2. If no recording exists or the input is incompatible, it captures a new
 *    ``Layout`` characterizing the input configuration. When the "auto-opaque"
 *    feature is enabled, it may materialize literal input arrays into buffers.
 *    It then compares the layout against a potentially larger set of cached
 *    recordings, replays the match if found, or otherwise records.
 *
 * The implementation optimizes for the fast path in (1) where a single
 * recording is reused over and over again.
 *
 * The interaction with the backend involves the following functions:
 *
 * - ``jit_freeze_start()`` enters the recording mode, with the variables bound
 *   to the layout's slots as inputs.
 *
 * - ``jit_freeze_stop()`` ends the recording and registers its outputs.
 *
 * - ``jit_freeze_abort()`` discards a recording interrupted by an exception.
 *
 * - ``jit_freeze_dry_run()`` checks whether a recording can be replayed with
 *   the sizes of the current input. For example, block reductions may not be
 *   compatible. If not, the function is recorded again in place.
 *
 * - ``jit_freeze_replay()`` launches the recorded kernels and returns the
 *   outputs, from which the frontend constructs the result and writes modified
 *   inputs back into the input PyTree.
 *
 * - ``jit_freeze_destroy()`` releases a recording.
 */

#include "freeze.h"

#include <drjit-core/jit.h>
#include <drjit/autodiff.h>
#include <drjit/extra.h>
#include <nanobind/nanobind.h>
#include <exception>
#include "common.h"

/// Call ``func`` with the positional and keyword arguments of the call
static nb::object call_python(nb::handle func, nb::handle args,
                              nb::handle kwargs) {
    nb::object result =
        nb::steal(PyObject_Call(func.ptr(), args.ptr(), kwargs.ptr()));
    if (!result.is_valid())
        nb::raise_python_error();
    return result;
}

struct ADScopeContext {
    bool process_postponed;
    ADScopeContext(drjit::ADScope type, size_t size, const uint64_t *indices,
                   int symbolic, bool process_postponed)
        : process_postponed(process_postponed) {
        ad_scope_enter(type, size, indices, symbolic);
    }
    ~ADScopeContext() { ad_scope_leave(process_postponed); }
};

/// Aborts a recording when an exception unwinds the scope
struct freeze_abort_guard {
    JitBackend backend;
    bool armed = true;
    ~freeze_abort_guard() {
        if (armed)
            jit_freeze_abort(backend);
    }
};

nb::object FunctionRecording::record(nb::handle func, nb::handle root,
                                     const uint32_t *inputs) {
    JitBackend backend = layout->backend;
    uint32_t n_inputs  = (uint32_t) layout->slots.size();

    jit_freeze_start(backend, inputs, n_inputs);
    freeze_abort_guard abort_guard { backend };

    nb::object output;
    {
        ProfilerPhase profiler2("dr.freeze(): executing function");
        state_unlock_guard guard;
        output = call_python(func,
                             PyTuple_GetItem(root.ptr(), 0),
                             PyTuple_GetItem(root.ptr(), 1));
    }

    // Collect AD nodes postponed by the isolation scope.
    tsl::robin_set<uint32_t, UInt32Hasher> postponed;
    {
        drjit::vector<uint32_t> postponed_vec;
        ad_scope_postponed(&postponed_vec);
        for (uint32_t index : postponed_vec)
            postponed.insert(index);
    }

    // Describe the result and the input after the call
    SlotBindings out_bindings;
    {
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);
        out_layout = build_output_layout(output, root, layout->arg_names,
                                         backend, postponed, out_bindings);
    }

    drjit::vector<uint32_t> outputs =
        plan_outputs(out_layout, out_bindings.indices.data(), *layout, inputs);

    jit_freeze_pause(backend);

    // Exceptions raised by the recorded functions are re-raised here
    recording = jit_freeze_stop(backend, outputs.data(),
                                (uint32_t) outputs.size());
    abort_guard.armed = false;

    // Construct the result and write the input back right away, so that
    // errors surface when recording rather than when replaying
    {
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);
        output = construct_output(out_layout, outputs.data());
        update_input(out_layout, root, outputs.data());
    }

    return output;
}

bool FunctionRecording::dry_run(const uint32_t *inputs) {
    return jit_freeze_dry_run(recording, inputs) != 0;
}

nb::object FunctionRecording::replay(nb::handle root, const uint32_t *inputs) {
    drjit::detail::index32_vector out_values(out_layout.n_outputs);
    {
        state_unlock_guard guard;
        nb::gil_scoped_release guard2;
        jit_freeze_replay(recording, inputs, out_values.data());
    }

    nb::object output;
    {
        // Enter Resume scope, so we can track gradients
        ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1, false);
        output = construct_output(out_layout, out_values.data());
        update_input(out_layout, root, out_values.data());
    }

    return output;
}

nb::object FrozenFunction::operator()(nb::args args, nb::kwargs kwargs) {
    if (!enabled)
        return call_python(env.func, args, kwargs);

    ProfilerPhase profiler("drjit.freeze()");

    // Kernel freezing can be disabled with ``JitFlag::KernelFreezing``. Nested
    // calls to frozen functions are ignored and baked into the current
    // recording.
    bool freeze = jit_flag(JitFlag::KernelFreezing) &&
                  !jit_flag(JitFlag::FreezingScope) && max_cache_size != 0;

    // Call the function without recording it
    if (!freeze) {
        ProfilerPhase profiler2("drjit.freeze(): executing function");
        state_lock_guard guard;
        ADScopeContext ad_scope(drjit::ADScope::Isolate, 0, nullptr, -1, true);
        state_unlock_guard guard2;
        return call_python(env.func, args, kwargs);
    }

    // The input of the call: its arguments, the values of the globals and
    // closure variables that the function reads, and the optional state
    nb::object state =
        state_fn.is_none() ? nb::none() : call_python(state_fn, args, kwargs);
    nb::tuple root = nb::make_tuple(args, kwargs, env.capture(), state);

    state_lock_guard guard;
    nb::object result;
    {
        call_counter++;

        // RAII helper to release references in the SlotBindings
        struct release_bindings {
            SlotBindings &b;
            ~release_bindings() { b.release(); }
        } guard2 { bindings };

        // Fast path: verify the input against the recording used by the
        // previous call and replay it with the resulting slot bindings. The
        // Isolate scope keeps gradients from propagating outside of the
        // function, and ``call_slow()`` opens its own one.
        if (last_recording) {
            ADScopeContext ad_scope(drjit::ADScope::Isolate, 0, nullptr, -1,
                                    true);
            ADScopeContext ad_scope2(drjit::ADScope::Resume, 0, nullptr, 0,
                                     true);
            FunctionRecording *rec = last_recording;

            if (verify_layout(*rec->layout, root, scratch, bindings) &&
                rec->dry_run(bindings.indices.data())) {
                rec->last_used = call_counter;
                result = rec->replay(root, bindings.indices.data());
            }
        }

        if (!result.is_valid())
            result = call_slow(root);
    }
    ad_traverse(drjit::ADMode::Backward,
                (uint32_t) drjit::ADFlag::ClearVertices);
    return result;
}

std::shared_ptr<Layout> FrozenFunction::build_layout(nb::handle root) {
    // Enter Resume scope to track gradients
    ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, 0, true);

    // The layout of the previous recording flags the literals made opaque by
    // the auto-opaque feature so far, and the builder compares literals
    // against it to detect changing ones
    const Layout *prev = auto_opaque ? prev_layout.get() : nullptr;
    std::shared_ptr<Layout> layout =
        build_input_layout(root, env.arg_names, default_backend, prev,
                           !auto_opaque, bindings);

    // A previous layout with a different structure may have forced literals
    // at unrelated positions, in which case the layout is built again
    // without it
    if (prev && layout->nodes.size() != prev->nodes.size()) {
        bool forced = false;
        for (const Node &n : layout->nodes)
            forced |= n.has(NodeFlag::ForceOpaque);
        if (forced)
            layout = build_input_layout(root, env.arg_names, default_backend,
                                        nullptr, false, bindings);
    }

    raise_if(layout->backend == JitBackend::None,
             "drjit.freeze(): Cannot infer backend without providing input "
             "variable to frozen function!");

    return layout;
}

void FrozenFunction::warn_recordings(const Layout &layout, bool retried,
                                     bool evicted) {
    if (recording_counter <= warn_recording_count || recording_counter < 2)
        return;

    if (retried) {
        jit_log(LogLevel::Warn,
                "This frozen function was traced %u times. A recorded "
                "operation could not handle the sizes of the current "
                "arguments, so its recording was overwritten by a new "
                "trace. This happens for example when a block reduction "
                "receives an input whose size is not divisible by the "
                "block size.",
                recording_counter);
        return;
    }

    // Name the input that differs from the one of the previous recording
    std::string reason;
    if (prev_layout)
        reason = layout_diff(layout, *prev_layout);
    if (reason.empty())
        reason = "unknown";

    // A full cache had to evict a recording to make room for this one, so
    // that the two keep replacing each other
    if (evicted)
        jit_log(LogLevel::Warn,
                "This frozen function was traced %u times, while its cache "
                "holds at most %i recording(s) (``limit=%i`` was passed to "
                "@dr.freeze). Recordings are therefore evicted and traced "
                "again. Repeated tracing defeats the purpose of function "
                "freezing and is caused by structural changes of the "
                "function's inputs. The change that triggered this recording "
                "was: %s.",
                recording_counter, max_cache_size, max_cache_size,
                reason.c_str());
    else
        jit_log(LogLevel::Warn,
                "This frozen function was traced %u times. Repeated tracing "
                "defeats the purpose of function freezing and is caused by "
                "structural changes of the function's inputs. The change that "
                "triggered this recording was: %s.",
                recording_counter, reason.c_str());
}

nb::object FrozenFunction::call_slow(nb::handle root) {
    // Detect structural changes in the input and potentially
    // try recording once more
    for (bool retried = false;; retried = true) {
        ADScopeContext ad_scope(drjit::ADScope::Isolate, 0, nullptr, -1, true);

        std::shared_ptr<Layout> layout = build_layout(root);
        const uint32_t *inputs = bindings.indices.data();

        // Drops the cache entry at position ``i``
        auto erase = [&](size_t i) {
            if (recordings[i].get() == last_recording)
                last_recording = nullptr;
            recordings[i] = std::move(recordings.back());
            recordings.pop_back();
        };

        // Replay the cached recording if its dry run succeeds. Otherwise,
        // drop the cache entry and record once more.
        bool retry = false;
        for (size_t i = 0; i < recordings.size(); ++i) {
            FunctionRecording *rec = recordings[i].get();
            if (!layout_equal(*rec->layout, *layout))
                continue;
            if (rec->dry_run(inputs)) {
                rec->last_used = call_counter;
                last_recording = rec;
                return rec->replay(root, inputs);
            }
            jit_log(LogLevel::Info, "drjit.freeze(): dry run failed, re-recording.");
            erase(i);
            retry = true;
            break;
        }

        // Evict the least recently used recording if the cache is "full"
        bool evicted =
            max_cache_size > 0 && recordings.size() >= (uint32_t) max_cache_size;
        if (evicted) {
            size_t lru = 0;
            for (size_t i = 1; i < recordings.size(); ++i)
                if (recordings[i]->last_used < recordings[lru]->last_used)
                    lru = i;
            erase(lru);
        }

        auto rec = std::make_unique<FunctionRecording>();
        rec->last_used = call_counter;
        rec->layout    = layout;

        recording_counter++;
        warn_recordings(*layout, retry, evicted);

        nb::object result;
        try {
            result = rec->record(env.func, root, inputs);
        } catch (const InputChanged &e) {
            if (retried)
                nb::raise("drjit.freeze(): %s, and did so again when it was "
                          "recorded a second time. A frozen function may assign "
                          "new arrays to its input, but it must not change the "
                          "structure of the input or the Python values it holds, "
                          "since a replay cannot reproduce such a change.",
                          e.what());
            jit_log(LogLevel::Info, "drjit.freeze(): %s, re-recording.", e.what());

            // Restore gradient state
            bindings.restore_grads();
            ad_scope.process_postponed = false;
            continue;
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_RuntimeError,
                           "drjit.freeze(): error encountered while recording a frozen "
                           "function (see above).");
        } catch (const std::exception &e) {
            nb::chain_error(PyExc_RuntimeError,
                            "drjit.freeze(): error encountered while recording a frozen "
                            "function: %s", e.what());
            nb::raise_python_error();
        }

        prev_layout = layout;
        last_recording = rec.get();
        recordings.push_back(std::move(rec));

        return result;
    }
}

void FrozenFunction::clear() {
    last_recording = nullptr;
    bindings.release();
    recordings.clear();
    prev_layout.reset();
    recording_counter = 0;
    call_counter      = 0;
}

/// Offset of the instance dictionary (``nb::dynamic_attr()``)
static Py_ssize_t dict_offset = 0;

static PyObject **inst_dict_ptr(PyObject *self) {
    return (PyObject **) ((char *) self + dict_offset);
}

/// Enable Python's GC to collect reference cycles involving frozen functions
int frozen_function_tp_traverse(PyObject *self, visitproc visit,
                                void *arg) noexcept {
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(*inst_dict_ptr(self));

    // The C++ constructor may not have run yet
    if (!nb::inst_ready(self))
        return 0;

    FrozenFunction *f = nb::inst_ptr<FrozenFunction>(self);

    int rv = f->env.tp_traverse(visit, arg);
    if (rv)
        return rv;
    Py_VISIT(f->state_fn.ptr());

    for (auto &rec : f->recordings) {
        rv = rec->layout->tp_traverse(visit, arg);
        if (rv)
            return rv;
        rv = rec->out_layout.tp_traverse(visit, arg);
        if (rv)
            return rv;
    }

    if (f->prev_layout.use_count() == 1)
        return f->prev_layout->tp_traverse(visit, arg);

    return 0;
}

/// Enable Python's GC to clear reference cycles involving frozen functions
int frozen_function_tp_clear(PyObject *self) noexcept {
    Py_CLEAR(*inst_dict_ptr(self));

    if (!nb::inst_ready(self))
        return 0;

    FrozenFunction *f = nb::inst_ptr<FrozenFunction>(self);
    f->clear();
    f->env.tp_clear();
    f->state_fn.reset();
    return 0;
}

/// ``tp_descr_get`` slot. When ``@dr.freeze`` decorates a method, accessing
/// the frozen function through an instance must produce a bound method, just
/// as accessing a plain function would. This reimplements the ``__get__``
/// behavior of Python functions using ``types.MethodType``.
PyObject *frozen_function_descr_get(PyObject *self, PyObject *obj,
                                    PyObject *) noexcept {
    if (!obj || obj == Py_None)
        return Py_NewRef(self);
    return PyObject_CallFunctionObjArgs(
        lazy_import(LazyImport::TypesMethodType).ptr(), self, obj, nullptr);
}

PyObject *frozen_function_tp_call(PyObject *self, PyObject *args,
                                  PyObject *kwargs) noexcept {
    FrozenFunction *f = nb::inst_ptr<FrozenFunction>(self);
    try {
        // Python passes no dictionary when the call site has no keyword
        // arguments, while both the layout and ``PyObject_Call()`` want one
        nb::kwargs kwargs_o = kwargs ? nb::borrow<nb::kwargs>(kwargs)
                                     : nb::kwargs();
        return (*f)(nb::borrow<nb::args>(args), std::move(kwargs_o))
            .release().ptr();
    } catch (nb::python_error &e) {
        e.restore();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "FrozenFunction.__call__(): uncaught exception!");
    }
    return nullptr;
}

// Python type slots for frozen functions
static PyType_Slot slots[] = {
    { Py_tp_traverse, (void *) frozen_function_tp_traverse },
    { Py_tp_clear, (void *) frozen_function_tp_clear },
    { Py_tp_descr_get, (void *) frozen_function_descr_get },
    { Py_tp_call, (void *) frozen_function_tp_call },
    { 0, nullptr }
};

void export_freeze(nb::module_ &m) {
    // The instance dictionary lets ``functools.wraps()`` copy the metadata
    // of the decorated function (``__name__``, ``__doc__``, ``__wrapped__``, ..)
    nb::class_<FrozenFunction> cls(m, "FrozenFunction", nb::type_slots(slots),
                                   nb::dynamic_attr(),
                                   nb::is_weak_referenceable(),
                                   doc_FrozenFunction);
    cls.def(nb::init<nb::callable, nb::object, int, uint32_t, JitBackend, bool,
                     bool>(),
            "func"_a, "state_fn"_a.none(), "limit"_a, "warn_after"_a,
            "backend"_a, "auto_opaque"_a, "enabled"_a)
       .def_prop_ro(
           "n_cached_recordings",
           [](FrozenFunction &self) { return self.recordings.size(); })
       .def_ro("n_recordings", &FrozenFunction::recording_counter)
       .def_rw("enabled", &FrozenFunction::enabled)
       .def("clear", &FrozenFunction::clear)
       .freeze();

    dict_offset = nb::cast<Py_ssize_t>(cls.attr("__dictoffset__"));
}
