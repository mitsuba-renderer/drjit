/*
    freeze.h -- Bindings for drjit.freeze()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include "freeze_layout.h"
#include "funcenv.h"
#include <drjit-core/jit.h>
#include <drjit/autodiff.h>
#include <memory>

/// A recording of a frozen function, made with a certain input layout
struct FunctionRecording {
    /// Value of ``FrozenFunction::call_counter`` when this recording was
    /// last recorded or replayed (used to evict the least recently used one)
    uint32_t last_used = 0;

    /// The backend recording, held by this wrapper
    Recording *recording = nullptr;

    /// Description of the input with which this recording was made. It
    /// serves as the cache key. The reference is shared with
    /// ``FrozenFunction::prev_layout``, which can outlive the recording when
    /// it is dropped or evicted from the cache.
    std::shared_ptr<Layout> layout;

    /// Description of the function result and of the input after the call,
    /// from which the result is constructed and modified inputs are written
    /// back.
    Layout out_layout;

    FunctionRecording() = default;
    FunctionRecording(const FunctionRecording &) = delete;
    FunctionRecording &operator=(const FunctionRecording &) = delete;
    ~FunctionRecording() {
        if (recording)
            jit_freeze_destroy(recording);
    }

    /// Record ``func`` with the given input, whose slot bindings were computed
    /// with ``layout``, and return its result. Throws ``InputChanged`` when
    /// the call changed the structure of its input (see ``plan_outputs()``).
    nb::object record(nb::handle func, nb::handle root, const uint32_t *inputs);

    /// Check that the recording can be replayed with the given input slots
    bool dry_run(const uint32_t *inputs);

    /// Replay the recording with the given input slots, construct the output,
    /// and update the input
    nb::object replay(nb::handle root, const uint32_t *inputs);
};

struct FrozenFunction {
    /// The frozen callable and everything that was read from it
    FunctionEnvironment env;

    /// Optional callable that produces additional input state from the
    /// arguments of a call
    nb::object state_fn;

    /// When disabled, calls are forwarded to ``func`` without freezing
    bool enabled = true;

    /// List of recordings made so far
    drjit::vector<std::unique_ptr<FunctionRecording>> recordings;

    /// The layout of the most recent recording. The auto-opaque feature
    /// compares new layouts against it to detect literals that change between
    /// calls, and its ``ForceOpaque`` flags mark the literals made opaque so
    /// far.
    std::shared_ptr<Layout> prev_layout;

    /// The number of times this function has been recorded. Note, this can
    /// differ from the number of recordings actually cached in \c recordings,
    /// when dry running recordings failed.
    uint32_t recording_counter    = 0;

    /// A counter, incremented whenever this function is called. It is used to
    /// determine the least recently used recording in order to evict it if the
    /// \c max_cache_size is set.
    uint32_t call_counter         = 0;

    /// Maximum number of recordings that should be made before evicting the
    /// least recently used one. If this value is -1, recordings can be made
    /// without limit.
    int max_cache_size            = -1;

    /// The number of recordings after which a warning message will be
    /// displayed. This is useful to detect cases in which changing Python
    /// values prevents replay.
    uint32_t warn_recording_count = 10;

    /// If no JIT variable inputs are given to the function, this can indicate a
    /// default backend, on which the function is recorded and replayed.
    JitBackend default_backend    = JitBackend::None;

    /// Whether the auto opaque feature is enabled. It allows us find literal
    /// values that change between calls to the frozen function, and selectively
    /// make those opaque.
    bool auto_opaque = true;

    /// The recording that was used by the most recent call. Its layout is
    /// verified against the input of the next call before the cache is
    /// consulted.
    FunctionRecording *last_recording = nullptr;

    /// Working memory of the layout verifier, reused across calls
    VerifierScratch scratch;

    /// Slot bindings of the current call, whether verified or built. Reused
    /// across calls and released once the call is done.
    SlotBindings bindings;

    FrozenFunction(nb::callable func, nb::object state_fn, int max_cache_size,
                   uint32_t warn_recording_count, JitBackend backend,
                   bool auto_opaque, bool enabled)
        : env(func), state_fn(state_fn),
          enabled(enabled), max_cache_size(max_cache_size),
          warn_recording_count(warn_recording_count), default_backend(backend),
          auto_opaque(auto_opaque) { }

    FrozenFunction(const FrozenFunction &) = delete;
    FrozenFunction &operator=(const FrozenFunction &) = delete;

    /// Clears the frozen function recordings and resets the counters.
    void clear();

    /// Call the frozen function with the arguments of a Python call, either
    /// recording a new version or replaying an old one. The input that the
    /// layouts describe is the tuple ``(args, kwargs, closure, state)``,
    /// where ``closure`` holds the values of the globals and closure
    /// variables (see ``FunctionEnvironment::capture()``) and ``state`` is the
    /// result of ``state_fn`` (or ``None``). Calls arrive via a ``tp_call``
    /// type slot, which skips the generic nanobind method dispatch.
    nb::object operator()(nb::args args, nb::kwargs kwargs);

    /// Build the layout of the input, making literals opaque according to
    /// the auto-opaque feature. ``bindings`` receives the slot bindings.
    std::shared_ptr<Layout> build_layout(nb::handle root);

    /// Look up the recording for the input via the cache, replaying or
    /// recording as needed. A callable that changes the structure of its
    /// input while it is recorded is recorded once more.
    nb::object call_slow(nb::handle root);

    /// Warn when the function is recorded more often than expected.
    /// ``retried`` marks a re-recording after a failed dry run, and
    /// ``evicted`` that a full cache had to drop a recording for this one.
    void warn_recordings(const Layout &layout, bool retried, bool evicted);
};

extern void export_freeze(nb::module_ &);
