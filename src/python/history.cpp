/*
    history.cpp -- bindings for the kernel history

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "history.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <memory>
#include <algorithm>

/// Shared owner of a Dr.Jit-Core kernel history snapshot. Entries hold a
/// reference so that temporary snapshots of an active capture stay alive.
using Snapshot = std::shared_ptr<KernelHistory>;

static Snapshot snapshot_new(KernelHistory *value) {
    return Snapshot(value, jit_kernel_history_free);
}

static void warn_legacy() {
    int rv = PyErr_WarnEx(
        PyExc_DeprecationWarning,
        "You are using the legacy kernel history interface (a manually managed "
        "JitFlag.KernelHistory flag combined with dictionary-style access to "
        "the query result). Please switch to the context manager form of "
        "drjit.kernel_history(), which is documented at "
        "https://drjit.readthedocs.io/en/latest/bench.html", 1);
    if (rv < 0)
        throw nb::python_error();
}

static void warn_renamed(const char *old_name, const char *new_name) {
    if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                         "drjit.KernelHistoryEntry.%s was renamed to '%s'. The "
                         "old name is deprecated and will be removed in a "
                         "future release.", old_name, new_name) < 0)
        throw nb::python_error();
}

/// Forwards the deprecated ``entry[key]`` syntax to attribute access. The
/// legacy interface exposed the kernel source code under the key 'ir', as a
/// file-like object.
static nb::object legacy_getitem(nb::handle self, nb::handle key) {
    warn_legacy();

    nb::str name = nb::str(key);
    bool is_ir = name.equal(nb::str("ir"));
    if (is_ir)
        name = nb::str("source");

    nb::object value = nb::steal(PyObject_GetAttr(self.ptr(), name.ptr()));
    if (!value.is_valid()) {
        PyErr_Clear();
        throw nb::key_error(nb::str(key).c_str());
    }

    if (is_ir) {
        nb::object io = nb::module_::import_("io").attr("StringIO");
        return io(value.is_none() ? nb::str("") : value);
    }

    return value;
}

// ============================================================================
//  Entries
// ============================================================================

/// Lazy view of one snapshot entry. Attribute reads query Dr.Jit-Core on
/// demand; fields that only JIT-compiled kernels provide read as neutral
/// defaults for other operation types.
struct Entry {
    using Field = KernelHistoryField;

    Snapshot snapshot;
    size_t index;

    Entry(Snapshot snapshot, size_t index)
        : snapshot(std::move(snapshot)), index(index) { }

    uint64_t query(Field field) const {
        return jit_kernel_history_query(snapshot.get(), index, field);
    }

    uint32_t query_u32(Field field) const { return (uint32_t) query(field); }

    /// Convert a nanosecond-valued field into milliseconds
    float query_ms(Field field) const {
        return (float) ((double) query(field) * 1e-6);
    }

    JitBackend backend() const { return (JitBackend) query_u32(Field::Backend); }
    KernelType type() const { return (KernelType) query_u32(Field::Type); }
    KernelRecordingMode recording_mode() const {
        return (KernelRecordingMode) query_u32(Field::RecordingMode);
    }
    uint32_t size() const { return query_u32(Field::Size); }
    uint32_t input_count() const { return query_u32(Field::InputCount); }
    uint32_t output_count() const { return query_u32(Field::OutputCount); }
    uint32_t operation_count() const { return query_u32(Field::OperationCount); }
    bool uses_optix() const { return query(Field::UsesOptix) != 0; }
    bool cache_hit() const { return query(Field::CacheHit) != 0; }
    bool cache_disk() const { return query(Field::CacheDisk) != 0; }
    float codegen_time() const { return query_ms(Field::CodegenTime); }
    float backend_time() const { return query_ms(Field::BackendTime); }

    /// Waits for the operation to finish
    float execution_time() const {
        nb::gil_scoped_release guard;
        return query_ms(Field::ExecutionTime);
    }

    std::optional<std::string> hash() const {
        if (type() != KernelType::JIT)
            return std::nullopt;
        char buf[33];
        snprintf(buf, sizeof(buf), "%016llx%016llx",
                 (unsigned long long) query(Field::HashHigh),
                 (unsigned long long) query(Field::HashLow));
        return std::string(buf);
    }

    std::optional<std::string> source() const {
        const char *value;
        {
            nb::gil_scoped_release guard;
            value = jit_kernel_history_source(snapshot.get(), index);
        }
        if (!value)
            return std::nullopt;
        return std::string(value);
    }

    std::string type_name() const {
        return nb::cast<std::string>(nb::cast(type()).attr("name"));
    }
};

// ============================================================================
//  Tabular rendering
// ============================================================================

static constexpr size_t n_cols = 11;

static const char *table_header[n_cols] = { "#",     "Type",    "Size",
                                            "In",    "Out",     "Ops",
                                            "Cache", "Codegen", "Compile",
                                            "Execute", "Hash" };

/// Columns holding text rather than numbers are aligned to the left
static bool align_left(size_t col) { return col == 1 || col == 6 || col == 10; }

/// Number of leading/trailing rows retained when abbreviating a long table
static constexpr size_t n_head = 30, n_tail = 9;

struct Row {
    bool ellipsis = false;
    std::string cells[n_cols];
};

static std::string time_str(float ms) {
    char buf[32];
    if (ms >= 1000)
        snprintf(buf, sizeof(buf), "%.3g s", ms / 1000);
    else if (ms >= 1)
        snprintf(buf, sizeof(buf), "%.3g ms", ms);
    else
        snprintf(buf, sizeof(buf), "%.3g µs", ms * 1000);
    return buf;
}

static Row make_row(size_t index, const Entry &e) {
    bool jit = e.type() == KernelType::JIT;

    std::string type = e.type_name();
    KernelRecordingMode rec = e.recording_mode();
    if (rec == KernelRecordingMode::Recorded)
        type += " [rec]";
    else if (rec == KernelRecordingMode::Replayed)
        type += " [rep]";

    std::optional<std::string> hash = e.hash();
    float backend_time = e.backend_time();

    Row r;
    r.cells[0]  = std::to_string(index);
    r.cells[1]  = type;
    r.cells[2]  = std::to_string(e.size());
    r.cells[3]  = std::to_string(e.input_count());
    r.cells[4]  = std::to_string(e.output_count());
    r.cells[5]  = jit ? std::to_string(e.operation_count()) : "-";
    r.cells[6]  = jit ? (e.cache_disk() ? "disk" : (e.cache_hit() ? "hit" : "miss")) : "-";
    r.cells[7]  = jit ? time_str(e.codegen_time()) : "-";
    r.cells[8]  = backend_time > 0 ? time_str(backend_time) : "-";
    r.cells[9]  = time_str(e.execution_time());
    r.cells[10] = hash ? hash->substr(0, 16) : "-";
    return r;
}

/// Width of a UTF-8 string in characters (the time column contains 'µ')
static size_t width(const std::string &s) {
    size_t result = 0;
    for (char c : s)
        result += ((uint8_t) c & 0xC0) != 0x80;
    return result;
}

// ============================================================================
//  History
// ============================================================================

struct History {
    /// Legacy: wraps launches that accumulated under a manually set
    /// JitFlag.KernelHistory; Active/Closed: 'with' block in progress/finished
    enum class Stage { Legacy, Active, Closed };

    Snapshot snapshot;
    std::optional<std::vector<KernelType>> types;
    nb::object cache;
    uint64_t start = 0;
    int flag = 0;
    Stage stage = Stage::Legacy;

    History(std::optional<std::vector<KernelType>> types)
        : types(std::move(types)) {
        // Legacy usage accumulates launches in the global log while the
        // JitFlag.KernelHistory flag is set by hand. Snapshot and clear the
        // log here, which also gives 'with' usage a clean slate.
        snapshot = snapshot_new(jit_kernel_history_view(0));
        jit_kernel_history_clear();
    }

    void enter() {
        if (stage == Stage::Active)
            nb::raise("drjit.kernel_history: this capture was already entered");

        stage = Stage::Active;
        snapshot.reset();
        cache.reset();
        flag = jit_flag(JitFlag::KernelHistory);
        jit_set_flag(JitFlag::KernelHistory, 1);
        start = jit_kernel_history_begin();
    }

    void exit() {
        jit_set_flag(JitFlag::KernelHistory, flag);
        snapshot = snapshot_new(jit_kernel_history_end(start));
        stage = Stage::Closed;
    }

    nb::list build(const Snapshot &s) const {
        nb::list result;
        for (size_t i = 0, n = jit_kernel_history_size(s.get()); i < n; ++i) {
            Entry e(s, i);
            if (types && std::find(types->begin(), types->end(), e.type()) ==
                             types->end())
                continue;
            result.append(nb::cast(std::move(e)));
        }
        return result;
    }

    /// Return the captured entries. A capture that is still in progress
    /// reports the launches recorded so far.
    nb::list entries() {
        if (stage == Stage::Active)
            return build(snapshot_new(jit_kernel_history_view(start)));
        if (stage == Stage::Legacy)
            warn_legacy();
        if (!cache.is_valid())
            cache = build(snapshot);
        return nb::borrow<nb::list>(cache);
    }

    std::vector<Row> rows(const nb::list &entries) const {
        size_t n = entries.size();
        std::vector<Row> result;

        for (size_t i = 0; i < n; ++i) {
            if (n > n_head + n_tail + 1 && i == n_head) {
                Row r;
                r.ellipsis = true;
                result.push_back(r);
                i = n - n_tail - 1;
                continue;
            }
            result.push_back(make_row(i, nb::cast<const Entry &>(entries[i])));
        }

        return result;
    }

    std::string header(const nb::list &entries) const {
        size_t n = entries.size();

        std::string result = "Kernel history (" + std::to_string(n) +
                             (n == 1 ? " entry" : " entries");
        if (n) {
            float total = 0.f;
            for (nb::handle h : entries)
                total += nb::cast<const Entry &>(h).execution_time();
            result += ", total device time: " + time_str(total);
        }
        return result + ")";
    }

    std::string repr() {
        nb::list entries = this->entries();
        std::string result = header(entries);
        if (entries.size() == 0)
            return result;

        std::vector<Row> rows = this->rows(entries);
        size_t widths[n_cols];
        for (size_t c = 0; c < n_cols; ++c) {
            widths[c] = width(table_header[c]);
            for (const Row &r : rows)
                if (!r.ellipsis)
                    widths[c] = std::max(widths[c], width(r.cells[c]));
        }

        auto put_row = [&](const std::string *cells) {
            std::string line;
            for (size_t c = 0; c < n_cols; ++c) {
                size_t pad = widths[c] - width(cells[c]);
                if (c)
                    line += "  ";
                if (align_left(c))
                    line += cells[c] + std::string(pad, ' ');
                else
                    line += std::string(pad, ' ') + cells[c];
            }
            while (!line.empty() && line.back() == ' ')
                line.pop_back();
            result += "\n" + line;
        };

        std::string cells[n_cols];
        for (size_t c = 0; c < n_cols; ++c)
            cells[c] = table_header[c];
        put_row(cells);
        for (size_t c = 0; c < n_cols; ++c)
            cells[c] = std::string(widths[c], '-');
        put_row(cells);

        for (const Row &r : rows) {
            if (r.ellipsis)
                result += "\n...";
            else
                put_row(r.cells);
        }

        return result;
    }
};

void export_history(nb::module_ &m) {
    nb::class_<Entry>(m, "KernelHistoryEntry", doc_KernelHistoryEntry)
        .def_prop_ro("backend", &Entry::backend,
                     doc_KernelHistoryEntry_backend)
        .def_prop_ro("type", &Entry::type,
                     doc_KernelHistoryEntry_type)
        .def_prop_ro("recording_mode", &Entry::recording_mode,
                     doc_KernelHistoryEntry_recording_mode)
        .def_prop_ro("size", &Entry::size,
                     doc_KernelHistoryEntry_size)
        .def_prop_ro("input_count", &Entry::input_count,
                     doc_KernelHistoryEntry_input_count)
        .def_prop_ro("output_count", &Entry::output_count,
                     doc_KernelHistoryEntry_output_count)
        .def_prop_ro("hash", &Entry::hash,
                     doc_KernelHistoryEntry_hash)
        .def_prop_ro("operation_count", &Entry::operation_count,
                     doc_KernelHistoryEntry_operation_count)
        .def_prop_ro("codegen_time", &Entry::codegen_time,
                     doc_KernelHistoryEntry_codegen_time)
        .def_prop_ro("backend_time", &Entry::backend_time,
                     doc_KernelHistoryEntry_backend_time)
        .def_prop_ro("uses_optix", &Entry::uses_optix,
                     doc_KernelHistoryEntry_uses_optix)
        .def_prop_ro("cache_hit", &Entry::cache_hit,
                     doc_KernelHistoryEntry_cache_hit)
        .def_prop_ro("cache_disk", &Entry::cache_disk,
                     doc_KernelHistoryEntry_cache_disk)
        .def_prop_ro("execution_time", &Entry::execution_time,
                     doc_KernelHistoryEntry_execution_time)
        .def_prop_ro("source", &Entry::source,
                     doc_KernelHistoryEntry_source)
        .def_prop_ro("ir", [](const Entry &e) {
            warn_renamed("ir", "source");
            return e.source();
        })
        .def("__getitem__", &legacy_getitem,
             nb::sig("def __getitem__(self, arg: str, /) -> object"))
        .def("__repr__", [](const Entry &e) {
            std::string result =
                "<KernelHistoryEntry: " + e.type_name() +
                ", size=" + std::to_string(e.size());
            if (std::optional<std::string> hash = e.hash())
                result += ", hash=" + hash->substr(0, 16);
            return result + ">";
        });

    nb::class_<History>(m, "kernel_history", doc_kernel_history)
        .def(nb::init<std::optional<std::vector<KernelType>>>(),
             "types"_a.none() = nb::none())
        .def("__enter__", [](History &h) -> History & { h.enter(); return h; },
             nb::rv_policy::none)
        .def("__exit__",
             [](History &h, nb::handle, nb::handle, nb::handle) { h.exit(); },
             nb::arg().none(), nb::arg().none(), nb::arg().none())
        .def("__len__", [](History &h) { return h.entries().size(); })
        .def("__getitem__",
             [](History &h, Py_ssize_t index) {
                 nb::object result = nb::steal(
                     PySequence_GetItem(h.entries().ptr(), index));
                 if (!result.is_valid())
                     throw nb::python_error();
                 return result;
             },
             nb::sig("def __getitem__(self, arg: int, /) -> KernelHistoryEntry"))
        .def("__getitem__",
             [](History &h, nb::slice slice) {
                 nb::object result = nb::steal(
                     PyObject_GetItem(h.entries().ptr(), slice.ptr()));
                 if (!result.is_valid())
                     throw nb::python_error();
                 return result;
             },
             nb::sig("def __getitem__(self, arg: slice, /) -> list[KernelHistoryEntry]"))
        // The list holds strong references to the entries, which in turn keep
        // the underlying snapshot alive
        .def("__iter__", [](History &h) { return nb::iter(h.entries()); },
             nb::sig("def __iter__(self, /) -> Iterator[KernelHistoryEntry]"))
        .def("__repr__", &History::repr);

    m.def("kernel_history_clear", &jit_kernel_history_clear,
          doc_kernel_history_clear);

    nb::enum_<KernelType>(m, "KernelType", doc_KernelType)
        .value("JIT", KernelType::JIT)
        .value("BlockReduce", KernelType::BlockReduce)
        .value("BlockPrefixReduce", KernelType::BlockPrefixReduce)
        .value("Dot", KernelType::Dot)
        .value("BatchedGemm", KernelType::BatchedGemm)
        .value("Compress", KernelType::Compress)
        .value("MkPerm", KernelType::MkPerm)
        .value("Memcpy", KernelType::Memcpy)
        .value("Memset", KernelType::Memset)
        .value("Poke", KernelType::Poke)
        .value("Aggregate", KernelType::Aggregate)
        .value("LLVMHostFunc", KernelType::LLVMHostFunc);

    nb::enum_<KernelRecordingMode>(m, "KernelRecordingMode",
                                   doc_KernelRecordingMode)
        .value("Inactive", KernelRecordingMode::Inactive)
        .value("Recorded", KernelRecordingMode::Recorded)
        .value("Replayed", KernelRecordingMode::Replayed);
}
