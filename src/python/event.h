#pragma once
#include "common.h"

template <JitBackend Backend>
class Event {
public:
    Event(bool enable_timing = true)
        : m_event(jit_event_create(Backend, enable_timing)) {}

    ~Event() {
        if (m_event)
            jit_event_destroy(m_event);
    }

    // Disable copy
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    // Enable move
    Event(Event&& other) noexcept : m_event(other.m_event) {
        other.m_event = nullptr;
    }

    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (m_event)
                jit_event_destroy(m_event);
            m_event = other.m_event;
            other.m_event = nullptr;
        }
        return *this;
    }

    void record() {
        jit_event_record(m_event);
    }

    bool query() {
        return jit_event_query(m_event) != 0;
    }

    void wait() {
        jit_event_wait(m_event);
    }

    float elapsed_time(const Event& end) {
        return jit_event_elapsed_time(m_event, end.m_event);
    }

    uintptr_t handle() const {
        return (uintptr_t) jit_event_handle(m_event);
    }

private:
    JitEvent m_event;
};


template <JitBackend Backend>
void bind_event(nb::module_ &m, const char* name) {
    using EventType = Event<Backend>;

    nb::class_<EventType>(m, name, doc_Event)
        .def(nb::init<bool>(), "enable_timing"_a = true, doc_Event_init)
        .def("record", &EventType::record, doc_Event_record)
        .def("query", &EventType::query, doc_Event_query)
        .def("wait", &EventType::wait, doc_Event_wait,
             nb::call_guard<nb::gil_scoped_release>())
        .def("elapsed_time", &EventType::elapsed_time, "end_event"_a,
             doc_Event_elapsed_time,
             nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("handle", &EventType::handle, doc_Event_handle);
}
