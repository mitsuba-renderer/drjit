/*
    drjit/unique_buffer.h -- RAII wrapper around a jit_malloc() allocation

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <utility>

NAMESPACE_BEGIN(drjit)

/**
 * \brief Owning, move-only wrapper around a raw ``jit_malloc()`` allocation
 *
 * This helper manages a typed memory buffer allocated through Dr.Jit's
 * caching allocator. It is useful for host-side staging storage that is
 * later either released via ``jit_free()`` (which correctly defers the
 * release past enqueued asynchronous operations) or handed over to a JIT
 * variable via \ref release() (e.g. through ``jit_var_mem_map()`` with
 * ownership transfer).
 *
 * The ``JitBackend::None`` backend provides plain host memory that behaves
 * like ``malloc()`` and requires no initialized JIT backend, so the wrapper
 * is also usable in purely scalar code.
 */
template <typename T> struct unique_buffer {
    unique_buffer() = default;

    /// Allocate ``size`` elements on the given backend. When ``shared`` is
    /// set, the buffer is host-writable immediately (see ``jit_malloc()``).
    unique_buffer(JitBackend backend, size_t size, bool shared = false)
        : m_data((T *) jit_malloc(backend, size * sizeof(T), shared ? 1 : 0)),
          m_size(size) { }

    unique_buffer(const unique_buffer &) = delete;
    unique_buffer &operator=(const unique_buffer &) = delete;

    unique_buffer(unique_buffer &&other) noexcept
        : m_data(other.m_data), m_size(other.m_size) {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    unique_buffer &operator=(unique_buffer &&other) noexcept {
        reset();
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
        return *this;
    }

    ~unique_buffer() { reset(); }

    /// Free the buffer (deferred past enqueued work on JIT backends)
    void reset() {
        if (m_data) {
            jit_free(m_data);
            m_data = nullptr;
            m_size = 0;
        }
    }

    /// Give up ownership; the caller becomes responsible for ``jit_free()``
    T *release() {
        T *tmp = m_data;
        m_data = nullptr;
        m_size = 0;
        return tmp;
    }

    T *data() { return m_data; }
    const T *data() const { return m_data; }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }
    explicit operator bool() const { return m_data != nullptr; }

    T &operator[](size_t i) { return m_data[i]; }
    const T &operator[](size_t i) const { return m_data[i]; }

private:
    T *m_data = nullptr;
    size_t m_size = 0;
};

NAMESPACE_END(drjit)
