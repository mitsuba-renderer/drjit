/*
    drjit/string.h -- Simple and efficient buffer class for constructing
    strings incrementally. This is an alternative to std::string and
    std::ostringstream, which are fairly bloated.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array_utils.h>

NAMESPACE_BEGIN(drjit)

// Some forward declarations
template <typename T> bool schedule(const T &value);

namespace detail {
    template <bool Abbrev, typename Array, typename... Indices>
    void to_string(StringBuffer &buf, const Array &a, const size_t *shape, Indices... indices);
    template <typename T> bool put_shape(const T &array, size_t *shape);

    template <typename T> using has_c_str     = decltype(std::declval<T>().c_str());
    template <typename T> using has_to_string = decltype(to_string(std::declval<T>()));
};


/**
 * \brief Helper class for string construction
 *
 * This class provides a convenient wrapper around a heap-allocated string that
 * grows in response to append operations carried out via operations including
 * \ref StringBuffer::put() and \ref StringBuffer::fmt().
 */
struct StringBuffer {
    /// Default constructor, creates an empty buffer
    StringBuffer() = default;

    /// Preallocate memory for 'size' characters
    StringBuffer(size_t size) {
        m_start = (char *) malloc(size);
        m_end = m_start + size;
        clear();
    }

    /// Destructor, release the internal buffer
    ~StringBuffer() { free(m_start); }

    /// Move constructor
    StringBuffer(StringBuffer &&b) { swap(b); }

    /// Move assignment operator
    StringBuffer &operator=(StringBuffer &&b) {
        swap(b);
        return *this;
    }

    /// Disable copy construction
    StringBuffer(const StringBuffer &) = delete;

    /// Disable copy assignment
    StringBuffer &operator=(const StringBuffer &) = delete;

    /// Return the underlying null-terminated string buffer
    const char *get() { return m_start; }

    /// Reset back to the empty string
    void clear() {
        m_cur = m_start;
        if (m_start)
            m_start[0] = '\0';
    }

    /// Append a single character to the buffer
    StringBuffer &put(char c) {
        if (DRJIT_UNLIKELY(m_cur + 1 >= m_end))
            expand();

        *m_cur++ = c;
        *m_cur = '\0';
        return *this;
    }

    /// Append a string to the buffer
    StringBuffer &put(const char *str) {
        return put_str(str, strlen(str));
    }

    /// Append a boolean to the buffer
    StringBuffer &put(bool value) { return put(value ? '1' : '0'); }

    /// Append a pointer to the buffer
    template <typename T, enable_if_t<std::is_pointer_v<T>> = 0>
    StringBuffer &put(T ptr) {
        size_t value = size_t(ptr);
        const char *num = "0123456789abcdef";
        char buf[18];
        int i = 18;

        do {
            buf[--i] = num[value % 16];
            value /= 16;
        } while (value);

        buf[--i] = 'x';
        buf[--i] = '0';

        return put_str(buf + i, 18 - i);
    }

    /// Append an integral value
    template <typename T, enable_if_t<std::is_integral_v<T>> = 0>
    StringBuffer &put(T value_) {
        constexpr int Digits =
            (std::is_signed_v<T> ? 1 : 0) + int(sizeof(T) * 5 + 1) / 2;

        using UInt = std::make_unsigned_t<T>;
        bool neg = std::is_signed_v<T> && value_ < 0;

        UInt value = UInt(neg ? -int(value_) : value_);

        const char *num = "0123456789";
        char buf[Digits];
        int i = Digits;

        do {
            buf[--i] = num[value % 10];
            value /= 10;
        } while (value);

        if (neg)
            buf[--i] = '-';

        return put_str(buf + i, Digits - i);
    }

    /// Append a floating point value to the buffer
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    StringBuffer &put(T value) { return fmt("%.6g", (double) value); }

    /// Append an Dr.Jit array to the buffer
    template <typename T, enable_if_t<is_array_v<T>> = 0>
    StringBuffer &put(const T &value) {
        size_t shape[std::decay_t<T>::Depth + 1 /* avoid zero-sized array */ ] { };

        if (!detail::put_shape(value, shape)) {
            put("[ragged array]");
        } else {
            drjit::schedule(value.derived());
            detail::to_string<true>(*this, value.derived(), shape);
        }

        return *this;
    }

    /// Append a string (or similar class) that exposes a <tt>c_str()</tt> method
    template <typename T, enable_if_t<is_detected_v<detail::has_c_str, const T &>> = 0>
    StringBuffer &put(const T &value) { return put(value.c_str()); }

    /// Handle instances providing <tt>to_string(const T &)</tt> that can be found via ADL
    template <typename T,
              enable_if_t<!std::is_pointer_v<T> && !is_array_v<T> &&
                          is_detected_v<detail::has_to_string, const T &>> = 0>
    StringBuffer &put(const T &value) {
        return put(to_string(value));
    }

    /// Append nothing (no-op)
    StringBuffer &put() { return *this; }

    /// Append multiple values to the buffer
    template <typename T1, typename T2, typename... Ts>
    StringBuffer &put(const T1 &t1, const T2 &t2, const Ts &... ts) {
        return put(t1).put(t2).put(ts...);
    }

    /// Append a formatted (printf-style) string to the buffer
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
    StringBuffer &fmt(const char *format, ...) {
        do {
            size_t size = remain();
            va_list args;
            va_start(args, format);
            size_t written = (size_t) vsnprintf(m_cur, size, format, args);
            va_end(args);

            if (DRJIT_LIKELY(written + 1 < size)) {
                m_cur += written;
                return *this;
            }

            expand();
        } while (true);
    }

    /// Return the remaining space in bytes
    size_t remain() const { return m_end - m_cur; }

    /// Return the size of the string, excluding the null byte
    size_t size() const { return m_cur - m_start; }

    /// Swap contents with another StringBuffer
    void swap(StringBuffer &b) {
        std::swap(m_start, b.m_start);
        std::swap(m_cur, b.m_cur);
        std::swap(m_end, b.m_end);
    }

private:
    StringBuffer &put_str(const char *str, size_t size) {
        if (DRJIT_UNLIKELY(size >= remain()))
            expand(size + 1 - remain());

        memcpy(m_cur, str, size);
        m_cur += size;
        *m_cur = '\0';
        return *this;
    }

    DRJIT_NOINLINE void expand(size_t minval = 2) {
        size_t old_alloc_size = m_end - m_start,
               new_alloc_size = 2 * old_alloc_size + minval,
               used_size      = m_cur - m_start,
               used_size_p    = used_size + 1,
               copy_size      = used_size_p < old_alloc_size ?
                                used_size_p : old_alloc_size;

        char *tmp = (char *) malloc(new_alloc_size);
        memcpy(tmp, m_start, copy_size);
        free(m_start);

        m_start = tmp;
        m_end = m_start + new_alloc_size;
        m_cur = m_start + used_size;
    }

private:
    char *m_start = nullptr;
    char *m_cur = nullptr;
    char *m_end = nullptr;
};

NAMESPACE_END(drjit)
