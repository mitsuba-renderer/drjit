/*
    enoki/vcall.h -- Vectorized method call support, via jump table

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

extern "C" {
    extern ENOKI_IMPORT uint32_t jitc_registry_get_max(const char *domain);
    extern ENOKI_IMPORT uint32_t jitc_capture_var(
        int cuda, const uint32_t *in, uint32_t n_in, const uint32_t *out,
        uint32_t n_out, uint32_t n_side_effects, uint64_t *hash_out,
        uint32_t **extra_out, uint32_t *extra_count_out);
    extern ENOKI_IMPORT uint32_t jitc_side_effect_counter(int cuda);
};

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <typename T>
void read_indices(uint32_t *out, uint32_t &count, const T &value) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            read_indices(out, count, value.derived().entry(i));
    } else if constexpr (is_diff_array_v<T>) {
        read_indices(out, count, value.detach_());
    } else if constexpr (is_jit_array_v<T>) {
        if (out)
            out[count] = value.index();
        count += 1;
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](const auto &x) { read_indices(out, count, x); });
    }
}

template <typename T>
void write_indices(uint32_t *out, uint32_t &count, T &value) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            write_indices(out, count, value.derived().entry(i));
    } else if constexpr (is_diff_array_v<T>) {
        write_indices(out, count, value.detach_());
    } else if constexpr (is_jit_array_v<T>) {
        value = T::steal(out[count++]);
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](auto &x) { write_indices(out, count, x); });
    }
}

template <bool IsCUDA, typename Func, typename... Args>
bool record(uint32_t &id, uint64_t &hash, detail::ek_vector<uint32_t> &extra,
            Func func, const Args &... args) {
    using Result       = decltype(func(args...));
    uint32_t se_before = jitc_side_effect_counter(IsCUDA);

    Result result = func(args...);

    uint32_t se_total = jitc_side_effect_counter(IsCUDA) - se_before;

    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    read_indices(nullptr, out_count, result);

    detail::ek_unique_ptr<uint32_t[]> in(new uint32_t[in_count]),
                                       out(new uint32_t[out_count]);

    in_count = 0, out_count = 0;
    (read_indices(in.get(), in_count, args), ...);
    read_indices(out.get(), out_count, result);

    uint32_t *extra_p = nullptr;
    uint32_t extra_count_p = 0;
    id = jitc_capture_var(IsCUDA, in.get(), in_count, out.get(), out_count,
                          se_total, &hash, &extra_p, &extra_count_p);

    for (uint32_t i = 0; i < extra_count_p; ++i)
        extra.push_back(extra_p[i]);

    return se_total != 0;
}

struct jit_flag_guard {
public:
    jit_flag_guard() : flags(jitc_flags()) {
        jitc_set_flags(flags | (uint32_t) JitFlag::RecordingVCall);
    }
    ~jit_flag_guard() { jitc_set_flags(flags); }

private:
    uint32_t flags;
};

template <typename Result, typename Func, typename Self, typename... Args>
ENOKI_INLINE Result dispatch_jit_symbolic(Func func, const Self &self, const Args&... args) {
    using Class = std::remove_pointer_t<scalar_t<Self>>;

    constexpr bool IsCUDA = is_cuda_array_v<Self>;

    jit_flag_guard guard;
    Result result = zero<Result>();

    // Determine # of existing instances, and preallocate memory for IR codegen
    uint32_t n_inst = jitc_registry_get_max(Class::Domain) + 1;

    detail::ek_unique_ptr<uint32_t[]> call_id(new uint32_t[n_inst]);
    detail::ek_unique_ptr<uint64_t[]> call_hash(new uint64_t[n_inst]);
    detail::ek_unique_ptr<uint32_t[]> extra_offset(new uint32_t[n_inst]);
    detail::ek_vector<uint32_t> extra;
    bool side_effects = false;

    // Call each instance symbolically and record!
    for (uint32_t i = 0; i < n_inst; ++i) {
        Class *ptr = (Class *) jitc_registry_get_ptr(Class::Domain, i);

        extra_offset[i] = (uint32_t) (extra.size() * sizeof(void *));

        if (ptr)
            side_effects |= record<IsCUDA>(
                call_id[i], call_hash[i], extra,
                [&](const Args &... args) { return func(ptr, args...); },
                placeholder<Args>(args)...);
        else
            side_effects |= record<IsCUDA>(
                call_id[i], call_hash[i], extra,
                [&](const Args &...) { return result; },
                placeholder<Args>(args)...);
    }

    // Collect input arguments
    uint32_t in_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    detail::ek_unique_ptr<uint32_t[]> in(new uint32_t[in_count]);
    in_count = 0;
    (read_indices(in.get(), in_count, args), ...);

    // Collect output arguments
    uint32_t out_count = 0;
    read_indices(nullptr, out_count, result);
    detail::ek_unique_ptr<uint32_t[]> out(new uint32_t[out_count]);
    out_count = 0;
    read_indices(out.get(), out_count, result);

    jitc_var_vcall(IsCUDA, detach(self).index(), n_inst, call_id.get(),
                   call_hash.get(), in_count, in.get(), out_count, out.get(),
                   (uint32_t) extra.size(), extra.data(),
                   extra_offset.get(), side_effects);

    out_count = 0;
    write_indices(out.get(), out_count, result);

    return result;
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
