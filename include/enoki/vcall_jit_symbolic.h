/*
    enoki/vcall.h -- Vectorized method call support, via jump table

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

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
        uint32_t i = value.index();
        if (i == 0)
            jitc_fail("enoki::detail::read_indices(): uninitialized variable!");
        if (out)
            out[count] = i;
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
bool record(const char *domain, const char *name, uint32_t &id, uint64_t &hash,
            uint32_t *in, uint32_t *out, uint32_t *need_in, uint32_t *need_out,
            detail::ek_vector<uint32_t> &extra, Func func,
            const Args &... args) {
    using Result = decltype(func(args...));

    uint32_t se_before = jitc_side_effect_counter(IsCUDA);
    Result result = func(args...);
    uint32_t se_total = jitc_side_effect_counter(IsCUDA) - se_before;

    uint32_t in_count = 0, out_count = 0;
    (read_indices(in, in_count, args), ...);
    read_indices(out, out_count, result);

    uint32_t *extra_p = nullptr;
    uint32_t extra_count_p = 0;
    id = jitc_capture_var(IsCUDA, domain, name, in, in_count, out, out_count,
                          need_in, need_out, se_total, &hash, &extra_p,
                          &extra_count_p);

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
ENOKI_INLINE Result dispatch_jit_symbolic(const char *name, Func func, const Self &self, const Args&... args) {
    using Class = std::remove_pointer_t<scalar_t<Self>>;

    constexpr bool IsCUDA = is_cuda_array_v<Self>;

    jit_flag_guard guard;
    Result result = zero<Result>();

    // Determine # of existing instances, and preallocate memory for IR codegen
    uint32_t n_inst = jitc_registry_get_max(Class::Domain) + 1;

    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    read_indices(nullptr, out_count, result);

    detail::ek_unique_ptr<uint32_t[]> call_id(new uint32_t[n_inst]);
    detail::ek_unique_ptr<uint64_t[]> call_hash(new uint64_t[n_inst]);
    detail::ek_unique_ptr<uint32_t[]> extra_offset(new uint32_t[n_inst]);
    detail::ek_unique_ptr<uint32_t[]> in(new uint32_t[in_count]),
                                      need_in(new uint32_t[in_count]),
                                      out(new uint32_t[out_count]),
                                      need_out(new uint32_t[out_count]);

    detail::ek_vector<uint32_t> extra;
    bool side_effects = false;

    int need_init = (jitc_flags() & (uint32_t) JitFlag::OptimizeVCalls) ? 0 : 1;
    memset(need_in.get(), need_init, in_count * sizeof(uint32_t));
    memset(need_out.get(), need_init, out_count * sizeof(uint32_t));

    /* Call each instance symbolically and record. Do this twice
       so irrelevant parameters can be optimized away */
    for (uint32_t j = 0; j < 2; ++j) {
        for (uint32_t i = 0; i < n_inst; ++i) {
            Class *ptr = (Class *) jitc_registry_get_ptr(Class::Domain, i);

            extra_offset[i] = (uint32_t) (extra.size() * sizeof(void *));

            if (ptr)
                side_effects |= record<IsCUDA>(
                    Class::Domain, name, call_id[i], call_hash[i], in.get(),
                    out.get(), need_in.get(), need_out.get(), extra,
                    [&](const Args &... args) { return func(ptr, args...); },
                    placeholder<Args>(args)...);
            else
                record<IsCUDA>(
                    Class::Domain, name, call_id[i], call_hash[i], in.get(),
                    out.get(), need_in.get(), need_out.get(), extra,
                    [&](const Args &...) -> Result { return result; },
                    placeholder<Args>(args)...);
        }

        if (j == 0) {
            for (uint32_t i = 0; i < n_inst; ++i)
                jitc_var_dec_ref_ext(call_id[i]);
            for (uint32_t i = 0; i < extra.size(); ++i)
                jitc_var_dec_ref_ext(extra[i]);
            extra.clear();
        }
    }

    // Collect input + output arguments
    in_count = 0;
    (read_indices(in.get(), in_count, args), ...);
    out_count = 0;
    read_indices(out.get(), out_count, result);

    jitc_var_vcall(IsCUDA, Class::Domain, name, detach(self).index(), n_inst,
                   call_id.get(), call_hash.get(), in_count, in.get(),
                   out_count, out.get(), need_in.get(), need_out.get(),
                   (uint32_t) extra.size(), extra.data(), extra_offset.get(),
                   side_effects);

    out_count = 0;
    write_indices(out.get(), out_count, result);

    return result;
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
