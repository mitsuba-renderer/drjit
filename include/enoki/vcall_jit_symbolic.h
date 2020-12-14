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
    extern ENOKI_IMPORT uint32_t jitc_eval_ir_var(int cuda,
                                                  const uint32_t *in, uint32_t n_in,
                                                  const uint32_t *out, uint32_t n_out,
                                                  uint32_t n_side_effects,
                                                  uint64_t *hash_out);
    extern ENOKI_IMPORT uint32_t jitc_side_effect_counter(int cuda);
};

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <typename T>
void read_indices(uint32_t *out, const T &value, uint32_t &count) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            read_indices(out, value.derived().entry(i), count);
    } else if constexpr (is_diff_array_v<T>) {
        read_indices(out, value.detach_(), count);
    } else if constexpr (is_jit_array_v<T>) {
        if (out)
            out[count] = value.index();
        count += 1;
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](const auto &x) {
                 read_indices(out, x, count);
            });
    }
}

template <typename T>
void write_indices(uint32_t *out, T &value, uint32_t &count) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            write_indices(out, value.derived().entry(i), count);
    } else if constexpr (is_diff_array_v<T>) {
        write_indices(out, value.detach_(), count);
    } else if constexpr (is_jit_array_v<T>) {
        value = T::steal(out[count++]);
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](auto &x) {
                 write_indices(out, x, count);
            });
    }
}

template <bool IsCUDA, typename Func, typename... Args>
std::pair<uint32_t, uint64_t> record(Func func, const Args&... args) {
    using Result = decltype(func(args...));
    uint32_t se_before = jitc_side_effect_counter(IsCUDA);

    Result result = func(args...);

    uint32_t se_total = jitc_side_effect_counter(IsCUDA) - se_before;

    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, args, in_count), ...);
    read_indices(nullptr, result, out_count);

    detail::tiny_unique_ptr<uint32_t> in(new uint32_t[in_count]),
                                     out(new uint32_t[out_count]);

    in_count = 0, out_count = 0;
    (read_indices(in.data, args, in_count), ...);
    read_indices(out.data, result, out_count);

    uint64_t func_hash = 0;
    uint32_t id = jitc_eval_ir_var(IsCUDA, in.data, in_count, out.data,
                                   out_count, se_total, &func_hash);
    return { id, func_hash };
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
    uint32_t n_inst   = jitc_registry_get_max(Class::Domain) + 1,
             buf_size = 22 + n_inst * 23 + 4;
    if (buf_size < 128)
        buf_size = 128;

    detail::tiny_unique_ptr<char> buf(new char[buf_size]);
    char *buf_ptr = buf.data;

    memcpy(buf_ptr, ".const .u64 $r0[] = { ", 22);
    buf_ptr += 22;

    // Call each instance symbolically and record!
    uint32_t index = jitc_var_new_0(1, VarType::Global, "", 1, 1);
    uint32_t se_before = jitc_side_effect_counter(IsCUDA);
    for (uint32_t i = 0; i < n_inst; ++i) {
        Class *ptr = (Class *) jitc_registry_get_ptr(Class::Domain, i);
        uint32_t prev = index;
        std::pair<uint32_t, uint64_t> id_and_hash;

        if (ptr)
            id_and_hash = record<IsCUDA>(
                [&](const Args &...args) { return func(ptr, args...); }, placeholder<Args>(args)...);
        else
            id_and_hash = record<IsCUDA>(
                [&](const Args &...) { return result; }, placeholder<Args>(args)...);

        index = jitc_var_new_2(1, VarType::Global, "", 1, index, id_and_hash.first);
        jitc_var_dec_ref_ext(id_and_hash.first);
        jitc_var_dec_ref_ext(prev);

        buf_ptr +=
            snprintf(buf_ptr, 23 + 1, "func_%016llx%s",
                     (unsigned long long) id_and_hash.second, i + 1 < n_inst ? ", " : " ");
    }
    uint32_t se_total = jitc_side_effect_counter(IsCUDA) - se_before;

    memcpy(buf_ptr, "};\n", 4);
    uint32_t call_table = jitc_var_new_1(1, VarType::Global, buf.data, 0, index);
    jitc_var_dec_ref_ext(index);

    uint32_t offset = jitc_var_new_2(1, VarType::UInt64,
            "mov.$t0 $r0, $r2$n"
            "mad.wide.u32 $r0, $r1, 8, $r0$n"
            "ld.const.$t0 $r0, [$r0]", 1, detach(self).index(), call_table);

    const uint32_t var_type_size[(int) VarType::Count] {
        0, 0, 1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8, 8
    };

    // Collect input arguments
    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, args, in_count), ...);
    read_indices(nullptr, result, out_count);
    detail::tiny_unique_ptr<uint32_t> in(new uint32_t[in_count]),
                                     out(new uint32_t[out_count]);
    in_count = 0; out_count = 0;
    (read_indices(in.data, args, in_count), ...);
    read_indices(out.data, result, out_count);

    index = jitc_var_new_1(1, VarType::Invalid, "", 1, offset);
    uint32_t offset_in = 0, align_in = 1;
    for (uint32_t i = 0; i < in_count; ++i) {
        uint32_t prev = index;
        index = jitc_var_new_2(1, VarType::Invalid, "", 1, in[i], index);
        jitc_var_dec_ref_ext(prev);
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(in[i])];
        offset_in = (offset_in + size - 1) / size * size;
        offset_in += size;
        if (size > align_in)
            align_in = size;
    }

    uint32_t offset_out = 0, align_out = 1;
    for (uint32_t i = 0; i < out_count; ++i) {
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(out[i])];
        offset_out = (offset_out + size - 1) / size * size;
        offset_out += size;
        if (size > align_out)
            align_out = size;
    }

    if (offset_in == 0)
        offset_in = 1;
    if (offset_out == 0)
        offset_out = 1;

    snprintf(buf.data, buf_size,
            "\n    {\n"
	        "        .param .align %u .b8 param_in[%u];\n"
	        "        .param .align %u .b8 param_out[%u]",
	        align_in, offset_in, align_out, offset_out);

    uint32_t prev = index;
    index = jitc_var_new_1(1, VarType::Invalid, buf.data, 0, index);
    jitc_var_dec_ref_ext(prev);

    offset_in = 0;
    for (uint32_t i = 0; i < in_count; ++i) {
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(in[i])];
        offset_in = (offset_in + size - 1) / size * size;
        snprintf(buf.data, buf_size, "    st.param.$t1 [param_in+%u], $r1", offset_in);
        uint32_t prev = index;
        index = jitc_var_new_2(1, VarType::Invalid, buf.data, 0, in[i], index);
        jitc_var_dec_ref_ext(prev);
        offset_in += size;
    }

    prev = index;
    index = jitc_var_new_3(1, VarType::Invalid,
            "    call (param_out), $r1, (param_in), $r2",
            1, offset, call_table, index);
    jitc_var_dec_ref_ext(prev);
    jitc_var_dec_ref_ext(offset);
    jitc_var_dec_ref_ext(call_table);

    offset_out = 0;
    for (uint32_t i = 0; i < out_count; ++i) {
        VarType type = jitc_var_type(out[i]);
        uint32_t size = var_type_size[(uint32_t) type];
        offset_out = (offset_out + size - 1) / size * size;
        uint32_t prev = index;
        snprintf(buf.data, buf_size, "    ld.param.$t0 $r0, [param_out+%u]", offset_out);
        index = jitc_var_new_1(1, type, buf.data, 0, index);
        out[i] = index;
        jitc_var_dec_ref_ext(prev);
        offset_out += size;
    }

    prev = index;
    index = jitc_var_new_1(1, VarType::Invalid, "}\n",
            1, index);

    jitc_var_dec_ref_ext(prev);

    if (se_total > 0) {
        jitc_var_inc_ref_ext(index);
        jitc_var_mark_scatter(index, 0);
    }

    for (uint32_t i = 0; i < out_count; ++i) {
        out[i] = jitc_var_new_2(1, jitc_var_type(out[i]),
                                "mov.$t0 $r0, $r1", 0,
                                out[i], index);
    }

    jitc_var_dec_ref_ext(index);
    out_count = 0;
    write_indices(out.data, result, out_count);

    return result;
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
