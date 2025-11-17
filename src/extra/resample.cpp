/*
    resample.cpp -- Python bindings for array resampling operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/resample.h>
#include <drjit/while_loop.h>
#include <drjit/math.h>
#include <nanothread/nanothread.h>
#include <cmath>
#include <algorithm>
#include <any>

NAMESPACE_BEGIN(drjit)

/// Internal storage of 'Resampler' (hidden via pImpl pattern)
struct Resampler::Impl {
    uint32_t source_res;
    uint32_t target_res;
    uint32_t taps;
    WrapMode wrap_mode;
    unique_ptr<uint32_t[]> offset;
    unique_ptr<double[]> weights;
    mutable std::any offset_cache;
    mutable std::any weights_cache;

    Impl(uint32_t source_res, uint32_t target_res, Resampler::Filter filter,
         const void *payload, double radius, double radius_scale,
         WrapMode wrap_mode)
        : source_res(source_res), target_res(target_res), wrap_mode(wrap_mode) {
        if (source_res == 0 || target_res == 0)
            drjit_raise("drjit.Resampler(): source/target resolution cannot be zero!");

        // Low-pass filter: scale reconstruction filters when downsampling
        double scale = (double) source_res / (double) target_res;
        double filter_scale = 1;
        if (target_res < source_res) {
            filter_scale = (double) target_res / (double) source_res;
            radius *= scale;
        }

        if (source_res == target_res) {
            // Convolution mode, adapt to filter size scale factor
            radius *= radius_scale;
            filter_scale /= radius_scale;
        }

        taps = (uint32_t) std::ceil(radius * 2);

        offset = unique_ptr<uint32_t[]>(new uint32_t[target_res]);
        weights = unique_ptr<double[]>(new double[taps * target_res]);

        for (uint32_t i = 0; i < target_res; i++) {
            // Fractional pos. of the new sample in the original coordinates
            double center = (i + 0.5) * scale;

            // Index of the first original sample that might contribute
            uint32_t offset_i =
                (uint32_t) std::max(0, (int) (center - radius + .5));
            offset[i] = offset_i;

            double sum = 0.0;
            for (uint32_t l = 0; l < taps; l++) {
                // Relative position of sample in the filter frame
                double rel = (offset_i - center + l + .5) * filter_scale;

                double weight = 0.0;
                // For clamp mode: only compute weight if in bounds (optimization)
                // For wrap/mirror: always compute weight, boundary handling done during sampling
                if (wrap_mode != WrapMode::Clamp || offset_i + l < source_res)
                    weight = filter(rel, payload);
                weights[i * taps + l] = weight;
                sum += weight;
            }

            if (sum == 0)
                drjit_raise(
                    "drjit.Resampler(): the filter footprint is too small; the "
                    "support of some output samples does not contain any input "
                    "samples!");

            // Normalize the weights for each output sample
            // For wrap/mirror modes, don't renormalize since all taps have valid samples
            // Only clamp mode needs renormalization due to boundary truncation
            if (wrap_mode == WrapMode::Clamp) {
                double normalization = 1.0 / sum;
                for (uint32_t l = 0; l < taps; l++)
                    weights[i * taps + l] *= normalization;
            }
        }
    }

    /// Apply boundary mode to convert potentially out-of-bounds index to valid index
    inline uint32_t apply_wrap_mode(int32_t idx) const {
        switch (wrap_mode) {
            case WrapMode::Clamp:
                return (uint32_t) std::max(0, std::min((int32_t)source_res - 1, idx));

            case WrapMode::Repeat: {
                idx = idx % (int32_t)source_res;
                if (idx < 0)
                    idx += source_res;
                return (uint32_t) idx;
            }

            case WrapMode::Mirror: {
                // Mirror the index at boundaries
                if (idx < 0)
                    idx = -idx - 1;
                int32_t period = 2 * (int32_t)source_res - 2;
                if (period > 0) {
                    idx = idx % period;
                    if (idx >= (int32_t)source_res)
                        idx = period - idx;
                }
                return (uint32_t) std::max(0, std::min((int32_t)source_res - 1, idx));
            }

            default:
                return (uint32_t) idx;
        }
    }

    /// Cast the resampling weights into a device array of the desired precision
    /// The implementation caches the result in case the Resampler is reused.
    template <typename T> const T& get_weights() const {
        const T *r = std::any_cast<T>(&weights_cache);
        if (!r) {
            using Scalar = scalar_t<T>;
            unique_ptr<Scalar[]> weights_tmp;
            size_t size = taps * target_res;
            const Scalar *weights_p;
            if constexpr (std::is_same_v<Scalar, double>) {
                weights_p = weights.get();
            } else {
                weights_tmp = new Scalar[size];
                for (size_t i = 0; i < size; ++i)
                    weights_tmp[i] = (Scalar) weights[i];
                weights_p = weights_tmp.get();
            }
            weights_cache = T::load_(weights_p, size);
            r = std::any_cast<T>(&weights_cache);
        }
        return *r;
    }

    /// Cast the resampling offset into a device array. The implementation
    /// caches the result in case the Resampler is reused.
    template <typename T> const T& get_offset() const {
        const T *r = std::any_cast<T>(&offset_cache);
        if (!r) {
            offset_cache = T::load_(offset.get(), target_res);
            r = std::any_cast<T>(&offset_cache);
        }
        return *r;
    }
};

static inline double sinc(double x) {
    if (x == 0.0)
        return 1.0;
    x *= Pi<double>;
    return std::sin(x) / x;
}

Resampler::Resampler(uint32_t source_res, uint32_t target_res, const char *filter,
                     double radius_scale, WrapMode wrap_mode) {
    Resampler::Filter filter_cb = nullptr;
    double radius = 0.0;

    if (strcmp(filter, "box") == 0) {
        filter_cb = [](double x, const void *) {
            if (x <= -0.5 || x > 0.5)
                return 0.0;
            return 1.0;
        };
        radius = .5f;
    } else if (strcmp(filter, "linear") == 0) {
        filter_cb = [](double x, const void *) {
            return fmax(1.0 - std::abs(x), 0.0);
        };
        radius = 1.f;
    } else if (strcmp(filter, "hamming") == 0) {
        filter_cb = [](double x, const void *) {
            x = std::abs(x);
            if (x == 0.0)
                return 1.0;
            if (x >= 1.0)
                return 0.0;
            x = x * Pi<double>;
            return std::sin(x) / x * (0.54 + 0.46 * std::cos(x));
        };
        radius = 1.f;
    } else if (strcmp(filter, "cubic") == 0) {
        filter_cb = [](double x, const void *) -> double {
            // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
            x = std::abs(x);
            const double a = -0.5;
            if (x < 1.0)
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            if (x < 2.0)
                return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        };
        radius = 2.f;
    } else if (strcmp(filter, "lanczos") == 0) {
        filter_cb = [](double x, const void *) -> double {
            if (x < -3.0 || x >= 3.0)
                return 0.0;
            return sinc(x) * sinc(x * (1.0 / 3.0));
        };
        radius = 3.f;
    } else if (strcmp(filter, "gaussian") == 0) {
        filter_cb = [](double x, const void *) -> double {
            if (x < -2.0 || x >= 2.0)
                return 0.0;
            double stddev = .5,
                   alpha = -1.0 / (2.0 * square(stddev));
            return maximum(0.f, exp(alpha * square(x)) - exp(alpha * square(2.0)));


        };
        radius = 2.f;
    } else {
        drjit_raise("'filter': unknown value ('box', 'linear', "
                    "'hamming', 'cubic', and 'lanczos' are supported).");
    }

    d = new Impl(source_res, target_res, filter_cb, nullptr, radius, radius_scale,
                 wrap_mode);
}

Resampler::Resampler(uint32_t source_res, uint32_t target_res,
                     Resampler::Filter filter, const void *payload,
                     double radius, WrapMode wrap_mode)
    : d(new Impl(source_res, target_res, filter, payload, radius, 1.0, wrap_mode)) {
}

Resampler::~Resampler() { }

/// Resample on the CPU and parallelize via the thread pool
template <typename Value>
void Resampler::resample(const Value *source, Value *target,
                         uint32_t source_size, uint32_t stride) const {
    struct Task {
        uint32_t source_res;
        uint32_t target_res;
        uint32_t taps;
        uint32_t stride;
        uint32_t inner_dim;
        const uint32_t *offset;
        const double *weights;
        const Value *in;
        Value *out;
        bool parallelize_outer;
        WrapMode wrap_mode;
    };

    auto callback = [](uint32_t outer, void *payload) {
        using Accum =
            std::conditional_t<sizeof(Value) < sizeof(float), float, Value>;

        Task t = *(Task *) payload;

        for (uint32_t inner = 0; inner < t.inner_dim; ++inner) {
            uint32_t i = t.parallelize_outer ? outer : inner,
                     j = t.parallelize_outer ? inner : outer;

            // For clamp mode (default), use the optimized offset pointer
            // For wrap/mirror modes, need to index from the start of the row
            const Value *in_base = t.in + (i * t.source_res * t.stride);
            const Value *in = (t.wrap_mode == WrapMode::Clamp)
                ? in_base + (t.offset[j] * t.stride)
                : in_base;
            Value *out = t.out + (i * t.target_res + j) * t.stride;

            for (uint32_t k = 0; k < t.stride; ++k) {
                Accum accum = 0;
                for (uint32_t l = 0; l < t.taps; ++l) {
                    double weight = t.weights[j * t.taps + l];

                    if (t.wrap_mode == WrapMode::Clamp) {
                        // Clamp mode: weights are only non-zero for in-bounds samples
                        // Use optimized pointer with offset already applied
                        if (weight != 0) {
                            Value value = in[l * t.stride + k];
                            accum = fmadd((Accum) weight, (Accum) value, accum);
                        }
                    } else {
                        // Wrap/Mirror modes: weights are computed for all taps
                        // Apply boundary mode to get the wrapped index
                        int32_t abs_idx = (int32_t)(t.offset[j] + l);

                        if (t.wrap_mode == WrapMode::Repeat) {
                            abs_idx = abs_idx % (int32_t)t.source_res;
                            if (abs_idx < 0)
                                abs_idx += t.source_res;
                        } else { // Mirror
                            if (abs_idx < 0)
                                abs_idx = -abs_idx - 1;
                            int32_t period = 2 * (int32_t)t.source_res - 2;
                            if (period > 0) {
                                abs_idx = abs_idx % period;
                                if (abs_idx >= (int32_t)t.source_res)
                                    abs_idx = period - abs_idx;
                            }
                            abs_idx = std::max(0, std::min((int32_t)t.source_res - 1, abs_idx));
                        }

                        Value value = in_base[abs_idx * t.stride + k];
                        accum = fmadd((Accum) weight, (Accum) value, accum);
                    }
                }
                if constexpr (std::is_same_v<Value, uint8_t>)
                    accum = clip(accum, (uint8_t) 0, (uint8_t) 255);
                *out++ = (Value) accum;
            }
        }
    };

    uint32_t n_passes = source_size / (d->source_res * stride),
             target_size = n_passes * d->target_res * stride;

    bool small_workload = std::max(source_size, target_size) < 256 * 256;
    bool parallelize_outer = small_workload || n_passes >= pool_size();

    uint32_t outer_dim = parallelize_outer ? n_passes : d->target_res,
             inner_dim = parallelize_outer ? d->target_res : n_passes;

    Task task {
        d->source_res,
        d->target_res,
        d->taps,
        stride,
        inner_dim,
        d->offset.get(),
        d->weights.get(),
        source,
        target,
        parallelize_outer,
        d->wrap_mode
    };

    if (small_workload) {
        for (uint32_t i = 0; i < outer_dim; ++i)
            callback(i, &task);
    } else {
        task_submit_and_wait(nullptr, outer_dim, callback, &task);
    }
}

template <typename Array>
Array Resampler::resample_fwd(const Array &source, uint32_t stride) const {
    using Accum = std::conditional_t<sizeof(scalar_t<Array>) <= 4,
                                     float32_array_t<Array>, Array>;
    using UInt32 = uint32_array_t<Array>;
    using Int32 = int32_array_t<Array>;

    const Accum &weights = d->get_weights<Accum>();
    const UInt32 &offset = d->get_offset<UInt32>();

    uint32_t source_size = (uint32_t) source.size(),
             n_passes = source_size / (d->source_res * stride),
             target_size = n_passes * d->target_res * stride;

    // Get pass index ('i'). The divisions below are all by scalar constants
    UInt32 idx = arange<UInt32>(target_size),
           i   = idx / (d->target_res * stride);

    idx = fmadd(i, (uint32_t)(-(int32_t)(d->target_res * stride)), idx);

    // Get index in output axis ('i') and channel ('k')
    UInt32 j = idx / stride,
           k = fmadd(j, (uint32_t)(-(int32_t)stride), idx),
           l = zeros<UInt32>(target_size);

    UInt32 offset_j = gather<UInt32>(offset, j),
           weight_offset = j * d->taps;

    UInt32 base_offset = i * d->source_res * stride + k;

    Accum target = zeros<Accum>(target_size);

    // For clamp mode, use optimized path (original behavior)
    if (d->wrap_mode == WrapMode::Clamp) {
        tie(l, target) = while_loop(
            make_tuple(l, target),
            // Loop condition
            [taps = d->taps](const UInt32 &l, const Accum &) {
                return l < taps;
            },
            // Loop body
            [base_offset, offset_j, source, weight_offset, weights,
             stride](UInt32 &l, Accum &target) {
                Accum weight = gather<Accum>(weights, weight_offset + l);
                UInt32 source_offset = base_offset + offset_j * stride + l * stride;
                Array value = gather<Array>(source, source_offset,
                                            weight != Accum(0));
                target = fmadd(weight, Accum(value), target);
                l += 1;
            });
    } else {
        // For wrap/mirror modes, apply boundary mode to indices
        tie(l, target) = while_loop(
            make_tuple(l, target),
            // Loop condition
            [taps = d->taps](const UInt32 &l, const Accum &) {
                return l < taps;
            },
            // Loop body
            [base_offset, offset_j, source, weight_offset, weights,
             stride, source_res = d->source_res, wrap_mode = d->wrap_mode]
            (UInt32 &l, Accum &target) {
                Accum weight = gather<Accum>(weights, weight_offset + l);

                // Compute the actual source index
                Int32 source_idx = Int32(offset_j) + Int32(l);

                // Apply boundary mode
                UInt32 wrapped_idx;
                if (wrap_mode == WrapMode::Repeat) {
                    Int32 wrapped = source_idx % Int32(source_res);
                    wrapped = select(wrapped < 0, wrapped + Int32(source_res), wrapped);
                    wrapped_idx = UInt32(wrapped);
                } else { // Mirror
                    Int32 mirrored = select(source_idx < 0, -source_idx - 1, source_idx);
                    int32_t period = 2 * (int32_t)source_res - 2;
                    if (period > 0) {
                        mirrored = mirrored % period;
                        mirrored = select(mirrored >= Int32(source_res),
                                        period - mirrored, mirrored);
                    }
                    wrapped_idx = UInt32(maximum(Int32(0), minimum(Int32(source_res - 1), mirrored)));
                }

                UInt32 source_offset = base_offset + wrapped_idx * stride;
                Array value = gather<Array>(source, source_offset);
                target = fmadd(weight, Accum(value), target);
                l += 1;
            });
    }

    return Array(target);
}

template <typename Array>
Array Resampler::resample_bwd(const Array &target, uint32_t stride) const {
    using Accum = std::conditional_t<sizeof(scalar_t<Array>) <= 4,
                                     float32_array_t<Array>, Array>;
    using UInt32 = uint32_array_t<Array>;
    using Int32 = int32_array_t<Array>;

    const Accum &weights = d->get_weights<Accum>();
    const UInt32 &offset = d->get_offset<UInt32>();

    uint32_t target_size = (uint32_t) target.size(),
             n_passes = target_size / (d->target_res * stride),
             source_size = n_passes * d->source_res * stride;

    // Get pass index ('i'). The divisions below are all by scalar constants
    UInt32 idx = arange<UInt32>(target_size),
           i   = idx / (d->target_res * stride);

    idx = fmadd(i, (uint32_t)(-(int32_t)(d->target_res * stride)), idx);

    // Get index in output axis ('i') and channel ('k')
    UInt32 j = idx / stride,
           k = fmadd(j, (uint32_t)(-(int32_t)stride), idx),
           l = zeros<UInt32>(target_size);

    UInt32 offset_j = gather<UInt32>(offset, j),
           weight_offset = j * d->taps;

    UInt32 base_offset = i * d->source_res * stride + k;

    Accum source = zeros<Accum>(source_size);

    // For clamp mode, use optimized path (original behavior)
    if (d->wrap_mode == WrapMode::Clamp) {
        tie(l, source) = while_loop(
            make_tuple(l, source),
            // Loop condition
            [taps = d->taps](const UInt32 &l, const Accum &) {
                return l < taps;
            },
            // Loop body
            [base_offset, offset_j, target, weight_offset, weights,
             stride](UInt32 &l, Accum &source) {
                Accum weight = gather<Accum>(weights, weight_offset + l);
                UInt32 source_offset = base_offset + offset_j * stride + l * stride;
                scatter_add(
                    source,
                    Accum(target) * weight,
                    source_offset,
                    weight != Accum(0)
                );
                l += 1;
            });
    } else {
        // For wrap/mirror modes, apply boundary mode to indices
        tie(l, source) = while_loop(
            make_tuple(l, source),
            // Loop condition
            [taps = d->taps](const UInt32 &l, const Accum &) {
                return l < taps;
            },
            // Loop body
            [base_offset, offset_j, target, weight_offset, weights,
             stride, source_res = d->source_res, wrap_mode = d->wrap_mode]
            (UInt32 &l, Accum &source) {
                Accum weight = gather<Accum>(weights, weight_offset + l);

                // Compute the actual source index
                Int32 source_idx = Int32(offset_j) + Int32(l);

                // Apply boundary mode
                UInt32 wrapped_idx;
                if (wrap_mode == WrapMode::Repeat) {
                    Int32 wrapped = source_idx % Int32(source_res);
                    wrapped = select(wrapped < 0, wrapped + Int32(source_res), wrapped);
                    wrapped_idx = UInt32(wrapped);
                } else { // Mirror
                    Int32 mirrored = select(source_idx < 0, -source_idx - 1, source_idx);
                    int32_t period = 2 * (int32_t)source_res - 2;
                    if (period > 0) {
                        mirrored = mirrored % period;
                        mirrored = select(mirrored >= Int32(source_res),
                                        period - mirrored, mirrored);
                    }
                    wrapped_idx = UInt32(maximum(Int32(0), minimum(Int32(source_res - 1), mirrored)));
                }

                UInt32 source_offset = base_offset + wrapped_idx * stride;
                scatter_add(
                    source,
                    Accum(target) * weight,
                    source_offset
                );
                l += 1;
            });
    }

    return Array(source);
}

uint32_t Resampler::source_res() const { return d->source_res; }
uint32_t Resampler::target_res() const { return d->target_res; }
uint32_t Resampler::taps() const { return d->taps; }

template DRJIT_EXTRA_EXPORT void Resampler::resample(const uint8_t *, uint8_t *, uint32_t, uint32_t) const;
template DRJIT_EXTRA_EXPORT void Resampler::resample(const half *, half *, uint32_t, uint32_t) const;
template DRJIT_EXTRA_EXPORT void Resampler::resample(const float *, float *, uint32_t, uint32_t) const;
template DRJIT_EXTRA_EXPORT void Resampler::resample(const double *, double *, uint32_t, uint32_t) const;

#if defined(DRJIT_ENABLE_CUDA)
template CUDAArray<half> Resampler::resample_fwd(const CUDAArray<half> &, uint32_t) const;
template CUDAArray<float> Resampler::resample_fwd(const CUDAArray<float> &, uint32_t) const;
template CUDAArray<double> Resampler::resample_fwd(const CUDAArray<double> &, uint32_t) const;
template CUDAArray<half> Resampler::resample_bwd(const CUDAArray<half> &, uint32_t) const;
template CUDAArray<float> Resampler::resample_bwd(const CUDAArray<float> &, uint32_t) const;
template CUDAArray<double> Resampler::resample_bwd(const CUDAArray<double> &, uint32_t) const;
#endif

#if defined(DRJIT_ENABLE_LLVM)
template LLVMArray<half> Resampler::resample_fwd(const LLVMArray<half> &, uint32_t) const;
template LLVMArray<float> Resampler::resample_fwd(const LLVMArray<float> &, uint32_t) const;
template LLVMArray<double> Resampler::resample_fwd(const LLVMArray<double> &, uint32_t) const;
template LLVMArray<half> Resampler::resample_bwd(const LLVMArray<half> &, uint32_t) const;
template LLVMArray<float> Resampler::resample_bwd(const LLVMArray<float> &, uint32_t) const;
template LLVMArray<double> Resampler::resample_bwd(const LLVMArray<double> &, uint32_t) const;
#endif

NAMESPACE_END(drjit)
