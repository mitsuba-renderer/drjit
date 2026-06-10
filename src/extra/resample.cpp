/*
    resample.cpp -- Python bindings for array resampling operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/resample.h>
#include <drjit/while_loop.h>
#include <drjit/if_stmt.h>
#include <drjit/math.h>
#include <nanothread/nanothread.h>
#include <cmath>
#include <algorithm>
#include <any>
#include <utility>

NAMESPACE_BEGIN(drjit)

/// Internal representation of 'Resampler'
struct Resampler::Impl {
    uint32_t source_res;
    uint32_t target_res;
    uint32_t taps;
    Boundary boundary;
    unique_ptr<int32_t[]> offset;
    unique_ptr<double[]> weights;
    mutable std::any offset_cache;
    mutable std::any weights_cache;

    // Codegen strategy: symbolic tap loop vs. unrolled + evaluated
    bool symbolic = false;

    // Transpose fast path: a shift-invariant operator (every window start equals
    // the affine 'j - conv_origin') has a windowed-convolution adjoint.
    bool shift_invariant = false;
    int32_t conv_origin = 0;
    mutable unique_ptr<Impl> transpose_impl;

    // Interior fast path: the in-bounds run [interior_lo, interior_hi) shares the
    // weight row 'interior_weights', which resample_fwd() bakes as constants.
    bool has_interior = false;
    uint32_t interior_lo = 0, interior_hi = 0;
    unique_ptr<double[]> interior_weights;

    Impl() = default;

    /// Construct a resampler from a continuous reconstruction filter
    Impl(uint32_t source_res, uint32_t target_res, Resampler::Filter filter,
         const void *payload, double radius, double radius_scale,
         Boundary boundary, bool normalize, bool flip)
        : source_res(source_res), target_res(target_res), boundary(boundary) {
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

        offset = unique_ptr<int32_t[]>(new int32_t[target_res]);
        weights = unique_ptr<double[]>(new double[taps * target_res]);

        // 'flip': convolve by flipping the sign of the filter kenrle
        double sign = flip ? -1.0 : 1.0;

        unique_ptr<double[]> raw(new double[taps]);

        for (uint32_t i = 0; i < target_res; i++) {
            // Fractional pos. of the new sample in the original coordinates
            double center = (i + 0.5) * scale;

            // Signed index of the first original sample that might contribute
            int32_t start = (int32_t) (center - radius + .5);

            for (uint32_t l = 0; l < taps; l++) {
                // Relative position of sample in the filter frame
                double rel = (start + (int32_t) l - center + .5) * filter_scale;
                raw[l] = filter(sign * rel, payload);
            }

            store_row(i, start, raw.get(), normalize);
        }

        configure();
    }

    /// Construct a resampler from a discrete filter kernel
    Impl(uint32_t res, const double *kernel, size_t kernel_size, int origin,
         Boundary boundary, bool normalize, bool flip)
        : source_res(res), target_res(res), boundary(boundary) {
        if (res == 0)
            drjit_raise("drjit.Resampler(): source/target resolution cannot be zero!");
        if (kernel_size == 0)
            drjit_raise("drjit.Resampler(): the filter kernel cannot be empty!");
        if (boundary != Boundary::Zero && kernel_size > res)
            drjit_raise("drjit.Resampler(): the filter kernel cannot be larger "
                        "than the array for this boundary mode!");

        taps = (uint32_t) kernel_size;

        offset = unique_ptr<int32_t[]>(new int32_t[res]);
        weights = unique_ptr<double[]>(new double[taps * res]);

        unique_ptr<double[]> raw(new double[taps]);

        // 'flip': convolve by reversing the kernel and mirroring the origin.
        int o = origin;
        for (uint32_t l = 0; l < taps; l++)
            raw[l] = flip ? kernel[taps - 1 - l] : kernel[l];
        if (flip)
            o = (int) (taps - 1) - origin;

        for (uint32_t i = 0; i < res; i++)
            store_row(i, (int32_t) i - o, raw.get(), normalize);

        configure();
    }

    /// Populate the fast-path variables from the precomputed tables
    void configure() {
        has_interior = false;
        shift_invariant = false;
        if (source_res != target_res || taps >= source_res)
            return;

        uint32_t res = target_res, mid = res / 2;
        int32_t R = (int32_t) res;
        conv_origin = (int32_t) mid - offset[mid];

        // Shift-invariant if every window start is the affine 'j - conv_origin'.
        shift_invariant = true;
        for (uint32_t j = 0; j < res; ++j) {
            if (offset[j] != (int32_t) j - conv_origin) {
                shift_invariant = false;
                break;
            }
        }

        interior_weights = unique_ptr<double[]>(new double[taps]);
        for (uint32_t l = 0; l < taps; ++l)
            interior_weights[l] = weights[(size_t) mid * taps + l];

        auto uniform = [&](uint32_t j) -> bool {
            int32_t s = offset[j];
            if (s != (int32_t) j - conv_origin)   // affine window start
                return false;
            if (s < 0 || s + (int32_t) taps > R)  // window fully in bounds
                return false;
            for (uint32_t l = 0; l < taps; ++l)   // canonical weight row
                if (weights[(size_t) j * taps + l] != interior_weights[l])
                    return false;
            return true;
        };

        if (!uniform(mid))
            return;

        uint32_t lo = mid, hi = mid + 1;
        while (lo > 0 && uniform(lo - 1)) --lo;
        while (hi < res && uniform(hi)) ++hi;
        interior_lo = lo;
        interior_hi = hi;
        has_interior = true;
    }

    /**
     * \brief Apply the boundary mode and normalization to one output's weights
     * and store the resulting row in the tables.
     *
     * For the ``Zero`` boundary, taps outside ``[0, source_res)`` are masked to
     * zero; other modes keep the raw weight and redirect the tap at evaluation.
     *
     * \param i Output sample index
     * \param start Signed start index of the tap window
     * \param raw The ``taps`` raw filter weights
     * \param normalize Whether to rescale the row to sum to one
     */
    void store_row(uint32_t i, int32_t start, const double *raw, bool normalize) {
        offset[i] = start;

        double sum = 0.0;
        for (uint32_t l = 0; l < taps; l++) {
            int32_t g = start + (int32_t) l;
            double weight = raw[l];
            if (boundary == Boundary::Zero &&
                (g < 0 || g >= (int32_t) source_res))
                weight = 0.0;
            weights[i * taps + l] = weight;
            sum += weight;
        }

        if (normalize) {
            if (sum == 0)
                drjit_raise(
                    "drjit.Resampler(): the filter footprint is too small; the "
                    "support of some output samples does not contain any input "
                    "samples!");
            double normalization = 1.0 / sum;
            for (uint32_t l = 0; l < taps; l++)
                weights[i * taps + l] *= normalization;
        }
    }

    /// Lazily build and cache the transpose operator. Supported only with
    /// a shift-invariant filter and self-transposing boundary (Zero, Wrap).
    const Impl *get_transpose() const {
        if (transpose_impl || !shift_invariant ||
            (boundary != Boundary::Zero && boundary != Boundary::Wrap))
            return transpose_impl.get();

        int32_t Ri = (int32_t) source_res;
        unique_ptr<Impl> t(new Impl());
        t->source_res = t->target_res = source_res;
        t->taps = taps;
        t->boundary = boundary;
        t->symbolic = symbolic;
        t->offset = unique_ptr<int32_t[]>(new int32_t[source_res]);
        t->weights = unique_ptr<double[]>(new double[(size_t) taps * source_res]);

        // The forward computes out[j] = sum_l weights[j][l] * in[j - conv_origin
        // + l]. Its transpose reads, for input cell 'i', the outputs j = i +
        // conv_origin - l (a contiguous window), with flipped per-output weights.
        for (uint32_t i = 0; i < source_res; ++i) {
            int32_t start = (int32_t) i + conv_origin - (int32_t) (taps - 1);
            t->offset[i] = start;
            for (uint32_t m = 0; m < taps; ++m) {
                int32_t j = start + (int32_t) m;
                uint32_t l = taps - 1 - m;   // flipped tap index
                double wv = 0.0;
                if (boundary == Boundary::Wrap) {        // wrap the output index
                    int32_t jp = ((j % Ri) + Ri) % Ri;
                    wv = weights[(size_t) jp * taps + l];
                } else if (j >= 0 && j < Ri) {           // Zero: mask if out of range
                    wv = weights[(size_t) j * taps + l];
                }
                t->weights[(size_t) i * taps + m] = wv;
            }
        }

        t->configure();
        transpose_impl = std::move(t);
        return transpose_impl.get();
    }

    // -----------------------------------------------------------------------

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

    /// Decompose the flat output index ``[0, target_size)`` into the output
    /// position ``j`` along the resampled axis, the channel ``k``, and the pass
    /// base ``i_base = i * source_res``. Shared by resample_fwd()/_bwd().
    template <typename UInt32>
    void decompose(uint32_t target_size, uint32_t stride, UInt32 &j, UInt32 &k,
                   UInt32 &i_base) const {
        // The divisions below are all by scalar constants
        UInt32 idx = arange<UInt32>(target_size),
               i   = idx / (target_res * stride);
        idx = fmadd(i, (uint32_t)(-(int32_t)(target_res * stride)), idx);
        j = idx / stride;
        k = fmadd(j, (uint32_t)(-(int32_t)stride), idx);
        i_base = i * source_res;
    }

    /// Traced forward evaluation (defined out-of-line, after tap_address())
    template <typename Array>
    Array forward(const Array &source, uint32_t stride) const;
};

static inline double sinc(double x) {
    if (x == 0.0)
        return 1.0;
    x *= Pi<double>;
    return std::sin(x) / x;
}

/// Map an index to an in-bounds index in ``[0, R)`` according to the boundary mode
static inline int32_t remap_scalar(int32_t g, int32_t R, Boundary boundary) {
    switch (boundary) {
        case Boundary::Wrap:    return g < 0 ? g + R     : (g >= R ? g - R         : g);
        case Boundary::Reflect: return g < 0 ? -1 - g    : (g >= R ? 2 * R - 1 - g : g);
        case Boundary::Mirror:  return g < 0 ? -g        : (g >= R ? 2 * R - 2 - g : g);
        default:                return g < 0 ? 0         : (g >= R ? R - 1         : g);
    }
}

/// Vectorized counterpart of \ref remap_scalar()
template <typename Int32>
static Int32 remap_array(const Int32 &g, int32_t R, Boundary boundary) {
    switch (boundary) {
        case Boundary::Wrap:
            return select(g < 0, g + R, select(g >= R, g - R, g));
        case Boundary::Reflect:
            return select(g < 0, -1 - g, select(g >= R, 2 * R - 1 - g, g));
        case Boundary::Mirror:
            return select(g < 0, -g, select(g >= R, 2 * R - 2 - g, g));
        default:
            return clip(g, Int32(0), Int32(R - 1));
    }
}

/**
 * \brief Source address and bounds mask for tap ``l`` of one output.
 *
 * Shared by the gather/scatter sites in resample_fwd()/_bwd(). For ``Zero``,
 * ``active`` drops out-of-bounds taps; other modes redirect into bounds and stay
 * active.
 *
 * \param boundary Boundary handling mode
 * \param offset_j Window start of the output
 * \param l Tap index (compile-time in the unrolled path, traced in the symbolic one)
 * \param source_offset Signed ``Zero`` base ``(i_base + offset_j) * stride + k``
 * \param i_base Pass base ``i * source_res``
 * \param k Channel offset
 * \param stride Element stride along the resampled axis
 * \param R Source resolution
 * \param active Output bounds mask for the gather/scatter
 * \return Source address for the tap
 */
template <typename UInt32, typename Int32, typename L>
static UInt32 tap_address(Boundary boundary, const Int32 &offset_j, const L &l,
                          const UInt32 &source_offset, const UInt32 &i_base,
                          const UInt32 &k, uint32_t stride, int32_t R,
                          mask_t<UInt32> &active) {
    Int32 g = offset_j + Int32(l);
    if (boundary == Boundary::Zero) {
        active = (g >= 0) & (g < R);
        return source_offset + l * stride;
    }
    active = mask_t<UInt32>(true);
    return (i_base + UInt32(remap_array(g, R, boundary))) * stride + k;
}

Resampler::Resampler(uint32_t source_res, uint32_t target_res, const char *filter,
                     double radius_scale, Boundary boundary, bool normalize,
                     bool symbolic) {
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
        drjit_raise("'filter': unknown value ('box', 'linear', 'hamming', "
                    "'cubic', 'lanczos', and 'gaussian' are supported).");
    }

    d = new Impl(source_res, target_res, filter_cb, nullptr, radius, radius_scale,
                 boundary, normalize, /* flip */ false);
    d->symbolic = symbolic;
}

Resampler::Resampler(uint32_t source_res, uint32_t target_res,
                     Resampler::Filter filter, const void *payload,
                     double radius, Boundary boundary, bool normalize, bool flip,
                     bool symbolic)
    : d(new Impl(source_res, target_res, filter, payload, radius, 1.0,
                 boundary, normalize, flip)) {
    d->symbolic = symbolic;
}

Resampler::Resampler(uint32_t res, const double *kernel, size_t kernel_size,
                     int origin, Boundary boundary, bool normalize, bool flip,
                     bool symbolic)
    : d(new Impl(res, kernel, kernel_size, origin, boundary, normalize, flip)) {
    d->symbolic = symbolic;
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
        Boundary boundary;
        const int32_t *offset;
        const double *weights;
        const Value *in;
        Value *out;
        bool parallelize_outer;
    };

    auto callback = [](uint32_t outer, void *payload) {
        using Accum =
            std::conditional_t<sizeof(Value) < sizeof(float), float, Value>;

        Task t = *(Task *) payload;
        int32_t R = (int32_t) t.source_res;
        for (uint32_t inner = 0; inner < t.inner_dim; ++inner) {
            uint32_t i = t.parallelize_outer ? outer : inner,
                     j = t.parallelize_outer ? inner : outer;

            int32_t base = (int32_t) (i * t.source_res);
            Value *out = t.out + (i * t.target_res + j) * t.stride;

            for (uint32_t k = 0; k < t.stride; ++k) {
                Accum accum = 0;
                for (uint32_t l = 0; l < t.taps; ++l) {
                    double weight = t.weights[j * t.taps + l];
                    if (weight != 0) {
                        int32_t phys = remap_scalar(t.offset[j] + (int32_t) l,
                                                    R, t.boundary);
                        Value value = t.in[(base + phys) * (int32_t) t.stride + k];
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
        d->boundary,
        d->offset.get(),
        d->weights.get(),
        source,
        target,
        parallelize_outer
    };

    if (small_workload) {
        for (uint32_t i = 0; i < outer_dim; ++i)
            callback(i, &task);
    } else {
        task_submit_and_wait(nullptr, outer_dim, callback, &task);
    }
}

template <typename Array>
Array Resampler::Impl::forward(const Array &source, uint32_t stride) const {
    using Accum = std::conditional_t<sizeof(scalar_t<Array>) <= 4,
                                     float32_array_t<Array>, Array>;
    using UInt32 = uint32_array_t<Array>;
    using Int32  = int32_array_t<Array>;

    const Accum &weights_v = get_weights<Accum>();
    const Int32 &offset_v = get_offset<Int32>();

    uint32_t source_size = (uint32_t) source.size(),
             n_passes = source_size / (source_res * stride),
             target_size = n_passes * target_res * stride;

    UInt32 j, k, i_base;
    decompose(target_size, stride, j, k, i_base);

    Boundary boundary_v = this->boundary;
    int32_t R = (int32_t) source_res;

    Accum target = zeros<Accum>(target_size);

    uint32_t taps_v = this->taps;

    if (!symbolic) {
        // Table-driven per-output accumulation (handles any boundary). Unrolled
        // at trace time so the per-tap gathers are independent loads; the table
        // lookups stay inside the lambda so the if_stmt below keeps them out of
        // the interior branch.
        auto accum = [&](const UInt32 &ibase, const UInt32 &kk,
                         const UInt32 &jj) -> Accum {
            Int32  offj = gather<Int32>(offset_v, jj);
            UInt32 woff = jj * taps_v;
            UInt32 soff = UInt32(
                (Int32(ibase) + offj) * (int32_t) stride + Int32(kk));
            Accum acc = zeros<Accum>(target_size);
            for (uint32_t l = 0; l < taps_v; ++l) {
                Accum weight = gather<Accum>(weights_v, woff + l);
                mask_t<UInt32> active;
                UInt32 addr = tap_address(boundary_v, offj, l, soff, ibase, kk,
                                          stride, R, active);
                acc = fmadd(weight,
                            Accum(gather<Array>(source, addr, active)), acc);
            }
            return acc;
        };

        if (has_interior) {
            // Interior: bake the weights as constants and gather directly (in
            // bounds by construction); if_stmt routes the border to accum().
            int32_t corg = conv_origin;
            uint32_t lo = interior_lo, hi = interior_hi;

            target = if_stmt(
                make_tuple(i_base, k, j),
                (j >= lo) & (j < hi),
                // interior (fast): baked weights, arithmetic start, plain loads
                [&](const UInt32 &ibase, const UInt32 &kk,
                    const UInt32 &jj) -> Accum {
                    UInt32 base = (ibase + UInt32(Int32(jj) - corg)) * stride + kk;
                    Accum acc = zeros<Accum>(target_size);
                    for (uint32_t l = 0; l < taps_v; ++l)
                        acc = fmadd(
                            Accum((scalar_t<Accum>) interior_weights[l]),
                            Accum(gather<Array>(source, base + l * stride)), acc);
                    return acc;
                },
                // border: table-driven path
                [&](const UInt32 &ibase, const UInt32 &kk,
                    const UInt32 &jj) -> Accum {
                    return accum(ibase, kk, jj);
                });
        } else {
            target = accum(i_base, k, j);
        }
    } else {
        Int32  offset_j = gather<Int32>(offset_v, j);
        UInt32 weight_offset = j * taps_v;
        UInt32 source_offset = UInt32(
            (Int32(i_base) + offset_j) * (int32_t) stride + Int32(k));

        UInt32 l = zeros<UInt32>(target_size);
        tie(l, target) = while_loop(
            make_tuple(l, target),
            // Loop condition
            [taps_v](const UInt32 &l, const Accum &) {
                return l < taps_v;
            },
            // Loop body
            [source_offset, source, weight_offset, weights_v, stride, boundary_v,
             offset_j, i_base, k, R](UInt32 &l, Accum &target) {
                Accum weight = gather<Accum>(weights_v, weight_offset + l);
                mask_t<UInt32> active;
                UInt32 addr = tap_address(boundary_v, offset_j, l, source_offset,
                                          i_base, k, stride, R, active);
                target = fmadd(weight,
                               Accum(gather<Array>(source, addr, active)),
                               target);
                l += 1;
            });
    }

    Array result(target);

    // Explicitly evaluate to unrolled kernel in evaluated mode
    if (!symbolic)
        eval(result);

    return result;
}

template <typename Array>
Array Resampler::resample_fwd(const Array &source, uint32_t stride) const {
    return d->forward(source, stride);
}

template <typename Array>
Array Resampler::resample_bwd(const Array &target, uint32_t stride) const {
    // When the adjoint is itself a windowed convolution (Zero/Wrap boundary),
    // evaluate it as a forward *gather* of the flipped operator instead of an
    // atomic scatter.
    if (const Impl *t = d->get_transpose())
        return t->forward(target, stride);

    using Accum = std::conditional_t<sizeof(scalar_t<Array>) <= 4,
                                     float32_array_t<Array>, Array>;
    using UInt32 = uint32_array_t<Array>;
    using Int32  = int32_array_t<Array>;

    const Accum &weights = d->get_weights<Accum>();
    const Int32 &offset = d->get_offset<Int32>();

    uint32_t target_size = (uint32_t) target.size(),
             n_passes = target_size / (d->target_res * stride),
             source_size = n_passes * d->source_res * stride;

    UInt32 j, k, i_base;
    d->decompose(target_size, stride, j, k, i_base);

    Int32  offset_j = gather<Int32>(offset, j);
    UInt32 weight_offset = j * d->taps;

    // Signed base address used by the masked 'Zero' fast path
    UInt32 source_offset = UInt32(
        (Int32(i_base) + offset_j) * (int32_t) stride + Int32(k));

    Boundary boundary = d->boundary;
    int32_t R = (int32_t) d->source_res;

    Accum source = zeros<Accum>(source_size);

    uint32_t taps = d->taps;

    if (!d->symbolic) {
        // Unrolled adjoint scatter (no interior fast path: the 'source'-sized
        // accumulator is not aligned with the per-output interior predicate).
        for (uint32_t l = 0; l < taps; ++l) {
            Accum weight = gather<Accum>(weights, weight_offset + l);
            mask_t<UInt32> active;
            UInt32 addr = tap_address(boundary, offset_j, l, source_offset,
                                      i_base, k, stride, R, active);
            scatter_add(source, Accum(target) * weight, addr, active);
        }
    } else {
        UInt32 l = zeros<UInt32>(target_size);
        tie(l, source) = while_loop(
            make_tuple(l, source),
            // Loop condition
            [taps = d->taps](const UInt32 &l, const Accum &) {
                return l < taps;
            },
            // Loop body
            [source_offset, target, weight_offset, weights, stride, boundary,
             offset_j, i_base, k, R](UInt32 &l, Accum &source) {
                Accum weight = gather<Accum>(weights, weight_offset + l);
                mask_t<UInt32> active;
                UInt32 addr = tap_address(boundary, offset_j, l, source_offset,
                                          i_base, k, stride, R, active);
                scatter_add(source, Accum(target) * weight, addr, active);
                l += 1;
            });
    }

    Array result(source);
    if (!d->symbolic)
        eval(result);
    return result;
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

#if defined(DRJIT_ENABLE_METAL)
template DRJIT_EXTRA_EXPORT MetalArray<half> Resampler::resample_fwd(const MetalArray<half> &, uint32_t) const;
template DRJIT_EXTRA_EXPORT MetalArray<float> Resampler::resample_fwd(const MetalArray<float> &, uint32_t) const;
template DRJIT_EXTRA_EXPORT MetalArray<half> Resampler::resample_bwd(const MetalArray<half> &, uint32_t) const;
template DRJIT_EXTRA_EXPORT MetalArray<float> Resampler::resample_bwd(const MetalArray<float> &, uint32_t) const;
#endif

NAMESPACE_END(drjit)
