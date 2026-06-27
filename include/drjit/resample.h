/*
    resample.h -- Python bindings for array resampling operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/dynamic.h>
#include <drjit/jit.h>
#include <drjit/extra.h>

NAMESPACE_BEGIN(drjit)

/**
 * \brief Boundary handling mode used by \ref Resampler.
 *
 * The mode determines how filter taps that reach beyond the array bounds are
 * treated. The names mirror those of ``scipy.ndimage``. Given an array
 * ``[a b c d]``, the left boundary is extended as follows:
 *
 * - ``Zero``:    ``0 0 0 | a b c d`` (out-of-bounds taps contribute nothing)
 * - ``Nearest``: ``a a a | a b c d`` (clamp to the edge sample)
 * - ``Wrap``:    ``b c d | a b c d`` (periodic, period equal to the array size)
 * - ``Reflect``: ``c b a | a b c d`` (reflect, the edge sample is duplicated)
 * - ``Mirror``:  ``d c b | a b c d`` (reflect, the edge sample is not duplicated)
 */
enum class Boundary : uint32_t {
    Zero, Nearest, Wrap, Reflect, Mirror
};

/**
 * \brief Helper data structure to increase or decrease the resolution of an
 * array/tensor or convolve it along a set of axes.
 *
 * The ``Resampler`` class represents a precomputed windowed transformation. It
 * serves two purposes: *resampling* a signal to a different resolution
 * (``source_res != target_res``, e.g. up-/downsampling images or volumes), and
 * *convolution* at the same resolution (``source_res == target_res``, e.g.
 * filtering with a discrete kernel or a continuous filter).
 *
 * Constructing a ``Resampler`` object incurs a one-time cost. It is
 * advantageous to reuse the sampler a number of times when possible.
 */
class DRJIT_EXTRA_EXPORT Resampler {
public:
    using Filter = double (*)(double x, const void *payload);

    /**
     * Create a Resampler that uses a predefined reconstruction filter to
     * resample a signal from resolution ``source_res`` to ``target_res``.
     *
     * The following ``filter`` presets are available:
     *
     * - ``"box"``: use nearest-neighbor interpolation/averaging. This is
     *   very efficient but generally produces sub-par output that is either
     *   pixelated (when upsampling) or aliased (when downsampling).
     *
     * - ``"linear"``: use linear ramp / tent filter that uses 2 neighbors to
     *    reconstruct each output sample when upsampling. Tends to produce
     *    relatively blurry results.
     *
     * - ``"hamming"``: uses the same number of input samples as ``"linear"``
     *    but better preserves sharpness when downscaling. Do not use for
     *    upscaling.
     *
     * - ``"cubic"``: use cubic filter kernel that queries 4 neighbors to
     *   reconstruct each output sample when upsampling. Produces high-quality
     *   results.
     *
     * - ``"lanczos"``: use a windowed Lanczos filter that queries 6 neighbors
     *   to reconstruct each output sample when upsampling. This is the best
     *   filter for smooth signals, but also the costliest. The Lanczos filter
     *   is susceptible to ringing when the input array contains discontinuities.
     *
     * - ``"gaussian"``: use a Gaussian filter that queries 4 neighbors to
     *    reconstruct each output sample when upsampling. The Gaussian has a
     *    standard deviation of 0.5 and is truncated after 4 standard
     *    deviations. This filter is mainly useful when intending to blur a signal.
     *
     * The optional ``radius_scale`` scales the filter kernel radius.
     *
     * See \ref Boundary and the discrete-kernel constructor for ``boundary`` and
     * ``normalize``. In traced modes, ``symbolic`` generates a symbolic loop over
     * the taps, while the default unrolls the loop and evaluates the result at
     * the end (usually faster). The built-in filters are symmetric, so there is
     * no ``flip`` parameter.
     */
    Resampler(uint32_t source_res, uint32_t target_res, const char *filter,
              double radius_scale = 1.0, Boundary boundary = Boundary::Zero,
              bool normalize = true, bool symbolic = false);

    /**
     * \brief Construct a Resampler using a custom (continuous) filter kernel.
     *
     * The filter radius on the target domain must be specified. The payload
     * parameter will be passed to the provided filter callback and can contain
     * arbitrary additional information.
     *
     * The implementation will invoke ``filter(x, payload)`` a number of times
     * to precompute internal tables used for resampling. The filter must return
     * zero for positions ``x`` outside of the interval ``[-radius, radius]``.
     *
     * See \ref Boundary and the discrete-kernel constructor for ``boundary``,
     * ``normalize``, and ``flip`` (unlike the preset filters, a custom filter may
     * be asymmetric, so ``flip`` is meaningful here). ``symbolic`` selects the
     * traced codegen as above.
     */
    Resampler(uint32_t source_res, uint32_t target_res, Filter filter,
              const void *payload, double radius,
              Boundary boundary = Boundary::Zero, bool normalize = true,
              bool flip = false, bool symbolic = false);

    /**
     * \brief Construct a Resampler using a discrete filter kernel.
     *
     * This constructor builds a resampler that convolves a signal with the
     * provided sequence of ``kernel_size`` discrete coefficients while leaving
     * the resolution unchanged (``source_res == target_res == res``).
     *
     * The ``origin`` parameter specifies which kernel entry is aligned with the
     * output sample (typically ``kernel_size / 2``). The ``boundary`` parameter
     * selects how out-of-bounds accesses are handled (see \ref Boundary).
     *
     * When ``normalize`` is set, the effective per-output weights are rescaled
     * to sum to one (this is the behavior needed to filter a signal without
     * affecting its overall magnitude). When it is unset, the raw kernel
     * coefficients are used, matching the convention of ``numpy.convolve``.
     *
     * When ``flip`` is set, the kernel is reversed prior to application,
     * realizing a true convolution (:math:`\\sum_l k[l]\\,x[i-l+o]`) as opposed
     * to a correlation (:math:`\\sum_l k[l]\\,x[i+l-o]`).
     *
     * ``symbolic`` selects the traced codegen: it generates a symbolic loop over
     * the taps, while the default unrolls and evaluates the result (usually
     * faster).
     */
    Resampler(uint32_t res, const double *kernel, size_t kernel_size,
              int origin, Boundary boundary, bool normalize, bool flip,
              bool symbolic = false);

    /// Free the resampler object
    ~Resampler();

    /// Return the Resampler's source resolution
    uint32_t source_res() const;

    /// Return the Resampler's target resolution
    uint32_t target_res() const;

    /**
     * \brief Return the number of filter taps
     *
     * This refers to how many source values may be read to
     * produce one target value.
     */
    uint32_t taps() const;

    /**
     * \brief Resample a memory buffer on the CPU
     *
     * This function resamples the memory buffer ``source`` containing
     * ``source_size`` elements of type ``Value`` and writes the output to
     * ``target``, which must have sufficient space for ``source_size /
     * source_res() * target_res()`` elements.
     *
     * The function can resample nd-arrays. In particular, adjacent
     * elements along the resampled axis are assumed to be separated by
     * ``stride`` elements. For a flat 1D array, specify ``stride=1``.
     *
     * When either the input or output array has more than 64K elements, the
     * implementation uses the nanothread thread pool to parallelize the
     * resampling operation.
     *
     * The following ``Value`` types are currently supported:
     *
     * - ``uint8_t``
     * - ``half``
     * - ``float``
     * - ``double``
     */
    template <typename Value>
    void resample(const Value *source, Value *target, uint32_t source_size,
                  uint32_t stride) const;

    /// Convenience wrapper around \ref resample() for dynamic CPU arrays
    template <typename Scalar>
    DynamicArray<Scalar> resample_fwd(const DynamicArray<Scalar> &source,
                                      uint32_t stride) const {
        uint32_t source_size = (uint32_t) source.size(),
                 target_size = source_size / source_res() * target_res();
        DynamicArray<Scalar> target = empty<DynamicArray<Scalar>>(target_size);
        resample(source.data(), target.data(), source_size, stride);
        return target;
    }

    /**
     * \brief Resample a JIT-compiled (CUDA/LLVM) array
     *
     * This function is analogous to the ``resample_*`` functions above.
     * It resamples the input array with the stride ``stride``.
     *
     * The main difference is that this version *traces* the resampling step.
     * It is usable with LLVM, CUDA, AMD, and Metal arrays.
     */
    template <typename Array>
    Array resample_fwd(const Array &source, uint32_t stride) const;

    /**
     * \brief Backward derivative of \ref resample_fwd()
     *
     * This function computes the backward derivative of \ref resample_fwd()
     * and traces the computation.
     */
    template <typename Array>
    Array resample_bwd(const Array &target, uint32_t stride) const;

protected:
    struct Impl;
    unique_ptr<Impl> d;
};

extern template DRJIT_EXTRA_EXPORT void Resampler::resample(const uint8_t *, uint8_t *, uint32_t, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT void Resampler::resample(const half *, half *, uint32_t, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT void Resampler::resample(const float *, float *, uint32_t, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT void Resampler::resample(const double *, double *, uint32_t, uint32_t) const;

#if defined(DRJIT_ENABLE_CUDA)
extern template DRJIT_EXTRA_EXPORT CUDAArray<half> Resampler::resample_fwd(const CUDAArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT CUDAArray<float> Resampler::resample_fwd(const CUDAArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT CUDAArray<double> Resampler::resample_fwd(const CUDAArray<double> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT CUDAArray<half> Resampler::resample_bwd(const CUDAArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT CUDAArray<float> Resampler::resample_bwd(const CUDAArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT CUDAArray<double> Resampler::resample_bwd(const CUDAArray<double> &, uint32_t) const;
#endif

#if defined(DRJIT_ENABLE_AMD)
extern template DRJIT_EXTRA_EXPORT AMDArray<half> Resampler::resample_fwd(const AMDArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT AMDArray<float> Resampler::resample_fwd(const AMDArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT AMDArray<double> Resampler::resample_fwd(const AMDArray<double> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT AMDArray<half> Resampler::resample_bwd(const AMDArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT AMDArray<float> Resampler::resample_bwd(const AMDArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT AMDArray<double> Resampler::resample_bwd(const AMDArray<double> &, uint32_t) const;
#endif

#if defined(DRJIT_ENABLE_LLVM)
extern template DRJIT_EXTRA_EXPORT LLVMArray<half> Resampler::resample_fwd(const LLVMArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<float> Resampler::resample_fwd(const LLVMArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<double> Resampler::resample_fwd(const LLVMArray<double> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<half> Resampler::resample_bwd(const LLVMArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<float> Resampler::resample_bwd(const LLVMArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<double> Resampler::resample_bwd(const LLVMArray<double> &, uint32_t) const;
#endif

NAMESPACE_END(drjit)
