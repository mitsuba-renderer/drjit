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
 * \brief Boundary handling modes for resampling operations
 */
enum class BoundaryMode {
    /// Clamp coordinates to edges (default behavior)
    Clamp = 0,
    /// Wrap coordinates periodically
    Wrap = 1,
    /// Mirror/reflect coordinates at boundaries
    Mirror = 2
};

/**
 * \brief Helper data structure to increase or decrease the resolution of an
 * array/tensor along a set of axes.
 *
 * The ``Resampler`` class represents a precomputed transformation that
 * resamples a signal from one resolution to another. It can be used to
 * downsample/upsample images or volumes, or to filter data while leaving the
 * input resolution unchanged.
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
     * The optional ``radius_scale`` parameter can be used to scale the
     * filter kernel radius.
     *
     * The optional ``boundary_mode`` parameter specifies how out-of-bounds
     * samples are handled:
     * - ``BoundaryMode::Clamp``: clamp to edge values (default)
     * - ``BoundaryMode::Wrap``: wrap around periodically
     * - ``BoundaryMode::Mirror``: mirror/reflect at boundaries
     */
    Resampler(uint32_t source_res, uint32_t target_res, const char *filter,
              double radius_scale = 1.0,
              BoundaryMode boundary_mode = BoundaryMode::Clamp);

    /**
     * \brief Construct a Resampler using a custom filter kernel.
     *
     * The filter radius on the target domain must be specified. The payload
     * parameter will be passed to the provided filter callback and can contain
     * arbitrary additional information.
     *
     * The implementation will invoke ``filter(x, payload)`` a number of times
     * to precompute internal tables used for resampling. The filter must return
     * zero for positions ``x`` outside of the interval ``[-radius, radius]``.
     *
     * The optional ``boundary_mode`` parameter specifies how out-of-bounds
     * samples are handled (see the documentation for the other constructor).
     */
    Resampler(uint32_t source_res, uint32_t target_res, Filter filter,
              const void *payload, double radius,
              BoundaryMode boundary_mode = BoundaryMode::Clamp);

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
     * It is usable with LLVM and CUDA arrays.
     */
    template <typename Array>
    Array resample_fwd(const Array &source, uint32_t stride) const;

    /**
     * \brief Backward derivative of \ref resample_fwd()
     *
     * This function computes the backward derivative of \ref resample_fwd().
     * Given the drivative of the resampled output array, it computes the
     * derivative of the input. The function is usable with LLVM and CUDA arrays.
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

#if defined(DRJIT_ENABLE_LLVM)
extern template DRJIT_EXTRA_EXPORT LLVMArray<half> Resampler::resample_fwd(const LLVMArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<float> Resampler::resample_fwd(const LLVMArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<double> Resampler::resample_fwd(const LLVMArray<double> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<half> Resampler::resample_bwd(const LLVMArray<half> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<float> Resampler::resample_bwd(const LLVMArray<float> &, uint32_t) const;
extern template DRJIT_EXTRA_EXPORT LLVMArray<double> Resampler::resample_bwd(const LLVMArray<double> &, uint32_t) const;
#endif

NAMESPACE_END(drjit)
