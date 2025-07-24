/*
 * The PCG32 part of this file uses the following license:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#pragma once

#include <drjit/array.h>
#include <drjit/idiv.h>
#include <drjit/math.h>
#include <drjit/while_loop.h>

NAMESPACE_BEGIN(drjit)

/**
 * \brief Implementation of PCG32, a member of the PCG family of random number
 * generators proposed by Melissa O'Neill.
 *
 * PCG32 is a stateful pseudorandom number generator that combines a linear
 * congruential generator (LCG) with a permutation function. It provides high
 * statistical quality with a remarkably fast and compact implementation.
 * Details on the PCG family of pseudorandom number generators can be found
 * `here <https://www.pcg-random.org/index.html>`__.
 *
 * To create random tensors of different sizes in Python, prefer the
 * higher-level :py:func:`dr.rng() <drjit.rng>` interface, which internally uses
 * the :py:class:`Philox4x32` generator. The properties of PCG32 makes it most
 * suitable for Monte Carlo applications requiring long sequences of random
 * variates.
 *
 * Key properties of the PCG variant implemented here include:
 *
 * * **Compact**: 128 bits total state (64-bit state + 64-bit increment)
 *
 * * **Output**: 32-bit output with a period of 2^64 per stream
 *
 * * **Streams**: Multiple independent streams via the increment parameter
 *   (with caveats, see below)
 *
 * * **Low-cost sample generation**: a single 64 bit integer multiply-add plus
 *   a bit permutation applied to the output.
 *
 * * **Extra features**: provides fast multi-step advance/rewind functionality.
 *
 * **Caveats**: PCG32 produces random high-quality variates within each random
 * number stream. For a given initial state, PCG32 can also produce multiple
 * output streams by specifying a different sequence increment (``initseq``) to
 * the constructor. However, the level of statistical independence *across
 * streams* is generally insufficient when doing so. To obtain a series of
 * high-quality independent parallel streams, it is recommended to use another
 * method (e.g., the Tiny Encryption Algorithm) to seed the `state` and `inc`
 * parameters. This ensures independence both within and across streams.
 *
 * In Python, the :py:class:`PCG32` class is implemented as a :ref:`PyTree
 * <pytrees>`, which means that it is compatible with symbolic function calls,
 * loops, etc.
 *
 * .. note::
 *
 *    Please watch out for the following pitfall when using the PCG32 class in
 *    long-running Dr.Jit calculations (e.g., steps of a gradient-based
 * optimizer). Consuming random variates (e.g., through :py:func:`next_float`)
 * changes the internal RNG state. If this state is never explicitly evaluated,
 *    the computation graph describing the state transformation keeps growing
 *    without bound, causing kernel compilation of increasingly large programs
 *    to eventually become a bottleneck. To evaluate the RNG, simply run
 *
 *    .. code-block:: python
 *
 *       rng: PCG32 = ....
 *       dr.eval(rng)
 *
 *    For computation involving very large arrays, storing the RNG state (16
 *    bytes per entry) can be prohibitive. In this case, it is better to keep
 *    the RNG in symbolic form and re-seed it at every optimization iteration.
 *
 *    In cases where a sampler is repeatedly used in a symbolic loop, it is
 *    more efficient to use the PCG32 API directly to seed once and reuse the
 *    random number generator throughout the loop.
 *
 * Comparison with \ref Philox4x32:
 *
 * * :py:class:`PCG32 <drjit.auto.PCG32>`: State-based, better for sequential
 *   generation, low per-sample cost.
 *
 * * :py:class:`Philox4x32 <drjit.auto.Philox4x32>`: Counter-based, better for
 *   parallel generation, higher per-sample cost.
 */
template <typename T> struct PCG32 {
    static constexpr uint64_t PCG32_DEFAULT_STATE  = 0x853c49e6748fea9bULL;
    static constexpr uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
    static constexpr uint64_t PCG32_MULT           = 0x5851f42d4c957f2dULL;
    static constexpr uint64_t PCG32_MULT_REV       = 0xc097ef87329e28a5ULL;

    // Type aliases for vectorization
    using  Int64     = int64_array_t<T>;
    using  Int32     = int32_array_t<T>;
    using UInt64     = uint64_array_t<T>;
    using UInt32     = uint32_array_t<T>;
    using Float16    = float16_array_t<T>;
    using Float64    = float64_array_t<T>;
    using Float32    = float32_array_t<T>;
    using Mask       = mask_t<UInt64>;

    /// Initialize the pseudorandom number generator with the \ref seed() function
    PCG32(size_t size = 1,
          const UInt64 &initstate = PCG32_DEFAULT_STATE,
          const UInt64 &initseq   = PCG32_DEFAULT_STREAM) {
        UInt64 counter = arange<UInt64>(size);
        seed(initstate + counter, initseq + counter);
    }

    /**
     * \brief Seed the pseudorandom number generator
     *
     * Specified in two parts: a state initializer and a sequence selection
     * constant (a.k.a. stream id)
     */
    void seed(const UInt64 &initstate = PCG32_DEFAULT_STATE,
              const UInt64 &initseq   = PCG32_DEFAULT_STREAM) {
        state = zeros<UInt64>();
        inc = sl<1>(initseq) | 1u;
        next_uint32();
        state += initstate;
        next_uint32();
    }

    /// Generate a uniformly distributed unsigned 32-bit random number
    DRJIT_INLINE UInt32 next_uint32() {
        UInt64 oldstate = state;

        state = fmadd(oldstate, uint64_t(PCG32_MULT), inc);

        UInt32 xorshifted = UInt32(sr<27>(sr<18>(oldstate) ^ oldstate)),
               rot = UInt32(sr<59>(oldstate));

        return (xorshifted >> rot) | (xorshifted << ((-Int32(rot)) & 31));
    }

    /// Generate previous uniformly distributed unsigned 32-bit random number
    DRJIT_INLINE UInt32 prev_uint32() {
        state = uint64_t(PCG32_MULT_REV) * (state - inc);

        UInt32 xorshifted = UInt32(sr<27>(sr<18>(state) ^ state)),
               rot = UInt32(sr<59>(state));

        return (xorshifted >> rot) | (xorshifted << ((-Int32(rot)) & 31));
    }

    /// Masked version of \ref next_uint32
    DRJIT_INLINE UInt32 next_uint32(const Mask &mask) {
        UInt64 oldstate = state;

        masked(state, mask) = fmadd(oldstate, uint64_t(PCG32_MULT), inc);

        UInt32 xorshifted = UInt32(sr<27>(sr<18>(oldstate) ^ oldstate)),
               rot = UInt32(sr<59>(oldstate));

        return (xorshifted >> rot) | (xorshifted << ((-Int32(rot)) & 31));
    }

    /// Masked version of \ref prev_uint32
    DRJIT_INLINE UInt32 prev_uint32(const Mask &mask) {
        masked(state, mask) = uint64_t(PCG32_MULT_REV) * (state - inc);

        UInt32 xorshifted = UInt32(sr<27>(sr<18>(state) ^ state)),
               rot = UInt32(sr<59>(state));

        return (xorshifted >> rot) | (xorshifted << ((-Int32(rot)) & 31));
    }

    /// Generate a uniformly distributed unsigned 64-bit random number
    DRJIT_INLINE UInt64 next_uint64() {
        /* v0, v1 computed as separate statements to ensure a consistent
           evaluation order across compilers */
        UInt32 v0 = next_uint32();
        UInt32 v1 = next_uint32();

        return UInt64(v0) | sl<32>(UInt64(v1));
    }

    /// Generate previous uniformly distributed unsigned 64-bit random number
    DRJIT_INLINE UInt64 prev_uint64() {
        /* v0, v1 computed as separate statements to ensure a consistent
           evaluation order across compilers */
        UInt32 v1 = prev_uint32();
        UInt32 v0 = prev_uint32();

        return UInt64(v0) | sl<32>(UInt64(v1));
    }

    /// Masked version of \ref next_uint64
    DRJIT_INLINE UInt64 next_uint64(const Mask &mask) {
        UInt32 v0 = next_uint32(mask);
        UInt32 v1 = next_uint32(mask);

        return UInt64(v0) | sl<32>(UInt64(v1));
    }

    /// Masked version of \ref prev_uint64
    DRJIT_INLINE UInt64 prev_uint64(const Mask &mask) {
        UInt32 v1 = prev_uint32(mask);
        UInt32 v0 = prev_uint32(mask);

        return UInt64(v0) | sl<32>(UInt64(v1));
    }

    /// Forward \ref next_uint call to the correct method based given type size
    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, uint32_t> ||
                          std::is_same_v<scalar_t<Value>, uint64_t>> = 0>
    DRJIT_INLINE Value next_uint() {
        if constexpr (std::is_same_v<scalar_t<Value>, uint64_t>)
            return next_uint64();
        else
            return next_uint32();
    }

    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, uint32_t> ||
                          std::is_same_v<scalar_t<Value>, uint64_t>> = 0>
    DRJIT_INLINE Value prev_uint() {
        if constexpr (std::is_same_v<scalar_t<Value>, uint64_t>)
            return prev_uint64();
        else
            return prev_uint32();
    }

    /// Forward \ref next_uint call to the correct method based given type size (masked version)
    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, uint32_t> ||
                          std::is_same_v<scalar_t<Value>, uint64_t>> = 0>
    DRJIT_INLINE Value next_uint(const Mask &mask) {
        if constexpr (std::is_same_v<scalar_t<Value>, uint64_t>)
            return next_uint64(mask);
        else
            return next_uint32(mask);
    }

    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, uint32_t> ||
                          std::is_same_v<scalar_t<Value>, uint64_t>> = 0>
    DRJIT_INLINE Value prev_uint(const Mask &mask) {
        if constexpr (std::is_same_v<scalar_t<Value>, uint64_t>)
            return prev_uint64(mask);
        else
            return prev_uint32(mask);
    }

    /// Generate a half precision floating point value on the half-open interval :math:`[0, 1)`.
    DRJIT_INLINE Float16 next_float16() {
        return Float16(next_float32());
    }

    /// Generate a half precision floating point value on the half-open interval :math:`[0, 1)`.
    DRJIT_INLINE Float16 prev_float16() {
        return Float16(prev_float32());
    }

    /// Masked version of \ref next_float16
    DRJIT_INLINE Float16 next_float16(const Mask &mask) {
        return Float16(next_float32(mask));
    }

    /// Masked version of \ref prev_float16
    DRJIT_INLINE Float16 prev_float16(const Mask &mask) {
        return Float16(prev_float32(mask));
    }

    /// Generate a single precision floating point value on the half-open interval :math:`[0, 1)`.
    DRJIT_INLINE Float32 next_float32() {
        return reinterpret_array<Float32>(sr<9>(next_uint32()) | 0x3f800000u) - 1.f;
    }

    /// Generate previous precision floating point value on the half-open interval :math:`[0, 1)`.
    DRJIT_INLINE Float32 prev_float32() {
        return reinterpret_array<Float32>(sr<9>(prev_uint32()) | 0x3f800000u) - 1.f;
    }

    /// Masked version of \ref next_float32
    DRJIT_INLINE Float32 next_float32(const Mask &mask) {
        return reinterpret_array<Float32>(sr<9>(next_uint32(mask)) | 0x3f800000u) - 1.f;
    }

    /// Masked version of \ref prev_float32
    DRJIT_INLINE Float32 prev_float32(const Mask &mask) {
        return reinterpret_array<Float32>(sr<9>(prev_uint32(mask)) | 0x3f800000u) - 1.f;
    }

    /**
     * \brief Generate a double precision floating point value on the half-open interval :math:`[0, 1)`.
     *
     * \remark Since the underlying random number generator produces 32 bit output,
     * only the first 32 mantissa bits will be filled (however, the resolution is still
     * finer than in \ref next_float(), which only uses 23 mantissa bits)
     */
    DRJIT_INLINE Float64 next_float64() {
        /* Trick from MTGP: generate an uniformly distributed
           double precision number in [1,2) and subtract 1. */
        return reinterpret_array<Float64>(sl<20>(UInt64(next_uint32())) |
                                          0x3ff0000000000000ull) - 1.0;
    }

    DRJIT_INLINE Float64 prev_float64() {
        /* Trick from MTGP: generate an uniformly distributed
           double precision number in [1,2) and subtract 1. */
        return reinterpret_array<Float64>(sl<20>(UInt64(prev_uint32())) |
                                          0x3ff0000000000000ull) - 1.0;
    }

    /// Masked version of next_float64
    DRJIT_INLINE Float64 next_float64(const Mask &mask) {
        return reinterpret_array<Float64>(sl<20>(UInt64(next_uint32(mask))) |
                                          0x3ff0000000000000ull) - 1.0;
    }

    DRJIT_INLINE Float64 prev_float64(const Mask &mask) {
        return reinterpret_array<Float64>(sl<20>(UInt64(prev_uint32(mask))) |
                                          0x3ff0000000000000ull) - 1.0;
    }

    /// Forward \ref next_float call to the correct method based given type size
    template <typename Value>
    DRJIT_INLINE Value next_float() {
        using Scalar = scalar_t<Value>;
        constexpr size_t Size = sizeof(Scalar);
        static_assert((Size == 2 || Size == 4 || Size == 8) &&
                      std::is_floating_point_v<Scalar>);

        if constexpr (Size == 2)
            return next_float16();
        else if constexpr (Size == 4)
            return next_float32();
        else
            return next_float64();
    }

    /// Forward \ref prev_float call to the correct method based given type size
    template <typename Value>
    DRJIT_INLINE Value prev_float() {
        using Scalar = scalar_t<Value>;
        constexpr size_t Size = sizeof(Scalar);
        static_assert((Size == 2 || Size == 4 || Size == 8) &&
                      std::is_floating_point_v<Scalar>);

        if constexpr (Size == 2)
            return prev_float16();
        else if constexpr (Size == 4)
            return prev_float32();
        else
            return prev_float64();
    }

    /// Forward \ref next_float call to the correct method based given type size (masked version)
    template <typename Value>
    DRJIT_INLINE Value next_float(const Mask &mask) {
        using Scalar = scalar_t<Value>;
        constexpr size_t Size = sizeof(Scalar);
        static_assert((Size == 2 || Size == 4 || Size == 8) &&
                      std::is_floating_point_v<Scalar>);

        if constexpr (Size == 2)
            return next_float16(mask);
        else if constexpr (Size == 4)
            return next_float32(mask);
        else
            return next_float64(mask);
    }

    /// Forward \ref prev_float call to the correct method based given type size (masked version)
    template <typename Value>
    DRJIT_INLINE Value prev_float(const Mask &mask) {
        using Scalar = scalar_t<Value>;
        constexpr size_t Size = sizeof(Scalar);
        static_assert((Size == 2 || Size == 4 || Size == 8) &&
                      std::is_floating_point_v<Scalar>);

        if constexpr (Size == 2)
            return prev_float16(mask);
        else if constexpr (Size == 4)
            return prev_float32(mask);
        else
            return prev_float64(mask);
    }

    /// Generate a normally distributed single precision floating point value
    template <typename T2, typename...Args>
    DRJIT_INLINE T2 next_float_normal(const Args&... args) {
        T2 value = next_float<T2>(args...);
        value = clip(value, Epsilon<T2>, OneMinusEpsilon<T2>);
        return -SqrtTwo<T2> * erfinv(fmadd(value, -2.f, 1.f));
    }

    /// Generate a previously generated normally distributed single precision floating point value
    template <typename T2, typename...Args>
    DRJIT_INLINE T2 prev_float_normal(const Args&... args) {
        T2 value = prev_float<T2>(args...);
        value = clip(value, Epsilon<T2>, OneMinusEpsilon<T2>);
        return -SqrtTwo<T2> * erfinv(fmadd(value, -2.f, 1.f));
    }

    DRJIT_INLINE Float16 next_float16_normal() { return Float16(next_float_normal<Float32>()); }
    DRJIT_INLINE Float32 next_float32_normal() { return next_float_normal<Float32>(); }
    DRJIT_INLINE Float64 next_float64_normal() { return next_float_normal<Float64>(); }
    DRJIT_INLINE Float16 next_float16_normal(const Mask &mask) { return Float16(next_float_normal<Float32>(mask)); }
    DRJIT_INLINE Float32 next_float32_normal(const Mask &mask) { return next_float_normal<Float32>(mask); }
    DRJIT_INLINE Float64 next_float64_normal(const Mask &mask) { return next_float_normal<Float64>(mask); }

    DRJIT_INLINE Float16 prev_float16_normal() { return Float16(prev_float_normal<Float32>()); }
    DRJIT_INLINE Float32 prev_float32_normal() { return prev_float_normal<Float32>(); }
    DRJIT_INLINE Float64 prev_float64_normal() { return prev_float_normal<Float64>(); }
    DRJIT_INLINE Float16 prev_float16_normal(const Mask &mask) { return Float16(prev_float_normal<Float32>(mask)); }
    DRJIT_INLINE Float32 prev_float32_normal(const Mask &mask) { return prev_float_normal<Float32>(mask); }
    DRJIT_INLINE Float64 prev_float64_normal(const Mask &mask) { return prev_float_normal<Float64>(mask); }

    /// Generate a uniformly distributed integer r, where 0 <= r < bound
    UInt32 next_uint32_bounded(uint32_t bound, Mask mask = true) {
        if constexpr (std::is_scalar_v<UInt64>) {
            DRJIT_MARK_USED(mask);

            /* To avoid bias, we need to make the range of the RNG a multiple of
               bound, which we do by dropping output less than a threshold.
               A naive scheme to calculate the threshold would be to do

                   uint32_t threshold = 0x1'0000'0000ull % bound;

               but 64-bit div/mod is slower than 32-bit div/mod (especially on
               32-bit platforms).  In essence, we do

                   uint32_t threshold = (0x1'0000'0000ull-bound) % bound;

               because this version will calculate the same modulus, but the LHS
               value is less than 2^32.
            */

            uint32_t threshold = (~bound + 1u) % bound;

            /* Uniformity guarantees that this loop will terminate.  In practice, it
               should usually terminate quickly; on average (assuming all bounds are
               equally likely), 82.25% of the time, we can expect it to require just
               one iteration.  In the worst case, someone passes a bound of 2^31 + 1
               (i.e., 2147483649), which invalidates almost 50% of the range.  In
               practice, bounds are typically small and only a tiny amount of the range
               is eliminated.
            */

            while (true) {
                UInt32 result = next_uint32();

                if (result >= threshold)
                    return result % bound;
            }
        } else {
            if ((bound & (bound - 1)) == 0)
                return next_uint32() % bound;

            divisor<uint32_t> div(bound);
            UInt32 threshold = imod(~bound + 1u, div);

            auto [_, rng, result] = while_loop(
                // Initial loop state
                make_tuple(mask, PCG32(*this), UInt32(0)),

                // Loop condition
                [](Mask &m, PCG32 &, UInt32 &) { return m; },

                // Loop update step
                [threshold](Mask &m, PCG32 &rng, UInt32 &result) {
                    result = rng.next_uint32();

                    /* Keep track of which SIMD lanes have already
                       finished and stop advancing the associated PRNGs */
                    m &= result < threshold;
                },

                // Descriptive label
                "drjit::PCG32::next_uint32_bounded()"
            );

            state = rng.state;

            return imod(result, div);
        }
    }

    /// Generate a uniformly distributed integer r, where 0 <= r < bound
    UInt64 next_uint64_bounded(uint64_t bound, Mask mask = true) {
        if constexpr (std::is_scalar_v<UInt64>) {
            DRJIT_MARK_USED(mask);

            uint64_t threshold = (~bound + (uint64_t) 1) % bound;

            while (true) {
                uint64_t result = next_uint64();

                if (result >= threshold)
                    return result % bound;
            }
        } else {
            if ((bound & (bound - 1)) == 0)
                return next_uint64() % bound;

            divisor<uint64_t> div(bound);
            UInt64 threshold = imod(~bound + (uint64_t) 1, div);

            auto [_, rng, result] = while_loop(
                // Initial loop state
                make_tuple(mask, PCG32(*this), UInt64(0)),

                // Loop condition
                [](Mask &active, PCG32 &, UInt64 &) { return active; },

                // Loop update step
                [threshold](Mask &active, PCG32 &rng, UInt64 &result) {
                    result = rng.next_uint64();

                    /* Keep track of which SIMD lanes have already
                       finished and stop advancing the associated PRNGs */
                    active &= result < threshold;
                },

                // Descriptive label
                "drjit::PCG32::next_uint64_bounded()"
            );

            state = rng.state;

            return imod(result, div);
        }
    }

    /// Forward \ref next_uint_bounded call to the correct method based given type size
    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, uint32_t> ||
                          std::is_same_v<scalar_t<Value>, uint64_t>> = 0>
    DRJIT_INLINE Value next_uint_bounded(scalar_t<Value> bound,
                                         const mask_t<Value> &mask = true) {
        if constexpr (std::is_same_v<scalar_t<Value>, uint64_t>)
            return next_uint64_bounded(bound, mask);
        else
            return next_uint32_bounded(bound, mask);
    }

    /**
     * \brief Multi-step advance function (jump-ahead, jump-back)
     *
     * The method used here is based on Brown, "Random Number Generation with
     * Arbitrary Stride", Transactions of the American Nuclear Society (Nov.
     * 1994). The algorithm is very similar to fast exponentiation.
     */
    PCG32 operator+(const Int64 &delta_) const {
        /* Even though delta is an unsigned integer, we can pass a signed
           integer to go backwards, it just goes "the long way round". */

        auto [cur_mult, cur_plus, acc_mult, acc_plus, delta] = while_loop(
            // Initial loop state
            make_tuple(
                /* cur_mult = */ UInt64(PCG32_MULT),
                /* cur_plus = */ UInt64(inc),
                /* acc_mult = */ UInt64(1),
                /* acc_plus = */ UInt64(0),
                /* delta    = */ UInt64(delta_)
            ),

            // Loop condition
            [](UInt64 &, UInt64 &, UInt64 &, UInt64 &, UInt64 &delta) {
                return delta != 0;
            },

            // Loop update step
            [](UInt64 &cur_mult, UInt64 &cur_plus, UInt64 &acc_mult,
               UInt64 &acc_plus, UInt64 &delta) {
                Mask mask = (delta & 1) != 0;
                delta = sr<1>(delta);

                masked(acc_mult, mask) *= cur_mult;
                masked(acc_plus, mask) = fmadd(acc_plus, cur_mult, cur_plus);
                cur_plus *= cur_mult + 1;
                cur_mult *= cur_mult;
            }
        );

        return PCG32(initialize_state{}, acc_mult * state + acc_plus, inc);
    }

    PCG32 operator-(const Int64 &delta) const {
        return operator+(-delta);
    }

    PCG32 &operator+=(const Int64 &delta) { *this = operator+(delta); return *this; }
    PCG32 &operator-=(const Int64 &delta) { *this = operator+(-delta); return *this; }

    /// Compute the distance between two PCG32 pseudorandom number generators
    Int64 operator-(const PCG32 &other) const {
        UInt64 state_value = state;

        auto [cur_state, cur_plus, cur_mult, distance, bit] = while_loop(
            // Initial loop state
            make_tuple(
                /* cur_state = */ UInt64(other.state),
                /* cur_plus  = */ UInt64(inc),
                /* cur_mult  = */ UInt64(PCG32_MULT),
                /* distance  = */ UInt64(0),
                /* bit       = */ UInt64(1)
            ),

            // Loop condition
            [state_value](UInt64 &cur_state, UInt64 &, UInt64 &, UInt64 &, UInt64 &) {
                return cur_state != state_value;
            },

            // Loop update step
            [state_value](UInt64 &cur_state, UInt64 &cur_plus, UInt64 &cur_mult,
               UInt64 &distance, UInt64 &bit) {
                Mask mask = (state_value & bit) != (cur_state & bit);
                masked(cur_state, mask) = fmadd(cur_state, cur_mult, cur_plus);
                masked(distance, mask) |= bit;
                cur_plus *= cur_mult + 1;
                cur_mult *= cur_mult;
                bit = sl<1>(bit);
            }
        );

        return Int64(distance);
    }

    /**
     * \brief Draw uniformly distributed permutation and permute the
     * given container
     *
     * From: Knuth, TAoCP Vol. 2 (3rd 3d), Section 3.4.2
     */
    template <typename Iterator, typename T2 = T,
              enable_if_t<std::is_scalar_v<T2>> = 0>
    void shuffle(Iterator begin, Iterator end) {
        for (Iterator it = end - 1; it > begin; --it)
            std::swap(*it, *(begin + next_uint32_bounded((uint32_t) (it - begin + 1))));
    }

    /// Equality operator
    bool operator==(const PCG32 &other) const { return state == other.state && inc == other.inc; }

    /// Inequality operator
    bool operator!=(const PCG32 &other) const { return state != other.state || inc != other.inc; }

public:
    UInt64 state; //< RNG state.  All values are possible.
    UInt64 inc;   //< Controls which RNG sequence (stream) is selected. Must *always* be odd.

    DRJIT_STRUCT_NODEF(PCG32, state, inc)

private:
    struct initialize_state { };
    PCG32(initialize_state, const UInt64 &state, const UInt64 &inc)
        : state(state), inc(inc) { }
};

/**
 * \brief Philox4x32 counter-based PRNG
 *
 * This class implements the Philox 4x32 counter-based pseudo-random number
 * generator based on the paper `Parallel Random Numbers: As Easy as 1, 2, 3
 * <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>`__ by
 * Salmon et al. [2011]. It uses strength-reduced cryptographic primitives to
 * realize a complex transition function that turns a seed and set of counter
 * values onto 4 pseudorandom outputs. Incrementing any of the counters or
 * choosing a different seed produces statistically independent samples.
 *
 * The implementation here uses a reduced number of bits (32) for the
 * arithmetic and sets the default number of rounds to 7. However, even with
 * these simplifications it passes the `Test01
 * <https://en.wikipedia.org/wiki/TestU01>`__ stringent ``BigCrush`` tests (a
 * battery of statistical tests for non-uniformity and correlations). Please
 * see the paper `Random number generators for massively parallel simulations
 * on GPU <https://arxiv.org/abs/1204.6193>`__ by Manssen et al. [2012] for
 * details.
 *
 * Functions like :py:func:`next_uint32x4()` or :py:func:`next_float32x4()`
 * advance the PRNG state by incrementing the counter ``counter[3]``.
 *
 * Key properties include:
 * - Counter-based design: generation from counter + key
 * - 192-bit bit state: 4x32-bit counters, 64-bit key
 * - Trivial jump-ahead capability through counter manipulation
 *
 * The :py:class:`Philox4x32` class is implemented as a :ref:`PyTree <pytrees>`,
 * making it compatible with symbolic function calls, loops, etc.
 *
 * .. note::
 *
 *    :py:class:`Philox4x32` naturally produces 4 samples at a time, which may
 *    be awkward for applications that need individual random values.
 *
 * .. note::
 *
 *    For a comparison of use cases between :py:class:`Philox4x32` and
 *    :py:class:`PCG32`, see the :py:class:`PCG32` class documentation. In
 *    brief: use :py:class:`PCG32` for sequential generation with lowest cost
 *    per sample; use :py:class:`Philox4x32` for parallel generation where
 *    independent streams are critical.
 */
template <typename T> struct Philox4x32 {
    /// Multipliers
    static constexpr uint32_t PHILOX_M0 = 0xD2511F53;
    static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57;

    // Key offsets
    static constexpr uint32_t PHILOX_W0 = 0x9E3779B9;
    static constexpr uint32_t PHILOX_W1 = 0xBB67AE85;

    // Type aliases for vectorization
    using  Int64    = int64_array_t<T>;
    using  Int32    = int32_array_t<T>;
    using UInt64    = uint64_array_t<T>;
    using UInt32    = uint32_array_t<T>;
    using Float16   = float16_array_t<T>;
    using Float32   = float32_array_t<T>;
    using Float64   = float64_array_t<T>;
    using Array2u   = Array<UInt32, 2>;
    using Array2u64 = Array<UInt64, 2>;
    using Array4u   = Array<UInt32, 4>;
    using Array4f   = Array<Float32, 4>;
    using Array4f16 = Array<Float16, 4>;
    using Array2f64 = Array<Float64, 2>;
    using Mask      = mask_t<UInt64>;

    /**
     * \brief Construct a Philox PRNG
     *
     * The function takes a ``seed`` and three of four ``counter`` component.
     * The last component is zero-initialized and incremented by calls to the
     * ``sample_*`` methods.
     *
     * \param seed The 64-bit seed value used as the key for the Philox step function
     * \param counter_0 The first 32-bit counter (least significant)
     * \param counter_1 The second 32-bit counter (default: 0)
     * \param counter_2 The third 32-bit counter (default: 0)
     * \param iterations Number of rounds (default: 7)
     */
    Philox4x32(const UInt64 &seed,
               const UInt32 &counter_0,
               const UInt32 &counter_1 = 0,
               const UInt32 &counter_2 = 0,
               uint32_t iterations = 7)
        : seed(UInt32(seed >> 32), UInt32(seed)),
          counter(counter_0, counter_1, counter_2, 0),
          iterations(iterations) { }

    /**
     * \brief Generate 4 random 32-bit unsigned integers
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 4 independent 32-bit random values.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array4u next_uint32x4(const Mask &mask = true) {
        auto [state, _1, _2] = while_loop(
            // Initial loop state
            make_tuple(counter, seed, UInt32(0)),

            // Loop condition
            [iterations=iterations](Array4u &, Array2u &, UInt32 &it) {
                return it < iterations;
            },

            // Loop update step
            [](Array4u &state, Array2u &key, UInt32 &it) {
                state = step(state, key);
                key += Array2u(PHILOX_W0, PHILOX_W1);
                it += 1;
            },

            // Descriptive label
            "drjit::Philox4x32::next_uint32x4()"
        );

        counter[3] = select(mask, counter[3] + 1, counter[3]);

        return state;
    }

    /**
     * \brief Generate 2 random 64-bit unsigned integers
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 2 independent 64-bit random values.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array2u64 next_uint64x2(const Mask &mask = true) {
        Array4u result = next_uint32x4(mask);
        return Array2u64(
            (UInt64(result.y()) << 32) | result.x(),
            (UInt64(result.w()) << 32) | result.z()
        );
    }

    /** \brief Generate 4 uniformly distributed half-precision floats in on
     * the half-open interval :math:`[0, 1)`.
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 4 independent half precision floats.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array4f16 next_float16x4(const Mask &mask = true) {
        return Array4f16(next_float32x4(mask));
    }

    /** \brief Generate 4 uniformly distributed single-precision floats in on
     * the half-open interval :math:`[0, 1)`.
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 4 independent single precision floats.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array4f next_float32x4(const Mask &mask = true) {
        return reinterpret_array<Array4f>(sr<9>(next_uint32x4(mask)) |
                                          0x3f800000u) - 1.f;
    }

    /** \brief Generate 2 uniformly distributed double-precision floats in on
     * the half-open interval :math:`[0, 1)`.
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 2 independent double precision floats.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array2f64 next_float64x2(const Mask &mask = true) {
        return reinterpret_array<Array2f64>(sr<12>(next_uint64x2(mask)) |
                                            0x3ff0000000000000ull) - 1.0;
    }

    /** \brief Generate 4 normally distributed half-precision floats
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 4 half precision floats following a standard normal distribution.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array4f16 next_float16x4_normal(const Mask &mask = true) {
        return Array4f16(next_float32x4_normal(mask));
    }

    /** \brief Generate 4 normally distributed single-precision floats
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 4 single precision floats following a standard normal distribution.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array4f next_float32x4_normal(const Mask &mask = true) {
        Array4f value = next_float32x4(mask);
        auto [s0, c0] = sincos(TwoPi<Float32> * value.x());
        auto [s1, c1] = sincos(TwoPi<Float32> * value.y());
        Float32 scale0 = sqrt(-log2(1.f - value.z())) * sqrt(2 * LogTwo<Float32>),
                scale1 = sqrt(-log2(1.f - value.w())) * sqrt(2 * LogTwo<Float32>);
        return Array4f(c0 * scale0, s0 * scale0, c1 * scale1, s1 * scale1);
    }

    /** \brief Generate 2 normally distributed double-precision floats
     *
     * Advances the internal counter and applies the Philox mapping to
     * produce 2 double precision floats following a standard normal distribution.
     *
     * \param mask Optional mask to control which lanes are updated
     */
    Array2f64 next_float64x2_normal(const Mask &mask = true) {
        Array2f64 value = next_float64x2(mask);
        auto [s, c] = sincos(TwoPi<Float64> * value.y());
        Float64 scale = sqrt(-log2(1.0 - value.x())) * sqrt(2 * LogTwo<Float64>);
        return Array2f64(c * scale, s * scale);
    }

public:
    Array2u seed;
    Array4u counter;
    uint32_t iterations;

    DRJIT_STRUCT_NODEF(Philox4x32, iterations, seed, counter)

private:
    static Array4u step(const Array4u &state, const Array2u &key) {
        UInt64 hilo0 = mul_wide(PHILOX_M0, state[0]),
               hilo1 = mul_wide(PHILOX_M1, state[2]);

        UInt32 lo0 = UInt32(hilo0),
               lo1 = UInt32(hilo1),
               hi0 = UInt32(hilo0 >> 32),
               hi1 = UInt32(hilo1 >> 32);

        return Array4u(
            hi1 ^ state[1] ^ key[0],
            lo1,
            hi0 ^ state[3] ^ key[1],
            lo0
        );
    }
};

NAMESPACE_END(drjit)
