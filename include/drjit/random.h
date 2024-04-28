/*
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
#include <drjit/while_loop.h>

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

NAMESPACE_BEGIN(drjit)

/// PCG32 pseudorandom number generator proposed by Melissa O'Neill
template <typename T> struct PCG32 {
    /* Some convenient type aliases for vectorization */
    using  Int64     = int64_array_t<T>;
    using  Int32     = int32_array_t<T>;
    using UInt64     = uint64_array_t<T>;
    using UInt32     = uint32_array_t<T>;
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

    /// Masked version of \ref next_uint32
    DRJIT_INLINE UInt32 next_uint32(const Mask &mask) {
        UInt64 oldstate = state;

        masked(state, mask) = fmadd(oldstate, uint64_t(PCG32_MULT), inc);

        UInt32 xorshifted = UInt32(sr<27>(sr<18>(oldstate) ^ oldstate)),
               rot = UInt32(sr<59>(oldstate));

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

    /// Masked version of \ref next_uint64
    DRJIT_INLINE UInt64 next_uint64(const Mask &mask) {
        UInt32 v0 = next_uint32(mask);
        UInt32 v1 = next_uint32(mask);

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

    /// Generate a single precision floating point value on the interval [0, 1)
    DRJIT_INLINE Float32 next_float32() {
        return reinterpret_array<Float32>(sr<9>(next_uint32()) | 0x3f800000u) - 1.f;
    }

    /// Masked version of \ref next_float32
    DRJIT_INLINE Float32 next_float32(const Mask &mask) {
        return reinterpret_array<Float32>(sr<9>(next_uint32(mask)) | 0x3f800000u) - 1.f;
    }

    /**
     * \brief Generate a double precision floating point value on the interval [0, 1)
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

    /// Masked version of next_float64
    DRJIT_INLINE Float64 next_float64(const Mask &mask) {
        return reinterpret_array<Float64>(sl<20>(UInt64(next_uint32(mask))) |
                                          0x3ff0000000000000ull) - 1.0;
    }

    /// Forward \ref next_float call to the correct method based given type size
    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, float> ||
                          std::is_same_v<scalar_t<Value>, double>> = 0>
    DRJIT_INLINE Value next_float() {
        if constexpr (std::is_same_v<scalar_t<Value>, double>)
            return next_float64();
        else
            return next_float32();
    }

    /// Forward \ref next_float call to the correct method based given type size (masked version)
    template <typename Value,
              enable_if_t<std::is_same_v<scalar_t<Value>, float> ||
                          std::is_same_v<scalar_t<Value>, double>> = 0>
    DRJIT_INLINE Value next_float(const Mask &mask) {
        if constexpr (std::is_same_v<scalar_t<Value>, double>)
            return next_float64(mask);
        else
            return next_float32(mask);
    }

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

    UInt64 state;  // RNG state.  All values are possible.
    UInt64 inc;    // Controls which RNG sequence (stream) is selected. Must *always* be odd.

    DRJIT_STRUCT_NODEF(PCG32, state, inc)
private:
    struct initialize_state { };
    PCG32(initialize_state, const UInt64 &state, const UInt64 &inc)
        : state(state), inc(inc) { }
};

NAMESPACE_END(drjit)
