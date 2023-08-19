
    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray and_(const MaskType &mask) const {
        return select(mask, *this, DiffArray(Scalar(0)));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray or_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            const Scalar value = memcpy_cast<Scalar>(int_array_t<Scalar>(-1));
            if (m_index)
                return select(mask, DiffArray(value), *this);
        }
        return DiffArray::create(0, detail::or_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray xor_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
                drjit_raise("xor_(): operation not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray andnot_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
                drjit_raise("andnot_(): operation not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::andnot_(m_value, mask.m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Transcendental functions
    // -----------------------------------------------------------------------

    DiffArray sin_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sin_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    index_new = detail::ad_new<Type>("sin", width(s),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cos_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -s };
                    index_new = detail::ad_new<Type>("cos", width(c),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sincos_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            uint32_t index_s = 0, index_c = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights_s[1] = { c }, weights_c[1] = { -s };
                    uint32_t w = (uint32_t) width(s);
                    index_s = detail::ad_new<Type>("sincos[s]", w, 1, indices,
                                                   weights_s);
                    index_c = detail::ad_new<Type>("sincos[c]", w, 1, indices,
                                                   weights_c);
                }
            }
            return {
                DiffArray::create(index_s, std::move(s)),
                DiffArray::create(index_c, std::move(c)),
            };
        }
    }

    DiffArray csc_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("csc_(): invalid operand type!");
        } else {
            Type result = csc(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -result * cot(m_value) };
                    index_new = detail::ad_new<Type>(
                        "csc", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sec_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sec_(): invalid operand type!");
        } else {
            Type result = sec(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { result * tan(m_value) };
                    index_new = detail::ad_new<Type>(
                        "sec", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray tan_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("tan_(): invalid operand type!");
        } else {
            Type result = tan(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { sqr(sec(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "tan", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray cot_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cot_(): invalid operand type!");
        } else {
            Type result = cot(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -sqr(csc(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "cot", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asin_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("asin_(): invalid operand type!");
        } else {
            Type result = asin(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt(fnmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "asin", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray acos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("acos_(): invalid operand type!");
        } else {
            Type result = acos(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -rsqrt(fnmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "acos", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atan_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("atan_(): invalid operand type!");
        } else {
            Type result = atan(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(fmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "atan", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atan2_(const DiffArray &x) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("atan2_(): invalid operand type!");
        } else {
            Type result = atan2(m_value, x.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || x.m_index) {
                    Type il2 = rcp(fmadd(m_value, m_value, sqr(x.m_value)));
                    uint32_t indices[2] = { m_index, x.m_index };
                    Type weights[2] = { il2 * x.m_value, -il2 * m_value };
                    index_new = detail::ad_new<Type>(
                        "atan2", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray exp_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("exp_(): invalid operand type!");
        } else {
            Type result = exp(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { result };
                    index_new = detail::ad_new<Type>(
                        "exp", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray exp2_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("exp2_(): invalid operand type!");
        } else {
            Type result = exp2(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { result * LogTwo<Value> };
                    index_new = detail::ad_new<Type>(
                        "exp2", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("log_(): invalid operand type!");
        } else {
            Type result = log(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(m_value) };
                    index_new = detail::ad_new<Type>(
                        "log", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log2_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("log2_(): invalid operand type!");
        } else {
            Type result = log2(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(m_value) * InvLogTwo<Value> };
                    index_new = detail::ad_new<Type>(
                        "log2", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sinh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sinh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    index_new = detail::ad_new<Type>("sinh", width(s),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cosh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { s };
                    index_new = detail::ad_new<Type>("cosh", width(c),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sincosh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            uint32_t index_s = 0, index_c = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights_s[1] = { c }, weights_c[1] = { s };
                    size_t w = width(s);
                    index_s =
                        detail::ad_new<Type>("sincosh[s]", w, 1, indices, weights_s);
                    index_c =
                        detail::ad_new<Type>("sincosh[c]", w, 1, indices, weights_c);
                }
            }
            return {
                DiffArray::create(index_s, std::move(s)),
                DiffArray::create(index_c, std::move(c)),
            };
        }
    }

    DiffArray tanh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("tanh_(): invalid operand type!");
        } else {
            Type result = tanh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { sqr(sech(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "tanh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asinh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("asinh_(): invalid operand type!");
        } else {
            Type result = asinh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt((Scalar) 1 + sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "asinh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray acosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("acosh_(): invalid operand type!");
        } else {
            Type result = acosh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt(sqr(m_value) - (Scalar) 1) };
                    index_new = detail::ad_new<Type>(
                        "acosh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atanh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("atanh_(): invalid operand type!");
        } else {
            Type result = atanh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp((Scalar) 1 - sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "atanh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Operations that don't require derivatives
    // -----------------------------------------------------------------------


    DiffArray or_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("or_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::or_(m_value, a.m_value));
    }

    DiffArray and_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("and_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::and_(m_value, a.m_value));
    }

    DiffArray xor_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("xor_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, a.m_value));
    }

    DiffArray andnot_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("andnot_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::andnot_(m_value, a.m_value));
    }

    DiffArray floor_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("floor_(): invalid operand type!");
        else
            return DiffArray::create(0, floor(m_value));
    }

    DiffArray ceil_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("ceil_(): invalid operand type!");
        else
            return DiffArray::create(0, ceil(m_value));
    }

    DiffArray trunc_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("trunc_(): invalid operand type!");
        else
            return DiffArray::create(0, trunc(m_value));
    }

    DiffArray round_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("round_(): invalid operand type!");
        else
            return DiffArray::create(0, round(m_value));
    }

    template <typename T> T ceil2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("ceil2int_(): invalid operand type!");
        else
            return T::create(0, ceil2int<typename T::Type>(m_value));
    }

    template <typename T> T floor2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("floor2int_(): invalid operand type!");
        else
            return T::create(0, floor2int<typename T::Type>(m_value));
    }

    template <typename T> T round2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("round2int_(): invalid operand type!");
        else
            return T::create(0, round2int<typename T::Type>(m_value));
    }

    template <typename T> T trunc2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("trunc2int_(): invalid operand type!");
        else
            return T::create(0, trunc2int<typename T::Type>(m_value));
    }

    DiffArray sl_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sl_(): invalid operand type!");
        else
            return DiffArray::create(0, m_value << a.m_value);
    }

    DiffArray sr_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sr_(): invalid operand type!");
        else
            return DiffArray::create(0, m_value >> a.m_value);
    }

    template <int Imm> DiffArray sl_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sl_(): invalid operand type!");
        else
            return DiffArray::create(0, sl<Imm>(m_value));
    }

    template <int Imm> DiffArray sr_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sr_(): invalid operand type!");
        else
            return DiffArray::create(0, sr<Imm>(m_value));
    }

    DiffArray tzcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("tzcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, tzcnt(m_value));
    }

    DiffArray lzcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("lzcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, lzcnt(m_value));
    }

    DiffArray popcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("popcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, popcnt(m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const {
        if constexpr (!is_mask_v<Type>)
            drjit_raise("all_(): invalid operand type!");
        else
            return all(m_value);
    }

    bool any_() const {
        if constexpr (!is_mask_v<Type>)
            drjit_raise("any_(): invalid operand type!");
        else
            return any(m_value);
    }

    size_t count_() const {
        if constexpr (!is_mask_v<Type>)
            drjit_raise("count_(): invalid operand type!");
        else
            return count(m_value);
    }

    DiffArray sum_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("sum_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { 1 };
                    index_new = detail::ad_new<Type>(
                        "sum", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, sum(m_value));
        }
    }

    DiffArray prod_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("prod_(): invalid operand type!");
        } else {
            Type result = prod(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { select(eq(m_value, (Scalar) 0),
                                               (Scalar) 0, result / m_value) };
                    index_new = detail::ad_new<Type>(
                        "prod", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray min_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("min_(): invalid operand type!");
        } else {
            Type result = min(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the minimum , which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { select(
                        eq(m_value, result), Type(1), Type(0)) };
                    index_new = detail::ad_new<Type>(
                        "min", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray max_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("max_(): invalid operand type!");
        } else {
            Type result = max(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the maximum, which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { select(
                        eq(m_value, result), Type(1), Type(0)) };
                    index_new = detail::ad_new<Type>(
                        "max", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray dot_(const DiffArray &a) const {
        return sum(*this * a);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather operations
    // -----------------------------------------------------------------------

    template <bool Permute>
    static DiffArray gather_(const DiffArray &src, const IndexType &offset,
                             const MaskType &mask = true) {
        if constexpr (std::is_scalar_v<Type>) {
            drjit_raise("Array gather operation not supported for scalar array type.");
        } else {
            Type result = gather<Type>(src.m_value, offset.m_value, mask.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (src.m_index)
                    index_new = detail::ad_new_gather<Type>(
                        Permute ? "gather[permute]" : "gather",
                        width(result), src.m_index, offset.m_value,
                        mask.m_value, Permute);
            }
            return create(index_new, std::move(result));
        }
    }

    template <bool Permute>
    void scatter_(DiffArray &dst, const IndexType &offset,
                  const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            (void) dst; (void) offset; (void) mask;
            drjit_raise("Array scatter operation not supported for scalar array type.");
        } else {
            scatter(dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index || (dst.m_index && !Permute)) {
                    uint32_t index = detail::ad_new_scatter<Type>(
                        Permute ? "scatter[permute]" : "scatter", width(dst),
                        ReduceOp::None, m_index, dst.m_index, offset.m_value,
                        mask.m_value, Permute);
                    detail::ad_dec_ref<Type>(dst.m_index);
                    dst.m_index = index;
                }
            }
        }
    }

    void scatter_reduce_(ReduceOp op, DiffArray &dst, const IndexType &offset,
                         const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            (void) op; (void) dst; (void) offset; (void) mask;
            drjit_raise("Array scatter_reduce operation not supported for scalar array type.");
        } else {
            scatter_reduce(op, dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index) { // safe to ignore dst.m_index in the case of scatter_reduce
                    uint32_t index = detail::ad_new_scatter<Type>(
                        "scatter_reduce", width(dst), op, m_index,
                        dst.m_index, offset.m_value, mask.m_value, false);
                    detail::ad_dec_ref<Type>(dst.m_index);
                    dst.m_index = index;
                }
            }
        }
    }

    void scatter_reduce_kahan_(DiffArray &dst_1, DiffArray &dst_2, const IndexType &offset,
                         const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            (void) dst_1; (void) dst_2; (void) offset; (void) mask;
            drjit_raise("Array scatter_reduce operation not supported for scalar array type.");
        } else {
            scatter_reduce_kahan(dst_1.m_value, dst_2.m_value, m_value,
                                 offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index) { // safe to ignore dst_1.m_index in the case of scatter_reduce
                    uint32_t index = detail::ad_new_scatter<Type>(
                        "scatter_reduce_kahan", width(dst_1), ReduceOp::Add,
                        m_index, dst_1.m_index, offset.m_value, mask.m_value,
                        false);
                    detail::ad_dec_ref<Type>(dst_1.m_index);
                    dst_1.m_index = index;
                }
            }
        }
    }

    template <bool>
    static DiffArray gather_(const void *src, const IndexType &offset,
                             const MaskType &mask = true) {
        return create(0, gather<Type>(src, offset.m_value, mask.m_value));
    }

    template <bool>
    void scatter_(void *dst, const IndexType &offset,
                  const MaskType &mask = true) const {
        scatter(dst, m_value, offset.m_value, mask.m_value);
    }

    void scatter_reduce_(ReduceOp op, void *dst, const IndexType &offset,
                         const MaskType &mask = true) const {
        scatter_reduce(op, dst, m_value, offset.m_value, mask.m_value);
    }

    auto compress_() const {
        if constexpr (!is_mask_v<Type>)
            drjit_raise("compress_(): invalid operand type!");
        else
            return uint32_array_t<ArrayType>::create(0, compress(m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Standard initializers
    // -----------------------------------------------------------------------

    static DiffArray empty_(size_t size) {
        return drjit::empty<Type>(size);
    }

    static DiffArray zero_(size_t size) {
        return zeros<Type>(size);
    }

    static DiffArray full_(Value value, size_t size) {
        return full<Type>(value, size);
    }

    static DiffArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        return arange<Type>(start, stop, step);
    }

    static DiffArray linspace_(Value min, Value max, size_t size, bool endpoint) {
        return linspace<Type>(min, max, size, endpoint);
    }

    static DiffArray map_(void *ptr, size_t size, bool free = false) {
        DRJIT_MARK_USED(size);
        DRJIT_MARK_USED(free);
        DRJIT_MARK_USED(ptr);
        if constexpr (is_jit_v<Type>)
            return Type::map_(ptr, size, free);
        else
            drjit_raise("map_(): not supported in scalar mode!");
    }

    static DiffArray load_(const void *ptr, size_t size) {
        return load<Type>(ptr, size);
    }

    void store_(void *ptr) const {
        store(ptr, m_value);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    DiffArray copy() const {
        if constexpr (IsEnabled) {
            if (m_index) {
                uint32_t indices[1] = { m_index };
                Type weights[1] = { 1 };

                uint32_t index_new = detail::ad_new<Type>(
                    "copy", width(m_value), 1, indices, weights);

                return DiffArray::create(index_new, std::move(Type(m_value)));
            }
        }

        return *this;
    }

    auto vcall_() const {
        if constexpr (is_jit_v<Type>)
            return m_value.vcall_();
        else
            drjit_raise("vcall_(): not supported in scalar mode!");
    }

    DiffArray block_sum_(size_t block_size) {
        if constexpr (is_jit_v<Type>) {
            if (m_index)
                drjit_raise("block_sum_(): not supported for attached arrays!");
            return m_value.block_sum_(block_size);
        } else {
            DRJIT_MARK_USED(block_size);
            drjit_raise("block_sum_(): not supported in scalar mode!");
        }
    }

    static DiffArray steal(uint32_t index) {
        DRJIT_MARK_USED(index);
        if constexpr (is_jit_v<Type>)
            return Type::steal(index);
        else
            drjit_raise("steal(): not supported in scalar mode!");
    }

    static DiffArray borrow(uint32_t index) {
        DRJIT_MARK_USED(index);
        if constexpr (is_jit_v<Type>)
            return Type::borrow(index);
        else
            drjit_raise("borrow(): not supported in scalar mode!");
    }

    bool grad_enabled_() const {
        if constexpr (IsEnabled) {
            if (!m_index)
                return false;
            else
                return detail::ad_grad_enabled<Type>(m_index);
        } else {
            return false;
        }
    }

    void set_grad_enabled_(bool value) {
        DRJIT_MARK_USED(value);
        if constexpr (IsEnabled) {
            if (value) {
                if (m_index)
                    return;
                m_index = detail::ad_new<Type>(nullptr, width(m_value));
                if constexpr (is_jit_v<Type>) {
                    const char *label = m_value.label_();
                    if (label)
                        detail::ad_set_label<Type>(m_index, label);
                }
            } else {
                if (m_index == 0)
                    return;
                detail::ad_dec_ref<Type>(m_index);
                m_index = 0;
            }
        }
    }

    DiffArray migrate_(AllocType type) const {
        DRJIT_MARK_USED(type);
        if constexpr (is_jit_v<Type_>)
            return m_value.migrate_(type);
        else
            return *this;
    }

    bool schedule_() const {
        if constexpr (is_jit_v<Type_>)
            return m_value.schedule_();
        else
            return false;
    }

    bool eval_() const {
        if constexpr (is_jit_v<Type_>)
            return m_value.eval_();
        else
            return false;
    }

    void enqueue_(ADMode mode) const {
        DRJIT_MARK_USED(mode);
        if constexpr (IsEnabled)
            detail::ad_enqueue<Type>(mode, m_index);
    }

    static void traverse_(ADMode mode, uint32_t flags) {
        DRJIT_MARK_USED(flags);
        if constexpr (IsEnabled)
            detail::ad_traverse<Type>(mode, flags);
    }

    void set_label_(const char *label) {
        set_label(m_value, label);

        if constexpr (IsEnabled) {
            if (m_index)
                detail::ad_set_label<Type>(m_index, label);
        }
    }

    const char *label_() const {
        const char *result = nullptr;
        if constexpr (IsEnabled) {
            if (m_index)
                result = detail::ad_label<Type>(m_index);
        }
        if constexpr (is_jit_v<Type>) {
            if (!result)
                result = m_value.label_();
        }
        return result;
    }

    static const char *graphviz_() {
        if constexpr (IsEnabled)
            return detail::ad_graphviz<Type>();
    }

    const Type &detach_() const {
        return m_value;
    }

    Type &detach_() {
        return m_value;
    }

    const Type grad_(bool fail_if_missing = false) const {
        DRJIT_MARK_USED(fail_if_missing);
        if constexpr (IsEnabled)
            return detail::ad_grad<Type>(m_index, fail_if_missing);
        else
            return zeros<Type>();
    }

    void set_grad_(const Type &value, bool fail_if_missing = false) {
        DRJIT_MARK_USED(value);
        DRJIT_MARK_USED(fail_if_missing);
        if constexpr (IsEnabled)
            detail::ad_set_grad<Type>(m_index, value, fail_if_missing);
    }

    void accum_grad_(const Type &value, bool fail_if_missing = false) {
        DRJIT_MARK_USED(value);
        DRJIT_MARK_USED(fail_if_missing);
        if constexpr (IsEnabled)
            detail::ad_accum_grad<Type>(m_index, value, fail_if_missing);
    }

    size_t size() const {
        if constexpr (std::is_scalar_v<Type>)
            return 1;
        else
            return m_value.size();
    }

    Value entry(size_t offset) const {
        DRJIT_MARK_USED(offset);
        if constexpr (std::is_scalar_v<Type>)
            return m_value;
        else
            return m_value.entry(offset);
    }

    void set_entry(size_t offset, Value value) {
        if (m_index)
            drjit_raise("Attempted to overwrite entries of a variable that is "
                        "attached to the AD graph. This is not allowed.");

        if constexpr (is_dynamic_v<Type_>) {
            m_value.set_entry(offset, value);
        } else {
            DRJIT_MARK_USED(offset);
#if !defined(NDEBUG) && !defined(DRJIT_DISABLE_RANGE_CHECK)
            if (offset != 0)
                drjit_raise("Out of range access (tried to access index %u in "
                            "an array of size 1)", offset);
#endif
            m_value = value;
        }
    }

    void resize(size_t size) {
        DRJIT_MARK_USED(size);
        if constexpr (is_dynamic_v<Type>)
            m_value.resize(size);
    }

    Scalar *data() {
        if constexpr (std::is_scalar_v<Type>)
            return &m_value;
        else
            return m_value.data();
    }

    const Scalar *data() const {
        if constexpr (std::is_scalar_v<Type>)
            return &m_value;
        else
            return m_value.data();
    }

    static DiffArray create(uint32_t index, Type&& value) {
        DiffArray result;
        result.m_index = index;
        result.m_value = std::move(value);
        return result;
    }

    static DiffArray create_borrow(uint32_t index, const Type &value) {
        DiffArray result;
        result.m_index = index;
        result.m_value = value;
        if constexpr (IsEnabled)
            detail::ad_inc_ref<Type>(index);
        return result;
    }

    void init_(size_t size) {
        DRJIT_MARK_USED(size);
        if constexpr (is_dynamic_v<Type>)
            m_value.init_(size);
    }

    bool is_literal() const {
        if constexpr (is_jit_v<Type>)
            return m_value.is_literal();
        else
            drjit_raise("is_literal(): expected a JIT array type");
    }

    bool is_evaluated() const {
        if constexpr (is_jit_v<Type>)
            return m_value.is_evaluated();
        else
            drjit_raise("is_evaluated(): expected a JIT array type");
    }

    uint32_t index() const {
        if constexpr (is_jit_v<Type>)
            return m_value.index();
        else
            drjit_raise("index(): expected a JIT array type");
    }

    uint32_t* index_ptr() {
        if constexpr (is_jit_v<Type>)
            return m_value.index_ptr();
        else
            drjit_raise("index_ptr(): expected a JIT array type");
    }

    uint32_t index_ad() const { return m_index; }
    uint32_t* index_ad_ptr() { return &m_index; }

