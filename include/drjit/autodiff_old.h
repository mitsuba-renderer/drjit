
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
