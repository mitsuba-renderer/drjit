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

