#include "common.h"
#include "quat.h"
#include <drjit/autodiff.h>
#include <drjit/sphere.h>

template <typename T> void export_t(nb::module_ &m) {
    m.def(
        "quat_to_matrix",
        [](nb::type_object_t<dr::ArrayBase> dtype, const dr::Quaternion<T> &x) -> nb::object {
            nb::handle dtype_3 = nb::type<dr::Matrix<T, 3>>(),
                       dtype_4 = nb::type<dr::Matrix<T, 4>>();
            if (dtype.is(dtype_3))
                return nb::cast(dr::quat_to_matrix<dr::Matrix<T, 3>>(x));
            else if (dtype.is(dtype_4))
                return nb::cast(dr::quat_to_matrix<dr::Matrix<T, 4>>(x));
            else
                nb::raise_type_error("dtype must be '%s' or '%s' (got '%s')",
                                     nb::type_name(dtype_3).c_str(),
                                     nb::type_name(dtype_4).c_str(),
                                     nb::type_name(dtype).c_str());
        },
        nb::sig("def quat_to_matrix(dtype: typing.Type[ArrayT], q: dr.ArrayBase) -> ArrayT"),
        "dtype"_a, "q"_a, doc_quat_to_matrix);

    m.def(
        "matrix_to_quat",
        [](const dr::Matrix<T, 3> &x) { return dr::matrix_to_quat(x); },
        doc_matrix_to_quat);

    m.def(
        "matrix_to_quat",
        [](const dr::Matrix<T, 4> &x) { return dr::matrix_to_quat(x); },
        doc_matrix_to_quat);

    m.def(
        "quat_to_euler",
        [](const dr::Quaternion<T> &x) { return dr::quat_to_euler(x); },
        doc_quat_to_euler);

    m.def(
        "euler_to_quat",
        [](const dr::Array<T, 3> &x) { return dr::euler_to_quat(x); },
        doc_euler_to_quat);

    m.def(
        "slerp",
        [](dr::Quaternion<T> &a, const dr::Quaternion<T> &b, nb::handle t) {
            return dr::slerp(a, b, nb::cast<T>(t));
        },
        nb::sig("def slerp(a: ArrayT, b: ArrayT, t: dr.ArrayBase | float) -> ArrayT"),
        "a"_a, "b"_a, "t"_a, doc_slerp);

    m.def(
        "rotate",
        [](nb::type_object_t<dr::ArrayBase> dtype, dr::Array<T, 3> &axis, nb::handle angle) {
            nb::handle dtype_q = nb::type<dr::Quaternion<T>>();
            if (dtype.is(dtype_q))
                return dr::rotate<dr::Quaternion<T>>(axis, nb::cast<T>(angle));
            else
                nb::raise_type_error("dtype must be '%s', got '%s'",
                                     nb::type_name(dtype_q).c_str(),
                                     nb::type_name(dtype).c_str());
        },
        nb::sig("def rotate(dtype: typing.Type[ArrayT], axis: dr.ArrayBase, angle: dr.ArrayBase | float) -> ArrayT"),
        "dtype"_a, "axis"_a, "angle"_a, doc_rotate);

    m.def(
        "quat_apply",
        [](const dr::Quaternion<T> &q, const dr::Array<T, 3> &v) {
            return dr::quat_apply(q, v);
        },
        doc_quat_apply);
}

void export_quat(nb::module_ &m) {
    export_t<float>(m);
    export_t<double>(m);

#if defined(DRJIT_ENABLE_CUDA)
    export_t<dr::CUDADiffArray<float>>(m);
    export_t<dr::CUDADiffArray<double>>(m);
    export_t<dr::CUDAArray<float>>(m);
    export_t<dr::CUDAArray<double>>(m);
#endif

#if defined(DRJIT_ENABLE_LLVM)
    export_t<dr::LLVMDiffArray<float>>(m);
    export_t<dr::LLVMDiffArray<double>>(m);
    export_t<dr::LLVMArray<float>>(m);
    export_t<dr::LLVMArray<double>>(m);
#endif
}
