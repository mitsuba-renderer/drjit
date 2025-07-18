/*
    cuda.cpp -- instantiates the drjit.cuda.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "cuda.h"
#include "random.h"
#include "texture.h"

#include <drjit-core/gl_interop.h>

#if defined(DRJIT_ENABLE_CUDA)
void export_cuda(nb::module_ &m) {
    using Guide = dr::CUDAArray<float>;

    ArrayBinding b;
    dr::bind_all<Guide>(b);
    bind_rng<Guide>(m);
    bind_texture_all<Guide>(m);

    m.attr("Float32") = m.attr("Float");
    m.attr("Int32") = m.attr("Int");
    m.attr("UInt32") = m.attr("UInt");

    // CUDA / OpenGL interop
    m.def("register_gl_buffer", &jit_register_gl_buffer, "gl_buffer"_a)
     .def("register_gl_texture", &jit_register_gl_texture, "gl_texture"_a)
     .def("unregister_cuda_resource", &jit_unregister_cuda_resource, "cuda_resource"_a)
     .def(
         "map_graphics_resource_ptr",
         [](void *cuda_resource) {
             size_t n_bytes;
             void *ptr = jit_map_graphics_resource_ptr(cuda_resource, &n_bytes);
             return std::make_pair((std::uintptr_t) ptr, n_bytes);
         },
         "cuda_resource"_a) // TODO: return value policy needed?
     .def(
         "map_graphics_resource_array",
         [](void *cuda_resource, uint32_t array_index, uint32_t mip_level) {
             return (std::uintptr_t) jit_map_graphics_resource_array(
                 cuda_resource, array_index, mip_level);
         },
         "cuda_resource"_a,
         "array_index"_a = 0,
         "mip_level"_a = 0) // TODO: return value policy needed?
     .def("unmap_graphics_resource", &jit_unmap_graphics_resource, "cuda_resource"_a)
    //  .def(
    //      "memcpy_2d",
    //      [](std::uintptr_t dst, size_t dst_pitch, std::uintptr_t src, size_t src_pitch,
    //          size_t width, size_t height) {
    //          jit_memcpy_2d(reinterpret_cast<void *>(dst), dst_pitch,
    //                        reinterpret_cast<void *>(src), src_pitch,
    //                        width, height);
    //      },
    //      "dst"_a, "dst_pitch"_a, "src"_a, "src_pitch"_a, "width"_a, "height"_a)
    //  .def(
    //      "memcpy_2d_to_array",
    //      [](std::uintptr_t dst, size_t w_offset, size_t h_offset, std::uintptr_t src, size_t src_pitch,
    //          size_t width, size_t height) {
    //          jit_memcpy_2d_to_array(reinterpret_cast<void *>(dst), w_offset, h_offset,
    //                                 reinterpret_cast<void *>(src), src_pitch,
    //                                 width, height);
    //      },
    //      "dst"_a, "w_offset"_a, "h_offset"_a, "src"_a, "src_pitch"_a, "width"_a, "height"_a)
     .def(
         "memcpy_2d_to_array_async",
         [](std::uintptr_t dst, std::uintptr_t src, size_t src_pitch,
            size_t component_size_bytes, size_t width, size_t height, bool from_host) {
             jit_memcpy_2d_to_array_async(reinterpret_cast<void *>(dst),
                                          reinterpret_cast<void *>(src), src_pitch,
                                          component_size_bytes, width, height, from_host);
         },
         "dst"_a, "src"_a, "src_pitch"_a, "component_size_bytes"_a, "width"_a, "height"_a, "from_host"_a = false);
     }
#endif
