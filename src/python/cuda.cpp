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
#include "event.h"

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

    // CUDA / OpenGL interop -- for Polyscope
    m.def("register_gl_buffer", &jit_register_gl_buffer, "gl_buffer"_a,
          doc_cuda_register_gl_buffer)
     .def("register_gl_texture", &jit_register_gl_texture, "gl_texture"_a,
          doc_cuda_register_gl_texture)
     .def("unregister_cuda_resource", &jit_unregister_cuda_resource, "cuda_resource"_a,
          doc_cuda_unregister_cuda_resource)
     .def("map_graphics_resource_ptr", [](void *cuda_resource) {
             size_t n_bytes;
             void *ptr = jit_map_graphics_resource_ptr(cuda_resource, &n_bytes);
             return std::make_pair((std::uintptr_t) ptr, n_bytes);
          },
          "cuda_resource"_a, doc_cuda_map_graphics_resource_ptr)
     .def("map_graphics_resource_array",
          [](void *cuda_resource, uint32_t mip_level) {
              return (std::uintptr_t) jit_map_graphics_resource_array(
                  cuda_resource, mip_level
              );
          },
          "cuda_resource"_a, "mip_level"_a = 0,
          doc_cuda_map_graphics_resource_array)
     .def("unmap_graphics_resource", &jit_unmap_graphics_resource, "cuda_resource"_a,
          doc_cuda_unmap_graphics_resource)
     .def("memcpy_2d_to_array_async",
          [](std::uintptr_t dst, std::uintptr_t src, size_t src_pitch,
             size_t height, bool from_host) {
              jit_memcpy_2d_to_array_async((void *) dst, (void *) src,
                                           src_pitch, height, from_host);
          },
          "dst"_a, "src"_a, "src_pitch"_a, "height"_a,
          "from_host"_a = false, doc_cuda_memcpy_2d_to_array_async);

    struct GLInterop {
        explicit GLInterop(void *handle, bool is_texture) : handle(handle), is_texture(is_texture) { }
        GLInterop(const GLInterop &) = delete;
        GLInterop(GLInterop &&r) : handle(r.handle), is_texture(r.is_texture), ptr(r.ptr), buf_size(r.buf_size) {
            r.handle = nullptr;
            r.is_texture = false;
            r.ptr = nullptr;
            r.buf_size = 0;
        }

        static GLInterop from_buffer(uint32_t gl_buffer) {
            return GLInterop(jit_register_gl_buffer(gl_buffer), false);
        }

        static GLInterop from_texture(uint32_t gl_texture) {
            return GLInterop(jit_register_gl_texture(gl_texture), true);
        }

        GLInterop *map(uint32_t mip_level = 0) {
            if (ptr)
                nb::raise("GLInterop: already mapped!");
            if (is_texture)
                ptr = jit_map_graphics_resource_array(handle, mip_level);
            else
                ptr = jit_map_graphics_resource_ptr(handle, &buf_size);
            return this;
        }

        GLInterop *upload(const nb::ndarray<nb::ro, nb::device::cuda, nb::c_contig> &buffer) {
            if (!ptr)
                nb::raise("GLInterop: not mapped!");
            if (is_texture) {
                if (buffer.ndim() != 2 && buffer.ndim() != 3)
                    nb::raise("GLInterop::write(): expected a 2D input buffer!");
              jit_memcpy_2d_to_array_async(
                    ptr,
                    buffer.data(),
                    buffer.shape(1) * (buffer.ndim() == 3 ? buffer.shape(2) : 1) * buffer.itemsize(),
                    buffer.shape(0),
                    false
                );
            } else {
                size_t input_size = buffer.size() * buffer.itemsize();
                if (input_size != buf_size)
                    nb::raise("GLInterop::write(): expected an input of size %zu, got %zu",
                              buf_size, input_size);
                jit_memcpy_async(JitBackend::CUDA, ptr, buffer.data(),
                                 input_size);
            }
            return this;
        }

        GLInterop *unmap() {
            if (!ptr)
                nb::raise("GLInterop: not mapped!");
            jit_unmap_graphics_resource(handle);
            ptr = nullptr;
            buf_size = 0;
            return this;
        }

        ~GLInterop() {
            if (ptr)
                unmap();
            if (handle)
                jit_unregister_cuda_resource(handle);
        }

        void *handle = nullptr;
        bool is_texture = false;
        void *ptr = nullptr;
        size_t buf_size = 0;
    };

    nb::class_<GLInterop>(m, "GLInterop", doc_cuda_GLInterop)
        .def_static("from_buffer", &GLInterop::from_buffer, doc_cuda_GLInterop_from_buffer)
        .def_static("from_texture", &GLInterop::from_texture, doc_cuda_GLInterop_from_texture)
        .def("map", &GLInterop::map, "mip_level"_a = 0, nb::rv_policy::none, doc_cuda_GLInterop_map)
        .def("upload", &GLInterop::upload, nb::rv_policy::none, doc_cuda_GLInterop_upload)
        .def("unmap", &GLInterop::unmap, nb::rv_policy::none, doc_cuda_GLInterop_unmap);

    bind_event<JitBackend::CUDA>(m, "Event");
 }
#endif
