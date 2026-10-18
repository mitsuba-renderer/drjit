/*
    src/extra/bc.cpp -- software decoder for block-compressed textures

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <drjit/extra.h>
#include <nanothread/nanothread.h>
#include <cstring>

#define BCDEC_IMPLEMENTATION
#define BCDECDEF static inline
#define BCDEC_BC4BC5_PRECISE
#include <bcdec.h>

void ad_tex_bc_decode(int block_format, uint32_t width, uint32_t height,
                      const uint8_t *blocks, uint8_t *out) {
    uint32_t channels, block_bytes;
    switch (block_format) {
        case 4: channels = 1; block_bytes = 8; break;
        case 5: channels = 2; block_bytes = 16; break;
        case 7: channels = 4; block_bytes = 16; break;
        default:
            jit_raise("ad_tex_bc_decode(): invalid block format %i (expected "
                      "4, 5, or 7)!", block_format);
    }

    uint32_t bw = (width + 3) / 4, bh = (height + 3) / 4;

    struct Payload {
        int block_format;
        uint32_t width, height, bw, channels, block_bytes;
        const uint8_t *blocks;
        uint8_t *out;
    } payload { block_format, width, height, bw, channels, block_bytes, blocks, out };

    // Decode one row of blocks per work item, cropping partial edge blocks
    auto body = [](uint32_t by, void *ptr) {
        const Payload &p = *(const Payload *) ptr;
        uint8_t texels[16 * 4];
        int pitch = (int) (4 * p.channels);
        for (uint32_t bx = 0; bx < p.bw; ++bx) {
            const uint8_t *block = p.blocks + (by * p.bw + bx) * p.block_bytes;
            switch (p.block_format) {
                case 4: bcdec_bc4(block, texels, pitch, 0); break;
                case 5: bcdec_bc5(block, texels, pitch, 0); break;
                default: bcdec_bc7(block, texels, pitch); break;
            }
            for (uint32_t y = 0; y < 4; ++y) {
                uint32_t py = by * 4 + y;
                if (py >= p.height)
                    break;
                for (uint32_t x = 0; x < 4; ++x) {
                    uint32_t px = bx * 4 + x;
                    if (px >= p.width)
                        break;
                    memcpy(p.out + ((size_t) py * p.width + px) * p.channels,
                           texels + (y * 4 + x) * p.channels, p.channels);
                }
            }
        }
    };

    Task *task = task_submit_dep(nullptr, nullptr, 0, bh, body, &payload,
                                 sizeof(Payload));
    task_wait_and_release(task);
}
