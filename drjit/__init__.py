import sys
import os

if sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

# Implementation details accessed by both C++ and Python
import drjit.detail as detail # noqa

if os.name != 'nt':
    # Use RTLD_DEEPBIND to prevent the DLL to search symbols in the global scope
    old_flags = sys.getdlopenflags()
    new_flags = os.RTLD_LAZY | os.RTLD_LOCAL
    if sys.platform != 'darwin':
        new_flags |= os.RTLD_DEEPBIND
    sys.setdlopenflags(new_flags)
    del new_flags

# Native extension defining low-level arrays
import drjit.drjit_ext as drjit_ext  # noqa

if os.name != 'nt':
    sys.setdlopenflags(old_flags)
    del old_flags

# Routing functionality (type promotion, broadcasting, etc.)
import drjit.router as router  # noqa

# Generic fallback implementations of array operations
import drjit.generic as generic  # noqa

# Type traits analogous to the ones provided in C++
import drjit.traits as traits  # noqa

# Math library and const
import drjit.const as const  # noqa

# Matrix-related functions
import drjit.matrix as matrix # noqa

# Tensor-related functions
import drjit.tensor as tensor # noqa

# Install routing functions in ArrayBase and global scope
self = vars()
base = self['ArrayBase']
for k, v in router.__dict__.items():
    if k.startswith('_') or (k[0].isupper() and not k == 'CustomOp'):
        continue
    if k.startswith('op_'):
        setattr(base, '__' + k[3:] + '__', v)
    else:
        self[k] = v

# Install generic array functions in ArrayBase
for k, v in generic.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    if k.startswith('op_'):
        setattr(base, '__' + k[3:] + '__', v)
    else:
        setattr(base, k, v)


# Install type traits in global scope
for k, v in traits.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v


# Install constants in global scope
for k, v in const.__dict__.items():
    if k.startswith('_'):
        continue
    self[k] = v


# Install matrix-related functions in global scope
for k, v in matrix.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v


# Install tensor-related functions
for k, v in tensor.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v

del sys, os
del k, v, self, base, generic, router, matrix, tensor, traits, const, drjit_ext
