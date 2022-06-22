import sys
import os

if sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

if os.name == 'nt':
    # Specify DLL search path for windows (no rpath on this platform..)
    d = __file__
    for i in range(3):
        d = os.path.dirname(d)
    try: # try to use Python 3.8's DLL handling
        os.add_dll_directory(d)
    except AttributeError:  # otherwise use PATH
        os.environ['PATH'] += os.pathsep + d
    del d, i

del sys, os

# Implementation details accessed by both C++ and Python
import drjit.detail as detail # noqa

# Native extension defining low-level arrays
import drjit.drjit_ext as drjit_ext  # noqa

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


# Install matrix-related functions
for k, v in matrix.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v


# Install tensor-related functions
for k, v in tensor.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v


del k, v, self, base, generic, router, matrix, tensor, traits, const, drjit_ext
