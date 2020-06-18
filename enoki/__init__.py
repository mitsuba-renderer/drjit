import sys

if sys.version_info < (3, 5):
    raise ImportError("Enoki requires Python >= 3.5")

del sys

# Implementation details accessed by both C++ and Python
import enoki.detail as detail # noqa

# Native extension defining low-level arrays
import enoki.enoki_ext as enoki_ext  # noqa

# Routing functionality (type promotion, broadcasting, etc.)
import enoki.router as router  # noqa

# Generic fallback implementations of array operations
import enoki.generic as generic  # noqa

# Type traits analogous to the ones provided in C++
import enoki.traits as traits  # noqa

# Math library and const
import enoki.const as const  # noqa

# Matrix-related functions
import enoki.matrix as matrix # noqa

# Install routing functions in ArrayBase and global scope
self = vars()
base = self['ArrayBase']
for k, v in router.__dict__.items():
    if k.startswith('_') or k[0].isupper():
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

del k, v, self, base, generic, router, matrix, traits, const, enoki_ext
