# Implementation details accessed by both C++ and Python
import enoki.detail

# Native extension defining low-level arrays
import enoki.enoki_ext as enoki_ext

# Routing functionality (type promotion, broadcasting, etc.)
import enoki.router as router

# Generic fallback implementations of array operations
import enoki.generic as generic

# Install routing functions in ArrayBase and global scope
self = vars()
base = self['ArrayBase']
for k, v in enoki.router.__dict__.items():
    if k.startswith('_'):
        continue
    if k.startswith('op_'):
        setattr(base, '__' + k[3:] + '__', v)
    else:
        self[k] = v

# Install generic array functions in ArrayBase
for k, v in enoki.generic.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    setattr(base, k + '_', v)

del k, v, self, base, generic, router, enoki_ext
