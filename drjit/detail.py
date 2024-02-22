import sys as _sys
from .config import CXX_COMPILER as _CXX_COMPILER
from .config import PYTHON_VERSION as _PYTHON_VERSION
import drjit

if _sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

# On Clang/Linux, dynamic module loading can lead to an issue where libc++
# symbols resolve to libstdc++ implementations present in the parent process,
# causing crashes during dynamic casts and exception resolution. This is
# related to some other packages not playing nice and aggressively exporting
# these symbols. See
#
# https://codeutility.org/linux-weird-interaction-of-rtld_deepbind-position-independent-code-pic-and-c-stl-stdcout-stack-overflow/
# and https://github.com/pytorch/pytorch/commit/ddff4efa26d527c99cd9892278a32529ddc77e66
#
# We work around this issue by importing Dr.Jit with the RTLD_DEEPBIND flag
# when it has been compiled with Clang on Linux (on macOS, everything is
# compiled with Clang, so the issue doesn't exist, and other compilers are
# used on Windows).
#
# Note that this workaround has been found to cause problems when the parent
# process is not compiled with -fPIC. Hopefully we can remove all of this
# in a couple of years.

_use_deepbind = _sys.platform == "linux" and "Clang" in _CXX_COMPILER

class scoped_rtld_deepbind:
    '''Python context manager to import extensions with RTLD_DEEPBIND if needed'''
    def __enter__(self):
        if _use_deepbind:
            import os
            self.backup = _sys.getdlopenflags()
            _sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_LOCAL | os.RTLD_DEEPBIND)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _use_deepbind:
            _sys.setdlopenflags(self.backup)
