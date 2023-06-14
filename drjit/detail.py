import sys
import os
from .config import CXX_COMPILER

if sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

# On Clang/Linux, dynamic module loading can lead to an issue where libc++
# symbols resolve to libstdc++ implementations present in the parent process,
# causing crashes during dynamic casts and exception resolution. The flag
# RTLD_DEEPBIND ensures that the loaded library gets the right symbols. Note
# that this can cause problems when the parent process is not compiled with
# -fPIC. See
# https://codeutility.org/linux-weird-interaction-of-rtld_deepbind-position-independent-code-pic-and-c-stl-stdcout-stack-overflow/
use_deepbind = sys.platform == 'linux' and 'Clang' in CXX_COMPILER

class scoped_rtld_deepbind:
    def __enter__(self):
        if use_deepbind:
            self.backup = sys.getdlopenflags()
            sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_LOCAL | os.RTLD_DEEPBIND)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if use_deepbind:
            sys.setdlopenflags(self.backup)
