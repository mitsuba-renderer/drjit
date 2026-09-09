.. py:currentmodule:: drjit

Debugging
=========

This section presents strategies for debugging Dr.Jit-based programs that
do not behave as expected.

Undefined behavior
------------------

Dr.Jit operations that acess memory (e.g., :py:func:`dr.gather() <gather>`,
:py:func:`dr.scatter(), dynamic slicing) emphasize performance and assume that
provided indices are in bounds. Violating this rule can easily crash the
process or produce other kinds of undefined behavior. To track down such
issues, enable *debug mode* (:py:attr:`drjit.JitFlag.Debug`) at the beginning
of your program.

.. code-block:: python

   dr.set_flag(drjit.JitFlag.Debug, True)

Debug mode adds bounds checks that report all undefined behavior along with the
responsible Python source code location. It is expensive and should not be
enabled by default. used to periodically detect issues. Debug mode comes at a
significant additional cost and is not a good default setting. We recommend
enabling it occasionally to flush out errors.

Further, consider using the functions

- :py:func:`drjit.assert_true`,
- :py:func:`drjit.assert_false`,
- :py:func:`drjit.assert_equal`.

to assert program invariants. They are only active in debug mode and can
also check symbolic variables.

In general, it should not be possible to crash Dr.Jit or run into undefined
behavior when debug mode is enabled. If you do, then you have likely found a
bug and we would appreciate a bug report with a minimal reproducer.

Debugging within Python
-----------------------

You can use the built-in `Python debugger
<https://docs.python.org/3/library/pdb.html>`__ or an IDE such as `VS Code
<https://code.visualstudio.com/docs/python/debugging>`__ to set breakpoints and
step through Dr.Jit programs. In this case, it may be helpful to disable
Dr.Jit's symbolic loops, conditionals, and calls so that variable contents are
inspectable.

.. code-block:: python

   dr.set_flag(drjit.JitFlag.SymbolicLoops, True)
   dr.set_flag(drjit.JitFlag.SymbolicCalls, True)
   dr.set_flag(drjit.JitFlag.SymbolicConditionals, True)

This will switch control flow to the less efficient but functionally equivalent
*evaluated mode* that is compatible with interactive debugging.

.. _debug_kernels:

Debugging using LLDB or GDB
---------------------------

The LLVM backend also supports attaching *external debuggers* like `LLDB
<https://lldb.llvm.org>`__ and `GDB <https://www.gnu.org/software/gdb/>`__ on
macOS and Linux. It tags compiled code with source location information to map
from machine instructions to Python code. You must enable *debug mode*
(:py:attr:`drjit.JitFlag.Debug`) for this feature.

Native debuggers can then treat kernels much like ordinary compiled code:

- Interrupting a running program shows the Python line teach thread is
  currently executing, even when the thread is deep inside a kernel.

- Breakpoints on Python source lines resolve to the corresponding machine code
  within kernels.

This reveals where a long-running kernel spends its time and enables low-level
inspection (e.g., disassembly and single-stepping) of the code generated for a
specific Python line.

LLDB does not consult the debugger interface for JIT-compiled code by default.
Enable it once and for all by adding the following line to ``~/.lldbinit``:

.. code-block:: text

   settings set plugin.jit-loader.gdb.enable on

GDB supports this interface out of the box and needs no configuration.


The following shows an example LLDB session:

.. code-block:: console

   $ lldb -- python3 test.py
   (lldb) run
   running (interrupt with Ctrl-C in the debugger)
   ^C
   Process 85611 stopped
   * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
       frame #0: 0x00000001011b81c8 JIT(0x1011b4000)`drjit.switch() + 456 at test.py:21
      18   def f_sin(x):
      19       y = x
      20       for _ in range(8):
   -> 21           y = dr.sin(y) * 1.5 + 0.25
      22       return y
   (lldb) bt
   * frame #0: 0x00000001011b81c8 JIT(0x1011b4000)`drjit.switch() + 456 at test.py:21
     frame #1: 0x00000001011a8144 JIT(0x1011a4000)`drjit_kernel + 324 at test.py:34
     frame #2: 0x000000010005fb58 libnanothread.dylib`pool_execute_task(...) + 72
     ...
   (lldb) thread backtrace all
   ...
   (lldb) breakpoint set -f test.py -l 26
   Breakpoint 1: where = JIT(0x1011b4000)`drjit.switch() + 156 at test.py:26
   (lldb) continue

A few things are worth knowing when reading such output:

- Kernel entry points appear as ``drjit_kernel``, and callables invoked via
  :py:func:`drjit.switch` or :py:func:`drjit.dispatch` appear under the name
  of the call. The underlying symbols are named ``drjit_<hash>`` and
  ``func_<hash>``, where ``<hash>`` identifies the :ref:`kernel cache
  <caching>` entry.

- A line refers to the Python code that *created* an operation. Since Dr.Jit
  fuses and reorders operations, neighboring instructions may belong to
  different lines, and a single line may occur in several places.

- Variables of the Python program are not accessible from within a kernel.

.. _debugging_drjit:

Advanced: finding bugs within Dr.Jit itself
-------------------------------------------

To debug Dr.Jit itself, begin making a debug build (i.e., manually compile it with
``-DCMAKE_BUILD_TYPE=Debug``). Furthermore, you may want to enable some of the
following sanitization flags:

- ``DRJIT_SANITIZE_ASAN``: Enable the `Address Sanitizer <https://github.com/google/sanitizers/wiki/AddressSanitizer>`__.
- ``DRJIT_SANITIZE_UBSAN``: Enable the `Undefined Behavior Sanitizer
  <https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html>`__.
- ``DRJIT_SANITIZE_INTENSE``: Insert sanitization "checkpoints" into Dr.Jit that aggressively flush out undefined behavior
  involving its internal variable data structures. This setting only makes sense combined with ASan and/or UBSan.


Sanitizing Python sessions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Getting the sanitizers to play well with Python requires a few extra steps.
First, unless you have manually compiled Python with sanitization, you will
need to preload ``libasan`` using ``LD_PRELOAD`` (Linux)` or
``DYLD_INSERT_LIBRARIES`` (macOS). The precise path will depend on the details
of your development environment. For example, I use the following on macOS and
Linux.

.. code-block:: bash

   # macOS
   DYLD_INSERT_LIBRARIES="$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)" python <...>

   # Linux
   LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)" python

On Linux, both ``libasan`` and ``libstdc++`` or ``libc++`` need to be preloaded
at the same time.

On macOS, the ``DYLD_INSERT_LIBRARIES`` environment variable isn't enough:
``libasan`` needs to be preloaded into the actual Python binary, and the
``python3`` binary is generally just a thin wrapper. To determine the path of
the actual Python executable, run ``whoami.py`` by `Jonas Devlieghere
<https://jonasdevlieghere.com/post/sanitizing-python-modules/>`__.

.. code-block:: python

   import ctypes
   dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
   namelen = ctypes.c_ulong(1024)
   name = ctypes.create_string_buffer(b'\000', namelen.value)
   dyld._NSGetExecutablePath(ctypes.byref(name), ctypes.byref(namelen))
   print(name.value)

On my machine, this, e.g., prints
``b'/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python'``.

Putting both together, we can then, e.g., run the Python test suite via ``pytest``. (Don't forget to specify ``--capture no`` to ensure
that the sanitizer messages are visible).

.. code-block:: bash

   PYTHON_BIN="/opt/homebrew/Cellar/python@3.12/3.12.1/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
   PYTHON_DYLD="$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)"
   DYLD_INSERT_LIBRARIES=$PYTHON_DYLD $PYTHON_BIN -m pytest --capture no

On Linux, ASAN conflicts with CUDA because both very aggressively map the
entire virtual memory space and cause each other to run out of memory. A
workaround seems to be to set the environment variable

.. code-block:: bash

   ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0

.. _inspect_kernels:

Advanced: Inspecting compiled kernels
-------------------------------------

It is sometimes useful to look at the machine code that Dr.Jit generated for a
kernel, for example to disassemble it or to load its symbols into a debugger
such as GDB or LLDB. The :ref:`kernel cache <caching>` provides a convenient
way to access this code.

With the exception of the OptiX database, every cache entry is a standard `LZ4
<https://lz4.org>`__ frame. Dr.Jit compresses entries using a dictionary that
depends on the file type, which improves the compression ratio of small
kernels. The ``lz4`` command line tool can decompress an entry given the
matching dictionary from the ``ext/drjit-core/resources`` directory of the
source tree. For example, the following command extracts an object file from
the cache on Linux:

.. code-block:: bash

   lz4 -d -D lz4_dict_elf ~/.drjit/<hash>.o.lz4 kernel.o

The decompressed contents depend on the backend:

- **LLVM Backend**: each entry decompresses to a native object file in the
  platform's standard format, i.e., ELF on Linux, Mach-O on macOS, and COFF on
  Windows. The matching dictionaries are ``lz4_dict_elf``, ``lz4_dict_macho``,
  and ``lz4_dict_coff``.
  Tools such as ``objdump``, ``otool``, or ``dumpbin`` can disassemble these
  files.

- **Metal Backend**: the cache holds three kinds of Metal library files.
  Entries with the extension ``.air.metallib.lz4`` hold an intermediate
  library image produced
  by the shader compiler front end. Entries with the extension
  ``.func.metallib.lz4`` are binary archives with the device-specific machine
  code of individual callables. Entries with the extension
  ``.pso.metallib.lz4`` are binary archives with the pipeline state of complete
  kernels. The first two use the ``lz4_dict_metallib`` dictionary, the last
  one uses ``lz4_dict_mpso``.

- **CUDA Backend**: Dr.Jit relies on the driver's own cache, whose format is
  not documented. Use the :py:class:`drjit.kernel_history` API to retrieve the
  PTX source of a kernel instead.

