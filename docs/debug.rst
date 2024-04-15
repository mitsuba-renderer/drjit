.. py:currentmodule:: drjit

Debugging
=========

This section presents strategies for debugging Dr.Jit-based programs that
do not behave as expected.

Suppressing undefined behavior
------------------------------

Several operations elide bounds checks for performance reasons, which can lead
to undefined behavior.

For example, calling :py:func:`drjit.gather` with an incorrect index could
cause the operation to read beyond the end of an array, producing bogus results
or even crashing the Python session due to a CPU or GPU page fault. Similarly,
incorrect indices passed to :py:func:`drjit.scatter`,
:py:func:`drjit.scatter_reduce`, :py:func:`drjit.scatter_add`, etc., might
cause these operations to write beyond the end of an array and crash Python or,
worse, introduce data corruption elsewhere that shows up much later.

To track down such issues, enable *debug mode*
(:py:attr:`drjit.JitFlag.Debug`). Debug mode instruments compiled kernels with
additional checks that suppress and report all undefined behavior along with
the responsible Python source code location.

To enable it, set the associated flag at the beginning of your program.

.. code-block:: python

   dr.set_flag(drjit.JitFlag.Debug)

Alternatively, you can enable debug mode locally for a block of code.

.. code-block:: python

   with dr.scoped_flag(drjit.JitFlag.Debug):

       # .. code goes here

(Due to how this instrumentation works internally, Python source code locations
will be tracked following the next function call)

Debug mode comes at a significant additional cost and is not a good default
setting. We recommend enabling it occasionally to flush out errors.

In general, it should not be possible to crash Dr.Jit or encounter undefined
behavior when debug mode is enabled. If you can break things with this flag
set, then you have likely found a bug within Dr.Jit (see the next section).

Debug assertions
----------------

Dr.Jit offers the following assertion helper functions that perform additional
check when the program runs in the *debug mode* explained above. Otherwise,
they are optimized away.

- :py:func:`drjit.assert_true`,
- :py:func:`drjit.assert_false`,
- :py:func:`drjit.assert_equal`.

A useful feature of these functions is that they also work in a symbolic
context, in which case they report errors asynchronously when code eventually
runs on the device.

Stepping through programs
-------------------------

If debug mode did not change the behavior of the program, then it may be
helpful to isolate the issue using traditional debugging techniques
(visualizing variable contents, setting breakpoints, and single-stepping
through the program using the built-in `Python debugger
<https://docs.python.org/3/library/pdb.html>`__ or an IDE such as `VS Code
<https://code.visualstudio.com/docs/python/debugging>`__.

Dr.Jit's symbolic loops, conditionals, and calls can sometimes interfere with
this kind of debugging methodology because they prevent access to symbolic
variable contents. In this case, you can temporarily disable all symbolic
program features by setting :py:attr:`drjit.JitFlag.SymbolicLoops`,
:py:attr:`drjit.JitFlag.SymbolicCalls`, and
:py:attr:`drjit.JitFlag.SymbolicConditionals` to ``False``. This will switch
control flow to the less efficient but functionally equivalent *evaluated mode*
that is compatible with interactive debugging.

Localizing bugs within Dr.Jit
-----------------------------

To debug Dr.Jit, begin making a debug build (i.e., manually compile it with
``-DCMAKE_BUILD_TYPE=Debug``). Furthermore, you may want to enable some of the
following sanitization flags:

- ``DRJIT_SANITIZE_ASAN``: Enable the `Address Sanitizer <https://github.com/google/sanitizers/wiki/AddressSanitizer>`__.
- ``DRJIT_SANITIZE_UBSAN``: Enable the `Undefined Behavior Sanitizer
  <https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html>`__.
- ``DRJIT_SANITIZE_INTENSE``: Insert sanitization "checkpoints" into Dr.Jit that aggressively flush out undefined behavior
  involving its internal variable data structures. This setting only makes sense combined with ASan and/or UBSan.


Sanitizing Python sessions
--------------------------

Getting the sanitizers to play well with Python requires a few extra steps.
First, unless you have manually compiled Python with sanitization, you will
need to preload ``libasan`` using ``LD_PRELOAD`` (Linux)` or
``DYLD_INSERT_LIBRARIES`` (macOS). The precise path will depend on the details
of your development environment. For example, I use the following on macOS and
Linux.

.. code-block:: bash

   # macOS
   DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/15.0.0/lib/darwin/libclang_rt.asan_osx_dynamic.dylib python <...>

   # Linux
   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.6:/usr/lib/x86_64-linux-gnu/libstdc++.so.6

On Linux, both ``libasan`` and ``libstdc++`` or ``libc++`` need to be preloaded
at the same time (be careful to use the right version of ``libasan`` in case
multiple ones are installed on your system).

On macOS, the ``DYLD_INSERT_LIBRARIES`` environment variable isn't enough:
``libasan`` needs to be preloaded into the actual Python binary, and the
``python3`` binary is generally just a thin wrapper. To determine the path of
the actual Python executable, run ``whoami.py`` by `Jonas Devlieghere
<https://jonasdevlieghere.com/post/sanitizing-python-modules/>`.

.. code-block:: pycon

   import ctypes
   dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
   namelen = ctypes.c_ulong(1024)
   name = ctypes.create_string_buffer(b'\000', namelen.value)
   dyld._NSGetExecutablePath(ctypes.byref(name), ctypes.byref(namelen))
   print(name.value)

On my machine, this, e.g., prints
```b'/opt/homebrew/Cellar/python@3.12/3.12.2_1/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python'``.

Putting both together, we can then, e.g., run the Python test suite via ``pytest``. (Don't forget to specify ``--capture no`` to ensure
that the sanitizer messages are visible).

.. code-block:: bash

   DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/15.0.0/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
/opt/homebrew/Cellar/python@3.12/3.12.1/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python
-m pytest --capture no

On Linux, ASAN conflicts with CUDA because both very aggressively map the
entire virtual memory space and cause each other to run out of memory. A
workaround seems to be to set the environment variable

.. code-block:: bash

   ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0
