Dr.Jit — A Just-In-Time-Compiler for Differentiable Rendering
=============================================================

.. only:: not latex

   .. image:: ../ext/drjit-core/resources/drjit-logo-dark.svg
     :class: only-light
     :align: center

   .. image:: ../ext/drjit-core/resources/drjit-logo-light.svg
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: ../ext/drjit-core/resources/drjit-logo-dark.svg
     :align: center

**Dr.Jit** is a *just-in-time* (JIT) compiler for ordinary and differentiable
computation. It was originally created as the numerical foundation of `Mitsuba
3 <https://github.com/mitsuba-renderer/mitsuba3>`_, a differentiable `Monte
Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`_ renderer. Over time,
*Dr.Jit* has become a general-purpose tool that can also help with various
other types of embarrassingly parallel computation.

If you're new Dr.Jit, start by reading the section :ref:`"What is Dr.Jit?"
<what_is_drjit>` and continue through the subsequent sections in order. A
separate :ref:`reference <reference>` provides detailed specifications of the
public API.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   changelog
   what
   basics
   cflow
   eval
   autodiff
   types
   interop
   optim
   misc
   debug
   bench
   cpp
   textures
   freeze
   faq

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference
   type_ref
