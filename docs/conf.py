# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

try:
    import drjit
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import drjit

# -- Project information -----------------------------------------------------

project = 'drjit'
copyright = '2026, Realistic Graphics Lab'
author = 'Realistic Graphics Lab'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinxcontrib.katex',
    'enum_tools.autoenum',
    'sphinxcontrib.rsvgconverter'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [ 'drjit.css' ]

html_title=' '
html_theme_options = {
    "light_logo": "../_images/drjit-logo-dark.svg",
    "dark_logo": "../_images/drjit-logo-light.svg"
}

# -- Options for LaTeX output -------------------------------------------------

latex_engine = "pdflatex"

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    "classoptions": ",openany,oneside",
    "preamble": r"""
\usepackage[utf8]{inputenc}
\usepackage{fourier}
\DeclareUnicodeCharacter{2194}{\ensuremath{\leftrightarrow}}
\DeclareUnicodeCharacter{274C}{\ensuremath{\times}}
\DeclareUnicodeCharacter{26A0}{\warning}
""",
}

latex_documents = [
    ('index', 'drjit.tex', 'Dr.Jit Documentation', author, 'manual'),
]

# -- ``drjit-namespace`` directive -------------------------------------------
#
# Generate backend--specific submodules (``drjit.llvm``, ``drjit.cuda``,
# ``drjit.auto`` and their ``.ad`` variants) automatically without having
# to select each class/method via autodoc.

import importlib
from docutils.parsers.rst import Directive, directives

# Reference backend used to introspect class members
import drjit.llvm as _ref_backend


def _arrays(prefixes, suffixes):
    return [f"Array{p}{s}" for s in suffixes for p in prefixes]

# Ordered groups rendered for every JIT backend namespace. Each entry is
# ``(heading, kind, names)`` where ``kind`` is one of:
#
#   "array"   - a plain type deriving from ``drjit.ArrayBase``.
#   "members" - a class whose methods/properties are enumerated by introspection.
#   "members_canonical" - like "members", but only rendered in the namespace that
#                         *owns* the class. ``Event`` has no AD variant, so e.g.
#                         ``drjit.llvm.ad.Event`` is literally ``drjit.llvm.Event``
#                         (the same object). Documenting it in both ``drjit.llvm``
#                         and ``drjit.llvm.ad`` (or in ``drjit.auto``, an alias of
#                         a concrete backend) would register the same canonical
#                         name twice. We therefore only emit it where the class is
#                         defined, i.e. in the concrete, non-AD backends.
_NAMESPACE_GROUPS = [
    ("Scalar", "array",
     ["Bool", "Float16", "Float", "Float64",
      "UInt", "UInt8", "UInt64", "Int", "Int8", "Int64"]),
    ("1D arrays", "array",
     _arrays(["0", "1", "2", "3", "4", "X"],
             ["b", "f16", "f", "u", "i", "f64", "u64", "i64"])),
    ("2D arrays", "array",
     _arrays(["22", "33", "44"], ["b", "f", "f64"])),
    ("Special (complex numbers, etc.)", "array",
     ["Complex2f", "Complex2f64",
      "Quaternion4f16", "Quaternion4f", "Quaternion4f64",
      "Matrix2f16", "Matrix3f16", "Matrix4f16",
      "Matrix2f", "Matrix3f", "Matrix4f",
      "Matrix2f64", "Matrix3f64", "Matrix4f64"]),
    ("Tensors", "array",
     ["TensorX" + s for s in
      ["b", "f16", "f", "u", "i", "f64", "u64", "i64"]]),
    ("Textures", "members",
     [f"Texture{d}{s}" for s in ["f16", "f", "f64"] for d in [1, 2, 3]]),
    ("Events and synchronization", "members_canonical", ["Event"]),
    ("Random number generators", "members", ["PCG32", "Philox4x32"]),
]


def _class_members(name):
    """Enumerate the documentable members of a reference-backend class in
    binding order, returning ``(autodirective, member_name)`` pairs."""
    cls = getattr(_ref_backend, name)
    members = []
    for member, obj in vars(cls).items():
        if isinstance(obj, property):
            kind = "autoproperty"
        elif type(obj).__name__ in ("nb_method", "nb_func"):
            # A genuine bound method. This excludes auto-generated slots such
            # as ``__new__`` (a builtin) and plain data attributes such as
            # ``__doc__``, ``DRJIT_STRUCT`` or ``Texture.IsTexture``.
            kind = "automethod"
        else:
            continue
        # Keep ``__init__`` and operator dunders (``__add__`` etc.), but drop
        # private helpers such as ``_traverse_1_cb_ro``.
        if member.startswith("_") and not (
            member.startswith("__") and member.endswith("__")
        ):
            continue
        members.append((kind, member))
    return members


def _backend_available(module):
    """Whether *module* is a compiled-in, populated backend. Returns the
    imported module, or ``None`` if the backend is unavailable in this build
    (e.g. CUDA on a machine without the CUDA toolkit, or Metal on Linux/RTD)."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    # A populated backend exposes the array types. ``drjit.auto`` resolves its
    # attributes lazily but is always backed by some concrete backend.
    return mod if hasattr(mod, "Float16") else None


class DrjitNamespace(Directive):
    """Render the full array-type listing for one backend namespace.

    Usage::

        .. drjit-namespace:: drjit.llvm
           :title: LLVM array namespace (``drjit.llvm``)

           Prose describing the backend goes here.

    The directive owns the section heading and prose so that it can omit the
    *entire* section when the backend is not compiled into the build being
    documented. This keeps the same ``type_ref.rst`` working everywhere: the
    Metal sections render on macOS but vanish on Read the Docs, and likewise the
    CUDA sections vanish on a machine without CUDA -- in both cases without
    leaving broken autodoc entries or an empty heading behind.
    """

    required_arguments = 1
    has_content = True
    option_spec = {"title": directives.unchanged_required}

    def run(self):
        module = self.arguments[0].strip()
        mod = _backend_available(module)
        if mod is None:
            return []

        title = self.options["title"]
        out = [title, "_" * len(title), "",
               f".. py:module:: {module}", ""]
        out += list(self.content) + [""]

        for heading, kind, names in _NAMESPACE_GROUPS:
            out += [heading, "^" * len(heading), ""]
            for name in names:
                # ``Event`` has no AD variant, so it is the *same* object in a
                # backend and its ``.ad`` sibling (and in ``drjit.auto``, which
                # aliases a concrete backend). Only document it where it lives to
                # avoid registering the same canonical name twice.
                if kind == "members_canonical" and \
                        getattr(mod, name).__module__ != module:
                    continue
                out += [f".. autoclass:: {name}", ""]
                if kind == "array":
                    out += ["   Derives from :py:class:`drjit.ArrayBase`.", ""]
                else:
                    out += [f"   .. {d}:: {m}" for d, m in _class_members(name)]
                    out += [""]
        self.state_machine.insert_input(out, f"<drjit-namespace {module}>")
        return []


def setup(app):
    app.add_directive("drjit-namespace", DrjitNamespace)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
