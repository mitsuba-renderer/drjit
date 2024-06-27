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
copyright = '2024, Realistic Graphics Lab'
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
\DeclareUnicodeCharacter{2194}{\ensuremath{\leftrightarrow}}
\DeclareUnicodeCharacter{274C}{\ensuremath{\times}}
\DeclareUnicodeCharacter{26A0FE0F}{{\fontencoding{U}\fontfamily{futs}\selectfont\char66}}
""",
}

latex_documents = [
    ('index', 'drjit.tex', 'Dr.Jit Documentation', author, 'manual'),
]
