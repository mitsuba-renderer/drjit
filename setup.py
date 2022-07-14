# -*- coding: utf-8 -*-

from __future__ import print_function

import sys, re, os

try:
    from skbuild import setup
    import pybind11
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

VERSION_REGEX = re.compile(
    r"^\s*#\s*define\s+DRJIT_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE)

this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join("include/drjit/fwd.h")) as f:
    matches = dict(VERSION_REGEX.findall(f.read()))
    drjit_version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)

with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

long_description = long_description[long_description.find('## Introduction'):]

setup(
    name="drjit",
    version=drjit_version,
    author="Wenzel Jakob",
    author_email="wenzel.jakob@epfl.ch",
    description="Structured vectorization and differentiation on modern processor architectures",
    url="https://github.com/mitsuba-renderer/drjit",
    license="BSD",
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmake_args=[
        '-DDRJIT_ENABLE_JIT:BOOL=ON',
        '-DDRJIT_ENABLE_AUTODIFF:BOOL=ON',
        '-DDRJIT_ENABLE_PYTHON:BOOL=ON',
        '-DCMAKE_INSTALL_LIBDIR=drjit',
        '-DCMAKE_INSTALL_BINDIR=drjit',
        '-DCMAKE_INSTALL_INCLUDEDIR=drjit/include',
        '-DCMAKE_INSTALL_DATAROOTDIR=drjit/share'
    ],
    packages=['drjit'],
    python_requires=">=3.8"
)
