# -*- coding: utf-8 -*-

from __future__ import print_function

import sys, re

try:
    from skbuild import setup
except ImportError:
    print("Please update pip: version 10 or greater is required.",
          file=sys.stderr)
    raise

VERSION_REGEX = re.compile(
    r"^\s*#\s*define\s+ENOKI_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE)

with open("include/enoki/fwd.h") as f:
    matches = dict(VERSION_REGEX.findall(f.read()))
    enoki_version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)

setup(
    name="enoki",
    version=enoki_version,
    author="Wenzel Jakob",
    author_email="wenzel.jakob@epfl.ch",
    description="Structured vectorization and differentiation on modern processor architectures",
    url="https://github.com/mitsuba-renderer/enoki",
    license="BSD",
    cmake_args=[
        '-DENOKI_ENABLE_JIT:BOOL=ON',
        '-DENOKI_ENABLE_AUTODIFF:BOOL=ON',
        '-DENOKI_ENABLE_PYTHON:BOOL=ON',
        '-DCMAKE_INSTALL_LIBDIR=enoki',
        '-DCMAKE_INSTALL_INCLUDEDIR=enoki/include',
        '-DCMAKE_INSTALL_DATAROOTDIR=enoki/share'
    ],
    packages=['enoki']
)
