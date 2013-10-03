#!/usr/bin/env python

import sys, platform
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from distutils.version import LooseVersion
from os import chdir
from os.path import join, split


# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(Extension, object):
    """Extension type that adds the NumPy include directory to include_dirs."""

    def __init__(self, *args, **kwargs):
        super(NumpyExtension, self).__init__(*args, **kwargs)
        self._include_dirs = []
        self._macros = []

    @property
    def include_dirs(self):
        from numpy import get_include
        return self._include_dirs + [get_include()]

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs

    @property
    def define_macros(self):
        from numpy import __version__
        if LooseVersion(__version__) > LooseVersion("1.6.2"):
            self._macros += [("NPY_NO_DEPRECATED_API", "NPY_%s_%s_API_VERSION"
                             % tuple(__version__.split(".")[:2]))]
        return self._macros

    @define_macros.setter
    def define_macros(self, macros):
        self._macros = macros

scripts = [join("scripts", "ffc")]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

ext = NumpyExtension("ffc.time_elements_ext",
                     ["ffc/ext/time_elements_interface.cpp",
                      "ffc/ext/time_elements.cpp",
                      "ffc/ext/LobattoQuadrature.cpp",
                      "ffc/ext/RadauQuadrature.cpp",
                      "ffc/ext/Legendre.cpp"])

setup(name = "FFC",
      version = "1.2.0",
      description = "The FEniCS Form Compiler",
      author = "Anders Logg, Kristian Oelgaard, Marie Rognes et al.",
      author_email = "ffc@lists.launchpad.net",
      url = "http://www.fenicsproject.org",
      setup_requires=['numpy>=1.6'],
      packages = ["ffc",
                  "ffc.quadrature", "ffc.tensor", "ffc.uflacsrepr",
                  "ffc.errorcontrol",
                  "ffc.dolfin"],
      package_dir={"ffc": "ffc"},
      scripts = scripts,
      ext_modules = [ext],
      data_files = [(join("share", "man", "man1"),
                     [join("doc", "man", "man1", "ffc.1.gz")])])
