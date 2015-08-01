#!/usr/bin/env python

import os, sys, platform, re, subprocess, string, numpy, tempfile, shutil
from distutils import sysconfig, spawn
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from distutils.command import build_ext
from distutils.command.build import build
from distutils.ccompiler import new_compiler

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

VERSION = re.findall('__version__ = "(.*)"',
                     open('mffc/__init__.py', 'r').read())[0]

SCRIPTS = [os.path.join("scripts", "mffc")]

AUTHORS = """\
Anders Logg, Kristian Oelgaard, Marie Rognes, Garth N. Wells,
Martin Sandve Alnaes, Hans Petter Langtangen, Kent-Andre Mardal,
Ola Skavhaug, et al.
"""

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 (GPLv2)
License :: Public Domain
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: POSIX :: Linux
Programming Language :: C++
Programming Language :: Python
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries
"""

def get_installation_prefix():
    "Get installation prefix"
    try:
        prefix = [item for item in sys.argv[1:] \
                  if "--prefix=" in item][0].split("=")[1]
    except:
        try:
            prefix = sys.argv[sys.argv.index("--prefix")+1]
        except:
            if platform.system() == "Windows":
                prefix = sys.prefix
            else:
                prefix = "/usr/local"
    return os.path.abspath(os.path.expanduser(prefix))

def create_windows_batch_files(scripts):
    """Create Windows batch files, to get around problem that we
    cannot run Python scripts in the prompt without the .py
    extension."""
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write("python \"%%~dp0\%s\" %%*\n" % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)
    return scripts

def find_python_library():
    "Return the full path to the Python library (empty string if not found)"
    pyver = sysconfig.get_python_version()
    libpython_names = [
        "python%s" % pyver.replace(".", ""),
        "python%smu" % pyver,
        "python%sm" % pyver,
        "python%su" % pyver,
        "python%s" % pyver,
        ]
    dirs = [
        "%s/lib" % os.environ.get("PYTHON_DIR", ""),
        "%s" % sysconfig.get_config_vars().get("LIBDIR", ""),
        "/usr/lib/%s" % sysconfig.get_config_vars().get("MULTIARCH", ""),
        "/usr/local/lib",
        "/opt/local/lib",
        "/usr/lib",
        "/usr/lib64",
        ]
    libpython = None
    cc = new_compiler()
    for name in libpython_names:
        libpython = cc.find_library_file(dirs, name)
        if libpython is not None:
            break
    return libpython or ""

def has_cxx_flag(cc, flag):
    "Return True if compiler supports given flag"
    tmpdir = tempfile.mkdtemp(prefix="mffc-build-")
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, "flagname.cpp")
            f = open(fname, "w")
            f.write("int main() { return 0;}")
            f.close()
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            cc.compile([fname], output_dir=tmpdir, extra_preargs=[flag])
        except:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)

def run_install():
    "Run installation"

    # Create batch files for Windows if necessary
    scripts = SCRIPTS
    if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
        scripts = create_windows_batch_files(scripts)

    # Check that compiler supports C++11 features
    cc = new_compiler()
    CXX = os.environ.get("CXX")
    if CXX:
        cc.set_executables(compiler_so=CXX, compiler=CXX, compiler_cxx=CXX)
    CXX_FLAGS = os.environ.get("CXXFLAGS", "")
    if has_cxx_flag(cc, "-std=c++11"):
        CXX_FLAGS += " -std=c++11"
    elif has_cxx_flag(cc, "-std=c++0x"):
        CXX_FLAGS += " -std=c++0x"

    # Call distutils to perform installation
    setup(name             = "MFFC",
          description      = "The FEniCS Form Compiler Modified for Firedrake",
          version          = VERSION,
          author           = AUTHORS,
          classifiers      = [_f for _f in CLASSIFIERS.split('\n') if _f],
          license          = "LGPL version 3 or later",
          author_email     = "fenics@fenicsproject.org",
          maintainer_email = "fenics@fenicsproject.org",
          url              = "http://fenicsproject.org/",
          platforms        = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
          packages         = ["mffc",
                              "mffc.quadrature",
                              "mffc.tensor",
                              "mffc.uflacsrepr",
                              "mffc.errorcontrol"],
          package_dir      = {"mffc": "mffc"},
          scripts          = scripts,
          include_dirs     = [numpy.get_include()],
          data_files       = [(os.path.join("share", "man", "man1"),
                               [os.path.join("doc", "man", "man1", "mffc.1.gz")])])

if __name__ == "__main__":
    run_install()
