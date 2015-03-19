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
from distutils.version import LooseVersion

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

VERSION = re.findall('__version__ = "(.*)"',
                     open('ffc/__init__.py', 'r').read())[0]

SCRIPTS = [os.path.join("scripts", "ffc")]

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

def get_swig_executable():
    "Get SWIG executable"

    # Find SWIG executable
    swig_executable = None
    swig_minimum_version = "3.0.3"
    for executable in ["swig", "swig3.0"]:
        swig_executable = spawn.find_executable(executable)
        if swig_executable is not None:
            # Check that SWIG version is ok
            output = subprocess.check_output([swig_executable, "-version"]).decode('utf-8')
            swig_version = re.findall(r"SWIG Version ([0-9.]+)", output)[0]
            if LooseVersion(swig_version) >= LooseVersion(swig_minimum_version):
                break
            swig_executable = None
    if swig_executable is None:
        raise OSError("Unable to find SWIG version %s or higher." % swig_minimum_version)
    print("Found SWIG: %s (version %s)" % (swig_executable, swig_version))

    return swig_executable

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

def write_config_file(infile, outfile, variables={}):
    "Write config file based on template"
    class AtTemplate(string.Template):
        delimiter = "@"
    s = AtTemplate(open(infile, "r").read())
    s = s.substitute(**variables)
    a = open(outfile, "w")
    try:
        a.write(s)
    finally:
        a.close()

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

def generate_config_files(SWIG_EXECUTABLE, CXX_FLAGS):
    "Generate and install configuration files"

    # Get variables
    INSTALL_PREFIX = get_installation_prefix()
    PYTHON_LIBRARY = os.environ.get("PYTHON_LIBRARY", find_python_library())
    MAJOR, MINOR, MICRO = VERSION.split(".")

    # Generate UFCConfig.cmake
    write_config_file(os.path.join("cmake", "templates", "UFCConfig.cmake.in"),
                      os.path.join("cmake", "templates", "UFCConfig.cmake"),
                      variables=dict(INSTALL_PREFIX=INSTALL_PREFIX,
                                     CXX_FLAGS=CXX_FLAGS.strip(),
                                     PYTHON_INCLUDE_DIR=sysconfig.get_python_inc(),
                                     PYTHON_LIBRARY=PYTHON_LIBRARY,
                                     PYTHON_EXECUTABLE=sys.executable,
                                     SWIG_EXECUTABLE=SWIG_EXECUTABLE,
                                     FULLVERSION=VERSION))

    # Generate UFCConfigVersion.cmake
    write_config_file(os.path.join("cmake", "templates", \
                                   "UFCConfigVersion.cmake.in"),
                      os.path.join("cmake", "templates", \
                                   "UFCConfigVersion.cmake"),
                      variables=dict(FULLVERSION=VERSION,
                                     MAJOR=MAJOR, MINOR=MINOR, MICRO=MICRO))

    # Generate UseUFC.cmake
    write_config_file(os.path.join("cmake", "templates", "UseUFC.cmake.in"),
                      os.path.join("cmake", "templates", "UseUFC.cmake"))

    # FIXME: Generation of pkgconfig file may no longer be needed, so
    # FIXME: we may consider removing this.

    # Generate ufc-1.pc
    write_config_file(os.path.join("cmake", "templates", "ufc-1.pc.in"),
                      os.path.join("cmake", "templates", "ufc-1.pc"),
                      variables=dict(FULLVERSION=VERSION,
                                     INSTALL_PREFIX=INSTALL_PREFIX,
                                     CXX_FLAGS=CXX_FLAGS))

def has_cxx_flag(cc, flag):
    "Return True if compiler supports given flag"
    tmpdir = tempfile.mkdtemp(prefix="ffc-build-")
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

    # Subclass extension building command to ensure that distutils to
    # finds the correct SWIG executable
    SWIG_EXECUTABLE = get_swig_executable()
    class my_build_ext(build_ext.build_ext):
        def find_swig(self):
            return SWIG_EXECUTABLE

    # Subclass the build command to ensure that build_ext produces
    # ufc.py before build_py tries to copy it.
    class my_build(build):
        def run(self):
            self.run_command('build_ext')
            build.run(self)

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

    # Generate config files
    generate_config_files(SWIG_EXECUTABLE, CXX_FLAGS)

    # Setup extension module for FFC time elements
    ext_module_time = Extension("ffc_time_ext.time_elements_ext",
                                ["ffc_time_ext/time_elements_interface.cpp",
                                 "ffc_time_ext/time_elements.cpp",
                                 "ffc_time_ext/LobattoQuadrature.cpp",
                                 "ffc_time_ext/RadauQuadrature.cpp",
                                 "ffc_time_ext/Legendre.cpp"],
                                extra_link_args=["-Wl,-rpath,%s/lib" % sys.prefix])

    # Setup extension module for UFC
    swig_options = ["-c++", "-shadow", "-modern",
                    "-modernargs", "-fastdispatch",
                    "-fvirtual", "-nosafecstrings",
                    "-noproxydel", "-fastproxy",
                    "-fastinit", "-fastunpack",
                    "-fastquery", "-nobuildnone"]
    if sys.version_info[0] > 2: swig_options.insert(0, "-py3")
    ext_module_ufc = Extension("ufc._ufc",
                               sources=[os.path.join("ufc", "ufc.i")],
                               depends=[os.path.join("ufc", "ufc.h"),
                                        os.path.join("ufc", "ufc_geometry.h")],
                               swig_opts=swig_options,
                               extra_compile_args=CXX_FLAGS.split(),
                               extra_link_args=["-Wl,-rpath,%s/lib" % sys.prefix],
                               include_dirs=[os.path.join("ufc")])

    # Call distutils to perform installation
    setup(name             = "FFC",
          description      = "The FEniCS Form Compiler",
          version          = VERSION,
          author           = AUTHORS,
          classifiers      = [_f for _f in CLASSIFIERS.split('\n') if _f],
          license          = "LGPL version 3 or later",
          author_email     = "fenics@fenicsproject.org",
          maintainer_email = "fenics@fenicsproject.org",
          url              = "http://fenicsproject.org/",
          platforms        = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
          packages         = ["ffc",
                              "ffc.quadrature",
                              "ffc.tensor",
                              "ffc.uflacsrepr",
                              "ffc.errorcontrol",
                              "ffc.backends",
                              "ffc.backends.dolfin",
                              "ffc.backends.ufc",
                              "ufc"],
          package_dir      = {"ffc": "ffc",
                              "ufc": "ufc"},
          scripts          = scripts,
          include_dirs     = [numpy.get_include()],
          ext_modules      = [ext_module_time, ext_module_ufc],
          cmdclass         = {"build": my_build, "build_ext": my_build_ext},
          data_files       = [(os.path.join("share", "man", "man1"),
                               [os.path.join("doc", "man", "man1", "ffc.1.gz")]),
                              (os.path.join("include"),
                               [os.path.join("ufc", "ufc.h"),
                                os.path.join("ufc", "ufc_geometry.h")]),
                              (os.path.join("share", "ufc"),
                               [os.path.join("cmake", "templates", \
                                             "UFCConfig.cmake"),
                                os.path.join("cmake", "templates", \
                                             "UFCConfigVersion.cmake"),
                                os.path.join("cmake", "templates", \
                                             "UseUFC.cmake")]),
                              (os.path.join("lib", "pkgconfig"),
                               [os.path.join("cmake", "templates", "ufc-1.pc")]),
                              (os.path.join("include", "swig"),
                               [os.path.join("ufc", "ufc.i"),
                                os.path.join("ufc", "ufc_shared_ptr_classes.i")])])

if __name__ == "__main__":
    run_install()
