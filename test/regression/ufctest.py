# Copyright (C) 2010 Anders Logg, Kristian B. Oelgaard and Marie E. Rognes
#
# This file is part of FFC.
#
# FFC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFC. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2010-01-24
# Last changed: 2011-07-08

import os, sys
from ffc.log import begin, end, info, info_green, info_red, info_blue
from instant.output import get_status_output

_test_code = """\
#include "../../ufctest.h"
#include "%s.h"

int main()
{
%s

  return 0;
}
"""

def _generate_test_code(header_file, bench):
    "Generate test code for given header file."

    # Count the number of forms and elements
    prefix = header_file.split(".h")[0]
    generated_code = open(header_file).read()
    num_forms = generated_code.count("class %s_form_" % prefix.lower())
    num_elements = generated_code.count("class %s_finite_element_" % prefix.lower())

    # Generate tests, either based on forms or elements
    if num_forms > 0:
        tests = ["  %s_form_%d f%d; test_form(f%d, %d);" % (prefix.lower(), i, i, i, bench) for i in range(num_forms)]
    else:
        tests = ["  %s_finite_element_%d e%d; test_finite_element(e%d);" % (prefix.lower(), i, i, i) for i in range(num_elements)]

    # Write file
    test_file = open(prefix + ".cpp", "w")
    test_file.write(_test_code % (prefix, "\n".join(tests)))
    test_file.close()

def build_ufc_programs(bench, helper):
    "Build test programs for all test cases."

    # Get a list of all files
    header_files = [f for f in os.listdir(".") if f.endswith(".h")]
    header_files.sort()

    begin("Building test programs (%d header files found)" % len(header_files))

    # Get UFC flags
    ufc_cflags = get_status_output("pkg-config --cflags ufc-1")[1].strip()

    # Get Boost dir (code copied from ufc/src/utils/python/ufc_utils/build.py)
    # Set a default directory for the boost installation
    if sys.platform == "darwin":
        # Use MacPorts as default
        default = os.path.join(os.path.sep, "opt", "local")
    else:
        default = os.path.join(os.path.sep, "usr")

    # If BOOST_DIR is not set use default directory
    boost_inc_dir = ""
    boost_lib_dir = ""
    boost_dir = os.getenv("BOOST_DIR", default)
    boost_is_found = False
    for inc_dir in ["", "include"]:
        if os.path.isfile(os.path.join(boost_dir, inc_dir, "boost", "version.hpp")):
            boost_inc_dir = os.path.join(boost_dir, inc_dir)
            break
    for lib_dir in ["", "lib"]:
        if os.path.isfile(os.path.join(boost_dir, lib_dir, "libboost_math_tr1.so")) or\
           os.path.isfile(os.path.join(boost_dir, lib_dir, "libboost_math_tr1.dylib")):
            boost_lib_dir = os.path.join(boost_dir, lib_dir)
            break
    if boost_inc_dir != "" and boost_lib_dir != "":
        boost_is_found = True

    if not boost_is_found:
        raise OSError, """The Boost library was not found.
If Boost is installed in a nonstandard location,
set the environment variable BOOST_DIR.
"""

    ufc_cflags += " -I%s -L%s" % (boost_inc_dir, boost_lib_dir)

    # Set compiler options
    if bench > 0:
        info("Benchmarking activated")
        # Takes too long to build with -O2
        #compiler_options = "%s -Wall -Werror -O2" % ufc_cflags
        compiler_options = "%s -Wall -Werror" % ufc_cflags
    else:
        compiler_options = "%s -Wall -Werror -g" % ufc_cflags
    info("Compiler options: %s" % compiler_options)

    # Iterate over all files
    for f in header_files:

        # Generate test code
        filename = _generate_test_code(f, bench)

        # Compile test code
        prefix = f.split(".h")[0]
        command = "g++ %s -o %s.bin %s.cpp -lboost_math_tr1" % (compiler_options, prefix, prefix)
        ok = helper.run_command(command)

        # Check status
        if ok:
            info_green("%s OK" % prefix)
        else:
            info_red("%s failed" % prefix)

    end()


