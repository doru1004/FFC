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

from ffc.log import begin, end, info, info_green, info_red
import os
from testutils import run_command

_test_code = """\
#include "../../pyop2test.h"
#include "%s.h"

int main()
{
%s

  return 0;
}
"""

def _generate_test_code(header_file, bench):
    "Generate test code for given header file."

    # Load test code
    prefix = header_file.split(".h")[0]
    pl = prefix.lower()
    generated_code = open(header_file).readlines()

    # Get integrals
    cell_integrals = [line.partition("(")[0] for line in generated_code \
                      if line.count("%s_cell_integral" % pl)>0 ]
    exterior_facet_integrals = [line.partition("(")[0] for line in generated_code \
                                if line.count("%s_exterior_facet_integral" % pl)>0 ]
    interior_facet_integrals = [line.partition("(")[0] for line in generated_code \
                                if line.count("%s_interior_facet_integral" % pl)>0 ]

    # Generate tests
    tests  = [ ("  test_cell_integral(%s);" % i) for i in cell_integrals ]
    tests += [ ("  test_exterior_facet_integral(%s);" % i) for i in exterior_facet_integrals ]
    tests += [ ("  test_interior_facet_integral(%s);" % i) for i in interior_facet_integrals ]
    
    # Write file
    test_file = open(prefix + ".cpp", "w")
    test_file.write(_test_code % (prefix, "\n".join(tests)))
    test_file.close()

def build_pyop2_programs(bench, permissive, debug=False):

    # Get a list of all files
    header_files = [f for f in os.listdir(".") if f.endswith(".h")]
    header_files.sort()

    begin("Building test programs (%d header files found)" % len(header_files))
    
    # Set compiler options
    if not permissive:
        compiler_options = " -Werror"
    if bench > 0:
        info("Benchmarking activated")
        compiler_options = "-Wall -Werror"
    if debug:
        info("Debugging activated")
        compiler_options = "-Wall -Werror -g"
    info("Compiler options: %s" % compiler_options)

    # Iterate over all files
    for f in header_files:

        # Generate test code
        filename = _generate_test_code(f, bench)

        # Compile test code
        prefix = f.split(".h")[0]
        command = "g++ %s -o %s.bin %s.cpp -lboost_math_tr1" % (compiler_options, prefix, prefix)
        ok = run_command(command)

        # Check status
        if ok:
            info_green("%s OK" % prefix)
        else:
            info_red("%s failed" % prefix)

    end()
