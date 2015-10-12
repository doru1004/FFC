"""
This is the compiler, acting as the main interface for compilation
of forms and breaking the compilation into several sequential stages.
The output of each stage is the input of the next stage.

Compiler stage 0: Language, parsing
-----------------------------------

  Input:  Python code or .ufl file
  Output: UFL form

  This stage consists of parsing and expressing a form in the
  UFL form language.

  This stage is completely handled by UFL.

Compiler stage 1: Analysis
--------------------------

  Input:  UFL form
  Output: Preprocessed UFL form and FormData (metadata)

  This stage preprocesses the UFL form and extracts form metadata.
  It may also perform simplifications on the form.

Compiler stage 2: Code representation
-------------------------------------

  Input:  Preprocessed UFL form and FormData (metadata)
  Output: Intermediate Representation (IR)

  This stage examines the input and generates all data needed for code
  generation. This includes generation of finite element basis
  functions, extraction of data for mapping of degrees of freedom and
  possible precomputation of integrals.

  Most of the complexity of compilation is handled in this stage.

  The IR is stored as a dictionary, mapping names of UFC functions to
  data needed for generation of the corresponding code.

Compiler stage 3: Optimization
------------------------------

  Input:  Intermediate Representation (IR)
  Output: Optimized Intermediate Representation (OIR)

  This stage examines the IR and performs optimizations.

  Optimization is currently disabled as a separate stage
  but is implemented as part of the code generation for
  quadrature representation.

Compiler stage 4: Code generation
---------------------------------

  Input:  Optimized Intermediate Representation (OIR)
  Output: C++ code

  This stage examines the OIR and generates the actual C++ code for
  the body of each UFC function.

  The code is stored as a dictionary, mapping names of UFC functions
  to strings containing the C++ code of the body of each function.

Compiler stage 5: Code formatting
---------------------------------

  Input:  C++ code
  Output: C++ code files

  This stage examines the generated C++ code and formats it according
  to the UFC format, generating as output one or more .h/.cpp files
  conforming to the UFC format.

The main interface is defined by the following two functions:

  compile_form
  compile_element

The compiler stages are implemented by the following functions:

  analyze_forms
  or
  analyze_elements  (stage 1)
  compute_ir        (stage 2)
  optimize_ir       (stage 3)
  generate_code     (stage 4)
  format_code       (stage 5)
"""

# Copyright (C) 2007-2015 Anders Logg
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
# Modified by Kristian B. Oelgaard, 2010.
# Modified by Dag Lindbo, 2008.
# Modified by Garth N. Wells, 2009.
# Modified by Martin Alnaes, 2013-2015

__all__ = ["compile_form", "compile_element"]

# Python modules
from time import time
import os

# FFC modules
from ffc.log import info, info_green, warning
from ffc.parameters import default_parameters

# FFC modules
from ffc.analysis import analyze_forms, analyze_elements
from ffc.representation import compute_ir
from ffc.optimization import optimize_ir
from ffc.codegeneration import generate_code
from ffc.formatting import format_code, write_code
from ffc.wrappers import generate_wrapper_code

import ufl
from ffc.fiatinterface import create_actual_fiat_element
from ffc.cpp import format

def compile_form(forms, object_names=None, prefix="Form", parameters=None):
    """This function generates UFC code for a given UFL form or list
    of UFL forms."""

    info("Compiling form %s\n" % prefix)

    # Reset timing
    cpu_time_0 = time()

    # Check input arguments
    forms = _check_forms(forms)
    if not forms:
        return
    if prefix != os.path.basename(prefix):
        prefix = os.path.basename(prefix)
        warning("Invalid prefix, modified to {}.".format(prefix))
    if object_names is None:
        object_names = {}
    parameters = _check_parameters(parameters)

    # Stage 1: analysis
    cpu_time = time()
    analysis = analyze_forms(forms, parameters)
    _print_timing(1, time() - cpu_time)

    # Stage 2: intermediate representation
    cpu_time = time()
    ir = compute_ir(analysis, object_names, parameters)
    _print_timing(2, time() - cpu_time)

    # Stage 3: optimization
    cpu_time = time()
    oir = optimize_ir(ir, parameters)
    _print_timing(3, time() - cpu_time)

    # Return IR (PyOP2 mode) or code string (otherwise)
    if parameters["pyop2-ir"]:
        try:
            from ffc.quadrature.quadraturepyop2ir import generate_pyop2_ir
        except ImportError:
            raise ImportError("Format pyop2-ir depends on PyOP2, which is not available.")
        # Stage 4: build PyOP2 intermediate representation

        cpu_time = time()
        #FIXME: need a cleaner interface
        pyop2_ir = [generate_pyop2_ir(ir, prefix, parameters) for ir in oir[2]]
        _print_timing(4, time() - cpu_time)

        info_green("FFC finished in %g seconds.", time() - cpu_time_0)
        return pyop2_ir

    else:
        # Stage 4: code generation
        cpu_time = time()
        code = generate_code(oir, prefix, parameters)
        _print_timing(4, time() - cpu_time)

        # Stage 4.1: generate wrappers
        cpu_time = time()
        wrapper_code = generate_wrapper_code(analysis, prefix, object_names, parameters)
        _print_timing(4.1, time() - cpu_time)

        # Stage 5: format code
        cpu_time = time()
        code_h, code_c = format_code(code, wrapper_code, prefix, parameters)
        write_code(code_h, code_c, prefix, parameters) # FIXME: Don't write to file in this function (issue #72)
        _print_timing(5, time() - cpu_time)

        info_green("FFC finished in %g seconds.", time() - cpu_time_0)

        return code

def _dX_norm_square(topological_dimension):
    return " + ".join("dX[{0}]*dX[{0}]".format(i) for i in xrange(topological_dimension))

def _X_iadd_dX(topological_dimension):
    return "\n".join("\tX[{0}] -= dX[{0}];".format(i) for i in xrange(topological_dimension))

def _is_affine(ufl_element):
    return ufl_element.cell().cellname() in ufl.cell.affine_cells and ufl_element.degree() <= 1 and ufl_element.family() in ["Discontinuous Lagrange", "Lagrange"]

def compile_element(ufl_element, coordinates_ufl_element):
    parameters = _check_parameters(None)

    # Set code generation parameters
    from ffc.cpp import set_float_formatting
    set_float_formatting(int(parameters["precision"]))

    from ffc.codegeneration import _value_dimension, _inside_check, _to_reference_coordinates, _calculate_basisvalues, _init_X
    from ffc.representation import needs_oriented_jacobian

    # Create FIAT element
    element = create_actual_fiat_element(ufl_element)
    coordinates_element = create_actual_fiat_element(coordinates_ufl_element)
    domain, = ufl_element.domains() # Assuming single domain
    cell = domain.cell()

    # Compute data for each function
    if ufl_element.value_shape():
        odim = ufl_element.value_shape()[0]
    else:
        odim = 1

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "inside_predicate": _inside_check(cell, element.get_reference_element()),
        "to_reference_coords": _to_reference_coordinates(cell, coordinates_element, needs_oriented_jacobian(element)),
        "ndofs": element.space_dimension(),
        "n_coords_nodes": coordinates_element.space_dimension(),
        "calculate_basisvalues": _calculate_basisvalues(cell, element),
        "odim": odim,
        "init_X": _init_X(element),
        "max_iteration_count": 1 if _is_affine(coordinates_ufl_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": _dX_norm_square(cell.topological_dimension()),
        "X_iadd_dX": _X_iadd_dX(cell.topological_dimension()),
    }

    evaluate_template_c = """#include <math.h>

#include <evaluate.h>
#include <firedrake_geometry.h>

struct ReferenceCoords {
	double X[%(geometric_dimension)d];
	double J[%(geometric_dimension)d * %(topological_dimension)d];
	double K[%(topological_dimension)d * %(geometric_dimension)d];
	double detJ;
};

void to_reference_coords_kernel(void *result_, double *x0, int *return_value, double **C)
{
	struct ReferenceCoords *result = result_;

	const int space_dim = %(geometric_dimension)d;

	/*
	 * Mapping coordinates from physical to reference space
	 */

	double *X = result->X;
%(init_X)s
	double x[space_dim];
	double *J = result->J;
	double *K = result->K;
	double detJ;

    double dX[%(topological_dimension)d];
    int converged = 0;

    for (int it = 0; !converged && it < %(max_iteration_count)d; it++) {
%(to_reference_coords)s

         if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
             converged = 1;
         }

%(X_iadd_dX)s
    }

	result->detJ = detJ;

	// Are we inside the reference element?
	*return_value = %(inside_predicate)s;
}

extern void wrap_to_reference_coords(void *result_, double *x, int *return_value,
                                     double *coords, int *coords_map, int cell);

void wrap_to_reference_coords(void *result_, double *x, int *return_value, double *coords, int *coords_map, int cell)
{
	const int space_dim = %(geometric_dimension)d;

	double *C[%(n_coords_nodes)d];
	for (int r = 0; r < %(n_coords_nodes)d; r++) {
		C[r] = &coords[coords_map[cell * %(n_coords_nodes)d + r] * space_dim];
	}
	to_reference_coords_kernel(result_, x, return_value, C);
}

int to_reference_coords(void *result_, struct Function *f, int cell, double *x)
{
	int return_value;
	wrap_to_reference_coords(result_, x, &return_value, f->coords, f->coords_map, cell);
	return return_value;
}

int evaluate(struct Function *f, double *x, double *result)
{
	struct ReferenceCoords reference_coords;
	int cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &reference_coords);
	if (cell == -1) {
		return -1;
	}

	if (!result) {
		return 0;
	}

%(calculate_basisvalues)s

    const int odim = %(odim)d;
    for (int q = 0; q < odim; q++) {
        result[q] = 0.0;
    }

    // Wrapper stuff
    double *F[%(ndofs)d];
    for (int r = 0; r < %(ndofs)d; r++) {
        F[r] = &f->f[f->f_map[cell * %(ndofs)d + r] * odim];
    }

    for (int i = 0; i < %(ndofs)d; i++) {
        for (int q = 0; q < odim; q++) {
            result[q] += F[i][q] * phi[i];
        }
    }
	return 0;
}
"""

    return evaluate_template_c % code


def _check_forms(forms):
    "Initial check of forms."
    if not isinstance(forms, (list, tuple)):
        forms = (forms,)
    return forms

def _check_elements(elements):
    "Initial check of elements."
    if not isinstance(elements, (list, tuple)):
        elements = (elements,)
    return elements

def _check_parameters(parameters):
    "Initial check of parameters."
    if parameters is None:
        parameters = default_parameters()
    if "blas" in parameters:
        warning("BLAS mode unavailable (will return in a future version).")
    if "quadrature_points" in parameters:
        warning("Option 'quadrature_points' has been replaced by 'quadrature_degree'.")
    return parameters

def _print_timing(stage, timing):
    "Print timing results."
    info("Compiler stage %s finished in %g seconds.\n" % (str(stage), timing))
