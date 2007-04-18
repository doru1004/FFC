"""Code generation for evaluation of finite element basis values. This module generates
code which is more or less a C++ representation of FIAT code. More specifically the
functions from the modules expansion.py and jacobi.py are translated into C++"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-04-04 -- 2007-04-16"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU GPL Version 2"

# FFC common modules
from ffc.common.constants import *
from ffc.common.utils import *

# FFC fem modules
from ffc.fem.finiteelement import *
from ffc.fem.mixedelement import *

# FFC format modules
#from ffc.compiler.format.codesnippets import *

# Python modules
import math
import numpy

class IndentControl:
    "Class to control the indentation of code"

    def __init__(self):
        "Constructor"
        self.size = 0
        self.increment = 2

    def increase(self):
        "Increase indentation by increment"
        self.size += self.increment

    def decrease(self):
        "Decrease indentation by increment"
        self.size -= self.increment

    def indent(self, a):
        "Indent string input string by size"
        return indent(a, self.size)

def evaluate_basis(element, format):
    """Evaluate an element basisfunction at a point. The value(s) of the basisfunction is/are
    computed as in FIAT as the dot product of the coefficients (computed at compile time)
    and basisvalues which are dependent on the coordinate and thus have to be computed at
    run time.

    Currently the following elements are supported in 2D and 3D:

    Lagrange                + mixed/vector valued
    Discontinuous Lagrange  + mixed/vector valued
    Crouzeix-Raviart        + mixed/vector valued
    Brezzi-Douglas-Marini   + mixed/vector valued

    Not supported in 2D or 3D:

    Raviart-Thomas ? (not tested since it is broken in FFC, but should work)
    Nedelec (broken?)"""

# To be fixed:
# check that the code is language-independent

    code = []

    Indent = IndentControl()

    # Get coordinates and generate map
    code += generate_map(element, Indent, format)

    # Check if we have just one element
    if (element.num_sub_elements() == 1):
        code += generate_element_code(element, Indent, format)

    # If the element is vector valued or mixed
    else:
        code += generate_cases(element, Indent, format)

    return code

def generate_element_code(element, Indent, format):
    "Generate code for each basis element"

    code = []

    # Tabulate coefficients
    code += tabulate_coefficients(element, Indent, format)

    # Get coordinates and generate map
#    code += generate_map(element, Indent, format)

    # Compute scaling of y and z 1/2(1-y)^n and 1/2(1-z)^n
    code += compute_scaling(element, Indent, format)

    # Compute auxilliary functions currently only 2D and 3D is supported
    if (element.cell_shape() == 2):
        code += compute_psitilde_a(element, Indent, format)
        code += compute_psitilde_b(element, Indent, format)
    elif (element.cell_shape() == 3):
        code += compute_psitilde_a(element, Indent, format)
        code += compute_psitilde_b(element, Indent, format)
        code += compute_psitilde_c(element, Indent, format)
    else:
        raise RuntimeError(), "Cannot compute auxilliary functions for shape: %d" %(element.cell_shape())

    # Compute the basisvalues
    code += compute_basisvalues(element, Indent, format)

    # Compute the value of the basisfunction as the dot product of the coefficients
    # and basisvalues
    code += dot_product(element, Indent, format)

    return code

def tabulate_coefficients(element, Indent, format):
    """This function tabulates the element coefficients that are generated by FIAT at
    compile time."""

    code = []

    # Prefetch formats to speed up code generation
    format_block_begin  = format["block begin"]
    format_block_end    = format["block end"]

    # Get coefficients from basis functions, computed by FIAT at compile time
    coefficients = element.basis().coeffs

    # Get shape of coefficients
    shape = numpy.shape(coefficients)

    # Scalar valued basis element [Lagrange, Discontinuous Lagrange, Crouzeix-Raviart]
    if (len(shape) == 2):
        num_components = 1
        poly_dim = shape[1]
        coefficients = [coefficients]

    # Vector valued basis element [Raviart-Thomas, Brezzi-Douglas-Marini (BDM)]
    elif (len(shape) == 3):
        num_components = shape[1]
        poly_dim = shape[2]
        coefficients = numpy.transpose(coefficients, [1,0,2])

    # ???
    else:
        raise RuntimeError(), "These coefficients have a strange shape!"

    # Get the number of dofs from element
    num_dofs = element.space_dimension()

    code += [Indent.indent(format["comment"]("Table(s) of coefficients"))]

    # Generate tables for each component
    for i in range(num_components):

        # Extract coefficients for current component
        coeffs = coefficients[i]

        # Declare varable name for coefficients
        name = format["table declaration"] + "coefficients%d[%d][%d]" %(i, num_dofs, poly_dim,)

        # Generate array of values
        value = "\\\n" + format_block_begin
        rows = []
        for j in range(num_dofs):
            rows += [format_block_begin + ", ".join([format["floating point"](coeffs[j,k])\
                     for k in range(poly_dim)]) + format_block_end]

        value += ",\n".join(rows)
        value += format_block_end

        code += [(Indent.indent(name), Indent.indent(value))] + [""]

    return code


def generate_map(element, Indent, format):
    """Generates map from reference triangle/tetrahedron to reference square/cube.
    The function is an implementation of the FIAT functions, eta_triangle( xi )
    and eta_tetrahedron( xi ) from expansions.py"""

    code = []

    # Prefetch formats to speed up code generation
    format_comment      = format["comment"]
    format_float        = format["float declaration"]
    format_floating_point = format["floating point"]
    format_coordinates  = format["coordinate access"]

    # Code snippets reproduced from FIAT: expansions.py: eta_triangle(xi) & eta_tetrahedron(xi)
    eta_triangle = [Indent.indent(format["snippet eta_triangle"]) %(format_floating_point(FFC_EPSILON))]

    eta_tetrahedron = [Indent.indent(format["snippet eta_tetrahedron"]) %(format_floating_point(FFC_EPSILON),\
                       format_floating_point(FFC_EPSILON))]

    # Dictionaries
    reference = {2:"square", 3:"cube"}
    mappings = {2:eta_triangle, 3:eta_tetrahedron}

    # Generate code
    # Get coordinates and map to the reference (FIAT) element from codesnippets.py
    code += [Indent.indent(format["coordinate map"](element.cell_shape()))] + [""]

    # Map coordinates to the reference square/cube
    code += [Indent.indent(format_comment("Map coordinates to the reference %s") % (reference[element.cell_shape()]))]
    code += mappings[element.cell_shape()]

    return code + [""]

def compute_scaling(element, Indent, format):
    """Generate the scalings of y and z coordinates. This function is an implementation of
    the FIAT function make_scalings( etas ) from expasions.py"""

    code = []

    # Get the element degree
    degree = element.degree()

    # Get the element shape
    element_shape = element.cell_shape()

    # Currently only 2D and 3D is supported
    if (element_shape == 2):
        scalings = ["y"]
        # Scale factor, for triangles 1/2*(1-y)^i i being the order of the element
        factors = ["(0.5 - 0.5 * y)"]
    elif (element_shape == 3):
        scalings = ["y", "z"]
        factors = ["(0.5 - 0.5 * y)", "(0.5 - 0.5 * z)"]
    else:
        raise RuntimeError(), "Cannot compute scaling for shape: %d" %(elemet_shape)

    code += [Indent.indent(format["comment"]("Generate scalings"))]

    for i in range(len(scalings)):

      # Old 'array' code
      # Declare scaling variable
#      name = format["const float declaration"] + "scalings_%s[%d]" %(scalings[i], degree+1,)
#      value = format["block begin"] + "1.0"
#      if degree > 0:
#          value += ", " + ", ".join(["scalings_%s[%d]*%s" %(scalings[i],j-1,factors[i])\
#                                     for j in range(1, degree+1)])
#      value += format["block end"]
#      code += [(Indent.indent(name), Indent.indent(value))] + [""]

      name = format["const float declaration"] + "scalings_%s_%d" %(scalings[i], 0,)
      value = "1.0"
      code += [(Indent.indent(name), Indent.indent(value))]

      for j in range(1, degree+1):
          name = format["const float declaration"] + "scalings_%s_%d" %(scalings[i], j,)
          value = "scalings_%s_%d*%s" %(scalings[i],j-1,factors[i])
          code += [(Indent.indent(name), value)]

    return code + [""]

def compute_psitilde_a(element, Indent, format):
    """Compute Legendre functions in x-direction. The function relies on
    eval_jacobi_batch(a,b,n) to compute the coefficients.

    The format is:
    psitilde_a[0] = 1.0
    psitilde_a[1] = a + b * x
    psitilde_a[n] = a * psitilde_a[n-1] + b * psitilde_a[n-1] * x + c * psitilde_a[n-2]
    where a, b and c are coefficients computed by eval_jacobi_batch(0,0,n)
    and n is the element degree"""

    code = []

    # Prefetch formats to speed up code generation
    format_float = format["floating point"]

    # Get the element degree
    degree = element.degree()

    code += [Indent.indent(format["comment"]("Compute psitilde_a"))]

    # Create list of variable names
    variables = ["x","psitilde_a"]

    # Old 'array' code
    # Declare variable
#    name = format["const float declaration"] + variables[1] + "[%d]" %(degree+1,)

    # Compute values
#    value = eval_jacobi_batch_array(0, 0, degree, variables, format)

#    code += [(Indent.indent(name), Indent.indent(value))]

    for n in range(degree+1):
        # Declare variable
        name = format["const float declaration"] + variables[1] + "_%d" %(n,)

        # Compute value
        value = eval_jacobi_batch_scalar(0, 0, n, variables, format)

        code += [(Indent.indent(name), Indent.indent(value))]

    return code + [""]

def compute_psitilde_b(element, Indent, format):
    """Compute Legendre functions in y-direction. The function relies on
    eval_jacobi_batch(a,b,n) to compute the coefficients.

    The format is:
    psitilde_bs_0[0] = 1.0
    psitilde_bs_0[1] = a + b * y
    psitilde_bs_0[n] = a * psitilde_bs_0[n-1] + b * psitilde_bs_0[n-1] * x + c * psitilde_bs_0[n-2]
    psitilde_bs_(n-1)[0] = 1.0
    psitilde_bs_(n-1)[1] = a + b * y
    psitilde_bs_n[0] = 1.0
    where a, b and c are coefficients computed by eval_jacobi_batch(2*i+1,0,n-i) with i in range(0,n+1)
    and n is the element degree + 1"""

    code = []

    # Prefetch formats to speed up code generation
    format_float = format["floating point"]

    # Get the element degree
    degree = element.degree()

    code += [Indent.indent(format["comment"]("Compute psitilde_bs"))]

    for i in range(0, degree + 1):

        # Compute constants for jacobi function
        a = 2*i+1
        b = 0
        n = degree - i
            
        # Create list of variable names
        variables = ["y", "psitilde_bs_%d" %(i)]

        # Old 'array' code
        # Declare variable
#        name = format["const float declaration"] + variables[1] + "[%d]" %(n+1)

        # Compute values
#        value = eval_jacobi_batch_array(a, b, n, variables, format)

#        code += [(Indent.indent(name), Indent.indent(value))] + [""]

        for j in range(n+1):
            # Declare variable
            name = format["const float declaration"] + variables[1] + "_%d" %(j)

            # Compute values
            value = eval_jacobi_batch_scalar(a, b, j, variables, format)

            code += [(Indent.indent(name), value)]

    return code + [""]

def compute_psitilde_c(element, Indent, format):
    """Compute Legendre functions in y-direction. The function relies on
    eval_jacobi_batch(a,b,n) to compute the coefficients.

    The format is:
    psitilde_cs_0[0] = 1.0
    psitilde_cs_0[1] = a + b * y
    psitilde_cs_0[n] = a * psitilde_cs_0[n-1] + b * psitilde_cs_0[n-1] * x + c * psitilde_cs_0[n-2]
    psitilde_cs_(n-1)[0] = 1.0
    psitilde_cs_(n-1)[1] = a + b * y
    psitilde_cs_n[0] = 1.0
    where a, b and c are coefficients computed by 

    [[jacobi.eval_jacobi_batch(2*(i+j+1),0, n-i-j) for j in range(0,n+1-i)] for i in range(0,n+1)]"""

    code = []

    # Prefetch formats to speed up code generation
    format_float = format["floating point"]

    # Get the element degree
    degree = element.degree()

    code += [Indent.indent(format["comment"]("Compute psitilde_cs"))]

    for i in range(0, degree + 1):
        for j in range(0, degree + 1 - i):

            # Compute constants for jacobi function
            a = 2*(i+j+1)
            b = 0
            n = degree - i - j
            
            # Create list of variable names
            variables = ["z", "psitilde_cs_%d%d" %(i,j)]

            # Old 'array' code
            # Declare variable
#            name = format["const float declaration"] + variables[1] + "[%d]" %(n+1)

            # Compute values
#            value = eval_jacobi_batch_array(a, b, n, variables, format)

#            code += [(Indent.indent(name), Indent.indent(value))] + [""]

            for k in range(n+1):
                # Declare variable
                name = format["const float declaration"] + variables[1] + "_%d" %(k)

                # Compute values
                value = eval_jacobi_batch_scalar(a, b, k, variables, format)

                code += [(Indent.indent(name), value)]

    return code + [""]


def compute_basisvalues(element, Indent, format):
    """This function is an implementation of the loops inside the FIAT functions
    tabulate_phis_triangle( n , xs ) and tabulate_phis_tetrahedron( n , xs ) in
    expansions.py. It computes the basis values from all the previously tabulated variables."""

    code = []
    code += [Indent.indent(format["comment"]("Compute basisvalues"))]

    # Get coefficients from basis functions, computed by FIAT at compile time
    coefficients = element.basis().coeffs

    # Get shape of coefficients
    shape = numpy.shape(coefficients)

    # Scalar valued basis element [Lagrange, Discontinuous Lagrange, Crouzeix-Raviart]
    if (len(shape) == 2):
        poly_dim = shape[1]

    # Vector valued basis element [Raviart-Thomas, Brezzi-Douglas-Marini (BDM)]
    elif (len(shape) == 3):
        poly_dim = shape[2]

    # ???
    else:
        raise RuntimeError(), "These coefficients have a strange shape!"

    # Declare variable
    name = format["const float declaration"] + "basisvalues[%d]" %(poly_dim,)
    value = "\\\n"

    # Get the element shape
    element_shape = element.cell_shape()

    # Currently only 2D and 3D is supported
    # 2D
    if (element_shape == 2):
        value += format["block begin"]
        var = []
        for k in range(0,element.degree() + 1):
            for i in range(0,k + 1):
                ii = k-i
                jj = i
                factor = math.sqrt( (ii+0.5)*(ii+jj+1.0) )
                var += [format["multiply"](["psitilde_a_%d" %(ii), "scalings_y_%d" %(ii),\
                        "psitilde_bs_%d_%d" %(ii,jj), format["floating point"](factor)])]

        value += ",\n".join(var)
        value += format["block end"]
    # 3D
    elif (element_shape == 3):
        value += format["block begin"]
        var = []

        for k in range(0, element.degree()+1):  # loop over degree
            for i in range(0, k+1):
                for j in range(0, k - i + 1):
                    ii = k-i-j
                    jj = j
                    kk = i
                    factor = math.sqrt( (ii+0.5) * (ii+jj+1.0) * (ii+jj+kk+1.5) )
                    var += []
                    var += [format["multiply"](["psitilde_a_%d" %(ii), "scalings_y_%d" %(ii),\
                        "psitilde_bs_%d_%d" %(ii,jj), "scalings_z_%d" %(ii+jj),\
                        "psitilde_cs_%d%d_%d" %(ii,jj,kk), format["floating point"](factor)])]

        value += ",\\\n ".join(var)
        value += format["block end"]
    else:
        raise RuntimeError(), "Cannot compute basis values for shape: %d" %(elemet_shape)

    code += [(Indent.indent(name), Indent.indent(value))]

    return code + [""]

def dot_product(element, Indent, format):
    """This function computes the value of the basisfunction as the dot product of the
    coefficients and basisvalues """

    code = []

    code += [Indent.indent(format["comment"]("Compute value(s)"))]

    # Get coefficients from basis functions, computed by FIAT at compile time
    coefficients = element.basis().coeffs

    # Get shape of coefficients
    shape_coeff = numpy.shape(coefficients)

    # Scalar valued basis element [Lagrange, Discontinuous Lagrange, Crouzeix-Raviart]
    if (len(shape_coeff) == 2):
        poly_dim = shape_coeff[1]

        # Reset value as it is a pointer
        code += [(Indent.indent("*values"), "0.0")]

        # Loop dofs to generate dot product, 3D ready
        code += [Indent.indent(format["loop"]("j", "j", poly_dim, "j"))]

        # Increase indentation
        Indent.increase()

        code += [Indent.indent(format["add equal"]("*values","coefficients0[i][j]*basisvalues[j]"))]

        # Decrease indentation
        Indent.decrease()

    # Vector valued basis element [Raviart-Thomas, Brezzi-Douglas-Marini (BDM)]
    elif (len(shape_coeff) == 3):
        num_components = shape_coeff[1]
        poly_dim = shape_coeff[2]

        # Reset value as it is a pointer
        code += [(Indent.indent("values[%d]" %(i)), "0.0") for i in range(num_components)]

        # Loop dofs to generate dot product, 3D ready
        code += [Indent.indent(format["loop"]("j", "j", poly_dim, "j"))]
        code += [Indent.indent(format["block begin"])]

        # Increase indentation
        Indent.increase()

        code += [Indent.indent(format["add equal"]("values[%d]" %(i),\
                 "coefficients%d[i][j]*basisvalues[j]" %(i))) for i in range(num_components)]

        # Decrease indentation
        Indent.decrease()

        code += [Indent.indent(format["block end"])]

    # ???
    else:
        raise RuntimeError(), "These coefficients have a strange shape!"

    return code

def eval_jacobi_batch_array(a, b, n, variables, format):
    """Implementation of FIAT function eval_jacobi_batch(a,b,n,xs) from jacobi.py"""

    # Prefetch formats to speed up code generation
    format_float  = format["floating point"]
    format_mult   = format["multiply"]
    format_add   = format["add"]

    # Format variables
    access = lambda i: variables[1] + "[%d]" %(i)
    coord = variables[0]

    # Entry 0 is always 1.0
    value = format["block begin"] + "1.0"

    if n > 0:
        # Results for entry 1, of type (a + b * coordinate) (coordinate = x, y or z)
        res0 = 0.5 * (a - b)
        res1 = 0.5 * ( a + b + 2.0 )

        val0, val1 = ("", "")
        if (res0 != 0.0): # Only include if the value is not zero
            val0 = format_float(res0)

        if (res1 != 0.0): # Only include if the value is not zero
            if (res1 < 0.0): # If value is less than zero minus sign is needed
                val1 = format_mult([format_float(res1), coord])
            else:
                if (val0): # If the value in front is present plus sign is needed
                    val1 = " + " + format_mult([format_float(res1), coord])
                else:
                    val1 = format_mult([format_float(res1), coord])

        value += ", " + "".join([val0, val1])

        apb = a + b
        # Compute remaining entries, of type (a + b * coordinate) * psitilde[k-1] - c * psitilde[k-2])
        for k in range(2,n+1):
            a1 = 2.0 * k * ( k + apb ) * ( 2.0 * k + apb - 2.0 )
            a2 = ( 2.0 * k + apb - 1.0 ) * ( a * a - b * b )
            a3 = ( 2.0 * k + apb - 2.0 ) * ( 2.0 * k + apb - 1.0 ) * ( 2.0 * k + apb )
            a4 = 2.0 * ( k + a - 1.0 ) * ( k + b - 1.0 ) * ( 2.0 * k + apb )
            a2 = a2 / a1
            a3 = a3 / a1
            # Note:  we subtract the value of a4!
            a4 = -a4 / a1

            val2, val3, val4 = ("", "", "")
            if (a2 != 0.0): # Only include if the value is not zero
                val2 = format_mult([format_float(a2), access(k-1)])

            if (a3 != 0.0): # Only include if the value is not zero
                if (a3 < 0.0): # If value is less than zero minus sign is needed
                   val3 = format_mult([format_float(a3), coord, access(k-1)])
                else:
                    if (val2): # If the value in front is present plus sign is needed
                        val3 = " + " + format_mult([format_float(a3), coord, access(k-1)])
                    else:
                        val3 = format_mult([format_float(a3), coord, access(k-1)])

            if (a4 != 0.0): # Only include if the value is not zero
                if (a4 < 0.0): # If value is less than zero minus sign is needed
                    val4 = format_mult([format_float(a4), access(k-2)])
                else:
                    if (val2 or val3): # If the value(s) in front is/are present plus sign is needed
                        val4 = " + " + format_mult([format_float(a4), access(k-2)])
                    else:
                        val4 = format_mult([format_float(a4), access(k-2)])

            value += ", " + "".join([val2, val3, val4])

    value += format["block end"]

    return value

def eval_jacobi_batch_scalar(a, b, n, variables, format):
    """Implementation of FIAT function eval_jacobi_batch(a,b,n,xs) from jacobi.py"""

    # Prefetch formats to speed up code generation
    format_float  = format["floating point"]
    format_mult   = format["multiply"]
    format_add   = format["add"]

    # Format variables
    access = lambda i: variables[1] + "_%d" %(i)
    coord = variables[0]

    # Entry 0 is always 1.0
    value = format["block begin"] + "1.0"

    if n == 0:
        return "1.0"
    if n == 1:
        # Results for entry 1, of type (a + b * coordinate) (coordinate = x, y or z)
        res0 = 0.5 * (a - b)
        res1 = 0.5 * ( a + b + 2.0 )
        val0, val1 = ("", "")
        if (res0 != 0.0): # Only include if the value is not zero
            val0 = format_float(res0)

        if (res1 != 0.0): # Only include if the value is not zero
            if (res1 < 0.0): # If value is less than zero minus sign is needed
                val1 = format_mult([format_float(res1), coord])
            else:
                if (val0): # If the value in front is present plus sign is needed
                    val1 = " + " + format_mult([format_float(res1), coord])
                else:
                    val1 = format_mult([format_float(res1), coord])

        return "".join([val0, val1])

    else:
        apb = a + b
        # Compute remaining entries, of type (a + b * coordinate) * psitilde[n-1] - c * psitilde[n-2])
        a1 = 2.0 * n * ( n + apb ) * ( 2.0 * n + apb - 2.0 )
        a2 = ( 2.0 * n + apb - 1.0 ) * ( a * a - b * b )
        a3 = ( 2.0 * n + apb - 2.0 ) * ( 2.0 * n + apb - 1.0 ) * ( 2.0 * n + apb )
        a4 = 2.0 * ( n + a - 1.0 ) * ( n + b - 1.0 ) * ( 2.0 * n + apb )
        a2 = a2 / a1
        a3 = a3 / a1
        # Note:  we subtract the value of a4!
        a4 = -a4 / a1

        val2, val3, val4 = ("", "", "")
        if (a2 != 0.0): # Only include if the value is not zero
            val2 = format_mult([format_float(a2), access(n-1)])

        if (a3 != 0.0): # Only include if the value is not zero
            if (a3 < 0.0): # If value is less than zero minus sign is needed
                val3 = format_mult([format_float(a3), coord, access(n-1)])
            else:
                if (val2): # If the value in front is present plus sign is needed
                    val3 = " + " + format_mult([format_float(a3), coord, access(n-1)])
                else:
                    val3 = format_mult([format_float(a3), coord, access(n-1)])

        if (a4 != 0.0): # Only include if the value is not zero
            if (a4 < 0.0): # If value is less than zero minus sign is needed
                val4 = format_mult([format_float(a4), access(n-2)])
            else:
                if (val2 or val3): # If the value(s) in front is/are present plus sign is needed
                    val4 = " + " + format_mult([format_float(a4), access(n-2)])
                else:
                    val4 = format_mult([format_float(a4), access(n-2)])

        return "".join([val2, val3, val4])

def generate_cases(element, Indent, format):
    "Generate cases in the event of vector valued elements or mixed elements"

    # Prefetch formats to speed up code generation
    format_block_begin = format["block begin"]
    format_block_end = format["block end"]

    # Extract basis elements
    elements = extract_elements(element)

    code, unique_elements = element_types(elements, Indent, format)

    code += dof_map(elements, Indent, format) + [""]

    code += [Indent.indent(format["comment"]("Switch for each of the unique sub elements"))]

    # Get number of unique sub elements
    num_unique_elements = len(unique_elements)

    # Generate switch
    if (num_unique_elements > 1):
        code += [Indent.indent(format["switch"]("element"))]
        code += [Indent.indent(format_block_begin)]

        # Increase indentation
        Indent.increase()

        # Generate case
        for i in range(len(unique_elements)):
            code += [Indent.indent(format["case"](i))]
            code += [Indent.indent(format_block_begin)]

            # Increase indentation
            Indent.increase()

            # Get unique sub element
            element = unique_elements[i]

            # Generate code for unique sub element
            code += generate_element_code(element, Indent, format)

            code += [Indent.indent(format["break"])]

            # Decrease indentation
            Indent.decrease()

            code += [Indent.indent(format_block_end)]

        # Decrease indentation
        Indent.decrease()

        code += [Indent.indent(format_block_end)]

    else:
        element = unique_elements[0]
        code += generate_element_code(element, Indent, format)

    return code

def extract_elements(element):
    """This function extracts the individual elements from vector elements and mixed elements.
    Example, the following mixed element:

    element1 = FiniteElement("Lagrange", "triangle", 1)
    element2 = VectorElement("Lagrange", "triangle", 2)

    element  = element2 + element1

    has the structure: mixed-element[mixed-element[Lagrange order 2, Lagrange order 2], Lagrange order 1]

    This function returns the list of basis elements:
    elements = [Lagrange order 2, Lagrange order 2, Lagrange order 1]"""

    elements = [element.sub_element(i) for i in range(element.num_sub_elements())]

    mixed = True
    while (mixed == True):
        mixed = False
        for i in range(len(elements)):
            sub_element = elements[i]
            if isinstance(sub_element, MixedElement):
                mixed = True
                elements.pop(i)
                for j in range(sub_element.num_sub_elements()):
                    elements.insert(i+j, sub_element.sub_element(j))

    return elements

def element_types(elements, Indent, format):
    """This function creates a list of element types and a list of unique elements.

    Example, the following mixed element:

    element1 = FiniteElement("Lagrange", "triangle", 1)
    element2 = VectorElement("Lagrange", "triangle", 2)

    element  = element2 + element1

    has the element list, elements = [Lagrange order 2, Lagrange order 2, Lagrange order 1]

    Unique elements are: unique_elements = [Lagrange order 2, Lagrange order 1]
    and the element types, element_types = [0, 0, 1]"""

    code = []

    # Prefetch formats to speed up code generation
    format_block_begin = format["block begin"]
    format_block_end = format["block end"]

    unique_elements = [elements[0]]
    types = [0]

    for i in range(1, len(elements)):
        unique = True
        element = elements[i]
        elem_type = len(unique_elements)
        for j in range(elem_type):
            if (element.signature() == unique_elements[j].signature()):
                unique = False
                elem_type = j
                break
        if unique:
            unique_elements += [element]
        types += [elem_type]

    code += [Indent.indent(format["comment"]("Element types"))]

    # Declare element types and tabulate
    name = format["const uint declaration"] + "element_types[%d]" %(len(elements),)
    value = format_block_begin
    value += ", ".join(["%d" %(element_type) for element_type in types])
    value += format_block_end
    code += [(Indent.indent(name), value)] + [""]

    # Declare dofs_per_element variable and tabulate
    code += [(format["comment"]("Number of degrees of freedom per element"))]
    name = format["const uint declaration"] + "dofs_per_element[%d]" %(len(elements),)
    value = format_block_begin
    value += ", ".join(["%d" %(element.space_dimension()) for element in elements])
    value += format_block_end
    code += [(Indent.indent(name), value)] + [""]


    return (code, unique_elements)


def dof_map(elements, Indent, format):
    """This function creates code to map a basis function to a local basis function.
    Example, the following mixed element:

    element = VectorElement("Lagrange", "triangle", 2)

    has the element list, elements = [Lagrange order 2, Lagrange order 2] and 12 dofs (6 each).

    However since only one unique element exists, the evaluation of basis function 8 is 
    mapped to 2 (8-6) for the unique element."""

    # Use snippet from codesnippets.py    
    code = [Indent.indent(format["comment"]("Map basis function to local basisfunction"))]
    code += [Indent.indent(format["snippet dof map"] % len(elements))]

    return code