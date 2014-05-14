"This module defines rules and algorithms for generating C++ code."

# Copyright (C) 2009-2013 Anders Logg
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
# Modified by Kristian B. Oelgaard 2011
# Modified by Marie E. Rognes 2010
# Modified by Martin Alnaes 2013
#
# First added:  2009-12-16
# Last changed: 2014-04-02

# Python modules
import re, numpy, platform

# FFC modules
from ffc.log import debug, error

# Mapping of restrictions
_fixed_map = {None: "", "+": "_0", "-": "_1"}
_choose_map = lambda r: _fixed_map[r] if r in _fixed_map else "_%s" % str(r)

# FIXME: MSA: Using a dict to collect functions in a namespace is weird
#             and makes the code harder to follow, change to a class
#             with member functions instead!
# FIXME: KBO: format is a builtin_function, i.e., we should use a different name.
# Formatting rules
format = {}

# Program flow
format.update({
    "return":         lambda v: "return %s;" % str(v),
    "grouping":       lambda v: "(%s)" % v,
    "block":          lambda v: "{%s}" % v,
    "block begin":    "{",
    "block end":      "}",
    "list":           lambda v: format["block"](format["list separator"].join([str(l) for l in v])),
    "switch":         lambda v, cases, default=None, numbers=None: _generate_switch(v, cases, default, numbers),
    "exception":      lambda v: "throw std::runtime_error(\"%s\");" % v,
    "warning":        lambda v: 'std::cerr << "*** FFC warning: " << "%s" << std::endl;' % v,
    "comment":        lambda v: "// %s" % v,
    "if":             lambda c, v: "if (%s)\n{\n%s\n}\n" % (c, v),
    "loop":           lambda i, j, k: "for (unsigned int %s = %s; %s < %s; %s++)"% (i, j, i, k, i),
    "generate loop":  lambda v, w, _indent=0: _generate_loop(v, w, _indent),
    "is equal":       " == ",
    "not equal":      " != ",
    "less than":      " < ",
    "greater than":   " > ",
    "less equal":     " <= ",
    "greater equal":  " >= ",
    "and":            " && ",
    "or":             " || ",
    "not":            lambda v: "!(%s)" % v,
    "do nothing":     "// Do nothing"
})

# Declarations
format.update({
    "declaration":                    lambda t, n, v=None: _declaration(t, n, v),
    "float declaration":              "double",
    "int declaration":                "int",
    "uint declaration":               "unsigned int",
    "static const uint declaration":  {"ufc": "static const unsigned int",
                                       "pyop2": "const unsigned uint"},
    "static const float declaration": {"ufc": "static const double",
                                       "pyop2": "const double"},
    "vector table declaration":       "std::vector< std::vector<double> >",
    "double array declaration":       "double*",
    "const double array declaration": "const double*",
    "const float declaration":        lambda v, w: "const double %s = %s;" % (v, w),
    "const uint declaration":         lambda v, w: "const unsigned int %s = %s;" % (v, w),
    "dynamic array":                  lambda t, n, s: "%s *%s = new %s[%s];" % (t, n, t, s),
    "static array":                   lambda t, n, s: "static %s %s[%d];" % (t, n, s),
    "fixed array":                    lambda t, n, s: "%s %s[%d];" % (t, n, s),
    "delete dynamic array":           lambda n, s=None: _delete_array(n, s),
    "create foo":                     lambda v: "new %s()" % v
})

# Mathematical operators
format.update({
    "add":            lambda v: " + ".join(v),
    "iadd":           lambda v, w: "%s += %s;" % (str(v), str(w)),
    "sub":            lambda v: " - ".join(v),
    "neg":            lambda v: "-%s" % v,
    "mul":            lambda v: "*".join(v),
    "imul":           lambda v, w: "%s *= %s;" % (str(v), str(w)),
    "div":            lambda v, w: "%s/%s" % (str(v), str(w)),
    "inverse":        lambda v: "(1.0/%s)" % v,
    "std power":      {'ufc': lambda base, exp: "std::pow(%s, %s)" % (base, exp), 'pyop2': lambda base, exp: "pow(%s, %s)" % (base, exp)},
    "exp":            {'ufc': lambda v: "std::exp(%s)" % str(v), 'pyop2': lambda v: "exp(%s)" % str(v)},
    "ln":             {'ufc': lambda v: "std::log(%s)" % str(v), 'pyop2': lambda v: "log(%s)" % str(v)},
    "cos":            {'ufc': lambda v: "std::cos(%s)" % str(v), 'pyop2': lambda v: "cos(%s)" % str(v)},
    "sin":            {'ufc': lambda v: "std::sin(%s)" % str(v), 'pyop2': lambda v: "sin(%s)" % str(v)},
    "tan":            {'ufc': lambda v: "std::tan(%s)" % str(v), 'pyop2': lambda v: "tan(%s)" % str(v)},
    "cosh":           {'ufc': lambda v: "std::cosh(%s)" % str(v), 'pyop2': lambda v: "cosh(%s)" % str(v)},
    "sinh":           {'ufc': lambda v: "std::sinh(%s)" % str(v), 'pyop2': lambda v: "sinh(%s)" % str(v)},
    "tanh":           {'ufc': lambda v: "std::tanh(%s)" % str(v), 'pyop2': lambda v: "tanh(%s)" % str(v)},
    "acos":           {'ufc': lambda v: "std::acos(%s)" % str(v), 'pyop2': lambda v: "acos(%s)" % str(v)},
    "asin":           {'ufc': lambda v: "std::asin(%s)" % str(v), 'pyop2': lambda v: "asin(%s)" % str(v)},
    "atan":           {'ufc': lambda v: "std::atan(%s)" % str(v), 'pyop2': lambda v: "atan(%s)" % str(v)},
    "atan_2":         lambda v1,v2: "std::atan2(%s,%s)" % (str(v1),str(v2)),
    "erf":            {'ufc': lambda v: "erf(%s)" % str(v), 'pyop2': lambda v: "erf(%s)" % str(v)},
    "bessel_i":       lambda v, n: "boost::math::cyl_bessel_i(%s, %s)" % (str(n), str(v)),
    "bessel_j":       lambda v, n: "boost::math::cyl_bessel_j(%s, %s)" % (str(n), str(v)),
    "bessel_k":       lambda v, n: "boost::math::cyl_bessel_k(%s, %s)" % (str(n), str(v)),
    "bessel_y":       lambda v, n: "boost::math::cyl_neumann(%s, %s)" % (str(n), str(v)),
    "absolute value": {'ufc': lambda v: "std::abs(%s)" % str(v), 'pyop2': lambda v: "fabs(%s)" % str(v)},
    "sqrt":           {'ufc': lambda v: "std::sqrt(%s)" % str(v), 'pyop2': lambda v: "sqrt(%s)" % str(v)},
    "addition":       lambda v: _add(v),
    "multiply":       lambda v: _multiply(v),
    "power":          lambda base, exp: _power(base, exp),
    "inner product":  lambda v, w: _inner_product(v, w),
    "assign":         lambda v, w: "%s = %s;" % (v, str(w)),
    "component":      lambda v, k: _component(v, k)
})

# Formatting used in tabulate_tensor
format.update({
    "geometry tensor": lambda j, a: "G%d_%s" % (j, "_".join(["%d" % i for i in a]))
})

# Geometry related variable names (from code snippets).
format.update({
    "entity index":       "c.entity_indices",
    "num entities":       "num_global_entities",
    "cell":               lambda s: "ufc::%s" % s,
    "J":                  lambda i, j, m, n: "J[%d]" % _flatten(i, j, m, n),
    "inv(J)":             lambda i, j, m, n: "K[%d]" % _flatten(i, j, m, n),
    "det(J)":             lambda r=None: "detJ%s" % _choose_map(r),
    "cell volume":        lambda r=None: "volume%s" % _choose_map(r),
    "circumradius":       lambda r=None: "circumradius%s" % _choose_map(r),
    "facet area":         "facet_area",
    "min facet edge length": lambda r: "min_facet_edge_length",
    "max facet edge length": lambda r: "max_facet_edge_length",
    "scale factor":       "det",
    "transform":          lambda t, i, j, m, n, r: _transform(t, i, j, m, n, r),
    "normal component":   lambda r, j: "n%s%s" % (_choose_map(r), j),
    "x coordinate":       "X",
    "y coordinate":       "Y",
    "z coordinate":       "Z",
    "ip coordinates":     lambda i, j: "X%d[%d]" % (i, j),
    "affine map table":   lambda i, j: "FEA%d_f%d" % (i, j),
    "vertex_coordinates": lambda r=None: "vertex_coordinates%s" % _choose_map(r)
})

# UFC function arguments and class members (names)
def _pyop2_element_tensor(entries):
    A = "A"
    for i in entries:
        A += ("[%s]" % i)
    return A

def _ufc_element_tensor(i):
    return "A[%s]" % i

def _ufc_coefficient(count, index):
    return format["component"]("w", [count, index])

def _pyop2_coefficient(count, indices):
    if not isinstance(indices, list):
        indices = [indices, '0']
    return format["component"]("w%s" % count, indices)

format.update({
    "element tensor":             { "ufc"  : _ufc_element_tensor,
                                    "pyop2": _pyop2_element_tensor },
    "element tensor term":        lambda i, j: "A%d[%s]" % (j, i),
    "coefficient":                { "ufc": _ufc_coefficient, "pyop2": _pyop2_coefficient },
    "argument basis num":         "i",
    "argument derivative order":  "n",
    "argument values":            "values",
    "argument coordinates":       "dof_coordinates",
    "facet":                      lambda r: "facet%s" % _choose_map(r),
    "vertex":                     "vertex",
    "argument axis":              "i",
    "argument dimension":         "d",
    "argument entity":            "i",
    "member global dimension":    "_global_dimension",
    "argument dofs":              "dofs",
    "argument dof num":           "i",
    "argument dof values":        "dof_values",
    "argument vertex values":     "vertex_values",
    "argument sub":               "i" # sub domain, sub element
})

# Formatting used in evaluatedof.
format.update({
    "dof vals":                 "vals",
    "dof result":               "result",
    "dof X":                    lambda i: "X_%d" % i,
    "dof D":                    lambda i: "D_%d" % i,
    "dof W":                    lambda i: "W_%d" % i,
    "dof copy":                 lambda i: "copy_%d" % i,
    "dof physical coordinates": "y"
})


# Formatting used in evaluate_basis, evaluate_basis_derivatives and quadrature
# code generators.
format.update({
    # evaluate_basis and evaluate_basis_derivatives
    "tmp value":                  lambda i: "tmp%d" % i,
    "tmp ref value":              lambda i: "tmp_ref%d" % i,
    "local dof":                  "dof",
    "basisvalues":                "basisvalues",
    "coefficients":               lambda i: "coefficients%d" %(i),
    "num derivatives":            lambda t_or_g :"num_derivatives" + t_or_g,
    "derivative combinations":    lambda t_or_g :"combinations" + t_or_g,
    "transform matrix":           "transform",
    "transform Jinv":             "Jinv",
    "dmats":                      lambda i: "dmats%s" %(i),
    "dmats old":                  "dmats_old",
    "reference derivatives":      "derivatives",
    "dof values":                 "dof_values",
    "dof map if":                 lambda i,j: "%d <= %s && %s <= %d"\
                                  % (i, format["argument basis num"], format["argument basis num"], j),
    "dereference pointer":        lambda n: "*%s" % n,
    "reference variable":         lambda n: "&%s" % n,
    "call basis":                 lambda i, s: "_evaluate_basis(%s, %s, x, vertex_coordinates, cell_orientation);" % (i, s),
    "call basis_all":             "_evaluate_basis_all(values, x, vertex_coordinates, cell_orientation);",
    "call basis_derivatives":     lambda i, s: "_evaluate_basis_derivatives(%s, n, %s, x, vertex_coordinates, cell_orientation);" % (i, s),
    "call basis_derivatives_all": lambda i, s: "_evaluate_basis_derivatives_all(n, %s, x, vertex_coordinates, cell_orientation);" % s,

    # quadrature code generators
    "integration points": "ip",
    "first free index":   "j",
    "second free index":  "k",
    "geometry constant":  lambda i: "G[%d]" % i,
    "ip constant":        lambda i: "I[%d]" % i,
    "basis constant":     lambda i: "B[%d]" % i,
    "conditional":        lambda i: "C[%d]" % i,
    "evaluate conditional":lambda i,j,k: "(%s) ? %s : %s" % (i,j,k),
#    "geometry constant":  lambda i: "G%d" % i,
#    "ip constant":        lambda i: "I%d" % i,
#    "basis constant":     lambda i: "B%d" % i,
    "function value":     lambda i: "F%d" % i,
    "nonzero columns":    lambda i: "nzc%d" % i,
    "weight":             lambda i: "W" if i is None else "W%d" % (i),
    "psi name":           lambda c, et, e, co, d, a: _generate_psi_name(c, et, e, co, d, a),
    # both
    "free indices":       ["r","s","t","u"],
    "matrix index":       lambda i, j, range_j: _matrix_index(i, str(j), str(range_j)),
    "quadrature point":   lambda i, gdim: "quadrature_points + %s*%d" % (i, gdim)
})

# Misc
format.update({
    "bool":             lambda v: {True: "true", False: "false"}[v],
    "str":              lambda v: "%s" % v,
    "int":              lambda v: "%d" % v,
    "list separator":   ", ",
    "block separator":  ",\n",
    "new line":         "\\\n",
    "tabulate tensor":  lambda m: _tabulate_tensor(m),
})

# Code snippets
from codesnippets import *

format.update({
    "compute_jacobian":         lambda cell, r=None: \
                                compute_jacobian[cell] % {"restriction": _choose_map(r)},
    "compute_jacobian_interior":     lambda cell, r=None: \
                                compute_jacobian_interior[cell] % {"restriction": _choose_map(r)},
    "compute_jacobian_inverse": lambda cell, r=None: \
                                compute_jacobian_inverse[cell] % {"restriction": _choose_map(r)},
    "orientation":              {"ufc": lambda tdim, gdim, r=None: ufc_orientation_snippet % {"restriction": _choose_map(r)} if tdim != gdim else "",
                                 "pyop2": lambda tdim, gdim, r=None: pyop2_orientation_snippet % {"restriction": _choose_map(r)} if tdim != gdim else ""},
    "facet determinant":        lambda cell, p_format, integral_type, r=None: _generate_facet_determinant(cell, p_format, integral_type, r),
    "fiat coordinate map":      lambda cell, gdim: fiat_coordinate_map[cell][gdim],
    "generate normal":          lambda cell, p_format, integral_type: _generate_normal(cell, p_format, integral_type),
    "generate cell volume":     {"ufc": lambda tdim, gdim, i, r=None: _generate_cell_volume(tdim, gdim, i, ufc_cell_volume, r),
                                 "pyop2": lambda tdim, gdim, i, r=None: _generate_cell_volume(tdim, gdim, i, pyop2_cell_volume, r)},
    "generate circumradius":    {"ufc": lambda tdim, gdim, i, r=None: _generate_circumradius(tdim, gdim, i, ufc_circumradius, r),
                                 "pyop2": lambda tdim, gdim, i, r=None: _generate_circumradius(tdim, gdim, i, pyop2_circumradius, r)},
    "generate facet area":      lambda tdim, gdim: facet_area[tdim][gdim],
    "generate min facet edge length": lambda tdim, gdim, r=None: min_facet_edge_length[tdim][gdim] % {"restriction": _choose_map(r)},
    "generate max facet edge length": lambda tdim, gdim, r=None: max_facet_edge_length[tdim][gdim] % {"restriction": _choose_map(r)},
    "generate ip coordinates":  lambda g, num_ip, name, ip, r=None: (ip_coordinates[g][0], ip_coordinates[g][1] % \
                                {"restriction": _choose_map(r), "ip": ip, "name": name, "num_ip": num_ip}),
    "scale factor snippet":     {"ufc": ufc_scale_factor,
                                 "pyop2": pyop2_scale_factor},
    "map onto physical":        map_onto_physical,
    "evaluate basis snippet":   eval_basis,
    "combinations":             combinations_snippet,
    "transform snippet":        transform_snippet,
    "evaluate function":        evaluate_f,
    "ufc comment":              comment_ufc,
    "dolfin comment":           comment_dolfin,
    "pyop2 comment":            comment_pyop2,
    "header_h":                 {"ufc": header_h,
                                 "pyop2": ""},
    "header_c":                 header_c,
    "footer":                   {"ufc": footer,
                                 "pyop2": ""},
    "eval_basis_decl":          eval_basis_decl,
    "eval_basis":               eval_basis,
    "eval_basis_copy":          eval_basis_copy,
    "eval_derivs_decl":         eval_derivs_decl,
    "eval_derivs":              eval_derivs,
    "eval_derivs_copy":         eval_derivs_copy})

# Class names
format.update({
    "classname finite_element": lambda prefix, i:\
               "%s_finite_element_%d" % (prefix.lower(), i),

    "classname dofmap":  lambda prefix, i: "%s_dofmap_%d" % (prefix.lower(), i),

    "classname cell_integral":  lambda prefix, form_id, sub_domain:\
               "%s_cell_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname exterior_facet_integral":  lambda prefix, form_id, sub_domain:\
              "%s_exterior_facet_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname exterior_facet_bottom_integral":  lambda prefix, form_id, sub_domain:\
              "%s_exterior_facet_bottom_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname exterior_facet_top_integral":  lambda prefix, form_id, sub_domain:\
              "%s_exterior_facet_top_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname exterior_facet_vert_integral":  lambda prefix, form_id, sub_domain:\
              "%s_exterior_facet_vert_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname interior_facet_integral":  lambda prefix, form_id, sub_domain:\
              "%s_interior_facet_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname interior_facet_horiz_integral":  lambda prefix, form_id, sub_domain:\
              "%s_interior_facet_horiz_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname interior_facet_vert_integral":  lambda prefix, form_id, sub_domain:\
              "%s_interior_facet_vert_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname point_integral":  lambda prefix, form_id, sub_domain:\
              "%s_point_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname custom_integral":  lambda prefix, form_id, sub_domain:\
              "%s_custom_integral_%d_%s" % (prefix.lower(), form_id, sub_domain),

    "classname form": lambda prefix, i: "%s_form_%d" % (prefix.lower(), i)
})

# Helper functions for formatting

def _declaration(type, name, value=None):
    if value is None:
        return "%s %s;" % (type, name);
    return "%s %s = %s;" % (type, name, str(value));

def _component(var, k):
    if not isinstance(k, (list, tuple)):
        k = [k]
    return "%s" % var + "".join("[%s]" % str(i) for i in k)

def _delete_array(name, size=None):
    if size is None:
        return "delete [] %s;" % name
    f_r = format["free indices"][0]
    code = format["generate loop"](["delete [] %s;" % format["component"](name, f_r)], [(f_r, 0, size)])
    code.append("delete [] %s;" % name)
    return "\n".join(code)

def _multiply(factors):
    """
    Generate string multiplying a list of numbers or strings.  If a
    factor is zero, the whole product is zero. Any factors equal to
    one are ignored.
    """

    # FIXME: This could probably be way more robust and elegant.

    cpp_str = format["str"]
    non_zero_factors = []
    for f in factors:

        # Round-off if f is smaller than epsilon
        if isinstance(f, (int, float)):
            if abs(f) < format["epsilon"]:
                return cpp_str(0)
            if abs(f - 1.0) < format["epsilon"]:
                continue

        # Convert to string
        f = cpp_str(f)

        # Return zero if any factor is zero
        if f == "0":
            return cpp_str(0)

        # If f is 1, don't add it to list of factors
        if f == "1":
            continue

        # If sum-like, parentheseze factor
        if "+" in f or "-" in f:
            f = "(%s)" % f

        non_zero_factors += [f]

    if len(non_zero_factors) == 0:
        return cpp_str(1.0)

    return "*".join(non_zero_factors)

def _add(terms):
    "Generate string summing a list of strings."

    # FIXME: Subtract absolute value of negative numbers
    result = " + ".join([str(t) for t in terms if (str(t) != "0")])
    if result == "":
        return format["str"](0)
    return result

def _power(base, exponent):
    "Generate code for base^exponent."
    if exponent >= 0:
        return _multiply(exponent*(base,))
    else:
        return "1.0 / (%s)" % _power(base, -exponent)

def _inner_product(v, w):
    "Generate string for v[0]*w[0] + ... + v[n]*w[n]."

    # Check that v and w have same length
    assert(len(v) == len(w)), "Sizes differ in inner-product!"

    # Special case, zero terms
    if len(v) == 0: return format["float"](0)

    # Straightforward handling when we only have strings
    if isinstance(v[0], str):
        return _add([_multiply([v[i], w[i]]) for i in range(len(v))])

    # Fancy handling of negative numbers etc
    result = None
    eps = format["epsilon"]
    add = format["add"]
    sub = format["sub"]
    neg = format["neg"]
    mul = format["mul"]
    fl  = format["float"]
    for (c, x) in zip(v, w):
        if result:
            if abs(c - 1.0) < eps:
                result = add([result, x])
            elif abs(c + 1.0) < eps:
                result = sub([result, x])
            elif c > eps:
                result = add([result, mul([fl(c), x])])
            elif c < -eps:
                result = sub([result, mul([fl(-c), x])])
        else:
            if abs(c - 1.0) < eps:
                result = x
            elif abs(c + 1.0) < eps:
                result = neg(x)
            elif c > eps:
                result = mul([fl(c), x])
            elif c < -eps:
                result = neg(mul([fl(-c), x]))

    return result

def _transform(type, i, j, m, n, r):
    map_name = {"J": "J", "JINV": "K"}[type] + _choose_map(r)
    return (map_name + "[%d]") % _flatten(i, j, m, n)

# FIXME: Input to _generate_switch should be a list of tuples (i, case)
def _generate_switch(variable, cases, default=None, numbers=None):
    "Generate switch statement from given variable and cases"

    # Special case: no cases and no default
    if len(cases) == 0 and default is None:
        return format["do nothing"]
    elif len(cases) == 0:
        return default

    # Special case: one case and no default
    if len(cases) == 1 and default is None:
        return cases[0]

    # Create numbers for switch
    if numbers is None:
        numbers = range(len(cases))

    # Create switch
    code = "switch (%s)\n{\n" % variable
    for (i, case) in enumerate(cases):
        code += "case %d:\n  {\n  %s\n    break;\n  }\n" % (numbers[i], indent(case, 2))
    code += "}\n"

    # Default value
    if default:
        code += "\n" + default

    return code

def _tabulate_tensor(vals):
    "Tabulate a multidimensional tensor. (Replace tabulate_matrix and tabulate_vector)."

    # Prefetch formats to speed up code generation
    f_block     = format["block"]
    f_list_sep  = format["list separator"]
    f_block_sep = format["block separator"]
    # FIXME: KBO: Change this to "float" once issue in set_float_formatting is fixed.
    f_float     = format["floating point"]
    f_epsilon   = format["epsilon"]

    # Create numpy array and get shape.
    tensor = numpy.array(vals)
    shape = numpy.shape(tensor)
    if len(shape) == 1:
        # Create zeros if value is smaller than tolerance.
        values = []
        for v in tensor:
            if abs(v) < f_epsilon:
                values.append(f_float(0.0))
            else:
                values.append(f_float(v))
        # Format values.
        return f_block(f_list_sep.join(values))
    elif len(shape) > 1:
        return f_block(f_block_sep.join([_tabulate_tensor(tensor[i]) for i in range(shape[0])]))
    else:
        error("Not an N-dimensional array:\n%s" % tensor)

def _generate_loop(lines, loop_vars, _indent):
    "This function generates a loop over a vector or matrix."

    # Prefetch formats to speed up code generation.
    f_loop     = format["loop"]
    f_begin    = format["block begin"]
    f_end      = format["block end"]
    f_comment  = format["comment"]

    if not loop_vars:
        return lines

    code = []
    for ls in loop_vars:
        # Get index and lower and upper bounds.
        index, lower, upper = ls
        # Loop index.
        code.append(indent(f_loop(index, lower, upper), _indent))
        code.append(indent(f_begin, _indent))

        # Increase indentation.
        _indent += 2

        # If this is the last loop, write values.
        if index == loop_vars[-1][0]:
            for l in lines:
                code.append(indent(l, _indent))

    # Decrease indentation and write end blocks.
    indices = [var[0] for var in loop_vars]
    indices.reverse()
    for index in indices:
        _indent -= 2
        code.append(indent(f_end + " " + f_comment("end loop over '%s'" % index), _indent))

    return code

def _matrix_index(i, j, range_j):
    "Map the indices in a matrix to an index in an array i.e., m[i][j] -> a[i*range(j)+j]"
    if i == 0:
        access = j
    elif i == 1:
        access = format["add"]([range_j, j])
    else:
        irj = format["mul"]([format["str"](i), range_j])
        access = format["add"]([irj, j])
    return access

def _generate_psi_name(counter, entity_type, entity, component, derivatives, avg):
    """Generate a name for the psi table of the form:
    FE#_f#_v#_C#_D###_A#, where '#' will be an integer value.

    FE  - is a simple counter to distinguish the various bases, it will be
          assigned in an arbitrary fashion.

    f   - denotes facets if applicable, range(element.num_facets()).

    fh, fv - denotes horiz_facets and vert_facets

    v   - denotes vertices if applicable, range(num_vertices).

    C   - is the component number if any (flattened in the case of tensor valued functions)

    D   - is the number of derivatives in each spatial direction if any.
          If the element is defined in 3D, then D012 means d^3(*)/dydz^2.

    A   - denotes averaged over cell (AC) or facet (AF)
    """

    name = "FE%d" % counter

    if entity_type == "facet":
        if entity is None:
            name += "_f0"
        else:
            name += "_f%d" % entity
    elif entity_type == "horiz_facet":
        name += "_fh%d" % entity
    elif entity_type == "vert_facet":
        name += "_fv%d" % entity
    elif entity_type == "vertex":
        name += "_v%d" % entity

    if component != () and component != []:
        name += "_C%d" % component

    if any(derivatives):
        name += "_D" + "".join(map(str,derivatives))

    if avg == "cell":
        name += "_AC"
    elif avg == "facet":
        name += "_AF"

    return name

def _generate_facet_determinant(cell, p_format, integral_type, r):
    "Generate code for computing facet determinant"

    tdim = cell.topological_dimension()
    gdim = cell.geometric_dimension()
    if p_format == "ufc":
        code = ufc_facet_determinant[tdim][gdim] % {"restriction": _choose_map(r)}
    elif p_format == "pyop2":
        if integral_type == "exterior_facet":
            code = pyop2_facet_determinant[tdim][gdim] % {"restriction": _choose_map(r)}
        elif integral_type == "interior_facet":
            code = pyop2_facet_determinant_interior[tdim][gdim] % {"restriction": _choose_map(r)}
        elif integral_type == "exterior_facet_bottom":
            code = bottom_facet_determinant[cell] % {"restriction": _choose_map(r)}
        elif integral_type == "exterior_facet_top":
            code = top_facet_determinant[cell] % {"restriction": _choose_map(r)}
        elif integral_type == "interior_facet_horiz":
            code = top_facet_determinant_interior[cell] % {"restriction": _choose_map(r)}
        elif integral_type == "exterior_facet_vert":
            code = vert_facet_determinant[cell] % {"restriction": _choose_map(r)}
        elif integral_type == "interior_facet_vert":
            code = vert_facet_determinant_interior[cell] % {"restriction": _choose_map(r)}
        else:
            raise RuntimeError("Invalid integral_type")
    else:
        raise RuntimeError("Invalid p_format")

    return code

def _generate_normal(cell, p_format, integral_type, reference_normal=False):
    "Generate code for computing normal"

    if p_format == "ufc":
        normal_direction = ufc_normal_direction
        facet_normal = ufc_facet_normal
    elif p_format == "pyop2":
        if integral_type == "exterior_facet":
            normal_direction = pyop2_normal_direction
            facet_normal = pyop2_facet_normal
        elif integral_type == "interior_facet":
            normal_direction = pyop2_normal_direction_interior
            facet_normal = pyop2_facet_normal_interior
        elif integral_type == "exterior_facet_bottom":
            normal_direction = bottom_normal_direction
            facet_normal = bottom_facet_normal
        elif integral_type == "exterior_facet_top":
            normal_direction = top_normal_direction
            facet_normal = top_facet_normal
        elif integral_type == "interior_facet_horiz":
            normal_direction = top_normal_direction_interior
            facet_normal = top_facet_normal_interior
        elif integral_type == "exterior_facet_vert":
            normal_direction = vert_normal_direction
            facet_normal = vert_facet_normal
        elif integral_type == "interior_facet_vert":
            normal_direction = vert_normal_direction_interior
            facet_normal = vert_facet_normal_interior
        else:
            raise RuntimeError("Invalid integral_type")
    else:
        raise RuntimeError("Invalid p_format")

    if integral_type in ("exterior_facet", "interior_facet"):
        # Choose snippets
        tdim = cell.topological_dimension()
        gdim = cell.geometric_dimension()
        direction = normal_direction[tdim][gdim]

        assert (facet_normal[tdim].has_key(gdim)),\
            "Facet normal not yet implemented for this tdim/gdim combo"
        normal = facet_normal[tdim][gdim]
    else:
        # Choose snippets
        direction = normal_direction[cell]

        assert (facet_normal.has_key(cell)),\
            "Facet normal not yet implemented for this cell"
        normal = facet_normal[cell]
    
    # Choose restrictions
    if integral_type in ("exterior_facet", "exterior_facet_vert"):
        code = direction % {"restriction": "", "facet" : "facet"}
        code += normal % {"direction" : "", "restriction": ""}
    elif integral_type in ("exterior_facet_bottom", "exterior_facet_top"):
        code = direction % {"restriction": ""}
        code += normal % {"direction" : "", "restriction": ""}
    elif integral_type in ("interior_facet", "interior_facet_vert"):
        code = direction % {"restriction": _choose_map("+"), "facet": "facet_0"}
        code += normal % {"direction" : "", "restriction": _choose_map("+")}
        code += normal % {"direction" : "!", "restriction": _choose_map("-")}
    elif integral_type == "interior_facet_horiz":
        code = direction % {"restriction": _choose_map("+")}
        code += normal % {"direction" : "", "restriction": _choose_map("+")}
        code += normal % {"direction" : "!", "restriction": _choose_map("-")}
    else:
        error("Unsupported integral_type: %s" % str(integral_type))
    return code

def _generate_cell_volume(tdim, gdim, integral_type, cell_volume, r=None):
    "Generate code for computing cell volume."

    # Choose snippets
    volume = cell_volume[tdim][gdim]

    # Choose restrictions
    if integral_type in ("cell", "exterior_facet", "exterior_facet_bottom",
                         "exterior_facet_top", "exterior_facet_vert"):
        code = volume % {"restriction": ""}
    elif integral_type in ("interior_facet", "interior_facet_horiz",
                           "interior_facet_vert"):
        code = volume % {"restriction": _choose_map("+")}
        code += volume % {"restriction": _choose_map("-")}
    elif domain_type == "custom":
        code = volume % {"restriction": _choose_map(r)}
    else:
        error("Unsupported integral_type: %s" % str(integral_type))
    return code

def _generate_circumradius(tdim, gdim, integral_type, circumradius, r=None):
    "Generate code for computing a cell's circumradius."

    # Choose snippets
    radius = circumradius[tdim][gdim]

    # Choose restrictions
    if integral_type in ("cell", "exterior_facet", "point"):
        code = radius % {"restriction": ""}
    elif integral_type == "interior_facet":
        code = radius % {"restriction": _choose_map("+")}
        code += radius % {"restriction": _choose_map("-")}
    elif domain_type == "custom":
        code = radius % {"restriction": _choose_map(r)}
    else:
        error("Unsupported integral_type: %s" % str(integral_type))
    return code

def _flatten(i, j, m, n):
    return i*n + j

# Other functions

def indent(block, num_spaces):
    "Indent each row of the given string block with n spaces."
    indentation = " " * num_spaces
    return indentation + ("\n" + indentation).join(block.split("\n"))

def count_ops(code):
    "Count the number of operations in code (multiply-add pairs)."
    num_add = code.count(" + ") + code.count(" - ")
    num_multiply = code.count("*") + code.count("/")
    return (num_add + num_multiply) / 2

def set_float_formatting(precision):
    "Set floating point formatting based on precision."

    # Options for float formatting
    #f1     = "%%.%df" % precision
    #f2     = "%%.%de" % precision
    f1     = "%%.%dg" % precision
    f2     = "%%.%dg" % precision
    f_int  = "%%.%df" % 1

    eps = eval("1e-%s" % precision)

    # Regular float formatting
    def floating_point_regular(v):
        if abs(v - round(v, 1)) < eps:
            return f_int % v
        elif abs(v) < 100.0:
            return f1 % v
        else:
            return f2 % v

    # Special float formatting on Windows (remove extra leading zero)
    def floating_point_windows(v):
        return floating_point_regular(v).replace("e-0", "e-").replace("e+0", "e+")

    # Set float formatting
    if platform.system() == "Windows":
        format["float"] = floating_point_windows
    else:
        format["float"] = floating_point_regular

    # FIXME: KBO: Remove once we agree on the format of 'f1'
    format["floating point"] = format["float"]

    # Set machine precision
    format["epsilon"] = 10.0*eval("1e-%s" % precision)

def set_exception_handling(convert_exceptions_to_warnings):
    "Set handling of exceptions."
    if convert_exceptions_to_warnings:
        format["exception"] = format["warning"]

# Declarations to examine
types = [["double"],
         ["const", "double"],
         ["const", "double", "*", "const", "*"],
         ["int"],
         ["const", "int"],
         ["unsigned", "int"],
         ["bool"],
         ["const", "bool"],
         ["static", "unsigned", "int"],
         ["const", "unsigned", "int"]]

# Special characters and delimiters
special_characters = ["+", "-", "*", "/", "=", ".", " ", ";", "(", ")", "\\", "{", "}", "[","]", "!"]

def remove_unused(code, used_set=set()):
    """
    Remove unused variables from a given C++ code. This is useful when
    generating code that will be compiled with gcc and parameters -Wall
    -Werror, in which case gcc returns an error when seeing a variable
    declaration for a variable that is never used.

    Optionally, a set may be specified to indicate a set of variables
    names that are known to be used a priori.
    """

    # Dictionary of (declaration_line, used_lines) for variables
    variables = {}

    # List of variable names (so we can search them in order)
    variable_names = []

    lines = code.split("\n")
    for (line_number, line) in enumerate(lines):
        # Exclude commented lines.
        if line[:2] == "//" or line[:3] == "///":
            continue

        # Split words
        words = [word for word in line.split(" ") if not word == ""]

        # Remember line where variable is declared
        for type in [type for type in types if " ".join(type) in " ".join(words)]: # Fewer matches than line below.
        # for type in [type for type in types if len(words) > len(type)]:
            variable_type = words[0:len(type)]
            variable_name = words[len(type)]

            # Skip special characters
            if variable_name in special_characters:
                continue

            # Test if any of the special characters are present in the variable name
            # If this is the case, then remove these by assuming that the 'real' name
            # is the first entry in the return list. This is implemented to prevent
            # removal of e.g. 'double array[6]' if it is later used in a loop as 'array[i]'
            if variable_type == type:

                # Create correct variable name (e.g. y instead of
                # y[2]) for variables with separators
                seps_present = [sep for sep in special_characters if sep in variable_name]
                if seps_present:
                    variable_name = [variable_name.split(sep)[0] for sep in seps_present]
                    variable_name.sort()
                    variable_name = variable_name[0]

                variables[variable_name] = (line_number, [])
                if not variable_name in variable_names:
                    variable_names += [variable_name]

        # Mark line for used variables
        for variable_name in variables:
            (declaration_line, used_lines) = variables[variable_name]
            if _variable_in_line(variable_name, line) and line_number > declaration_line:
                variables[variable_name] = (declaration_line, used_lines + [line_number])

    # Reverse the order of the variable names to catch variables used
    # only by variables that are removed
    variable_names.reverse()

    # Remove declarations that are not used
    removed_lines = []
    for variable_name in variable_names:
        (declaration_line, used_lines) = variables[variable_name]
        for line in removed_lines:
            if line in used_lines:
                used_lines.remove(line)
        if not used_lines and not variable_name in used_set:
            debug("Removing unused variable: %s" % variable_name)
            lines[declaration_line] = None # KBO: Need to completely remove line for evaluate_basis* to work
            # lines[declaration_line] = "// " + lines[declaration_line]
            removed_lines += [declaration_line]
    return "\n".join([line for line in lines if not line is None])

def _variable_in_line(variable_name, line):
    "Check if variable name is used in line"
    if not variable_name in line:
        return False
    for character in special_characters:
        line = line.replace(character, "\\" + character)
    delimiter = "[" + ",".join(["\\" + c for c in special_characters]) + "]"
    return not re.search(delimiter + variable_name + delimiter, line) == None
