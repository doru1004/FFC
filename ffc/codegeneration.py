"""
Compiler stage 4: Code generation
---------------------------------

This module implements the generation of C++ code for the body of each
UFC function from an (optimized) intermediate representation (OIR).
"""

# Copyright (C) 2009-2015 Anders Logg
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
# Modified by Mehdi Nikbakht 2010
# Modified by Martin Alnaes, 2013-2015

# FFC modules
import ffc
import numpy as np
from ffc.log import info, begin, end, debug_code
from ffc.cpp import format, indent
from ffc.cpp import set_exception_handling

# FFC code generation modules
from ffc.evaluatebasis import _evaluate_basis, _evaluate_basis_all
from ffc.evaluatebasisderivatives import _evaluate_basis_derivatives
from ffc.evaluatebasisderivatives import _evaluate_basis_derivatives_all
from ffc.evaluatedof import evaluate_dof_and_dofs, affine_weights
from ffc.interpolatevertexvalues import interpolate_vertex_values

from ffc.representation import pick_representation, ufc_integral_types

from sympy import symbols
from sympy.printing import StrPrinter

# Errors issued for non-implemented functions
def _not_implemented(function_name, return_null=False):
    body = format["exception"]("%s not yet implemented." % function_name)
    if return_null:
        body += "\n" + format["return"](0)
    return body


def generate_code(ir, prefix, parameters):
    "Generate code from intermediate representation."

    begin("Compiler stage 4: Generating code")

    # FIXME: Document option -fconvert_exceptions_to_warnings
    # FIXME: Remove option epsilon and just rely on precision?

    # Set code generation parameters
#    set_float_formatting(int(parameters["precision"]))
    set_exception_handling(parameters["convert_exceptions_to_warnings"])

    # Extract representations
    ir_elements, ir_dofmaps, ir_integrals, ir_forms = ir

    # Generate code for elements
    info("Generating code for %d element(s)" % len(ir_elements))
    code_elements = [_generate_element_code(ir, prefix, parameters)
                     for ir in ir_elements]

    # Generate code for dofmaps
    info("Generating code for %d dofmap(s)" % len(ir_dofmaps))
    code_dofmaps = [_generate_dofmap_code(ir, prefix, parameters)
                    for ir in ir_dofmaps]

    # Generate code for integrals
    info("Generating code for integrals")
    code_integrals = [_generate_integral_code(ir, prefix, parameters)
                      for ir in ir_integrals]

    # Generate code for forms
    info("Generating code for forms")
    code_forms = [_generate_form_code(ir, prefix, parameters)
                  for ir in ir_forms]

    end()

    return code_elements, code_dofmaps, code_integrals, code_forms


def _generate_element_code(ir, prefix, parameters):
    "Generate code for finite element from intermediate representation."

    # Skip code generation if ir is None
    if ir is None:
        return None

    # Prefetch formatting to speedup code generation
    ret = format["return"]
    classname = format["classname finite_element"]
    do_nothing = format["do nothing"]
    create = format["create foo"]

    # Codes generated together
    (evaluate_dof_code, evaluate_dofs_code) \
        = evaluate_dof_and_dofs(ir["evaluate_dof"])

    # Generate code
    code = {}
    code["classname"] = classname(prefix, ir["id"])
    code["members"] = ""
    code["constructor"] = do_nothing
    code["constructor_arguments"] = ""
    code["initializer_list"] = ""
    code["destructor"] = do_nothing
    code["signature"] = ret('"%s"' % ir["signature"])
    code["cell_shape"] = ret(format["cell"](ir["cell"].cellname()))
    code["topological_dimension"] = ret(ir["cell"].topological_dimension())
    code["geometric_dimension"] = ret(ir["cell"].geometric_dimension())
    code["space_dimension"] = ret(ir["space_dimension"])
    code["value_rank"] = ret(ir["value_rank"])
    code["value_dimension"] = _value_dimension(ir["value_dimension"])
    code["evaluate_basis"] = _evaluate_basis(ir["evaluate_basis"])
    code["evaluate_basis_all"] = _evaluate_basis_all(ir["evaluate_basis"])
    code["evaluate_basis_derivatives"] \
        = _evaluate_basis_derivatives(ir["evaluate_basis"])
    code["evaluate_basis_derivatives_all"] \
        = _evaluate_basis_derivatives_all(ir["evaluate_basis"])
    code["evaluate_dof"] = evaluate_dof_code
    code["evaluate_dofs"] = evaluate_dofs_code
    code["interpolate_vertex_values"] \
        = interpolate_vertex_values(ir["interpolate_vertex_values"])
    code["map_from_reference_cell"] \
        = _not_implemented("map_from_reference_cell")
    code["map_to_reference_cell"] = _not_implemented("map_to_reference_cell")
    code["num_sub_elements"] = ret(ir["num_sub_elements"])
    code["create_sub_element"] = _create_sub_element(prefix, ir)
    code["create"] = ret(create(code["classname"]))

    # Postprocess code
    _postprocess_code(code, parameters)

    return code


def _inside_check(ufl_cell, fiat_cell):
    dim = ufl_cell.topological_dimension()
    point = tuple(symbols("X[%d]" % i) for i in xrange(dim))

    return " && ".join("(%s)" % arg for arg in fiat_cell.contains_point(point, epsilon=1e-14).args)


def _init_X(fiat_element):
    fiat_cell = fiat_element.get_reference_element()
    vertices = np.array(fiat_cell.get_vertices())
    X = np.average(vertices, axis=0)
    return "\n".join("\tX[%d] = %.12f;" % (i, v) for i, v in enumerate(X))  # TODO: FP formatting


def _calculate_basisvalues(ufl_cell, fiat_element):
    f_table         = format["static const float declaration"]["ufc"]
    f_component     = format["component"]
    f_decl          = format["declaration"]
    f_tensor        = format["tabulate tensor"]
    f_new_line      = format["new line"]

    code = _read_coordinates(ufl_cell.topological_dimension())
    tdim = ufl_cell.topological_dimension()
    refs, scode = ffc.tabulate(fiat_element, 0, ("X", "Y", "Z")[:tdim])
    for si in scode:
        code += ["\tdouble %s = %s;" % (si[0], CPrinter(dict(order=None)).doprint(si[1]))]

    code += ["", "\t// Values of basis functions"]
    code += [f_decl("double", f_component("phi", refs[(0,) * tdim].shape),
                    f_new_line + f_tensor(refs[(0,) * tdim]))]

    shape = refs[(0,) * tdim].shape
    if len(shape) <= 1:
        vdim = 1
    elif len(shape) == 2:
        vdim = shape[1]
    else:
        raise NotImplementedError("I am surprised.")
    return "\n".join(code), vdim


def _to_reference_coordinates(ufl_cell, fiat_element, needs_orientation):
    f_table         = format["static const float declaration"]["ufc"]
    f_component     = format["component"]
    f_decl          = format["declaration"]
    f_tensor        = format["tabulate tensor"]
    f_new_line      = format["new line"]

    # Get the element cell name and geometric dimension.
    cell = ufl_cell
    gdim = cell.geometric_dimension()
    tdim = cell.topological_dimension()

    code = []
    refs, scode = ffc.tabulate(fiat_element,
                               1,
                               ("X[0]", "X[1]", "X[2]")[:tdim],
                               prefix="t")
    for si in scode:
        code += ["\tdouble %s = %s;" % (si[0], CPrinter(dict(order=None)).doprint(si[1]))]

    from sympy import Symbol as Variable
    C = np.array([[Variable("C[%d][%d]" % (i, j))for j in range(gdim)]
                   for i in range(fiat_element.space_dimension())])

    x_phi = refs[(0,) * tdim]
    x = x_phi.dot(C)
    for i, e in enumerate(x):
        code += ["\tx[%d] = %s;" % (i, e)]

    refs_keys = reversed(sorted(refs.keys()))
    refs_keys = list(refs_keys)[:-1]  # skip all zeros
    x_grad_phi = np.vstack([refs[k] for k in refs_keys])
    J = np.transpose(x_grad_phi.dot(C))
    for i, row in enumerate(J):
        for j, e in enumerate(row):
            code += ["\tJ[%d * %d + %d] = %s;" % (i, tdim, j, e)]

    # Get code snippets for Jacobian, Inverse of Jacobian and mapping of
    # coordinates from physical element to the FIAT reference element.
    code_ = [format["compute_jacobian_inverse"](cell)]
    if needs_orientation:
        code_ += [format["orientation"]["ufc"](tdim, gdim)]
    # FIXME: very ugly hack!
    code_ = "\n".join(code_).split("\n")
    code_ = filter(lambda line: not line.startswith(("double J", "double K", "double detJ")), code_)
    code += code_

    x = np.array(map(lambda i: Variable("x[%d]" % i), range(gdim)))
    x0 = np.array(map(lambda i: Variable("x0[%d]" % i), range(gdim)))
    X = np.array(map(lambda i: Variable("X[%d]" % i), range(tdim)))
    K = np.array(map(lambda i: Variable("K[%d]" % i), range(tdim * gdim)))
    K = K.reshape(tdim, gdim)

    dX = K.dot(x - x0)
    for i, e in enumerate(dX):
        code += ["\tdX[%d] = %s;" % (i, e)]

    return "\n".join(code)


def _read_coordinates(topological_dimension):
    return ["\tdouble %s = reference_coords.X[%d];" % (v, i)
            for (i, v) in zip(range(topological_dimension), ["X", "Y", "Z"])]


def _generate_dofmap_code(ir, prefix, parameters):
    "Generate code for dofmap from intermediate representation."

    # Skip code generation if ir is None
    if ir is None:
        return None

    # Prefetch formatting to speedup code generation
    ret = format["return"]
    classname = format["classname dofmap"]
    declare = format["declaration"]
    assign = format["assign"]
    do_nothing = format["do nothing"]
    switch = format["switch"]
    f_int = format["int"]
    f_d = format["argument dimension"]
    create = format["create foo"]

    # Generate code
    code = {}
    code["classname"] = classname(prefix, ir["id"])
    code["members"] = ""
    code["constructor"] = do_nothing
    code["constructor_arguments"] = ""
    code["initializer_list"] = ""
    code["destructor"] = do_nothing
    code["signature"] = ret('"%s"' % ir["signature"])
    code["needs_mesh_entities"] \
        = _needs_mesh_entities(ir["needs_mesh_entities"])
    code["topological_dimension"] = ret(ir["cell"].topological_dimension())
    code["geometric_dimension"] = ret(ir["cell"].geometric_dimension())
    code["global_dimension"] = _global_dimension(ir["global_dimension"])
    code["num_element_dofs"] = ret(ir["num_element_dofs"])
    code["num_facet_dofs"] = ret(ir["num_facet_dofs"])
    code["num_entity_dofs"] \
        = switch(f_d, [ret(num) for num in ir["num_entity_dofs"]],
                 ret(f_int(0)))
    code["tabulate_dofs"] = _tabulate_dofs(ir["tabulate_dofs"])
    code["tabulate_facet_dofs"] \
        = _tabulate_facet_dofs(ir["tabulate_facet_dofs"])
    code["tabulate_entity_dofs"] \
        = _tabulate_entity_dofs(ir["tabulate_entity_dofs"])
    code["tabulate_coordinates"] \
        = _tabulate_coordinates(ir["tabulate_coordinates"])
    code["num_sub_dofmaps"] = ret(ir["num_sub_dofmaps"])
    code["create_sub_dofmap"] = _create_sub_dofmap(prefix, ir)
    code["create"] = ret(create(code["classname"]))

    # Postprocess code
    _postprocess_code(code, parameters)

    return code

def _generate_integral_code(ir, prefix, parameters):
    "Generate code for integrals from intermediate representation."

    # Skip code generation if ir is None
    if ir is None:
        return None

    # Select representation
    r = pick_representation(ir["representation"])

    # Generate code
    code = r.generate_integral_code(ir, prefix, parameters)

    # Indent code (unused variables should already be removed)
    # FIXME: Remove this quick hack
    if ir["representation"] != "uflacs":
        _indent_code(code)

    return code


def _generate_original_coefficient_position(original_coefficient_positions):
    # TODO: I don't know how to implement this using the format dict,
    # this will do for now:
    initializer_list = ", ".join(str(i)
                                 for i in original_coefficient_positions)
    code = '\n'.join([
        "static const std::vector<std::size_t> position({%s});"
        % initializer_list, "return position[i];",
        ])
    return code


def _generate_form_code(ir, prefix, parameters):
    "Generate code for form from intermediate representation."

    # Skip code generation if ir is None
    if ir is None:
        return None

    # Prefetch formatting to speedup code generation
    ret = format["return"]
    classname = format["classname form"]
    do_nothing = format["do nothing"]

    # Generate code
    code = {}
    code["classname"] = classname(prefix, ir["id"])
    code["members"] = ""

    code["constructor"] = do_nothing
    code["constructor_arguments"] = ""
    code["initializer_list"] = ""
    code["destructor"] = do_nothing

    code["signature"] = ret('"%s"' % ir["signature"])
    code["original_coefficient_position"] = _generate_original_coefficient_position(ir["original_coefficient_positions"])
    code["rank"] = ret(ir["rank"])
    code["num_coefficients"] = ret(ir["num_coefficients"])

    code["create_finite_element"] = _create_finite_element(prefix, ir)
    code["create_dofmap"] = _create_dofmap(prefix, ir)

    for integral_type in ufc_integral_types:
        code["max_%s_subdomain_id" % integral_type] = ret(ir["max_%s_subdomain_id" % integral_type])
        code["has_%s_integrals" % integral_type] = _has_foo_integrals(ir, integral_type)
        code["create_%s_integral" % integral_type] = _create_foo_integral(ir, integral_type, prefix)
        code["create_default_%s_integral" % integral_type] = _create_default_foo_integral(ir, integral_type, prefix)

    # Postprocess code
    _postprocess_code(code, parameters)

    return code

#--- Code generation for non-trivial functions ---

def _value_dimension(ir):
    "Generate code for value_dimension."
    ret = format["return"]
    axis = format["argument axis"]
    f_int = format["int"]

    if ir == ():
        return ret(1)
    return format["switch"](axis, [ret(n) for n in ir], ret(f_int(0)))


def _needs_mesh_entities(ir):
    """
    Generate code for needs_mesh_entities. ir is a list of num dofs
    per entity.
    """
    ret = format["return"]
    boolean = format["bool"]
    dimension = format["argument dimension"]

    return format["switch"](dimension, [ret(boolean(c)) for c in ir],
                            ret(boolean(False)))


def _global_dimension(ir):
    """Generate code for global_dimension. ir[0] is a list of num dofs per
    entity."""

    num_dofs = ir[0]
    component = format["component"]
    entities = format["num entities"]
    dimension = format["inner product"]([format["int"](d) for d in num_dofs],
                                        [component(entities, d)
                                         for d in range(len(num_dofs))])
    # Handle global "elements" if any
    if ir[1]:
        dimension = format["add"]([dimension, format["int"](ir[1])])
        try:
            dimension = format["int"](eval(dimension))
        except:
            pass

    code = format["return"](dimension)

    return code


def _tabulate_facet_dofs(ir):
    "Generate code for tabulate_facet_dofs."

    assign = format["assign"]
    component = format["component"]
    dofs = format["argument dofs"]
    cases = ["\n".join(assign(component(dofs, i), dof)
                       for (i, dof) in enumerate(facet))
             for facet in ir]
    return format["switch"](format["facet"](None), cases)


def _tabulate_dofs(ir):
    "Generate code for tabulate_dofs."

    # Prefetch formats
    add = format["addition"]
    iadd = format["iadd"]
    multiply = format["multiply"]
    assign = format["assign"]
    component = format["component"]
    entity_index = format["entity index"]
    num_entities_format = format["num entities"]
    unsigned_int = format["uint declaration"]
    dofs_variable = format["argument dofs"]

    if ir is None:
        return assign(component(dofs_variable, 0), 0)

    # Extract representation
    (dofs_per_element, num_dofs_per_element, num_entities,
     need_offset, fakes) = ir

    # Declare offset if needed
    code = []
    offset_name = "0"
    if need_offset:
        offset_name = "offset"
        code.append(format["declaration"](unsigned_int, offset_name, 0))

    # Generate code for each element
    i = 0
    for (no, num_dofs) in enumerate(dofs_per_element):

        # Handle fakes (Space of reals)
        if fakes[no] and num_dofs_per_element[no] == 1:
            code.append(assign(component(dofs_variable, i), offset_name))
            if offset_name != "0":
                code.append(iadd(offset_name, 1))
            i += 1
            continue

        # Generate code for each degree of freedom for each dimension
        for (dim, num) in enumerate(num_dofs):

            # Ignore if no dofs for this dimension
            if not num[0]:
                continue

            for (k, dofs) in enumerate(num):
                v = multiply([len(num[k]), component(entity_index, (dim, k))])
                for (j, dof) in enumerate(dofs):
                    value = add([offset_name, v, j])
                    code.append(assign(component(dofs_variable, dof+i), value))

            # Update offset corresponding to mesh entity:
            if need_offset:
                addition = multiply([len(num[0]),
                                     component(num_entities_format, dim)])
                code.append(iadd("offset", addition))

        i += num_dofs_per_element[no]

    return "\n".join(code)


def _tabulate_coordinates(ir):
    "Generate code for tabulate_coordinates."

    # Raise error if tabulate_coordinates is ill-defined
    if not ir:
        msg = "tabulate_coordinates is not defined for this element"
        return format["exception"](msg)

    # Extract formats:
    inner_product = format["inner product"]
    component = format["component"]
    precision = format["float"]
    assign = format["assign"]
    f_x = format["vertex_coordinates"]
    coordinates = format["argument coordinates"]

    # Extract coordinates and cell dimension
    gdim = ir["cell"].geometric_dimension()
    tdim = ir["cell"].topological_dimension()

    # Aid mapping points from reference to physical element
    coefficients = affine_weights(tdim)

    # Generate code for each point and each component
    code = []
    for (i, coordinate) in enumerate(ir["points"]):

        w = coefficients(coordinate)
        for j in range(gdim):
            # Compute physical coordinate
            coords = [component(f_x(), (k*gdim + j,)) for k in range(tdim + 1)]
            value = inner_product(w, coords)

            # Assign coordinate
            code.append(assign(component(coordinates, (i*gdim + j)), value))

    return "\n".join(code)


def _tabulate_entity_dofs(ir):
    "Generate code for tabulate_entity_dofs."

    # Extract variables from ir
    entity_dofs, num_dofs_per_entity = ir

    # Prefetch formats
    assign = format["assign"]
    component = format["component"]
    f_d = format["argument dimension"]
    f_i = format["argument entity"]
    dofs = format["argument dofs"]

    # Add check that dimension and number of mesh entities is valid
    dim = len(num_dofs_per_entity)
    excpt = format["exception"]("%s is larger than dimension (%d)"
                                % (f_d, dim - 1))
    code = [format["if"]("%s > %d" % (f_d, dim-1), excpt)]

    # Generate cases for each dimension:
    all_cases = ["" for d in range(dim)]
    for d in range(dim):

        # Ignore if no entities for this dimension
        if num_dofs_per_entity[d] == 0:
            continue

        # Add check that given entity is valid:
        num_entities = len(entity_dofs[d].keys())
        excpt = format["exception"]("%s is larger than number of entities (%d)"
                                    % (f_i, num_entities - 1))
        check = format["if"]("%s > %d" % (f_i, num_entities - 1), excpt)

        # Generate cases for each mesh entity
        cases = ["\n".join(assign(component(dofs, j), dof)
                           for (j, dof) in enumerate(entity_dofs[d][entity]))
                 for entity in range(num_entities)]

        # Generate inner switch with preceding check
        all_cases[d] = "\n".join([check, format["switch"](f_i, cases)])

    # Generate outer switch
    code.append(format["switch"](f_d, all_cases))

    return "\n".join(code)


#--- Utility functions ---


def _create_foo(prefix, class_name, postfix, arg, numbers=None):
    "Generate code for create_<foo>."
    ret = format["return"]
    create = format["create foo"]

    class_names = ["%s_%s_%d" % (prefix.lower(), class_name, i)
                   for i in postfix]
    cases = [ret(create(name)) for name in class_names]
    default = ret(0)
    return format["switch"](arg, cases, default=default, numbers=numbers)

def _create_finite_element(prefix, ir):
    f_i = format["argument sub"]
    return _create_foo(prefix, "finite_element", ir["create_finite_element"], f_i)

def _create_dofmap(prefix, ir):
    f_i = format["argument sub"]
    return _create_foo(prefix, "dofmap", ir["create_dofmap"], f_i)

def _create_sub_element(prefix, ir):
    f_i = format["argument sub"]
    return _create_foo(prefix, "finite_element", ir["create_sub_element"], f_i)

def _create_sub_dofmap(prefix, ir):
    f_i = format["argument sub"]
    return _create_foo(prefix, "dofmap", ir["create_sub_dofmap"], f_i)

def _create_foo_integral(ir, integral_type, prefix):
    "Generate code for create_<foo>_integral."
    f_i = format["argument subdomain"]
    class_name = integral_type + "_integral_" + str(ir["id"])
    postfix = ir["create_" + integral_type + "_integral"]
    return _create_foo(prefix, class_name, postfix, f_i, numbers=postfix)

def _has_foo_integrals(ir, integral_type):
    ret = format["return"]
    b = format["bool"]
    i = ir["has_%s_integrals" % integral_type]
    return ret(b(i))

def _create_default_foo_integral(ir, integral_type, prefix):
    "Generate code for create_default_<foo>_integral."
    ret = format["return"]
    postfix = ir["create_default_" + integral_type + "_integral"]
    if postfix is None:
        return ret(0)
    else:
        create = format["create foo"]
        class_name = integral_type + "_integral_" + str(ir["id"])
        name = "%s_%s_%s" % (prefix.lower(), class_name, postfix)
        return ret(create(name))


def _postprocess_code(code, parameters):
    "Postprocess generated code."
    _indent_code(code)
    _remove_code(code, parameters)


def _indent_code(code):
    "Indent code that should be indented."
    for key in code:
        if not key in ("classname", "members", "constructor_arguments",
                       "initializer_list", "additional_includes_set",
                       "restrict", "class_type"):
            code[key] = indent(code[key], 4)


def _remove_code(code, parameters):
    "Remove code that should not be generated."
    for key in code:
        flag = "no-" + key
        if flag in parameters and parameters[flag]:
            msg = "// Function %s not generated (compiled with -f%s)" \
                  % (key, flag)
            code[key] = format["exception"](msg)


class CPrinter(StrPrinter):
    def _print_Pow(self, expr, rational=False):
        # WARNING: Code mostly copied from sympy source code!
        from sympy.core import S
        from sympy.printing.precedence import precedence

        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that will
                # match -0.5, which we don't want.
                return "1/sqrt(%s)" % self._print(expr.base)
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '1/%s' % self.parenthesize(expr.base, PREC)

        e = self.parenthesize(expr.exp, PREC)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                e = e[1:-1]

        if e == "2":
            return '{0}*{0}'.format(self.parenthesize(expr.base, PREC))
        elif e == "3":
            return '{0}*{0}*{0}'.format(self.parenthesize(expr.base, PREC))
        else:
            return 'pow(%s,%s)' % (self.parenthesize(expr.base, PREC), e)
