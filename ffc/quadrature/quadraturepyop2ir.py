"Build a PyOP2-conforming Abstract Syntax Tree for quadrature representation."

# Copyright (C) 2009-2013 Kristian B. Oelgaard and Imperial College London
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
# Modified by Anders Logg 2013
# Modified by Martin Alnaes, 2013
# Modified by Fabio Luporini, 2013

# Python modules.
import functools
import numpy

# UFL modules.
from ufl.algorithms.printing import tree_format

## FFC modules.
from ffc.log import info, debug, ffc_assert
from ffc.cpp import format, remove_unused, _choose_map

# PyOP2 IR modules.
from pyop2.coffee import ast_base as pyop2
from pyop2.coffee.ast_base import c_sym

from ffc.representationutils import initialize_integral_code

# Utility and optimisation functions for quadraturegenerator.
from symbolics import generate_aux_constants

def generate_pyop2_ir(ir, prefix, parameters):
    "Generate code for integral from intermediate representation."
    code = initialize_integral_code(ir, prefix, parameters)
    code["additional_includes_set"] = ir["additional_includes_set"]
    code["metadata"] = ""

    body_ir = _tabulate_tensor(ir, parameters)
    return pyop2.FunDecl("void", code["classname"], _arglist(ir), body_ir, ["static", "inline"])

def _arglist(ir):
    "Generate argument list for tensor tabulation function (only for pyop2)"

    rank  = len(ir['prim_idims'])
    f_j   = format["first free index"]
    f_k   = format["second free index"]
    float = format['float declaration']
    int   = format['int declaration']
    prim_idims  = ir["prim_idims"]
    integral_type = ir["integral_type"]

    if integral_type in ("interior_facet", "interior_facet_horiz", "interior_facet_vert"):
        prim_idims = [d*2 for d in prim_idims]
    localtensor = pyop2.Decl(float, pyop2.Symbol("A", tuple(prim_idims) or (1,)))

    coordinates = pyop2.Decl(float, pyop2.Symbol("**vertex_coordinates", ()))

    coeffs = []
    for n, e in zip(ir['coefficient_names'], ir['coefficient_elements']):
        coeffs.append(pyop2.Decl(float, pyop2.Symbol("*%s%s" % \
            ("c" if e.family() == 'Real' else "*", n[1:] if e.family() == 'Real' else n), ())))

    arglist = [localtensor, coordinates]
    # embedded manifold, passing in cell_orientation
    if ir['needs_oriented'] and \
        ir['cell'].topological_dimension() != ir['cell'].geometric_dimension():
        cell_orientation = pyop2.Decl(int, pyop2.Symbol("**cell_orientation_", ()))
        arglist.append(cell_orientation)
    arglist += coeffs
    if integral_type in ("exterior_facet", "exterior_facet_vert"):
        arglist.append(pyop2.Decl(int, pyop2.Symbol("*facet_p", ()), qualifiers=["unsigned"]))
    if integral_type in ("interior_facet", "interior_facet_vert"):
        arglist.append(pyop2.Decl(int, pyop2.Symbol("facet_p", (2,)), qualifiers=["unsigned"]))

    return arglist

def _tabulate_tensor(ir, parameters):
    "Generate code for a single integral (tabulate_tensor())."

    p_format        = parameters["format"]

    f_comment       = format["comment"]
    f_G             = format["geometry constant"]
    f_const_double  = format["assign"]
    f_switch        = format["switch"]
    f_float         = format["float"]
    f_assign        = format["assign"]
    f_A             = format["element tensor"][p_format]
    f_r             = format["free indices"][0]
    f_j             = format["first free index"]
    f_k             = format["second free index"]
    f_loop          = format["generate loop"]
    f_int           = format["int"]
    f_facet         = format["facet"]
    f_weight        = format["weight"]

    # Get data.
    opt_par     = ir["optimise_parameters"]
    integral_type = ir["integral_type"]
    cell        = ir["cell"]
    gdim        = cell.geometric_dimension()
    tdim        = cell.topological_dimension()
    num_facets  = ir["num_facets"]
    num_vertices= ir["num_vertices"]
    integrals   = ir["trans_integrals"]
    geo_consts  = ir["geo_consts"]
    oriented    = ir["needs_oriented"]

    # Create sets of used variables.
    used_weights    = set()
    used_psi_tables = set()
    used_nzcs       = set()
    trans_set       = set()
    sets = [used_weights, used_psi_tables, used_nzcs, trans_set]

    affine_tables = {} # TODO: This is not populated anywhere, remove?
    quadrature_weights = ir["quadrature_weights"]

    #The pyop2 format requires dereferencing constant coefficients since
    # these are passed in as double *
    common = []
    if p_format == "pyop2":
        for n, c in zip(ir["coefficient_names"], ir["coefficient_elements"]):
            if c.family() == 'Real':
                # Second index is always? 0, so we cast to (double (*)[1]).
                common += ['double (*w%(n)s)[1] = (double (*)[1])c%(n)s;\n' %
                           {'n': n[1:]}]

    operations = []
    if integral_type == "cell":
        # Update transformer with facets and generate code + set of used geometry terms.
        nest_ir, num_ops = _generate_element_tensor(integrals, sets, \
                                                    opt_par, parameters)

        # Set operations equal to num_ops (for printing info on operations).
        operations.append([num_ops])

        # Generate code for basic geometric quantities
        # @@@: Jacobian snippet
        jacobi_code  = ""
        jacobi_code += format["compute_jacobian"](cell)
        jacobi_code += "\n"
        jacobi_code += format["compute_jacobian_inverse"](cell)
        if oriented and tdim != gdim:
            # NEED TO THINK ABOUT THIS FOR EXTRUSION
            jacobi_code += format["orientation"][p_format](tdim, gdim)
        jacobi_code += "\n"
        jacobi_code += format["scale factor snippet"][p_format]

        # Generate code for cell volume and circumradius -- note that the
        # former will be incorrect on extruded meshes by a constant factor.
        jacobi_code += "\n\n" + format["generate cell volume"][p_format](tdim, gdim, integral_type)
        jacobi_code += "\n\n" + format["generate circumradius"][p_format](tdim, gdim, integral_type)

    elif integral_type in ("exterior_facet", "exterior_facet_vert"):
        if p_format == 'pyop2':
            common += ["unsigned int facet = *facet_p;\n"]

        cases = [None for i in range(num_facets)]
        for i in range(num_facets):
            # Update transformer with facets and generate case code + set of used geometry terms.
            nest_ir, ops = _generate_element_tensor(integrals[i], sets, opt_par, parameters)
            case = [f_comment("Total number of operations to compute element tensor (from this point): %d" % ops)]
            case += [nest_ir.gencode()]
            cases[i] = "\n".join(case)

            # Save number of operations (for printing info on operations).
            operations.append([i, ops])

        # Generate tensor code for all cases using a switch.
        tensor_code = f_switch(f_facet(None), cases)
        nest_ir = pyop2.FlatBlock(tensor_code)

        # Generate code for basic geometric quantities
        # @@@: Jacobian snippet
        jacobi_code  = ""
        jacobi_code += format["compute_jacobian"](cell)
        jacobi_code += "\n"
        jacobi_code += format["compute_jacobian_inverse"](cell)
        if oriented and tdim != gdim:
            # NEED TO THINK ABOUT THIS FOR EXTRUSION
            jacobi_code += format["orientation"][p_format](tdim, gdim)
        jacobi_code += "\n"
        if integral_type == "exterior_facet":
            jacobi_code += "\n\n" + format["facet determinant"](cell, p_format, integral_type)
            jacobi_code += "\n\n" + format["generate normal"](cell, p_format, integral_type)
            jacobi_code += "\n\n" + format["generate facet area"](tdim, gdim)
            if tdim == 3:
                jacobi_code += "\n\n" + format["generate min facet edge length"](tdim, gdim)
                jacobi_code += "\n\n" + format["generate max facet edge length"](tdim, gdim)

            # Generate code for cell volume and circumradius
            jacobi_code += "\n\n" + format["generate cell volume"][p_format](tdim, gdim, integral_type)
            jacobi_code += "\n\n" + format["generate circumradius"][p_format](tdim, gdim, integral_type)

        elif integral_type == "exterior_facet_vert":
            jacobi_code += "\n\n" + format["facet determinant"](cell, p_format, integral_type)
            jacobi_code += "\n\n" + format["generate normal"](cell, p_format, integral_type)
            # OTHER THINGS NOT IMPLEMENTED YET
        else:
            raise RuntimeError("Invalid integral_type")

    # If we have an extruded horizontal facet, we don't want a switch
    # statement to be generated. Ideally, we would stop unnecessary
    # bits of code from being generated much earlier on, but this
    # is a start.
    elif integral_type in ("exterior_facet_top", "exterior_facet_bottom"):
        if integral_type == "exterior_facet_bottom":
            nest_ir, ops = _generate_element_tensor(integrals[0], sets, opt_par, parameters)
            operations.append([ops])
        elif integral_type == "exterior_facet_top":
            nest_ir, ops = _generate_element_tensor(integrals[1], sets, opt_par, parameters)
            operations.append([ops])
        else:
            raise RuntimeError("Invalid integral_type")

        # Generate code for basic geometric quantities
        # @@@: Jacobian snippet
        jacobi_code  = ""
        jacobi_code += format["compute_jacobian"](cell)
        jacobi_code += "\n"
        jacobi_code += format["compute_jacobian_inverse"](cell)
        if oriented:
            # NEED TO THINK ABOUT THIS FOR EXTRUSION
            jacobi_code += format["orientation"][p_format](tdim, gdim)
        jacobi_code += "\n"
        jacobi_code += "\n\n" + format["facet determinant"](cell, p_format, integral_type)
        jacobi_code += "\n\n" + format["generate normal"](cell, p_format, integral_type)
        # THE REST IS NOT IMPLEMENTED YET

    elif integral_type in ("interior_facet", "interior_facet_vert"):
        if p_format == 'pyop2':
            common += ["unsigned int facet_0 = facet_p[0];"]
            common += ["unsigned int facet_1 = facet_p[1];"]
            common += ["double **vertex_coordinates_0 = vertex_coordinates;"]
            # Note that the following line is unsafe for isoparametric elements.
            common += ["double **vertex_coordinates_1 = vertex_coordinates + %d;" % num_vertices]

        cases = [[None for j in range(num_facets)] for i in range(num_facets)]
        for i in range(num_facets):
            for j in range(num_facets):
                # Update transformer with facets and generate case code + set of used geometry terms.
                nest_ir, ops = _generate_element_tensor(integrals[i][j], sets, \
                                                        opt_par, parameters)
                case = [f_comment("Total number of operations to compute element tensor (from this point): %d" % ops)]
                case += [nest_ir.gencode()]
                cases[i][j] = "\n".join(case)

                # Save number of operations (for printing info on operations).
                operations.append([i, j, ops])

        # Generate tensor code for all cases using a switch.
        tensor_code = f_switch(f_facet("+"), [f_switch(f_facet("-"), cases[i]) for i in range(len(cases))])
        nest_ir = pyop2.FlatBlock(tensor_code)

        # Generate code for basic geometric quantities
        # @@@: Jacobian snippet
        jacobi_code  = ""
        for _r in ["+", "-"]:
            if p_format == "pyop2":
                jacobi_code += format["compute_jacobian_interior"](cell, r=_r)
            else:
                jacobi_code += format["compute_jacobian"](cell, r=_r)

            jacobi_code += "\n"
            jacobi_code += format["compute_jacobian_inverse"](cell, r=_r)
            if oriented and tdim != gdim:
                # NEED TO THINK ABOUT THIS FOR EXTRUSION
                jacobi_code += format["orientation"][p_format](tdim, gdim, r=_r)
            jacobi_code += "\n"

        if integral_type == "interior_facet":
            jacobi_code += "\n\n" + format["facet determinant"](cell, p_format, integral_type, r="+")
            jacobi_code += "\n\n" + format["generate normal"](cell, p_format, integral_type)

            jacobi_code += "\n\n" + format["generate facet area"](tdim, gdim)
            if tdim == 3:
                jacobi_code += "\n\n" + format["generate min facet edge length"](tdim, gdim, r="+")
                jacobi_code += "\n\n" + format["generate max facet edge length"](tdim, gdim, r="+")

            # Generate code for cell volume and circumradius
            jacobi_code += "\n\n" + format["generate cell volume"][p_format](tdim, gdim, integral_type)
            jacobi_code += "\n\n" + format["generate circumradius interior"](tdim, gdim, integral_type)

        elif integral_type == "interior_facet_vert":
            # THE REST IS NOT IMPLEMENTED YET
            jacobi_code += "\n\n" + format["facet determinant"](cell, p_format, integral_type, r="+")
            jacobi_code += "\n\n" + format["generate normal"](cell, p_format, integral_type)
        else:
            raise RuntimeError("Invalid integral_type")

    # As before, all interior horizontal facets are identical, so
    # don't write out a double-switch statement
    elif integral_type == "interior_facet_horiz":
        common += ["double **vertex_coordinates_0 = vertex_coordinates;"]
        # Note that the following line is unsafe for isoparametric elements.
        common += ["double **vertex_coordinates_1 = vertex_coordinates + %d;" % num_vertices]

        # Generate the code we need, corresponding to facet 1 [top] of
        # the lower element, and facet 0 [bottom] of the top element
        nest_ir, ops = _generate_element_tensor(integrals[1][0], sets, \
                                                opt_par, parameters)

        # Save number of operations (for printing info on operations).
        operations.append([ops])

        # Generate code for basic geometric quantities
        # @@@: Jacobian snippet
        jacobi_code  = ""
        for _r in ["+", "-"]:
            jacobi_code += format["compute_jacobian_interior"](cell, r=_r)
            jacobi_code += "\n"
            jacobi_code += format["compute_jacobian_inverse"](cell, r=_r)
            if oriented:
                # NEED TO THINK ABOUT THIS FOR EXTRUSION
                jacobi_code += format["orientation"][p_format](tdim, gdim, r=_r)
            jacobi_code += "\n"

        # TODO: verify that this is correct (we think it is)
        jacobi_code += "\n\n" + format["facet determinant"](cell, p_format, integral_type, r="+")
        jacobi_code += "\n\n" + format["generate normal"](cell, p_format, integral_type)
        # THE REST IS NOT IMPLEMENTED YET

    elif integral_type == "point":
        cases = [None for i in range(num_vertices)]
        for i in range(num_vertices):
            # Update transformer with vertices and generate case code +
            # set of used geometry terms.
            nest_ir, ops = _generate_element_tensor(integrals[i],
                                                    sets, opt_par, parameters)
            case = [f_comment("Total number of operations to compute element tensor (from this point): %d" % ops)]
            case += c
            cases[i] = "\n".join(case)

            # Save number of operations (for printing info on operations).
            operations.append([i, ops])

        # Generate tensor code for all cases using a switch.
        tensor_code = f_switch(format["vertex"], cases)

        # Generate code for basic geometric quantities
        # @@@: Jacobian snippet
        jacobi_code  = ""
        jacobi_code += format["compute_jacobian"](cell)
        jacobi_code += "\n"
        jacobi_code += format["compute_jacobian_inverse"](cell)
        if oriented and tdim != gdim:
            jacobi_code += format["orientation"][p_format](tdim, gdim)
        jacobi_code += "\n"

    else:
        error("Unhandled integral type: " + str(integral_type))

    # Embedded manifold, need to pass in cell orientations
    if oriented and tdim != gdim and p_format == 'pyop2':
        if integral_type in ("interior_facet", "interior_facet_vert", "interior_facet_horiz"):
            common += ["const int cell_orientation%s = cell_orientation_[0][0];" % _choose_map('+'),
                       "const int cell_orientation%s = cell_orientation_[1][0];" % _choose_map('-')]
        else:
            common += ["const int cell_orientation = cell_orientation_[0][0];"]
    # After we have generated the element code for all facets we can remove
    # the unused transformations and tabulate the used psi tables and weights.
    common += [remove_unused(jacobi_code, trans_set)]
    jacobi_ir = pyop2.FlatBlock("\n".join(common))

    # @@@: const double W3[3] = {{...}}
    pyop2_weights = []
    for weights, points in [quadrature_weights[p] for p in used_weights]:
        n_points = len(points)
        value = f_float(weights[0])
        w_sym = pyop2.Symbol(f_weight(n_points), () if n_points == 1 else (n_points,))
        values = f_float(weights[0]) if n_points == 1 else "{%s}" % ", ".join(map(str, [f_float(i) for i in weights]))
        pyop2_weights.append(pyop2.Decl("double", w_sym, pyop2.ArrayInit(values), qualifiers=["static", "const"]))

    name_map = ir["name_map"]
    tables = ir["unique_tables"]
    tables.update(affine_tables) # TODO: This is not populated anywhere, remove?

    # @@@: const double FE0[] = {{...}}
    code, decl = _tabulate_psis(tables, used_psi_tables, name_map, used_nzcs, opt_par, parameters)
    pyop2_basis = []
    for name, data in decl.items():
        rank, value = data
        feo_sym = pyop2.Symbol(name, rank)
        pyop2_basis.append(pyop2.Decl("double", feo_sym, pyop2.ArrayInit(value), qualifiers=["static", "const"]))

    # Build the root of the PyOP2' ast
    pyop2_tables = pyop2_weights + [tab for tab in pyop2_basis]
    root = pyop2.Root([jacobi_ir] + pyop2_tables + [nest_ir])

    return root

def _generate_element_tensor(integrals, sets, optimise_parameters, parameters):
    "Construct quadrature code for element tensors."

    # Prefetch formats to speed up code generation.
    f_comment    = format["comment"]
    f_ip         = format["integration points"]
    f_I          = format["ip constant"]
    f_loop       = format["generate loop"]
    f_ip_coords  = format["generate ip coordinates"]
    f_coords     = format["vertex_coordinates"]
    f_double     = format["float declaration"]
    f_decl       = format["declaration"]
    f_X          = format["ip coordinates"]
    f_C          = format["conditional"]


    # Initialise return values.
    tensor_ops_count = 0

    # We receive a dictionary {num_points: form,}.
    # Loop points and forms.
    for points, terms, functions, ip_consts, coordinate, conditionals in integrals:

        ip_code = []
        ip_ir = []
        num_ops = 0

        # Generate code to compute coordinates if used.
        if coordinate:
            raise RuntimeError("Don't know how to compute coordinates")
            # Left in place for posterity
            name, gdim, ip, r = coordinate
            element_code += ["", f_comment("Declare array to hold physical coordinate of quadrature point.")]
            element_code += [f_decl(f_double, f_X(points, gdim))]
            ops, coord_code = f_ip_coords(gdim, points, name, ip, r)
            ip_code += ["", f_comment("Compute physical coordinate of quadrature point, operations: %d." % ops)]
            ip_code += [coord_code]
            num_ops += ops
            # Update used psi tables and transformation set.
            sets[1].add(name)
            sets[3].add(f_coords(r))

        # Generate code to compute function values.
        if functions:
            func_code, ops = _generate_functions(functions, sets)
            ip_ir += func_code
            num_ops += ops

        # Generate code to compute conditionals (might depend on coordinates
        # and function values so put here).
        # TODO: Some conditionals might only depend on geometry so they
        # should be moved outside if possible.
        if conditionals:
            ip_ir.append(pyop2.Decl(f_double, c_sym(f_C(len(conditionals)))))
            # Sort conditionals (need to in case of nested conditionals).
            reversed_conds = dict([(n, (o, e)) for e, (t, o, n) in conditionals.items()])
            for num in range(len(conditionals)):
                name = format["conditional"](num)
                ops, expr = reversed_conds[num]
                ip_ir.append(pyop2.Assign(c_sym(name), c_sym(str(expr))))
                num_ops += ops

        # Generate code for ip constant declarations.
        # TODO: this code should be removable as only executed when ffc's optimisations are on
        ip_const_ops, ip_const_code = generate_aux_constants(ip_consts, f_I,\
                                        format["assign"], True)
        if len(ip_const_code) > 0:
            raise RuntimeError("IP Const code not supported")
        num_ops += ip_const_ops
        if ip_const_code:
            ip_code += ["", f_comment("Number of operations to compute ip constants: %d" %ip_const_ops)]
            ip_code += [format["declaration"](format["float declaration"], f_I(len(ip_consts)))]
            ip_code += ip_const_code

        # Generate code to evaluate the element tensor.
        nest_ir, ops = _generate_integral_ir(points, terms, sets, optimise_parameters, parameters)
        num_ops += ops
        tensor_ops_count += num_ops*points
        ip_ir += nest_ir

        # Loop code over all IPs.
        # @@@: for (ip ...) { A[0][0] += ... }
        if points > 1:
            it_var = pyop2.Symbol(f_ip, ())
            nest_ir = pyop2.For(pyop2.Decl("int", it_var, c_sym(0)), pyop2.Less(it_var, c_sym(points)), \
                        pyop2.Incr(it_var, c_sym(1)), pyop2.Block(ip_ir, open_scope=True), "#pragma pyop2 integration")
        else:
            nest_ir = pyop2.Block(ip_ir, open_scope=False)

    return (nest_ir, tensor_ops_count)


def visit_rhs(node):
    """Create a PyOP2 AST-conformed object starting from a FFC node. """

    def create_pyop2_node(typ, exp1, exp2):
        """Create an expr node starting from two FFC symbols."""
        if typ == 2:
            return pyop2.Prod(exp1, exp2)
        if typ == 3:
            return pyop2.Sum(exp1, exp2)
        if typ == 4:
            return pyop2.Div(exp1, exp2)

    def create_nested_pyop2_node(typ, nodes):
        """Create a subtree for the PyOP2 AST from a generic FFC expr. """
        if len(nodes) == 2:
            return create_pyop2_node(typ, nodes[0], nodes[1])
        else:
            return create_pyop2_node(typ, nodes[0], \
                    create_nested_pyop2_node(typ, nodes[1:]))

    if node._prec == 0:
        # Float
        return pyop2.Symbol(node.val, ())
    if node._prec == 1:
        # Symbol
        return pyop2.Symbol(node.ide, tuple(node.loop_index))
    if node._prec in [2, 3] and len(node.vrs) == 1:
        # "Fake" Product, "Fake" Sum
        return pyop2.Par(visit_rhs(node.vrs[0]))
    children = []
    if node._prec == 4:
        # Fraction
        children = [visit_rhs(node.num), visit_rhs(node.denom)]
    else:
        # Product, Sum
        children = [visit_rhs(n) for n in reversed(node.vrs)]
    # PyOP2's ast expr are binary, so we deal with this here
    return pyop2.Par(create_nested_pyop2_node(node._prec, children))


def _generate_functions(functions, sets):
    "Generate declarations for functions and code to compute values."

    f_comment      = format["comment"]
    f_double       = format["float declaration"]
    f_F            = format["function value"]
    f_float        = format["floating point"]
    f_decl         = format["declaration"]
    f_r            = format["free indices"]
    f_iadd         = format["iadd"]
    f_loop         = format["generate loop"]

    ast_items = []

    # Create the function declarations -- only the (unique) variables we need
    vardecls = set([functions[function][0] for function in functions])
    ast_items += [pyop2.Decl(f_double, c_sym(f_F(n)), c_sym(f_float(0))) \
                    for n in vardecls]

    # Get sets.
    used_psi_tables = sets[1]
    used_nzcs = sets[2]

    # Sort functions after loop ranges.
    function_list = {}
    for key, val in functions.items():
        if val[1] in function_list:
            function_list[val[1]].append(key)
        else:
            function_list[val[1]] = [key]

    total_ops = 0
    # Loop ranges and get list of functions.
    for loop_range, list_of_functions in function_list.items():
        function_expr = []
        function_numbers = []
        # Loop functions.
        func_ops = 0
        for function in list_of_functions:
            # Get name and number.
            number, range_i, ops, psi_name, u_nzcs, ufl_element = functions[function]
            if not isinstance(range_i, tuple):
                range_i = tuple([range_i])

            # Add name to used psi names and non zeros name to used_nzcs.
            used_psi_tables.add(psi_name)
            used_nzcs.update(u_nzcs)

            # # TODO: This check can be removed for speed later.
            # REMOVED this, since we might need to increment into the same
            # number more than once for mixed element + interior facets
            # ffc_assert(number not in function_expr, "This is definitely not supposed to happen!")

            # Convert function to COFFEE ast node, save string
            # representation for sorting (such that we're reproducible
            # in parallel).
            function = visit_rhs(function)
            key = str(function)
            function_expr.append((number, function, key))

            # Get number of operations to compute entry and add to function operations count.
            func_ops += (ops + 1)*sum(range_i)

        # Gather, sorted by string rep of function.
        lines = [pyop2.Incr(c_sym(f_F(n)), fn) for n, fn, _ in sorted(function_expr, key=lambda x: x[2])]
        if isinstance(loop_range, tuple):
            if not all(map(lambda x: x==loop_range[0], loop_range)):
                raise RuntimeError("General mixed elements not yet supported in PyOP2")
            loop_vars = [ (f_r[0], 0, loop_range[0]), (f_r[1], 0, len(loop_range)) ]
        else:
            loop_vars = [(f_r[0], 0, loop_range)]
        # TODO: If loop_range == 1, this loop may be unneccessary. Not sure if it's safe to just skip it.
        it_var = c_sym(loop_vars[0][0])
        loop_size = c_sym(loop_vars[0][2])
        ast_items.append(pyop2.For(pyop2.Decl("int", it_var, c_sym(0)), pyop2.Less(it_var, loop_size), \
                    pyop2.Incr(it_var, c_sym(1)), pyop2.Block(lines, open_scope=True)))

    return ast_items, total_ops

def _generate_integral_ir(points, terms, sets, optimise_parameters, parameters):
    "Generate code to evaluate the element tensor."

    # For checking if the integral code is for a matrix
    def is_matrix(loop):
        loop_indices = [ l[0] for l in loop ]
        return (format["first free index"] in loop_indices and \
                format["second free index"] in loop_indices)

    # Prefetch formats to speed up code generation.
    p_format        = parameters["format"]
    f_comment       = format["comment"]
    f_mul           = format["mul"]
    f_scale_factor  = format["scale factor"]
    f_iadd          = format["iadd"]
    f_add           = format["add"]
    f_A             = format["element tensor"][p_format]
    f_j             = format["first free index"]
    f_k             = format["second free index"]
    f_loop          = format["generate loop"]
    f_B             = format["basis constant"]

    # Initialise return values.
    code = []
    num_ops = 0
    loops = {}

    # Extract sets.
    used_weights, used_psi_tables, used_nzcs, trans_set = sets

    nests = []
    # Loop terms and create code.
    for loop, (data, entry_vals) in terms.items():
        # If we don't have any entry values, there's no need to generate the loop.
        if not entry_vals:
            continue

        # Get data.
        t_set, u_weights, u_psi_tables, u_nzcs, basis_consts = data

        # If we have a value, then we also need to update the sets of used variables.
        trans_set.update(t_set)
        used_weights.update(u_weights)
        used_psi_tables.update(u_psi_tables)
        used_nzcs.update(u_nzcs)

        # @@@: A[0][0] += FE0[ip][j]*FE0[ip][k]*W24[ip]*det;

        entry_ir = []
        for entry, value, ops in entry_vals:
            # Left hand side
            it_vars = entry if len(loop) > 0 else (0,)
            local_tensor = pyop2.Symbol(f_A(''), it_vars)
            # Right hand side
            pyop2_rhs = visit_rhs(value)
            pragma = "#pragma pyop2 assembly(j,k)" if len(loop) == 2 else ""
            entry_ir.append(pyop2.Incr(local_tensor, pyop2_rhs, pragma))

        if len(loop) == 0:
            nest = pyop2.Block(entry_ir, open_scope=True)
        elif len(loop) in [1, 2]:
            it_var = c_sym(loop[0][0])
            end = c_sym(loop[0][2])
            nest = pyop2.For(pyop2.Decl("int", it_var, c_sym(0)), pyop2.Less(it_var, end), \
                             pyop2.Incr(it_var, c_sym(1)), pyop2.Block(entry_ir, open_scope=True), "#pragma pyop2 itspace")
        if len(loop) == 2:
            it_var = c_sym(loop[1][0])
            end = c_sym(loop[1][2])
            nest_k = pyop2.For(pyop2.Decl("int", it_var, c_sym(0)), pyop2.Less(it_var, end), \
                               pyop2.Incr(it_var, c_sym(1)), pyop2.Block(entry_ir, open_scope=True), "#pragma pyop2 itspace")
            nest.children[0] = pyop2.Block([nest_k], open_scope=True)
        nests.append(nest)

    return nests, num_ops

def _tabulate_weights(quadrature_weights, parameters):
    "Generate table of quadrature weights."

    # Prefetch formats to speed up code generation.
    p_format    = parameters["format"]

    f_float     = format["floating point"]
    f_table     = format["static const float declaration"][p_format]
    f_sep       = format["list separator"]
    f_weight    = format["weight"]
    f_component = format["component"]
    f_group     = format["grouping"]
    f_decl      = format["declaration"]
    f_tensor    = format["tabulate tensor"]
    f_comment   = format["comment"]
    f_int       = format["int"]

    code = ["", f_comment("Array of quadrature weights.")]

    # Loop tables of weights and create code.
    for weights, points in quadrature_weights:
        # FIXME: For now, raise error if we don't have weights.
        # We might want to change this later.
        ffc_assert(weights.any(), "No weights.")

        # Create name and value for weight.
        num_points = len(points)
        name = f_weight(num_points)
        value = f_float(weights[0])
        if len(weights) > 1:
            name += f_component("", f_int(num_points))
            value = f_tensor(weights)
        code += [f_decl(f_table, name, value)]

        # Tabulate the quadrature points (uncomment for different parameters).
        # 1) Tabulate the points as: p0, p1, p2, with p0 = (x0, y0, z0) etc.
        # Use f_float to format the value (enable variable precision).
        formatted_points = [f_group(f_sep.join([f_float(val) for val in point]))
                            for point in points]

        # Create comment.
        comment = "Quadrature points on the UFC reference element: " \
                  + f_sep.join(formatted_points)
        code += [f_comment(comment)]

        # 2) Tabulate the coordinates of the points p0, p1, p2 etc.
        #    X: x0, x1, x2
        #    Y: y0, y1, y2
        #    Z: z0, z1, z2
#            comment = "Quadrature coordinates on the UFC reference element: "
#            code += [format["comment"](comment)]

#            # All points have the same number of coordinates.
#            num_coord = len(points[0])

#            # All points have x-coordinates.
#            xs = [f_float(p[0]) for p in points]
#            comment = "X: " + f_sep.join(xs)
#            code += [format["comment"](comment)]

#            ys = []
#            zs = []
#            # Tabulate y-coordinate if we have 2 or more coordinates.
#            if num_coord >= 2:
#                ys = [f_float(p[1]) for p in points]
#                comment = "Y: " + f_sep.join(ys)
#                code += [format["comment"](comment)]
#            # Only tabulate z-coordinate if we have 3 coordinates.
#            if num_coord == 3:
#                zs = [f_float(p[2]) for p in points]
#                comment = "Z: " + f_sep.join(zs)
#                code += [format["comment"](comment)]

        code += [""]

    return code

def _tabulate_psis(tables, used_psi_tables, inv_name_map, used_nzcs, optimise_parameters, parameters):
    "Tabulate values of basis functions and their derivatives at quadrature points."

    # Prefetch formats to speed up code generation.
    p_format    = parameters["format"]

    f_comment     = format["comment"]
    f_table       = format["static const float declaration"][p_format]
    f_component   = format["component"]
    f_const_uint  = format["static const uint declaration"][p_format]
    f_nzcolumns   = format["nonzero columns"]
    f_list        = format["list"]
    f_decl        = format["declaration"]
    f_tensor      = format["tabulate tensor"]
    f_new_line    = format["new line"]
    f_int         = format["int"]

    # FIXME: Check if we can simplify the tabulation
    code = []
    code += [f_comment("Value of basis functions at quadrature points.")]

    # Get list of non zero columns, if we ignore ones, ignore columns with one component.
    if optimise_parameters["ignore ones"]:
        nzcs = []
        for key, val in inv_name_map.items():
            # Check if we have a table of ones or if number of non-zero columns
            # is larger than one.
            if val[1] and len(val[1][1]) > 1 or not val[3]:
                nzcs.append(val[1])
    else:
        nzcs = [val[1] for key, val in inv_name_map.items()\
                                        if val[1]]

    # TODO: Do we get arrays that are not unique?
    new_nzcs = []
    for nz in nzcs:
        # Only get unique arrays.
        if not nz in new_nzcs:
            new_nzcs.append(nz)

    # Construct name map.
    name_map = {}
    if inv_name_map:
        for name in inv_name_map:
            if inv_name_map[name][0] in name_map:
                name_map[inv_name_map[name][0]].append(name)
            else:
                name_map[inv_name_map[name][0]] = [name]

    # Loop items in table and tabulate.
    pyop2_decl = {}
    for name in sorted(list(used_psi_tables)):
        # Only proceed if values are still used (if they're not remapped).
        vals = tables[name]
        if not vals is None:
            # Add declaration to name.
            ip, dofs = numpy.shape(vals)
            decl_name = f_component(name, [ip, dofs])

            # Generate array of values.
            value = f_tensor(vals)
            code += [f_decl(f_table, decl_name, f_new_line + value), ""]

            # Store the information for creating PyOP2'ast declarations
            pyop2_decl[name] = ((ip, dofs), value)

        # Tabulate non-zero indices.
        if optimise_parameters["eliminate zeros"]:
            if name in name_map:
                for n in name_map[name]:
                    if inv_name_map[n][1] and inv_name_map[n][1] in new_nzcs:
                        i, cols = inv_name_map[n][1]
                        if not i in used_nzcs:
                            continue
                        code += [f_comment("Array of non-zero columns")]
                        value = f_list([f_int(c) for c in list(cols)])
                        name_col = f_component(f_nzcolumns(i), len(cols))
                        code += [f_decl(f_const_uint, name_col, value), ""]

                        # Store the nzc info for creating PyOP2'ast declarations
                        # TODO: to be tested yet, and lack the size of the array
                        #pyop2_decl[name] = value

                        # Remove from list of columns.
                        new_nzcs.remove(inv_name_map[n][1])
    return code, pyop2_decl
