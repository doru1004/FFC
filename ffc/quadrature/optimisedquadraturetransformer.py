"QuadratureTransformer (optimised) for quadrature code generation to translate UFL expressions."

# Copyright (C) 2009-2011 Kristian B. Oelgaard
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
# Modified by Anders Logg, 2009

# Python modules.
from numpy import shape

from six import iteritems, iterkeys
from six.moves import xrange as range
from six import advance_iterator as next
def firstkey(d):
    return next(iterkeys(d))

# UFL common.
from ufl.common import product
from ufl.utils.sorting import sorted_by_key

# UFL Classes.
from ufl.classes import FixedIndex
from ufl.classes import IntValue
from ufl.classes import FloatValue
from ufl.classes import Coefficient
from ufl.classes import Operator

# FFC modules.
from ffc.log import info, debug, error, ffc_assert
from ffc.cpp import format
from ffc.quadrature.quadraturetransformerbase import QuadratureTransformerBase
from ffc.quadrature.quadratureutils import create_permutations

# Symbolics functions
#from ffc.quadrature.symbolics import set_format
from ffc.quadrature.symbolics import create_float, create_symbol, create_product,\
                                     create_sum, create_fraction, BASIS, IP, GEO, CONST,\
                                     create_funcall, create_expression
                                     
from ffc.cpp import _choose_map
from coffee import base as ast


class QuadratureTransformerOpt(QuadratureTransformerBase):
    "Transform UFL representation to quadrature code."

    def __init__(self, *args):

        # Initialise base class.
        QuadratureTransformerBase.__init__(self, *args)
#        set_format(format)

    # -------------------------------------------------------------------------
    # Start handling UFL classes.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # AlgebraOperators (algebra.py).
    # -------------------------------------------------------------------------
    def sum(self, o, *operands):
        #print("Visiting Sum: " + repr(o) + "\noperands: " + "\n".join(map(repr, operands)))

        code = {}
        # Loop operands that has to be summend.
        for op in operands:
            # If entries does already exist we can add the code, otherwise just
            # dump them in the element tensor.
            for key, val in sorted(op.items()):
                if key in code:
                    code[key].append(val)
                else:
                    code[key] = [val]

        # Add sums and group if necessary.
        for key, val in sorted_by_key(code):
            if len(val) > 1:
                code[key] = create_sum(val)
            elif val:
                code[key] = val[0]
            else:
                error("Where did the values go?")
            # If value is zero just ignore it.
            if abs(code[key].val) < format["epsilon"]:
                del code[key]

        return code

    def product(self, o, *operands):
        #print("\n\nVisiting Product:\n" + str(tree_format(o)))

        permute = []
        not_permute = []

        # Sort operands in objects that needs permutation and objects that does not.
        for op in operands:
            # If we get an empty dict, something was zero and so is the product.
            if not op:
                return {}
            if len(op) > 1 or (op and firstkey(op) != ()):
                permute.append(op)
            elif op and firstkey(op) == ():
                not_permute.append(op[()])

        # Create permutations.
        # TODO: After all indices have been expanded I don't think that we'll
        # ever get more than a list of entries and values.
        #print("\npermute: " + repr(permute))
        #print("\nnot_permute: " + repr(not_permute))
        permutations = create_permutations(permute)
        #print("\npermutations: " + repr(permutations))

        # Create code.
        code ={}
        if permutations:
            for key, val in permutations.items():
                # Sort key in order to create a unique key.
                l = sorted(key)

                # TODO: I think this check can be removed for speed since we
                # just have a list of objects we should never get any conflicts here.
                ffc_assert(tuple(l) not in code, "This key should not be in the code.")

                code[tuple(l)] = create_product(val + not_permute)
        else:
            return {():create_product(not_permute)}
        return code

    def division(self, o, *operands):
        #print("\n\nVisiting Division: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))

        ffc_assert(len(operands) == 2, "Expected exactly two operands (numerator and denominator): " + repr(operands))

        # Get the code from the operands.
        numerator_code, denominator_code = operands

        # TODO: Are these safety checks needed?
        ffc_assert(() in denominator_code and len(denominator_code) == 1, \
                   "Only support function type denominator: " + repr(denominator_code))

        code = {}
        # Get denominator and create new values for the numerator.
        denominator = denominator_code[()]
        for key, val in numerator_code.items():
            code[key] = create_fraction(val, denominator)

        return code

    def power(self, o):
        #print("\n\nVisiting Power: " + repr(o))

        # Get base and exponent.
        base, expo = o.ufl_operands

        # Visit base to get base code.
        base_code = self.visit(base)

        # TODO: Are these safety checks needed?
        ffc_assert(() in base_code and len(base_code) == 1, "Only support function type base: " + repr(base_code))

        # Get the base code and create power.
        val = base_code[()]

        # Handle different exponents
        if isinstance(expo, IntValue):
            return {(): create_product([val]*expo.value())}
        elif isinstance(expo, FloatValue):
            if self.parameters["format"] == "pyop2":
                exp = create_float(expo.value())
            else:
                exp = format["floating point"](expo.value())
            var = format["std power"][self.parameters["format"]](str(val), exp)
            if self.parameters["format"] == "pyop2":
                sym = create_funcall(var, [val, exp])
            else:
                sym = create_symbol(var, val.t, val, 1, iden=var)
            return {(): sym}
        elif isinstance(expo, (Coefficient, Operator)):
            exp = self.visit(expo)[()]
#            print "pow exp: ", exp
#            print "pow val: ", val
            var = format["std power"][self.parameters["format"]](str(val), exp)
            if self.parameters["format"] == "pyop2":
                sym = create_funcall(var, [val, exp])
            else:
                sym = create_symbol(var, val.t, val, 1, iden=var)
            return {(): sym}
        else:
            error("power does not support this exponent: " + repr(expo))

    def abs(self, o, *operands):
        #print("\n\nVisiting Abs: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))

        # TODO: Are these safety checks needed?
        ffc_assert(len(operands) == 1 and () in operands[0] and len(operands[0]) == 1, \
                   "Abs expects one operand of function type: " + repr(operands))

        # Take absolute value of operand.
        val = operands[0][()]
        var = format["absolute value"][self.parameters["format"]]()
        new_val = create_funcall(var, [val])
        return {():new_val}

    def min_value(self, o, *operands):
        # Take minimum value of operands.
        val0 = operands[0][()]
        val1 = operands[1][()]
        t = min(val0.t, val1.t)
        # FIXME: I don't know how to implement this the optimized way. Is this right?
        if self.parameters["format"] == "pyop2":
            new_val = create_funcall("fmin", [val0, val1])
        else:
            new_val = create_symbol(format["min value"](str(val0), str(val1)), t)
        return {():new_val}

    def max_value(self, o, *operands):
        # Take maximum value of operands.
        val0 = operands[0][()]
        val1 = operands[1][()]
        t = min(val0.t, val1.t)
        # FIXME: I don't know how to implement this the optimized way. Is this right?
        if self.parameters["format"] == "pyop2":
            new_val = create_funcall("fmax", [val0, val1])
        else:
            new_val = create_symbol(format["max value"](str(val0), str(val1)), t)
        return {():new_val}

    # -------------------------------------------------------------------------
    # Condition, Conditional (conditional.py).
    # -------------------------------------------------------------------------
    def not_condition(self, o, *operands):
        # This is a Condition but not a BinaryCondition, and the operand will be another Condition
        # Get condition expression and do safety checks.
        # Might be a bit too strict?
        c, = operands
        ffc_assert(len(c) == 1 and firstkey(c) == (),\
            "Condition for NotCondition should only be one function: " + repr(c))
        if self.parameters["format"] == "pyop2":
            return {(): create_expression(ast.Not, [c[()]], c[()].t)}
        var = format["not"](str(c[()]))
        sym = create_symbol(var, c[()].t, base_op=c[()].ops()+1, iden=var)
        return {(): sym}

    def binary_condition(self, o, *operands):

        # Get LHS and RHS expressions and do safety checks.
        # Might be a bit too strict?
        lhs, rhs = operands
        ffc_assert(len(lhs) == 1 and firstkey(lhs) == (),\
            "LHS of Condtion should only be one function: " + repr(lhs))
        ffc_assert(len(rhs) == 1 and firstkey(rhs) == (),\
            "RHS of Condtion should only be one function: " + repr(rhs))

        # Map names from UFL to cpp.py.
        name_map = {"==":"is equal", "!=":"not equal",\
                    "<":"less than", ">":"greater than",\
                    "<=":"less equal", ">=":"greater equal",\
                    "&&": "and", "||": "or"}

        # Get the minimum type
        t = min(lhs[()].t, rhs[()].t)
        ops = lhs[()].ops() + rhs[()].ops() + 1
        if self.parameters["format"] == "pyop2":
            name_map = {"==": ast.Eq,
                        "!=" : ast.NEq,
                        "<": ast.Less,
                        ">": ast.Greater,
                        "<=": ast.LessEq,
                        ">=": ast.GreaterEq,
                        "&&": ast.And,
                        "||": ast.Or}
            expr = create_expression(name_map[o._name], [lhs[()], rhs[()]], t)
            return {(): expr}
        cond = str(lhs[()])+format[name_map[o._name]]+str(rhs[()])
        var = format["grouping"](cond)
        sym = create_symbol(var, t, base_op=ops, iden=var)
        return {(): sym}

    def conditional(self, o, *operands):
        # Get condition and return values; and do safety check.
        cond, true, false = operands
        ffc_assert(len(cond) == 1 and firstkey(cond) == (),\
            "Condtion should only be one function: " + repr(cond))
        ffc_assert(len(true) == 1 and firstkey(true) == (),\
            "True value of Condtional should only be one function: " + repr(true))
        ffc_assert(len(false) == 1 and firstkey(false) == (),\
            "False value of Condtional should only be one function: " + repr(false))

        # Get values and test for None
        t_val = true[()]
        f_val = false[()]

        # Get the minimum type and number of operations
        # TODO: conditionals are currently always located inside the ip loop,
        # therefore the type has to be at least IP (fix bug #1082048). This can
        # be optimised.
        t = min([cond[()].t, t_val.t, f_val.t, IP])
        ops = sum([cond[()].ops(), t_val.ops(), f_val.ops()])

        # Create expression for conditional
        # TODO: Handle this differently to expose the variables which are used
        # to create the expressions.
        var = format["evaluate conditional"](cond[()], t_val, f_val)
        if self.parameters["format"] == "pyop2":
            expr = create_expression(ast.Ternary, [cond[()], t_val, f_val], t, base_op=ops)
        else:
            expr = create_symbol(var, t, iden=var)
        num = len(self.conditionals)
        var = format["conditional"](num)
        name = create_symbol(var, t, iden=var)
        if not expr in self.conditionals:
            self.conditionals[expr] = (t, ops, num)
        else:
            num = self.conditionals[expr][2]
            var = format["conditional"](num)
            name = create_symbol(var, t, iden=var)
        return {():name}

    # -------------------------------------------------------------------------
    # FacetNormal, CellVolume, Circumradius, FacetArea (geometry.py).
    # -------------------------------------------------------------------------
    def cell_coordinate(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def facet_coordinate(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def cell_origin(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def facet_origin(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def cell_facet_origin(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def jacobian(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def jacobian_determinant(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def jacobian_inverse(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def facet_jacobian(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def facet_jacobian_determinant(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def facet_jacobian_inverse(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def cell_facet_jacobian(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def cell_facet_jacobian_determinant(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def cell_facet_jacobian_inverse(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def facet_normal(self, o):
        components = self.component()

        # Safety check.
        ffc_assert(len(components) == 1,
                   "FacetNormal expects 1 component index: " + repr(components))

        # Handle 1D as a special case.
        # FIXME: KBO: This has to change for mD elements in R^n : m < n
        if self.gdim == 1: # FIXME: MSA UFL uses shape (1,) now, can we remove the special case here then?
            normal_component = format["normal component"](self.restriction, "")
        else:
            normal_component = format["normal component"](self.restriction, components[0])
        self.trans_set.add(normal_component)

        return {(): create_symbol(normal_component, GEO, iden=normal_component)}

    def cell_normal(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def cell_volume(self, o):
        # FIXME: KBO: This has to change for higher order elements
        #detJ = format["det(J)"](self.restriction)
        #volume = format["absolute value"](detJ)
        #self.trans_set.add(detJ)

        volume = format["cell volume"](self.restriction)
        self.trans_set.add(volume)

        return {():create_symbol(volume, GEO, iden=volume)}

    def circumradius(self, o):
        # FIXME: KBO: This has to change for higher order elements
        circumradius = format["circumradius"](self.restriction)
        self.trans_set.add(circumradius)

        return {():create_symbol(circumradius, GEO, iden=circumradius)}

    def facet_area(self, o):
        # FIXME: KBO: This has to change for higher order elements
        # NOTE: Omitting restriction because the area of a facet is the same
        # on both sides.
        # FIXME: Since we use the scale factor, facet area has no meaning
        # for cell integrals. (Need check in FFC or UFL).
        area = format["facet area"]
        self.trans_set.add(area)

        return {():create_symbol(area, GEO, iden=area)}

    def min_facet_edge_length(self, o):
        # FIXME: this has no meaning for cell integrals. (Need check in FFC or UFL).

        tdim = self.tdim # FIXME: o.domain().topological_dimension() ???
        if tdim < 3:
            return self.facet_area(o)

        edgelen = format["min facet edge length"](self.restriction)
        self.trans_set.add(edgelen)

        return {():create_symbol(edgelen, GEO, iden=edgelen)}

    def max_facet_edge_length(self, o):
        # FIXME: this has no meaning for cell integrals. (Need check in FFC or UFL).

        tdim = self.tdim # FIXME: o.domain().topological_dimension() ???
        if tdim < 3:
            return self.facet_area(o)

        edgelen = format["max facet edge length"](self.restriction)
        self.trans_set.add(edgelen)

        return {():create_symbol(edgelen, GEO, iden=edgelen)}

    def cell_orientation(self, o): # FIXME
        error("This object should be implemented by the child class.")

    def quadrature_weight(self, o):
        # I don't think this should exist at all, but I left it in
        # so as not to break upstream compatibility...
        return {(): create_float(1.0)}

    # -------------------------------------------------------------------------

    def reference_value(self, o):
        return self.visit(o.ufl_operands[0])

    def create_argument(self, ufl_argument, derivatives, component, local_comp,
                        local_offset, ffc_element, transformation,
                        tdim, gdim, avg):
        "Create code for basis functions, and update relevant tables of used basis."

        # Prefetch formats to speed up code generation.
        f_transform     = format["transform"]
        f_detJ          = format["det(J)"]

        # Reset code
        code = {}

        if transformation != "affine":
            ffc_assert(avg is None, "Taking average is not supported for non-affine mappings.")

        # Get derivatives.
        deriv = [derivatives.count(i) for i in range(tdim)]
        if not any(deriv):
            deriv = []

        # Create mapping and basis name.
        mapping, basis = self._create_mapping_basis(component, deriv, avg, ufl_argument, ffc_element)

        # VFS-on-interior-facets case
        if isinstance(mapping, list):
            for ma, ba in zip(mapping, basis):
                if not ma in code:
                    code[ma] = []
                if basis is not None:
                    # Add transformation if needed.
                    code[ma].append(ba)
        else:
            if not mapping in code:
                code[mapping] = []
            if basis is not None:
                # Add transformation if needed.
                code[mapping].append(basis)

        # Add sums and group if necessary.
        for key, val in list(code.items()):
            if len(val) > 1:
                code[key] = create_sum(val)
            else:
                code[key] = val[0]

        return code

    def create_function(self, ufl_function, derivatives, component, local_comp,
                       local_offset, ffc_element, is_quad_element, transformation,
                       tdim, gdim, avg):
        "Create code for basis functions, and update relevant tables of used basis."
        ffc_assert(ufl_function in self._function_replace_values, "Expecting ufl_function to have been mapped prior to this call.")

        # Prefetch formats to speed up code generation.
        f_transform     = format["transform"]
        f_detJ          = format["det(J)"]

        # Reset code
        code = []

        if transformation != "affine":
            ffc_assert(avg is None, "Taking average is not supported for non-affine mappings.")

        # Get derivatives.
        deriv = [derivatives.count(i) for i in range(tdim)]
        if not any(deriv):
            deriv = []

        # Create function name.
        function_name = self._create_function_name(component, deriv, avg, is_quad_element, ufl_function, ffc_element)

        if function_name:
            code.append(function_name)

        if not code:
            return create_float(0.0)
        elif len(code) > 1:
            code = create_sum(code)
        else:
            code = code[0]

        return code


    # -------------------------------------------------------------------------
    # Helper functions for transformation of UFL objects in base class
    # -------------------------------------------------------------------------
    def _create_symbol(self, symbol, domain, _loop_index=[], _iden=None):
        return {():create_symbol(symbol, domain, loop_index=_loop_index, iden=_iden)}

    def _create_product(self, symbols):
        return create_product(symbols)

    def _format_scalar_value(self, value):
        #print("format_scalar_value: %d" % value)
        if value is None:
            return {():create_float(0.0)}
        return {():create_float(value)}

    def _math_function(self, operands, format_function):
        #print("Calling _math_function() of optimisedquadraturetransformer.")
        # TODO: Are these safety checks needed?
        ffc_assert(len(operands) == 1 and () in operands[0] and len(operands[0]) == 1, \
                   "MathFunctions expect one operand of function type: " + repr(operands))
        # Use format function on value of operand.
        operand = operands[0]
        for key, val in list(operand.items()):
            fun_name = format_function()
            new_val = create_funcall(format_function(), [val])
            operand[key] = new_val
        return operand

    def _bessel_function(self, operands, format_function):
        # TODO: Are these safety checks needed?
        # TODO: work on reference instead of copies? (like math_function)
        ffc_assert(len(operands) == 2,\
          "BesselFunctions expect two operands of function type: " + repr(operands))
        nu, x = operands
        ffc_assert(len(nu) == 1 and () in nu,\
          "Expecting one operand of function type as first argument to BesselFunction : " + repr(nu))
        ffc_assert(len(x) == 1 and () in x,\
          "Expecting one operand of function type as second argument to BesselFunction : " + repr(x))
        nu = nu[()]
        x = x[()]
        if nu is None:
            nu = format["floating point"](0.0)
        if x is None:
            x = format["floating point"](0.0)

        sym = create_symbol(format_function(x,nu), x.t, x, 1)
        return {():sym}

    # -------------------------------------------------------------------------
    # Helper functions for code_generation()
    # -------------------------------------------------------------------------
    def _count_operations(self, expression):
        return expression.ops()

    def _create_entry_data(self, val, integral_type):
#        zero = False
        # Multiply value by weight and determinant
        ACCESS = GEO
        weight = format["weight"](self.points)
        iden = weight
        loop_index = ()
        if self.points > 1:
            weight += format["component"]("", format["integration points"])
            ACCESS = IP
            loop_index = [format["integration points"]]
        weight = self._create_symbol(weight, ACCESS, _loop_index=loop_index, _iden=iden)[()]
        # Create value.
        trans_set = set()
        value = create_product([val, weight])

        # Update sets of used variables (if they will not be used because of
        # optimisations later, they will be reset).
        trans_set.update([str(x) for x in value.get_unique_vars(GEO)])
        used_points = set([self.points])
        ops = self._count_operations(value)
        used_psi_tables = set([self.psi_tables_map[b]
                               for b in value.get_unique_vars(BASIS)])

        return (value, ops, [trans_set, used_points, used_psi_tables])
