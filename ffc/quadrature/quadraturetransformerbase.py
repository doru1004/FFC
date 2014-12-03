"""QuadratureTransformerBase, a common class for quadrature
transformers to translate UFL expressions."""

# Copyright (C) 2009-2013 Kristian B. Oelgaard
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
# Modified by Martin Alnaes, 2013
# Modified by Garth N. Wells, 2013
#
# First added:  2009-10-13
# Last changed: 2014-10-08

# Python modules.
from six.moves import zip
from numpy import shape, allclose, array

# UFL Classes.
from ufl.classes import FixedIndex, Index
from ufl.common import StackDict, Stack, product
from ufl.permutation import build_component_numbering

# UFL Algorithms.
from ufl.algorithms import Transformer

# FFC modules.
from ffc.log import ffc_assert, error, info
from ffc.fiatinterface import create_element, map_facet_points
from ffc.mixedelement import MixedElement
from ffc.cpp import format
from ffc.utils import Bunch

# FFC tensor modules.
from ffc.tensor.multiindex import MultiIndex as FFCMultiIndex

# Utility and optimisation functions for quadraturegenerator.
from ffc.quadrature.quadratureutils import create_psi_tables
from ffc.quadrature.symbolics import BASIS, IP, GEO, CONST


class EmptyIntegrandError(RuntimeError):
    pass


class QuadratureTransformerBase(Transformer):
    "Transform UFL representation to quadrature code."

    def __init__(self,
                 psi_tables,
                 quad_weights,
                 gdim,
                 tdim,
                 entity_type,
                 function_replace_map,
                 optimise_parameters,
                 parameters):

        Transformer.__init__(self)

        # Save optimise_parameters, weights and fiat_elements_map.
        self.optimise_parameters = optimise_parameters

        # Save parameters
        self.parameters = parameters

        # Map from original functions with possibly incomplete elements
        # to functions with properly completed elements
        self._function_replace_map = function_replace_map
        self._function_replace_values = set(function_replace_map.values()) # For assertions

        # Create containers and variables.
        self.used_psi_tables = set()
        self.psi_tables_map = {}
        self.used_weights = set()
        self.quad_weights = quad_weights
        self.used_nzcs = set()
        self.ip_consts = {}
        self.trans_set = set()
        self.function_data = {}
        self.tdim = tdim
        self.gdim = gdim
        self.entity_type = entity_type
        self.points = 0
        self.facet0 = None
        self.facet1 = None
        self.vertex = None
        self.restriction = None
        self.avg = None
        self.coordinate = None
        self.conditionals = {}
        self.additional_includes_set = set()
        self.__psi_tables = psi_tables # TODO: Unused? Remove?

        # Stacks.
        self._derivatives = []
        self._index2value = StackDict()
        self._components = Stack()

        self.element_map, self.name_map, self.unique_tables =\
            create_psi_tables(psi_tables, self.optimise_parameters["eliminate zeros"], self.entity_type)

        # Cache.
        self.argument_cache = {}
        self.function_cache = {}

    def update_cell(self):
        ffc_assert(self.entity_type == "cell", "Not expecting update_cell on a %s." % self.entity_type)
        self.facet0 = None
        self.facet1 = None
        self.vertex = None
        self.coordinate = None
        self.conditionals = {}

    def update_facets(self, facet0, facet1):
        ffc_assert(self.entity_type in ("facet", "horiz_facet", "vert_facet"), "Not expecting update_facets on a %s." % self.entity_type)
        self.facet0 = facet0
        self.facet1 = facet1
        self.vertex = None
        self.coordinate = None
        self.conditionals = {}

    def update_vertex(self, vertex):
        ffc_assert(self.entity_type == "vertex", "Not expecting update_vertex on a %s." % self.entity_type)
        self.facet0 = None
        self.facet1 = None
        self.vertex = vertex
        self.coordinate = None
        self.conditionals = {}

    def update_points(self, points):
        self.points = points
        self.coordinate = None
        # Reset functions everytime we move to a new quadrature loop
        self.conditionals = {}
        self.function_data = {}

        # Reset cache
        self.argument_cache = {}
        self.function_cache = {}

    def disp(self):
        print("\n\n **** Displaying QuadratureTransformer ****")
        print("\nQuadratureTransformer, element_map:\n", self.element_map)
        print("\nQuadratureTransformer, name_map:\n", self.name_map)
        print("\nQuadratureTransformer, unique_tables:\n", self.unique_tables)
        print("\nQuadratureTransformer, used_psi_tables:\n", self.used_psi_tables)
        print("\nQuadratureTransformer, psi_tables_map:\n", self.psi_tables_map)
        print("\nQuadratureTransformer, used_weights:\n", self.used_weights)

    def component(self):
        "Return current component tuple."
        if len(self._components):
            return self._components.peek()
        return ()

    def derivatives(self):
        "Return all derivatives tuple."
        if len(self._derivatives):
            return tuple(self._derivatives[:])
        return ()

    # -------------------------------------------------------------------------
    # Start handling UFL classes.
    # -------------------------------------------------------------------------
    # Nothing in expr.py is handled. Can only handle children of these clases.
    def expr(self, o):
        print("\n\nVisiting basic Expr:", repr(o), "with operands:")
        error("This expression is not handled: " + repr(o))

    # Nothing in terminal.py is handled. Can only handle children of these clases.
    def terminal(self, o):
        print("\n\nVisiting basic Terminal:", repr(o), "with operands:")
        error("This terminal is not handled: " + repr(o))

    # -------------------------------------------------------------------------
    # Things which should not be here (after expansion etc.) from:
    # algebra.py, differentiation.py, finiteelement.py,
    # form.py, geometry.py, indexing.py, integral.py, tensoralgebra.py, variable.py.
    # -------------------------------------------------------------------------
    def derivative(self, o, *operands):
        print("\n\nVisiting Derivative: ", repr(o))
        error("All derivatives apart from Grad should have been expanded!!")

    def compound_tensor_operator(self, o):
        print("\n\nVisiting CompoundTensorOperator: ", repr(o))
        error("CompoundTensorOperator should have been expanded.")

    def label(self, o):
        print("\n\nVisiting Label: ", repr(o))
        error("What is a Lable doing in the integrand?")

    # -------------------------------------------------------------------------
    # Things which are not supported yet, from:
    # condition.py, constantvalue.py, function.py, geometry.py, lifting.py,
    # mathfunctions.py, restriction.py
    # -------------------------------------------------------------------------
    def condition(self, o):
        print("\n\nVisiting Condition:", repr(o))
        error("This type of Condition is not supported (yet).")

    def constant_value(self, o):
        print("\n\nVisiting ConstantValue:", repr(o))
        error("This type of ConstantValue is not supported (yet).")

    def geometric_quantity(self, o):
        print("\n\nVisiting GeometricQuantity:", repr(o))
        error("This type of GeometricQuantity is not supported (yet).")

    def math_function(self, o):
        print("\n\nVisiting MathFunction:", repr(o))
        error("This MathFunction is not supported (yet).")

    def atan_2_function(self, o):
        print("\n\nVisiting Atan2Function:", repr(o))
        error("Atan2Function is not implemented (yet).")

    def bessel_function(self, o):
        print("\n\nVisiting BesselFunction:", repr(o))
        error("BesselFunction is not implemented (yet).")

    def restricted(self, o):
        print("\n\nVisiting Restricted:", repr(o))
        error("This type of Restricted is not supported (only positive and negative are currently supported).")

    # -------------------------------------------------------------------------
    # Handlers that should be implemented by child classes.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # AlgebraOperators (algebra.py).
    # -------------------------------------------------------------------------
    def sum(self, o, *operands):
        print("\n\nVisiting Sum: ", repr(o))
        error("This object should be implemented by the child class.")

    def product(self, o, *operands):
        print("\n\nVisiting Product: ", repr(o))
        error("This object should be implemented by the child class.")

    def division(self, o, *operands):
        print("\n\nVisiting Division: ", repr(o))
        error("This object should be implemented by the child class.")

    def power(self, o):
        print("\n\nVisiting Power: ", repr(o))
        error("This object should be implemented by the child class.")

    def abs(self, o, *operands):
        print("\n\nVisiting Abs: ", repr(o))
        error("This object should be implemented by the child class.")

    # -------------------------------------------------------------------------
    # FacetNormal, CellVolume, Circumradius (geometry.py).
    # -------------------------------------------------------------------------
    def cell_coordinate(self, o):
        error("This object should be implemented by the child class.")

    def facet_coordinate(self, o):
        error("This object should be implemented by the child class.")

    def cell_origin(self, o):
        error("This object should be implemented by the child class.")

    def facet_origin(self, o):
        error("This object should be implemented by the child class.")

    def cell_facet_origin(self, o):
        error("This object should be implemented by the child class.")

    def jacobian(self, o):
        error("This object should be implemented by the child class.")

    def jacobian_determinant(self, o):
        error("This object should be implemented by the child class.")

    def jacobian_inverse(self, o):
        error("This object should be implemented by the child class.")

    def facet_jacobian(self, o):
        error("This object should be implemented by the child class.")

    def facet_jacobian_determinant(self, o):
        error("This object should be implemented by the child class.")

    def facet_jacobian_inverse(self, o):
        error("This object should be implemented by the child class.")

    def cell_facet_jacobian(self, o):
        error("This object should be implemented by the child class.")

    def cell_facet_jacobian_determinant(self, o):
        error("This object should be implemented by the child class.")

    def cell_facet_jacobian_inverse(self, o):
        error("This object should be implemented by the child class.")

    def facet_normal(self, o):
        error("This object should be implemented by the child class.")

    def cell_normal(self, o):
        error("This object should be implemented by the child class.")

    def cell_volume(self, o):
        error("This object should be implemented by the child class.")

    def circumradius(self, o):
        error("This object should be implemented by the child class.")

    def facet_area(self, o):
        error("This object should be implemented by the child class.")

    def min_facet_edge_length(self, o):
        error("This object should be implemented by the child class.")

    def max_facet_edge_length(self, o):
        error("This object should be implemented by the child class.")

    def cell_orientation(self, o):
        error("This object should be implemented by the child class.")

    def quadrature_weight(self, o):
        error("This object should be implemented by the child class.")

    # -------------------------------------------------------------------------
    # Things that can be handled by the base class.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Argument (basisfunction.py).
    # -------------------------------------------------------------------------
    def argument(self, o):
        #print("\nVisiting Argument:" + repr(o))

        # Create aux. info.
        components = self.component()
        derivatives = self.derivatives()
        # Splat restriction for Real arguments (it has no meaning anyway)
        if self.restriction and o.element().family() == "Real":
            self.restriction = None

        # Check if basis is already in cache
        key = (o, components, derivatives, self.restriction, self.avg)
        basis = self.argument_cache.get(key, None)

        tdim = self.tdim # FIXME: o.domain().topological_dimension() ???

        # FIXME: Why does using a code dict from cache make the expression manipulations blow (MemoryError) up later?
        # PyOP2 format always uses optimised quadrature transformer,
        # hence we must take this route
        if basis is None or self.parameters["format"] == "pyop2" or self.optimise_parameters["optimisation"]:
            # Get auxiliary variables to generate basis
            (component, local_elem,
             ffc_element, transformation) = self._get_auxiliary_variables(o, components)

            # Create mapping and code for basis function and add to dict.
            basis = self.create_argument(o, derivatives, component,
                                         ffc_element, transformation,
                                         tdim, self.avg)
            self.argument_cache[key] = basis

        return basis

    # -------------------------------------------------------------------------
    # Constant values (constantvalue.py).
    # -------------------------------------------------------------------------
    def identity(self, o):
        #print "\n\nVisiting Identity: ", repr(o)

        # Get components
        i, j = self.component()

        # Only return a value if i==j
        if i == j:
            return self._format_scalar_value(1.0)
        else:
            return self._format_scalar_value(None)

    def scalar_value(self, o):
        "ScalarValue covers IntValue and FloatValue"
        #print "\n\nVisiting ScalarValue: ", repr(o)
        return self._format_scalar_value(o.value())

    def zero(self, o):
        #print "\n\nVisiting Zero:", repr(o)
        return self._format_scalar_value(None)

    # -------------------------------------------------------------------------
    # Grad (differentiation.py).
    # -------------------------------------------------------------------------
    def grad(self, o):
        # This should be turned into J^(-T)[i, j]*ReferenceGrad[j]
        error("This object should be implemented by the child class.")

    def reference_grad(self, o):
        #print("\n\nVisiting ReferenceGrad: " + repr(o))

        # Get expression
        derivative_expr, = o.ufl_operands

        # Get components
        components = self.component()

        en = derivative_expr.rank()
        cn = len(components)
        ffc_assert(o.rank() == cn, "Expecting rank of grad expression to match components length.")

        # Get direction of derivative
        if cn == en+1:
            der = components[en]
            self._components.push(components[:en])
        elif cn == en:
            # This happens in 1D, sligtly messy result of defining grad(f) == f.dx(0)
            der = 0
        else:
            error("Unexpected rank %d and component length %d in grad expression." % (en, cn))

        # Add direction to list of derivatives
        self._derivatives.append(der)

        # Visit children to generate the derivative code.
        code = self.visit(derivative_expr)

        # Remove the direction from list of derivatives
        self._derivatives.pop()
        if cn == en+1:
            self._components.pop()
        return code

    # -------------------------------------------------------------------------
    # Coefficient and Constants (function.py).
    # -------------------------------------------------------------------------
    def coefficient(self, o):
        #print("\nVisiting Coefficient: " + repr(o))

        # Map o to object with proper element and count
        o = self._function_replace_map[o]

        # Splat restriction for Real coefficients (it has no meaning anyway)
        if self.restriction and o.element().family() == "Real":
            self.restriction = None
        # Create aux. info.
        components = self.component()
        derivatives = self.derivatives()

        # Check if function is already in cache
        key = (o, components, derivatives, self.restriction, self.avg)
        function_code = self.function_cache.get(key)

        # FIXME: Why does using a code dict from cache make the expression manipulations blow (MemoryError) up later?
        # PyOP2 format always uses optimised quadrature transformer,
        # hence we must take this route
        if function_code is None or self.parameters["format"] == "pyop2" or self.optimise_parameters["optimisation"]:
            # Get auxiliary variables to generate function
            (component, local_elem,
             ffc_element, transformation) = self._get_auxiliary_variables(o, components)

            # Check that we don't take derivatives of QuadratureElements.
            is_quad_element = local_elem.family() == "Quadrature"
            ffc_assert(not (derivatives and is_quad_element), \
                       "Derivatives of Quadrature elements are not supported: " + repr(o))

            tdim = self.tdim # FIXME: o.domain().topological_dimension() ???

            # Create code for function and add empty tuple to cache dict.
            function_code = {(): self.create_function(o, derivatives, component,
                                                      ffc_element, is_quad_element,
                                                      transformation, tdim, self.avg)}

            self.function_cache[key] = function_code

        return function_code

    # -------------------------------------------------------------------------
    # SpatialCoordinate (geometry.py).
    # -------------------------------------------------------------------------
    def spatial_coordinate(self, o):
        #print "\n\nVisiting SpatialCoordinate:", repr(o)
        #print "\n\nVisiting SpatialCoordinate:", repr(operands)

        # Get the component.
        components = self.component()
        c, = components

        if self.vertex is not None:
            error("Spatial coordinates (x) not implemented for point measure (dP)") # TODO: Implement this, should be just the point.
        else:
            # Generate the appropriate coordinate and update tables.
            coordinate = format["ip coordinates"](self.points, c)
            self._generate_affine_map()
            return self._create_symbol(coordinate, IP)

    # -------------------------------------------------------------------------
    # Indexed (indexed.py).
    # -------------------------------------------------------------------------
    def indexed(self, o):
        #print("\n\nVisiting Indexed:" + repr(o))

        # Get indexed expression and index, map index to current value
        # and update components
        indexed_expr, index = o.ufl_operands
        self._components.push(self.visit(index))

        # Visit expression subtrees and generate code.
        code = self.visit(indexed_expr)

        # Remove component again
        self._components.pop()

        return code

    # -------------------------------------------------------------------------
    # MultiIndex (indexing.py).
    # -------------------------------------------------------------------------
    def multi_index(self, o):
        #print("\n\nVisiting MultiIndex:" + repr(o))

        # Loop all indices in MultiIndex and get current values
        subcomp = []
        for i in o:
            if isinstance(i, FixedIndex):
                subcomp.append(i._value)
            elif isinstance(i, Index):
                subcomp.append(self._index2value[i])

        return tuple(subcomp)

    # -------------------------------------------------------------------------
    # IndexSum (indexsum.py).
    # -------------------------------------------------------------------------
    def index_sum(self, o):
        #print("\n\nVisiting IndexSum: " + str(tree_format(o)))

        # Get expression and index that we're summing over
        summand, multiindex = o.ufl_operands
        index, = multiindex

        # Loop index range, update index/value dict and generate code
        ops = []
        for i in range(o.dimension()):
            self._index2value.push(index, i)
            ops.append(self.visit(summand))
            self._index2value.pop()

        # Call sum to generate summation
        code = self.sum(o, *ops)

        return code

    # -------------------------------------------------------------------------
    # MathFunctions (mathfunctions.py).
    # -------------------------------------------------------------------------
    def sqrt(self, o, *operands):
        #print("\n\nVisiting Sqrt: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["sqrt"][self.parameters["format"]])

    def exp(self, o, *operands):
        #print("\n\nVisiting Exp: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["exp"][self.parameters["format"]])

    def ln(self, o, *operands):
        #print("\n\nVisiting Ln: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["ln"][self.parameters["format"]])

    def cos(self, o, *operands):
        #print("\n\nVisiting Cos: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["cos"][self.parameters["format"]])

    def sin(self, o, *operands):
        #print("\n\nVisiting Sin: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["sin"][self.parameters["format"]])

    def tan(self, o, *operands):
        #print("\n\nVisiting Tan: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["tan"][self.parameters["format"]])

    def cosh(self, o, *operands):
        #print("\n\nVisiting Cosh: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["cosh"][self.parameters["format"]])

    def sinh(self, o, *operands):
        #print("\n\nVisiting Sinh: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["sinh"][self.parameters["format"]])

    def tanh(self, o, *operands):
        #print("\n\nVisiting Tanh: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["tanh"][self.parameters["format"]])

    def acos(self, o, *operands):
        #print("\n\nVisiting Acos: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["acos"][self.parameters["format"]])

    def asin(self, o, *operands):
        #print("\n\nVisiting Asin: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["asin"][self.parameters["format"]])

    def atan(self, o, *operands):
        #print("\n\nVisiting Atan: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["atan"][self.parameters["format"]])

    def atan_2(self, o, *operands):
        #print("\n\nVisiting Atan2: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        self.additional_includes_set.add("#include <cmath>")
        return self._atan_2_function(operands, format["atan_2"])

    def erf(self, o, *operands):
        #print("\n\nVisiting Erf: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        return self._math_function(operands, format["erf"][self.parameters["format"]])

    def bessel_i(self, o, *operands):
        #print("\n\nVisiting Bessel_I: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        #self.additional_includes_set.add("#include <tr1/cmath>")
        self.additional_includes_set.add("#include <boost/math/special_functions.hpp>")
        return self._bessel_function(operands, format["bessel_i"])

    def bessel_j(self, o, *operands):
        #print("\n\nVisiting Bessel_J: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        #self.additional_includes_set.add("#include <tr1/cmath>")
        self.additional_includes_set.add("#include <boost/math/special_functions.hpp>")
        return self._bessel_function(operands, format["bessel_j"])

    def bessel_k(self, o, *operands):
        #print("\n\nVisiting Bessel_K: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        #self.additional_includes_set.add("#include <tr1/cmath>")
        self.additional_includes_set.add("#include <boost/math/special_functions.hpp>")
        return self._bessel_function(operands, format["bessel_k"])

    def bessel_y(self, o, *operands):
        #print("\n\nVisiting Bessel_Y: " + repr(o) + "with operands: " + "\n".join(map(repr,operands)))
        #self.additional_includes_set.add("#include <tr1/cmath>")
        self.additional_includes_set.add("#include <boost/math/special_functions.hpp>")
        return self._bessel_function(operands, format["bessel_y"])

    # -------------------------------------------------------------------------
    # PositiveRestricted and NegativeRestricted (restriction.py).
    # -------------------------------------------------------------------------
    def positive_restricted(self, o):
        #print("\n\nVisiting PositiveRestricted: " + repr(o))

        # Just get the first operand, there should only be one.
        restricted_expr = o.ufl_operands
        ffc_assert(len(restricted_expr) == 1, "Only expected one operand for restriction: " + repr(restricted_expr))
        ffc_assert(self.restriction is None, "Expression is restricted twice: " + repr(restricted_expr))

        # Set restriction, visit operand and reset restriction
        self.restriction = "+"
        code = self.visit(restricted_expr[0])
        self.restriction = None

        return code

    def negative_restricted(self, o):
        #print("\n\nVisiting NegativeRestricted: " + repr(o))

        # Just get the first operand, there should only be one.
        restricted_expr = o.ufl_operands
        ffc_assert(len(restricted_expr) == 1, "Only expected one operand for restriction: " + repr(restricted_expr))
        ffc_assert(self.restriction is None, "Expression is restricted twice: " + repr(restricted_expr))

        # Set restriction, visit operand and reset restriction
        self.restriction = "-"
        code = self.visit(restricted_expr[0])
        self.restriction = None

        return code

    def cell_avg(self, o):
        ffc_assert(self.avg is None, "Not expecting nested averages.")

        # Just get the first operand, there should only be one.
        expr, = o.ufl_operands

        # Set average marker, visit operand and reset marker
        self.avg = "cell"
        code = self.visit(expr)
        self.avg = None

        return code

    def facet_avg(self, o):
        ffc_assert(self.avg is None, "Not expecting nested averages.")
        ffc_assert(self.entity_type != "cell", "Cannot take facet_avg in a cell integral.")

        # Just get the first operand, there should only be one.
        expr, = o.ufl_operands

        # Set average marker, visit operand and reset marker
        self.avg = "facet"
        code = self.visit(expr)
        self.avg = None

        return code

    # -------------------------------------------------------------------------
    # ComponentTensor (tensors.py).
    # -------------------------------------------------------------------------
    def component_tensor(self, o):
        #print("\n\nVisiting ComponentTensor:\n" + str(tree_format(o)))

        # Get expression and indices
        component_expr, indices = o.ufl_operands

        # Get current component(s)
        components = self.component()

        ffc_assert(len(components) == len(indices), \
                   "The number of known components must be equal to the number of components of the ComponentTensor for this to work.")

        # Update the index dict (map index values of current known indices to
        # those of the component tensor)
        for i, v in zip(indices._indices, components):
            self._index2value.push(i, v)

        # Push an empty component tuple
        self._components.push(())

        # Visit expression subtrees and generate code.
        code = self.visit(component_expr)

        # Remove the index map from the StackDict
        for i in range(len(components)):
            self._index2value.pop()

        # Remove the empty component tuple
        self._components.pop()

        return code

    def list_tensor(self, o):
        #print("\n\nVisiting ListTensor: " + repr(o))

        # Get the component
        component = self.component()

        # Extract first and the rest of the components
        c0, c1 = component[0], component[1:]

        # Get first operand
        op = o.ufl_operands[c0]

        # Evaluate subtensor with this subcomponent
        self._components.push(c1)
        code = self.visit(op)
        self._components.pop()

        return code

    # -------------------------------------------------------------------------
    # Variable (variable.py).
    # -------------------------------------------------------------------------
    def variable(self, o):
        #print("\n\nVisiting Variable: " + repr(o))
        # Just get the expression associated with the variable
        return self.visit(o.expression())

    # -------------------------------------------------------------------------
    # Generate terms for representation.
    # -------------------------------------------------------------------------
    def generate_terms(self, integrand, integral_type):
        "Generate terms for code generation."

        # Set domain type
        self.integral_type = integral_type

        # Get terms
        terms = self.visit(integrand)

        # Get formatting
        f_nzc = format["nonzero columns"](0).split("0")[0]

        # Loop code and add weight and scale factor to value and sort after
        # loop ranges.
        new_terms = {}
        if len(terms) == 0 and self.parameters["format"] == "pyop2":
            # Integrand simplified to zero (i.e. empty) so raise
            # exception (Firedrake catches this later)
            raise EmptyIntegrandError('Integrand %s is empty' % integrand)
        for key, val in sorted(terms.items()):
            # If value was zero continue.
            if val is None or (self.parameters["format"] == "pyop2" and val.val == 0.0):
                continue
            # Create data.
            value, ops, sets = self._create_entry_data(val, integral_type)
            # Extract nzc columns if any and add to sets.
            used_nzcs = set([int(k[1].split(f_nzc)[1].split("[")[0]) for k in key if f_nzc in k[1]])
            sets.append(used_nzcs)

            # Create loop information and entry from key info and insert into dict.
            loop, entry = self._create_loop_entry(key, f_nzc)
            if not loop in new_terms:
                sets.append({})
                new_terms[loop] = [sets, [(entry, value, ops)]]
            else:
                for i, s in enumerate(sets):
                    new_terms[loop][0][i].update(s)
                new_terms[loop][1].append((entry, value, ops))

        if len(new_terms) == 0 and self.parameters["format"] == "pyop2":
            # Integrand simplified to zero (i.e. empty) so raise
            # exception (Firedrake catches this later)
            raise EmptyIntegrandError('Integrand %s is empty' % integrand)
        return new_terms

    def _create_loop_entry(self, key, f_nzc):

        indices = {0: format["first free index"],  1: format["second free index"]}

        # Create appropriate entries.
        # FIXME: We only support rank 0, 1 and 2.
        entry = ""
        loop = ()
        if len(key) == 0:
            entry = "0"
        elif len(key) == 1:
            key = key[0]
            # Checking if the basis was a test function.
            # TODO: Make sure test function indices are always rearranged to 0.
            ffc_assert(key[0] == -2 or key[0] == 0, \
                        "Linear forms must be defined using test functions only: " + repr(key))
            index_j, entry, range_j, space_dim_j = key
            loop = ((indices[index_j], 0, range_j),)
            if range_j == 1 and self.optimise_parameters["ignore ones"] and not (f_nzc in entry):
                loop = ()
            if self.parameters["format"] == "pyop2":
                entry = (entry, )
        elif len(key) == 2:
            # PyOP2 mixed element assembling a rank-1 form
            if key[0][0]==key[1][0]:
                key = (key[0], key[1])
                for k in key:
                    ffc_assert(key[0][0] == -2 or key[0][0] == 0, \
                    "Linear forms must be defined using test functions only: " + repr(key))

                index_j, entry_j, range_j, space_dim_j = key[0]
                index_r, entry_r, range_r, space_dim_r = key[1]
                entry = (entry_r)
                loop = ((entry_r, 0, range_r),)
            else:
                # Extract test and trial loops in correct order and check if for is legal.
                key0, key1 = (0, 0)
                for k in key:
                    ffc_assert(k[0] in indices, \
                    "Bilinear forms must be defined using test and trial functions (index -2, -1, 0, 1): " + repr(k))
                    if k[0] == -2 or k[0] == 0:
                        key0 = k
                    else:
                        key1 = k
                index_j, entry_j, range_j, space_dim_j = key0
                index_k, entry_k, range_k, space_dim_k = key1

                loop = []
                if not (range_j == 1 and self.optimise_parameters["ignore ones"]) or f_nzc in entry_j:
                    loop.append((indices[index_j], 0, range_j))
                if not (range_k == 1 and self.optimise_parameters["ignore ones"]) or f_nzc in entry_k:
                    loop.append((indices[index_k], 0, range_k))
                if self.parameters["format"] == "pyop2":
                    entry = (entry_j, entry_k)
                else:
                    entry = format["add"]([format["mul"]([entry_j, str(space_dim_k)]), entry_k])
                loop = tuple(loop)
        elif len(key) == 4:
            # PyOP2 mixed element case only.
            key0, key1 = ([], [])
            for k in key:
                ffc_assert(k[0] in indices, \
                "Bilinear forms must be defined using test and trial functions (index -2, -1, 0, 1): " + repr(k))
                if k[0] == -2 or k[0] == 0:
                    key0.append(k)
                else:
                    key1.append(k)

            index_j, entry_j, range_j, space_dim_j = key0[0]
            index_r, entry_r, range_r, space_dim_r = key0[1]
            index_k, entry_k, range_k, space_dim_k = key1[0]
            index_s, entry_s, range_s, space_dim_s = key1[1]

            entry = (entry_r, entry_s)

            loop = ((entry_j, 0, range_j), (entry_r, 0, range_r), (entry_k, 0, range_k), (entry_s, 0, range_s))
        else:
            error("Only rank 0, 1 and 2 tensors are currently supported: " + repr(key))
        # Generate the code line for the entry.
        # Try to evaluate entry ("3*6 + 2" --> "20").
        try:
            entry = str(eval(entry))
        except:
            pass
        return loop, entry

    # -------------------------------------------------------------------------
    # Helper functions for transformation of UFL objects in base class
    # -------------------------------------------------------------------------
    def _create_symbol(self, symbol, domain):
        error("This function should be implemented by the child class.")

    def _create_product(self, symbols):
        error("This function should be implemented by the child class.")

    def _format_scalar_value(self, value):
        error("This function should be implemented by the child class.")

    def _math_function(self, operands, format_function):
        error("This function should be implemented by the child class.")

    def _atan_2_function(self, operands, format_function):
        error("This function should be implemented by the child class.")

    def _get_auxiliary_variables(self,
                                 ufl_function,
                                 component):
        "Helper function for both Coefficient and Argument."

        # Get UFL element.
        ufl_element = ufl_function.element()

        # Get subelement and the relative (flattened) component (in case we have mixed elements).
        local_comp, local_elem = ufl_element.extract_component(component)
        ffc_assert(len(local_comp) <= 1, "Assuming there are no tensor-valued basic elements.")
        local_comp = local_comp[0] if local_comp else 0

        # Check that component != not () since the UFL component map will turn
        # it into 0, and () does not mean zeroth component in this context.
        if len(component):
            # Map component using component map from UFL. (TODO: inefficient use of this function)
            comp_map, comp_num = build_component_numbering(ufl_element.value_shape(), ufl_element.symmetry())
            component = comp_map[component]

        # Create FFC element.
        ffc_element = create_element(ufl_element)

        # Assuming that mappings for all basisfunctions are equal
        ffc_sub_element = create_element(local_elem)
        transformation = ffc_sub_element.mapping()[0]
        ffc_assert(all(transformation == mapping for mapping in ffc_sub_element.mapping()),
                   "Assuming subelement mappings are equal but they differ.")
        return (component, local_elem, ffc_element, transformation)

    def _get_current_entity(self):
        if self.entity_type == "cell":
            # If we add macro cell integration, I guess the 'current cell number' would go here?
            return 0
        elif self.entity_type in ("facet", "horiz_facet", "vert_facet"):
            # Handle restriction through facet.
            return {"+": self.facet0, "-": self.facet1, None: self.facet0}[self.restriction]
        elif self.entity_type == "vertex":
            return self.vertex
        else:
            error("Unknown entity type %s." % self.entity_type)

    def _create_mapping_basis(self, component, deriv, avg, ufl_argument, ffc_element):
        "Create basis name and mapping from given basis_info."

        # Get string for integration points.
        f_ip = "0" if (avg or self.points == 1) else format["integration points"]
        generate_psi_name = format["psi name"]

        # Only support test and trial functions.
        indices = {0: format["first free index"],
                   1: format["second free index"]}

        # Check that we have a basis function.
        ffc_assert(ufl_argument.number() in indices,
                   "Currently, Argument number must be either 0 or 1: " + repr(ufl_argument))
        ffc_assert(ufl_argument.part() is None,
                   "Currently, Argument part is not supporte: " + repr(ufl_argument))

        # Get element counter and loop index.
        element_counter = self.element_map[1 if avg else self.points][ufl_argument.element()]
        loop_index = indices[ufl_argument.number()]

        # Offset element space dimension in case of negative restriction,
        # need to use the complete element for offset in case of mixed element.
        space_dim = ffc_element.space_dimension()
        offset = {"+": "", "-": str(space_dim), None: ""}[self.restriction]

        # If we are in PyOP2 mode with mixed elements on interior facets, the
        # offsets are different; see below for fuller explanation
        self.mixed_elt_int_facet_mode = self.parameters["pyop2-ir"] and \
            isinstance(ffc_element, MixedElement) and \
            self.restriction in ["+", "-"]

        if self.mixed_elt_int_facet_mode:
            # Exposition: see below
            loop_index_range = [e.space_dimension() for e in ffc_element._elements]
            if self.restriction == "+":
                offset = ["0"]  # needs to be "0" rather than "", since we won't
                                # try to eval the string below
                cur = 0
                for e in ffc_element._elements:
                    cur += 2*e.space_dimension()
                    offset.append(str(cur))
                offset.pop()
            if self.restriction == "-":
                offset = []
                cur = 0
                for e in ffc_element._elements:
                    cur += e.space_dimension()
                    offset.append(str(cur))
                    cur += e.space_dimension()

        # If we have a restricted function multiply space_dim by two.
        if self.restriction in ("+", "-"):
            space_dim *= 2

        # Get current cell entity, with current restriction considered
        entity = self._get_current_entity()
        name = generate_psi_name(element_counter, self.entity_type, component, deriv, avg)
        name, non_zeros, zeros, ones = self.name_map[name]
        # don't overwrite this if we set it already
        if not self.mixed_elt_int_facet_mode:
            loop_index_range = shape(self.unique_tables[name])[-1]

        index_func = _make_index_function(shape(self.unique_tables[name]), entity, f_ip)

        # Create basis access, we never need to map the entry in the basis table
        # since we will either loop the entire space dimension or the non-zeros.
        # NOT TRUE FOR MIXED-ELT-INT-FACET MODE
        if self.mixed_elt_int_facet_mode:
            index_calc = []
            cur = 0
            for e in ffc_element._elements:
                index_calc.append(Access(loop_index, str(cur)))
                cur += e.space_dimension()
            basis_access = [format["component"]("", index_func(bi)) for bi in index_calc]
        else:
            if self.points == 1:
                f_ip = "0"
            index_calc = loop_index
            if self.restriction in ("+", "-") and self.integral_type == "custom" and offset != "":
                # Special case access for custom integrals (all basis functions stored in flattened array)
                basis_access = format["component"]("", index_func(format["add"]([loop_index, offset])))
                index_calc = Access(loop_index, offset)
            else:
                # Normal basis function access
                basis_access = format["component"]("", index_func(loop_index))

        # If domain type is custom, then special-case set loop index
        # range since table is empty
        if self.integral_type == "custom":
            loop_index_range = ffc_element.space_dimension() # different from `space_dimension`...

        basis = ""
        # Ignore zeros if applicable
        if zeros and (self.optimise_parameters["ignore zero tables"] or self.optimise_parameters["remove zero terms"]):
            basis = self._format_scalar_value(None)[()]
        # If the loop index range is one we can look up the first component
        # in the psi array. If we only have ones we don't need the basis.
        elif self.optimise_parameters["ignore ones"] and loop_index_range == 1 and ones:
            loop_index = "0"
            basis = self._format_scalar_value(1.0)[()]
        else:
            # Add basis name to the psi tables map for later use.
            if self.mixed_elt_int_facet_mode:
                basis = [self._create_symbol(name + ba, BASIS, index_func(ic), _iden=name)[()] for ba, ic in zip(basis_access, index_calc)]
                for ba in basis:
                    self.psi_tables_map[ba] = name
            else:
                basis = self._create_symbol(name + basis_access, BASIS, index_func(index_calc), _iden=name)[()]
                self.psi_tables_map[basis] = name

        # Create the correct mapping of the basis function into the local element tensor.
        basis_map = Access(loop_index)
        if non_zeros and basis_map == "0":
            basis_map = Access(str(non_zeros[1][0]))
        elif non_zeros:
            basis_map = Access(format["component"](format["nonzero columns"](non_zeros[0]), basis_map))
        if offset:
            if self.mixed_elt_int_facet_mode:
                basis_map = [Access(basis_map.loop_index, o) for o in offset]
            else:
                basis_map = Access(basis_map.loop_index, offset)

        # Try to evaluate basis map ("3 + 2" --> "5").
        try:
            basis_map = str(eval(basis_map))
        except:
            pass

        # Create mapping (index, map, loop_range, space_dim).
        # Example dx and ds: (0, j, 3, 3)
        # Example dS: (0, (j + 3), 3, 6), 6=2*space_dim
        # Example dS optimised: (0, (nz2[j] + 3), 2, 6), 6=2*space_dim
        if self.mixed_elt_int_facet_mode:
            mapping = [((ufl_argument.number(), bm, lir, space_dim),) for bm, lir in zip(basis_map, loop_index_range)]
        else:
            mapping = ((ufl_argument.number(), basis_map, loop_index_range, space_dim),)

        return (mapping, basis)

    def _create_function_name(self, component, deriv, avg, is_quad_element, ufl_function, ffc_element):
        ffc_assert(ufl_function in self._function_replace_values,
                   "Expecting ufl_function to have been mapped prior to this call.")

        # Get format
        p_format = self.parameters["format"]

        # Get string for integration points.
        f_ip = "0" if (avg or self.points == 1) else format["integration points"]

        # Get the element counter.
        element_counter = self.element_map[1 if avg else self.points][ufl_function.element()]

        # Get current cell entity, with current restriction considered
        entity = self._get_current_entity()

        # Set to hold used nonzero columns
        used_nzcs = set()

        # Create basis name and map to correct basis and get info.
        generate_psi_name = format["psi name"]
        psi_name = generate_psi_name(element_counter, self.entity_type, component, deriv, avg)
        psi_name, non_zeros, zeros, ones = self.name_map[psi_name]

        # If all basis are zero we just return None.
        if zeros and self.optimise_parameters["ignore zero tables"]:
            return self._format_scalar_value(None)[()]

        # Get the index range of the loop index.
        loop_index_range = shape(self.unique_tables[psi_name])[-1]

        # Drop integration point if function is cellwise constant
        unique_table = self.unique_tables[psi_name]
        if allclose(unique_table, unique_table.mean(axis=-2, keepdims=True)):
            f_ip = "0"

        index_func = _make_index_function(shape(unique_table), entity, f_ip)

        # If domain type is custom, then special-case set loop index
        # range since table is empty
        if self.integral_type == "custom":
            loop_index_range = ffc_element.space_dimension()

        # Create loop index
        if loop_index_range > 1:
            # Pick first free index of secondary type
            # (could use primary indices, but it's better to avoid confusion).
            loop_index = format["free indices"][0]

        # If we have a quadrature element we can use the ip number to look
        # up the value directly. Need to add offset in case of components.
        if is_quad_element:
            quad_offset = 0
            if component:
                # FIXME: Should we add a member function elements() to FiniteElement?
                if isinstance(ffc_element, MixedElement):
                    for i in range(component):
                        quad_offset += ffc_element.elements()[i].space_dimension()
                elif component != 1:
                    error("Can't handle components different from 1 if we don't have a MixedElement.")
                else:
                    quad_offset += ffc_element.space_dimension()
            if quad_offset:
                coefficient_access = format["add"]([f_ip, str(quad_offset)])
            else:
                if non_zeros and f_ip == "0":
                    # If we have non zero column mapping but only one value just pick it.
                    # MSA: This should be an exact refactoring of the previous logic,
                    #      but I'm not sure if these lines were originally intended
                    #      here in the quad_element section, or what this even does:
                    coefficient_access = str(non_zeros[1][0])
                else:
                    coefficient_access = f_ip

        elif non_zeros:
            if loop_index_range == 1:
                # If we have non zero column mapping but only one value just pick it.
                coefficient_access = str(non_zeros[1][0])
            else:
                used_nzcs.add(non_zeros[0])
                coefficient_access = format["component"](format["nonzero columns"](non_zeros[0]), loop_index)

        elif loop_index_range == 1:
            # If the loop index range is one we can look up the first component
            # in the coefficient array.
            coefficient_access = "0"

        else:
            # Or just set default coefficient access.
            coefficient_access = loop_index

        # Offset by element space dimension in case of negative restriction.
        offset = {"+": "", "-": str(ffc_element.space_dimension()), None: ""}[self.restriction]

        # If we are in PyOP2 mode with mixed elements on interior facets, the
        # offsets are totally different, since the coefficient passed in to the
        # kernel contains data for both cells, interleaved. This means we need
        # to splat the offset and loop_index_range variables and replace them
        # with something appropriate for our data layout.
        self.mixed_elt_int_facet_mode = self.parameters["pyop2-ir"] and \
            isinstance(ffc_element, MixedElement) and \
            self.restriction in ["+", "-"]

        if self.mixed_elt_int_facet_mode:
            # Exposition: suppose our mixed element has 6 dofs in the first
            # sub-element, 3 in the second, and 4 in the third.  The data looks
            # like 0-5 | 6-11 | 12-14 | 15-17 | 18-21 | 22-25
            # If restriction is "+", we want the offsets to be 0, 12, 18
            # If restriction is "-", we want the offsets to be 6, 15, 22
            loop_index_range = [e.space_dimension() for e in ffc_element._elements]
            if self.restriction == "+":
                offset = ["0"]  # needs to be "0" rather than "", since we won't
                                # try to eval the string below
                cur = 0
                for e in ffc_element._elements:
                    cur += 2*e.space_dimension()
                    offset.append(str(cur))
                offset.pop()
            if self.restriction == "-":
                offset = []
                cur = 0
                for e in ffc_element._elements:
                    cur += e.space_dimension()
                    offset.append(str(cur))
                    cur += e.space_dimension()

        if self.mixed_elt_int_facet_mode:
            coefficient_access = [Access(coefficient_access, o) for o in offset]
        else:
            if offset:
                coefficient_access = Access(coefficient_access, offset)

        # Try to evaluate coefficient access ("3 + 2" --> "5").
        try:
            coefficient_access = str(eval(coefficient_access))
            C_ACCESS = GEO
        except:
            # Guaranteed to fail in "borkmode", i.e. mixed_elt_int_facet_mode
            C_ACCESS = IP

        # Format coefficient access
        if self.mixed_elt_int_facet_mode:
            coefficient = [format["coefficient"][p_format](str(ufl_function.count()), ca) for ca in coefficient_access]
            coefficient_name = [format["coefficient"][p_format](str(ufl_function.count())) for ca in coefficient_access]
            coefficient_li = [[ca, 0] for ca in coefficient_access]
        else:
            coefficient = format["coefficient"][p_format](str(ufl_function.count()), coefficient_access)
            coefficient_name = format["coefficient"][p_format](str(ufl_function.count()))
            coefficient_li = [coefficient_access, 0] 

        # Build and cache some function data only if we need the basis
        # MSA: I don't understand the mix of loop index range check and ones check here, but that's how it was.
        if is_quad_element or (loop_index_range == 1 and ones and self.optimise_parameters["ignore ones"]):
            # If we only have ones or if we have a quadrature element we don't need the basis.
            function_symbol_name = coefficient
            F_ACCESS = C_ACCESS

        else:
            # Add basis name to set of used tables and add matrix access.
            # TODO: We should first add this table if the function is used later
            # in the expressions. If some term is multiplied by zero and it falls
            # away there is no need to compute the function value
            self.used_psi_tables.add(psi_name)

            # Create basis access, we never need to map the entry in the basis
            # table since we will either loop the entire space dimension or the
            # non-zeros.
            # Edit: unfortunately this is not true if we are in m_e_i_f_mode!
            if self.mixed_elt_int_facet_mode:
                basis_index = []
                cur = 0
                for e in ffc_element._elements:
                    basis_index.append(Access(loop_index, str(cur)))
                    cur += e.space_dimension()
                basis_li = [index_func(bi) for bi in basis_index]
                basis_access = [format["component"]("", bi) for bi in basis_li]
                basis_name = [psi_name + ba for ba in basis_access]
            else:
                basis_index = "0" if loop_index_range == 1 else loop_index
                basis_li = index_func(basis_index)
                basis_access = format["component"]("", basis_li)
                basis_name = psi_name + basis_access
            # Try to set access to the outermost possible loop
            if f_ip == "0" and basis_access == "0":
                B_ACCESS = GEO
                F_ACCESS = C_ACCESS
            else:
                B_ACCESS = IP
                F_ACCESS = IP

            # Format expression for function
            if self.mixed_elt_int_facet_mode:
                function_expr = [self._create_product(\
                           [self._create_symbol(b, B_ACCESS, _iden=psi_name, _loop_index=bli)[()], \
                            self._create_symbol(c, C_ACCESS, _iden=cn, _loop_index=cli)[()]]) \
                            for b, bli, c, cn, cli in zip(basis_name, basis_li, coefficient,
                                                          coefficient_name, coefficient_li)]
            else:
                function_expr = self._create_product(\
                            [self._create_symbol(basis_name, B_ACCESS, _iden=psi_name,
                                                 _loop_index=basis_li)[()], \
                             self._create_symbol(coefficient, C_ACCESS, _iden=coefficient_name,
                                                 _loop_index=coefficient_li)[()]])
             
            # Check if the expression to compute the function value is already in
            # the dictionary of used function. If not, generate a new name and add.
            if self.mixed_elt_int_facet_mode:
                function_count = len(self.function_data)
                for fe, lir in zip(function_expr, loop_index_range):
                    data = self.function_data.get(fe)
                    if data is None:
                        data = Bunch(id=function_count,
                                     cellwise_constant=(f_ip == "0"),
                                     loop_range=lir,
                                     ops=self._count_operations(fe),
                                     psi_name=psi_name,
                                     used_nzcs=used_nzcs,
                                     ufl_element=ufl_function.element())
                        self.function_data[fe] = data
            else:
                data = self.function_data.get(function_expr)
                if data is None:
                    function_count = len(self.function_data)
                    data = Bunch(id=function_count,
                                 cellwise_constant=(f_ip == "0"),
                                 loop_range=loop_index_range,
                                 ops=self._count_operations(function_expr),
                                 psi_name=psi_name,
                                 used_nzcs=used_nzcs,
                                 ufl_element=ufl_function.element())
                    self.function_data[function_expr] = data

            function_symbol_name = format["function value"](data.id)

        # TODO: This access stuff was changed subtly during my refactoring, the
        # X_ACCESS vars is an attempt at making it right, make sure it is correct now!
        return self._create_symbol(function_symbol_name, F_ACCESS, _iden=function_symbol_name)[()]

    def _generate_affine_map(self):
        """Generate psi table for affine map, used by spatial coordinate to map
        integration point to physical element."""

        # TODO: KBO: Perhaps it is better to create a fiat element and tabulate
        # the values at the integration points?
        f_FEA = format["affine map table"]
        f_ip  = format["integration points"]

        affine_map = {1: lambda x: [1.0 - x[0],               x[0]],
                      2: lambda x: [1.0 - x[0] - x[1],        x[0], x[1]],
                      3: lambda x: [1.0 - x[0] - x[1] - x[2], x[0], x[1], x[2]]}

        num_ip = self.points
        w, points = self.quad_weights[num_ip]

        if self.facet0 is not None:
            points = map_facet_points(points, self.facet0, self.entity_type)
            name = f_FEA(num_ip, self.facet0)
        elif self.vertex is not None:
            error("Spatial coordinates (x) not implemented for point measure (dP)") # TODO: Implement this, should be just the point.
            #name = f_FEA(num_ip, self.vertex)
        else:
            name = f_FEA(num_ip, 0)

        if name not in self.unique_tables:
            self.unique_tables[name] = array([affine_map[len(p)](p) for p in points])

        if self.coordinate is None:
            ip = f_ip if num_ip > 1 else 0
            r = None if self.facet1 is None else "+"
            self.coordinate = [name, self.gdim, ip, r]

    # -------------------------------------------------------------------------
    # Helper functions for code_generation()
    # -------------------------------------------------------------------------
    def _count_operations(self, expression):
        error("This function should be implemented by the child class.")

    def _create_entry_data(self, val):
        error("This function should be implemented by the child class.")


def _make_index_function(shape, entity, ip):
    if len(shape) == 2:
        def f(dof):
            return [ip, dof]
    elif len(shape) == 3:
        def f(dof):
            return [entity, ip, dof]
    else:
        raise RuntimeError("Tensor of dimension 2 or 3 expected.")
    return f


class Access():
    """Class to represent an access into a basis function array or similar.
    Stores the loop index (letter) and offset (number) separately, but
    pretends to be a string such as "j + 3" when needed"""

    def __init__(self, loop_index, offset=None):
        self.loop_index = loop_index
        try:
            self.offset = int(offset)
        except:
            self.offset = offset or 0
        if offset:
            self.access = format["grouping"](format["add"]([loop_index, offset]))
        else:
            self.access = loop_index

    def __getattr__(self, m):
        return getattr(self.access, m)

    def __eq__(self, other):
        return self.loop_index == other.loop_index and self.offset == other.offset
