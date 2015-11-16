# Copyright (C) 2009-2013 Kristian B. Oelgaard and Anders Logg
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
# Modified by Garth N. Wells, 2009.
# Modified by Marie Rognes, 2009-2013.
# Modified by Martin Alnaes, 2013
# Modified by Andrew T. T. McRae, 2013
# Modified by Lizao Li, 2015

# Python modules
from numpy import array, asarray, polymul, zeros, ones
import six
import weakref

# UFL and FIAT modules
import ufl
from ufl.utils.sorting import sorted_by_key
import FIAT

from FIAT.trace import DiscontinuousLagrangeTrace

# FFC modules
from ffc.log import debug, error, ffc_assert
from ffc.quadratureelement import QuadratureElement as FFCQuadratureElement


from ffc.mixedelement import MixedElement
from ffc.restrictedelement import RestrictedElement
from ffc.enrichedelement import SpaceOfReals

# Dictionary mapping from cellname to dimension
from ufl.cell import cell2dim

# Element families supported by FFC
supported_families = ("Brezzi-Douglas-Marini",
                      "Brezzi-Douglas-Fortin-Marini",
                      "Crouzeix-Raviart",
                      "Discontinuous Lagrange",
                      "Discontinuous Taylor",
                      "Discontinuous Raviart-Thomas",
                      "Discontinuous Lagrange Trace",
                      "Lagrange",
                      "Lobatto",
                      "Nedelec 1st kind H(curl)",
                      "Nedelec 2nd kind H(curl)",
                      "Q",
                      "DQ",
                      "Radau",
                      "Raviart-Thomas",
                      "Real",
                      "RTCE",
                      "RTCF",
                      "Bubble",
                      "Quadrature",
                      "OuterProductElement",
                      "EnrichedElement",
                      "BrokenElement",
                      "TraceElement",
                      "FacetElement",
                      "InteriorElement",
                      "Regge")

# Cache for computed elements

_cache = weakref.WeakKeyDictionary()

# Quadrilateral OuterProductCell
_quad_opc = ufl.OuterProductCell(ufl.Cell("interval"), ufl.Cell("interval"))

def reference_cell(cell):
    # really want to be using cells only, but sometimes only cellname is passed
    # in. FIAT handles the cases.
    
    # I hope nothing is still passing in just dimension...
    if isinstance(cell, int):
        error("%s was passed into reference_cell(). Need cell or cellname." % str(cell))

    return FIAT.ufc_cell(cell)

def reference_cell_vertices(cellname):
    "Return dict of coordinates of reference cell vertices for this 'dim'."
    cell = reference_cell(cellname)
    return cell.get_vertices()

def create_element(ufl_element):

    # Create element signature for caching (just use UFL element)
    element_signature = ufl_element

    # Check cache
    element = _cache.get(element_signature, None)
    if element:
        debug("Reusing element from cache")
        return element

    if isinstance(ufl_element, ufl.MixedElement):
        # Create mixed element (implemented by FFC)
        elements = _extract_elements(ufl_element)
        element = MixedElement(elements)
    elif isinstance(ufl_element, ufl.RestrictedElement):
        # Create restricted element(implemented by FFC)
        element = _create_restricted_element(ufl_element)
    elif isinstance(ufl_element, (ufl.FiniteElement, ufl.OuterProductElement, ufl.EnrichedElement, ufl.BrokenElement, ufl.TraceElement, ufl.FacetElement, ufl.InteriorElement)):
        # Create regular FIAT finite element
        element = _create_fiat_element(ufl_element)
    else:
        error("Cannot handle this element type: %s" % str(ufl_element))

    # Store in cache
    _cache[element_signature] = element

    return element

def _create_fiat_element(ufl_element):
    "Create FIAT element corresponding to given finite element."

    # Get element data
    family = ufl_element.family()
    domain, = ufl_element.domains() # Assuming single domain
    cellname = domain.cell().cellname() # Assuming single cell in domain
    degree = ufl_element.degree()

    # Check that FFC supports this element
    ffc_assert(family in supported_families,
               "This element family (%s) is not supported by FFC." % family)

    # Handle the space of the constant
    if family == "Real":
        domain, = ufl_element.domains() # Assuming single domain

        if not isinstance(ufl_element.cell(), ufl.OuterProductCell):
            dg0_element = ufl.FiniteElement("DG", domain, 0)
        else:
            dg0_element_A = ufl.FiniteElement("DG", ufl_element.cell()._A, 0)
            dg0_element_B = ufl.FiniteElement("DG", ufl_element.cell()._B, 0)
            dg0_element = ufl.OuterProductElement(dg0_element_A, dg0_element_B).reconstruct(domain=domain)

        constant = _create_fiat_element(dg0_element)
        return SpaceOfReals(constant)

    # FIXME: AL: Should this really be here?
    # Handle QuadratureElement
    elif family == "Quadrature":
        return FFCQuadratureElement(ufl_element)

    else:
        return create_actual_fiat_element(ufl_element)

    raise Exception("Something strange happened: reached end of function without returning an element")


def create_actual_fiat_element(ufl_element):
    fiat_element = None

    # Check if finite element family is supported by FIAT
    family = ufl_element.family()
    if not family in FIAT.supported_elements:
        # We support RTCE and RTCF elements on quadrilaterals,
        # even though they are not supported by FIAT.
        if ufl_element.cell().cellname() == "quadrilateral":
            fiat_element = create_actual_fiat_element(ufl_element.reconstruct(domain=_quad_opc))
        else:
            if family in ("FacetElement", "InteriorElement"):
                # rescue these
                pass
            else:
                error("Sorry, finite element of type \"%s\" are not supported by FIAT.", family)

    # Skip all cases if FIAT element is ready already
    if fiat_element is not None:
        pass
    # HDiv and HCurl elements have family "OuterProductElement",
    # so get matching FIAT element directly rather than via lookup
    elif isinstance(ufl_element, ufl.HDiv):
        fiat_element = FIAT.Hdiv(create_element(ufl_element._element))
    elif isinstance(ufl_element, ufl.HCurl):
        fiat_element = FIAT.Hcurl(create_element(ufl_element._element))
    elif isinstance(ufl_element, ufl.FacetElement):
        fiat_element = FIAT.RestrictedElement(create_element(ufl_element._element), restriction_domain="facet")
    elif isinstance(ufl_element, ufl.InteriorElement):
        fiat_element = FIAT.RestrictedElement(create_element(ufl_element._element), restriction_domain="interior")
    else:
        # Look up FIAT element
        ElementClass = FIAT.supported_elements[family]

        if isinstance(ufl_element, ufl.EnrichedElement):
            A = create_element(ufl_element._elements[0])
            B = create_element(ufl_element._elements[1])
            fiat_element = ElementClass(A, B)
        # OPVE is only here to satisfy calls from Firedrake
        elif isinstance(ufl_element, (ufl.OuterProductElement, ufl.OuterProductVectorElement, ufl.OuterProductTensorElement)):
            domain, = ufl_element.domains() # Assuming single domain
            cell = domain.cell()            # Assuming single cell in domain
            if not isinstance(cell, ufl.OuterProductCell):
                error("An OuterProductElement must have an OuterProductCell as domain, sorry.")

            A = create_element(ufl_element._A)
            B = create_element(ufl_element._B)
            fiat_element = ElementClass(A, B)
        elif isinstance(ufl_element, (ufl.BrokenElement, ufl.TraceElement)):
            fiat_element = ElementClass(create_element(ufl_element._element))
        elif ufl_element.cell().cellname() == "quadrilateral":
            fiat_element = create_actual_fiat_element(ufl_element.reconstruct(domain=_quad_opc))
        else:
            # "Normal element" case
            domain, = ufl_element.domains() # Assuming single domain
            cell = domain.cell()            # Assuming single cell in domain
            degree = ufl_element.degree()
            fiat_cell = reference_cell(cell)
            fiat_element = ElementClass(fiat_cell, degree)

    if fiat_element is None:
        raise Exception("Something strange happened: reached end of function without returning an element")

    if ufl_element.cell().cellname() == "quadrilateral" and \
            isinstance(fiat_element.get_reference_element(),
                       FIAT.reference_element.two_product_cell):
        # Flatten tensor product element

        from FIAT.reference_element import FiredrakeQuadrilateral
        from FIAT.dual_set import DualSet

        nodes = fiat_element.dual.nodes
        ref_el = FiredrakeQuadrilateral()

        entity_ids = fiat_element.dual.entity_ids
        flat_entity_ids = {}
        flat_entity_ids[0] = entity_ids[(0, 0)]
        flat_entity_ids[1] = dict(enumerate(entity_ids[(0, 1)].values() + entity_ids[(1, 0)].values()))
        flat_entity_ids[2] = entity_ids[(1, 1)]

        fiat_element.dual = DualSet(nodes, ref_el, flat_entity_ids)
        fiat_element.ref_el = ref_el

    return fiat_element


def create_quadrature(cell, num_points):
    """
    Generate quadrature rule (points, weights) for given shape with
    num_points points in each direction.
    """

    if isinstance(cell, int) and cell == 0:
        return ([()], array([1.0,]))

    if cell2dim(cell) == 0:
        return ([()], array([1.0,]))

    quad_rule = FIAT.make_quadrature(reference_cell(cell), num_points)
    return quad_rule.get_points(), quad_rule.get_weights()

def map_facet_points(*arg, **kwargs):
    # This is here to prevent import failures from dead code.
    raise NotImplementedError("Function removed!")

def _extract_elements(ufl_element, restriction_domain=None):
    "Recursively extract un-nested list of (component) elements."

    elements = []
    if isinstance(ufl_element, ufl.MixedElement):
        for sub_element in ufl_element.sub_elements():
            elements += _extract_elements(sub_element, restriction_domain)
        return elements

    # Handle restricted elements since they might be mixed elements too.
    if isinstance(ufl_element, ufl.RestrictedElement):
        base_element = ufl_element.sub_element()
        restriction_domain = ufl_element.restriction_domain()
        return _extract_elements(base_element, restriction_domain)

    if restriction_domain:
        ufl_element = ufl.RestrictedElement(ufl_element, restriction_domain)

    elements += [create_element(ufl_element)]

    return elements

def _create_restricted_element(ufl_element):
    "Create an FFC representation for an UFL RestrictedElement."

    if not isinstance(ufl_element, ufl.RestrictedElement):
        error("create_restricted_element expects an ufl.RestrictedElement")

    base_element = ufl_element.sub_element()
    restriction_domain = ufl_element.restriction_domain()

    # If simple element -> create RestrictedElement from fiat_element
    if isinstance(base_element, ufl.FiniteElement):
        element = _create_fiat_element(base_element)
        tdim = ufl_element.cell().topological_dimension()
        return RestrictedElement(element, _indices(element, restriction_domain, tdim), restriction_domain)

    # If restricted mixed element -> convert to mixed restricted element
    if isinstance(base_element, ufl.MixedElement):
        elements = _extract_elements(base_element, restriction_domain)
        return MixedElement(elements)

    error("Cannot create restricted element from %s" % str(ufl_element))

def _indices(element, restriction_domain, tdim):
    "Extract basis functions indices that correspond to restriction_domain."

    # FIXME: The restriction_domain argument in FFC/UFL needs to be re-thought and
    # cleaned-up.

    # If restriction_domain is "interior", pick basis functions associated with
    # cell.
    if restriction_domain == "interior":
        return element.entity_dofs()[tdim][0]

    # Pick basis functions associated with
    # the topological degree of the restriction_domain and of all lower
    # dimensions.
    if restriction_domain == "facet":
        rdim = tdim-1
    elif restriction_domain == "face":
        rdim = 2
    elif restriction_domain == "edge":
        rdim = 1
    elif restriction_domain == "vertex":
        rdim = 0
    else:
        error("Restriction to domain: %s, is not supported." % repr(restriction_domain))

    entity_dofs = element.entity_dofs()
    indices = []
    for dim in range(rdim + 1):
        entities = entity_dofs[dim]
        for (entity, index) in sorted_by_key(entities):
            indices += index
    return indices
