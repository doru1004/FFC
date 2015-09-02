"""Code generation format strings for UFC (Unified Form-assembly Code) v. 1.6.0dev

Three format strings are defined for each of the following UFC classes:

    function
    finite_element
    dofmap
    cell_integral
    exterior_facet_integral
    interior_facet_integral
    form

The strings are named '<classname>_header', '<classname>_implementation',
and '<classname>_combined'. The header and implementation contain the
definition and declaration respectively, and are meant to be placed in
.h and .cpp files, while the combined version is for an implementation
within a single .h header.

Each string has the following format variables: 'classname',
'members', 'constructor', 'destructor', plus one for each interface
function with name equal to the function name.

For more information about UFC and the FEniCS Project, visit

    http://www.fenicsproject.org
    https://bitbucket.org/fenics-project/ufc

"""

# -*- coding: utf-8 -*-
__author__  = "Martin Sandve Alnaes, Anders Logg, Kent-Andre Mardal, Ola Skavhaug, and Hans Petter Langtangen"
__date__    = "2015-01-12"
__version__ = "1.6.0dev"
__license__ = "This code is released into the public domain"

from .function import *
from .finite_element import *
from .dofmap import *
from .integrals import *
from .form import *
from .build import build_ufc_module

templates = {"function_header":                          function_header,
             "function_implementation":                  function_implementation,
             "function_combined":                        function_combined,
             "finite_element_header":                    finite_element_header,
             "finite_element_implementation":            finite_element_implementation,
             "finite_element_combined":                  finite_element_combined,
             "dofmap_header":                            dofmap_header,
             "dofmap_implementation":                    dofmap_implementation,
             "dofmap_combined":                          dofmap_combined,
             "cell_integral_header":                     cell_integral_header,
             "cell_integral_implementation":             cell_integral_implementation,
             "cell_integral_combined":                   cell_integral_combined,
             "exterior_facet_integral_header":           exterior_facet_integral_header,
             "exterior_facet_integral_implementation":   exterior_facet_integral_implementation,
             "exterior_facet_integral_combined":         exterior_facet_integral_combined,
             "interior_facet_integral_header":           interior_facet_integral_header,
             "interior_facet_integral_implementation":   interior_facet_integral_implementation,
             "interior_facet_integral_combined":         interior_facet_integral_combined,
             "vertex_integral_header":                    vertex_integral_header,
             "vertex_integral_implementation":            vertex_integral_implementation,
             "vertex_integral_combined":                  vertex_integral_combined,
             "custom_integral_header":                   custom_integral_header,
             "custom_integral_implementation":           custom_integral_implementation,
             "custom_integral_combined":                 custom_integral_combined,
             "form_header":                              form_header,
             "form_implementation":                      form_implementation,
             "form_combined":                            form_combined}
