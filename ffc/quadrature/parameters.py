"Quadrature representation class for UFL"

# Copyright (C) 2009-2014 Kristian B. Oelgaard
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
# Modified by Anders Logg 2009, 2014
# Modified by Martin Alnaes 2013-2014

# FFC modules
from ffc.log import warning

def parse_optimise_parameters(parameters, itg_data):

    # Initialize parameters
    optimise_parameters = {"eliminate zeros":     False,
                           "optimisation":        False,
                           "ignore ones":         False,
                           "remove zero terms":   False,
                           "ignore zero tables":  False}


    # Set optimized parameters
    if parameters["optimize"] and itg_data.integral_type == "custom":
        warning("Optimization not available for custom integrals, skipping optimization.")
    elif parameters["optimize"]:
        # Disable "ignore ones", because it is broken
        # optimise_parameters["ignore ones"]        = True
        optimise_parameters["remove zero terms"]  = True
        optimise_parameters["ignore zero tables"] = True

        if parameters["optimize"] == "ffc -O":
            optimise_parameters["eliminate zeros"] = True
            optimise_parameters["optimisation"]    = "simplify_expressions"

    return optimise_parameters
