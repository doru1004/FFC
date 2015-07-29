"This file implements a class to represent a function call."

# Copyright (C) 2015 Fabio Luporini
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
# Last changed: 2015-01-13

# FFC modules.
from ffc.log import error
from ffc.cpp import format

# FFC quadrature modules.
from .expr import Expr

class FunCall(Expr):
    __slots__ = ("funname", "vrs")
    def __init__(self, funname, vrs=[]):
        """Initialise a FunCall object, it derives from Expr and contains
        the additional variables:

        funname - string, function name
        vrs     - list, function arguments
        NOTE: self._prec = 5."""

        # Initialise variable, type and class.
        self.val = 1.0
        self.funname = funname
        self.vrs = vrs

        self._prec = 5
        self._repr = "FunCall(%s, [%s])" % (funname, ', '.join(v._repr for v in self.vrs))

        # The type is equal to the lowest variable type.
        self.t = min([v.t for v in self.vrs])

        # Use repr as hash value.
        self._hash = hash(self._repr)

    # Print functions.
    def __str__(self):
        "Simple string representation which will appear in the generated code."
        return "%s(%s)" % (self.funname, ', '.join(str(v) for v in self.vrs))
