# Copyright (C) 2010-2014 Anders Logg
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

__all__ = ["generate_wrapper_code"]

# FIXME: More clean-ups needed here.

def generate_wrapper_code(analysis, prefix, object_names, parameters):
    "Generate code for additional wrappers."

    # Skip if wrappers not requested
    if not parameters["format"] == "dolfin":
        return None

    raise NotImplementedError("For dolfin support please use FFC, not MFFC")
