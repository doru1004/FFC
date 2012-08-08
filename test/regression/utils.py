# Copyright (C) 2010 Anders Logg, Kristian B. Oelgaard and Marie E. Rognes
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

def run_command(command):
    "Run command and collect errors in log file."
    (status, output) = get_status_output(command)
    if status == 0:
        return True
    global logfile
    if logfile is None:
        logfile = open("../../error.log", "w")
    logfile.write(output + "\n")
    print output
    return False


