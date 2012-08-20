// Copyright (C) 2010 Anders Logg
//
// This file is part of FFC.
//
// FFC is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FFC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with FFC. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-01-24
// Last changed: 2011-07-04
//
// Functions for calling generated UFC functions with "random" (but
// fixed) data and print the output to screen. Useful for running
// regression tests.

#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <ufc.h>

typedef unsigned int uint;

// How many derivatives to test
const uint max_derivative = 2;

// Precision in output of floats
const uint precision = 16;
const double epsilon = 1e-16;

// Parameters for adaptive timing
const uint initial_num_reps = 10;
const double minimum_timing = 1.0;

// Global counter for results
uint counter = 0;

// Function for timing
double time()
{
  clock_t __toc_time = std::clock();
  return ((double) (__toc_time)) / CLOCKS_PER_SEC;
}

// Function for printing a single value
template <class value_type>
void print_value(value_type value)
{
  std::cout.precision(precision);
  if (std::abs(static_cast<double>(value)) < epsilon)
    std::cout << "0";
  else
    std::cout << value;
}

// Function for printing scalar result
template <class value_type>
void print_scalar(std::string name, value_type value, int i=-1, int j=-1)
{
  std::stringstream s;
  s << counter++ << "_";
  s << name;
  if (i >= 0) s << "_" << i;
  if (j >= 0) s << "_" << j;
  std::cout << s.str() << " = ";
  print_value(value);
  std::cout << std::endl;
}

// Function for printing array result
template <class value_type>
void print_array(std::string name, unsigned int n, value_type* values, int i=-1, int j=-1)
{
  std::stringstream s;
  s << counter++ << "_";
  s << name;
  if (i >= 0) s << "_" << i;
  if (j >= 0) s << "_" << j;
  std::cout << s.str() << " =";
  for (uint i = 0; i < n; i++)
  {
    std::cout << " ";
    print_value(values[i]);
  }
  std::cout << std::endl;
}

// Class for creating "random" ufc::cell objects
class test_cell : public ufc::cell
{
public:

  test_cell(ufc::shape cell_shape, int offset=0)
  {
    // Store cell shape
    this->cell_shape = cell_shape;

    // Store dimensions
    switch (cell_shape)
    {
    case ufc::interval:
      topological_dimension = 1;
      geometric_dimension = 1;
      break;
    case ufc::triangle:
      topological_dimension = 2;
      geometric_dimension = 2;
      break;
    case ufc::tetrahedron:
      topological_dimension = 3;
      geometric_dimension = 3;
      break;
    default:
      throw std::runtime_error("Unhandled cell shape.");
    }

    // Generate some "random" entity indices
    entity_indices = new uint * [4];
    for (uint i = 0; i < 4; i++)
    {
      entity_indices[i] = new uint[6];
      for (uint j = 0; j < 6; j++)
        entity_indices[i][j] = i*j + offset;
    }

    // Generate some "random" coordinates
    double** x = new double * [4];
    for (uint i = 0; i < 4; i++)
      x[i] = new double[3];
    x[0][0] = 0.903; x[0][1] = 0.341; x[0][2] = 0.457;
    x[1][0] = 0.561; x[1][1] = 0.767; x[1][2] = 0.833;
    x[2][0] = 0.987; x[2][1] = 0.783; x[2][2] = 0.191;
    x[3][0] = 0.123; x[3][1] = 0.561; x[3][2] = 0.667;
    coordinates = x;
  }

  ~test_cell()
  {
    for (uint i = 0; i < 4; i++)
    {
      delete [] entity_indices[i];
      delete [] coordinates[i];
    }
    delete [] entity_indices;
    delete [] coordinates;
  }

};

void test_cell_integral(void (*test_integral_fn)(double *, double **, double **), 
                        uint tensor_size,
                        uint num_coefficients,
                        uint space_dimension,
                        ufc::shape cell_shape,
                        bool bench)
{
  if (bench)
  {
    std::cerr << "Benchmarking PyOP2 not yet supported." << std::endl;
    std::exit(1);
  }

  std::cout << std::endl;
  std::cout << "Testing cell_integral" << std::endl;
  std::cout << "---------------------" << std::endl;

  // Prepare dummy coefficients
  double** w = 0;
  if (num_coefficients > 0)
  {
    w = new double * [num_coefficients];
    for (uint i = 0; i < num_coefficients; i++)
    {
      const uint macro_dim = 2*space_dimension; // *2 for interior facet integrals
      w[i] = new double[macro_dim];
      for (uint j = 0; j < macro_dim; j++)
        w[i][j] = 0.1*static_cast<double>((i + 1)*(j + 1));
    }
  }

  // Prepare local tensor
  double* A = new double[tensor_size];
  for(uint i = 0; i < tensor_size; i++)
    A[i] = 0.0;

  // Prepare coordinates
  test_cell c(cell_shape);
  double **coords = c.coordinates;

  // Call integral function
  test_integral_fn(A, coords, w);
  print_array("tabulate_tensor", tensor_size, A);

  // Cleanup
  delete [] A;
}
