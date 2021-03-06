//
// This code complies with UFC version 1.0, and is generated with SyFi version 0.4.0.
//
// http://www.fenics.org/syfi/
// http://www.fenics.org/ufc/
//


#include <stdexcept>
#include <math.h>
#include <ufc.h>
#include <pycc/Functions/Ptv.h>
#include <pycc/Functions/Ptv_tools.h>
#include <pycc/Functions/Dof_Ptv.h>
#include <pycc/Functions/OrderedPtvSet.h>
#include <pycc/Functions/Dof_OrderedPtvSet.h>
#include "fe_Lagrange_1_2D.h"


namespace pycc
{


/// Constructor
fe_Lagrange_1_2D::fe_Lagrange_1_2D() : ufc::finite_element()
{
  
}

/// Destructor
fe_Lagrange_1_2D::~fe_Lagrange_1_2D()
{
  
}

/// Return a string identifying the finite element
const char* fe_Lagrange_1_2D::signature() const
{
  return "fe_Lagrange_1_2D // generated by SyFi";
}

/// Return the cell shape
ufc::shape fe_Lagrange_1_2D::cell_shape() const
{
  return ufc::triangle;
}

/// Return the dimension of the finite element function space
unsigned int fe_Lagrange_1_2D::space_dimension() const
{
  return 3;
}

/// Return the rank of the value space
unsigned int fe_Lagrange_1_2D::value_rank() const
{
  return 0;
}

/// Return the dimension of the value space for axis i
unsigned int fe_Lagrange_1_2D::value_dimension(unsigned int i) const
{
  return 1;
}

/// Evaluate basis function i at given point in cell
void fe_Lagrange_1_2D::evaluate_basis(unsigned int i,
                                   double* values,
                                   const double* coordinates,
                                   const ufc::cell& c) const
{
  const double x = coordinates[0];
  const double y = coordinates[1];
  switch(i)
  {
  case 0:
    values[0] = -x-y+1.0;
    break;
  case 1:
    values[0] = x;
    break;
  case 2:
    values[0] = y;
    break;
  }
}

/// Evaluate order n derivatives of basis function i at given point in cell
void fe_Lagrange_1_2D::evaluate_basis_derivatives(unsigned int i,
                                               unsigned int n,
                                               double* values,
                                               const double* coordinates,
                                               const ufc::cell& c) const
{
    throw std::runtime_error("gen_evaluate_basis_derivatives not implemented yet.");
}

/// Evaluate linear functional for dof i on the function f
double fe_Lagrange_1_2D::evaluate_dof(unsigned int i,
                                   const ufc::function& f,
                                   const ufc::cell& c) const
{
  // coordinates
  double x0 = c.coordinates[0][0]; double y0 = c.coordinates[0][1];
  double x1 = c.coordinates[1][0]; double y1 = c.coordinates[1][1];
  double x2 = c.coordinates[2][0]; double y2 = c.coordinates[2][1];
  
  // affine map
  double G00 = x1 - x0;
  double G01 = x2 - x0;
  
  double G10 = y1 - y0;
  double G11 = y2 - y0;
  
  double v[1];
  double x[2];
  switch(i)
    {
  case 0:
    x[0] = x0;
    x[1] = y0;
    break;
  case 1:
    x[0] = x0+G00;
    x[1] = G10+y0;
    break;
  case 2:
    x[0] = G01+x0;
    x[1] = y0+G11;
    break;
  }
  f.evaluate(v, x, c);
  return v[i % 1];

}

/// Interpolate vertex values from dof values
void fe_Lagrange_1_2D::interpolate_vertex_values(double* vertex_values,
                                              const double* dof_values,
                                              const ufc::cell& c) const
{
  vertex_values[0] = dof_values[0];
  vertex_values[1] = dof_values[1];
  vertex_values[2] = dof_values[2];
}

/// Return the number of sub elements (for a mixed element)
unsigned int fe_Lagrange_1_2D::num_sub_elements() const
{
  return 1;
}

/// Create a new finite element for sub element i (for a mixed element)
ufc::finite_element* fe_Lagrange_1_2D::create_sub_element(unsigned int i) const
{
  return new fe_Lagrange_1_2D();
}


} // namespace
