# Code generation format strings for UFC (Unified Form-assembly Code) v. 2.3.0+.
# This code is released into the public domain.
#
# The FEniCS Project (http://www.fenicsproject.org/) 2006-2014

cell_integral_combined = """\
/// This class defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over a cell.

class %(classname)s: public ufc::cell_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s) : ufc::cell_integral()%(initializer_list)s
  {
%(constructor)s
  }

  /// Destructor
  virtual ~%(classname)s()
  {
%(destructor)s
  }

  /// Tabulate the tensor for the contribution from a local cell
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               int cell_orientation) const
  {
%(tabulate_tensor)s
  }

};
"""

cell_integral_header = """\
/// This class defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over a cell.

class %(classname)s: public ufc::cell_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s);

  /// Destructor
  virtual ~%(classname)s();

  /// Tabulate the tensor for the contribution from a local cell
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               int cell_orientation) const;

};
"""

cell_integral_implementation = """\
/// Constructor
%(classname)s::%(classname)s(%(constructor_arguments)s) : ufc::cell_integral()%(initializer_list)s
{
%(constructor)s
}

/// Destructor
%(classname)s::~%(classname)s()
{
%(destructor)s
}

/// Tabulate the tensor for the contribution from a local cell
void %(classname)s::tabulate_tensor(double* A,
                                    const double * const * w,
                                    const double* vertex_coordinates,
                                    int cell_orientation) const
{
%(tabulate_tensor)s
}
"""

exterior_facet_integral_combined = """\
/// This class defines the interface for the tabulation of the
/// exterior facet tensor corresponding to the local contribution to
/// a form from the integral over an exterior facet.

class %(classname)s: public ufc::exterior_facet_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s) : ufc::exterior_facet_integral()%(initializer_list)s
  {
%(constructor)s
  }

  /// Destructor
  virtual ~%(classname)s()
  {
%(destructor)s
  }

  /// Tabulate the tensor for the contribution from a local exterior facet
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               std::size_t facet,
                               int cell_orientation) const
  {
%(tabulate_tensor)s
  }

};
"""

exterior_facet_integral_header = """\
/// This class defines the interface for the tabulation of the
/// exterior facet tensor corresponding to the local contribution to
/// a form from the integral over an exterior facet.

class %(classname)s: public ufc::exterior_facet_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s);

  /// Destructor
  virtual ~%(classname)s();

  /// Tabulate the tensor for the contribution from a local exterior facet
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               std::size_t facet,
                               int cell_orientation) const;

};
"""

exterior_facet_integral_implementation = """\
/// Constructor
%(classname)s::%(classname)s(%(constructor_arguments)s) : ufc::exterior_facet_integral()%(initializer_list)s
{
%(constructor)s
}

/// Destructor
%(classname)s::~%(classname)s()
{
%(destructor)s
}

/// Tabulate the tensor for the contribution from a local exterior facet
void %(classname)s::tabulate_tensor(double* A,
                                    const double * const * w,
                                    const double* vertex_coordinates,
                                    std::size_t facet,
                                    int cell_orientation) const
{
%(tabulate_tensor)s
}
"""

interior_facet_integral_combined = """\
/// This class defines the interface for the tabulation of the
/// interior facet tensor corresponding to the local contribution to
/// a form from the integral over an interior facet.

class %(classname)s: public ufc::interior_facet_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s) : ufc::interior_facet_integral()%(initializer_list)s
  {
%(constructor)s
  }

  /// Destructor
  virtual ~%(classname)s()
  {
%(destructor)s
  }

  /// Tabulate the tensor for the contribution from a local interior facet
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates_0,
                               const double* vertex_coordinates_1,
                               std::size_t facet_0,
                               std::size_t facet_1,
                               int cell_orientation_0,
                               int cell_orientation_1) const
  {
%(tabulate_tensor)s
  }

};
"""

interior_facet_integral_header = """\
/// This class defines the interface for the tabulation of the
/// interior facet tensor corresponding to the local contribution to
/// a form from the integral over an interior facet.

class %(classname)s: public ufc::interior_facet_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s);

  /// Destructor
  virtual ~%(classname)s();

  /// Tabulate the tensor for the contribution from a local interior facet
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates_0,
                               const double* vertex_coordinates_1,
                               std::size_t facet_0,
                               std::size_t facet_1,
                               int cell_orientation_0,
                               int cell_orientation_1) const;

};
"""

interior_facet_integral_implementation = """\
/// Constructor
%(classname)s::%(classname)s(%(constructor_arguments)s) : ufc::interior_facet_integral()%(initializer_list)s
{
%(constructor)s
}

/// Destructor
%(classname)s::~%(classname)s()
{
%(destructor)s
}

/// Tabulate the tensor for the contribution from a local interior facet
void %(classname)s::tabulate_tensor(double* A,
                                    const double * const * w,
                                    const double* vertex_coordinates_0,
                                    const double* vertex_coordinates_1,
                                    std::size_t facet_0,
                                    std::size_t facet_1,
                                    int cell_orientation_0,
                                    int cell_orientation_1) const
{
%(tabulate_tensor)s
}
"""

point_integral_combined = """\
/// This class defines the interface for the tabulation of
/// an expression evaluated at exactly one point.

class %(classname)s: public ufc::point_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s) : ufc::point_integral()%(initializer_list)s
  {
%(constructor)s
  }

  /// Destructor
  virtual ~%(classname)s()
  {
%(destructor)s
  }

  /// Tabulate the tensor for the contribution from the local vertex
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               std::size_t vertex,
                               int cell_orientation) const
  {
%(tabulate_tensor)s
  }

};
"""

point_integral_header = """\
/// This class defines the interface for the tabulation of
/// an expression evaluated at exactly one point.

class %(classname)s: public ufc::point_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s);

  /// Destructor
  virtual ~%(classname)s();

  /// Tabulate the tensor for the contribution from the local vertex
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               std::size_t vertex,
                               int cell_orientation) const;

};
"""

point_integral_implementation = """\
/// Constructor
%(classname)s::%(classname)s(%(constructor_arguments)s) : ufc::point_integral()%(initializer_list)s
{
%(constructor)s
}

/// Destructor
%(classname)s::~%(classname)s()
{
%(destructor)s
}

/// Tabulate the tensor for the contribution from the local vertex
void %(classname)s::tabulate_tensor(double* A,
                                    const double * const * w,
                                    const double* vertex_coordinates,
                                    std::size_t vertex,
                                    int cell_orientation) const
{
%(tabulate_tensor)s
}
"""

quadrature_cell_integral_combined = """\
/// This class defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over an unknown cell fragment with quadrature
/// points given.

class %(classname)s: public ufc::quadrature_cell_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s) : ufc::quadrature_cell_integral()%(initializer_list)s
  {
%(constructor)s
  }

  /// Destructor
  virtual ~%(classname)s()
  {
%(destructor)s
  }

  /// Tabulate the tensor for the contribution from a cell fragment
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               std::size_t num_quadrature_points,
                               const double* quadrature_points,
                               const double* quadrature_weights,
                               int cell_orientation) const
  {
%(tabulate_tensor)s
  }

};
"""

quadrature_cell_integral_header = """\
/// This class defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over an unknown cell fragment with quadrature
/// points given.

class %(classname)s: public ufc::quadrature_cell_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s);

  /// Destructor
  virtual ~%(classname)s();

  /// Tabulate the tensor for the contribution from a cell fragment
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates,
                               std::size_t num_quadrature_points,
                               const double* quadrature_points,
                               const double* quadrature_weights,
                               int cell_orientation) const;
};
"""

quadrature_cell_integral_implementation = """\
/// Constructor
%(classname)s::%(classname)s(%(constructor_arguments)s) : ufc::quadrature_cell_integral()%(initializer_list)s
{
%(constructor)s
}

/// Destructor
%(classname)s::~%(classname)s()
{
%(destructor)s
}

/// Tabulate the tensor for the contribution from a local cell
void %(classname)s::tabulate_tensor(double* A,
                                    const double * const * w,
                                    const double* vertex_coordinates,
                                    std::size_t num_quadrature_points,
                                    const double* quadrature_points,
                                    const double* quadrature_weights,
                                    int cell_orientation) const;
{
%(tabulate_tensor)s
}
"""

quadrature_facet_integral_combined = """\
/// This class defines the interface for the tabulation of the facet
/// tensor corresponding to the local contribution to a form from
/// the integral over an unknown facet fragment with quadrature
/// points given.

class %(classname)s: public ufc::quadrature_facet_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s) : ufc::quadrature_facet_integral()%(initializer_list)s
  {
%(constructor)s
  }

  /// Destructor
  virtual ~%(classname)s()
  {
%(destructor)s
  }

  /// Tabulate the tensor for the contribution from a facet fragment
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates_0
                               std::size_t num_cells_1,
                               const double* vertex_coordinates_1,
                               std::size_t num_quadrature_points,
                               const double* quadrature_points,
                               const double* quadrature_weights,
                               std::size_t* quadrature_point_to_cell_map_1) const
  {
%(tabulate_tensor)s
  }

};
"""

quadrature_facet_integral_header = """\
/// This class defines the interface for the tabulation of the facet
/// tensor corresponding to the local contribution to a form from
/// the integral over an unknown facet fragment with quadrature
/// points given.

class %(classname)s: public ufc::quadrature_facet_integral
{%(members)s
public:

  /// Constructor
  %(classname)s(%(constructor_arguments)s);

  /// Destructor
  virtual ~%(classname)s();

  /// Tabulate the tensor for the contribution from a facet fragment
  virtual void tabulate_tensor(double* A,
                               const double * const * w,
                               const double* vertex_coordinates_0
                               std::size_t num_cells_1,
                               const double* vertex_coordinates_1,
                               std::size_t num_quadrature_points,
                               const double* quadrature_points,
                               const double* quadrature_weights,
                               std::size_t* quadrature_point_to_cell_map_1) const

};
"""

quadrature_facet_integral_implementation = """\
/// Constructor
%(classname)s::%(classname)s(%(constructor_arguments)s) : ufc::quadrature_facet_integral()%(initializer_list)s
{
%(constructor)s
}

/// Destructor
%(classname)s::~%(classname)s()
{
%(destructor)s
}

/// Tabulate the tensor for the contribution from a local facet
void %(classname)s::tabulate_tensor(double* A,
                                    const double * const * w,
                                    const double* vertex_coordinates_0
                                    std::size_t num_cells_1,
                                    const double* vertex_coordinates_1,
                                    std::size_t num_quadrature_points,
                                    const double* quadrature_points,
                                    const double* quadrature_weights,
                                    std::size_t* quadrature_point_to_cell_map_1) const
{
%(tabulate_tensor)s
}
"""
