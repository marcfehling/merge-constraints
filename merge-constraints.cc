// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the deal.II authors
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

using namespace dealii;

template <int dim, int spacedim = dim> class Problem {
public:
  Problem();

  void run();

private:
  Triangulation<dim, spacedim> triangulation;
  DoFHandler<dim, spacedim> dof_handler;
};

template <int dim, int spacedim>
Problem<dim, spacedim>::Problem() : dof_handler(triangulation) {}

template <int dim, int spacedim> void Problem<dim, spacedim>::run() {}

int main() {
  Problem<3> problem_3d;
  problem_3d.run();
}
