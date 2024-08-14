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

#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>

using namespace dealii;


template <int dim, int spacedim = dim>
class Problem {
public:
  Problem();

  void run();

private:
  parallel::distributed::Triangulation<dim, spacedim> triangulation;
  DoFHandler<dim, spacedim> dof_handler;

  hp::FECollection<dim, spacedim> fe_collection;

  IndexSet locally_owned_dofs;
  IndexSet locally_active_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_OLDSTYLE;
};



template <int dim, int spacedim>
Problem<dim, spacedim>::Problem()
 : triangulation(MPI_COMM_WORLD)
 , dof_handler(triangulation)
{}



template <int dim, int spacedim>
void
Problem<dim, spacedim>::run()
{
  GridGenerator::hyper_cube(triangulation);

  for (unsigned int degree = 1; degree <= 2; ++degree)
    fe_collection.push_back(FE_Q<dim>(degree));
  dof_handler.distribute_dofs(fe_collection);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_active_dofs   = DoFTools::extract_locally_active_dofs(dof_handler);
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler); 

  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints_OLDSTYLE.copy_from(constraints);

  constraints.make_consistent_in_parallel(locally_owned_dofs, locally_active_dofs, dof_handler.get_communicator());

  constraints_OLDSTYLE.make_consistent_in_parallel_OLDSTYLE(locally_owned_dofs, locally_active_dofs, dof_handler.get_communicator());
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Problem<3> problem_3d;
  problem_3d.run();
}
