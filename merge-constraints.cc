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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;

template <int dim, int spacedim = dim>
class Problem
{
public:
  Problem();

  void
  run();

private:
  parallel::distributed::Triangulation<dim, spacedim> triangulation;
  DoFHandler<dim, spacedim>                           dof_handler;

  hp::FECollection<dim, spacedim> fe_collection;

  IndexSet locally_active_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  ConditionalOStream pcout;
  TimerOutput        timer;
};

template <int dim, int spacedim>
Problem<dim, spacedim>::Problem()
  : triangulation(MPI_COMM_WORLD)
  , dof_handler(triangulation)
  , pcout(std::cout,
          Utilities::MPI::this_mpi_process(triangulation.get_communicator()) ==
            0)
  , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
{}

template <int dim, int spacedim>
void
Problem<dim, spacedim>::run()
{
  timer.enter_subsection("run");

  pcout << "Set up grid and dofs." << std::endl;
  {
    TimerOutput::Scope t(timer, "set_up");
    // Y-pipe domain from hpbox checkpoint

    // coarse mesh
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{-2., 0., 0.}, 1.},
       {{1., 1. * std::sqrt(3.), 0.}, 1.},
       {{1., -1. * std::sqrt(3.), 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    GridGenerator::pipe_junction(triangulation, openings, bifurcation);

    // load checkpoint
    triangulation.load("critical_hp.cycle-07.checkpoint");
    dof_handler.deserialize_active_fe_indices();

    pcout << "  Number of cells: " << triangulation.n_global_active_cells()
          << std::endl;

    // set dofs
    for (unsigned int degree = 1; degree <= 10; ++degree)
      fe_collection.push_back(FE_Q<dim>(degree));
    dof_handler.distribute_dofs(fe_collection);
    pcout << "  Number of DoFs:  " << dof_handler.n_dofs() << std::endl;
  }

  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_active_dofs   = DoFTools::extract_locally_active_dofs(dof_handler);
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


  pcout << "Make hanging node constraints." << std::endl;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  {
    TimerOutput::Scope t(timer, "make_hanging_node_constraints");
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  }


  pcout << "Make constraints consistent." << std::endl;
  constraints.make_consistent_in_parallel(locally_owned_dofs,
                                          locally_active_dofs,
                                          dof_handler.get_communicator(),
                                          timer);

  timer.leave_subsection();

  // print timing results
  timer.print_wall_time_statistics(dof_handler.get_communicator());
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Problem<3> problem_3d;
  problem_3d.run();
}
