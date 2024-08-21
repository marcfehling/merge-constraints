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

  AffineConstraints<double> constraints_NEWSTYLE;

  ConditionalOStream pcout;
};

template <int dim, int spacedim>
Problem<dim, spacedim>::Problem()
  : triangulation(MPI_COMM_WORLD)
  , dof_handler(triangulation)
  , pcout(std::cout,
          Utilities::MPI::this_mpi_process(triangulation.get_communicator()) ==
            0)
{}

template <int dim, int spacedim>
void
Problem<dim, spacedim>::run()
{
  // --------------------
  pcout << "Set up grid and dofs." << std::endl;
  // --------------------

  // set triangulation
  if (true)
    {
      // L-shaped domain as in mg-ev-estimator

      std::vector<unsigned int> repetitions(dim);
      Point<dim>                bottom_left, top_right;
      for (unsigned int d = 0; d < dim; ++d)
        if (d < 2)
          {
            repetitions[d] = 2;
            bottom_left[d] = -1.;
            top_right[d]   = 1.;
          }
        else
          {
            repetitions[d] = 1;
            bottom_left[d] = 0.;
            top_right[d]   = 1.;
          }

      std::vector<int> cells_to_remove(dim, 1);
      cells_to_remove[0] = -1;

      GridGenerator::subdivided_hyper_L(
        triangulation, repetitions, bottom_left, top_right, cells_to_remove);

      triangulation.refine_global(2);

      // hp-refine center part
      for (const auto &cell : dof_handler.active_cell_iterators() |
                                IteratorFilters::LocallyOwnedCell())
        {
          // set all cells to second to last FE
          cell->set_active_fe_index(1);

          const auto &center = cell->center();
          if (std::abs(center[0]) < 0.5 && std::abs(center[1]) < 0.5)
            {
              if (center[0] < -0.25 || center[1] > 0.25)
                // outer layer gets p-refined
                cell->set_active_fe_index(2);
              else
                // inner layer gets h-refined
                cell->set_refine_flag();
            }
        }

      triangulation.execute_coarsening_and_refinement();
    }
  else if (false)
    {
      // Y-pipe domain from hpbox checkpoint

      // coarse mesh
      const std::vector<std::pair<Point<spacedim>, double>> openings = {
        {{{-2., 0., 0.}, 1.},
         {{1., 1. * std::sqrt(3.), 0.}, 1.},
         {{1., -1. * std::sqrt(3.), 0.}, 1.}}};

      const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

      GridGenerator::pipe_junction(triangulation, openings, bifurcation);

      // load checkpoint
      triangulation.load("critical_hp.cycle-01.checkpoint");
      dof_handler.deserialize_active_fe_indices();
    }

  pcout << "  Number of cells: " << triangulation.n_global_active_cells()
        << std::endl;

  // set dofs
  for (unsigned int degree = 1; degree <= 10; ++degree)
    fe_collection.push_back(FE_Q<dim>(degree));
  dof_handler.distribute_dofs(fe_collection);

  pcout << "  Number of DoFs:  " << dof_handler.n_dofs() << std::endl;

  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_active_dofs   = DoFTools::extract_locally_active_dofs(dof_handler);
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


  // --------------------
  pcout << "Make hanging node constraints." << std::endl;
  // --------------------

  constraints_NEWSTYLE.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_NEWSTYLE);

  {
    std::cout << "  hanging node constraints on process "
              << Utilities::MPI::this_mpi_process(
                   dof_handler.get_communicator())
              << ": " << constraints_NEWSTYLE.n_constraints() << std::endl;
    MPI_Barrier(dof_handler.get_communicator());
  }


  // --------------------
  pcout << "Make constraints consistent." << std::endl;
  // --------------------

  constraints_NEWSTYLE.make_consistent_in_parallel(
    locally_owned_dofs, locally_active_dofs, dof_handler.get_communicator());

  {
    // constraints.print(std::cout);

    std::cout << "  consistent constraints NEWSTYLE on process "
              << Utilities::MPI::this_mpi_process(
                   dof_handler.get_communicator())
              << ": " << constraints_NEWSTYLE.n_constraints() << std::endl;
    MPI_Barrier(dof_handler.get_communicator());
  }

  // --------------------
  pcout << "Compare differences in constraints." << std::endl;
  // --------------------
  // manually specify problematic dofs
  std::set<types::global_dof_index> problematic_dofs;

  // add dofs whose constraint lines differ after calling
  // both versions of make_consistent_in_parallel
  problematic_dofs.insert({774, 852});
  // add dofs to which 774 and 852 are constrained against
  // in both versions of make_consistent_in_parallel
  problematic_dofs.insert({209, 627, 773, 627, 667, 851});
  // add dofs to which 774 and 852 are additionally constrained against
  // in the NEWSTYLE version of make_consistent_in_parallel
  problematic_dofs.insert({630, 670});

  {
    // merge on all processes
    problematic_dofs =
      Utilities::MPI::compute_set_union(problematic_dofs,
                                        dof_handler.get_communicator());

    MPI_Barrier(dof_handler.get_communicator());
  }


  // --------------------
  pcout << "Write results." << std::endl;
  // --------------------

  Vector<float> fe_degrees(triangulation.n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
    fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;

  Vector<float> subdomain(triangulation.n_active_cells());
  for (auto &subd : subdomain)
    subd = triangulation.locally_owned_subdomain();

  Vector<float>                        mask(triangulation.n_active_cells());
  std::vector<types::global_dof_index> local_dofs;
  for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
    {
      local_dofs.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(local_dofs);
      for (unsigned int i = 0; i < local_dofs.size(); ++i)
        if (problematic_dofs.contains(local_dofs[i]))
          {
            std::cout << "  DoF " << local_dofs[i] << " is the " << i
                      << "th DoF on cell " << cell->global_active_cell_index()
                      << " with FE_Q(" << cell->get_fe().degree
                      << ") on process "
                      << Utilities::MPI::this_mpi_process(
                           dof_handler.get_communicator())
                      << std::endl;

            mask(cell->active_cell_index()) = local_dofs[i];
          }
    }

  DataOut<dim, spacedim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(fe_degrees, "fe_degrees");
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.add_data_vector(mask, "mask");
  data_out.build_patches();

  data_out.write_vtu_in_parallel("result.vtu", dof_handler.get_communicator());
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Problem<3> problem_3d;
  problem_3d.run();
}
