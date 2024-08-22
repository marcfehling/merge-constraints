Merge constraints
=================

Compare the [AffineConstraints::make_consistent_in_parallel()](https://dealii.org/developer/doxygen/deal.II/classAffineConstraints.html#ab0217a83250614a473e29096d9a7f515) implementations within the hp-context: The old version with the new one introduced in [#14905](https://github.com/dealii/dealii/pull/14905).

Setup
=====

You need to work on the [deal.II master branch](https://github.com/dealii/dealii/tree/master).

Configure this project as an in-source build as follows:

    cmake -DDEAL_II_DIR=/path/to/dealii .
    make

To reproduce the assertion, run in debug mode on 2 MPI processes:

    make debug
    make
    mpirun -np 2 ./merge-constraints
