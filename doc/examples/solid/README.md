# README #

This example demonstrates how to set up a quasi-static solid mechanics elasticity problem. The deformation of a steel cantilever under transverse conservative load is simulated. The structure 
is fixed on one end. Quadratic 27-node hexahedral finite elements are used for the discretization of the domain.
The well-known St. Venant-Kirchhoff material is used as constitutive law, which is a generalization of Hooke's law to the nonlinear realm.

![Simulation setup.](https://github.com/marchirschvogel/ambit/assets/52761273/55463473-93e6-4dd5-82e6-7fce65c3d0a5) \
**Simulation setup.**

### Instructions ###

Run the simulation, either in one of the provided Docker containers or using your own FEniCSx/Ambit installation, using the command
```
mpiexec -n 1 python3 solid_cantilever.py
```
It is fully sufficient to use one core (mpiexec -n 1) for the presented setup.

Open the results file results_solid_cantilever_displacement.xdmf in Paraview, and visualize the deformation over time.

### Solution

The figure shows the displacement magnitude at the end of the simulation.

![Deformed cantilever, color indicates the magnitude of the displacement.](https://github.com/marchirschvogel/ambit/assets/52761273/32235966-603b-4f79-ac98-a8f898bd6d78) \
**Deformed cantilever, color indicates the magnitude of the displacement.**
