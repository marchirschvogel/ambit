# README #

This example demonstrates how to couple 3D fluid flow to a 0D lumped-parameter model.


### Instructions ###

Run the simulation, either in one of the provided Docker containers or using your own FEniCSx/Ambit installation, using the command
```
mpiexec -n 1 python3 fluid_flow0d_pipe.py
```
It is fully sufficient to use one core (mpiexec -n 1) for the presented setup.

Open the results file results_fluid_flow0d_pipe_velocity.xdmf in Paraview, and visualize the velocity over time.

### Solution

The figure shows the velocity streamlines and magnitude at the end of the simulation.
