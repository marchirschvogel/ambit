# README #

This example shows how to set up 2D fluid flow in a channel around a rigid obstacle. Incompressible Navier-Stokes flow is solved using Taylor-Hood elements
(9-node biquadratic quadrilaterals for the velocity, 4-node bilinear quadrilaterals for the pressure).

![Simulation setup.](https://github.com/marchirschvogel/ambit/assets/52761273/494703fd-72cb-4f83-9984-8f3e9b3b63c8) \
**Simulation setup.**

### Instructions ###

Study the setup and the comments in the input file `fluid_channel.py`. Run the simulation, either in one of the provided Docker containers or using your own FEniCSx/Ambit installation, using the command
```
mpiexec -n 1 python3 fluid_channel.py
```
It is fully sufficient to use one core (mpiexec -n 1) for the presented setup.

Open the results file results_fluid_channel_velocity.xdmf and results_fluid_channel_pressure.xdmf in Paraview, and visualize the velocity as well as the pressure over time.

### Solution

The figure shows the velocity magnitude (top) as well as the pressure (bottom part) at the end of the simulation.

![Velocity magnitude (top part) and pressure (bottom part) at end of simulation.](https://github.com/marchirschvogel/ambit/assets/52761273/fb3445f8-b928-4379-9c91-7c001fb6671b) \
**Velocity magnitude (top part) and pressure (bottom part) at end of simulation.**
