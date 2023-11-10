# README #

This example shows how to set up 2D fluid flow in a channel around a rigid obstacle, using Taylor-Hood elements.


### Instructions ###

Run the simulation, either in one of the provided Docker containers or using your own FEniCSx/Ambit installation, using the command
```
mpiexec -n 1 python3 fluid_channel.py
```
It is fully sufficient to use one core (mpiexec -n 1) for the presented setup.

Open the results file results_channel_velocity.xdmf and results_channel_pressure.xdmf in Paraview, and visualize the velocity as well as the pressure over time.

### Solution

The figure shows the velocity magnitude (top) as well as the pressure (bottom part) at the end of the simulation.
