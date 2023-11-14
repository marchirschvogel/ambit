# README #

This example demonstrates how to couple 3D fluid flow to a 0D lumped-parameter model. Incompressible transient Navier-Stokes flow in a pipe with prescibed inflow is solved,
with the special constraint that an internal boundary (all-time closed valve) separates region 1 and region 2 of the pipe. This internal Dirichlet condition can only be achieved
by splitting the pressure space, hence having duplicate pressure nodes at the valve plane. Otherwise, fluid would experience deceleration towards the valve and unphysical acceleration
behind it, since the pressure gradient drives fluid flow. To achieve this, the mixed Dolfinx branch instead of the main branch is used. It is installed inside the Ambit devenv Docker
container. In the future, this functionality is expected to be merged into the Dolfinx main branch (at least it was announced...).

This example demonstrates how the closed valve can be bypassed by a 0D flow model that links the 3D fluid out-flow of one region to the in-flow of the other region. The 0D model consists
of two Windkessel models in series, each having compliance, resistance, and inertance elements.

![pipe_0d_setup](https://github.com/marchirschvogel/ambit/assets/52761273/d83cb0d4-74f1-4d5b-b41b-bf9f772bfbd3) \
**Simulation setup.**

### Instructions ###

Study the setup and the comments in the input file `fluid_flow0d_pipe.py`. Run the simulation, either in one of the provided Docker containers or using your own FEniCSx/Ambit installation, using the command
```
mpiexec -n 1 python3 fluid_flow0d_pipe.py
```
It is fully sufficient to use one core (mpiexec -n 1) for the presented setup.

Open the results file results_fluid_flow0d_pipe_velocity.xdmf in Paraview, and visualize the velocity over time.

Think of which parameter(s) of the 0D model to tweak in order to achieve a) little to no fluid in-flow (into $\mathit{\Gamma}_{\mathrm{in}}^{\mathrm{f-0d}}$), b) almost the same flow across $\mathit{\Gamma}_{\mathrm{out}}^{\mathrm{f-0d}}$ and $\mathit{\Gamma}_{\mathrm{in}}^{\mathrm{f-0d}}$. Think of where the flow is going to in case of a).

### Solution

The figure shows the velocity streamlines and magnitude at the end of the simulation.

![Streamlines of velocity at end of simulation, color indicates velcity magnitude.](https://github.com/marchirschvogel/ambit/assets/52761273/00b4819c-25a9-4079-a5f7-03b49de7a9af) \
**Streamlines of velocity at end of simulation, color indicates velcity magnitude.**
