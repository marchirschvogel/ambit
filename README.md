# README #

[![DOI](https://joss.theoj.org/papers/10.21105/joss.05744/status.svg)](https://doi.org/10.21105/joss.05744)

Ambit is an open-source multi-physics finite element solver written in Python, supporting solid and fluid mechanics, fluid-structure interaction (FSI), and lumped-parameter models.
It is tailored towards solving problems in cardiac mechanics, but may also be used for more general nonlinear finite element analysis. It uses the finite element backend FEniCSx and
linear algebra library PETSc.

https://github.com/marchirschvogel/ambit/assets/52761273/a438ff55-9b37-4572-a1c5-499dd3cfba73

**Heart cycle simulation of a generic bi-ventricular heart model coupled to a closed-loop circulation model.**

https://github.com/marchirschvogel/ambit/assets/52761273/8e681cb6-7a4f-4d1f-b34a-cefb642f44b7

**FSI simulation (Turek benchmark, FSI2 case) of an elastic flag in incompressible channel flow.**

The following is supported:

* Solid mechanics
  - Finite strain elastodynamics, implementing a range of hyperelastic isotropic and anisotropic as well as viscous constitutive laws
  - Active stress for modeling of cardiac contraction mechanics
  - Quasi-static, generalized-alpha, or one-step theta time integration
  - Nearly incompressible as well as fully incompressible formulations (latter using pressure dofs)
  - Prestressing using MULF method in displacement formulation
  - Volumetric growth & remodeling
* Fluid dynamics
  - Incompressible Navier-Stokes/Stokes equations, either in nonconservative or conservative formulation
  - Navier-Stokes/Stokes flow in an Arbitrary Lagrangian Eulerian (ALE) reference frame
  - One-step theta, or generalized-alpha time integration
  - SUPG/PSPG stabilization for equal-order approximations of velocity and pressure
* Lumped (0D) models
  - Systemic and pulmonary circulation flow models
  - 2-element as well as 4-element Windkessel models
  - Signalling network model
* Coupling of different physics:
  - Fluid-solid interaction (FSI): Monolithic FSI in ALE formulation using Lagrange multiplier
  - Monolithic coupling of 3D solid/fluid/ALE-fluid with lumped 0D flow models
  - Multiscale-in-time analysis of growth & remodeling (staggered solution of 3D-0D coupled solid-flow0d and G&R solid problem)
* Fluid-reduced-solid interaction (FrSI)
  - Boundary subspace-projected physics-reduced solid model (incl. hyperelastic, viscous, and active parts) in an ALE fluid reference frame
* POD-based model order reduction (MOR)
  - Projection-based model order reduction applicable to main fluid or solid field (also in a coupled problem), by either projecting
    the full problem or a boundary to a lower dimensional subspace spanned by POD modes

- author: Dr.-Ing. Marc Hirschvogel, marc.hirschvogel@ambit.net

Still experimental / to-do:

- Finite strain plasticity
- Electrophysiology/scalar transport
- ... whatever might be wanted in some future ...

### Documentation ###

Documentation can be viewed at https://ambit.readthedocs.io

### Installation ###

In order to use Ambit, you need to [install FEniCSx](https://github.com/FEniCS/dolfinx#installation)

Latest Ambit-compatible dolfinx release version: N/A\
Latest tested Ambit-compatible dolfinx development version dating to 6 Aug 2025

Ambit can then be installed using pip, either the current release
```
python3 -m pip install ambit-fe
```

or latest development version:
```
python3 -m pip install git+https://github.com/marchirschvogel/ambit.git
```

Alternatively, you can pull a pre-built Docker image with FEniCSx and Ambit installed:
```
docker pull ghcr.io/marchirschvogel/ambit:latest
```

If a Docker image for development is desired, the following image contains all dependencies needed to install and run Ambit:
```
docker pull ghcr.io/marchirschvogel/ambit:devenv
```

### Usage ###

Check out the examples for the basic problem types in demos/ to quickly get started running solid, fluid, or 0D model problems. Further, you can have a look
at input files in tests/ and the file ambit_template.py in the main folder as example of all available input options.


Best, check if all testcases run and pass, by navigating to tests/ and executing
```
./runtests.py
```

Build your input file and run it with the command
```
mpiexec -n <NUMBER_OF_CORES> python3 your_file.py
```

### References ###

Software paper:

* Hirschvogel M. **Ambit – A FEniCS-based cardiovascular multi-physics solver.** *Journal of Open Source Software,* 9(93), 2024, DOI: https://doi.org/10.21105/joss.05744

Papers explicitly using and citing Ambit:

* Hirschvogel M, Bonini M, Balmus M, Nordsletten D. **Effective Block Preconditioners for Fluid Dynamics Coupled to Reduced Models of a Non-Local Nature.** *Computer Methods in Applied Mechanics and Engineering,* 435, 2025, DOI: https://doi.org/10.1016/j.cma.2024.117541
* Aróstica R, Nolte D, Brown A, Gebauer A, Karabelas E, Jilberto J, Salvador M, Bucelli M, Piersanti R, Osouli K, Augustin C, Finsberg H, Shi L, Hirschvogel M, Pfaller M, Africa PC, Gsell M, Marsden A, Nordsletten D, Regazzoni F, Plank G, Sundnes J, Dede' L, Peirlinck M, Vedula V, Wall WA, Bertoglio C. **A software benchmark for cardiac elastodynamics.** *Computer Methods in Applied Mechanics and Engineering,* 435, 2025, DOI: https://doi.org/10.1016/j.cma.2024.117485
