# README #

* Ambit - A FEniCS-based cardiovascular physics solver

3D nonlinear solid and fluid mechanics finite element Python code using FEniCS and PETSc libraries, supporting

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
  - Monolithic coupling of ALE and fluid, 3D solid/fluid/ALE-fluid with lumped 0D flow models
  - Multiscale-in-time analysis of growth & remodeling (staggered solution of 3D-0D coupled solid-flow0d and G&R solid problem)
* Fluid-reduced-solid interaction (FrSI)
  - Boundary subspace-projected physics-reduced solid model (incl. hyperelastic, viscous, and active parts) in an ALE fluid reference frame
* POD-based model order reduction (MOR)
  - Projection-based model order reduction applicable to main fluid or solid field (also in a coupled problem), by either projecting
    the full problem or a boundary to a lower dimensional subspace spanned by POD modes

- author: Dr.-Ing. Marc Hirschvogel, marc.hirschvogel@deepambit.com

Still experimental / to-do:

- Fluid-solid interaction (FSI) (started)
- Finite strain plasticity
- Electrophysiology/scalar transport
- ... whatever might be wanted in some future ...


### Installation ###

In order to use Ambit, you need to [install FEniCSx](https://github.com/FEniCS/dolfinx#installation) (latest Ambit-compatible dolfinx development version dates to 19 Aug 2023)

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

If a Docker image for development is desired, the following image contains all dependencies needed to install and run Ambit (including the dolfinx mixed branch):
```
docker pull ghcr.io/marchirschvogel/ambit:devenv
```

### Usage ###

Have a look at example input files in ambit/tests and the file ambit_template.py in the main folder as example of all available input options

Best, check if all testcases run and pass, by navigating to ambit/tests and executing
```
./runtests.py
```

* Build your input file and run it with the command
```
mpiexec -n <NUMBER_OF_CORES> python3 your_file.py
```
