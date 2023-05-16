# README #

* AMBIT - A FEniCS-based cardiovascular physics solver

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
- Enhance PETSc linear iterative solver interface, enable surface-projected problems
- Finite strain plasticity
- ... whatever might be wanted in some future ...


### How do I get set up? ###

* Clone the repo:

``git clone https://github.com/marchirschvogel/ambit.git``

* RECOMMENDED: FEniCS should run in a Docker container unless one wants to run through installation from source (see https://github.com/FEniCS/dolfinx if needed)

* Best, use rootless Docker (https://docs.docker.com/engine/security/rootless)
if not present, install: (seems that uidmap needs to be installed, which requires to be root... :-/)

``sudo apt install uidmap``\
``curl -fsSL https://get.docker.com/rootless | sh``

* Get latest tested ambit-compatible digest (16 May 2023) of dolfinx Docker image:

``docker pull dolfinx/dolfinx@sha256:8681f25150240cee71ccdccfc98763ab5debc252ab9a2e07391aab5102c59dd0``

* To get dolfinx nightly build (may or may not work with ambit code):

``docker pull dolfinx/dolfinx:nightly``

* Put the following shortcut in .bashrc (replacing <PATH_TO_AMBIT_FOLDER> with the path to the ambit folder):

``alias fenicsdocker='docker run -ti -v $HOME:/home/shared -v <PATH_TO_AMBIT_FOLDER>:/home/ambit -w /home/shared/ --env-file <PATH_TO_AMBIT_FOLDER>/.env.list --rm dolfinx/dolfinx@sha256:8681f25150240cee71ccdccfc98763ab5debc252ab9a2e07391aab5102c59dd0'``

* If 0D models should be used, it seems that we have to install sympy (not part of docker container anymore) - in the folder where you pulled ambit to, do:

``cd ambit && mkdir modules/ext && pip3 install --system --target=modules/ext mpmath --no-deps --no-cache-dir && pip3 install --system --target=modules/ext sympy --no-deps --no-cache-dir && cd ..``

* Launch the container in a konsole/terminal window by simply typing

``fenicsdocker``

* Have a look at example input files in ambit/testing and the file ambit_template.py in the main folder as example of all available input options

* Launch your input file with

``mpiexec -n <NUMBER_OF_CORES> python3 your_file.py``
