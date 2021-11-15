# README #

* AMBIT - A FEniCS-based cardiovascular physics solver

3D nonlinear solid (and fluid) mechanics Python code using FEniCS and PETSc libraries, supporting

- Hyperelastic isotropic and anisotropic materials
- Inelastic constitutive laws (growth & remodeling, G&R)
- Dynamics
- Incompressibility in solid mechanics (2-field functional with displacement and pressure dofs)
- 0D lumped systemic and pulmonary circulation flow, Windkessel models
- Monolithic coupling of 3D solid (and fluid) mechanics with lumped 0D flow models
- Multiscale-in-time analysis of growth & remodeling (staggered solution of 3D-0D coupled solid-flow0d and G&R solid problem)

- author: Dr.-Ing. Marc Hirschvogel, marc.hirschvogel@deepambit.com

Still experimental / to-do:

- Fluid mechanics is not tested!
- Inf-sup stable equal order fluid mechanics formulation for Navier Stokes (SUPG/PSPG stabilization)
- ALE fluid
- Linear solvers and preconditioners (working, but best choices for specific problems still need investigation)
- ... whatever might be wanted in some future ...


### How do I get set up? ###

* Clone the repo:

``git clone https://github.com/marchirschvogel/ambit.git``

* RECOMMENDED: FEniCS should run in a Docker container unless one wants to run through installation from source (see https://github.com/FEniCS/dolfinx if needed)

* best, use rootless Docker (https://docs.docker.com/engine/security/rootless)
if not present, install: (seems that uidmap needs to be installed, which requires to be root... :-/)

``sudo apt install uidmap``\
``curl -fsSL https://get.docker.com/rootless | sh``

* Get latest tested ambit-compatible digest (14 Nov 2021) of dolfinx Docker image (the experimental FEniCS version, https://hub.docker.com/u/dolfinx):
(latest nightly image can be pulled without @..., but may or may not work with current ambit code):

``docker pull dolfinx/dolfinx@sha256:29602f44137c80910a1093013a3fdc8a7b7f1eb3368caa95b016043efbcc2455``

* put the following shortcut in .bashrc (replacing <PATH_TO_AMBIT_FOLDER> with the path to the ambit folder):

``alias fenicsdocker='docker run -ti -v $HOME:/home/shared -v <PATH_TO_AMBIT_FOLDER>:/home/ambit -w /home/shared/ --env-file <PATH_TO_AMBIT_FOLDER>/.env.list --rm dolfinx/dolfinx@sha256:29602f44137c80910a1093013a3fdc8a7b7f1eb3368caa95b016043efbcc2455'``

* if 0D models should be used, it seems that we have to install sympy (not part of docker container anymore) - in the folder where you pulled ambit to, do:

``cd ambit && mkdir modules/ext && pip3 install --system --target=modules/ext mpmath --no-deps --no-cache-dir && pip3 install --system --target=modules/ext sympy --no-deps --no-cache-dir && cd ..``

* then launch the container in a konsole/terminal window by simply typing

``fenicsdocker``

* have a look at example input files in ambit/testing and the file ambit_template.py in the main folder as example of all available input options

* launch your input file with

``mpiexec -n <NUMBER_OF_CORES> python3 your_file.py``
