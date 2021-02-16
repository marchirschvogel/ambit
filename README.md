# README #

* AMBIT - A FEniCS-based cardiovascular physics solver

3D nonlinear solid (and fluid) mechanics Python code using FEniCS and PETSc libraries, supporting

- Hyperelastic isotropic and anisotropic materials
- Inelastic constitutive laws (growth & remodeling, G&R)
- Dynamics
- 0D lumped systemic and pulmonary circulation, Windkessel models
- Monolithic coupling of 3D solid (and fluid) mechanics with lumped 0D flow models
- Multiscale-in-time analysis of growth & remodeling (staggered solution of 3D-0D coupled solid and solid G&R problem)

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

* RECOMMENDED: FEniCS should run in a Docker container unless one wants to run through installation from source (see below)

* best, use rootless Docker (https://docs.docker.com/engine/security/rootless)
if not present, install: (seems that uidmap needs to be installed, which requires to be root... :-/)

``sudo apt install uidmap``
``curl -fsSL https://get.docker.com/rootless | sh``

* Get latest tested Ambit-compatible digest (4 Feb 2021) of dolfinx Docker image (the experimental FEniCS version, https://hub.docker.com/u/dolfinx):
(latest nightly image can be pulled without @..., but may or may not work with current Ambit code):

``docker pull dolfinx/dolfinx@sha256:a4115d7a11d64d09deed26a53d53efc0feb95d97379d3bad38b8e537ce763481``

* put the following shortcut in .bashrc (replacing <PATH_TO_AMBIT_FOLDER> with the path to the ambit folder):

``alias fenicsdocker='docker run -ti -v $HOME:/home/shared -v <PATH_TO_AMBIT_FOLDER>:/home/ambit -w /home/shared/ --env-file <PATH_TO_AMBIT_FOLDER>/.env.list --rm dolfinx/dolfinx@sha256:a4115d7a11d64d09deed26a53d53efc0feb95d97379d3bad38b8e537ce763481'``

* if 0D models should be used, it seems that we have to install sympy (not part of docker container anymore) - in the folder where you pulled ambit to, do:

``cd ambit && mkdir modules/ext && pip3 install --system --target=modules/ext mpmath --no-deps --no-cache-dir && pip3 install --system --target=modules/ext sympy --no-deps --no-cache-dir && cd ..``

* then launch the container in a konsole/terminal window by simply typing

``fenicsdocker``

* have a look at example input files in ambit/testing and the file ambit_template.py in the main folder as example of all available input options

* launch your input file with

``mpirun -np <NUMBER_OF_CORES> python3 your_file.py``


*************************************************


### Other useful Linux stuff:

* in general, before installing other things, do an update:

``sudo apt update``
``sudo apt upgrade``

* if not pre-installed, install Python3 stuff

``sudo apt install python3``
``sudo apt install python3-distutils``
``sudo apt install python3-pip``
``sudo apt install python3-dev``

* compiler stuff:

``sudo apt install build-essential``
``sudo apt install gfortran``
``sudo apt install libgfortran3``

* an mpich compiler (i.e. mpich-3.3.2 from http://www.mpich.org/downloads)

``mkdir mpich-3.3.2_install && cd mpich-3.3.2 && ./configure --prefix=/home/mh/mpich-3.3.2_install && make -j 6 && make install && cd ..``

* after install, need to set in ~/.bashrc:

``export MPI_HOME=$HOME/mpich-3.3.2_install``
``export MPI_ROOT=$MPI_HOME``
``export PATH=$MPI_HOME/bin:$PATH``

* Paraview's Python interface

``sudo apt install paraview-python``

* useful Python stuff

``pip3 install numpy mpmath sympy``
``pip3 install -v --no-deps --no-cache-dir mpi4py``
``pip3 install -v --no-deps --no-cache-dir cffi``
``pip3 install progress``

* command to delete old / all docker images

``docker image prune -a``

* command to delete a docker image by ID

``docker image rm <IMAGE_ID>``

### Building FEniCS from source

* If you want to build FEniCS (dolfin-x) from source (can be tedious) instead of using the convenient docker image, then try the following:

* install Eigen3 and Boost stuff

``sudo apt install libeigen3-dev``
``sudo apt install libboost-all-dev libgmp-dev libmpfr-dev``

* optional but useful tools

``sudo apt install libblas-dev liblapack-dev``
``sudo apt install libhdf5-mpich-dev``
``sudo apt install libsuitesparse-dev``
``sudo apt install hdf5-tools``

* install PETSc first (latest release version)

``git clone -b release https://gitlab.com/petsc/petsc.git petsc``
``./configure --with-mpi-dir=$MPI_HOME --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch --download-scotch --download-blacs --download-fblaslapack --download-hypre --download-suitsparse --download-superlu --download-superlu_dist``

* install mpi4py and petsc4py

``pip3 install -v --no-deps --no-cache-dir mpi4py``
``pip3 install -v --no-deps --no-cache-dir petsc4py``

* install Pybind

``PYBIND11_VERSION=2.5.0``
``wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz``
``tar -xf v${PYBIND11_VERSION}.tar.gz && cd pybind11-${PYBIND11_VERSION}``
``mkdir build && mkdir -p ../pybind_install && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=$HOME/pybind_install .. && sudo make install && cd ../.. && rm v${PYBIND11_VERSION}.tar.gz``

* get all the FEniCS stuff

* wherever you are, make a fenics and a dolfinx_install folder and navigate to the fenics folder:

``mkdir -p fenics/dolfinx_install && cd fenics``

* in there, do the following:

``git clone https://github.com/FEniCS/fiat.git``
``git clone https://bitbucket.org/fenics-project/dijitso``
``git clone https://github.com/FEniCS/ufl.git``
``git clone https://github.com/FEniCS/ffcx.git``
``git clone https://github.com/fenics/dolfinx.git``
``cd fiat    && pip3 -v install --no-deps --no-cache-dir . && cd ..``
``cd dijitso && pip3 -v install --no-deps --no-cache-dir . && cd ..``
``cd ufl     && pip3 -v install --no-deps --no-cache-dir . && cd ..``
``cd ffcx    && pip3 -v install --no-deps --no-cache-dir . && cd ..``
``mkdir -p dolfinx/build && mkdir -p dolfinx_install && cd dolfinx/build && cmake -DCMAKE_INSTALL_PREFIX=dolfinx_install ../cpp && sudo make install -j 6 && cd ../..``
``source dolfinx_install/share/dolfinx/dolfinx.conf``
``cd dolfinx/python && pip3 -v install --no-deps --no-cache-dir . && cd ../..``

* for update of already installed dolfin-x, do the following inside the fenics folder:

``cd ffcx && git pull && pip3 -v install --no-deps --no-cache-dir . && cd ..``
``cd dolfinx && git pull && cd build && cmake -DCMAKE_INSTALL_PREFIX=dolfinx_install ../cpp && sudo make install -j 6 && source dolfinx_install/share/dolfinx/dolfinx.conf && cd ../python && pip3 -v install --no-deps . && cd ../..``
