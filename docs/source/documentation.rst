==========================================================
Ambit – A FEniCS-based cardiovascular multi-physics solver
==========================================================

:Author: Dr.-Ing. Marc Hirschvogel

.. role:: raw-latex(raw)
   :format: latex
..

Preface
=======

| Ambit :cite:p:`hirschvogel2024-ambit` is an open-source
  multi-physics finite element solver written in Python, supporting
  solid and fluid mechanics, fluid-structure interaction (FSI), and
  lumped-parameter models. It is tailored towards solving problems in
  cardiac mechanics, but may also be used for more general nonlinear
  finite element analysis. The code encompasses re-implementations and
  generalizations of methods developed by the author for his PhD thesis
  :cite:p:`hirschvogel2019disspub` and beyond. Ambit makes use
  of the open-source finite element library FEniCS/dolfinx
  (https://fenicsproject.org) :cite:p:`logg2012-fenics` along
  with the linear algebra package PETSc (https://petsc.org)
  :cite:p:`balay2022-petsc`, hence guaranteeing a
  state-of-the-art finite element and linear algebra backend. It is
  constantly updated to ensure compatibility with a recent dolfinx
  development version. I/O routines are designed such that the user only
  needs to provide input files that define parameters through Python
  dictionaries, hence no programming or in-depth knowledge of any
  library-specific syntax is required.
| Ambit provides general nonlinear (compressible or incompressible)
  finite strain solid dynamics :cite:p:`holzapfel2000`,
  implementing a range of hyperelastic, viscous, and active material
  models. Specifically, the well-known anisotropic Holzapfel-Ogden
  :cite:p:`holzapfel2009` and Guccione models
  :cite:p:`guccione1995` for structural description of the
  myocardium are provided, along with a bunch of other models. It
  further implements strain- and stress-mediated volumetric growth
  models :cite:p:`goektepe2010` that allow to model
  (maladaptive) ventricular shape and size changes. Inverse mechanics
  approaches to imprint loads into a reference state are implemented
  using the so-called prestressing method :cite:p:`gee2010` in
  displacement formulation :cite:p:`schein2021`.
| Furthermore, fluid dynamics in terms of incompressible
  Navier-Stokes/Stokes equations – either in Eulerian or Arbitrary
  Lagrangian-Eulerian (ALE) reference frames – are implemented.
  Taylor-Hood elements or equal-order approximations with SUPG/PSPG
  stabilization :cite:p:`tezduyar2000` can be used.
| A variety of reduced 0D lumped models targeted at blood circulation
  modeling are implemented, including 3- and 4-element Windkessel models
  :cite:p:`westerhof2009` as well as closed-loop full
  circulation :cite:p:`hirschvogel2017` and coronary flow
  models :cite:p:`arthurs2016`.
| Monolithic fluid-solid interaction (FSI)
  :cite:p:`nordsletten2011` in ALE formulation using a
  Lagrange multiplier field is supported, along with coupling of 3D and
  0D models (solid or fluid with 0D lumped circulation systems) such
  that cardiovascular simulations with realistic boundary conditions can
  be performed.
| Implementations for a recently proposed novel physics- and
  projection-based model reduction for FSI, denoted as
  fluid-reduced-solid interaction (FrSI)
  :cite:p:`hirschvogel2024-frsi`, are provided, along with
  POD-based Galerkin model reduction techniques
  :cite:p:`farhat2014` using full or boundary subspaces.
| The nonlinear (single- or multi-field) problems are solved with a
  customized Newton solver with PTC :cite:p:`gee2009`
  adaptivity in case of divergence, providing robustness for numerically
  challenging problems. Linear solvers and preconditioners can be chosen
  from the PETSc repertoire, and specific block preconditioners are made
  available for coupled problems.

| Avenues for future functionality include cardiac electrophysiology,
  scalar transport, or finite strain plasticity.
| In the following, a brief description of the supported problem types
  is given, including the strong and weak form of the underlying
  equations as well as the discrete assembled systems that are solved.
| Examples of input files for the respective problem types can be found
  in the folder ``demos`` (with detailed setup descriptions) or amogst
  the test cases in the folder ``tests``.
| This documentation is structured as follows. In sec.
  `2 <#installation>`__, instructions on how to install and use Ambit
  are given. The relevant supported physics models are described in sec.
  `4 <#physics-models>`__. Demos are presented in sec. `5 <#demos>`__.

Installation
============

| In order to use Ambit, you need to install FEniCSx
  (https://github.com/FEniCS/dolfinx#installation) (latest
  Ambit-compatible dolfinx development version dates to 28 Aug 2024).
| Ambit can then be installed using pip, either the current release

::

   python3 -m pip install ambit-fe

or latest development version:

::

   python3 -m pip install git+https://github.com/marchirschvogel/ambit.git

Alternatively, you can pull a pre-built Docker image with FEniCSx and
Ambit installed:

::

   docker pull ghcr.io/marchirschvogel/ambit:latest

If a Docker image for development is desired, the following image
contains all dependencies needed to install and run Ambit:

::

   docker pull ghcr.io/marchirschvogel/ambit:devenv

Ambit input
===========

Here, a minimal Ambit input file is shown, exemplarily for a
single-field problem. The mandatory parameter dictionaries to provide
are input parameters (IO), time parameters (TME), solver parameters
(SOL), finite element parameters (FEM), and constitutive/material
parameters (MAT). For multi-physics problems, each field needs
individual time, finite element, and constitutive parameters.

::

   #!/usr/bin/env python3

   # Minimal input file for an elastodynamics problem

   import ambit_fe
   import numpy as np

   def main():

       # Input/output
       IO = {"problem_type"        : "solid",                   # type of physics to solve
             "mesh_domain"         : "/path/mesh_d.xdmf",       # path to domain mesh
             "mesh_boundary"       : "/path/mesh_b.xdmf",       # ath to boundary mesh
             "meshfile_type"       : "HDF5",                    # encoding (HDF5 or ASCII)
             "write_results_every" : 1,                         # step frequency for output
             "output_path"         : "/path/...",               # path to output to
             "results_to_write"    : ["displacement",
                                      "vonmises_cauchystress"], # results to output
             "simname"             : "my_results_name"}         # midfix of output name
       
       # Time discretization  
       TME = {"maxtime"            : 1.0,        # maximum simulation time
              "dt"                 : 0.01,       # time step size
              "timint"             : "genalpha", # time integration: Generalized-alpha
              "rho_inf_genalpha"   : 1.0}        # spectral radius of Gen-alpha scheme
       
       # Solver
       SOL = {"solve_type"         : "direct", # direct linear solver
              "tol_res"            : 1.0e-8,   # residual tolerance
              "tol_inc"            : 1.0e-8}   # increment tolerance
       
       # Finite element discretization
       FEM = {"order_disp"         : 1, # FEM degree for displacement field
              "quad_degree"        : 2} # quadrature scheme degree
       
       # Time curves
       class TC:
           # user defined load curves, to be used in boundary conditions (BC)
           def tc1(self, t):
               load_max = 5.0
               return load_max * np.sin(t)

       # Materials
       MAT = {"MAT1" : {"neohooke_dev" : {"mu" : 10.0},      # isochoric NeoHookean material
                        "ogden_vol"    : {"kappa" : 1.0e3},  # volumetric Ogden material
                        "inertia"      : {"rho0" : 1.0e-6}}} # density

       # Boundary conditions
       BC = {"dirichlet" : [{"id" : [<SURF_IDs>], # list of surfaces for Dirichlet BC
                             "dir" : "all",       # all directions
                             "val" : 0.0}],       # set to zero
             "neumann"   : [{"id" : [<SURF_IDs>], # list of surfaces for Neumann BC
                             "dir" : "xyz_ref",   # in cartesian reference directions
                             "curve" : [1,0,0]}]} # load in x-direction controlled by curve #1 (see time curves)

       # Problem setup
       problem = ambit_fe.ambit_main.Ambit(io_params=IO, time_params=TME, solver_params=SOL, fem_params=FEM, constitutive_params=MAT, boundary_conditions=BC, time_curves=TC)

       # Run: solve the problem
       problem.solve_problem()
       
   if __name__ == "__main__":

       main()

Physics Models
==============

Solid mechanics
---------------

| – Example: Sec. `5.1 <#demo-solid>`__ and ``demos/solid``
| – ``problem_type : "solid"``
| – Solid mechanics are formulated in a Total Lagrangian frame

Strong form
~~~~~~~~~~~

| **Displacement-based**
| – Primary variable: displacement :math:`\boldsymbol{u}`

  .. math::
     :label: solid-strong-form

     \begin{aligned}
     \nabla_{0} \cdot \boldsymbol{P}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) + \hat{\boldsymbol{b}}_{0} &= \rho_{0} \boldsymbol{a}(\boldsymbol{u}) &&\text{in} \; \mathit{\Omega}_{0} \times [0, T], \\
     \boldsymbol{u} &= \hat{\boldsymbol{u}} &&\text{on} \; \mathit{\Gamma}_{0}^{\mathrm{D}} \times [0, T],\\
     \boldsymbol{t}_{0} = \boldsymbol{P}\boldsymbol{n}_{0} &= \hat{\boldsymbol{t}}_{0} &&\text{on} \; \mathit{\Gamma}_{0}^{\mathrm{N}} \times [0, T],\\
     \boldsymbol{u}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{u}}_{0}(\boldsymbol{x}_{0}) &&\text{in} \; \mathit{\Omega}_{0},\\
     \boldsymbol{v}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}_{0}) &&\text{in} \; \mathit{\Omega}_{0},
     \end{aligned}

| **Incompressible mechanics**
| – Primary variables: displacement :math:`\boldsymbol{u}` and pressure
  :math:`p`

  .. math::
     :label: solid-strong-form-inc

     \begin{aligned}
     \nabla_{0} \cdot \boldsymbol{P}(\boldsymbol{u},p,\boldsymbol{v}(\boldsymbol{u})) + \hat{\boldsymbol{b}}_{0} &= \rho_{0} \boldsymbol{a}(\boldsymbol{u}) &&\text{in} \; \mathit{\Omega}_{0} \times [0, T], \\
     J(\boldsymbol{u})-1 &= 0 &&\text{in} \; \mathit{\Omega}_{0} \times [0, T], \\
     \boldsymbol{u} &= \hat{\boldsymbol{u}} &&\text{on} \; \mathit{\Gamma}_{0}^{\mathrm{D}} \times [0, T],\\
     \boldsymbol{t}_{0} = \boldsymbol{P}\boldsymbol{n}_{0} &= \hat{\boldsymbol{t}}_{0} &&\text{on} \; \mathit{\mathit{\Gamma}}_{0}^{\mathrm{N}} \times [0, T],\\
     \boldsymbol{u}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{u}}_{0}(\boldsymbol{x}_{0}) &&\text{in} \; \mathit{\mathit{\Omega}}_{0},\\
     \boldsymbol{v}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}_{0}) &&\text{in} \; \mathit{\mathit{\Omega}}_{0},
     \end{aligned}

Weak form
~~~~~~~~~

| **Displacement-based**
| – Primary variable: displacement :math:`\boldsymbol{u}`
| – Principal of Virtual Work:

  .. math::
     :label: solid-weak-form

     \begin{aligned}
     r(\boldsymbol{u};\delta\boldsymbol{u}) := \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{u};\delta\boldsymbol{u}) + \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u};\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}\end{aligned}

  – Kinetic virtual work:

  .. math::
     :label: deltaw-kin

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{u};\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \rho_{0}\,\boldsymbol{a}(\boldsymbol{u}) \cdot \delta\boldsymbol{u} \,\mathrm{d}V
     \end{aligned}

  – Internal virtual work:

  .. math::
     :label: deltaw-int

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u};\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \boldsymbol{P}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) : \nabla_{0} \delta\boldsymbol{u} \,\mathrm{d}V = \int\limits_{\mathit{\Omega}_{0}} \boldsymbol{S}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) : \frac{1}{2}\delta\boldsymbol{C}(\boldsymbol{u}) \,\mathrm{d}V
     \end{aligned}

  – External virtual work:

-  conservative Neumann load:

   .. math::
      :label: deltaw-ext-pk1

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Gamma}_{0}^{\mathrm{N}}} \hat{\boldsymbol{t}}_{0}(t) \cdot \delta\boldsymbol{u} \,\mathrm{d}A
      \end{aligned}

-  Neumann pressure load in current normal direction:

   .. math::
      :label: deltaw-ext-cur-p

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= -\int\limits_{\mathit{\Gamma}_{0}^{\mathrm{N}}} \hat{p}(t)\,J \boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0} \cdot \delta\boldsymbol{u} \,\mathrm{d}A \end{aligned}

-  general Neumann load in current direction:

   .. math::
      :label: deltaw-ext-cur

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Gamma}_0} J\boldsymbol{F}^{-\mathrm{T}}\,\hat{\boldsymbol{t}}_{0}(t) \cdot \delta\boldsymbol{u} \,\mathrm{d}A
      \end{aligned}

-  body force:

   .. math::
      :label: deltaw-ext-body

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \hat{\boldsymbol{b}}_{0}(t) \cdot \delta\boldsymbol{u} \,\mathrm{d}V
      \end{aligned}

-  generalized Robin condition:

   .. math::
      :label: deltaw-ext-robin

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= -\int\limits_{\mathit{\Gamma}_{0}^{\mathrm{R}}} \left[k\,\boldsymbol{u} + c\,\boldsymbol{v}(\boldsymbol{u})\right] \cdot \delta\boldsymbol{u}\,\mathrm{d}A
      \end{aligned}

-  generalized Robin condition in reference surface normal direction:

   .. math::
      :label: deltaw-ext-robin-n

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= -\int\limits_{\mathit{\Gamma}_{0}^{\mathrm{R}}} (\boldsymbol{n}_0 \otimes \boldsymbol{n}_0)\left[k\,\boldsymbol{u} + c\,\boldsymbol{v}(\boldsymbol{u})\right] \cdot \delta\boldsymbol{u}\,\mathrm{d}A
      \end{aligned}

| **Incompressible mechanics: 2-field displacement and pressure
  variables**
| – Primary variables: displacement :math:`\boldsymbol{u}` and pressure
  :math:`p`

  .. math::
     :label: solid-weak-form-inc

     \begin{aligned}
     r_u(\boldsymbol{u},p;\delta\boldsymbol{u}) &:= \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{u};\delta\boldsymbol{u}) + \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u},p;\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u} \\
     r_p(\boldsymbol{u};\delta p) &:= \delta \mathcal{W}_{\mathrm{pres}}(\boldsymbol{u};\delta p) = 0, \quad \forall \; \delta p
     \end{aligned}

| – Kinetic virtual work:
  (`[equation-deltaw-kin] <#equation-deltaw-kin>`__)
| – Internal virtual work:

  .. math::
     :label: deltaw-int-inc

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u},p;\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \boldsymbol{P}(\boldsymbol{u},p,\boldsymbol{v}(\boldsymbol{u})) : \nabla_{0} \delta\boldsymbol{u} \,\mathrm{d}V = \int\limits_{\mathit{\Omega}_{0}} \boldsymbol{S}(\boldsymbol{u},p,\boldsymbol{v}(\boldsymbol{u})) : \frac{1}{2}\delta\boldsymbol{C}(\boldsymbol{u}) \,\mathrm{d}V
     \end{aligned}

  – Pressure virtual work:

  .. math::
     :label: deltaw-p

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{pres}}(\boldsymbol{u};\delta p) &= \int\limits_{\mathit{\Omega}_{0}} (J(\boldsymbol{u}) - 1) \,\delta p \,\mathrm{d}V 
     \end{aligned}

| **Material models**
| – hyperelastic material models

  .. math::
     \begin{aligned}
     \boldsymbol{S} = 2\frac{\partial\mathit{\Psi}}{\partial \boldsymbol{C}}
     \end{aligned}

| - MAT ``"neohooke_dev"``
| 

  .. math::
     \begin{aligned}
     \mathit{\Psi} &= \frac{\mu}{2}\left(\bar{I}_C - 3\right)
     \end{aligned}

| - MAT ``"holzapfelogden_dev"``
| 

  .. math::
     \begin{aligned}
     \mathit{\Psi} &= \frac{a_0}{2b_0}\left(e^{b_0(\bar{I}_C - 3)} - 1\right) + \sum\limits_{i\in\{f,s\}}\frac{a_i}{2b_i}\left(e^{b_i(I_{4,i}-1)^2}-1\right) + \frac{a_{fs}}{2b_{fs}}\left(e^{b_{fs}I_{8}^2} - 1\right), \\ & I_{4,f} = \boldsymbol{f}_0 \cdot \boldsymbol{C}\boldsymbol{f}_0, \quad I_{4,s} = \boldsymbol{s}_0 \cdot \boldsymbol{C}\boldsymbol{s}_0, \quad I_8 = \boldsymbol{f}_0 \cdot \boldsymbol{C}\boldsymbol{s}_0
     \end{aligned}

– viscous material models

.. math::
   \begin{aligned}
   \boldsymbol{S} = 2\frac{\partial\mathit{\Psi}_{\mathrm{v}}}{\partial \dot{\boldsymbol{C}}}
   \end{aligned}

| **Time integration**
| – time scheme ``timint : "static"``
| 

  .. math::
     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n+1};\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n+1};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}\end{aligned}

  – Generalized-alpha time scheme ``timint : "genalpha"``

  .. math::
     \begin{aligned}
     \boldsymbol{v}_{n+1} &= \frac{\gamma}{\beta\Delta t}(\boldsymbol{u}_{n+1}-\boldsymbol{u}_{n}) - \frac{\gamma-\beta}{\beta} \boldsymbol{v}_{n} - \frac{\gamma-2\beta}{2\beta}\Delta t\,\boldsymbol{a}_{n} \\
     \boldsymbol{a}_{n+1} &= \frac{1}{\beta\Delta t^2}(\boldsymbol{u}_{n+1}-\boldsymbol{u}_{n}) - \frac{1}{\beta\Delta t} \boldsymbol{v}_{n} - \frac{1-2\beta}{2\beta}\boldsymbol{a}_{n}
     \end{aligned}

  - option ``eval_nonlin_terms : "midpoint"``:

  .. math::
     :label: solid-midpoint-genalpha

     \begin{aligned}
     \boldsymbol{u}_{n+1-\alpha_{\mathrm{f}}} &= (1-\alpha_{\mathrm{f}})\boldsymbol{u}_{n+1} + \alpha_{\mathrm{f}} \boldsymbol{v}_{n} \\
     \boldsymbol{v}_{n+1-\alpha_{\mathrm{f}}} &= (1-\alpha_{\mathrm{f}})\boldsymbol{v}_{n+1} + \alpha_{\mathrm{f}} \boldsymbol{v}_{n} \\
     \boldsymbol{a}_{n+1-\alpha_{\mathrm{m}}} &= (1-\alpha_{\mathrm{m}})\boldsymbol{a}_{n+1} + \alpha_{\mathrm{m}} \boldsymbol{a}_{n}
     \end{aligned}

  .. math::
     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{a}_{n+1-\alpha_{m}};\delta\boldsymbol{u}) + \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n+1-\alpha_{f}};\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n+1-\alpha_{f}};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}\end{aligned}

  - option ``eval_nonlin_terms : "trapezoidal"``:

  .. math::
     \begin{aligned}
     &(1-\alpha_{\mathrm{m}})\,\delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{a}_{n+1};\delta\boldsymbol{u}) + \alpha_{\mathrm{m}}\,\delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{a}_{n};\delta\boldsymbol{u}) + \\
     & (1-\alpha_{\mathrm{f}})\,\delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n+1};\delta\boldsymbol{u}) + \alpha_{\mathrm{f}}\,\delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n};\delta\boldsymbol{u}) - \\
     & (1-\alpha_{f})\,\delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n+1};\delta\boldsymbol{u}) - \alpha_{\mathrm{f}}\,\delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}\end{aligned}

| – One-Step-theta time scheme ``timint : "ost"``
| 

  .. math::
     \begin{aligned}
     \boldsymbol{v}_{n+1} &= \frac{1}{\theta\Delta t}(\boldsymbol{u}_{n+1}-\boldsymbol{u}_{n}) - \frac{1-\theta}{\theta} \boldsymbol{v}_{n} \\
     \boldsymbol{a}_{n+1} &= \frac{1}{\theta^2\Delta t^2}(\boldsymbol{u}_{n+1}-\boldsymbol{u}_{n}) - \frac{1}{\theta^2\Delta t} \boldsymbol{v}_{n} - \frac{1-\theta}{\theta}\boldsymbol{a}_{n}
     \end{aligned}

  - option ``eval_nonlin_terms : "midpoint"``:

  .. math::
     :label: solid-midpoint-ost

     \begin{aligned}
     \boldsymbol{u}_{n+\theta} &= \theta \boldsymbol{u}_{n+1} + (1-\theta) \boldsymbol{u}_{n} \\
     \boldsymbol{v}_{n+\theta} &= \theta \boldsymbol{v}_{n+1} + (1-\theta) \boldsymbol{v}_{n} \\
     \boldsymbol{a}_{n+\theta} &= \theta \boldsymbol{a}_{n+1} + (1-\theta) \boldsymbol{a}_{n}
     \end{aligned}

  .. math::
     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{a}_{n+\theta};\delta\boldsymbol{u}) + \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n+\theta};\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n+\theta};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}\end{aligned}

  - option ``eval_nonlin_terms : "trapezoidal"``:

  .. math::
     \begin{aligned}
     &\theta\,\delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{a}_{n+1};\delta\boldsymbol{u}) + (1-\theta)\,\delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{a}_{n};\delta\boldsymbol{u}) + \\
     & \theta\,\delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n+1};\delta\boldsymbol{u}) + (1-\theta)\,\delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u}_{n};\delta\boldsymbol{u}) - \\
     & \theta\,\delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n+1};\delta\boldsymbol{u}) - (1-\theta)\,\delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u}_{n};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}
     \end{aligned}

| Note the equivalence of ``"midpoint"`` and ``"trapezoidal"`` for all
  linear terms, e.g. :math:`\delta \mathcal{W}_{\mathrm{kin}}`, or for
  no or only linear dependence of
  :math:`\delta \mathcal{W}_{\mathrm{ext}}` on the solution.
| Note that, for incompressible mechanics, the pressure kinematic
  constraint is always evaluated at :math:`t_{n+1}`:

  .. math::
     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{pres}}(\boldsymbol{u}_{n+1};\delta p) = 0, \quad \forall \; \delta p,
     \end{aligned}

  and the pressure in :math:`\delta \mathcal{W}_{\mathrm{int}}` is set
  according to
  (`[equation-solid-midpoint-genalpha] <#equation-solid-midpoint-genalpha>`__)
  or (`[equation-solid-midpoint-ost] <#equation-solid-midpoint-ost>`__),
  respectively.

| **Spatial discretization and solution**
| – Discrete nonlinear system to solve in each time step :math:`n`
  (displacement-based):

  .. math::
     :label: nonlin-sys-solid

     \begin{aligned}
     \left.\boldsymbol{\mathsf{r}}_{u}(\boldsymbol{\mathsf{u}})\right|_{n+1} = \boldsymbol{\mathsf{0}}
     \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`
(displacement-based):

.. math::
   :label: lin-sys-solid

   \begin{aligned}
   \left. \boldsymbol{\mathsf{K}}_{uu} \right|_{n+1}^{k} \Delta\boldsymbol{\mathsf{u}}_{n+1}^{k+1}=-\left. \boldsymbol{\mathsf{r}}_{u} \right|_{n+1}^{k}
   \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n`
(incompressible):

.. math::
   :label: nonlin-sys-solid-inc

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{p}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{u}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`
(incompressible):

.. math::
   :label: lin-sys-solid-inc

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{up} \\ \\ \boldsymbol{\mathsf{K}}_{pu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u} \\ \\ \boldsymbol{\mathsf{r}}_{p} \end{bmatrix}_{n+1}^{k}
   \end{aligned}

Fluid mechanics
---------------

Eulerian reference frame
~~~~~~~~~~~~~~~~~~~~~~~~

| – Example: Sec. `5.2 <#demo-fluid>`__ and ``demos/fluid``
| – Problem type: ``fluid``
| – Incompressible Navier-Stokes equations in Eulerian reference frame
| **Strong Form**
| – Primary variables: velocity :math:`\boldsymbol{v}` and pressure
  :math:`p`

  .. math::
     :label: fluid-strong-form

     \begin{aligned}
     \nabla \cdot \boldsymbol{\sigma}(\boldsymbol{v},p) + \hat{\boldsymbol{b}} &= \rho\left(\frac{\partial\boldsymbol{v}}{\partial t} + (\nabla\boldsymbol{v})\,\boldsymbol{v}\right) &&\text{in} \; \mathit{\mathit{\Omega}}_t \times [0, T], \\
     \nabla\cdot \boldsymbol{v} &= 0 &&\text{in} \; \mathit{\mathit{\Omega}}_t \times [0, T],\\
     \boldsymbol{v} &= \hat{\boldsymbol{v}} &&\text{on} \; \mathit{\mathit{\Gamma}}_t^{\mathrm{D}} \times [0, T],\\
     \boldsymbol{t} = \boldsymbol{\sigma}\boldsymbol{n} &= \hat{\boldsymbol{t}} &&\text{on} \; \mathit{\mathit{\Gamma}}_t^{\mathrm{N}} \times [0, T],\\
     \boldsymbol{v}(\boldsymbol{x},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}) &&\text{in} \; \mathit{\mathit{\Omega}}_t,
     \end{aligned}

with a Newtonian fluid constitutive law

.. math::
   \begin{aligned}
   \boldsymbol{\sigma} = -p \boldsymbol{I} + 2 \mu\,\boldsymbol{\gamma} = -p \boldsymbol{I} + \mu \left(\nabla \boldsymbol{v} + (\nabla \boldsymbol{v})^{\mathrm{T}}\right)
   \end{aligned}

| **Weak Form**
| – Primary variables: velocity :math:`\boldsymbol{v}` and pressure
  :math:`p`
| – Principle of Virtual Power

  .. math::
     :label: fluid-weak-form

     \begin{aligned}
     r_v(\boldsymbol{v},p;\delta\boldsymbol{v}) &:= \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v};\delta\boldsymbol{v}) + \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p;\delta\boldsymbol{v}) - \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{v};\delta\boldsymbol{v}) = 0, \quad \forall \; \delta\boldsymbol{v} \\
     r_p(\boldsymbol{v};\delta p) &:= \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v};\delta p), \quad \forall \; \delta p
     \end{aligned}

– Kinetic virtual power:

.. math::
   :label: deltap-kin

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v};\delta\boldsymbol{v}) = \int\limits_{\mathit{\Omega}_t} \rho\left[\frac{\partial\boldsymbol{v}}{\partial t} + (\nabla\boldsymbol{v})\,\boldsymbol{v}\right] \cdot \delta\boldsymbol{v} \,\mathrm{d}v
   \end{aligned}

– Internal virtual power:

.. math::
   :label: deltap-int

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p;\delta\boldsymbol{v}) = 
   \int\limits_{\mathit{\Omega}_t} \boldsymbol{\sigma}(\boldsymbol{v},p) : \nabla \delta\boldsymbol{v} \,\mathrm{d}v 
   \end{aligned}

– Pressure virtual power:

.. math::
   :label: deltap-p

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v};\delta p) = 
   \int\limits_{\mathit{\Omega}_t} (\nabla\cdot\boldsymbol{v})\,\delta p\,\mathrm{d}v
   \end{aligned}

| – External virtual power:

-  conservative Neumann load:

   .. math::
      :label: deltap-ext-cur

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Gamma}_t^{\mathrm{N}}} \hat{\boldsymbol{t}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}a
      \end{aligned}

-  pressure Neumann load:

   .. math::
      :label: deltap-ext-cur-p

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= -\int\limits_{\mathit{\Gamma}_t^{\mathrm{N}}} \hat{p}(t)\,\boldsymbol{n} \cdot \delta\boldsymbol{v} \,\mathrm{d}a
      \end{aligned}

-  body force:

   .. math::
      :label: deltap-ext-body

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Omega}_t} \hat{\boldsymbol{b}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
      \end{aligned}

| **Stabilization**
| Streamline-upwind Petrov-Galerkin/pressure-stabilizing Petrov-Galerkin
  (SUPG/PSPG) methods are implemented, either using the full or a
  reduced scheme
| – to the fluid FEM params, add the dict entry:

::

   "stabilization" : {"scheme" : <SCHEME>, "vscale" : 1e3, "dscales" : [<d1>,<d2>,<d3>],
                      "symmetric" : False, "reduced_scheme" : False}

| Full scheme according to :cite:p:`tezduyar2000`:
  ``"supg_pspg"``:
| – Velocity residual operator in
  (`[equation-fluid-weak-form] <#equation-fluid-weak-form>`__) is
  augmented by the following terms:

  .. math::
     \begin{aligned}
     r_v \leftarrow r_v &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_t} \tau_{\mathrm{SUPG}}\,(\nabla\delta\boldsymbol{v})\,\boldsymbol{v} \cdot \left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\nabla\boldsymbol{v})\,\boldsymbol{v}\right) - \nabla \cdot \boldsymbol{\sigma}(\boldsymbol{v},p)\right]\,\mathrm{d}v \\
     & + \int\limits_{\mathit{\Omega}_t} \tau_{\mathrm{LSIC}}\,\rho\,(\nabla\cdot\delta\boldsymbol{v})(\nabla\cdot\boldsymbol{v})\,\mathrm{d}v
     \end{aligned}

  – Pressure residual operator in
  (`[equation-fluid-weak-form] <#equation-fluid-weak-form>`__) is
  augmented by the following terms:

  .. math::
     \begin{aligned}
     r_p \leftarrow r_p &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_t} \tau_{\mathrm{PSPG}}\,(\nabla\delta p) \cdot \left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\nabla\boldsymbol{v})\,\boldsymbol{v}\right) - \nabla \cdot \boldsymbol{\sigma}(\boldsymbol{v},p)\right]\,\mathrm{d}v 
     \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n`:

.. math::
   :label: nonlin-sys-fluid

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{v}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`:

.. math::
   :label: lin-sys-fluid

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \end{bmatrix}_{n+1}^{k}
   \end{aligned}

– Note that :math:`\boldsymbol{\mathsf{K}}_{pp}` is zero for Taylor-Hood
elements (without stabilization)

ALE reference frame
~~~~~~~~~~~~~~~~~~~

| – Problem type: ``fluid_ale``
| – Incompressible Navier-Stokes equations in Arbitrary Lagrangian
  Eulerian (ALE) reference frame
| – ALE domain problem deformation governed by linear-elastic, nonlinear
  hyperelastic solid, or a diffusion problem, displacement field
  :math:`\boldsymbol{d}`
| – Fluid mechanics formulated with respect to the reference frame,
  using ALE deformation gradient
  :math:`\widehat{\boldsymbol{F}}(\boldsymbol{d}) = \boldsymbol{I} + \nabla_0\boldsymbol{d}`
  and its determinant,
  :math:`\widehat{J}(\boldsymbol{d})=\det \widehat{\boldsymbol{F}}(\boldsymbol{d})`
| **ALE problem**
| – Primary variable: domain displacement :math:`\boldsymbol{d}`
| – Strong form:

  .. math::
     :label: ale-strong-form

     \begin{aligned}
     \nabla_{0} \cdot \boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) &= \boldsymbol{0} &&\text{in} \; \mathit{\mathit{\Omega}}_0, \\
     \boldsymbol{d} &= \hat{\boldsymbol{d}} &&\text{on} \; \mathit{\mathit{\Gamma}}_0^{\mathrm{D}},
     \end{aligned}

  – ALE material ``linelast``:

  .. math::
     \begin{aligned}
     \boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) = 2\mu \,\boldsymbol{\varepsilon} + \lambda \,\mathrm{tr}\boldsymbol{\varepsilon}\,\boldsymbol{I}, \qquad \text{with}\quad \boldsymbol{\varepsilon} = \frac{1}{2}\left(\nabla_0\boldsymbol{d} + (\nabla_0\boldsymbol{d})^{\mathrm{T}}\right)
     \end{aligned}

  – ALE material ``diffusion``:

  .. math::
     \begin{aligned}
     \boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) = D \,\nabla_0\boldsymbol{d}
     \end{aligned}

  – ALE material ``neohooke`` (fully nonlinear model):

  .. math::
     \begin{aligned}
     \boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) = \frac{\partial \mathit{\Psi}}{\partial \widehat{\boldsymbol{F}}}, \qquad \text{with}\quad \mathit{\Psi} = \frac{\mu}{2}\left(\mathrm{tr}(\widehat{\boldsymbol{F}}^{\mathrm{T}}\widehat{\boldsymbol{F}}) - 3\right) + \frac{\mu}{2\beta} \left(\widehat{J}^{-2\beta} - 1\right)
     \end{aligned}

– weak form:

.. math::
   :label: ale-weak-form

   \begin{aligned}
   r_{d}(\boldsymbol{d};\delta\boldsymbol{d}) := \int\limits_{\mathit{\Omega}_0}\boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) : \nabla_{0}\delta\boldsymbol{d}\,\mathrm{d}V = 0, \quad \forall \; \delta\boldsymbol{d}
   \end{aligned}

| **Strong form (ALE)**
| – Primary variables: velocity :math:`\boldsymbol{v}`, pressure
  :math:`p`, and domain displacement :math:`\boldsymbol{d}`

  .. math::
     :label: fluid-ale-strong-form

     \begin{aligned}
     \nabla_{0} \boldsymbol{\sigma}(\boldsymbol{v},\boldsymbol{d},p) : \widehat{\boldsymbol{F}}^{-\mathrm{T}} + \hat{\boldsymbol{b}} &= \rho\left[\left.\frac{\partial\boldsymbol{v}}{\partial t}\right|_{\boldsymbol{x}_{0}} + (\nabla_0\boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1})\,(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right] &&\text{in} \; \mathit{\mathit{\Omega}}_0 \times [0, T],\\
     \nabla_{0}\boldsymbol{v} : \widehat{\boldsymbol{F}}^{-\mathrm{T}} &= 0 &&\text{in} \; \mathit{\mathit{\Omega}}_0 \times [0, T],\\
     \boldsymbol{v} &= \hat{\boldsymbol{v}} &&\text{on} \; \mathit{\mathit{\Gamma}}_0^{\mathrm{D}} \times [0, T], \\
     \boldsymbol{t} = \boldsymbol{\sigma}\boldsymbol{n} &= \hat{\boldsymbol{t}} &&\text{on} \; \mathit{\mathit{\Gamma}}_0^{\mathrm{N}} \times [0, T], \\
     \boldsymbol{v}(\boldsymbol{x},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}) &&\text{in} \; \mathit{\mathit{\Omega}}_0,
     \end{aligned}

with a Newtonian fluid constitutive law

.. math::
   \begin{aligned}
   \boldsymbol{\sigma} = -p \boldsymbol{I} + 2 \mu \boldsymbol{\gamma} = -p \boldsymbol{I} + \mu \left(\nabla_0 \boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1} + \widehat{\boldsymbol{F}}^{-\mathrm{T}}(\nabla_0 \boldsymbol{v})^{\mathrm{T}}\right)
   \end{aligned}

| **Weak form (ALE)**
| – Primary variables: velocity :math:`\boldsymbol{v}`, pressure
  :math:`p`, and domain displacement :math:`\boldsymbol{d}`
| – Principle of Virtual Power

  .. math::
     :label: fluid-ale-weak-form

     \begin{aligned}
     r_v(\boldsymbol{v},p,\boldsymbol{d};\delta\boldsymbol{v}) &:= \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v},\boldsymbol{d};\delta\boldsymbol{v}) + \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p,\boldsymbol{d};\delta\boldsymbol{v}) - \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{v},\boldsymbol{d};\delta\boldsymbol{v}) = 0, \quad \forall \; \delta\boldsymbol{v} \\
     r_p(\boldsymbol{v},\boldsymbol{d};\delta p) &:= \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v},\boldsymbol{d};\delta p), \quad \forall \; \delta p
     \end{aligned}

– Kinetic virtual power:

.. math::
   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v},\boldsymbol{d};\delta\boldsymbol{v}) = \int\limits_{\mathit{\Omega}_0} \widehat{J} \rho\left[\left.\frac{\partial\boldsymbol{v}}{\partial t}\right|_{\boldsymbol{x}_{0}} + (\nabla_{0}\boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1})\,(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right] \cdot \delta\boldsymbol{v} \,\mathrm{d}V
   \end{aligned}

– Internal virtual power:

.. math::
   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p,\boldsymbol{d};\delta\boldsymbol{v}) = 
   \int\limits_{\mathit{\Omega}_0} \widehat{J}\boldsymbol{\sigma}(\boldsymbol{v},p,\boldsymbol{d}) : \nabla_{0} \delta\boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1} \,\mathrm{d}V
   \end{aligned}

– Pressure virtual power:

.. math::
   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v},\boldsymbol{d};\delta p) = 
   \int\limits_{\mathit{\Omega}_0} \widehat{J}\,\nabla_{0}\boldsymbol{v} : \widehat{\boldsymbol{F}}^{-\mathrm{T}}\delta p\,\mathrm{d}V
   \end{aligned}

| – External virtual power:

-  conservative Neumann load:

   .. math::
      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Gamma}_0^{\mathrm{N}}} \hat{\boldsymbol{t}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}A
      \end{aligned}

-  pressure Neumann load:

   .. math::
      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{d};\delta\boldsymbol{v}) &= -\int\limits_{\mathit{\Gamma}_0^{\mathrm{N}}} \hat{p}(t)\,\widehat{J}\widehat{\boldsymbol{F}}^{-\mathrm{T}}\boldsymbol{n}_{0} \cdot \delta\boldsymbol{v} \,\mathrm{d}A 
      \end{aligned}

-  body force:

   .. math::
      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{d};\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Omega}_0} \widehat{J}\,\hat{\boldsymbol{b}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
      \end{aligned}

| **Stabilization (ALE)**
| ``"supg_pspg"``:
| – Velocity residual operator in
  (`[equation-fluid-ale-weak-form] <#equation-fluid-ale-weak-form>`__)
  is augmented by the following terms:

  .. math::
     \begin{aligned}
     r_v \leftarrow r_v &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_0}\widehat{J}\, \tau_{\mathrm{SUPG}}\,(\nabla_0\delta\boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1})\,\boldsymbol{v}\;\cdot \\
     & \qquad\quad \cdot\left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\nabla_0\boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1})\,(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right) - \nabla_{0} \boldsymbol{\sigma}(\boldsymbol{v},\boldsymbol{d},p) : \widehat{\boldsymbol{F}}^{-\mathrm{T}}\right]\,\mathrm{d}V \\
     & + \int\limits_{\mathit{\Omega}_0}\widehat{J}\, \tau_{\mathrm{LSIC}}\,\rho\,(\nabla_{0}\delta\boldsymbol{v} : \widehat{\boldsymbol{F}}^{-\mathrm{T}})(\nabla_{0}\boldsymbol{v} : \widehat{\boldsymbol{F}}^{-\mathrm{T}})\,\mathrm{d}V
     \end{aligned}

  – Pressure residual operator in
  (`[equation-fluid-ale-weak-form] <#equation-fluid-ale-weak-form>`__)
  is augmented by the following terms:

  .. math::
     \begin{aligned}
     r_p \leftarrow r_p &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_0}\widehat{J}\, \tau_{\mathrm{PSPG}}\,(\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_{0}\delta p) \;\cdot \\
     & \qquad\quad \cdot \left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\nabla_0\boldsymbol{v}\,\widehat{\boldsymbol{F}}^{-1})\,(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right) - \nabla_{0} \boldsymbol{\sigma}(\boldsymbol{v},\boldsymbol{d},p) : \widehat{\boldsymbol{F}}^{-\mathrm{T}}\right]\,\mathrm{d}V
     \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n`:

.. math::
   :label: nonlin-sys-fluid-ale

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{v}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`:

.. math::
   :label: lin-sys-fluid-ale

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{d} \end{bmatrix}_{n+1}^{k} 
   \end{aligned}

– note that :math:`\boldsymbol{\mathsf{K}}_{pp}` is zero for Taylor-Hood
elements (without stabilization)

0D flow: Lumped parameter models
--------------------------------

| – Example: Sec. `5.3 <#demo-0d-flow>`__ and ``demos/flow0d``
| – Problem type: ``flow0d``
| – 0D model concentrated elements are resistances (:math:`R`),
  impedances (:math:`Z`, technically are resistances), compliances
  (:math:`C`), inertances (:math:`L` or :math:`I`), and elastances
  (:math:`E`)
| – 0D variables are pressures (:math:`p`), fluxes (:math:`q` or
  :math:`Q`), or volumes (:math:`V`)

2-element Windkessel
~~~~~~~~~~~~~~~~~~~~

| – Model type : ``2elwindkessel``

4-element Windkessel (inertance parallel to impedance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Model type : ``4elwindkesselLpZ``

4-element Windkessel (inertance serial to impedance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Model type : ``4elwindkesselLsZ``

In-outflow CRL link
~~~~~~~~~~~~~~~~~~~

| – Model type : ``CRLinoutlink``

Systemic and pulmonary circulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Model type : ``syspul``
| – Allows to link in a coronary flow model

.. math::
   :label: syspul-1

   \begin{aligned}
   &\text{left heart and systemic circulation} && \nonumber\\
   &-Q_{\mathrm{at}}^{\ell} = \sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{pul}}}q_{\mathrm{ven},i}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell} && \text{left atrium flow balance}\\
   &q_{\mathrm{v,in}}^{\ell} = q_{\mathrm{mv}}(p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}) && \text{mitral valve momentum}\\
   &-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell} && \text{left ventricle flow balance}\\
   &q_{\mathrm{v,out}}^{\ell} = q_{\mathrm{av}}(p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}) && \text{aortic valve momentum}\\
   &-Q_{\mathrm{aort}}^{\mathrm{sys}} = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}} - \mathbb{I}^{\mathrm{cor}}\sum\limits_{i\in\{\ell,r\}}q_{\mathrm{cor,p,in}}^{\mathrm{sys},i} && \text{aortic root flow balance}\\
   &I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}} && \text{aortic root inertia}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}} && \text{systemic arterial flow balance}\\
   &L_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}} && \text{systemic arterial momentum}\\
   &C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{sys}}}q_{\mathrm{ven},i}^{\mathrm{sys}}\ && \text{systemic venous flow balance}\\
   &L_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven},i}^{\mathrm{sys}}\, q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at},i}^{r} && \text{systemic venous momentum}\nonumber\\
   &\qquad\qquad i \in \{1,...,n_{\mathrm{ven}}^{\mathrm{sys}}\} && 
   \end{aligned}

.. math::
   :label: syspul-2

   \begin{aligned}
   &\text{right heart and pulmonary circulation} && \\
   &-Q_{\mathrm{at}}^{r} = \sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{sys}}}q_{\mathrm{ven},i}^{\mathrm{sys}} + \mathbb{I}^{\mathrm{cor}} q_{\mathrm{cor,d,out}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r} && \text{right atrium flow balance}\\
   &q_{\mathrm{v,in}}^{r} = q_{\mathrm{tv}}(p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}) && \text{tricuspid valve momentum}\\
   &-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r} && \text{right ventricle flow balance}\\
   &q_{\mathrm{v,out}}^{r} = q_{\mathrm{pv}}(p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}) && \text{pulmonary valve momentum}\\
   &C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}} && \text{pulmonary arterial flow balance}\\
   &L_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{ven}}^{\mathrm{pul}} && \text{pulmonary arterial momentum}\\
   &C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - \sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{pul}}}q_{\mathrm{ven},i}^{\mathrm{pul}} && \text{pulmonary venous flow balance}\nonumber\\
   &L_{\mathrm{ven},i}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ven},i}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven},i}^{\mathrm{pul}}\, q_{\mathrm{ven},i}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at},i}^{\ell} && \text{pulmonary venous momentum}\\
   &\qquad\qquad i \in \{1,...,n_{\mathrm{ven}}^{\mathrm{pul}}\} && 
   \end{aligned}

with:

.. math::
   \begin{aligned}
   Q_{\mathrm{at}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{at}}^{\ell}}{\mathrm{d}t}, \quad
   Q_{\mathrm{v}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{v}}^{\ell}}{\mathrm{d}t}, \quad
   Q_{\mathrm{at}}^{r} := -\frac{\mathrm{d}V_{\mathrm{at}}^{r}}{\mathrm{d}t}, \quad
   Q_{\mathrm{v}}^{r} := -\frac{\mathrm{d}V_{\mathrm{v}}^{r}}{\mathrm{d}t},
   \quad
   Q_{\mathrm{aort}}^{\mathrm{sys}} := -\frac{\mathrm{d}V_{\mathrm{aort}}^{\mathrm{sys}}}{\mathrm{d}t}
   \end{aligned}

and:

.. math::
   \begin{aligned}
   \mathbb{I}^{\mathrm{cor}} = \begin{cases} 1, & \text{if \, CORONARYMODEL}, \\ 0, & \text{else} \end{cases}
   \end{aligned}

The volume :math:`V` of the heart chambers (0D) is modeled by the
volume-pressure relationship

.. math::
   \begin{aligned}
   V(t) = \frac{p}{E(t)} + V_{\mathrm{u}},
   \end{aligned}

with the unstressed volume :math:`V_{\mathrm{u}}` and the time-varying
elastance

.. math::
   :label: at-elast

   \begin{aligned}
   E(t)=\left(E_{\mathrm{max}}-E_{\mathrm{min}}\right)\cdot \hat{y}(t)+E_{\mathrm{min}},
   \end{aligned}

where :math:`E_{\mathrm{max}}` and :math:`E_{\mathrm{min}}` denote the
maximum and minimum elastance, respectively. The normalized activation
function :math:`\hat{y}(t)` is input by the user.

Flow-pressure relations for the four valves, eq.
(`[eq:mv_flow] <#eq:mv_flow>`__), (`[eq:av_flow] <#eq:av_flow>`__),
(`[eq:tv_flow] <#eq:tv_flow>`__), (`[eq:pv_flow] <#eq:pv_flow>`__), are
functions of the pressure difference :math:`p-p_{\mathrm{open}}` across
the valve. The following valve models can be defined:

Valve model ``pwlin_pres``:

.. math::
   \begin{aligned}
   q(p-p_{\mathrm{open}}) = \frac{p-p_{\mathrm{open}}}{\tilde{R}}, \quad \text{with}\; \tilde{R} = \begin{cases} R_{\max}, & p < p_{\mathrm{open}} \\
   R_{\min}, & p \geq p_{\mathrm{open}} \end{cases}
   \end{aligned}

**Remark:** Non-smooth flow-pressure relationship

Valve model ``pwlin_time``:

.. math::
   \begin{aligned}
   q(p-p_{\mathrm{open}}) = \frac{p-p_{\mathrm{open}}}{\tilde{R}},\quad \text{with}\; \tilde{R} = \begin{cases} \begin{cases} R_{\max}, & t < t_{\mathrm{open}} \;\text{and}\; t \geq t_{\mathrm{close}} \\
   R_{\min}, & t \geq t_{\mathrm{open}} \;\text{or}\; t < t_{\mathrm{close}} \end{cases}, & t_{\mathrm{open}} > t_{\mathrm{close}} \\ \begin{cases} R_{\max}, & t < t_{\mathrm{open}} \;\text{or}\; t \geq t_{\mathrm{close}} \\
   R_{\min}, & t \geq t_{\mathrm{open}} \;\text{and}\; t < t_{\mathrm{close}} \end{cases}, & \text{else} \end{cases}
   \end{aligned}

**Remark:** Non-smooth flow-pressure relationship with resistance only
dependent on timings, not the pressure difference!

Valve model ``smooth_pres_resistance``:

.. math::
   \begin{aligned}
   q(p-p_{\mathrm{open}}) = \frac{p-p_{\mathrm{open}}}{\tilde{R}},\quad \text{with}\;\tilde{R} = 0.5\left(R_{\max}-R_{\min}\right)\left(\tanh\frac{p-p_{\mathrm{open}}}{\epsilon}+1\right) + R_{\min}
   \end{aligned}

**Remark:** Smooth but potentially non-convex flow-pressure
relationship!

Valve model ``smooth_pres_momentum``:

.. math::
   \begin{aligned}
   q(p-p_{\mathrm{open}}) = \begin{cases}\frac{p-p_{\mathrm{open}}}{R_{\max}}, & p < p_{\mathrm{open}}-0.5\epsilon \\ h_{00}p_{0} + h_{10}m_{0}\epsilon + h_{01}p_{1} + h_{11}m_{1}\epsilon, & p \geq p_{\mathrm{open}}-0.5\epsilon \;\text{and}\; p < p_{\mathrm{open}}+0.5\epsilon \\ \frac{p-p_{\mathrm{open}}}{R_{\min}}, & p \geq p_{\mathrm{open}}+0.5\epsilon  \end{cases}
   \end{aligned}

with

.. math::
   \begin{aligned}
   p_{0}=\frac{-0.5\epsilon}{R_{\max}}, \qquad m_{0}=\frac{1}{R_{\max}}, \qquad && p_{1}=\frac{0.5\epsilon}{R_{\min}}, \qquad m_{1}=\frac{1}{R_{\min}} 
   \end{aligned}

and

.. math::
   \begin{aligned}
   h_{00}=2s^3 - 3s^2 + 1, &\qquad h_{01}=-2s^3 + 3s^2, \\
   h_{10}=s^3 - 2s^2 + s, &\qquad h_{11}=s^3 - s^2 
   \end{aligned}

with

.. math::
   \begin{aligned}
   s=\frac{p-p_{\mathrm{open}}+0.5\epsilon}{\epsilon} 
   \end{aligned}

| **Remarks:**
| – Collapses to valve model ``pwlin_pres`` for :math:`\epsilon=0`
| – Smooth and convex flow-pressure relationship
| Valve model ``pw_pres_regurg``:

  .. math::
     \begin{aligned}
     q(p-p_{\mathrm{open}}) = \begin{cases} c A_{\mathrm{o}} \sqrt{p-p_{\mathrm{open}}}, & p < p_{\mathrm{open}} \\ \frac{p-p_{\mathrm{open}}}{R_{\min}}, & p \geq p_{\mathrm{open}}  \end{cases}
     \end{aligned}

  **Remark:** Model to allow a regurgitant valve in the closed state,
  degree of regurgitation can be varied by specifying the valve
  regurgitant area :math:`A_{\mathrm{o}}`

| – Coronary circulation model: ``ZCRp_CRd_lr``
| 

  .. math::
     \begin{aligned}
     &C_{\mathrm{cor,p}}^{\mathrm{sys},\ell} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys},\ell}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell} - q_{\mathrm{cor,p}}^{\mathrm{sys},\ell} && \text{left coronary proximal flow balance}\\
     &R_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,p}}^{\mathrm{sys},\ell}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys},\ell} - Z_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell} && \text{left coronary proximal momentum}\\
     &C_{\mathrm{cor,d}}^{\mathrm{sys},\ell} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys},\ell}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys},\ell} - q_{\mathrm{cor,d}}^{\mathrm{sys},\ell} && \text{left coronary distal flow balance}\\
     &R_{\mathrm{cor,d}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}=p_{\mathrm{cor,d}}^{\mathrm{sys},\ell}-p_{\mathrm{at}}^{r} && \text{left coronary distal momentum}\\
     &C_{\mathrm{cor,p}}^{\mathrm{sys},r} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys},r}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys},r}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys},r}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys},r} - q_{\mathrm{cor,p}}^{\mathrm{sys},r} && \text{right coronary proximal flow balance}\\
     &R_{\mathrm{cor,p}}^{\mathrm{sys},r}\,q_{\mathrm{cor,p}}^{\mathrm{sys},r}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys},r} - Z_{\mathrm{cor,p}}^{\mathrm{sys},r}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys},r} && \text{right coronary proximal momentum}\\
     &C_{\mathrm{cor,d}}^{\mathrm{sys},r} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys},r}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys},r} - q_{\mathrm{cor,d}}^{\mathrm{sys},r} && \text{right coronary distal flow balance}\nonumber\\
     &R_{\mathrm{cor,d}}^{\mathrm{sys},r}\,q_{\mathrm{cor,d}}^{\mathrm{sys},r}=p_{\mathrm{cor,d}}^{\mathrm{sys},r}-p_{\mathrm{at}}^{r} && \text{right coronary distal momentum}\\
     &0=q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}+q_{\mathrm{cor,d}}^{\mathrm{sys},r}-q_{\mathrm{cor,d,out}}^{\mathrm{sys}} && \text{distal coronary junction flow balance}
     \end{aligned}

| – Coronary circulation model: ``ZCRp_CRd``
| 

  .. math::
     \begin{aligned}
     &C_{\mathrm{cor,p}}^{\mathrm{sys}} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys}}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys}} - q_{\mathrm{cor,p}}^{\mathrm{sys}} && \text{coronary proximal flow balance}\\
     &R_{\mathrm{cor,p}}^{\mathrm{sys}}\,q_{\mathrm{cor,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys}} - Z_{\mathrm{cor,p}}^{\mathrm{sys}}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys}} && \text{coronary proximal momentum}\nonumber\\
     &C_{\mathrm{cor,d}}^{\mathrm{sys}} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys}}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys}} - q_{\mathrm{cor,d}}^{\mathrm{sys}} && \text{coronary distal flow balance}\\
     &R_{\mathrm{cor,d}}^{\mathrm{sys}}\,q_{\mathrm{cor,d}}^{\mathrm{sys}}=p_{\mathrm{cor,d}}^{\mathrm{sys}}-p_{\mathrm{at}}^{r} && \text{coronary distal momentum}
     \end{aligned}

Systemic and pulmonary circulation, including capillary flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Model type : ``syspulcap``, cf.
  :cite:p:`hirschvogel2019disspub`, p. 51ff.
| 

  .. math::
     :label: syspulcap-1

     \begin{aligned}
     &-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell} && \text{left atrium flow balance}\nonumber\\
     &\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell} && \text{mitral valve momentum}\nonumber\\
     &-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell} && \text{left ventricle flow balance}\nonumber\\
     &\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}} && \text{aortic valve momentum}\nonumber\\
     &0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}} && \text{aortic root flow balance}\nonumber\\
     &I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}} && \text{aortic root inertia}\nonumber\\
     &C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}} && \text{systemic arterial flow balance}\nonumber\\
     &L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}} -p_{\mathrm{ar,peri}}^{\mathrm{sys}} && \text{systemic arterial momentum}\nonumber\\
     &\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}} && \text{systemic capillary arterial flow balance}\nonumber\\
     &R_{\mathrm{ar},i}^{\mathrm{sys}}\,q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}} && \text{systemic capillary arterial momentum}\nonumber\\
     &C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}&& \text{systemic capillary venous flow balance}\nonumber\\
     &R_{\mathrm{ven},i}^{\mathrm{sys}}\,q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}} && \text{systemic capillary venous momentum}\nonumber\\
     &C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer,cor}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}} && \text{systemic venous flow balance}\nonumber\\
     &L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r} && \text{systemic venous momentum}\nonumber
     \end{aligned}

.. math::
   :label: syspulcap-2

   \begin{aligned}
   &-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r} && \text{right atrium flow balance}\\
   &\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r} && \text{tricuspid valve momentum}\\
   &-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r} && \text{right ventricle flow balance}\nonumber\\
   &\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}} && \text{pulmonary valve momentum}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}} && \text{pulmonary arterial flow balance}\\
   &L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{cap}}^{\mathrm{pul}} && \text{pulmonary arterial momentum}\\
   &C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}} && \text{pulmonary capillary flow balance}\\
   &R_{\mathrm{cap}}^{\mathrm{pul}}\,q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}} && \text{pulmonary capillary momentum}\\
   &C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}} && \text{pulmonary venous flow balance}\\
   &L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell} && \text{pulmonary venous momentum}
   \end{aligned}

with:

.. math::
   \begin{aligned}
   Q_{\mathrm{at}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{at}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{v}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{at}}^{r} := -\frac{\mathrm{d}V_{\mathrm{at}}^{r}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{r} := -\frac{\mathrm{d}V_{\mathrm{v}}^{r}}{\mathrm{d}t}\nonumber
   \end{aligned}

Systemic and pulmonary circulation, including capillary and coronary flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Model type : ``syspulcapcor``
| – Variant of ``syspulcap``, with coronaries branching off directly
  after aortic valve

.. math::
   :label: syspulcapcor-1

   \begin{aligned}
   &-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell} && \text{left atrium flow balance}\\
   &\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell} && \text{mitral valve momentum}\nonumber\\
   &-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell} && \text{left ventricle flow balance}\\
   &\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}} && \text{aortic valve momentum}\nonumber\\
   &0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar,cor,in}}^{\mathrm{sys}} && \text{aortic root flow balance}\\
   &I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}} && \text{aortic root inertia}\nonumber\\
   &C_{\mathrm{ar,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor,in}}^{\mathrm{sys}} - q_{\mathrm{ar,cor}}^{\mathrm{sys}} && \text{systemic arterial coronary flow balance}\nonumber\\
   &R_{\mathrm{ar,cor}}^{\mathrm{sys}}\,q_{\mathrm{ar,cor}}^{\mathrm{sys}} = p_{\mathrm{ar}}^{\mathrm{sys}} - p_{\mathrm{ven,cor}}^{\mathrm{sys}} && \text{systemic arterial coronary momentum}\\
   &C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}}&& \text{systemic arterial flow balance}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}} -p_{\mathrm{ar,peri}}^{\mathrm{sys}} && \text{systemic arterial flow balance}\\
   &\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}} && \text{systemic arterial capillary flow balance}\\
   &R_{\mathrm{ar},i}^{\mathrm{sys}}\,q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}} && \text{systemic arterial capillary momentum}\\
   &C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}} && \text{systemic venous capillary flow balance}\nonumber\\
   &R_{\mathrm{ven},i}^{\mathrm{sys}}\,q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}} && \text{systemic venous capillary momentum}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}} && \text{systemic venous flow balance}\\
   &L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r} && \text{systemic venous momentum}\nonumber\\
   &C_{\mathrm{ven,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven,cor}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor}}^{\mathrm{sys}}-q_{\mathrm{ven,cor}}^{\mathrm{sys}} && \text{systemic venous coronary flow balance}\nonumber\\
   &R_{\mathrm{ven,cor}}^{\mathrm{sys}}\,q_{\mathrm{ven,cor}}^{\mathrm{sys}} = p_{\mathrm{ven,cor}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r} && \text{systemic venous coronary momentum}
   \end{aligned}

.. math::
   :label: syspulcapcor-2

   \begin{aligned}
   &-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} + q_{\mathrm{ven,cor}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r} && \text{right atrium flow balance}\nonumber\\
   &\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r} && \text{tricuspid valve momentum}\\
   &-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r} && \text{right ventricle flow balance}\\
   &\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}} && \text{pulmonary valve momentum}\\
   &C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}} && \text{pulmonary arterial flow balance}\\
   &L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{cap}}^{\mathrm{pul}} && \text{pulmonary arterial momentum}\\
   &C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}} && \text{pulmonary capillary flow balance}\\
   &R_{\mathrm{cap}}^{\mathrm{pul}}\,q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}} && \text{pulmonary capillary momentum}\\
   &C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}} && \text{pulmonary venous flow balance}\\
   &L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell} && \text{pulmonary venous momentum}
   \end{aligned}

with:

.. math::
   \begin{aligned}
   Q_{\mathrm{at}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{at}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{v}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{at}}^{r} := -\frac{\mathrm{d}V_{\mathrm{at}}^{r}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{r} := -\frac{\mathrm{d}V_{\mathrm{v}}^{r}}{\mathrm{d}t} 
   \end{aligned}

Systemic and pulmonary circulation, capillary flow, and respiratory (gas transport + dissociation) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Model type : ``syspulcaprespir``
| – Model equations described in
  :cite:p:`hirschvogel2019disspub`, p. 51ff., 58ff.

Multi-physics coupling
----------------------

Solid + 0D flow
~~~~~~~~~~~~~~~

| – Example: Sec. `4.4.1 <#solid-0d-flow>`__ and ``demos/solid_flow0d``
| – Problem type: ``solid_flow0d``
| – solid momentum in
  (`[equation-solid-weak-form] <#equation-solid-weak-form>`__) or in
  (`[equation-solid-weak-form-inc] <#equation-solid-weak-form-inc>`__)
  augmented by following term:

  .. math::
     \begin{aligned}
     r_u \leftarrow r_u + \int\limits_{\mathit{\Gamma}_0^{\text{s}\text{-}\mathrm{0d}}}\!\mathit{\Lambda}\,J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\cdot\delta\boldsymbol{u}\,\mathrm{d}A
     \end{aligned}

– Multiplier constraint

.. math::
   \begin{aligned}
   r_{\mathit{\Lambda}}(\mathit{\Lambda},\boldsymbol{u};\delta\mathit{\Lambda}):= \left(\int\limits_{\mathit{\Gamma}_0^{\mathrm{\text{s}\text{-}0d}}}\! J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\boldsymbol{v}(\boldsymbol{u})\,\mathrm{d}A - Q^{\mathrm{0d}}(\mathit{\Lambda})\right) \delta\mathit{\Lambda}, \quad \forall \; \delta\mathit{\Lambda}
   \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n` for
displacement-based solid:

.. math::
   :label: nonlin-sys-solid-0d

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{\Lambda}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{u}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k` for
displacement-based solid:

.. math::
   :label: lin-sys-solid-0d

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{u\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}u} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n` for
incompressible solid:

.. math::
   :label: nonlin-sys-solid-inc-0d

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\Lambda}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{u}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{u}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k` for
incompressible solid:

.. math::
   :label: lin-sys-solid-inc-0d

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{up} & \boldsymbol{\mathsf{K}}_{u\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{K}}_{pu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\  \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}u} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

– sub-solves: 0D model has to hold true in each nonlinear iteration
:math:`k`:

.. math::
   :label: nonlin-lin-sys-0d

   \begin{aligned}
   \left.\boldsymbol{\mathsf{r}}^{0\mathrm{D}}(\boldsymbol{\mathsf{\Lambda}})\right|_{k} = \boldsymbol{\mathsf{0}}, \qquad \rightsquigarrow \boldsymbol{\mathsf{K}}_{k}^{0\mathrm{D},j}\Delta\boldsymbol{\mathsf{q}}_{k}^{j+1} = -\boldsymbol{\mathsf{r}}_{k}^{0\mathrm{D},j}
   \end{aligned}

Fluid + 0D flow
~~~~~~~~~~~~~~~

| – Example: Sec. `4.4.2 <#fluid-0d-flow>`__ ``demos/fluid_flow0d``
| – Problem type: ``fluid_flow0d``
| – fluid momentum in
  (`[equation-fluid-weak-form] <#equation-fluid-weak-form>`__) augmented
  by following term:

  .. math::
     \begin{aligned}
     r_v \leftarrow r_v + \int\limits_{\mathit{\Gamma}_t^{\text{f}\text{-}\mathrm{0d}}}\!\mathit{\Lambda}\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
     \end{aligned}

– Multiplier constraint

.. math::
   \begin{aligned}
   r_{\mathit{\Lambda}}(\mathit{\Lambda},\boldsymbol{v};\delta\mathit{\Lambda}):= \left(\int\limits_{\mathit{\Gamma}_t^{\mathrm{\text{f}\text{-}0d}}}\! \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}a - Q^{\mathrm{0d}}(\mathit{\Lambda})\right) \delta\mathit{\Lambda}, \quad \forall \; \delta\mathit{\Lambda}
   \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n`:

.. math::
   :label: nonlin-sys-fluid-0d

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{v}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\Lambda}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{v}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`:

.. math::
   :label: lin-sys-fluid-0d

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\  \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

ALE fluid + 0D flow
~~~~~~~~~~~~~~~~~~~

| – Problem type: ``fluid_ale_flow0d``
| – fluid momentum in
  (`[equation-fluid-ale-weak-form] <#equation-fluid-ale-weak-form>`__)
  augmented by following term:

  .. math::
     \begin{aligned}
     r_v \leftarrow r_v + \int\limits_{\mathit{\Gamma}_0^{\text{f}\text{-}\mathrm{0d}}}\!\mathit{\Lambda}\,\widehat{J}\widehat{\boldsymbol{F}}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{v}\,\mathrm{d}A
     \end{aligned}

– Multiplier constraint

.. math::
   \begin{aligned}
   r_{\mathit{\Lambda}}(\mathit{\Lambda},\boldsymbol{v},\boldsymbol{d};\delta\mathit{\Lambda}):= \left(\int\limits_{\mathit{\Gamma}_0^{\mathrm{\text{f}\text{-}0d}}}\! \widehat{J}\widehat{\boldsymbol{F}}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot(\boldsymbol{v}-\widehat{\boldsymbol{w}}(\boldsymbol{d}))\,\mathrm{d}A - Q^{\mathrm{0d}}(\mathit{\Lambda})\right) \delta\mathit{\Lambda}, \quad \forall \; \delta\mathit{\Lambda}
   \end{aligned}

| with
  :math:`\boldsymbol{w}(\boldsymbol{d})=\frac{\mathrm{d}\boldsymbol{d}}{\mathrm{d}t}`
| – Discrete nonlinear system to solve in each time step :math:`n`:

  .. math::
     :label: nonlin-sys-fluid-ale-0d

     \begin{aligned}
     \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{v}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
     \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`:

.. math::
   :label: lin-sys-fluid-ale-0d

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}d} \\ \\ \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

Fluid-Solid Interaction (FSI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Example: Sec. `5.6 <#demo-fsi>`__ and ``demos/fsi``
| – Problem type: ``fsi``
| – Solid momentum in
  (`[equation-solid-weak-form] <#equation-solid-weak-form>`__) or
  (`[equation-solid-weak-form-inc] <#equation-solid-weak-form-inc>`__)
  augmented by following term:

  .. math::
     \begin{aligned}
     r_u \leftarrow r_u + \int\limits_{\mathit{\Gamma}_0^{\text{f}\text{-}\text{s}}}\boldsymbol{\lambda}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
     \end{aligned}

  – Fluid momentum in
  (`[equation-fluid-ale-weak-form] <#equation-fluid-ale-weak-form>`__)
  is augmented by following term:

  .. math::
     \begin{aligned}
     r_v \leftarrow r_v - \int\limits_{\mathit{\Gamma}_0^{\text{f}\text{-}\text{s}}}\boldsymbol{\lambda}\cdot\delta\boldsymbol{v}\,\mathrm{d}A
     \end{aligned}

| Note the different signs (actio=reactio!)
| – Lagrange multiplier constraint:

-  “solid-governed”:

   .. math::
      \begin{aligned}
      r_{\lambda}(\boldsymbol{u},\boldsymbol{v};\delta\boldsymbol{\lambda}):= \int\limits_{\mathit{\Gamma}_0^{\mathrm{\text{f}\text{-}\text{s}}}} \left(\boldsymbol{u} - \boldsymbol{u}_{\mathrm{f}}(\boldsymbol{v})\right)\cdot\delta\boldsymbol{\lambda}\,\mathrm{d}A, \quad \forall \; \delta\boldsymbol{\lambda}
      \end{aligned}

-  “fluid-governed”:

   .. math::
      \begin{aligned}
      r_{\lambda}(\boldsymbol{v},\boldsymbol{u};\delta\boldsymbol{\lambda}):= \int\limits_{\mathit{\Gamma}_0^{\mathrm{\text{f}\text{-}\text{s}}}} \left(\boldsymbol{v} - \frac{\mathrm{d} \boldsymbol{u}}{\mathrm{d} t}\right)\cdot\delta\boldsymbol{\lambda}\,\mathrm{d}A, \quad \forall \; \delta\boldsymbol{\lambda}
      \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n` for
displacement-based solid:

.. math::
   :label: nonlin-sys-fsi

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{\lambda}}) \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\lambda}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{\lambda}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{v}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n` for
incompressible solid:

.. math::
   :label: nonlin-sys-fsi-inc

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{p}}^{\mathrm{s}},\boldsymbol{\mathsf{\lambda}}) \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{s}}(\boldsymbol{\mathsf{u}}) \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\lambda}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{\lambda}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{v}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k` for
displacement-based solid:

.. math::
   :label: lin-sys-fsi

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{u\lambda} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\lambda} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\lambda u} & \boldsymbol{\mathsf{K}}_{\lambda v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}&  \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}} \\ \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{\lambda} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k` for
incompressible solid:

.. math::
   :label: lin-sys-fsi-inc

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{up} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{u\lambda} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \boldsymbol{\mathsf{K}}_{pu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\lambda} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\lambda u} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\lambda v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}&  \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}} \\ \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{s}} \\ \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{\lambda} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

Fluid-Solid Interaction (FSI) + 0D flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| – Problem type: ``fsi_flow0d``
| – Discrete nonlinear system to solve in each time step :math:`n` for
  displacement-based solid:

  .. math::
     :label: nonlin-sys-fsi-0d

     \begin{aligned}
     \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{\lambda}}) \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\lambda}},\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{\lambda}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{v}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
     \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n` for
incompressible solid:

.. math::
   :label: nonlin-sys-fsi-0d-inc

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{p}}^{\mathrm{s}},\boldsymbol{\mathsf{\lambda}}) \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{s}}(\boldsymbol{\mathsf{u}}) \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}}(\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\lambda}},\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{\lambda}(\boldsymbol{\mathsf{u}},\boldsymbol{\mathsf{v}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{v}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k` for
displacement-based solid:

.. math::
   :label: lin-sys-fsi-0d

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{u\lambda} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\lambda} & \boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\lambda u} & \boldsymbol{\mathsf{K}}_{\lambda v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}d} \\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}&  \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\lambda}}\\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}} \\ \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{\lambda} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k` for
incompressible solid:

.. math::
   :label: lin-sys-fsi-0d-inc

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{up} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{u\lambda} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \boldsymbol{\mathsf{K}}_{pu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\lambda} & \boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\lambda u} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\lambda v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}d} \\ \\ \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}&  \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\lambda}}\\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u}^{\mathrm{s}} \\ \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{s}} \\ \\ \boldsymbol{\mathsf{r}}_{v}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{p}^{\mathrm{f}} \\ \\ \boldsymbol{\mathsf{r}}_{\lambda} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

Fluid-reduced-Solid Interaction (FrSI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

– Boundary term:

.. math::
   \begin{aligned}
       r_{\tilde{\text{s}}} := \int\limits_{\mathit{\Gamma}_{0}^{\text{f}\mhyphen\tilde{\text{s}}}} h_{0} \left[\rho_{0,s}\frac{\partial\boldsymbol{v}}{\partial t}\cdot\delta\boldsymbol{v} + \tilde{\boldsymbol{P}}(\boldsymbol{u}_{\text{f}}(\boldsymbol{v}) + \tilde{\boldsymbol{u}}_{\mathrm{pre}},\boldsymbol{v}) : \tilde{\nabla}_{0}\delta\boldsymbol{v}\right]\mathrm{d}A
   \end{aligned}

– Discrete nonlinear system to solve in each time step :math:`n`:

.. math::
   :label: nonlin-sys-frsi-0d

   \begin{aligned}
   \boldsymbol{\mathsf{r}}_{n+1} = \begin{bmatrix} \boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}^\mathrm{T}}\boldsymbol{\mathsf{r}}_{v}(\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}}\tilde{\boldsymbol{\mathsf{v}}},\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{p}(\boldsymbol{\mathsf{p}},\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}}\tilde{\boldsymbol{\mathsf{v}}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}(\boldsymbol{\mathsf{\Lambda}},\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}}\tilde{\boldsymbol{\mathsf{v}}},\boldsymbol{\mathsf{d}}) \\ \boldsymbol{\mathsf{r}}_{d}(\boldsymbol{\mathsf{d}},\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}}\tilde{\boldsymbol{\mathsf{v}}}) \end{bmatrix}_{n+1} = \boldsymbol{\mathsf{0}}
   \end{aligned}

– Discrete linear system to solve in each Newton iteration :math:`k`:

.. math::
   :label: lin-sys-frsi-0d

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}^\mathrm{T}}\boldsymbol{\mathsf{K}}_{vv}\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}} & \boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}^\mathrm{T}}\boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}^\mathrm{T}}\boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} & \boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}^\mathrm{T}}\boldsymbol{\mathsf{K}}_{vd} \\ \\ \boldsymbol{\mathsf{K}}_{pv}\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v}\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}d} \\ \\ \boldsymbol{\mathsf{K}}_{dv}\boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\tilde{\boldsymbol{\mathsf{v}}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{V}}_{v}^{\mathit{\Gamma}^\mathrm{T}}\boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k}
   \end{aligned}

Demos
=====

Demo: Solid
-----------

| – Physics description given in sec. `4.1 <#solid-mechanics>`__
| – Input files: ``demos/solid/``

Cantilever under tip load
~~~~~~~~~~~~~~~~~~~~~~~~~

| This example demonstrates how to set up a quasi-static solid mechanics
  elasticity problem. The deformation of a steel cantilever under
  transverse conservative load is simulated. The structure is fixed on
  one end. Quadratic 27-node hexahedral finite elements are used for the
  discretization of the domain. The well-known St. Venant-Kirchhoff
  material is used as constitutive law, which is a generalization of
  Hooke’s law to the nonlinear realm.

.. figure:: fig/cantilever_setup.png
   :name: fig:cantilever_setup
   :width: 85.0%

   Cantilever, problem setup.

Study the setup shown in fig. `1 <#fig:cantilever_setup>`__ and the
comments in the input file ``solid_cantilever.py`` Run the simulation,
either in one of the provided Docker containers or using your own
FEniCSx/Ambit installation, using the command

::

   mpiexec -n 1 python3 solid_cantilever.py

It is fully sufficient to use one core (``mpiexec -n 1``) for the
presented setup.

Open the results file ``results_solid_cantilever_displacement.xdmf`` in
Paraview, and visualize the deformation over time.

Figure `2 <#fig:cantilever_results>`__ shows the displacement magnitude
at the end of the simulation.

.. figure:: fig/cantilever_results.png
   :name: fig:cantilever_results
   :width: 85.0%

   Cantilever, tip deformation. Color shows displacement magnitude.

Demo: Fluid
-----------

| – Physics description given in sec. `4.2 <#fluid-mechanics>`__
| – Input files: ``demos/fluid``

2D channel flow
~~~~~~~~~~~~~~~

This example shows how to set up 2D fluid flow in a channel around a
rigid obstacle. Incompressible Navier-Stokes flow is solved using
Taylor-Hood elements (9-node biquadratic quadrilaterals for the
velocity, 4-node bilinear quadrilaterals for the pressure).

.. figure:: fig/channel_setup.png
   :name: fig:channel_setup
   :width: 90.0%

   Channel flow, problem setup.

Study the setup and the comments in the input file ``fluid_channel.py``.
Run the simulation, either in one of the provided Docker containers or
using your own FEniCSx/Ambit installation, using the command

::

   mpiexec -n 1 python3 fluid_channel.py

It is fully sufficient to use one core (``mpiexec -n 1``) for the
presented setup.

| Open the results file ``results_fluid_channel_velocity.xdmf`` and
| ``results_fluid_channel_pressure.xdmf`` in Paraview, and visualize the
  velocity as well as the pressure over time.

Fig. `4 <#fig:channel_results>`__ shows the velocity magnitude (top) as
well as the pressure (bottom part) at the end of the simulation.

.. figure:: fig/channel_results.png
   :name: fig:channel_results
   :width: 90.0%

   Velocity magnitude (top part) and pressure (bottom part) at end of
   simulation.

Demo: 0D flow
-------------

| – Physics description given in sec.
  `4.3 <#d-flow-lumped-parameter-models>`__
| – Input files: ``demos/flow0d``

.. _systemic-and-pulmonary-circulation-1:

Systemic and pulmonary circulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to simulate a cardiac cycle using a
lumped-parameter (0D) model for the heart chambers and the entire
circulation. Multiple heart beats are run until a periodic state
criterion is met (which compares variable values at the beginning to
those at the end of a cycle, and stops if the relative change is less
than a specified value, here :literal:`\`eps_periodic'` in the
``TIME_PARAMS`` dictionary). The problem is set up such that periodicity
is reached after 5 heart cycles.

.. figure:: fig/syspul_setup.png
   :name: fig:syspul_setup
   :width: 65.0%

   0D heart, systemic and pulmonary circulation, problem setup.

Study the setup in fig. `5 <#fig:syspul_setup>`__ and the comments in
the input file ``flow0d_heart_cycle.py``. Run the simulation, either in
one of the provided Docker containers or using your own FEniCSx/Ambit
installation, using the command

::

   python3 flow0d_heart_cycle.py

For postprocessing of the time courses of pressures, volumes, and fluxes
of the 0D model, either use your own tools to plot the text output files
(first column is time, second is the respective quantity), or make sure
to have Gnuplot (and TeX) installed and navigate to the output folder
(``tmp/``) in order to execute the script ``flow0d_plot.py`` (which lies
in ``ambit/src/ambit_fe/postprocess/``):

::

   flow0d_plot.py -s flow0d_heart_cycle -n 100

| A folder ``plot_flow0d_heart_cycle`` is created inside ``tmp/``. Look
  at the results of pressures (:math:`p`), volumes (:math:`V`), and
  fluxes (:math:`q`, :math:`Q`) over time. Subscripts ``v``, ``at``,
  ``ar``, ``ven`` refer to ‘ventricular’, ‘atrial’, ‘arterial’, and
  ‘venous’, respectively. Superscripts ``l``, ``r``, ``sys``, ``pul``
  refer to ‘left’, ‘right’, ‘systemic’, and ‘pulmonary’, respectively.
  Try to understand the time courses of the respective pressures, as
  well as the plots of ventricular pressure over volume. Check that the
  overall system volume is constant and around 4-5 liters.
| The solution is depicted in fig. `6 <#fig:syspul_results>`__, showing
  the time course of volumes and pressures of the circulatory system.

.. figure:: fig/syspul_results.png
   :name: fig:syspul_results
   :width: 100.0%

   A. Left heart and systemic pressures over time. B. Right heart and
   pulmonary pressures over time. C. Left and right ventricular and
   atrial volumes over time. D. Left and right ventricular
   pressure-volume relationships of periodic (5th) cycle.

Demo: Solid + 0D flow
---------------------

| – Physics description given in sec. `4.4.1 <#solid-0d-flow>`__
| – Input files: ``demos/solid_flow0d``

3D heart, coupled to systemic and pulmonary circulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| This example demonstrates how to set up and simulate a two-chamber
  (left and right ventricular) solid mechanics heart model coupled to a
  closed-loop 0D circulatory system. A full dynamic heart cycle of
  duration 1 s is simulated, where the active contraction is modeled by
  a prescribed active stress approach. Passive material behavior of the
  heart muscle is governed by the Holzapfel-Ogden anisotropic strain
  energy function :cite:p:`holzapfel2009` and a strain
  rate-dependent viscous model :cite:p:`chapelle2012`. We
  start the simulation with "prestressing" using the MULF method
  :cite:p:`gee2010,schein2021`, which allows to imprint loads
  without changing the geometry, where the solid is loaded to the
  initial left and right ventricular pressures. Thereafter, we kickstart
  the dynamic simulation with passive ventricular filling by the systole
  of the atria (0D chamber models). Ventricular systole happens in
  :math:`t \in [0.2\;\mathrm{s}, 0.53\;\mathrm{s}]`, hence lasting a
  third of the whole cycle time. After systole, the heart relaxes and
  eventually fills to about the same pressure as it has been initialized
  to.
| NOTE: For demonstrative purposes, a fairly coarse finite element
  discretization is chosen here, which by no means yields a spatially
  converged solution and which may be prone to locking phenomena. The
  user may increse the parameter :literal:`\`order_disp'` in the
  ``FEM_PARAMS`` section from ``1`` to ``2`` (and increase
  :literal:`\`quad_degree'` to ``6``) such that quadratic finite element
  ansatz functions (instead of linear ones) are used. While this will
  increase accuracy and mitigate locking, computation time will
  increase.

.. figure:: fig/heart_syspul_setup.png
   :name: fig:heart_syspul_setup
   :width: 65.0%

   Generic 3D ventricular heart model coupled to a closed-loop systemic
   and pulmonary circulation model.

Study the setup shown in fig. `7 <#fig:heart_syspul_setup>`__ and the
comments in the input file ``solid_flow0d_heart_cycle.py``. Run the
simulation, either in one of the provided Docker containers or using
your own FEniCSx/Ambit installation, using the command

::

   mpiexec -n 1 python3 solid_flow0d_heart_cycle.py

It is fully sufficient to use one core (``mpiexec -n 1``) for the
presented setup, while you might want to use more (e.g.,
``mpiexec -n 4``) if you increase :literal:`\`order_disp'` to ``2``.

Open the results file
``results_solid_flow0d_heart_cycle_displacement.xdmf`` in Paraview, and
visualize the deformation over the heart cycle.

For postprocessing of the time courses of pressures, volumes, and fluxes
of the 0D model, either use your own tools to plot the text output files
(first column is time, second is the respective quantity), or make sure
to have Gnuplot (and TeX) installed and navigate to the output folder
(``tmp/``) in order to execute the script ``flow0d_plot.py`` (which lies
in ``ambit/src/ambit_fe/postprocess/``):

::

   flow0d_plot.py -s solid_flow0d_heart_cycle -V0 117e3 93e3 0 0 0

| A folder ``plot_solid_flow0d_heart_cycle`` is created inside ``tmp/``.
  Look at the results of pressures (:math:`p`), volumes (:math:`V`), and
  fluxes (:math:`q`, :math:`Q`) over time. Subscripts ``v``, ``at``,
  ``ar``, ``ven`` refer to ‘ventricular’, ‘atrial’, ‘arterial’, and
  ‘venous’, respectively. Superscripts ``l``, ``r``, ``sys``, ``pul``
  refer to ‘left’, ‘right’, ‘systemic’, and ‘pulmonary’, respectively.
  Try to understand the time courses of the respective pressures, as
  well as the plots of ventricular pressure over volume. Check that the
  overall system volume is constant and around 4-5 liters.
| NOTE: This setup computes only one cardiac cycle, which does not yield
  a periodic state solution (compare e.g. initial and end-cyclic right
  ventricular pressures and volumes, which do not coincide). Change the
  parameter ``number_of_cycles`` from ``1`` to ``10`` and re-run the
  simulation. The simulation will stop when the cycle error (relative
  change in 0D variable quantities from beginning to end of a cycle)
  falls below the value of :literal:`\`eps_periodic'` (set to
  :math:`5 \%`). How many cycles are needed to reach periodicity?
| Figure `8 <#fig:heart_syspul_results>`__ shows a high-fidelity
  solution using a refined mesh and quadratic tetrahedral elements.
  Compare your solution from the coarser mesh. What is the deviation in
  ventricular volume?

.. figure:: fig/heart_syspul_results.png
   :name: fig:heart_syspul_results
   :width: 100.0%

   A. Left heart and systemic pressures over time. B. Left and right
   ventricular and atrial volumes over time. C. Left and right
   ventricular pressure-volume relationships. D. Snapshot of heart
   deformation at end-systole, color indicates displacement magnitude.

Demo: Fluid + 0D flow
---------------------

| – Physics description given in sec. `4.4.2 <#fluid-0d-flow>`__
| – Input files: ``demos/fluid_flow0d``

Blocked pipe flow with 0D model bypass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| This example demonstrates how to couple 3D fluid flow to a 0D
  lumped-parameter model. Incompressible transient Navier-Stokes flow in
  a pipe with prescribed inflow is solved, with the special constraint
  that an internal boundary (all-time closed valve) separates region 1
  and region 2 of the pipe. This internal Dirichlet condition can only
  be achieved by splitting the pressure space, hence having duplicate
  pressure nodes at the valve plane. Otherwise, fluid would experience
  deceleration towards the valve and unphysical acceleration behind it,
  since the pressure gradient drives fluid flow.
| This example demonstrates how the closed valve can be bypassed by a 0D
  flow model that links the 3D fluid out-flow of one region to the
  in-flow of the other region. The 0D model consists of two Windkessel
  models in series, each having compliance, resistance, and inertance
  elements.

.. figure:: fig/pipe_0d_setup.png
   :name: fig:pipe_0d_setup
   :width: 85.0%

   Blocked pipe with 0D model bypass, simulation setup.

Study the setup shown in fig. `9 <#fig:pipe_0d_setup>`__ and the
comments in the input file ``fluid_flow0d_pipe.py``. Run the simulation,
either in one of the provided Docker containers or using your own
FEniCSx/Ambit installation, using the command

::

   mpiexec -n 1 python3 fluid_flow0d_pipe.py

It is fully sufficient to use one core (``mpiexec -n 1``) for the
presented setup.

Open the results file ``results_fluid_flow0d_pipe_velocity.xdmf`` in
Paraview, and visualize the velocity over time.

| Think of which parameter(s) of the 0D model to tweak in order to
  achieve a) little to no fluid in-flow (into
  :math:`\mathit{\Gamma}_{\mathrm{in}}^{\mathrm{f\text{-}0d}}`), b)
  almost the same flow across
  :math:`\mathit{\Gamma}_{\mathrm{out}}^{\mathrm{f\text{-}0d}}` and
  :math:`\mathit{\Gamma}_{\mathrm{in}}^{\mathrm{f\text{-}0d}}`. Think of
  where the flow is going to in case of a).
| Figure shows the velocity streamlines and magnitude at the end of the
  simulation.

.. figure:: fig/pipe_0d_results.png
   :name: fig:pipe_0d_results
   :width: 85.0%

   Streamlines of velocity at end of simulation, color indicates velcity
   magnitude.

Demo: FSI
---------

| – Physics description given in sec.
  `4.4.4 <#fluid-solid-interaction-fsi>`__
| – Input files: ``demos/fsi``

Channel flow around elastic flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Incompressible fluid flow in a 2D channel around an elastic flag is
  studied. The setup corresponds to the well-known Turek benchmark
  :cite:p:`turek2006`. Here, the two cases FSI2 and FSI3 from
  the original setup are investigated. A prescribed inflow velocity with
  parabolic inflow profile is used:
| 

  .. math::
     \begin{aligned}
         \boldsymbol{v}_{\text{f}}= \bar{v}(t,y) \boldsymbol{e}_{x}
         \quad &  
         \text{on}\; \mathit{\Gamma}_{t,\mathrm{in}}^{D,\text{f}} \times [0,T],
         \label{eq:flag_dbc_in}
     \end{aligned}

  with

  .. math::
     \begin{aligned}
         \bar{v}(t,y) = \begin{cases} 1.5 \,\bar{U}\, \frac{y(H-y)}{\left(\frac{H}{2}\right)^2} \frac{1-\cos\left(\frac{\pi}{2}t\right)}{2}, & \text{if} \; t < 2, \\ 1.5 \,\bar{U}\, \frac{y(H-y)}{\left(\frac{H}{2}\right)^2}, & \text{else}, \end{cases}
         \label{eq:flag_dbcs_func}
     \end{aligned}

| with :math:`\bar{U}=10^{3}\;\mathrm{mm}/\mathrm{s}` (FSI2) and
  :math:`\bar{U}=2\cdot 10^{3}\;\mathrm{mm}/\mathrm{s}` (FSI3).

.. figure:: fig/channel_flag_setup.png
   :name: fig:channel_flag_setup
   :width: 100.0%

   Channel flow around an elastic flag :cite:p:`turek2006`,
   problem setup.

Geometrical parameters, given in :math:`[\mathrm{mm}]`, are:

.. container:: center

   +-----------+-----------+-----------+-----------+-----------+-------------+-------------+
   | :math:`L` | :math:`H` | :math:`r` | :math:`l` | :math:`h` | :math:`d_x` | :math:`d_y` |
   +===========+===========+===========+===========+===========+=============+=============+
   | 2500      | 410       | 50        | 350       | 20        | 200         | 200         |
   +-----------+-----------+-----------+-----------+-----------+-------------+-------------+

| Both solid and fluid are discretized with quadrilateral
  :math:`\mathbb{Q}^2`-:math:`\mathbb{Q}^1` Taylor-Hood finite elements,
  hence no stabilization for the fluid problem is needed. Temporal
  discretization for both the solid and the fluid are carried out with a
  Generalized-:math:`\alpha` scheme with no numerical damping
  (:math:`\rho_{\mathrm{inf}}=1`).
| Study the setup shown in fig. `11 <#fig:channel_flag_setup>`__
  together with the parameters in the table and the comments in the
  input file ``fsi_channel_flag.py``. Run the simulation for FSI2 and
  FSI3 cases, either in one of the provided Docker containers or using
  your own FEniCSx/Ambit installation, using the command

::

   mpiexec -n 1 python3 fsi_channel_flag.py

| If your system allows, use more than one core (e.g. ``mpiexec -n 4``)
  in order to speed up the simulation a bit.
| The physics of the problem are strongly time-dependent, and a
  (near-)periodic oscillation of the flag only occurs after
  :math:`t\approx 10\;\mathrm{s}` (FSI2) and
  :math:`t\approx 5\;\mathrm{s}` (FSI3). Run the problem to the end
  (:math:`t = 15\;\mathrm{s}` for FSI2, :math:`t = 7.5\;\mathrm{s}` for
  FSI3), be patient, and monitor the flag tip displacement over time.
| Figure `12 <#fig:channel_flag_results>`__ depicts the velocity at
  three instances in time towards the end of the simulation for the FSI2
  case, and figure `13 <#fig:channel_flag_results_verif>`__ shows the
  flag’s tip displacement over time compared to the reference solution,
  over a time interval where the solution has become periodic. (Note
  that the reference solution from the official link shown in the input
  file needs to be time-adjusted, i.e. synchronized with the first peak
  in the interval of interest, since the time column of the reference
  data does not correspond to the physical time of the problem setup.)

.. figure:: fig/channel_flag_results.png
   :name: fig:channel_flag_results
   :width: 85.0%

   FSI2 case: Magnitude of fluid velocity at three instances in time
   (:math:`t=10.5\;\mathrm{s}`, :math:`t=11.2\;\mathrm{s}`, and
   :math:`t=12\;\mathrm{s}`) towards end of simulation, color indicates
   velcity magnitude.

.. figure:: fig/channel_flag_results_verif.png
   :name: fig:channel_flag_results_verif
   :width: 100.0%

   Comparison to benchmark reference solution for the time course of the
   flag’s tip displacement for the two setups FSI2 and FSI3. A fairly
   coarse time step of :math:`\Delta t = 4 \;\mathrm{ms}` (FSI2) and
   :math:`\Delta t = 2 \;\mathrm{ms}` (FSI3) already allows a close
   match to the original results.

Table of symbols
================

.. math::
   \nonumber
   \begin{aligned}
   &\mathit{\Omega}_0,\mathit{\Omega}&&: \text{reference, current domain} \\
   &\mathit{\Gamma}_0,\mathit{\Gamma}&&: \text{reference, current boundary} \\
   &\boldsymbol{x}_0, \boldsymbol{x} &&: \text{coordinates of the reference, current frame} \\
   &\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z &&: \text{unit vectors of the cartesian reference frame} \\
   &\boldsymbol{n}_0, \boldsymbol{n} &&: \text{unit outward normal defined in the reference, current frame} \\
   &\nabla_{0},\nabla &&: \text{Nabla operator with respect to the reference, current frame} \\
   &\nabla\boldsymbol{v} := \frac{\partial v_i}{\partial x_j} \boldsymbol{e}_{i} \otimes\boldsymbol{e}_{j} &&: \text{Gradient of a vector field} \\
   &\nabla\cdot\boldsymbol{v} := \frac{\partial v_i}{\partial x_j} \boldsymbol{e}_{i} \cdot\boldsymbol{e}_{j} &&: \text{Divergence of a vector field} \\
   &\boldsymbol{e}_{i} &&: \text{Basis vectors}, i\in \{1,2,3\} \\
   &t, T &&: \text{current, end time of an initial boundary value problem} \\
   &\boldsymbol{u}, \hat{\boldsymbol{u}}_{0} &&: \text{solid mechanics displacement field, and prescribed initial value} \\
   &\delta\boldsymbol{u}, \Delta\boldsymbol{u} &&: \text{solid mechanics displacement test, trial function} \\
   & p &&: \text{solid mechanics hydrostatic pressure, or fluid mechanics pressure} \\
   & \delta p, \Delta p &&: \text{solid or fluid mechanics pressure test, trial function} \\
   &\boldsymbol{v}=\frac{\partial\boldsymbol{u}}{\partial t}, \hat{\boldsymbol{v}}_{0} &&: \text{solid mechanics velocity, and prescribed initial value} \\
   &\boldsymbol{a}=\frac{\partial^2\boldsymbol{u}}{\partial t^2} &&: \text{solid mechanics acceleration} \\
   &\boldsymbol{v}, \hat{\boldsymbol{v}}_{0} &&: \text{fluid mechanics velocity, and prescribed initial value} \\
   &\delta\boldsymbol{v}, \Delta\boldsymbol{v} &&: \text{fluid mechanics velocity test, trial function} \\
   &\boldsymbol{a}=\frac{\partial\boldsymbol{v}}{\partial t} &&: \text{fluid mechanics acceleration} \\
   &\boldsymbol{d} &&: \text{ALE domain displacement} \\
   &\delta\boldsymbol{d}, \Delta\boldsymbol{d} &&: \text{ALE domain displacement test, trial function} \\
   &\hat{\boldsymbol{b}}_0, \hat{\boldsymbol{b}} &&: \text{body force vector defined in the reference, current frame} \\
   &\widehat{\boldsymbol{w}}=\frac{\partial\boldsymbol{d}}{\partial t} &&: \text{ALE domain velocity} \\
   &\rho_0, \rho &&: \text{reference, current density} \\
   &\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S} &&: \text{1st Piola Kirchhoff stress tensor} \\
   &\boldsymbol{F}=\boldsymbol{I}+\nabla_{0}\boldsymbol{u} &&: \text{solid deformation gradient} \\
   &\widehat{\boldsymbol{F}}=\boldsymbol{I}+\nabla_{0}\boldsymbol{d} &&: \text{ALE deformation gradient} \\
   &J=\det \boldsymbol{F} &&: \text{determinant of solid deformation gradient} \\
   &\widehat{J}=\det \widehat{\boldsymbol{F}} &&: \text{determinant of ALE deformation gradient} \\
   &\boldsymbol{S} &&: \text{2nd Piola-Kirchhoff stress tensor} \\
   &\boldsymbol{\sigma} &&: \text{Cauchy stress tensor} \\
   &\boldsymbol{t}_0, \hat{\boldsymbol{t}}_{0} &&: \text{1st Piola-Kirchhoff traction, prescribed 1st Piola-Kirchhoff traction} \\
   &\boldsymbol{t}, \hat{\boldsymbol{t}} &&: \text{Cauchy traction, prescribed Cauchy traction} \\
   \end{aligned}
 
.. bibliography::
