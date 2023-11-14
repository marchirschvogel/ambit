====================================================
Ambit – A FEniCS-based cardiovascular physics solver
====================================================

:Author: Dr.-Ing. Marc Hirschvogel

Preface
=======

| Ambit is an open-source software tool written in Python for parallel
  multi-physics simulations focusing on – but not limited to – cardiac
  mechanics. Amongst others, it contains re-implementations and
  generalizations of methods developed by the author for his PhD thesis
  (Hirschvogel 2018). Ambit makes use of the open-source finite element
  library FEniCS/dolfinx (https://fenicsproject.org) (Logg, Mardal, and
  Wells 2012) along with the linear algebra package PETSc
  (https://petsc.org) (Balay et al. 2022). It is constantly updated to
  ensure compatibility with a recent dolfinx development version, hence
  guaranteeing a state-of-the-art finite element and linear algebra
  backend.
| Ambit is designed such that the user only needs to provide input files
  that define parameters through Python dictionaries, hence no
  programming or in-depth knowledge of any library-specific syntax is
  required.
| Ambit provides general nonlinear (compressible or incompressible)
  finite strain solid dynamics (Holzapfel 2000), implementing a range of
  hyperelastic, viscous, and active material models. Specifically, the
  well-known anisotropic Holzapfel-Ogden (Holzapfel and Ogden 2009) and
  Guccione models (Guccione, Costa, and McCulloch 1995) for structural
  description of the myocardium are provided, along with a bunch of
  other models. It further implements strain- and stress-mediated
  volumetric growth models (Göktepe et al. 2010) that allow to model
  (maladaptive) ventricular shape and size changes. Inverse mechanics
  approaches to imprint loads into a reference state are implemented
  using the so-called prestressing method (Gee, Förster, and Wall 2010)
  in displacement formulation (Schein and Gee 2021).
| Furthermore, fluid dynamics in terms of incompressible
  Navier-Stokes/Stokes equations – either in Eulerian or Arbitrary
  Lagrangian-Eulerian (ALE) reference frames – are implemented.
  Taylor-Hood elements or equal-order approximations with SUPG/PSPG
  stabilization (Tezduyar and Osawa 2000) can be used.
| A variety of reduced 0D lumped models targeted at blood circulation
  modeling are implemented, including 3- and 4-element Windkessel models
  (Westerhof, Lankhaar, and Westerhof 2009) as well as closed-loop full
  circulation (Hirschvogel et al. 2017) and coronary flow models
  (Arthurs et al. 2016).
| Monolithic multi-physics coupling of solid, fluid, and ALE-fluid with
  0D lumped models is implemented such that cardiovascular simulations
  with realistic boundary conditions can be performed. Monolithic
  fluid-solid interaction (FSI) will be provided in the near future once
  mixed domain functionality is fully supported in the finite element
  backend dolfinx.
| Implementations for a recently proposed novel physics- and
  projection-based model reduction for FSI, denoted as
  fluid-reduced-solid interaction (FrSI) (Hirschvogel et al. 2022), are
  provided, along with POD-based Galerkin model reduction techniques
  (Farhat et al. 2014) using full or boundary subspaces.
| The nonlinear (single- or multi-field) problems are solved with a
  customized Newton solver with PTC (Gee, Kelley, and Lehoucq 2009)
  adaptibity in case of divergence, providing robustness for numerically
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

Solid mechanics
===============

| – Example: ``demos/solid``
| – Problem type: ``solid``
| – Solid mechanics are formulated in a Total Lagrangian frame

Strong form
-----------

Displacement-based
~~~~~~~~~~~~~~~~~~

– Primary variable: displacement :math:`\boldsymbol{u}`

.. math::

   \begin{aligned}
   {3}
   \boldsymbol{\nabla}_{0} \cdot \boldsymbol{P}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) + \hat{\boldsymbol{b}}_{0} &= \rho_{0} \boldsymbol{a}(\boldsymbol{u}) \quad &&\text{in} \; \mathit{\Omega}_{0} \times [0, T], \label{eq:divP} \\
   \boldsymbol{u} &= \hat{\boldsymbol{u}} \quad &&\text{on} \; \mathit{\Gamma}_{0}^{\mathrm{D}} \times [0, T], \label{eq:bc_u}\\
   \boldsymbol{t}_{0} = \boldsymbol{P}\boldsymbol{n}_{0} &= \hat{\boldsymbol{t}}_{0} \quad &&\text{on} \; \mathit{\Gamma}_{0}^{\mathrm{N}} \times [0, T], \label{eq:bc_N}\\
   \boldsymbol{u}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{u}}_{0}(\boldsymbol{x}_{0}) \quad &&\text{in} \; \mathit{\Omega}_{0}, \label{eq:ini_u}\\
   \boldsymbol{v}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}_{0}) \quad &&\text{in} \; \mathit{\Omega}_{0}, \label{eq:ini_v}\end{aligned}

Incompressible mechanics
~~~~~~~~~~~~~~~~~~~~~~~~

– Primary variables: displacement :math:`\boldsymbol{u}` and pressure
:math:`p`

.. math::

   \begin{aligned}
   {3}
   \boldsymbol{\nabla}_{0} \cdot \boldsymbol{P}(\boldsymbol{u},p,\boldsymbol{v}(\boldsymbol{u})) + \hat{\boldsymbol{b}}_{0} &= \rho_{0} \boldsymbol{a}(\boldsymbol{u}) \quad &&\text{in} \; \mathit{\Omega}_{0} \times [0, T], \label{eq:divP} \\
   J(\boldsymbol{u})-1 &= 0 \quad &&\text{in} \; \mathit{\Omega}_{0} \times [0, T], \label{eq:J} \\
   \boldsymbol{u} &= \hat{\boldsymbol{u}} \quad &&\text{on} \; \mathit{\Gamma}_{0}^{\mathrm{D}} \times [0, T], \label{eq:bc_u}\\
   \boldsymbol{t}_{0} = \boldsymbol{P}\boldsymbol{n}_{0} &= \hat{\boldsymbol{t}}_{0} \quad &&\text{on} \; \mathit{\mathit{\Gamma}}_{0}^{\mathrm{N}} \times [0, T], \label{eq:bc_N}\\
   \boldsymbol{u}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{u}}_{0}(\boldsymbol{x}_{0}) \quad &&\text{in} \; \mathit{\mathit{\Omega}}_{0}, \label{eq:ini_u}\\
   \boldsymbol{v}(\boldsymbol{x}_{0},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}_{0}) \quad &&\text{in} \; \mathit{\mathit{\Omega}}_{0}, \label{eq:ini_v}\end{aligned}

with velocity and acceleration
:math:`\boldsymbol{v}=\frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}t}` and
:math:`\boldsymbol{a}=\frac{\mathrm{d}^2\boldsymbol{u}}{\mathrm{d}t^2}`,
respectively

Weak form
---------

.. _displacement-based-1:

Displacement-based
~~~~~~~~~~~~~~~~~~

| – Primary variable: displacement :math:`\boldsymbol{u}`
| – Principal of Virtual Work:

  .. math::

     \begin{aligned}
     r(\boldsymbol{u};\delta\boldsymbol{u}) := \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{u};\delta\boldsymbol{u}) + \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u};\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u}\label{eq:res_u_solid}\end{aligned}

  – Kinetic virtual work:

  .. math::

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{u};\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \rho_{0}\,\boldsymbol{a}(\boldsymbol{u}) \cdot \delta\boldsymbol{u} \,\mathrm{d}V \label{eq:deltaWkin}\end{aligned}

  – Internal virtual work:

  .. math::

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u};\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \boldsymbol{P}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) : \boldsymbol{\nabla}_{0} \delta\boldsymbol{u} \,\mathrm{d}V \label{eq:deltaWint}\end{aligned}

  – External virtual work:

-  conservative Neumann load:

   .. math::

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Gamma}_{0}^{\mathrm{N}}} \hat{\boldsymbol{t}}_{0}(t) \cdot \delta\boldsymbol{u} \,\mathrm{d}A \label{eq:deltaWext_pk1}\end{aligned}

-  Neumann pressure load in current normal direction:

   .. math::

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= -\int\limits_{\mathit{\Gamma}_{0}^{\mathrm{N}}} \hat{p}(t)\,J \boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0} \cdot \delta\boldsymbol{u} \,\mathrm{d}A \label{eq:deltaWext_cur_p}\end{aligned}

-  general Neumann load in current direction:

   .. math::

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Gamma}_0} J\boldsymbol{F}^{-\mathrm{T}}\,\hat{\boldsymbol{t}}_{0}(t) \cdot \delta\boldsymbol{u} \,\mathrm{d}A \label{eq:deltaWext_cur}\end{aligned}

-  body force:

   .. math::

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \hat{\boldsymbol{b}}_{0}(t) \cdot \delta\boldsymbol{u} \,\mathrm{d}V \label{eq:deltaWext_body}\end{aligned}

-  generalized Robin condition:

   .. math::

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= -\int\limits_{\mathit{\Gamma}_{0}^{\mathrm{N}}} \left[k\,\boldsymbol{u} + c\,\boldsymbol{v}(\boldsymbol{u})\right] \cdot \delta\boldsymbol{u}\,\mathrm{d}A \label{eq:deltaWext_rob}\end{aligned}

-  generalized Robin condition in reference surface normal direction:

   .. math::

      \begin{aligned}
      \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) &= -\int\limits_{\mathit{\Gamma}_{0}^{\mathrm{N}}} (\boldsymbol{n}_0 \otimes \boldsymbol{n}_0)\left[k\,\boldsymbol{u} + c\,\boldsymbol{v}(\boldsymbol{u})\right] \cdot \delta\boldsymbol{u}\,\mathrm{d}A \label{eq:deltaWext_robn}\end{aligned}

– Discrete linear system

.. math::

   \begin{aligned}
   \left. \boldsymbol{\mathsf{K}}_{uu} \right|_{n+1}^{k} \Delta\boldsymbol{\mathsf{u}}_{n+1}^{k+1}=-\left. \boldsymbol{\mathsf{r}}_{u} \right|_{n+1}^{k} \label{eq:lin_sys_solid}\end{aligned}

Incompressible mechanics: 2-field displacement and pressure variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

– Primary variables: displacement :math:`\boldsymbol{u}` and pressure
:math:`p`

.. math::

   \begin{aligned}
   r_u(\boldsymbol{u},p;\delta\boldsymbol{u}) &:= \delta \mathcal{W}_{\mathrm{kin}}(\boldsymbol{u};\delta\boldsymbol{u}) + \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u},p;\delta\boldsymbol{u}) - \delta \mathcal{W}_{\mathrm{ext}}(\boldsymbol{u};\delta\boldsymbol{u}) = 0, \quad \forall \; \delta\boldsymbol{u} \label{eq:res_u_solid_incomp}\\
   r_p(\boldsymbol{u};\delta p) &:= \delta \mathcal{W}_{\mathrm{pres}}(\boldsymbol{u};\delta p) = 0, \quad \forall \; \delta p\end{aligned}

| – Kinetic virtual work: (`[eq:deltaWkin] <#eq:deltaWkin>`__)
| – Internal virtual work:

  .. math::

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{int}}(\boldsymbol{u},p;\delta\boldsymbol{u}) &= \int\limits_{\mathit{\Omega}_{0}} \boldsymbol{P}(\boldsymbol{u},p,\boldsymbol{v}(\boldsymbol{u})) : \boldsymbol{\nabla}_{0} \delta\boldsymbol{u} \,\mathrm{d}V \label{eq:deltaWint_incomp}\end{aligned}

  – Pressure virtual work:

  .. math::

     \begin{aligned}
     \delta \mathcal{W}_{\mathrm{pres}}(\boldsymbol{u};\delta p) &= \int\limits_{\mathit{\Omega}_{0}} (J(\boldsymbol{u}) - 1) \,\delta p \,\mathrm{d}V \label{eq:deltaWpres}\end{aligned}

– Discrete linear system

.. math::

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{up} \\ \\ \boldsymbol{\mathsf{K}}_{pu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u} \\ \\ \boldsymbol{\mathsf{r}}_{p} \end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_solid_incomp}\end{aligned}

Fluid mechanics
===============

Eulerian reference frame
------------------------

| – Example: ``demos/fluid``
| – Problem type: ``fluid``
| – Incompressible Navier-Stokes equations in Eulerian reference frame

.. _strong-form-1:

Strong Form
~~~~~~~~~~~

– Primary variables: velocity :math:`\boldsymbol{v}` and pressure
:math:`p`

.. math::

   \begin{aligned}
   {3}
   \boldsymbol{\nabla} \cdot \boldsymbol{\sigma}(\boldsymbol{v},p) + \hat{\boldsymbol{b}} &= \rho\left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})\,\boldsymbol{v}\right) \quad &&\text{in} \; \mathit{\mathit{\Omega}}_t \times [0, T], \label{eq:divsigma_ns} \\
   \boldsymbol{\nabla}\cdot \boldsymbol{v} &= 0 \quad &&\text{in} \; \mathit{\mathit{\Omega}}_t \times [0, T],\label{eq:divv_ns}\\
   \boldsymbol{v} &= \hat{\boldsymbol{v}} \quad &&\text{on} \; \mathit{\mathit{\Gamma}}_t^{\mathrm{D}} \times [0, T], \label{eq:bc_v_ns}\\
   \boldsymbol{t} = \boldsymbol{\sigma}\boldsymbol{n} &= \hat{\boldsymbol{t}} \quad &&\text{on} \; \mathit{\mathit{\Gamma}}_t^{\mathrm{N}} \times [0, T], \label{eq:bc_N_ns}\\
   \boldsymbol{v}(\boldsymbol{x},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}) \quad &&\text{in} \; \mathit{\mathit{\Omega}}_t, \label{eq:ini_v_ns}\end{aligned}

with a Newtonian fluid constitutive law

.. math::

   \begin{aligned}
   \boldsymbol{\sigma} = -p \boldsymbol{I} + 2 \mu\,\boldsymbol{\gamma} = -p \boldsymbol{I} + \mu \left(\boldsymbol{\nabla} \boldsymbol{v} + (\boldsymbol{\nabla} \boldsymbol{v})^{\mathrm{T}}\right)\end{aligned}

.. _weak-form-1:

Weak Form
~~~~~~~~~

| – Primary variables: velocity :math:`\boldsymbol{v}` and pressure
  :math:`p`
| – Principle of Virtual Power

  .. math::

     \begin{aligned}
     r_v(\boldsymbol{v},p;\delta\boldsymbol{v}) &:= \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v};\delta\boldsymbol{v}) + \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p;\delta\boldsymbol{v}) - \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{v};\delta\boldsymbol{v}) = 0, \quad \forall \; \delta\boldsymbol{v} \label{eq:res_v_fluid}\\
     r_p(\boldsymbol{v};\delta p) &:= \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v};\delta p), \quad \forall \; \delta p \label{eq:res_p_fluid}\end{aligned}

– Kinetic virtual power:

.. math::

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v};\delta\boldsymbol{v}) = \int\limits_{\mathit{\Omega}_t} \rho\left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})\,\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v\end{aligned}

– Internal virtual power:

.. math::

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p;\delta\boldsymbol{v}) = 
   \int\limits_{\mathit{\Omega}_t} \boldsymbol{\sigma}(\boldsymbol{v},p) : \boldsymbol{\nabla} \delta\boldsymbol{v} \,\mathrm{d}v \end{aligned}

– Pressure virtual power:

.. math::

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v};\delta p) = 
   \int\limits_{\mathit{\Omega}_t} (\boldsymbol{\nabla}\cdot\boldsymbol{v})\,\delta p\,\mathrm{d}v\end{aligned}

| – External virtual power:

-  conservative Neumann load:

   .. math::

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Gamma}_t^{\mathrm{N}}} \hat{\boldsymbol{t}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}a \label{eq:deltaPext_neumann}\end{aligned}

-  pressure Neumann load:

   .. math::

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= -\int\limits_{\mathit{\Gamma}_t^{\mathrm{N}}} \hat{p}(t)\,\boldsymbol{n} \cdot \delta\boldsymbol{v} \,\mathrm{d}a \label{eq:deltaPext_neumann}\end{aligned}

-  body force:

   .. math::

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Omega}_t} \hat{\boldsymbol{b}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}V \label{eq:deltaPext_body}\end{aligned}

.. _subsubsec:stab:

Stabilization
~~~~~~~~~~~~~

| Streamline-upwind Petrov-Galerkin/pressure-stabilizing Petrov-Galerkin
  (SUPG/PSPG) methods are implemented, either using the full or a
  reduced scheme
| Full scheme according to (Tezduyar and Osawa 2000): ``supg_pspg``:
| – Velocity residual operator (`[eq:res_v_fluid] <#eq:res_v_fluid>`__)
  is augmented with the following terms:

  .. math::

     \begin{aligned}
     r_v \leftarrow r_v &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_t} \tau_{\mathrm{SUPG}}\,(\boldsymbol{\nabla}\delta\boldsymbol{v})\,\boldsymbol{v} \cdot \left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})\,\boldsymbol{v}\right) - \boldsymbol{\nabla} \cdot \boldsymbol{\sigma}(\boldsymbol{v},p)\right]\,\mathrm{d}v \\
     & + \int\limits_{\mathit{\Omega}_t} \tau_{\mathrm{LSIC}}\,\rho\,(\boldsymbol{\nabla}\cdot\delta\boldsymbol{v})(\boldsymbol{\nabla}\cdot\boldsymbol{v})\,\mathrm{d}v\end{aligned}

  – Pressure residual operator (`[eq:res_p_fluid] <#eq:res_p_fluid>`__)
  is augmented with the following terms:

  .. math::

     \begin{aligned}
     r_p \leftarrow r_p &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_t} \tau_{\mathrm{PSPG}}\,(\boldsymbol{\nabla}\delta p) \cdot \left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})\,\boldsymbol{v}\right) - \boldsymbol{\nabla} \cdot \boldsymbol{\sigma}(\boldsymbol{v},p)\right]\,\mathrm{d}v \end{aligned}

Reduced scheme (optimized for first-order): ``supg_pspg2``:

– Velocity residual operator (`[eq:res_v_fluid] <#eq:res_v_fluid>`__) is
augmented with the following terms:

.. math::

   \begin{aligned}
   r_v \leftarrow r_v &+ \int\limits_{\mathit{\Omega}_t} d_1\,((\boldsymbol{\nabla}\boldsymbol{v})\,\boldsymbol{v}) \cdot (\boldsymbol{\nabla}\delta\boldsymbol{v})\,\boldsymbol{v}\,\mathrm{d}v \\
   & + \int\limits_{\mathit{\Omega}_t} d_2\,(\boldsymbol{\nabla}\cdot\boldsymbol{v}) (\boldsymbol{\nabla}\cdot\delta\boldsymbol{v})\,\mathrm{d}v\\
   &+ \int\limits_{\mathit{\Omega}_t} d_3\,(\boldsymbol{\nabla}p) \cdot (\boldsymbol{\nabla}\delta\boldsymbol{v})\,\boldsymbol{v}\,\mathrm{d}v \end{aligned}

– Pressure residual operator (`[eq:res_p_fluid] <#eq:res_p_fluid>`__) is
augmented with the following terms:

.. math::

   \begin{aligned}
   r_p \leftarrow r_p &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_t} d_1\,((\boldsymbol{\nabla}\boldsymbol{v})\,\boldsymbol{v}) \cdot (\boldsymbol{\nabla}\delta p)\,\mathrm{d}v \\
   &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_t} d_3\,(\boldsymbol{\nabla}p) \cdot (\boldsymbol{\nabla}\delta p)\,\mathrm{d}v \end{aligned}

– Discrete linear system

.. math::

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_fluid}\end{aligned}

– Note that :math:`\boldsymbol{\mathsf{K}}_{pp}` is zero for Taylor-Hood
elements (without stabilization)

ALE reference frame
-------------------

| – Problem type: ``fluid_ale``
| – Incompressible Navier-Stokes equations in Arbitrary Lagrangian
  Eulerian (ALE) reference frame
| – ALE domain problem deformation governed by linear-elastic or
  nonlinear hyperelastic solid, displacement field
  :math:`\boldsymbol{d}`
| – Fluid mechanics formulated with respect to the reference frame,
  using ALE deformation gradient
  :math:`\boldsymbol{F}(\boldsymbol{d}) = \boldsymbol{I} + \boldsymbol{\nabla}_0\boldsymbol{d}`
  and its determinant,
  :math:`J(\boldsymbol{d})=\det \boldsymbol{F}(\boldsymbol{d})`

ALE problem
~~~~~~~~~~~

| – Displacement-based quasi-static solid
| – Primary variable: domain displacement :math:`\boldsymbol{d}`
| – Strong form:

  .. math::

     \begin{aligned}
     {3}
     \boldsymbol{\nabla}_{0} \cdot \boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) &= \boldsymbol{0} \quad &&\text{in} \; \mathit{\mathit{\Omega}}_0, \label{eq:divsigma_ale} \\
     \boldsymbol{d} &= \hat{\boldsymbol{d}} \quad &&\text{on} \; \mathit{\mathit{\Gamma}}_0^{\mathrm{D}}, \label{eq:dbc_ale}\end{aligned}

  with

  .. math::

     \begin{aligned}
     \boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) = E \,\frac{1}{2}\left(\boldsymbol{\nabla}_0\boldsymbol{d} + (\boldsymbol{\nabla}_0\boldsymbol{d})^{\mathrm{T}}\right) + \kappa \left(\boldsymbol{\nabla}_0 \cdot \boldsymbol{d}\right)\,\boldsymbol{I}\end{aligned}

– weak form:

.. math::

   \begin{aligned}
   r_{d}(\boldsymbol{d};\delta\boldsymbol{d}) := \int\limits_{\mathit{\Omega}_0}\boldsymbol{\sigma}^{\mathrm{G}}(\boldsymbol{d}) : \boldsymbol{\nabla}_{0}\delta\boldsymbol{d}\,\mathrm{d}V = 0, \quad \forall \; \delta\boldsymbol{d} \label{eq:r_d}\end{aligned}

.. _strong-form-2:

Strong form
~~~~~~~~~~~

– Primary variables: velocity :math:`\boldsymbol{v}` and pressure
:math:`p`

.. math::

   \begin{aligned}
   {3}
   \boldsymbol{\nabla}_{0} \boldsymbol{\sigma}(\boldsymbol{v},\boldsymbol{d},p) : \boldsymbol{F}^{-\mathrm{T}} + \hat{\boldsymbol{b}} &= \rho\left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_0\boldsymbol{v}\,\boldsymbol{F}^{-1})\,\boldsymbol{v}\right) \quad &&\text{in} \; \mathit{\mathit{\Omega}}_0 \times [0, T], \label{eq:divsigma_ns} \\
   \boldsymbol{\nabla}_{0}\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}} &= 0 \quad &&\text{in} \; \mathit{\mathit{\Omega}}_0 \times [0, T],\label{eq:divv_ns}\\
   \boldsymbol{v} &= \hat{\boldsymbol{v}} \quad &&\text{on} \; \mathit{\mathit{\Gamma}}_0^{\mathrm{D}} \times [0, T], \label{eq:bc_v_ns}\\
   \boldsymbol{t} = \boldsymbol{\sigma}\boldsymbol{n} &= \hat{\boldsymbol{t}} \quad &&\text{on} \; \mathit{\mathit{\Gamma}}_0^{\mathrm{N}} \times [0, T], \label{eq:bc_N_ns}\\
   \boldsymbol{v}(\boldsymbol{x},0) &= \hat{\boldsymbol{v}}_{0}(\boldsymbol{x}) \quad &&\text{in} \; \mathit{\mathit{\Omega}}_0, \label{eq:ini_v_ns}\end{aligned}

with a Newtonian fluid constitutive law

.. math::

   \begin{aligned}
   \boldsymbol{\sigma} = -p \boldsymbol{I} + 2 \mu \boldsymbol{\gamma} = -p \boldsymbol{I} + \mu \left(\boldsymbol{\nabla}_0 \boldsymbol{v}\,\boldsymbol{F}^{-1} + \boldsymbol{F}^{-\mathrm{T}}(\boldsymbol{\nabla}_0 \boldsymbol{v})^{\mathrm{T}}\right)\end{aligned}

.. _weak-form-2:

Weak form
~~~~~~~~~

| – Primary variables: velocity :math:`\boldsymbol{v}`, pressure
  :math:`p`, and domain displacement :math:`\boldsymbol{d}`
| – Principle of Virtual Power

  .. math::

     \begin{aligned}
     r_v(\boldsymbol{v},p,\boldsymbol{d};\delta\boldsymbol{v}) &:= \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v},\boldsymbol{d};\delta\boldsymbol{v}) + \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p,\boldsymbol{d};\delta\boldsymbol{v}) - \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{v},\boldsymbol{d};\delta\boldsymbol{v}) = 0, \quad \forall \; \delta\boldsymbol{v} \label{eq:res_v_fluid_ale}\\
     r_p(\boldsymbol{v},\boldsymbol{d};\delta p) &:= \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v},\boldsymbol{d};\delta p), \quad \forall \; \delta p \label{eq:res_p_fluid_ale}\end{aligned}

– Kinetic virtual power:

.. math::

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{kin}}(\boldsymbol{v},\boldsymbol{d};\delta\boldsymbol{v}) = \int\limits_{\mathit{\Omega}_0} J \rho\left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_{0}\boldsymbol{v}\,\boldsymbol{F}^{-1})\,\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V\end{aligned}

– Internal virtual power:

.. math::

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{int}}(\boldsymbol{v},p,\boldsymbol{d};\delta\boldsymbol{v}) = 
   \int\limits_{\mathit{\Omega}_0} J\boldsymbol{\sigma}(\boldsymbol{v},p,\boldsymbol{d}) : \boldsymbol{\nabla}_{0} \delta\boldsymbol{v}\,\boldsymbol{F}^{-1} \,\mathrm{d}V\end{aligned}

– Pressure virtual power:

.. math::

   \begin{aligned}
   \delta \mathcal{P}_{\mathrm{pres}}(\boldsymbol{v},\boldsymbol{d};\delta p) = 
   \int\limits_{\mathit{\Omega}_0} J\,\boldsymbol{\nabla}_{0}\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}}\delta p\,\mathrm{d}V\end{aligned}

| – External virtual power:

-  conservative Neumann load:

   .. math::

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Gamma}_0^{\mathrm{N}}} \hat{\boldsymbol{t}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}A \label{eq:deltaPext_neumann_ale}\end{aligned}

-  pressure Neumann load:

   .. math::

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{d};\delta\boldsymbol{v}) &= -\int\limits_{\mathit{\Gamma}_0^{\mathrm{N}}} \hat{p}(t)\,J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0} \cdot \delta\boldsymbol{v} \,\mathrm{d}A \label{eq:deltaPext_neumann_ale}\end{aligned}

-  body force:

   .. math::

      \begin{aligned}
      \delta \mathcal{P}_{\mathrm{ext}}(\boldsymbol{d};\delta\boldsymbol{v}) &= \int\limits_{\mathit{\Omega}_0} J\,\hat{\boldsymbol{b}}(t) \cdot \delta\boldsymbol{v} \,\mathrm{d}V \label{eq:deltaPext_body_ale}\end{aligned}

Stabilization
~~~~~~~~~~~~~

| ALE forms of stabilization introduced in sec.
  `3.1.3 <#subsubsec:stab>`__
| ``supg_pspg``:
| – Velocity residual operator
  (`[eq:res_v_fluid_ale] <#eq:res_v_fluid_ale>`__) is augmented with the
  following terms:

  .. math::

     \begin{aligned}
     r_v \leftarrow r_v &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_0}J\, \tau_{\mathrm{SUPG}}\,(\boldsymbol{\nabla}_0\delta\boldsymbol{v}\,\boldsymbol{F}^{-1})\,\boldsymbol{v}\;\cdot \\
     & \qquad\quad \cdot\left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_0\boldsymbol{v}\,\boldsymbol{F}^{-1})\,(\boldsymbol{v}-\boldsymbol{w})\right) - \boldsymbol{\nabla}_{0} \boldsymbol{\sigma}(\boldsymbol{v},\boldsymbol{d},p) : \boldsymbol{F}^{-\mathrm{T}}\right]\,\mathrm{d}V \\
     & + \int\limits_{\mathit{\Omega}_0}J\, \tau_{\mathrm{LSIC}}\,\rho\,(\boldsymbol{\nabla}_{0}\delta\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}})(\boldsymbol{\nabla}_{0}\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}})\,\mathrm{d}V\end{aligned}

  – Pressure residual operator
  (`[eq:res_p_fluid_ale] <#eq:res_p_fluid_ale>`__) is augmented with the
  following terms:

  .. math::

     \begin{aligned}
     r_p \leftarrow r_p &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_0}J\, \tau_{\mathrm{PSPG}}\,(\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{\nabla}_{0}\delta p) \;\cdot \\
     & \qquad\quad \cdot \left[\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_0\boldsymbol{v}\,\boldsymbol{F}^{-1})\,(\boldsymbol{v}-\boldsymbol{w})\right) - \boldsymbol{\nabla}_{0} \boldsymbol{\sigma}(\boldsymbol{v},\boldsymbol{d},p) : \boldsymbol{F}^{-\mathrm{T}}\right]\,\mathrm{d}V\end{aligned}

| ``supg_pspg2``:
| – Velocity residual operator
  (`[eq:res_v_fluid_ale] <#eq:res_v_fluid_ale>`__) is augmented with the
  following terms:

  .. math::

     \begin{aligned}
     r_v \leftarrow r_v &+ \int\limits_{\mathit{\Omega}_0} J\,d_1\,((\boldsymbol{\nabla}_{0}\boldsymbol{v}\,\boldsymbol{F}^{-1})\,(\boldsymbol{v}-\boldsymbol{w})) \cdot (\boldsymbol{\nabla}_{0}\delta\boldsymbol{v}\,\boldsymbol{F}^{-1})\,\boldsymbol{v}\,\mathrm{d}V \\
     & + \int\limits_{\mathit{\Omega}_0} J\,d_2\,(\boldsymbol{\nabla}_{0}\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}}) (\boldsymbol{\nabla}_{0}\delta\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}})\,\mathrm{d}V\\
     &+ \int\limits_{\mathit{\Omega}_0} J\,d_3\,(\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{\nabla}_{0}p) \cdot (\boldsymbol{\nabla}_{0}\delta\boldsymbol{v}\,\boldsymbol{F}^{-1})\,\boldsymbol{v}\,\mathrm{d}V\end{aligned}

  – Pressure residual operator
  (`[eq:res_p_fluid_ale] <#eq:res_p_fluid_ale>`__) is augmented with the
  following terms:

  .. math::

     \begin{aligned}
     r_p \leftarrow r_p &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_0} J\,d_1\,((\boldsymbol{\nabla}_{0}\boldsymbol{v}\,\boldsymbol{F}^{-1})\,(\boldsymbol{v}-\boldsymbol{w})) \cdot (\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{\nabla}_{0}\delta p)\,\mathrm{d}V \\
     &+ \frac{1}{\rho}\int\limits_{\mathit{\Omega}_0} J\,d_3\,(\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{\nabla}_{0}p) \cdot (\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{\nabla}_{0}\delta p)\,\mathrm{d}V\end{aligned}

– Discrete linear system

.. math::

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{d} \end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_fluid_ale}\end{aligned}

– note that :math:`\boldsymbol{\mathsf{K}}_{pp}` is zero for Taylor-Hood
elements (without stabilization)

Coupling
========

Solid + 0D flow
---------------

| – Example: ``demos/solid_flow0d``
| – Problem type: ``solid_flow0d``
| – (`[eq:res_u_solid] <#eq:res_u_solid>`__) or
  (`[eq:res_u_solid_incomp] <#eq:res_u_solid_incomp>`__) augmented by
  following term:

  .. math::

     \begin{aligned}
     r_u \leftarrow r_u + \int\limits_{\mathit{\Gamma}_0^{\text{s}\mhyphen \mathrm{0d}}}\!\mathit{\Lambda}\,J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\cdot\delta\boldsymbol{u}\,\mathrm{d}A\end{aligned}

– Multiplier constraint

.. math::

   \begin{aligned}
   r_{\lambda}(\mathit{\Lambda},\boldsymbol{u};\delta\mathit{\Lambda}):= \left(\int\limits_{\mathit{\Gamma}_0^{\mathrm{\text{s}\mhyphen 0d}}}\! J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\boldsymbol{v}(\boldsymbol{u})\,\mathrm{d}A - Q^{\mathrm{0d}}(\mathit{\Lambda})\right) \delta\mathit{\Lambda}, \quad \forall \; \delta\mathit{\Lambda}\end{aligned}

– Discrete linear system for displacement-based solid

.. math::

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{u\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}u} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_solid_0d}\end{aligned}

– Discrete linear system for incompressible solid

.. math::

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{uu} & \boldsymbol{\mathsf{K}}_{up} & \boldsymbol{\mathsf{K}}_{u\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{K}}_{pu} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\  \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}u} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{u}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{u} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_solid_incomp_0d}\end{aligned}

Fluid + 0D flow
---------------

| – Example: ``demos/fluid_flow0d``
| – Problem type: ``fluid_flow0d``
| – (`[eq:res_v_fluid] <#eq:res_v_fluid>`__) augmented by following
  term:

  .. math::

     \begin{aligned}
     r_v \leftarrow r_v + \int\limits_{\mathit{\Gamma}_t^{\text{f}\mhyphen \mathrm{0d}}}\!\mathit{\Lambda}\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a\end{aligned}

– Multiplier constraint

.. math::

   \begin{aligned}
   r_{\lambda}(\mathit{\Lambda},\boldsymbol{v};\delta\mathit{\Lambda}):= \left(\int\limits_{\mathit{\Gamma}_t^{\mathrm{\text{f}\mhyphen 0d}}}\! \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}a - Q^{\mathrm{0d}}(\mathit{\Lambda})\right) \delta\mathit{\Lambda}, \quad \forall \; \delta\mathit{\Lambda}\end{aligned}

– Discrete linear system

.. math::

   \begin{aligned}
   \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}\\ \\  \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}}\end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_fluid_0d}\end{aligned}

ALE fluid + 0D flow
-------------------

| – Problem type: ``fluid_ale_flow0d``
| – (`[eq:res_v_fluid_ale] <#eq:res_v_fluid_ale>`__) augmented by
  following term:

  .. math::

     \begin{aligned}
     r_v \leftarrow r_v + \int\limits_{\mathit{\Gamma}_0^{\text{f}\mhyphen \mathrm{0d}}}\!\mathit{\Lambda}\,J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{v}\,\mathrm{d}A\end{aligned}

– Multiplier constraint

.. math::

   \begin{aligned}
   r_{\lambda}(\mathit{\Lambda},\boldsymbol{v},\boldsymbol{d};\delta\mathit{\Lambda}):= \left(\int\limits_{\mathit{\Gamma}_0^{\mathrm{\text{f}\mhyphen 0d}}}\! J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot(\boldsymbol{v}-\boldsymbol{w}(\boldsymbol{d}))\,\mathrm{d}A - Q^{\mathrm{0d}}(\mathit{\Lambda})\right) \delta\mathit{\Lambda}, \quad \forall \; \delta\mathit{\Lambda}\end{aligned}

| with
  :math:`\boldsymbol{w}(\boldsymbol{d})=\frac{\mathrm{d}\boldsymbol{d}}{\mathrm{d}t}`
| – Discrete linear system

  .. math::

     \begin{aligned}
     \begin{bmatrix} \boldsymbol{\mathsf{K}}_{vv} & \boldsymbol{\mathsf{K}}_{vp} & \boldsymbol{\mathsf{K}}_{v\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{vd} \\ \\ \boldsymbol{\mathsf{K}}_{pv} & \boldsymbol{\mathsf{K}}_{pp} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{pd} \\ \\ \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}v} & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}\mathit{\Lambda}} & \boldsymbol{\mathsf{K}}_{\mathit{\Lambda}d} \\ \\ \boldsymbol{\mathsf{K}}_{dv}  & \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \textcolor{lightgray}{\boldsymbol{\mathsf{0}}}& \boldsymbol{\mathsf{K}}_{dd} \end{bmatrix}_{n+1}^{k}\begin{bmatrix} \Delta\boldsymbol{\mathsf{v}} \\ \\ \Delta\boldsymbol{\mathsf{p}} \\ \\ \Delta\boldsymbol{\mathsf{\Lambda}}\\ \\ \Delta\boldsymbol{\mathsf{d}} \end{bmatrix}_{n+1}^{k+1}=-\begin{bmatrix} \boldsymbol{\mathsf{r}}_{v} \\ \\ \boldsymbol{\mathsf{r}}_{p} \\ \\ \boldsymbol{\mathsf{r}}_{\mathit{\Lambda}} \\ \\ \boldsymbol{\mathsf{r}}_{d}\end{bmatrix}_{n+1}^{k} \label{eq:lin_sys_fluid_ale_0d}\end{aligned}

Fluid-Solid Interaction (FSI)
-----------------------------

| – Problem type: ``fsi``
| – Not yet fully implemented!

Fluid-Solid Interaction (FSI) + 0D flow
---------------------------------------

| – Problem type: ``fsi_flow0d``
| – Not yet fully implemented!

0D flow: Lumped parameter models
================================

| – Example: ``demos/flow0d``
| – Problem type: ``flow0d``

“Syspul” circulation model
--------------------------

.. math::

   \begin{aligned}
   {2}
   &\text{left heart and systemic circulation} && \nonumber\\
   &-Q_{\mathrm{at}}^{\ell} = \sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{pul}}}q_{\mathrm{ven},i}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell} && \text{\;left atrium flow balance}\nonumber\\
   &q_{\mathrm{v,in}}^{\ell} = q_{\mathrm{mv}}(p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}) && \text{\;mitral valve momentum}\label{eq:mv_flow}\\
   &-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell} && \text{\;left ventricle flow balance}\nonumber\\
   &q_{\mathrm{v,out}}^{\ell} = q_{\mathrm{av}}(p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}) && \text{\;aortic valve momentum}\label{eq:av_flow}\\
   &-Q_{\mathrm{aort}}^{\mathrm{sys}} = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}} - \mathbb{I}^{\mathrm{cor}}\sum\limits_{i=1}^{2}q_{\mathrm{ar,cor,in},i}^{\mathrm{sys}} && \text{\;aortic root flow balance}\nonumber\\
   &I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}} && \text{\;aortic root inertia}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}} && \text{\;systemic arterial flow balance}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}} && \text{\;systemic arterial momentum}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{sys}}}q_{\mathrm{ven},i}^{\mathrm{sys}}\ && \text{\;systemic venous flow balance}\nonumber\\
   &L_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven},i}^{\mathrm{sys}}\, q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at},i}^{r} && \text{\;systemic venous momentum}\nonumber\\
   &\qquad\qquad i \in \{1,...,n_{\mathrm{ven}}^{\mathrm{sys}}\} && \nonumber\end{aligned}

.. math::

   \begin{aligned}
   {2}
   &\text{right heart and pulmonary circulation} && \nonumber\\
   &-Q_{\mathrm{at}}^{r} = \sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{sys}}}q_{\mathrm{ven},i}^{\mathrm{sys}} - \mathbb{I}^{\mathrm{cor}} q_{\mathrm{ven,cor,out}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r} && \text{\;right atrium flow balance}\nonumber\\
   &q_{\mathrm{v,in}}^{r} = q_{\mathrm{tv}}(p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}) && \text{\;tricuspid valve momentum}\label{eq:tv_flow}\\
   &-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r} && \text{\;right ventricle flow balance}\nonumber\\
   &q_{\mathrm{v,out}}^{r} = q_{\mathrm{pv}}(p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}) && \text{\;pulmonary valve momentum}\label{eq:pv_flow}\\
   &C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}} && \text{\;pulmonary arterial flow balance}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{ven}}^{\mathrm{pul}} && \text{\;pulmonary arterial momentum}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - \sum\limits_{i=1}^{n_{\mathrm{ven}}^{\mathrm{pul}}}q_{\mathrm{ven},i}^{\mathrm{pul}} && \text{\;pulmonary venous flow balance}\nonumber\\
   &L_{\mathrm{ven},i}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ven},i}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven},i}^{\mathrm{pul}}\, q_{\mathrm{ven},i}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at},i}^{\ell} && \text{\;pulmonary venous momentum}\nonumber\\
   &\qquad\qquad i \in \{1,...,n_{\mathrm{ven}}^{\mathrm{pul}}\} && \nonumber\end{aligned}

with:

.. math::

   \begin{aligned}
   Q_{\mathrm{at}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{at}}^{\ell}}{\mathrm{d}t}, \quad
   Q_{\mathrm{v}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{v}}^{\ell}}{\mathrm{d}t}, \quad
   Q_{\mathrm{at}}^{r} := -\frac{\mathrm{d}V_{\mathrm{at}}^{r}}{\mathrm{d}t}, \quad
   Q_{\mathrm{v}}^{r} := -\frac{\mathrm{d}V_{\mathrm{v}}^{r}}{\mathrm{d}t},
   \quad
   Q_{\mathrm{aort}}^{\mathrm{sys}} := -\frac{\mathrm{d}V_{\mathrm{aort}}^{\mathrm{sys}}}{\mathrm{d}t}\nonumber\end{aligned}

and:

.. math::

   \begin{aligned}
   \mathbb{I}^{\mathrm{cor}} = \begin{cases} 1, & \text{if \, CORONARY\_MODEL}, \\ 0, & \text{else} \end{cases}\nonumber\end{aligned}

The volume :math:`V` of the heart chambers (0D) is modeled by the
volume-pressure relationship

.. math:: V(t) = \frac{p}{E(t)} + V_{\mathrm{u}},

with the unstressed volume :math:`V_{\mathrm{u}}` and the time-varying
elastance

.. math:: E(t)=\left(E_{\mathrm{max}}-E_{\mathrm{min}}\right)\cdot \hat{y}(t)+E_{\mathrm{min}} \label{at_elast},

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
   R_{\min}, & p \geq p_{\mathrm{open}} \end{cases}\nonumber\end{aligned}

**Remark:** Non-smooth flow-pressure relationship

Valve model ``pwlin_time``:

.. math::

   \begin{aligned}
   q(p-p_{\mathrm{open}}) = \frac{p-p_{\mathrm{open}}}{\tilde{R}},\quad \text{with}\; \tilde{R} = \begin{cases} \begin{cases} R_{\max}, & t < t_{\mathrm{open}} \;\text{and}\; t \geq t_{\mathrm{close}} \\
   R_{\min}, & t \geq t_{\mathrm{open}} \;\text{or}\; t < t_{\mathrm{close}} \end{cases}, & t_{\mathrm{open}} > t_{\mathrm{close}} \\ \begin{cases} R_{\max}, & t < t_{\mathrm{open}} \;\text{or}\; t \geq t_{\mathrm{close}} \\
   R_{\min}, & t \geq t_{\mathrm{open}} \;\text{and}\; t < t_{\mathrm{close}} \end{cases}, & \text{else} \end{cases}\nonumber\end{aligned}

**Remark:** Non-smooth flow-pressure relationship with resistance only
dependent on timings, not the pressure difference!

Valve model ``smooth_pres_resistance``:

.. math::

   \begin{aligned}
   q(p-p_{\mathrm{open}}) = \frac{p-p_{\mathrm{open}}}{\tilde{R}},\quad \text{with}\;\tilde{R} = 0.5\left(R_{\max}-R_{\min}\right)\left(\tanh\frac{p-p_{\mathrm{open}}}{\epsilon}+1\right) + R_{\min}\nonumber\end{aligned}

**Remark:** Smooth but potentially non-convex flow-pressure
relationship!

| Valve model ``smooth_pres_momentum``:

  .. math::

     \begin{aligned}
     q(p-p_{\mathrm{open}}) = \begin{cases}\frac{p-p_{\mathrm{open}}}{R_{\max}}, & p < p_{\mathrm{open}}-0.5\epsilon \\ h_{00}p_{0} + h_{10}m_{0}\epsilon + h_{01}p_{1} + h_{11}m_{1}\epsilon, & p \geq p_{\mathrm{open}}-0.5\epsilon \;\text{and}\; p < p_{\mathrm{open}}+0.5\epsilon \\ \frac{p-p_{\mathrm{open}}}{R_{\min}}, & p \geq p_{\mathrm{open}}+0.5\epsilon  \end{cases}\nonumber\end{aligned}

  with

  .. math::

     \begin{aligned}
     p_{0}=\frac{-0.5\epsilon}{R_{\max}}, \qquad m_{0}=\frac{1}{R_{\max}}, \qquad && p_{1}=\frac{0.5\epsilon}{R_{\min}}, \qquad m_{1}=\frac{1}{R_{\min}} \nonumber\end{aligned}

  and

  .. math::

     \begin{aligned}
     h_{00}=2s^3 - 3s^2 + 1, &\qquad h_{01}=-2s^3 + 3s^2, \nonumber\\
     h_{10}=s^3 - 2s^2 + s, &\qquad h_{11}=s^3 - s^2 \nonumber\end{aligned}

  with

  .. math::

     \begin{aligned}
     s=\frac{p-p_{\mathrm{open}}+0.5\epsilon}{\epsilon} \nonumber\end{aligned}

  **Remarks:** :math:`-` Collapses to valve model ``pwlin_pres`` for
  :math:`\epsilon=0`
| :math:`-` Smooth and convex flow-pressure relationship

Valve model ``pw_pres_regurg``:

.. math:: q(p-p_{\mathrm{open}}) = \begin{cases} c A_{\mathrm{o}} \sqrt{p-p_{\mathrm{open}}}, & p < p_{\mathrm{open}} \\ \frac{p-p_{\mathrm{open}}}{R_{\min}}, & p \geq p_{\mathrm{open}}  \end{cases}\nonumber

**Remark:** Model to allow a regurgitant valve in the closed state,
degree of regurgitation can be varied by specifying the valve
regurgitant area :math:`A_{\mathrm{o}}`

Coronary circulation model:

.. math::

   \begin{aligned}
   &C_{\mathrm{cor,p}}^{\mathrm{sys},\ell} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys},\ell}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell} - q_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\nonumber\\
   &R_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,p}}^{\mathrm{sys},\ell}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys},\ell} - Z_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell}\nonumber\\
   &C_{\mathrm{cor,d}}^{\mathrm{sys},\ell} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys},\ell}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys},\ell} - q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}\nonumber\\
   &R_{\mathrm{cor,d}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}=p_{\mathrm{cor,d}}^{\mathrm{sys},\ell}-p_{\mathrm{at}}^{r}\nonumber\\
   &C_{\mathrm{cor,p}}^{\mathrm{sys},r} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys},r}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys},r}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys},r}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys},r} - q_{\mathrm{cor,p}}^{\mathrm{sys},r}\nonumber\\
   &R_{\mathrm{cor,p}}^{\mathrm{sys},r}\,q_{\mathrm{cor,p}}^{\mathrm{sys},r}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys},r} - Z_{\mathrm{cor,p}}^{\mathrm{sys},r}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys},r}\nonumber\\
   &C_{\mathrm{cor,d}}^{\mathrm{sys},r} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys},r}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys},r} - q_{\mathrm{cor,d}}^{\mathrm{sys},r}\nonumber\\
   &R_{\mathrm{cor,d}}^{\mathrm{sys},r}\,q_{\mathrm{cor,d}}^{\mathrm{sys},r}=p_{\mathrm{cor,d}}^{\mathrm{sys},r}-p_{\mathrm{at}}^{r}\nonumber\\
   &0=q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}+q_{\mathrm{cor,d}}^{\mathrm{sys},r}-q_{\mathrm{cor,d,out}}^{\mathrm{sys}}\nonumber\end{aligned}

“Syspulcap” circulation model
-----------------------------

.. math::

   \begin{aligned}
   &-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\nonumber\\
   &\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\nonumber\\
   &-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\nonumber\\
   &\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
   &0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}}\nonumber\\
   &I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}} -p_{\mathrm{ar,peri}}^{\mathrm{sys}}\nonumber\\
   &\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}}\nonumber\\
   &R_{\mathrm{ar},i}^{\mathrm{sys}}\,q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
   &C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
   &R_{\mathrm{ven},i}^{\mathrm{sys}}\,q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer,cor}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
   &L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber\end{aligned}

.. math::

   \begin{aligned}
   &-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\nonumber\\
   &\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\nonumber\\
   &-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\nonumber\\
   &\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
   &C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
   &R_{\mathrm{cap}}^{\mathrm{pul}}\,q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
   &L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}\nonumber\end{aligned}

with:

.. math::

   \begin{aligned}
   Q_{\mathrm{at}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{at}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{v}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{at}}^{r} := -\frac{\mathrm{d}V_{\mathrm{at}}^{r}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{r} := -\frac{\mathrm{d}V_{\mathrm{v}}^{r}}{\mathrm{d}t}\nonumber\end{aligned}

“Syspulcapcor” circulation model
--------------------------------

.. math::

   \begin{aligned}
   &-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\nonumber\\
   &\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\nonumber\\
   &-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\nonumber\\
   &\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
   &0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar,cor,in}}^{\mathrm{sys}}\nonumber\\
   &I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}}\nonumber\\
   &C_{\mathrm{ar,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor,in}}^{\mathrm{sys}} - q_{\mathrm{ar,cor}}^{\mathrm{sys}}\nonumber\\
   &R_{\mathrm{ar,cor}}^{\mathrm{sys}}\,q_{\mathrm{ar,cor}}^{\mathrm{sys}} = p_{\mathrm{ar}}^{\mathrm{sys}} - p_{\mathrm{ven,cor}}^{\mathrm{sys}}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}} -p_{\mathrm{ar,peri}}^{\mathrm{sys}}\nonumber\\
   &\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}}\nonumber\\
   &R_{\mathrm{ar},i}^{\mathrm{sys}}\,q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}}\nonumber\\
   &C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}}\nonumber\\
   &R_{\mathrm{ven},i}^{\mathrm{sys}}\,q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
   &L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber\\
   &C_{\mathrm{ven,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven,cor}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor}}^{\mathrm{sys}}-q_{\mathrm{ven,cor}}^{\mathrm{sys}}\nonumber\\
   &R_{\mathrm{ven,cor}}^{\mathrm{sys}}\,q_{\mathrm{ven,cor}}^{\mathrm{sys}} = p_{\mathrm{ven,cor}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber\end{aligned}

.. math::

   \begin{aligned}
   &-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} + q_{\mathrm{ven,cor}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\nonumber\\
   &\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\nonumber\\
   &-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\nonumber\\
   &\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
   &C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
   &L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
   &C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
   &R_{\mathrm{cap}}^{\mathrm{pul}}\,q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
   &C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
   &L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}\nonumber\end{aligned}

with:

.. math::

   \begin{aligned}
   Q_{\mathrm{at}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{at}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{\ell} := -\frac{\mathrm{d}V_{\mathrm{v}}^{\ell}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{at}}^{r} := -\frac{\mathrm{d}V_{\mathrm{at}}^{r}}{\mathrm{d}t}, \qquad
   Q_{\mathrm{v}}^{r} := -\frac{\mathrm{d}V_{\mathrm{v}}^{r}}{\mathrm{d}t} \nonumber\end{aligned}

.. container:: references
   :name: refs

   .. container::
      :name: ref-arthurs2016

      Arthurs, C. J., K. D. Lau, K. N. Asrress, S. R. Redwood, and C. A.
      Figueroa. 2016. “A Mathematical Model of Coronary Blood Flow
      Control: Simulation of Patient-Specific Three-Dimensional
      Hemodynamics During Exercise.” *Am J Physiol Heart Circ Physiol*
      310 (9): H1242–H1258. https://doi.org/10.1152/ajpheart.00517.2015.

   .. container::
      :name: ref-petsc-user-ref

      Balay, Satish, Shrirang Abhyankar, Mark F. Adams, Steven Benson,
      Jed Brown, Peter Brune, Kris Buschelman, et al. 2022. “PETSc/TAO
      Users Manual.” ANL-21/39 - Revision 3.17. Argonne National
      Laboratory.

   .. container::
      :name: ref-farhat2014

      Farhat, C., P. Avery, T. Chapman, and J. Cortial. 2014.
      “Dimensional Reduction of Nonlinear Finite Element Dynamic Models
      with Finite Rotations and Energy-Based Mesh Sampling and Weighting
      for Computational Efficiency.” *International Journal for
      Numerical Methods in Engineering* 98 (9): 625–62.
      https://doi.org/10.1002/nme.4668.

   .. container::
      :name: ref-gee2010

      Gee, M. W., Ch. Förster, and W. A. Wall. 2010. “A Computational
      Strategy for Prestressing Patient-Specific Biomechanical Problems
      Under Finite Deformation.” *International Journal for Numerical
      Methods in Biomedical Engineering* 26 (1): 52–72.
      https://doi.org/10.1002/cnm.1236.

   .. container::
      :name: ref-gee2009

      Gee, M. W., C. T. Kelley, and R. B. Lehoucq. 2009.
      “Pseudo-Transient Continuation for Nonlinear Transient
      Elasticity.” *International Journal for Numerical Methods in
      Engineering* 78 (10): 1209–19. https://doi.org/10.1002/nme.2527.

   .. container::
      :name: ref-goektepe2010

      Göktepe, S., O. J. Abilez, K. K. Parker, and E. Kuhl. 2010. “A
      Multiscale Model for Eccentric and Concentric Cardiac Growth
      Through Sarcomerogenesis.” *J Theor Biol* 265 (3): 433–42.
      https://doi.org/10.1016/j.jtbi.2010.04.023.

   .. container::
      :name: ref-guccione1995

      Guccione, J. M., K. D. Costa, and A. D. McCulloch. 1995. “Finite
      Element Stress Analysis of Left Ventricular Mechanics in the
      Beating Dog Heart.” *J Biomech* 28 (10): 1167–77.
      https://doi.org/10.1016/0021-9290(94)00174-3.

   .. container::
      :name: ref-hirschvogel2018

      Hirschvogel, M. 2018. *Computational Modeling of Patient-Specific
      Cardiac Mechanics with Model Reduction-Based Parameter Estimation
      and Applications to Novel Heart Assist Technologies*. PhD thesis,
      Technische Universität München.

   .. container::
      :name: ref-hirschvogel2022preprint

      Hirschvogel, M., M. Balmus, M. Bonini, and D. Nordsletten. 2022.
      “Fluid-Reduced-Solid Interaction (FrSI): Physics- and
      Projection-Based Model Reduction for Cardiovascular Applications.”
      *Preprint, Submitted to Elsevier*.
      https://doi.org/10.2139/ssrn.4281317.

   .. container::
      :name: ref-hirschvogel2017

      Hirschvogel, M., M. Bassilious, L. Jagschies, S. M. Wildhirt, and
      M. W. Gee. 2017. “A Monolithic 3D-0D Coupled Closed-Loop Model of
      the Heart and the Vascular System: Experiment-Based Parameter
      Estimation for Patient-Specific Cardiac Mechanics.” *Int J Numer
      Method Biomed Eng* 33 (8): e2842.
      https://doi.org/10.1002/cnm.2842.

   .. container::
      :name: ref-holzapfel2000

      Holzapfel, G. A. 2000. *Nonlinear Solid Mechanics – A Continuum
      Approach for Engineering*. Wiley Press Chichester.

   .. container::
      :name: ref-holzapfel2009

      Holzapfel, G. A., and R. W. Ogden. 2009. “Constitutive Modelling
      of Passive Myocardium: A Structurally Based Framework for Material
      Characterization.” *Phil Trans R Soc A* 367 (1902): 3445–75.
      https://doi.org/10.1098/rsta.2009.0091.

   .. container::
      :name: ref-logg2012

      Logg, A., K.-A. Mardal, and G. N. Wells, eds. 2012. *Automated
      Solution of Differential Equations by the Finite Element Method –
      the FEniCS Book*. Springer.
      https://doi.org/10.1007/978-3-642-23099-8.

   .. container::
      :name: ref-schein2021

      Schein, A., and M. W. Gee. 2021. “Greedy Maximin Distance Sampling
      Based Model Order Reduction of Prestressed and Parametrized
      Abdominal Aortic Aneurysms.” *Advanced Modeling and Simulation in
      Engineering Sciences* 8 (18).
      https://doi.org/10.1186/s40323-021-00203-7.

   .. container::
      :name: ref-tezduyar2000

      Tezduyar, T. E., and Y. Osawa. 2000. “Finite Element Stabilization
      Parameters Computed from Element Matrices and Vectors.” *Computer
      Methods in Applied Mechanics and Engineering* 190 (3–4): 411–30.
      https://doi.org/10.1016/S0045-7825(00)00211-5.

   .. container::
      :name: ref-westerhof2009

      Westerhof, N., J.-W. Lankhaar, and B. E. Westerhof. 2009. “The
      Arterial Windkessel.” *Med Biol Eng Comput* 47 (2): H81–H88.
      https://doi.org/10.1007/s11517-008-0359-2.
