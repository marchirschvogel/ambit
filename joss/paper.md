---
title: 'Ambit – A FEniCS-based cardiovascular multi-physics solver'
tags:
  - Python
  - cardiovascular mechanics
  - finite strain solid mechanics
  - nonlinear elastodynamics
  - fluid dynamics
  - 0D lumped models
  - fluid-solid interaction
  - fsi
  - multi-physics coupling
authors:
  - name: Marc Hirschvogel
    orcid: 0000-0002-4575-9120
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted: "1, 2")
affiliations:
 - name: Department of Biomedical Engineering, School of Biomedical Engineering & Imaging Sciences, King's College London, London, United Kingdom
   index: 1
 - name: MOX, Dipartimento di Matematica, Politecnico di Milano, Milan, Italy
   index: 2
date: 7 July 2023
bibliography: paper.bib

---

# Summary

Ambit is an open-source multi-physics finite element solver written in Python, supporting solid and fluid mechanics, fluid-structure interaction (FSI), and lumped-parameter models. It is tailored towards solving problems in cardiac mechanics, but may also be used for more general nonlinear finite element analysis. The code encompasses re-implementations and generalizations of methods developed by the author for his PhD thesis [@hirschvogel2019disspub] and beyond. Ambit makes use of the open-source finite element library [FEniCS/dolfinx](https://fenicsproject.org) [@logg2012-fenics] along with the linear algebra package [PETSc](https://petsc.org) [@balay2022-petsc], hence guaranteeing a state-of-the-art finite element and linear algebra backend. It is constantly updated to ensure compatibility with a recent dolfinx development version. I/O routines are designed such that the user only needs to provide input files that define parameters through Python dictionaries, hence no programming or in-depth knowledge of any library-specific syntax is required.

Ambit provides general nonlinear (compressible or incompressible) finite strain solid dynamics [@holzapfel2000], implementing a range of hyperelastic, viscous, and active material models. Specifically, the well-known anisotropic Holzapfel-Ogden [@holzapfel2009] and Guccione models [@guccione1995] for structural description of the myocardium are provided, along with a bunch of other models. It further implements strain- and stress-mediated volumetric growth models [@goektepe2010] that allow to model (maladaptive) ventricular shape and size changes. Inverse mechanics approaches to imprint loads into a reference state are implemented using the so-called prestressing method [@gee2010] in displacement formulation [@schein2021].

Furthermore, fluid dynamics in terms of incompressible Navier-Stokes/Stokes equations – either in Eulerian or Arbitrary Lagrangian-Eulerian (ALE) reference frames – are implemented. Taylor-Hood elements or equal-order approximations with SUPG/PSPG stabilization [@tezduyar2000] can be used.

A variety of reduced 0D lumped models targeted at blood circulation modeling are implemented, including 3- and 4-element Windkessel models [@westerhof2009] as well as closed-loop full circulation [@hirschvogel2017] and coronary flow models [@arthurs2016].

Monolithic fluid-solid interaction (FSI) [@nordsletten2011] in ALE formulation using a Lagrange multiplier field is supported, along with coupling of 3D and 0D models (solid or fluid with 0D lumped circulation systems) such that cardiovascular simulations with realistic boundary conditions can be performed.

Implementations for a recently proposed novel physics- and projection-based model reduction for FSI, denoted as fluid-reduced-solid interaction (FrSI) [@hirschvogel2022preprint], are provided, along with POD-based Galerkin model reduction techniques [@farhat2014] using full or boundary subspaces.

The nonlinear (single- or multi-field) problems are solved with a customized Newton solver with PTC [@gee2009] adaptivity in case of divergence, providing robustness for numerically challenging problems. Linear solvers and preconditioners can be chosen from the PETSc repertoire, and specific block preconditioners are made available for coupled problems.

Avenues for future functionality include cardiac electrophysiology, scalar transport, or finite strain plasticity.

# Statement of need

Cardiovascular disease entities are the most prevalent ones in the industrialized world [@dimmeler2011; @luepker2011] and a leading cause of death worldwide. Therefore, models that promote a better understanding of cardiac diseases and their progression represent a valuable tool to guide or assist therapy planning, support device dimensioning and design [@hirschvogel2019], or help predict intervention planning [@bonini2022; @taylor2013].

Software packages that are tailored towards cardiac modeling have been provided to the open source community. Amongst them are the cardiovascular FSI solver svFSI [@zhu2022-svfsi] along with SimVascular [@updegrove2017-simvascular], providing a full medical image-to-model pipeline, as well as FEBio [@maas2012-febio], focusing on advanced structural mechanics of soft tissue. FEniCS-based open-source solvers are pulse [@finsberg2019-pulse] for cardiac solid mechanics and cbcbeat [@rognes2017-cbcbeat] for cardiac electrophysiology, both fused to a combined toolkit for cardiac electro-mechanics named simcardems [@finsberg2023-simcardems]. Another framework for simulating cardiac electrophysiology is openCARP [@plank2021-opencarp], and CRIMSON [@arthurs2021-crimson] provides a modeling suite for 3D and reduced-dimensional hemodynamics in arteries. A general purpose library that provides the building blocks for cardiac modeling is lifex [@africa2022-lifex], and a FEniCS-based monolithic FSI solver for general applications is turtleFSI [@bergersen2020-turtlefsi].

Ambit represents a complete open-source code for simulating cardiac mechanics, encompassing advanced structural mechanics of the myocardium, ventricular fluid dynamics, reduced-dimensional blood flow, and multi-physics coupling. Therefore, a wide range of mechanical problems can be simulated, and the code structure allows easy and straightforward extensibility (e.g. implementations of new constitutive models) without the need for low-level library-specific syntax or advanced programming. Due to its simple design in terms of clearly organized input files, Ambit is easy to use and hence represents a valuable tool for novices or advanced researchers who want to address cardiovascular mechanics problems.

# Basic code structure

\autoref{fig:codedesign} represents a basic sketch of the main building blocks of Ambit. Depending on the physics of interest, the respective problem class is instantiated along with all the necessary input parameters, including boundary conditions (Dirichlet, Neumann, Robin), load curves, specification of coupling interfaces, etc. Single-physics problems like nonlinear elastodynamics (problem type `solid`) or fluid mechanics (problem type `fluid`) as well as 0D blood flow (problem type `flow0d`) can be solved as standalone problems. Additionally, FSI (problem type `fsi`) and 3D-0D coupling for 0D flow to 3D solid or fluid domains is supported (problem types `solid_flow0d` and `fluid_flow0d`), as well as fluid mechanics in ALE description (problem type `fluid_ale`), plus coupling to 0D models (problem types `fluid_ale_flow0d` and `fsi_flow0d`).

The (coupled) problem object then is passed to a solver class, which calls the main routine to solve the nonlinear problem. This routine implements a time stepping scheme and a monolithic Newton solver which solves the (coupled multi-physics or single-field) problem and updates all variables simultaneously.

![Basic sketch of Ambit code structure: Problem class, solver class, and main code execution flow. Single-physics problems that can be solved encompass solid mechanics (`solid`), fluid mechanics (`fluid`), or 0D models (`flow0d`). Two-physics problems like 3D-0D coupling (`solid_flow0d`, `fluid_flow0d`), as well as fluid in ALE description (`fluid_ale`) are defined by instantiating the respective single-physics problems. Three-physics problems arise for coupling of ALE fluid to 0D models (`fluid_ale_flow0d`) or for fluid-solid interaction (`fsi`), whereas four-physics problems would encompass FSI linked to 0D models (`fsi_flow0d`). Note that the single-physics problem `ale` just mimics a dummy linear elastic solid and would be irrelevant as a standalone problem.\label{fig:codedesign}](code_design.pdf){ width=100% }

<!--# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.-->

# References
