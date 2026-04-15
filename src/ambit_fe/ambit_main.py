#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from mpi4py import MPI
from . import ioroutines
from .base import problem_base


class Ambit:
    """
    Ambit main class

    Attributes
    ----------
    comm :
        MPI communicator
    comm_sq :
        Sequential MPI communicator for one core
    entity_maps : list
        Entity maps for mixed domain problems
    mp :
        Model problem object
    ms :
        Model solver object
    """

    def __init__(
        self,
        io_params,
        ctrl_params,
        time_params,
        solver_params,
        fem_params={},
        constitutive_params={},
        boundary_conditions={},
        time_curves=None,
        coupling_params={},
        multiscale_params={},
        mor_params={},
    ):
        """
        Parameters
        ----------
        io_params : dict
            Input/output parameters
        ctrl_params : dict
            Global control parameters
        time_params : dict or list of dicts
            Time integration parameters
        solver_params : dict
            Solver parameters for nonlinear and linear solution schemes
        fem_params : dict or list of dicts, optional
            Finite element parameters
        constitutive_params : dict or list of dicts, optional
            Material parameters
        boundary_conditions : dict, optional
            Boundary conditions
        time_curves : class, optional
            Time functions
        coupling_params : dict or list of dicts, optional
            Parameters for multi-physics coupling
        multiscale_params : dict, optional
            Parameters for multiscale simulation (growth & remodeling)
        mor_params : dict, optional
            Model order reduction parameters
        """

        # MPI communicator
        self.comm = MPI.COMM_WORLD
        self.comm_sq = MPI.COMM_SELF

        problem_type = io_params["problem_type"]

        # entity maps for coupled/multi-mesh problems
        self.entity_maps = []

        if problem_type == "solid":
            from .solid import solid_main

            io = ioroutines.IO_solid(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_solid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = solid_main.SolidmechanicsProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = solid_main.SolidmechanicsSolver(self.mp, solver_params)

        elif problem_type == "fluid":
            from .fluid import fluid_main

            io = ioroutines.IO_fluid(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = fluid_main.FluidmechanicsProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_main.FluidmechanicsSolver(self.mp, solver_params)

        elif problem_type == "ale":
            from .ale import ale_main

            io = ioroutines.IO_ale(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_ale = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = ale_main.AleProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = ale_main.AleSolver(self.mp, solver_params)

        elif problem_type == "fluid_ale":
            from .coupling import fluid_ale_main

            io = ioroutines.IO_fluid_ale(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid, io.m_id_ale = 0, 0

            io.mesh_ = [io.mesh, io.mesh]
            io.mt_d_ = [io.mt_d, io.mt_d]
            io.mt_b_ = [io.mt_b, io.mt_b]
            io.mt_sb_ = [io.mt_sb, io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = fluid_ale_main.FluidmechanicsAleProblem(
                pbase,
                io_params,
                time_params,
                fem_params[0],
                fem_params[1],
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions[0],
                boundary_conditions[1],
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_ale_main.FluidmechanicsAleSolver(self.mp, solver_params)

        elif problem_type == "fluid_ale_flow0d":
            from .coupling import fluid_ale_flow0d_main

            io = ioroutines.IO_fluid_ale(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid, io.m_id_ale = 0, 0

            io.mesh_ = [io.mesh, io.mesh]
            io.mt_d_ = [io.mt_d, io.mt_d]
            io.mt_b_ = [io.mt_b, io.mt_b]
            io.mt_sb_ = [io.mt_sb, io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            self.mp = fluid_ale_flow0d_main.FluidmechanicsAleFlow0DProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params[0],
                fem_params[1],
                constitutive_params[0],
                constitutive_params[1],
                constitutive_params[2],
                boundary_conditions[0],
                boundary_conditions[1],
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_ale_flow0d_main.FluidmechanicsAleFlow0DSolver(self.mp, solver_params)

        elif problem_type == "fluid_multiphase":
            from .coupling import fluid_multiphase_main

            io = ioroutines.IO_fluid_multiphase(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid, io.m_id_phase = 0, 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = fluid_multiphase_main.FluidmechanicsMultiphaseProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params[0],
                fem_params[1],
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions[0],
                boundary_conditions[1],
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_multiphase_main.FluidmechanicsMultiphaseSolver(self.mp, solver_params)

        elif problem_type == "fluid_ale_multiphase":
            from .coupling import fluid_ale_multiphase_main

            io = ioroutines.IO_fluid_ale_multiphase(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid, io.m_id_phase, io.m_id_ale = 0, 0, 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = fluid_ale_multiphase_main.FluidmechanicsAleMultiphaseProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params[0],
                fem_params[1],
                fem_params[2],
                constitutive_params[0],
                constitutive_params[1],
                constitutive_params[2],
                boundary_conditions[0],
                boundary_conditions[1],
                boundary_conditions[2],
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_ale_multiphase_main.FluidmechanicsAleMultiphaseSolver(self.mp, solver_params)

        elif problem_type == "flow0d":
            from .flow0d import flow0d_main

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = flow0d_main.Flow0DProblem(pbase, io_params, time_params, constitutive_params, time_curves)
            self.ms = flow0d_main.Flow0DSolver(self.mp, solver_params)

        elif problem_type == "solid_flow0d":
            from .coupling import solid_flow0d_main

            io = ioroutines.IO_solid(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_solid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            self.mp = solid_flow0d_main.SolidmechanicsFlow0DProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params,
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = solid_flow0d_main.SolidmechanicsFlow0DSolver(self.mp, solver_params)

        elif problem_type == "solid_flow0d_periodicref":
            from .coupling import solid_flow0d_main
            from .coupling import solid_flow0d_periodicref_main

            io = ioroutines.IO_solid(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_solid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            self.mp = solid_flow0d_main.SolidmechanicsFlow0DProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params,
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = solid_flow0d_periodicref_main.SolidmechanicsFlow0DPeriodicRefSolver(self.mp, solver_params)

        elif problem_type == "fluid_flow0d":
            from .coupling import fluid_flow0d_main

            io = ioroutines.IO_fluid(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            self.mp = fluid_flow0d_main.FluidmechanicsFlow0DProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params,
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_flow0d_main.FluidmechanicsFlow0DSolver(self.mp, solver_params)

        elif problem_type == "solid_flow0d_multiscale_gandr":
            raise RuntimeError("Solid-flow0d multiscale G&R currently broken. To be fixed soon!")

            from .multiscale import solid_flow0d_growthremodel_main

            io = ioroutines.IO_solid(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_solid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            self.mp = solid_flow0d_growthremodel_main.SolidmechanicsFlow0DMultiscaleGrowthRemodelingProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                time_params[2],
                fem_params,
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions,
                time_curves,
                multiscale_params,
                io,
                coupling_params=coupling_params,
            )
            self.mp.set_variational_forms()
            self.ms = solid_flow0d_growthremodel_main.SolidmechanicsFlow0DMultiscaleGrowthRemodelingSolver(
                self.mp, solver_params
            )

        elif problem_type == "fsi":
            from .coupling import fsi_main

            io = ioroutines.IO_fsi(io_params, constitutive_params[0:2], entity_maps=self.entity_maps, comm=self.comm)
            io.readin_mesh()
            assert(fem_params[0]["quad_degree"]==fem_params[1]["quad_degree"])  # in FSI, these should be the same...
            io.create_integration_measures(io.mesh, io.domain_ids[0], io.domain_ids[1], coupling_params["coupling_fluid_ale"]["interface"], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)  # we want the fields on the master, entity maps will restrict
            io.create_submeshes()
            io.m_id_solid, io.m_id_fluid, io.m_id_ale = 0, 1, 1

            io.mesh_ = [io.msh_emap_solid[0], io.msh_emap_fluid[0]]
            io.mt_d_ = [io.mt_d_solid, io.mt_d_fluid]
            io.mt_b_ = [io.mt_b_solid, io.mt_b_fluid]
            io.mt_sb_ = [io.mt_sb_solid, io.mt_sb_fluid]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            try:
                bcs_lm = boundary_conditions[3]
            except:
                bcs_lm = None

            self.mp = fsi_main.FSIProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                fem_params[0],
                fem_params[1],
                fem_params[2],
                constitutive_params[0],
                constitutive_params[1],
                constitutive_params[2],
                boundary_conditions[0],
                boundary_conditions[1],
                boundary_conditions[2],
                bcs_lm,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fsi_main.FSISolver(self.mp, solver_params)

        elif problem_type == "fsi_flow0d":
            from .coupling import fsi_flow0d_main

            io = ioroutines.IO_fsi(io_params, constitutive_params[0:2], entity_maps=self.entity_maps, comm=self.comm)
            io.readin_mesh()
            assert(fem_params[0]["quad_degree"]==fem_params[1]["quad_degree"])  # in FSI, these should be the same...
            io.create_integration_measures(io.mesh, io.domain_ids[0], io.domain_ids[1], coupling_params[0]["coupling_fluid_ale"]["interface"], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)  # we want the fields on the master, entity maps will restrict
            io.create_submeshes()
            io.m_id_solid, io.m_id_fluid, io.m_id_ale = 0, 1, 1

            io.mesh_ = [io.msh_emap_solid[0], io.msh_emap_fluid[0]]
            io.mt_d_ = [io.mt_d_solid, io.mt_d_fluid]
            io.mt_b_ = [io.mt_b_solid, io.mt_b_fluid]
            io.mt_sb_ = [io.mt_sb_solid, io.mt_sb_fluid]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            try:
                bcs_lm = boundary_conditions[3]
            except:
                bcs_lm = None

            self.mp = fsi_flow0d_main.FSIFlow0DProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                time_params[2],
                fem_params[0],
                fem_params[1],
                fem_params[2],
                constitutive_params[0],
                constitutive_params[1],
                constitutive_params[2],
                constitutive_params[3],
                boundary_conditions[0],
                boundary_conditions[1],
                boundary_conditions[2],
                bcs_lm,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fsi_flow0d_main.FSIFlow0DSolver(self.mp, solver_params)

        elif problem_type == "fsi_multiphase":
            from .coupling import fsi_multiphase_main

            io = ioroutines.IO_fsi_multiphase(io_params, constitutive_params[0:2], entity_maps=self.entity_maps, comm=self.comm)
            io.readin_mesh()
            assert(fem_params[0]["quad_degree"]==fem_params[1]["quad_degree"])  # in FSI, these should be the same...
            io.create_integration_measures(io.mesh, io.domain_ids[0], io.domain_ids[1], coupling_params[0]["coupling_fluid_ale"]["interface"], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)  # we want the fields on the master, entity maps will restrict
            io.create_submeshes()
            io.m_id_solid, io.m_id_fluid, io.m_id_phase, io.m_id_ale = 0, 1, 1, 1

            io.mesh_ = [io.msh_emap_solid[0], io.msh_emap_fluid[0]]
            io.mt_d_ = [io.mt_d_solid, io.mt_d_fluid]
            io.mt_b_ = [io.mt_b_solid, io.mt_b_fluid]
            io.mt_sb_ = [io.mt_sb_solid, io.mt_sb_fluid]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            try:
                bcs_lm = boundary_conditions[4]
            except:
                bcs_lm = None

            self.mp = fsi_multiphase_main.FSIMultiphaseProblem(
                pbase,
                io_params,
                time_params[0],
                time_params[1],
                time_params[2],
                fem_params[0],
                fem_params[1],
                fem_params[2],
                fem_params[3],
                constitutive_params[0],
                constitutive_params[1],
                constitutive_params[2],
                constitutive_params[3],
                boundary_conditions[0],
                boundary_conditions[1],
                boundary_conditions[2],
                boundary_conditions[3],
                bcs_lm,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fsi_multiphase_main.FSIMultiphaseSolver(self.mp, solver_params)

        elif problem_type == "solid_constraint":
            from .coupling import solid_constraint_main

            io = ioroutines.IO_solid(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_solid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = solid_constraint_main.SolidmechanicsConstraintProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = solid_constraint_main.SolidmechanicsConstraintSolver(self.mp, solver_params)

        elif problem_type == "fluid_constraint":
            from .coupling import fluid_constraint_main

            io = ioroutines.IO_fluid(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = fluid_constraint_main.FluidmechanicsConstraintProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_constraint_main.FluidmechanicsConstraintSolver(self.mp, solver_params)

        elif problem_type == "fluid_ale_constraint":
            from .coupling import fluid_ale_constraint_main

            io = ioroutines.IO_fluid_ale(io_params, constitutive_params[0:1], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params[0]["quad_degree"], bcdict=boundary_conditions)
            io.set_mesh_fields(io.mesh)
            io.m_id_fluid, io.m_id_ale = 0, 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = fluid_ale_constraint_main.FluidmechanicsAleConstraintProblem(
                pbase,
                io_params,
                time_params,
                fem_params[0],
                fem_params[1],
                constitutive_params[0],
                constitutive_params[1],
                boundary_conditions[0],
                boundary_conditions[1],
                time_curves,
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = fluid_ale_constraint_main.FluidmechanicsAleConstraintSolver(self.mp, solver_params)

        elif problem_type == "phasefield":
            from .phasefield import phasefield_main

            io = ioroutines.IO_phasefield(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_phase = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = phasefield_main.PhasefieldProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = phasefield_main.PhasefieldSolver(self.mp, solver_params)

        elif problem_type == "electrophysiology":
            raise RuntimeError("Electrophysiology not yet fully implemented!")

            from .electrophysiology import electrophysiology_main

            io = ioroutines.IO_solid(io_params, [constitutive_params], self.entity_maps, self.comm)
            io.readin_mesh()
            io.dx, io.bmeasures = io.create_integration_measures(io.mesh, [io.mt_d, io.mt_b, io.mt_sb], fem_params["quad_degree"], bcdict=[boundary_conditions])
            io.set_mesh_fields(io.mesh)
            io.m_id_ep = 0

            io.mesh_ = [io.mesh]
            io.mt_d_ = [io.mt_d]
            io.mt_b_ = [io.mt_b]
            io.mt_sb_ = [io.mt_sb]

            pbase = problem_base(io_params, ctrl_params, comm=self.comm)

            self.mp = electrophysiology_main.ElectrophysiologyProblem(
                pbase,
                io_params,
                time_params,
                fem_params,
                constitutive_params,
                boundary_conditions,
                time_curves,
                io,
                mor_params=mor_params,
            )
            self.mp.set_variational_forms()
            self.ms = electrophysiology_main.ElectrophysiologySolver(self.mp, solver_params)

        elif problem_type == "signet":
            from .signet import signet_main

            pbase = problem_base(io_params, ctrl_params, comm=self.comm, comm_sq=self.comm_sq)

            self.mp = signet_main.SignallingNetworkProblem(
                pbase, io_params, time_params, constitutive_params, time_curves
            )
            self.ms = signet_main.SignallingNetworkSolver(self.mp, solver_params)

        else:
            raise NameError("Unknown problem type!")

    def solve_problem(self):
        """
        Main solve routine
        """

        self.ms.solve_problem()
