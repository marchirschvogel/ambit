#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from mpi4py import MPI
from . import ioroutines


class Ambit():
    """
    Ambit main class

    Attributes
    ----------
    comm :
        MPI communicator
    comm_sq :
        Sequential MPI communicator for one core
    entity_maps : dict
        Entity maps for mixed domain problems
    mp :
        Model problem object
    ms :
        Model solver object
    """

    def __init__(self, io_params, time_params, solver_params, fem_params={}, constitutive_params={}, bc_dict={}, time_curves=None, coupling_params={}, multiscale_params={}, mor_params={}):
        """
        Parameters
        ----------
        io_params : dict or list of dicts
            Input/output parameters
        time_params : dict or list of dicts
            Time integration parameters
        solver_params : dict or list of dicts
            Solver parameters for nonlinear and linear solution schemes
        fem_params : dict or list of dicts, optional
            Finite element parameters
        constitutive_params : dict or list of dicts, optional
            Material parameters
        bc_dict : dict, optional
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

        problem_type = io_params['problem_type']

        # entity maps for coupled/multi-mesh problems
        self.entity_maps = {}

        if problem_type == 'solid':

            from .solid import solid_main

            io = ioroutines.IO_solid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = solid_main.SolidmechanicsProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
            self.ms = solid_main.SolidmechanicsSolver(self.mp, solver_params)

        elif problem_type == 'fluid':

            from .fluid import fluid_main

            io = ioroutines.IO_fluid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = fluid_main.FluidmechanicsProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
            self.ms = fluid_main.FluidmechanicsSolver(self.mp, solver_params)

        elif problem_type == 'ale':

            from .ale import ale_main

            io = ioroutines.IO_ale(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = ale_main.AleProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
            self.ms = ale_main.AleSolver(self.mp, solver_params)

        elif problem_type == 'fluid_ale':

            from .coupling import fluid_ale_main

            io = ioroutines.IO_fluid_ale(io_params, fem_params[0], self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = fluid_ale_main.FluidmechanicsAleProblem(io_params, time_params, fem_params[0], fem_params[1], constitutive_params[0], constitutive_params[1], bc_dict[0], bc_dict[1], time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = fluid_ale_main.FluidmechanicsAleSolver(self.mp, solver_params)

        elif problem_type == 'fluid_ale_flow0d':

            from .coupling import fluid_ale_flow0d_main

            io = ioroutines.IO_fluid_ale(io_params, fem_params[0], self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = fluid_ale_flow0d_main.FluidmechanicsAleFlow0DProblem(io_params, time_params[0], time_params[1], fem_params[0], fem_params[1], constitutive_params[0], constitutive_params[1], constitutive_params[2], bc_dict[0], bc_dict[1], time_curves, coupling_params[0], coupling_params[1], io, mor_params=mor_params, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = fluid_ale_flow0d_main.FluidmechanicsAleFlow0DSolver(self.mp, solver_params)

        elif problem_type == 'flow0d':

            from .flow0d import flow0d_main

            self.mp = flow0d_main.Flow0DProblem(io_params, time_params, constitutive_params, time_curves, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = flow0d_main.Flow0DSolver(self.mp, solver_params)

        elif problem_type == 'solid_flow0d':

            from .coupling import solid_flow0d_main

            io = ioroutines.IO_solid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = solid_flow0d_main.SolidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = solid_flow0d_main.SolidmechanicsFlow0DSolver(self.mp, solver_params)

        elif problem_type == 'solid_flow0d_periodicref':

            from .coupling import solid_flow0d_main
            from .coupling import solid_flow0d_periodicref_main

            io = ioroutines.IO_solid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = solid_flow0d_main.SolidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = solid_flow0d_periodicref_main.SolidmechanicsFlow0DPeriodicRefSolver(self.mp, solver_params)

        elif problem_type == 'fluid_flow0d':

            from .coupling import fluid_flow0d_main

            io = ioroutines.IO_fluid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = fluid_flow0d_main.FluidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = fluid_flow0d_main.FluidmechanicsFlow0DSolver(self.mp, solver_params)

        elif problem_type == 'solid_flow0d_multiscale_gandr':

            from .multiscale import solid_flow0d_growthremodel_main

            io = ioroutines.IO_solid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = solid_flow0d_growthremodel_main.SolidmechanicsFlow0DMultiscaleGrowthRemodelingProblem(io_params, time_params[0], time_params[1], time_params[2], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, multiscale_params, io, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = solid_flow0d_growthremodel_main.SolidmechanicsFlow0DMultiscaleGrowthRemodelingSolver(self.mp, solver_params)

        elif problem_type == 'fsi':

            from .coupling import fsi_main

            io = ioroutines.IO_fsi(io_params, fem_params[0], self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_submeshes()
            io.create_integration_measures(io.mesh)

            # solid mesh and integration domain variables
            ios = ioroutines.IO_solid(io_params, fem_params[0], self.entity_maps, self.comm)
            ios.mesh = io.msh_emap_solid[0]

            ios.mesh_master = io.mesh
            ios.mt_d_master, ios.mt_b1_master, ios.mt_b2_master = io.mt_d_master, io.mt_b1_master, None
            ios.mt_d, ios.mt_b1, ios.mt_b2 = io.mt_d_solid, io.mt_b1_solid, None
            ios.dx, ios.ds, ios.dS = io.dx, io.ds, io.dS
            ios.bmeasures = io.bmeasures

            ios.set_mesh_fields(ios.mesh_master) # we want the fields on the master, entity maps will restrict

            # ALE-fluid mesh and integration domain variables
            iof = ioroutines.IO_fluid_ale(io_params, fem_params[0], self.entity_maps, self.comm)
            iof.mesh = io.msh_emap_fluid[0]

            iof.mesh_master = io.mesh
            iof.mt_d_master, iof.mt_b1_master, iof.mt_b2_master = io.mt_d_master, io.mt_b1_master, None
            iof.mt_d, iof.mt_b1, iof.mt_b2 = io.mt_d_fluid, io.mt_b1_fluid, None
            iof.dx, iof.ds, iof.dS = io.dx, io.ds, io.dS
            iof.bmeasures = io.bmeasures

            iof.set_mesh_fields(iof.mesh_master) # we want the fields on the master, entity maps will restrict

            self.mp = fsi_main.FSIProblem(io_params, time_params[0], time_params[1], fem_params[0], fem_params[1], fem_params[2], constitutive_params[0], [constitutive_params[1],constitutive_params[2]], bc_dict[0], [bc_dict[1],bc_dict[2]], bc_dict[3], time_curves, coupling_params, io, ios, iof, mor_params=mor_params, comm=self.comm)
            self.ms = fsi_main.FSISolver(self.mp, solver_params)

        elif problem_type == 'fsi_flow0d':

            from .coupling import fsi_flow0d_main

            io = ioroutines.IO_fsi(io_params, fem_params[0], self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_submeshes()
            io.create_integration_measures(io.mesh)

            # solid mesh and integration domain variables
            ios = ioroutines.IO_solid(io_params, fem_params[0], self.entity_maps, self.comm)
            ios.mesh = io.msh_emap_solid[0]

            ios.mesh_master = io.mesh
            ios.mt_d_master, ios.mt_b1_master, ios.mt_b2_master = io.mt_d_master, io.mt_b1_master, None
            ios.mt_d, ios.mt_b1, ios.mt_b2 = io.mt_d_solid, io.mt_b1_solid, None
            ios.dx, ios.ds, ios.dS = io.dx, io.ds, io.dS
            ios.bmeasures = io.bmeasures

            ios.set_mesh_fields(ios.mesh_master) # we want the fields on the master, entity maps will restrict

            # ALE-fluid mesh and integration domain variables
            iof = ioroutines.IO_fluid_ale(io_params, fem_params[0], self.entity_maps, self.comm)
            iof.mesh = io.msh_emap_fluid[0]

            iof.mesh_master = io.mesh
            iof.mt_d_master, iof.mt_b1_master, iof.mt_b2_master = io.mt_d_master, io.mt_b1_master, None
            iof.mt_d, iof.mt_b1, iof.mt_b2 = io.mt_d_fluid, io.mt_b1_fluid, None
            iof.dx, iof.ds, iof.dS = io.dx, io.ds, io.dS
            iof.bmeasures = io.bmeasures

            iof.set_mesh_fields(iof.mesh_master) # we want the fields on the master, entity maps will restrict

            self.mp = fsi_flow0d_main.FSIFlow0DProblem(io_params, time_params[0], time_params[1], time_params[2], fem_params[0], fem_params[1], fem_params[2], constitutive_params[0], [constitutive_params[1],constitutive_params[2]], constitutive_params[3], bc_dict[0], [bc_dict[1],bc_dict[2]], bc_dict[3], time_curves, coupling_params[0], coupling_params[1], io, ios, iof, mor_params=mor_params, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = fsi_flow0d_main.FSIFlow0DSolver(self.mp, solver_params)

        elif problem_type == 'solid_constraint':

            from .coupling import solid_constraint_main

            io = ioroutines.IO_solid(io_params, fem_params, self.entity_maps, self.comm)
            io.readin_mesh()
            io.create_integration_measures(io.mesh)
            io.set_mesh_fields(io.mesh)

            self.mp = solid_constraint_main.SolidmechanicsConstraintProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = solid_constraint_main.SolidmechanicsConstraintSolver(self.mp, solver_params)

        elif problem_type == 'signet':

            from .signet import signet_main

            self.mp = signet_main.SignallingNetworkProblem(io_params, time_params, constitutive_params, time_curves, comm=self.comm, comm_sq=self.comm_sq)
            self.ms = signet_main.SignallingNetworkSolver(self.mp, solver_params)

        else:
            raise NameError("Unknown problem type!")


    def solve_problem(self):
        """
        Main solve routine
        """

        self.ms.solve_problem()
