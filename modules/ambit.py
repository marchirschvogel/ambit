#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mpi4py import MPI
import ioroutines

class Ambit():
    
    def __init__(self, io_params, time_params, solver_params, fem_params={}, constitutive_params={}, bc_dict={}, time_curves=None, coupling_params={}, multiscale_params={}, mor_params={}):
    
        # MPI communicator
        self.comm = MPI.COMM_WORLD
    
        problem_type = io_params['problem_type']
        
        if problem_type == 'solid':
            
            import solid
            
            io = ioroutines.IO_solid(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = solid.SolidmechanicsProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
            self.ms = solid.SolidmechanicsSolver(self.mp, solver_params)
        
        elif problem_type == 'fluid':
            
            import fluid
            
            io = ioroutines.IO_fluid(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = fluid.FluidmechanicsProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
            self.ms = fluid.FluidmechanicsSolver(self.mp, solver_params)

        elif problem_type == 'fluid_ale':
            
            raise RuntimeError("ALE fluid mechanics not yet fully implemented!")
            
            import fluid_ale
            
            io = ioroutines.IO_ale(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = fluid_ale.FluidmechanicsAleProblem(io_params, time_params, fem_params, constitutive_params[0], constitutive_params[1], bc_dict[0], bc_dict[1], time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = fluid_ale.FluidmechanicsAleSolver(self.mp, solver_params)
        
        elif problem_type == 'flow0d':
            
            import flow0d
            
            self.mp = flow0d.Flow0DProblem(io_params, time_params, constitutive_params, time_curves, comm=self.comm)
            self.ms = flow0d.Flow0DSolver(self.mp, solver_params)
            
        elif problem_type == 'solid_flow0d':
            
            import solid_flow0d
            
            io = ioroutines.IO_solid(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = solid_flow0d.SolidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = solid_flow0d.SolidmechanicsFlow0DSolver(self.mp, solver_params[0], solver_params[1])

        elif problem_type == 'solid_flow0d_periodicref':
            
            import solid_flow0d, solid_flow0d_periodicref
            
            io = ioroutines.IO_solid(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = solid_flow0d.SolidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = solid_flow0d_periodicref.SolidmechanicsFlow0DPeriodicRefSolver(self.mp, solver_params[0], solver_params[1])
            
        elif problem_type == 'fluid_flow0d':
            
            import fluid_flow0d
            
            io = ioroutines.IO_fluid(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = fluid_flow0d.FluidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = fluid_flow0d.FluidmechanicsFlow0DSolver(self.mp, solver_params[0], solver_params[1])

        elif problem_type == 'solid_flow0d_multiscale_gandr':
            
            import solid_flow0d_growthremodel
            
            io = ioroutines.IO_solid(io_params, self.comm)
            io.readin_mesh()
            
            self.mp = solid_flow0d_growthremodel.SolidmechanicsFlow0DMultiscaleGrowthRemodelingProblem(io_params, time_params[0], time_params[1], time_params[2], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, multiscale_params, io, comm=self.comm)
            self.ms = solid_flow0d_growthremodel.SolidmechanicsFlow0DMultiscaleGrowthRemodelingSolver(self.mp, solver_params[0], solver_params[1])

        elif problem_type == 'fsi':
            
            raise RuntimeError("Monolithic FSI not yet fully implemented!")
            
            import fsi
            
            ios = ioroutines.IO_solid(io_params['io_solid'], self.comm)
            ios.readin_mesh()

            iof = ioroutines.IO_fluid(io_params['io_fluid'], self.comm)
            iof.readin_mesh()
            
            self.mp = fsi.FSIProblem(io_params, time_params[0], time_params[1], fem_params[0], fem_params[1], constitutive_params[0], constitutive_params[1], bc_dict[0], bc_dict[1], time_curves, coupling_params, ios, iof, mor_params=mor_params, comm=self.comm)
            self.ms = fsi.FSISolver(self.mp, solver_params[0], solver_params[1])

        elif problem_type == 'solid_constraint':
            
            import solid_constraint

            io = ioroutines.IO_solid(io_params, self.comm)
            io.readin_mesh()

            self.mp = solid_constraint.SolidmechanicsConstraintProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, coupling_params, io, mor_params=mor_params, comm=self.comm)
            self.ms = solid_constraint.SolidmechanicsConstraintSolver(self.mp, solver_params[0], solver_params[1])

        elif problem_type == 'signet':
            
            import signet
            
            self.mp = signet.SignallingNetworkProblem(io_params, time_params, constitutive_params, time_curves, comm=self.comm)
            self.ms = signet.SignallingNetworkSolver(self.mp, solver_params)

        else:
            raise NameError("Unknown problem type!")


    def solve_problem(self):
        
        self.ms.solve_problem()

