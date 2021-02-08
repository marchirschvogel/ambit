#!/usr/bin/env python3

from mpi4py import MPI

class Ambit():
    
    def __init__(self, io_params, time_params, solver_params, fem_params={}, constitutive_params={}, bc_dict={}, time_curves=None, coupling_params={}, multiscale_params={}):
    
        # MPI communicator
        self.comm = MPI.COMM_WORLD
    
        problem_type = io_params['problem_type']
        
        if problem_type == 'solid':
            import solid
            self.mp = solid.SolidmechanicsProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, comm=self.comm)
            self.ms = solid.SolidmechanicsSolver(self.mp, solver_params)
        
        elif problem_type == 'fluid':
            import fluid
            self.mp = fluid.FluidmechanicsProblem(io_params, time_params, fem_params, constitutive_params, bc_dict, time_curves, comm=self.comm)
            self.ms = fluid.FluidmechanicsSolver(self.mp, solver_params)
        
        elif problem_type == 'flow0d':
            import flow0d
            self.mp = flow0d.Flow0DProblem(io_params, time_params, constitutive_params, time_curves, comm=self.comm)
            self.ms = flow0d.Flow0DSolver(self.mp, solver_params)
            
        elif problem_type == 'solid_flow0d':
            import solid_flow0d
            self.mp = solid_flow0d.SolidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, comm=self.comm)
            self.ms = solid_flow0d.SolidmechanicsFlow0DSolver(self.mp, solver_params[0], solver_params[1])
            
        elif problem_type == 'fluid_flow0d':
            import fluid_flow0d
            self.mp = fluid_flow0d.FluidmechanicsFlow0DProblem(io_params, time_params[0], time_params[1], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, comm=self.comm)
            self.ms = fluid_flow0d.FluidmechanicsFlow0DSolver(self.mp, solver_params[0], solver_params[1])

        elif problem_type == 'solid_flow0d_multiscale_gandr':
            import solid_flow0d_growthremodel
            self.mp = solid_flow0d_growthremodel.SolidmechanicsFlow0DMultiscaleGrowthRemodelingProblem(io_params, time_params[0], time_params[1], time_params[2], fem_params, constitutive_params[0], constitutive_params[1], bc_dict, time_curves, coupling_params, multiscale_params, comm=self.comm)
            self.ms = solid_flow0d_growthremodel.SolidmechanicsFlow0DMultiscaleGrowthRemodelingSolver(self.mp, solver_params[0], solver_params[1])

        else:
            raise NameError("Unknown problem type!")


    def solve_problem(self):
        
        self.ms.solve_problem()

