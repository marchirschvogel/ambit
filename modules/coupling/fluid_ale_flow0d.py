#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, math
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import utilities
import solver_nonlin
import expression
from mpiroutines import allgather_vec

from fluid_flow0d import FluidmechanicsFlow0DProblem
from ale import AleProblem
from base import solver_base
from meshutils import gather_surface_dof_indices


class FluidmechanicsAleFlow0DProblem():

    def __init__(self, io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models_fluid, constitutive_models_ale, model_params_flow0d, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params_fluid_ale, coupling_params_fluid_flow0d, io, mor_params={}, comm=None):

        self.problem_physics = 'fluid_ale_flow0d'
        
        self.comm = comm
        
        try: self.coupling_type = coupling_params_fluid_ale['fluid_on_deformed']
        except: self.coupling_type = 'consistent'
        
        # initialize problem instances (also sets the variational forms for the fluid flow0d problem)
        self.pba  = AleProblem(io_params, time_params_fluid, fem_params, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params, comm=self.comm)
        self.pbf0 = FluidmechanicsFlow0DProblem(io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models_fluid, model_params_flow0d, bc_dict_fluid, time_curves, coupling_params_fluid_flow0d, io, mor_params=mor_params, comm=self.comm, aleproblem=[self.pba,self.coupling_type])

        self.pbf = self.pbf0.pbf
        self.pb0 = self.pbf0.pb0

        self.io = io
        
        self.fsi_interface = coupling_params_fluid_ale['surface_ids']

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.have_rom = False
        
        self.ufa = fem.Function(self.pba.V_u)

        self.set_variational_forms_and_jacobians()
        
        self.numdof = self.pbf.numdof + self.pb0.numdof + self.pba.numdof
        # fluid is 'master' problem - define problem variables based on its values
        self.simname = self.pbf.simname
        self.restart_step = self.pbf.restart_step
        self.numstep_stop = self.pbf.numstep_stop
        self.dt = self.pbf.dt
        self.have_rom = self.pbf.have_rom
        if self.have_rom: self.rom = self.pbf.rom

        self.sub_solve = True

        
    def get_problem_var_list(self):
        
        is_ghosted = [True, True, True, False]
        return [self.pbf0.pbf.v.vector, self.pbf0.pbf.p.vector, self.pba.u.vector, self.pbf0.lm], is_ghosted
        
        
    # defines the monolithic coupling forms for 0D flow and fluid mechanics
    def set_variational_forms_and_jacobians(self):
    
        dbcs_coup = []
        for i in range(len(self.fsi_interface)):
            dbcs_coup.append( fem.dirichletbc(self.ufa, fem.locate_dofs_topological(self.pba.V_u, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == self.fsi_interface[i]])) )
        
        # pay attention to order... first u=uf, then the others... hence re-set!
        self.pba.bc.dbcs = []
        self.pba.bc.dbcs += dbcs_coup
        # Dirichlet boundary conditions
        if 'dirichlet' in self.pba.bc_dict.keys():
            self.pba.bc.dirichlet_bcs(self.pba.V_u)

        fdi = set(gather_surface_dof_indices(self.pba, self.pba.V_u, self.fsi_interface, self.comm))

        # fluid and ALE actually should have same sizes...
        locmatsize_u = self.pba.V_u.dofmap.index_map.size_local * self.pba.V_u.dofmap.index_map_bs
        matsize_u = self.pba.V_u.dofmap.index_map.size_global * self.pba.V_u.dofmap.index_map_bs

        locmatsize_v = self.pbf.V_v.dofmap.index_map.size_local * self.pbf.V_v.dofmap.index_map_bs
        matsize_v = self.pbf.V_v.dofmap.index_map.size_global * self.pbf.V_v.dofmap.index_map_bs

        ## now we have to assemble the offdiagonal stiffness due to the DBCs u=uf set on the ALE surface - cannot be treated with "derivative" since DBCs are not present in form
        #self.K_uv = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_v,matsize_v)), bsize=None, nnz=None, csr=None, comm=self.comm)
        #self.K_uv.setUp()
        #for i in range(matsize_u):
            #if i in fdi:
                #self.K_uv[i,i] = -1.0/(self.pbf.timefac*self.pbf.dt)
        #self.K_uv.assemble()

        # derivative of fluid momentum w.r.t. ALE displacement
        self.jac_vu = ufl.derivative(self.pbf.weakform_v, self.pba.u, self.pba.du)
        
        # derivative of fluid continuity w.r.t. ALE displacement
        self.jac_pu = ufl.derivative(self.pbf.weakform_p, self.pba.u, self.pba.du)


    def set_forms_solver(self):
        pass


    def get_presolve_state(self):
        return False


    def assemble_residual_stiffness(self, t, subsolver=None):

        # we need a vector representation of ufluid to apply in ALE BCs
        uf_vec = self.pbf.ti.update_uf_ost(self.pbf.v.vector, self.pbf.v_old.vector, self.pbf.uf_old.vector, ufl=False)
        self.ufa.vector.axpby(1.0, 0.0, uf_vec)
        self.ufa.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        r_list_fluidflow0d, K_list_fluidflow0d = self.pbf0.assemble_residual_stiffness(t, subsolver=subsolver)

        r_list_ale, K_list_ale = self.pba.assemble_residual_stiffness(t)
        
        K_list = [[None]*4 for _ in range(4)]
        r_list = [None]*4
        
        K_list[0][0] = K_list_fluidflow0d[0][0]
        K_list[0][1] = K_list_fluidflow0d[0][1]
        
        K_list[0][3] = K_list_fluidflow0d[0][2]

        K_list[1][0] = K_list_fluidflow0d[1][0]
        K_list[1][1] = K_list_fluidflow0d[1][1]
        K_list[1][3] = K_list_fluidflow0d[1][2]
        
        K_list[3][0] = K_list_fluidflow0d[2][0]
        K_list[3][1] = K_list_fluidflow0d[2][1]
        K_list[3][3] = K_list_fluidflow0d[2][2]
        
        # derivative of fluid momentum w.r.t. ALE displacement
        K_vu = fem.petsc.assemble_matrix(fem.form(self.jac_vu), self.pbf.bc.dbcs)
        K_vu.assemble()
        K_list[0][2] = K_vu
        
        # derivative of fluid continuity w.r.t. ALE velocity
        K_pu = fem.petsc.assemble_matrix(fem.form(self.jac_pu), [])
        K_pu.assemble()
        K_list[1][2] = K_pu
        
        # derivative of ALE residual w.r.t. fluid velocities - needed due to DBCs u=uf added on the ALE surfaces
        # TODO: How to form this matrix efficiently?
        #K_list[2][0] = self.K_uv
        
        K_list[2][2] = K_list_ale[0][0]

        # fluid
        r_list[0] = r_list_fluidflow0d[0]
        r_list[1] = r_list_fluidflow0d[1]
        # ALE
        r_list[2] = r_list_ale[0]
        # flow0d
        r_list[3] = r_list_fluidflow0d[2]
        
        return r_list, K_list


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):

        # perform Proper Orthogonal Decomposition
        if self.have_rom:
            self.rom.POD(self, self.pbf.V_v)


    def read_restart(self, sname, N):

        # fluid+flow0d + ALE problem
        self.pbf.read_restart(sname, N)
        self.pba.read_restart(sname, N)


    def evaluate_initial(self):

        self.pbf0.evaluate_initial()


    def write_output_ini(self):

        self.pbf.write_output_ini()


    def get_time_offset(self):

        return 0.


    def evaluate_pre_solve(self, t):

        self.pbf.evaluate_pre_solve(t)
        self.pba.evaluate_pre_solve(t)
            
            
    def evaluate_post_solve(self, t, N):

        self.pbf.evaluate_post_solve(t, N)
        self.pba.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbf.set_output_state()
        self.pba.set_output_state()

            
    def write_output(self, N, t, mesh=False): 

        self.io.write_output(self, N=N, t=t)

            
    def update(self):

        # update time step - fluid+flow0d and ALE
        self.pbf.update()
        self.pba.update()


    def print_to_screen(self):

        self.pbf.print_to_screen()
        self.pba.print_to_screen()
    
    
    def induce_state_change(self):
        
        self.pbf.induce_state_change()
        self.pba.induce_state_change()


    def write_restart(self, sname, N):

        self.pbf.io.write_restart(self.pbf, N)

        
        
    def check_abort(self, t):
        
        self.pbf.check_abort(t)



class FluidmechanicsAleFlow0DSolver(solver_base):

    def __init__(self, problem, solver_params):
    
        self.pb = problem
        
        self.solver_params = solver_params
        
        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):
        
        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if self.pb.pbf.timint != 'static' and self.pb.pbf.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaP_kin_old + self.pb.pbf.deltaP_int_old - self.pb.pbf.deltaP_ext_old
            
            jac_a = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbf.a_old)


    def solve_nonlinear_problem(self, t):
        
        self.solnln.newton(t)
        

    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)
