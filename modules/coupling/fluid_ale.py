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

from fluid import FluidmechanicsProblem
from ale import AleProblem
from base import solver_base
from meshutils import gather_surface_dof_indices


class FluidmechanicsAleProblem():

    def __init__(self, io_params, time_params, fem_params, constitutive_models_fluid, constitutive_models_ale, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params, io, mor_params={}, comm=None):

        self.problem_physics = 'fluid_ale'
        
        self.comm = comm
        
        self.coupling_params = coupling_params
        
        self.fsi_interface = self.coupling_params['surface_ids']

        # initialize problem instances (also sets the variational forms for the fluid problem)
        self.pba = AleProblem(io_params, time_params, fem_params, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params, comm=self.comm)
        self.pbf = FluidmechanicsProblem(io_params, time_params, fem_params, constitutive_models_fluid, bc_dict_fluid, time_curves, io, mor_params=mor_params, comm=self.comm, aleproblem=self.pba)

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.incompressible_2field = self.pbf.incompressible_2field
        self.localsolve = False
        self.have_rom = False

        self.set_variational_forms_and_jacobians()
        
        self.numdof = self.pbf.numdof + self.pba.numdof
        # fluid is 'master' problem - define problem variables based on its values
        self.simname = self.pbf.simname
        self.restart_step = self.pbf.restart_step
        self.numstep_stop = self.pbf.numstep_stop
        self.dt = self.pbf.dt
        self.have_rom = self.pbf.have_rom
        if self.have_rom: self.rom = self.pbf.rom

        self.sub_solve = False

        
    def get_problem_var_list(self):
        
        is_ghosted = [True]*3
        return [self.pbf.v.vector, self.pbf.p.vector, self.pba.w.vector], is_ghosted
        
        
    # defines the monolithic coupling forms for 0D flow and fluid mechanics
    def set_variational_forms_and_jacobians(self):
    
        self.dbcs_coup = fem.dirichletbc(self.pbf.v, fem.locate_dofs_topological(self.pba.V_w, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == self.fsi_interface[0]]))
        #self.dbcs_coup = fem.dirichletbc(self.pbf.uf, fem.locate_dofs_topological(self.pba.V_w, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == self.fsi_interface[0]]))

        fdi = set(gather_surface_dof_indices(self.pba, self.pba.V_w, self.fsi_interface, self.comm))

        # fluid and ALE actually should have same sizes...
        locmatsize_w = self.pba.V_w.dofmap.index_map.size_local * self.pba.V_w.dofmap.index_map_bs
        matsize_w = self.pba.V_w.dofmap.index_map.size_global * self.pba.V_w.dofmap.index_map_bs

        locmatsize_v = self.pbf.V_v.dofmap.index_map.size_local * self.pbf.V_v.dofmap.index_map_bs
        matsize_v = self.pbf.V_v.dofmap.index_map.size_global * self.pbf.V_v.dofmap.index_map_bs

        # now we have to assemble the offdiagonal stiffness due to the DBCs w=v set on the ALE surface - cannot be treated with "derivative" since DBCs are not present in form
        self.K_wv = PETSc.Mat().createAIJ(size=((locmatsize_w,matsize_w),(locmatsize_v,matsize_v)), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_wv.setUp()
        
        for i in range(matsize_w):
            if i in fdi:
                self.K_wv[i,i] = -1.0
                
        self.K_wv.assemble()

        # derivative of fluid momentum w.r.t. ALE displacement
        self.jac_vw = ufl.derivative(self.pbf.weakform_v, self.pba.w, self.pba.dw)
        
        #db_ = ufl.ds(subdomain_data=self.pba.io.mt_b1, subdomain_id=self.fsi_interface[0], metadata={'quadrature_degree': self.pba.quad_degree})
        #wbound = 100.*(ufl.dot((self.pba.w-self.pbf.v), self.pba.var_w)*db_)
        #self.pba.weakform_w += wbound
        #self.pba.jac_ww += ufl.derivative(wbound, self.pba.w, self.pba.dw)
        #self.jac_wv = ufl.derivative(wbound, self.pbf.v, self.pbf.dv)


    def set_forms_solver(self):
        pass


    def get_presolve_state(self):
        return False


    def assemble_residual_stiffness(self, t, subsolver=None):

        r_list_fluid, K_list_fluid = self.pbf.assemble_residual_stiffness(t)
        
        #r_list_ale, K_list_ale = self.pba.assemble_residual_stiffness(t, dbcfluid=self.dbcs_coup)
        r_list_ale, K_list_ale = self.pba.assemble_residual_stiffness(t)
        
        K_list = [[None]*3 for _ in range(3)]
        r_list = [None]*3
        
        K_list[0][0] = K_list_fluid[0][0]
        K_list[0][1] = K_list_fluid[0][1]
        
        # derivate of fluid residual w.r.t. ALE velocity (appears in convective term)
        K_vw = fem.petsc.assemble_matrix(fem.form(self.jac_vw), [])
        K_vw.assemble()
        K_list[0][2] = K_vw

        K_list[1][0] = K_list_fluid[1][0]
        K_list[1][1] = K_list_fluid[1][1]
        
        # derivate of ALE residual w.r.t. fluid velocities - needed due to DBCs w=v added on the ALE surfaces
        #K_wv = fem.petsc.assemble_matrix(fem.form(self.jac_wv), self.pbf.bc.dbcs)
        #K_wv.assemble()
        #K_list[2][0] = K_wv
        
        #K_list[2][0] = self.K_wv
        
        K_list[2][2] = K_list_ale[0][0]

        r_list[0] = r_list_fluid[0]
        r_list[1] = r_list_fluid[1]
        r_list[2] = r_list_ale[0]
        
        return r_list, K_list


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):

        # perform Proper Orthogonal Decomposition
        if self.have_rom:
            self.rom.POD(self, self.pbf.V_v)


    def read_restart(self, sname, N):

        # fluid + ALE problem
        self.pbf.read_restart(sname, N)
        self.pba.read_restart(sname, N)


    def evaluate_initial(self):

        pass


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

        self.pbf.write_output(N, t)
        self.pba.write_output(N, t)

            
    def update(self):

        # update time step - fluid and ALE
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



class FluidmechanicsAleSolver(solver_base):

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
