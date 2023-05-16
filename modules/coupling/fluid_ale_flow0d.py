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

from fluid_ale import FluidmechanicsAleProblem
from fluid_flow0d import FluidmechanicsFlow0DProblem
from ale import AleProblem
from base import solver_base
from meshutils import gather_surface_dof_indices


class FluidmechanicsAleFlow0DProblem(FluidmechanicsAleProblem):

    def __init__(self, io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models_fluid, constitutive_models_ale, model_params_flow0d, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params_fluid_ale, coupling_params_fluid_flow0d, io, mor_params={}, comm=None):

        self.problem_physics = 'fluid_ale_flow0d'

        self.comm = comm

        try: self.coupling_fluid_ale = coupling_params_fluid_ale['coupling_fluid_ale']
        except: self.coupling_fluid_ale = {}

        try: self.coupling_ale_fluid = coupling_params_fluid_ale['coupling_ale_fluid']
        except: self.coupling_ale_fluid = {}

        try: self.fluid_on_deformed = coupling_params_fluid_ale['fluid_on_deformed']
        except: self.fluid_on_deformed = 'consistent'

        # initialize problem instances (also sets the variational forms for the fluid flow0d problem)
        self.pba  = AleProblem(io_params, time_params_fluid, fem_params, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params, comm=self.comm)
        # ALE variables that are handed to fluid problem
        alevariables = {'Fale' : self.pba.ki.F(self.pba.d), 'Fale_old' : self.pba.ki.F(self.pba.d_old), 'w' : self.pba.wel, 'w_old' : self.pba.w_old, 'fluid_on_deformed' : self.fluid_on_deformed}
        self.pbf0 = FluidmechanicsFlow0DProblem(io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models_fluid, model_params_flow0d, bc_dict_fluid, time_curves, coupling_params_fluid_flow0d, io, mor_params=mor_params, comm=self.comm, alevar=alevariables)

        self.pbf = self.pbf0.pbf
        self.pb0 = self.pbf0.pb0

        # modify results to write...
        self.pbf.results_to_write = io_params['results_to_write'][0]
        self.pba.results_to_write = io_params['results_to_write'][1]

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.have_rom = False

        # NOTE: Fluid and ALE function spaces should be of the same type, but are different objects.
        # For some reason, when applying a function from one funtion space as DBC to another function space,
        # errors occur. Therefore, we define these auxiliary variables and interpolate respectively...

        # fluid displacement, but defined within ALE function space
        self.ufa = fem.Function(self.pba.V_d)
        # ALE velocity, but defined within fluid function space
        self.wf = fem.Function(self.pbf.V_v)

        self.set_variational_forms()

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
        return [self.pbf0.pbf.v.vector, self.pbf0.pbf.p.vector, self.pba.d.vector, self.pbf0.lm], is_ghosted


    def assemble_residual_stiffness(self, t, subsolver=None):

        K_list = [[None]*4 for _ in range(4)]
        r_list = [None]*4

        if bool(self.coupling_fluid_ale):
            if self.coupling_fluid_ale['type'] == 'strong_dirichlet':
                # we need a vector representation of ufluid to apply in ALE DBCs
                uf_vec = self.pbf.ti.update_uf_ost(self.pbf.v.vector, self.pbf.v_old.vector, self.pbf.uf_old.vector, ufl=False)
                self.ufa.vector.axpby(1.0, 0.0, uf_vec)
                self.ufa.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            if self.coupling_fluid_ale['type'] == 'weak_dirichlet':
                K_uv = fem.petsc.assemble_matrix(fem.form(self.jac_uv), self.pba.bc.dbcs)
                K_uv.assemble()
                K_list[2][0] = K_uv

        if bool(self.coupling_ale_fluid):
            if self.coupling_ale_fluid['type'] == 'strong_dirichlet':
                #we need a vector representation of w to apply in fluid DBCs
                w_vec = self.pba.ti.update_w_ost(self.pba.d.vector, self.pba.d_old.vector, self.pba.w_old.vector, ufl=False)
                self.wf.vector.axpby(1.0, 0.0, w_vec)
                self.wf.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        r_list_fluidflow0d, K_list_fluidflow0d = self.pbf0.assemble_residual_stiffness(t, subsolver=subsolver)

        r_list_ale, K_list_ale = self.pba.assemble_residual_stiffness(t)

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
        K_vu = fem.petsc.assemble_matrix(self.jac_vd, self.pbf.bc.dbcs)
        K_vu.assemble()
        K_list[0][2] = K_vu

        # derivative of fluid continuity w.r.t. ALE velocity
        K_pu = fem.petsc.assemble_matrix(self.jac_pd, [])
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


    def get_index_sets(self):

        # block size of velocity field
        bs_v = self.pbf.V_v.dofmap.index_map_bs
        # block size of ALE displacement field
        bs_d = self.pba.V_d.dofmap.index_map_bs

        offset_v = self.pbf.V_v.dofmap.index_map.local_range[0]*bs_v + self.pbf.V_p.dofmap.index_map.local_range[0] + self.pba.V_d.dofmap.index_map.local_range[0]*bs_d + self.pbf0.lm.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(self.pbf.V_v.dofmap.index_map.size_local*bs_v, first=offset_v, step=1, comm=self.comm)

        offset_p = offset_v + self.pbf.V_v.dofmap.index_map.size_local*bs_v
        iset_p = PETSc.IS().createStride(self.pbf.V_p.dofmap.index_map.size_local, first=offset_p, step=1, comm=self.comm)

        offset_d = offset_p + self.pbf.V_p.dofmap.index_map.size_local
        iset_d = PETSc.IS().createStride(self.pba.V_d.dofmap.index_map.size_local*bs_d, first=offset_d, step=1, comm=self.comm)

        offset_s = offset_d + self.pba.V_d.dofmap.index_map.size_local*bs_d
        iset_s = PETSc.IS().createStride(self.pbf0.lm.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        return [iset_v, iset_p, iset_d, iset_s]


    ### now the base routines for this problem

    def pre_timestep_routines(self):

        # perform Proper Orthogonal Decomposition
        if self.have_rom:
            self.rom.prepare_rob()


    def read_restart(self, sname, N):

        # fluid+flow0d + ALE problem
        self.pbf0.read_restart(sname, N)
        self.pba.read_restart(sname, N)


    def evaluate_initial(self):

        self.pbf0.evaluate_initial()
        self.pba.evaluate_initial()


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):

        return 0.


    def evaluate_pre_solve(self, t):

        self.pbf0.evaluate_pre_solve(t)
        self.pba.evaluate_pre_solve(t)


    def evaluate_post_solve(self, t, N):

        self.pbf0.evaluate_post_solve(t, N)
        self.pba.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbf0.set_output_state()
        self.pba.set_output_state()


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t) # combined fluid-ALE output routine
        self.pb0.write_output(N, t)


    def update(self):

        # update time step - fluid+flow0d and ALE
        self.pbf0.update()
        self.pba.update()


    def print_to_screen(self):

        self.pbf0.print_to_screen()
        self.pba.print_to_screen()


    def induce_state_change(self):

        self.pbf0.induce_state_change()
        self.pba.induce_state_change()


    def write_restart(self, sname, N):

        self.pbf0.write_restart(sname, N)
        self.pba.write_restart(sname, N)


    def check_abort(self, t):

        self.pbf0.check_abort(t)
        self.pba.check_abort(t)



class FluidmechanicsAleFlow0DSolver(solver_base):

    def __init__(self, problem, solver_params):

        self.pb = problem

        self.solver_params = solver_params

        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbf.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            jac_a = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbf.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)
