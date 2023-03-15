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
from flow0d import Flow0DProblem
from base import solver_base


class FluidmechanicsFlow0DProblem():

    def __init__(self, io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None):
        
        self.problem_physics = 'fluid_flow0d'
        
        self.comm = comm
        
        self.coupling_params = coupling_params
        
        self.surface_vq_ids = self.coupling_params['surface_ids']
        try: self.surface_p_ids = self.coupling_params['surface_p_ids']
        except: self.surface_p_ids = self.surface_vq_ids

        self.num_coupling_surf = len(self.surface_vq_ids)
        
        try: self.cq_factor = self.coupling_params['cq_factor']
        except: self.cq_factor = [1.]*self.num_coupling_surf

        try: self.coupling_type = self.coupling_params['coupling_type']
        except: self.coupling_type = 'monolithic_direct'

        try: self.eps_fd = self.coupling_params['eps_fd']
        except: self.eps_fd = 1.0e-5

        try: self.print_subiter = self.coupling_params['print_subiter']
        except: self.print_subiter = False

        try: self.restart_periodicref = self.coupling_params['restart_periodicref']
        except: self.restart_periodicref = 0

        try: self.Nmax_periodicref = self.coupling_params['Nmax_periodicref']
        except: self.Nmax_periodicref = 10

        # assert that we do not have conflicting timings
        time_params_flow0d['maxtime'] = time_params_fluid['maxtime']
        time_params_flow0d['numstep'] = time_params_fluid['numstep']

        # initialize problem instances (also sets the variational forms for the fluid problem)
        self.pbs = FluidmechanicsProblem(io_params, time_params_fluid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
        self.pb0 = Flow0DProblem(io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params, comm=self.comm)

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.incompressible_2field = self.pbs.incompressible_2field

        self.set_variational_forms_and_jacobians()
        
        self.numdof = self.pbs.numdof + self.pb0.numdof
        # fluid is 'master' problem - define problem variables based on its values
        self.simname = self.pbs.simname
        self.restart_step = self.pbs.restart_step
        self.numstep_stop = self.pbs.numstep_stop
        self.dt = self.pbs.dt

        
    # defines the monolithic coupling forms for 0D flow and fluid mechanics
    def set_variational_forms_and_jacobians(self):

        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        self.coupfuncs, self.coupfuncs_old = [], []

        if self.coupling_type == 'monolithic_lagrange':
            
            # Lagrange multiplier stiffness matrix (currently treated with FD!)
            self.K_lm = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
            self.K_lm.setUp()

            # Lagrange multipliers
            self.lm, self.lm_old = self.K_lm.createVecLeft(), self.K_lm.createVecLeft()
            
            # 3D fluxes
            self.constr, self.constr_old = [], []

        self.power_coupling, self.power_coupling_old = ufl.as_ufl(0), ufl.as_ufl(0)
    
        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):
            
            self.pr0D = expression.template()
            
            self.coupfuncs.append(fem.Function(self.pbs.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbs.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)

            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):

                ds_vq = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_vq_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
          
                if self.coupling_params['coupling_quantity'][n] == 'flux':
                    assert(self.coupling_type == 'monolithic_direct')
                    cq_ += self.pbs.vf.flux(self.pbs.v, ds_vq)
                    cq_old_ += self.pbs.vf.flux(self.pbs.v_old, ds_vq)
                elif self.coupling_params['coupling_quantity'][n] == 'pressure':
                    assert(self.coupling_type == 'monolithic_lagrange' and self.coupling_params['variable_quantity'][n] == 'flux')
                    cq_ += self.pbs.vf.flux(self.pbs.v, ds_vq)
                    cq_old_ += self.pbs.vf.flux(self.pbs.v_old, ds_vq)
                else:
                    raise NameError("Unknown coupling quantity! Choose flux or pressure!")
            
            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbs.v, self.pbs.dv))

            df_ = ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):
                
                ds_p = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_p_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
                df_ += self.pbs.timefac*self.pbs.vf.surface(ds_p)
            
                # add to fluid rhs contributions
                self.power_coupling += self.pbs.vf.deltaP_ext_neumann_normal(self.coupfuncs[-1], ds_p)
                self.power_coupling_old += self.pbs.vf.deltaP_ext_neumann_normal(self.coupfuncs_old[-1], ds_p)
        
            self.dforce.append(df_)
        
        # minus sign, since contribution to external power!
        self.pbs.weakform_v += -self.pbs.timefac * self.power_coupling - (1.-self.pbs.timefac) * self.power_coupling_old
        
        # add to fluid Jacobian
        self.pbs.jac_vv += -self.pbs.timefac * ufl.derivative(self.power_coupling, self.pbs.v, self.pbs.dv)
        
        # for naming/access convention in solver... TODO: should go away after solver restructuring!
        self.jac_uu = self.pbs.jac_vv

        if self.coupling_type == 'monolithic_lagrange':
            # old Lagrange multipliers - initialize with initial pressures
            self.pb0.cardvasc0D.initialize_lm(self.lm, self.pb0.initialconditions)
            self.pb0.cardvasc0D.initialize_lm(self.lm_old, self.pb0.initialconditions)


    def assemble_residual_stiffness_main(self):

        return self.pbs.assemble_residual_stiffness_main()


    def assemble_residual_stiffness_incompressible(self):
        
        return self.pbs.assemble_residual_stiffness_incompressible()


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):

        self.pbs.pre_timestep_routines()
        self.pb0.pre_timestep_routines()


    def read_restart(self, sname, N):

        # fluid + flow0d problem
        self.pbs.read_restart(sname, N)
        self.pb0.read_restart(sname, N)

        if self.pbs.restart_step > 0:
            if self.coupling_type == 'monolithic_lagrange':
                self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm)
                self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm_old)


    def evaluate_initial(self):

        # set pressure functions for old state - s_old already initialized by 0D flow problem
        if self.coupling_type == 'monolithic_direct':
            self.pb0.cardvasc0D.set_pressure_fem(self.pb0.s_old, self.pb0.cardvasc0D.v_ids, self.pr0D, self.coupfuncs_old)
        
        if self.coupling_type == 'monolithic_lagrange':
            self.pb0.cardvasc0D.set_pressure_fem(self.lm_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)

        if self.coupling_type == 'monolithic_direct':
            # old 3D coupling quantities (volumes or fluxes)
            self.pb0.c = []
            for i in range(self.num_coupling_surf):
                cq = fem.assemble_scalar(fem.form(self.cq_old[i]))
                cq = self.comm.allgather(cq)
                self.pb0.c.append(sum(cq)*self.cq_factor[i])
        
        if self.coupling_type == 'monolithic_lagrange':
            self.pb0.c, self.constr, self.constr_old = [], [], []
            for i in range(self.num_coupling_surf):
                lm_sq, lm_old_sq = allgather_vec(self.lm, self.comm), allgather_vec(self.lm_old, self.comm)
                self.pb0.c.append(lm_sq[i])
                con = fem.assemble_scalar(fem.form(self.cq_old[i]))
                con = self.comm.allgather(con)
                self.constr.append(sum(con)*self.cq_factor[i])
                self.constr_old.append(sum(con)*self.cq_factor[i])

        if bool(self.pb0.chamber_models):
            self.pb0.y = []
            for ch in ['lv','rv','la','ra']:
                if self.pb0.chamber_models[ch]['type']=='0D_elast': self.pb0.y.append(self.pbs.ti.timecurves(self.pb0.chamber_models[ch]['activation_curve'])(self.pbs.t_init))
                if self.pb0.chamber_models[ch]['type']=='0D_elast_prescr': self.pb0.y.append(self.pbs.ti.timecurves(self.pb0.chamber_models[ch]['elastance_curve'])(self.pbs.t_init))
                if self.pb0.chamber_models[ch]['type']=='0D_prescr': self.pb0.c.append(self.pbs.ti.timecurves(self.pb0.chamber_models[ch]['prescribed_curve'])(self.pbs.t_init))

        # initially evaluate 0D model at old state
        self.pb0.cardvasc0D.evaluate(self.pb0.s_old, self.pbs.t_init, self.pb0.df_old, self.pb0.f_old, None, None, self.pb0.c, self.pb0.y, self.pb0.aux_old)
        self.pb0.auxTc_old[:] = self.pb0.aux_old[:]


    def write_output_ini(self):

        self.pbs.write_output_ini()


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t):

        self.pbs.evaluate_pre_solve(t)
        self.pb0.evaluate_pre_solve(t)
            
            
    def evaluate_post_solve(self, t, N):

        self.pbs.evaluate_post_solve(t, N)
        self.pb0.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbs.set_output_state()
        self.pb0.set_output_state()

            
    def write_output(self, N, t, mesh=False): 

        self.pbs.write_output(N, t)
        self.pb0.write_output(N, t)

            
    def update(self):

        # update time step - fluid and 0D model
        self.pbs.update()
        self.pb0.update()
        
        # update old pressures on fluid
        if self.coupling_type == 'monolithic_direct':
            self.pb0.cardvasc0D.set_pressure_fem(self.pb0.s_old, self.pb0.cardvasc0D.v_ids, self.pr0D, self.coupfuncs_old)
        if self.coupling_type == 'monolithic_lagrange':
            self.lm_old.axpby(1.0, 0.0, self.lm)
            self.pb0.cardvasc0D.set_pressure_fem(self.lm_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)
            # update old 3D fluxes
            self.constr_old[:] = self.constr[:]


    def print_to_screen(self):

        self.pbs.print_to_screen()
        self.pb0.print_to_screen()
    
    
    def induce_state_change(self):
        
        self.pbs.induce_state_change()
        self.pb0.induce_state_change()


    def write_restart(self, sname, N):

        self.pbs.io.write_restart(self.pbs, N)
        
        if self.pbs.io.write_restart_every > 0 and N % self.pbs.io.write_restart_every == 0:
            self.pb0.writerestart(sname, N)
            if self.coupling_type == 'monolithic_lagrange':
                self.pb0.cardvasc0D.write_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm)
        
        
    def check_abort(self, t):
        
        self.pb0.check_abort(t)



class FluidmechanicsFlow0DSolver(solver_base):

    def __init__(self, problem, solver_params_fluid, solver_params_flow0d):
    
        self.pb = problem
        
        self.solver_params_fluid = solver_params_fluid
        self.solver_params_flow0d = solver_params_flow0d
        
        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):
        
        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_constraint_monolithic(self.pb, self.pb.pbs.V_v, self.pb.pbs.V_p, self.solver_params_fluid, self.solver_params_flow0d)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaP_kin_old + self.pb.pbs.deltaP_int_old - self.pb.pbs.deltaP_ext_old - self.pb.power_coupling_old
            
            jac_a = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbs.a_old)


    def solve_nonlinear_problem(self, t):
        
        self.solnln.newton(self.pb.pbs.v, self.pb.pbs.p, self.pb.pb0.s, t)
        

    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.sepstring, self.pb.pbs.numstep, wt=wt)
