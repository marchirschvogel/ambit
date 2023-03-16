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
from projection import project
from mpiroutines import allgather_vec

from solid import SolidmechanicsProblem, SolidmechanicsSolver
from flow0d import Flow0DProblem
from base import solver_base


class SolidmechanicsFlow0DProblem():

    def __init__(self, io_params, time_params_solid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None):
        
        self.problem_physics = 'solid_flow0d'
        
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
        
        try: self.write_checkpoints_periodicref = self.coupling_params['write_checkpoints_periodicref']
        except: self.write_checkpoints_periodicref = False
        
        try: self.restart_periodicref = self.coupling_params['restart_periodicref']
        except: self.restart_periodicref = 0

        try: self.Nmax_periodicref = self.coupling_params['Nmax_periodicref']
        except: self.Nmax_periodicref = 10
        
        # assert that we do not have conflicting timings
        time_params_flow0d['maxtime'] = time_params_solid['maxtime']
        time_params_flow0d['numstep'] = time_params_solid['numstep']
        
        # initialize problem instances (also sets the variational forms for the solid problem)
        self.pbs = SolidmechanicsProblem(io_params, time_params_solid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
        self.pb0 = Flow0DProblem(io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params, comm=self.comm)

        self.incompressible_2field = self.pbs.incompressible_2field

        # for multiscale G&R analysis
        self.t_prev = 0
        self.t_gandr_setpoint = 0
        self.restart_multiscale = False
        
        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        if self.pbs.problem_type == 'solid_flow0d_multiscale_gandr': self.have_multiscale_gandr = True
        else: self.have_multiscale_gandr = False

        self.set_variational_forms_and_jacobians()
        
        self.numdof = self.pbs.numdof + self.pb0.numdof
        # solid is 'master' problem - define problem variables based on its values
        self.simname = self.pbs.simname
        self.restart_step = self.pbs.restart_step
        self.numstep_stop = self.pbs.numstep_stop
        self.dt = self.pbs.dt


    def get_problem_var_list(self):
        
        if self.coupling_type == 'monolithic_lagrange':
            if self.pbs.incompressible_2field:
                return {'field1' : [self.pbs.u, self.pbs.p], 'field2' : [self.lm]}
            else:
                return {'field1' : [self.pbs.u], 'field2' : [self.lm]}

        if self.coupling_type == 'monolithic_direct':
            if self.pbs.incompressible_2field:
                return {'field1' : [self.pbs.u, self.pbs.p], 'field2' : [self.pb0.s]}
            else:
                return {'field1' : [self.pbs.u], 'field2' : [self.pb0.s]}


    def get_problem_functionspace_list(self):
        
        if self.pbs.incompressible_2field:
            return {'field1' : [self.pbs.V_u, self.pbs.V_p], 'field2' : []}
        else:
            return {'field1' : [self.pbs.V_u], 'field2' : []}

        
    # defines the monolithic coupling forms for 0D flow and solid mechanics
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
        
        self.work_coupling, self.work_coupling_old, self.work_coupling_prestr = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        
        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):
            
            self.pr0D = expression.template()
            
            self.coupfuncs.append(fem.Function(self.pbs.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbs.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)
            
            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):
                
                ds_vq = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_vq_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
                
                if self.coupling_params['coupling_quantity'][n] == 'volume':
                    assert(self.coupling_type == 'monolithic_direct')
                    cq_ += self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                    cq_old_ += self.pbs.vf.volume(self.pbs.u_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                elif self.coupling_params['coupling_quantity'][n] == 'flux':
                    assert(self.coupling_type == 'monolithic_direct')
                    cq_ += self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                    cq_old_ += self.pbs.vf.flux(self.pbs.v_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                elif self.coupling_params['coupling_quantity'][n] == 'pressure':
                    assert(self.coupling_type == 'monolithic_lagrange')
                    if self.coupling_params['variable_quantity'][n] == 'volume':
                        cq_ += self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                        cq_old_ += self.pbs.vf.volume(self.pbs.u_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                    elif self.coupling_params['variable_quantity'][n] == 'flux':
                        cq_ += self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                        cq_old_ += self.pbs.vf.flux(self.pbs.v_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                    else:
                        raise NameError("Unknown variable quantity! Choose either volume or flux!")
                else:
                    raise NameError("Unknown coupling quantity! Choose either volume, flux, or pressure!")
            
            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbs.u, self.pbs.du))
            
            df_ = ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):
            
                ds_p = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_p_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
                df_ += self.pbs.timefac*self.pbs.vf.surface(self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_p)
            
                # add to solid rhs contributions
                self.work_coupling += self.pbs.vf.deltaW_ext_neumann_true(self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), self.coupfuncs[-1], ds_p)
                self.work_coupling_old += self.pbs.vf.deltaW_ext_neumann_true(self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), self.coupfuncs_old[-1], ds_p)
                
                # for prestressing, true loads should act on the reference, not the current configuration
                if self.pbs.prestress_initial:
                    self.work_coupling_prestr += self.pbs.vf.deltaW_ext_neumann_refnormal(self.coupfuncs_old[-1], ds_p)
            
            self.dforce.append(df_)
        
        # minus sign, since contribution to external work!
        self.pbs.weakform_u += -self.pbs.timefac * self.work_coupling - (1.-self.pbs.timefac) * self.work_coupling_old
        
        # add to solid Jacobian
        self.pbs.jac_uu += -self.pbs.timefac * ufl.derivative(self.work_coupling, self.pbs.u, self.pbs.du)

        # for naming/access convention in solver... TODO: should go away after solver restructuring!
        self.jac_uu = self.pbs.jac_uu

        if self.coupling_type == 'monolithic_lagrange':
            # old Lagrange multipliers - initialize with initial pressures
            self.pb0.cardvasc0D.initialize_lm(self.lm, self.pb0.initialconditions)
            self.pb0.cardvasc0D.initialize_lm(self.lm_old, self.pb0.initialconditions)


    # for multiscale G&R analysis
    def set_homeostatic_threshold(self, t):
        
        # time is absolute time (should only be set in first cycle)
        eps = 1.0e-14
        if t >= self.t_gandr_setpoint-eps and t < self.t_gandr_setpoint+self.pbs.dt-eps:

            if self.comm.rank == 0:
                print('Set homeostatic growth thresholds...')
                sys.stdout.flush()
            time.sleep(1)
            
            growth_thresolds = []
            for n in range(self.pbs.num_domains):
                
                if self.pbs.mat_growth[n]:
                    
                    growth_settrig = self.pbs.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_settrig']
                    
                    if growth_settrig == 'fibstretch':
                        growth_thresolds.append(self.pbs.ma[n].fibstretch_e(self.pbs.ki.C(self.pbs.u), self.pbs.theta, self.pbs.fib_func[0]))
                    elif growth_settrig == 'volstress':
                        growth_thresolds.append(tr(self.pbs.ma[n].M_e(self.pbs.u, self.pbs.p, self.pbs.ki.C(self.pbs.u), ivar=self.pbs.internalvars)))
                    else:
                        raise NameError("Unknown growth trigger to be set as homeostatic threshold!")
                
                else:
                    
                    growth_thresolds.append(ufl.as_ufl(0))
                
            growth_thres_proj = project(growth_thresolds, self.pbs.Vd_scalar, self.pbs.dx_)
            self.pbs.growth_param_funcs['growth_thres'].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.pbs.growth_param_funcs['growth_thres'].interpolate(growth_thres_proj)


    # for multiscale G&R analysis
    def set_growth_trigger(self, t):

        # time is relative time (w.r.t. heart cycle)
        eps = 1.0e-14
        if t >= self.t_gandr_setpoint-eps and t < self.t_gandr_setpoint+self.pbs.dt-eps:

            if self.comm.rank == 0:
                print('Set growth triggers...')
                sys.stdout.flush()
            time.sleep(1)
            
            self.pbs.u_set.vector.axpby(1.0, 0.0, self.pbs.u.vector)
            self.pbs.u_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            if self.pbs.incompressible_2field:
                self.pbs.p_set.vector.axpby(1.0, 0.0, self.pbs.p.vector)
                self.pbs.p_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.pbs.tau_a_set.vector.axpby(1.0, 0.0, self.pbs.tau_a.vector)
            self.pbs.tau_a_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            if self.pbs.have_frank_starling:
                self.pbs.amp_old_set.vector.axpby(1.0, 0.0, self.pbs.amp_old.vector)
                self.pbs.amp_old_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.pb0.s_set.axpby(1.0, 0.0, self.pb0.s)


    def assemble_residual_stiffness(self):

        return self.pbs.assemble_residual_stiffness()


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):
        
        self.pbs.pre_timestep_routines()
        self.pb0.pre_timestep_routines()


    def read_restart(self, sname, N):

        # solid + flow0d problem
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
    
        if self.have_multiscale_gandr:
            self.set_homeostatic_threshold(t), self.set_growth_trigger(t-t_off)


    def set_output_state(self):

        self.pbs.set_output_state()
        self.pb0.set_output_state()

            
    def write_output(self, N, t, mesh=False): 

        self.pbs.write_output(N, t)
        self.pb0.write_output(N, t)

            
    def update(self):

        # update time step - solid and 0D model
        self.pbs.update()
        self.pb0.update()

        # update old pressures on solid
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



class SolidmechanicsFlow0DSolver(solver_base):

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):
        
        self.pb = problem
        
        self.solver_params_solid = solver_params_solid
        self.solver_params_flow0d = solver_params_flow0d
        
        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_constraint_monolithic(self.pb, self.solver_params_solid, self.solver_params_flow0d)

        if self.pb.pbs.prestress_initial:
            # add coupling work to prestress weak form
            self.pb.pbs.weakform_prestress_u -= self.pb.work_coupling_prestr            
            # initialize solid mechanics solver
            self.solverprestr = SolidmechanicsSolver(self.pb.pbs, self.solver_params_solid)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.ksp.destroy()
        else:
            # set flag definitely to False if we're restarting
            self.pb.pbs.prestress_initial = False
            self.pb.pbs.set_forms_solver()
        
        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0 and not self.pb.restart_multiscale:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old - self.pb.work_coupling_old
            
            jac_a = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbs.a_old)


    def solve_nonlinear_problem(self, t):
        
        self.solnln.newton(t, localdata=self.pb.pbs.localdata)
        

    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.sepstring, self.pb.pbs.numstep, wt=wt)
