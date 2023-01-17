#!/usr/bin/env python3

# Copyright (c) 2019-2022, Dr.-Ing. Marc Hirschvogel
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
        
        try: self.restart_periodicref = self.coupling_params['restart_periodicref']
        except: self.restart_periodicref = 0

        try: self.Nmax_periodicref = self.coupling_params['Nmax_periodicref']
        except: self.Nmax_periodicref = 10
        
        # assert that we do not have conflicting timings
        time_params_flow0d['maxtime'] = time_params_solid['maxtime']
        time_params_flow0d['numstep'] = time_params_solid['numstep']
        
        # initialize problem instances (also sets the variational forms for the solid problem)
        self.pbs = SolidmechanicsProblem(io_params, time_params_solid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)
        self.pbf = Flow0DProblem(io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params, comm=self.comm)

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

        if self.coupling_type == 'monolithic_lagrange':
            # old Lagrange multipliers - initialize with initial pressures
            self.pbf.cardvasc0D.initialize_lm(self.lm, self.pbf.time_params['initial_conditions'])
            self.pbf.cardvasc0D.initialize_lm(self.lm_old, self.pbf.time_params['initial_conditions'])


    def induce_perturbation(self):
        
        if self.pbf.perturb_after_cylce > 0: # at least run through one healthy cycle
            
            if self.pbf.ti.cycle[0] > self.pbf.perturb_after_cylce:

                if self.comm.rank == 0:
                    print(">>> Induced cardiovascular disease type: %s" % (self.pbf.perturb_type))
                    sys.stdout.flush()
        
                self.pbf.cardvasc0D.induce_perturbation(self.pbf.perturb_type, self.pbf.perturb_factor)
                if self.pbf.perturb_type=='mi':
                    self.pbs.actstress[self.pbf.perturb_id].sigma0 = self.pbf.perturb_factor # FIXME: ID here is not mat id!!!
                        
                self.pbf.have_induced_pert = True



class SolidmechanicsFlow0DSolver():

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):
        
        self.pb = problem
        
        self.solver_params_solid = solver_params_solid
        self.solver_params_flow0d = solver_params_flow0d

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_constraint_monolithic(self.pb, self.pb.pbs.V_u, self.pb.pbs.V_p, self.solver_params_solid, self.solver_params_flow0d)
        
        
        if self.pb.pbs.prestress_initial:
            # add coupling work to prestress weak form
            self.pb.pbs.weakform_prestress_u -= self.pb.work_coupling_prestr            
            # initialize solid mechanics solver
            self.solverprestr = SolidmechanicsSolver(self.pb.pbs, self.solver_params_solid)


    def solve_problem(self):
        
        start = time.time()
        
        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.comm, self.pb.pbs.ndof)

        if self.pb.pbs.have_rom:
            self.pb.pbs.rom.POD(self.pb.pbs)

        # read restart information
        if self.pb.pbs.restart_step > 0:
            self.pb.pbs.io.readcheckpoint(self.pb.pbs, self.pb.pbs.restart_step)
            self.pb.pbf.readrestart(self.pb.pbs.simname, self.pb.pbs.restart_step)
            self.pb.pbs.simname += '_r'+str(self.pb.pbs.restart_step)
            
        # set pressure functions for old state - s_old already initialized by 0D flow problem
        if self.pb.coupling_type == 'monolithic_direct':
            self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.pbf.s_old, self.pb.pbf.cardvasc0D.v_ids, self.pb.pr0D, self.pb.coupfuncs_old)

        if self.pb.coupling_type == 'monolithic_lagrange':
            self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.lm_old, list(range(self.pb.num_coupling_surf)), self.pb.pr0D, self.pb.coupfuncs_old)

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.ksp.destroy()
        else:
            # set flag definitely to False if we're restarting
            self.pb.pbs.prestress_initial = False

        if self.pb.coupling_type == 'monolithic_direct':
            # old 3D coupling quantities (volumes or fluxes)
            self.pb.pbf.c = []
            for i in range(self.pb.num_coupling_surf):
                cq = fem.assemble_scalar(fem.form(self.pb.cq_old[i]))
                cq = self.pb.comm.allgather(cq)
                self.pb.pbf.c.append(sum(cq)*self.pb.cq_factor[i])

        if self.pb.coupling_type == 'monolithic_lagrange':
            self.pb.pbf.c, self.pb.constr, self.pb.constr_old = [], [], []
            for i in range(self.pb.num_coupling_surf):
                lm_sq, lm_old_sq = allgather_vec(self.pb.lm, self.pb.comm), allgather_vec(self.pb.lm_old, self.pb.comm)
                self.pb.pbf.c.append(lm_sq[i])
                con = fem.assemble_scalar(fem.form(self.pb.cq_old[i]))
                con = self.pb.comm.allgather(con)
                self.pb.constr.append(sum(con)*self.pb.cq_factor[i])
                self.pb.constr_old.append(sum(con)*self.pb.cq_factor[i])

        if bool(self.pb.pbf.chamber_models):
            self.pb.pbf.y = []
            for ch in ['lv','rv','la','ra']:
                if self.pb.pbf.chamber_models[ch]['type']=='0D_elast': self.pb.pbf.y.append(self.pb.pbs.ti.timecurves(self.pb.pbf.chamber_models[ch]['activation_curve'])(self.pb.pbs.t_init))
                if self.pb.pbf.chamber_models[ch]['type']=='0D_elast_prescr': self.pb.pbf.y.append(self.pb.pbs.ti.timecurves(self.pb.pbf.chamber_models[ch]['elastance_curve'])(self.pb.pbs.t_init))
                if self.pb.pbf.chamber_models[ch]['type']=='0D_prescr': self.pb.pbf.c.append(self.pb.pbs.ti.timecurves(self.pb.pbf.chamber_models[ch]['prescribed_curve'])(self.pb.pbs.t_init))

        # initially evaluate 0D model at old state
        self.pb.pbf.cardvasc0D.evaluate(self.pb.pbf.s_old, self.pb.pbs.t_init, self.pb.pbf.df_old, self.pb.pbf.f_old, None, None, self.pb.pbf.c, self.pb.pbf.y, self.pb.pbf.aux_old)
        
        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0 and not self.pb.restart_multiscale:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old - self.pb.work_coupling_old
            
            jac_a = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbs.a_old)

        # write mesh output
        self.pb.pbs.io.write_output(self.pb.pbs, writemesh=True)
        
        # solid 0D flow main time loop
        for N in range(self.pb.pbs.restart_step+1, self.pb.pbs.numstep_stop+1):

            wts = time.time()
            
            # current time
            t = N * self.pb.pbs.dt + self.pb.t_prev # t_prev for multiscale analysis (time from previous cycles)
            
            # offset time for multiple cardiac cycles
            t_off = (self.pb.pbf.ti.cycle[0]-1) * self.pb.pbf.cardvasc0D.T_cycl * self.pb.noperiodicref # zero if T_cycl variable is not specified

            # set time-dependent functions
            self.pb.pbs.ti.set_time_funcs(self.pb.pbs.ti.funcs_to_update, self.pb.pbs.ti.funcs_to_update_vec, t-t_off)
            
            # evaluate rate equations
            self.pb.pbs.evaluate_rate_equations(t, t_off)
            
            # activation curves for 0D chambers (if present)
            self.pb.pbf.evaluate_activation(t-t_off)

            # solve
            self.solnln.newton(self.pb.pbs.u, self.pb.pbs.p, self.pb.pbf.s, t-t_off, localdata=self.pb.pbs.localdata)

            # get midpoint dof values for post-processing (has to be called before update!)
            self.pb.pbf.cardvasc0D.midpoint_avg(self.pb.pbf.s, self.pb.pbf.s_old, self.pb.pbf.s_mid, self.pb.pbf.theta0d_timint(t)), self.pb.pbf.cardvasc0D.midpoint_avg(self.pb.pbf.aux, self.pb.pbf.aux_old, self.pb.pbf.aux_mid, self.pb.pbf.theta0d_timint(t))

            # write output
            self.pb.pbs.io.write_output(self.pb.pbs, N=N, t=t)
            # raw txt file output of 0D model quantities
            if self.pb.pbf.write_results_every_0D > 0 and N % self.pb.pbf.write_results_every_0D == 0:
                self.pb.pbf.cardvasc0D.write_output(self.pb.pbf.output_path_0D, t, self.pb.pbf.s_mid, self.pb.pbf.aux_mid, self.pb.pbs.simname)

            # update time step - solid and 0D model
            self.pb.pbs.ti.update_timestep(self.pb.pbs.u, self.pb.pbs.u_old, self.pb.pbs.v_old, self.pb.pbs.a_old, self.pb.pbs.p, self.pb.pbs.p_old, self.pb.pbs.internalvars, self.pb.pbs.internalvars_old, self.pb.pbs.ratevars, self.pb.pbs.ratevars_old, self.pb.pbs.ti.funcs_to_update, self.pb.pbs.ti.funcs_to_update_old, self.pb.pbs.ti.funcs_to_update_vec, self.pb.pbs.ti.funcs_to_update_vec_old)
            self.pb.pbf.cardvasc0D.update(self.pb.pbf.s, self.pb.pbf.df, self.pb.pbf.f, self.pb.pbf.s_old, self.pb.pbf.df_old, self.pb.pbf.f_old, self.pb.pbf.aux, self.pb.pbf.aux_old)

            if self.pb.have_multiscale_gandr:
                self.set_homeostatic_threshold(t), self.set_growth_trigger(t-t_off)

            # update old pressures on solid
            if self.pb.coupling_type == 'monolithic_direct':
                self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.pbf.s_old, self.pb.pbf.cardvasc0D.v_ids, self.pb.pr0D, self.pb.coupfuncs_old)
            if self.pb.coupling_type == 'monolithic_lagrange':
                self.pb.lm_old.axpby(1.0, 0.0, self.pb.lm)
                self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.lm_old, list(range(self.pb.num_coupling_surf)), self.pb.pr0D, self.pb.coupfuncs_old)
                # update old 3D fluxes
                self.pb.constr_old[:] = self.pb.constr[:]

            # solve time for time step
            wte = time.time()
            wt = wte - wts

            # print to screen
            self.pb.pbf.cardvasc0D.print_to_screen(self.pb.pbf.s_mid,self.pb.pbf.aux_mid)
            # print time step info to screen
            self.pb.pbf.ti.print_timestep(N, t, self.solnln.sepstring, self.pb.pbs.numstep, wt=wt)
            
            # check for periodicity in cardiac cycle and stop if reached (only for syspul* models - cycle counter gets updated here)
            is_periodic = self.pb.pbf.cardvasc0D.cycle_check(self.pb.pbf.s, self.pb.pbf.sTc, self.pb.pbf.sTc_old, t-t_off, self.pb.pbf.ti.cycle, self.pb.pbf.ti.cycleerror, self.pb.pbf.eps_periodic, check=self.pb.pbf.periodic_checktype, inioutpath=self.pb.pbf.output_path_0D, nm=self.pb.pbs.simname, induce_pert_after_cycl=self.pb.pbf.perturb_after_cylce)

            # induce some disease/perturbation for cardiac cycle (i.e. valve stenosis or leakage)
            if self.pb.pbf.perturb_type is not None and not self.pb.pbf.have_induced_pert: self.pb.induce_perturbation()

            # write restart info - old and new quantities are the same at this stage
            self.pb.pbs.io.write_restart(self.pb.pbs, N)
            # write 0D restart info - old and new quantities are the same at this stage (except cycle values sTc)
            if self.pb.pbs.io.write_restart_every > 0 and N % self.pb.pbs.io.write_restart_every == 0:
                self.pb.pbf.writerestart(self.pb.pbs.simname, N)

            if is_periodic and self.pb.noperiodicref==1:
                if self.pb.comm.rank == 0:
                    print("Periodicity reached after %i heart cycles with cycle error %.4f! Finished. :-)" % (self.pb.pbf.ti.cycle[0]-1,self.pb.pbf.ti.cycleerror[0]))
                    sys.stdout.flush()
                break
            
        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()


    # for multiscale G&R analysis
    def set_homeostatic_threshold(self, t):
        
        # time is absolute time (should only be set in first cycle)
        eps = 1.0e-14
        if t >= self.pb.t_gandr_setpoint-eps and t < self.pb.t_gandr_setpoint+self.pb.pbs.dt-eps:

            if self.pb.comm.rank == 0:
                print('Set homeostatic growth thresholds...')
                sys.stdout.flush()
            time.sleep(1)
            
            growth_thresolds = []
            for n in range(self.pb.pbs.num_domains):
                
                if self.pb.pbs.mat_growth[n]:
                    
                    growth_settrig = self.pb.pbs.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_settrig']
                    
                    if growth_settrig == 'fibstretch':
                        growth_thresolds.append(self.pb.pbs.ma[n].fibstretch_e(self.pb.pbs.ki.C(self.pb.pbs.u), self.pb.pbs.theta, self.pb.pbs.fib_func[0]))
                    elif growth_settrig == 'volstress':
                        growth_thresolds.append(tr(self.pb.pbs.ma[n].M_e(self.pb.pbs.u, self.pb.pbs.p, self.pb.pbs.ki.C(self.pb.pbs.u), ivar=self.pb.pbs.internalvars)))
                    else:
                        raise NameError("Unknown growth trigger to be set as homeostatic threshold!")
                
                else:
                    
                    growth_thresolds.append(ufl.as_ufl(0))
                
            growth_thres_proj = project(growth_thresolds, self.pb.pbs.Vd_scalar, self.pb.pbs.dx_)
            self.pb.pbs.growth_param_funcs['growth_thres'].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.pb.pbs.growth_param_funcs['growth_thres'].interpolate(growth_thres_proj)


    # for multiscale G&R analysis
    def set_growth_trigger(self, t):

        # time is relative time (w.r.t. heart cycle)
        eps = 1.0e-14
        if t >= self.pb.t_gandr_setpoint-eps and t < self.pb.t_gandr_setpoint+self.pb.pbs.dt-eps:

            if self.pb.comm.rank == 0:
                print('Set growth triggers...')
                sys.stdout.flush()
            time.sleep(1)
            
            self.pb.pbs.u_set.vector.axpby(1.0, 0.0, self.pb.pbs.u.vector)
            self.pb.pbs.u_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            if self.pb.pbs.incompressible_2field:
                self.pb.pbs.p_set.vector.axpby(1.0, 0.0, self.pb.pbs.p.vector)
                self.pb.pbs.p_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.pb.pbs.tau_a_set.vector.axpby(1.0, 0.0, self.pb.pbs.tau_a.vector)
            self.pb.pbs.tau_a_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            if self.pb.pbs.have_frank_starling:
                self.pb.pbs.amp_old_set.vector.axpby(1.0, 0.0, self.pb.pbs.amp_old.vector)
                self.pb.pbs.amp_old_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.pb.pbf.s_set.axpby(1.0, 0.0, self.pb.pbf.s)
