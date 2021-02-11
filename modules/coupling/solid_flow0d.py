#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np
from dolfinx import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, Function, DirichletBC
from dolfinx.fem import assemble_scalar
from ufl import TrialFunction, TestFunction, FiniteElement, derivative, diff, dx, ds, tr, as_ufl
from petsc4py import PETSc

import utilities
import solver_nonlin
import expression
from projection import project
from mpiroutines import allgather_vec

from solid import SolidmechanicsProblem
from flow0d import Flow0DProblem


class SolidmechanicsFlow0DProblem():

    def __init__(self, io_params, time_params_solid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, comm=None):
        
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
        
        # for multiscale G&R analysis
        self.t_prev = 0
        self.t_gandr_setpoint = 0
        self.have_set_homeostatic = False

        # initialize problem instances (also sets the variational forms for the solid problem)
        self.pbs = SolidmechanicsProblem(io_params, time_params_solid, fem_params, constitutive_models, bc_dict, time_curves, comm=self.comm)
        self.pbf = Flow0DProblem(io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params, comm=self.comm)

        self.set_variational_forms_and_jacobians()

        
    # defines the monolithic coupling forms for 0D flow and solid mechanics
    def set_variational_forms_and_jacobians(self):

        self.cq, self.dcq, self.dforce = [], [], []
        self.coupfuncs, self.coupfuncs_old = [], []
        
        if self.coupling_type == 'monolithic_lagrange':
            
            # Lagrange multiplier stiffness matrix (currently treated with FD!)
            self.K_lm = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
            self.K_lm.setUp()

            # Lagrange multipliers
            self.lm, self.lm_old = self.K_lm.createVecLeft(), self.K_lm.createVecLeft()
            
            # 3D fluxes
            self.flux3D, self.flux3D_old = [], []
        
        self.work_coupling, self.work_coupling_old, self.work_coupling_prestr = as_ufl(0), as_ufl(0), as_ufl(0)
        
        # coupling variational forms and Jacobian contributions
        for i in range(self.num_coupling_surf):
            
            self.pr0D = expression.template()
            
            self.coupfuncs.append(Function(self.pbs.Vd_scalar)), self.coupfuncs_old.append(Function(self.pbs.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)
            
            ds_vq = ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_vq_ids[i], metadata={'quadrature_degree': self.pbs.quad_degree})
            ds_p = ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_p_ids[i], metadata={'quadrature_degree': self.pbs.quad_degree})
       
            if self.coupling_params['coupling_quantity'] == 'volume':
                assert(self.coupling_type == 'monolithic_direct')
                self.cq.append(self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_vq))
            elif self.coupling_params['coupling_quantity'] == 'flux':
                assert(self.coupling_type == 'monolithic_direct')
                self.cq.append(self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_vq))
            elif self.coupling_params['coupling_quantity'] == 'pressure':
                assert(self.coupling_type == 'monolithic_lagrange')
                self.cq.append(self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_vq))
            else:
                raise NameError("Unknown coupling quantity! Choose either volume, flux, or pressure!")
            
            self.dcq.append(derivative(self.cq[-1], self.pbs.u, self.pbs.du))
            self.dforce.append(self.pbs.timefac*self.pbs.vf.surface(self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_p))

            # add to solid rhs contributions
            self.work_coupling += self.pbs.vf.deltaW_ext_neumann_true(self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), self.coupfuncs[-1], ds_p)
            self.work_coupling_old += self.pbs.vf.deltaW_ext_neumann_true(self.pbs.ki.J(self.pbs.u_old), self.pbs.ki.F(self.pbs.u_old), self.coupfuncs_old[-1], ds_p)
            
            # for prestressing, true loads should act on the reference, not the current configuration
            if self.pbs.prestress_initial:
                self.work_coupling_prestr += self.pbs.vf.deltaW_ext_neumann_refnormal(self.coupfuncs_old[-1], ds_p)
        
        # minus sign, since contribution to external work!
        self.pbs.weakform_u += -self.pbs.timefac * self.work_coupling - (1.-self.pbs.timefac) * self.work_coupling_old
        
        # add to solid Jacobian
        self.pbs.jac_uu += -self.pbs.timefac * derivative(self.work_coupling, self.pbs.u, self.pbs.du)

        if self.coupling_type == 'monolithic_lagrange':
            # old Lagrange multipliers - initialize with initial pressures
            self.pbf.cardvasc0D.initialize_lm(self.lm, self.pbf.time_params['initial_conditions'])
            self.pbf.cardvasc0D.initialize_lm(self.lm_old, self.pbf.time_params['initial_conditions'])


class SolidmechanicsFlow0DSolver():

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):
    
        self.pb = problem
        
        self.solver_params_solid = solver_params_solid
        self.solver_params_flow0d = solver_params_flow0d

        self.solve_type = self.solver_params_solid['solve_type']
        

    def solve_problem(self):
        
        start = time.time()
        
        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.pbs.comm, self.pb.pbs.ndof)

        # set pressure functions for old state - s_old already initialized by 0D flow problem
        if self.pb.coupling_type == 'monolithic_direct':
            self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.pbf.s_old, self.pb.pbf.cardvasc0D.v_ids, self.pb.pr0D, self.pb.coupfuncs_old)

        if self.pb.coupling_type == 'monolithic_lagrange':
            self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.lm_old, self.pb.pbf.cardvasc0D.c_ids, self.pb.pr0D, self.pb.coupfuncs_old)

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.prestress_initial:
            
            utilities.print_prestress('start', self.pb.comm)

            # quasi-static weak forms (don't dare to use fancy growth laws or other inelastic stuff during prestressing...)
            self.pb.pbs.weakform_prestress_u = self.pb.pbs.deltaW_int - self.pb.pbs.deltaW_prestr_ext - self.pb.work_coupling_prestr
            self.pb.pbs.jac_prestress_uu = derivative(self.pb.pbs.weakform_prestress_u, self.pb.pbs.u, self.pb.pbs.du)
            if self.pb.pbs.incompressible_2field:
                self.pb.pbs.weakform_prestress_p = self.pb.pbs.deltaW_p
                self.pb.pbs.jac_prestress_up = derivative(self.pb.pbs.weakform_prestress_u, self.pb.pbs.p, self.pb.pbs.dp)
                self.pb.pbs.jac_prestress_pu = derivative(self.pb.pbs.weakform_prestress_p, self.pb.pbs.u, self.pb.pbs.du)

            solnln_prestress = solver_nonlin.solver_nonlinear(self.pb.pbs, self.pb.pbs.V_u, self.pb.pbs.V_p, self.solver_params_solid)
            # pure solid problem during prestress
            solnln_prestress.ptype = 'solid'

            # solve in 1 load step using PTC!
            solnln_prestress.PTC = True
            solnln_prestress.k_PTC_initial = 0.1

            solnln_prestress.newton(self.pb.pbs.u, self.pb.pbs.p)
            

            # MULF update
            self.pb.pbs.ki.prestress_update(self.pb.pbs.u, self.pb.pbs.Vd_tensor, self.pb.pbs.dx_, self.pb.pbs.u_pre)

            # set flag to false again
            self.pb.pbs.prestress_initial = False

            utilities.print_prestress('end', self.pb.comm)
            # delete class instance
            del solnln_prestress

        if self.pb.coupling_type == 'monolithic_direct':
            # old 3D coupling quantities (volumes or fluxes)
            self.pb.pbf.c = []
            for i in range(self.pb.num_coupling_surf):
                cq = assemble_scalar(self.pb.cq[i])
                cq = self.pb.pbs.comm.allgather(cq)
                self.pb.pbf.c.append(sum(cq)*self.pb.cq_factor[i])

        if self.pb.coupling_type == 'monolithic_lagrange':
            self.pb.pbf.c, self.pb.flux3D, self.pb.flux3D_old = [], [], []
            for i in range(self.pb.num_coupling_surf):
                lm_sq, lm_old_sq = allgather_vec(self.pb.lm, self.pb.comm), allgather_vec(self.pb.lm_old, self.pb.comm)
                self.pb.pbf.c.append(lm_sq[i])
                fl = assemble_scalar(self.pb.cq[i])
                fl = self.pb.pbs.comm.allgather(fl)
                self.pb.flux3D.append(sum(fl)*self.pb.cq_factor[i])
                self.pb.flux3D_old.append(sum(fl)*self.pb.cq_factor[i])

        # initially evaluate 0D model at old state
        self.pb.pbf.cardvasc0D.evaluate(self.pb.pbf.s_old, 0., 0., self.pb.pbf.df_old, self.pb.pbf.f_old, None, self.pb.pbf.c, self.pb.pbf.aux_old)
        
        # initialize nonlinear solver class
        solnln = solver_nonlin.solver_nonlinear_3D0Dmonolithic(self.pb, self.pb.pbs.V_u, self.pb.pbs.V_p, self.solver_params_solid, self.solver_params_flow0d)

        # solve for consistent initial acceleration
        if self.pb.pbs.timint != 'static':
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old - self.pb.work_coupling_old

            jac_a = derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old and return forms for acc and vel
            solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbs.a_old)

        # write mesh output
        self.pb.pbs.io.write_output(writemesh=True)

        # load/time stepping
        interval = np.linspace(0, self.pb.pbs.maxtime, self.pb.pbs.numstep+1)


        # solid 0D flow main time loop
        for (N, dt) in enumerate(np.diff(interval)):
            
            wts = time.time()

            t = interval[N+1] + self.pb.t_prev # t_prev for multiscale analysis (time from previous cycles)
            
            # offset time for multiple cardiac cycles
            t_off = (self.pb.pbf.ti.cycle[0]-1) * self.pb.pbf.T_cycl # zero if T_cycl variable is not specified

            # set time-dependent functions
            self.pb.pbs.ti.set_time_funcs(self.pb.pbs.ti.funcs_to_update, self.pb.pbs.ti.funcs_to_update_vec, t-t_off)

            if self.pb.pbs.problem_type == 'solid_flow0d_multiscale_gandr':
                self.set_homeostatic_threshold(t-t_off, dt), self.set_growth_trigger(t-t_off, dt)

            # take care of active stress
            if self.pb.pbs.have_active_stress and self.pb.pbs.active_stress_trig == 'ode':
                self.pb.pbs.evaluate_active_stress_ode(t-t_off, dt)

            # solve
            solnln.newton(self.pb.pbs.u, self.pb.pbs.p, self.pb.pbf.s, t-t_off, dt, locvar=self.pb.pbs.theta, locresform=self.pb.pbs.r_growth, locincrform=self.pb.pbs.del_theta)

            # get midpoint dof values for post-processing (has to be called before update!)
            self.pb.pbf.cardvasc0D.midpoint_avg(self.pb.pbf.s, self.pb.pbf.s_old, self.pb.pbf.s_mid), self.pb.pbf.cardvasc0D.midpoint_avg(self.pb.pbf.aux, self.pb.pbf.aux_old, self.pb.pbf.aux_mid)

            # update time step - solid and 0D model
            self.pb.pbs.ti.update_timestep(self.pb.pbs.u, self.pb.pbs.u_old, self.pb.pbs.v_old, self.pb.pbs.a_old, self.pb.pbs.p, self.pb.pbs.p_old, self.pb.pbs.internalvars, self.pb.pbs.internalvars_old, self.pb.pbs.ti.funcs_to_update, self.pb.pbs.ti.funcs_to_update_old, self.pb.pbs.ti.funcs_to_update_vec, self.pb.pbs.ti.funcs_to_update_vec_old)
            self.pb.pbf.cardvasc0D.update(self.pb.pbf.s, self.pb.pbf.df, self.pb.pbf.f, self.pb.pbf.s_old, self.pb.pbf.df_old, self.pb.pbf.f_old, self.pb.pbf.aux, self.pb.pbf.aux_old)
            
            # update old pressures on solid
            if self.pb.coupling_type == 'monolithic_direct':
                self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.pbf.s_old, self.pb.pbf.cardvasc0D.v_ids, self.pb.pr0D, self.pb.coupfuncs_old)
            if self.pb.coupling_type == 'monolithic_lagrange':
                self.pb.lm.assemble(), self.pb.lm_old.axpby(1.0, 0.0, self.pb.lm)
                self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.lm_old, self.pb.pbf.cardvasc0D.c_ids, self.pb.pr0D, self.pb.coupfuncs_old)
                # update old 3D fluxes
                for i in range(self.pb.num_coupling_surf):
                    self.pb.flux3D_old[i] = self.pb.flux3D[i]

            # solve time for time step
            wte = time.time()
            wt = wte - wts
            
            # write output
            self.pb.pbs.io.write_output(pb=self.pb.pbs, N=N, t=t)
            # raw txt file output of 0D model quantities
            if (N+1) % self.pb.pbf.write_results_every_0D == 0:
                self.pb.pbf.cardvasc0D.write_output(self.pb.pbf.output_path_0D, t, self.pb.pbf.s_mid, self.pb.pbf.aux_mid)

            # print to screen
            self.pb.pbf.cardvasc0D.print_to_screen(self.pb.pbf.s_mid,self.pb.pbf.aux_mid)
            # print time step info to screen
            self.pb.pbf.ti.print_timestep(N, t, self.pb.pbs.numstep, wt=wt)
            
            # check for periodicity in cardiac cycle and stop if reached (only for syspul* models - cycle counter gets updated here)
            is_periodic = self.pb.pbf.cardvasc0D.cycle_check(self.pb.pbf.s, self.pb.pbf.sTc, self.pb.pbf.sTc_old, t, self.pb.pbf.ti.cycle, self.pb.pbf.ti.cycleerror, self.pb.pbf.eps_periodic, check=self.pb.pbf.periodic_checktype, inioutpath=self.pb.pbf.output_path_0D, induce_pert_after_cycl=self.pb.pbf.perturb_after_cylce)

            # induce some disease/perturbation for cardiac cycle (i.e. valve stenosis or leakage)
            if self.pb.pbf.perturb_type is not None: self.pb.pbf.cardvasc0D.induce_perturbation(self.pb.pbf.perturb_type, self.pb.pbf.ti.cycle[0], self.pb.pbf.perturb_after_cylce)

            if is_periodic:
                if self.pb.comm.rank == 0:
                    print("Periodicity reached after %i heart cycles with cycle error %.4f! Finished. :-)" % (self.pb.pbf.ti.cycle[0]-1,self.pb.pbf.ti.cycleerror[0]))
                    sys.stdout.flush()
                break
            
            # maximum number of steps to perform
            try:
                if N+1 == self.pb.pbs.numstep_stop:
                    break
            except:
                pass
            

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()


    # for multiscale G&R analysis
    def set_homeostatic_threshold(self, t, dt):
            
        if t >= self.pb.t_gandr_setpoint and not self.pb.have_set_homeostatic:

            if self.pb.comm.rank == 0:
                print('Set homeostatic growth thresholds...')
                sys.stdout.flush()
            time.sleep(1)
            
            growth_thresolds = []
            for n in range(self.pb.pbs.num_domains):
                growth_settrig = self.pb.pbs.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_settrig']
                if growth_settrig == 'fibstretch':
                    growth_thresolds.append(self.pb.pbs.ma[n].fibstretch_e(self.pb.pbs.ki.C(self.pb.pbs.u), self.pb.pbs.theta, self.pb.pbs.fib_func[0]))
                elif growth_settrig == 'volstress':
                    growth_thresolds.append(tr(self.pb.pbs.ma[n].M_e(self.pb.pbs.u, self.pb.pbs.p, self.pb.pbs.ki.C(self.pb.pbs.u), ivar=self.pb.pbs.internalvars)))
                else:
                    raise NameError("Unknown growth trigger to be set as homeostatic threshold!")
                
            growth_thres_proj = project(growth_thresolds, self.pb.pbs.Vd_scalar, self.pb.pbs.dx_)
            self.pb.pbs.growth_thres.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.pb.pbs.growth_thres.interpolate(growth_thres_proj)

            self.pb.have_set_homeostatic = True

        
    # for multiscale G&R analysis
    def set_growth_trigger(self, t, dt):

        if t >= self.pb.t_gandr_setpoint and t < self.pb.t_gandr_setpoint + dt:

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

            self.pb.pbf.s_set.axpby(1.0, 0.0, self.pb.pbf.s)
