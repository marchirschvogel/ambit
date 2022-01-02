#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
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
from mpiroutines import allgather_vec, allgather_vec_entry

from solid import SolidmechanicsProblem, SolidmechanicsSolver


class SolidmechanicsConstraintProblem():

    def __init__(self, io_params, time_params_solid, fem_params, constitutive_models, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None):
        
        self.problem_physics = 'solid_constraint'
        
        self.comm = comm
        
        self.coupling_params = coupling_params

        self.surface_c_ids = self.coupling_params['surface_ids']
        try: self.surface_p_ids = self.coupling_params['surface_p_ids']
        except: self.surface_p_ids = self.surface_c_ids
        
        self.num_coupling_surf = len(self.surface_c_ids)
        
        self.cq_factor = [1.]*self.num_coupling_surf

        self.coupling_type = 'monolithic_lagrange'

        self.prescribed_curve = self.coupling_params['prescribed_curve']

        # initialize problem instances (also sets the variational forms for the solid problem)
        self.pbs = SolidmechanicsProblem(io_params, time_params_solid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm)

        self.set_variational_forms_and_jacobians()

        
    # defines the monolithic coupling forms for constraints and solid mechanics
    def set_variational_forms_and_jacobians(self):

        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        self.coupfuncs, self.coupfuncs_old = [], []
        
        # Lagrange multiplier stiffness matrix (most likely to be zero!)
        self.K_lm = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_lm.setUp()

        # Lagrange multipliers
        self.lm, self.lm_old = self.K_lm.createVecLeft(), self.K_lm.createVecLeft()
        
        # 3D constraint variable (volume or flux)
        self.constr, self.constr_old = [], []
        
        self.work_coupling, self.work_coupling_old, self.work_coupling_prestr = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        
        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):
            
            self.pr0D = expression.template()
            
            self.coupfuncs.append(fem.Function(self.pbs.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbs.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)
            
            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_c_ids[n])):
                
                ds_vq = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_c_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
                
                # currently, only volume or flux constraints are supported
                if self.coupling_params['constraint_quantity'][n] == 'volume':
                    cq_ += self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_vq)
                    cq_old_ += self.pbs.vf.volume(self.pbs.u_old, self.pbs.ki.J(self.pbs.u_old), self.pbs.ki.F(self.pbs.u_old), ds_vq)
                elif self.coupling_params['constraint_quantity'][n] == 'flux':
                    cq_ += self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_vq)
                    cq_old_ += self.pbs.vf.flux(self.pbs.v_old, self.pbs.ki.J(self.pbs.u_old), self.pbs.ki.F(self.pbs.u_old), ds_vq)
                else:
                    raise NameError("Unknown constraint quantity! Choose either volume or flux!")
            
            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbs.u, self.pbs.du))
            
            df_ = ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):
            
                ds_p = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_p_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
                df_ += self.pbs.timefac*self.pbs.vf.surface(self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), ds_p)
            
                # add to solid rhs contributions
                self.work_coupling += self.pbs.vf.deltaW_ext_neumann_true(self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), self.coupfuncs[-1], ds_p)
                self.work_coupling_old += self.pbs.vf.deltaW_ext_neumann_true(self.pbs.ki.J(self.pbs.u_old), self.pbs.ki.F(self.pbs.u_old), self.coupfuncs_old[-1], ds_p)
                
                # for prestressing, true loads should act on the reference, not the current configuration
                if self.pbs.prestress_initial:
                    self.work_coupling_prestr += self.pbs.vf.deltaW_ext_neumann_refnormal(self.coupfuncs_old[-1], ds_p)

            self.dforce.append(df_)

        # minus sign, since contribution to external work!
        self.pbs.weakform_u += -self.pbs.timefac * self.work_coupling - (1.-self.pbs.timefac) * self.work_coupling_old
        
        # add to solid Jacobian
        self.pbs.jac_uu += -self.pbs.timefac * ufl.derivative(self.work_coupling, self.pbs.u, self.pbs.du)


    def set_pressure_fem(self, var, p0Da):
        
        # set pressure functions
        for i in range(self.num_coupling_surf):
            self.pr0D.val = -allgather_vec_entry(var, i, self.comm)
            p0Da[i].interpolate(self.pr0D.evaluate)



class SolidmechanicsConstraintSolver():

    def __init__(self, problem, solver_params_solid, solver_params_constr):
    
        self.pb = problem
        
        self.solver_params_solid = solver_params_solid
        self.solver_params_constr = solver_params_constr

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_constraint_monolithic(self.pb, self.pb.pbs.V_u, self.pb.pbs.V_p, self.solver_params_solid, self.solver_params_constr)
        
        if self.pb.pbs.prestress_initial:
            # add coupling work to prestress weak form
            self.pb.pbs.weakform_prestress_u -= self.pb.work_coupling_prestr            
            # initialize solid mechanics solver
            self.solverprestr = SolidmechanicsSolver(self.pb.pbs, self.solver_params_solid)


    def solve_problem(self):
        
        start = time.time()
        
        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.pbs.comm, self.pb.pbs.ndof)

        # read restart information
        if self.pb.pbs.restart_step > 0:
            self.pb.pbs.io.readcheckpoint(self.pb.pbs, self.pb.pbs.restart_step)
            self.pb.pbs.simname += '_r'+str(self.pb.pbs.restart_step)

        self.pb.set_pressure_fem(self.pb.lm_old, self.pb.coupfuncs_old)

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            del self.solverprestr
        else:
            # set flag definitely to False if we're restarting
            self.pb.pbs.prestress_initial = False

        self.pb.constr, self.pb.constr_old = [], []
        for i in range(self.pb.num_coupling_surf):
            lm_sq, lm_old_sq = allgather_vec(self.pb.lm, self.pb.comm), allgather_vec(self.pb.lm_old, self.pb.comm)
            con = fem.assemble_scalar(self.pb.cq[i])
            con = self.pb.pbs.comm.allgather(con)
            self.pb.constr.append(sum(con))
            self.pb.constr_old.append(sum(con))
                 
        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old - self.pb.work_coupling_old

            jac_a = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbs.a_old)

        # write mesh output
        self.pb.pbs.io.write_output(self.pb.pbs, writemesh=True)
        
        # solid constraint main time loop
        for N in range(self.pb.pbs.restart_step+1, self.pb.pbs.numstep_stop+1):

            wts = time.time()
            
            # current time
            t = N * self.pb.pbs.dt
            
            # set time-dependent functions
            self.pb.pbs.ti.set_time_funcs(self.pb.pbs.ti.funcs_to_update, self.pb.pbs.ti.funcs_to_update_vec, t)

            # take care of active stress
            if self.pb.pbs.have_active_stress and self.pb.pbs.active_stress_trig == 'ode':
                self.pb.pbs.evaluate_active_stress_ode(t)

            # solve
            self.solnln.newton(self.pb.pbs.u, self.pb.pbs.p, self.pb.lm, t, locvars=[self.pb.pbs.theta], locresforms=[self.pb.pbs.r_growth], locincrforms=[self.pb.pbs.del_theta])

            # write output
            self.pb.pbs.io.write_output(self.pb.pbs, N=N, t=t)

            # update time step
            self.pb.pbs.ti.update_timestep(self.pb.pbs.u, self.pb.pbs.u_old, self.pb.pbs.v_old, self.pb.pbs.a_old, self.pb.pbs.p, self.pb.pbs.p_old, self.pb.pbs.internalvars, self.pb.pbs.internalvars_old, self.pb.pbs.ratevars, self.pb.pbs.ratevars_old, self.pb.pbs.ti.funcs_to_update, self.pb.pbs.ti.funcs_to_update_old, self.pb.pbs.ti.funcs_to_update_vec, self.pb.pbs.ti.funcs_to_update_vec_old)

            # update old pressures on solid
            self.pb.lm.assemble(), self.pb.lm_old.axpby(1.0, 0.0, self.pb.lm)
            self.pb.set_pressure_fem(self.pb.lm_old, self.pb.coupfuncs_old)
            # update old 3D constraint variable
            for i in range(self.pb.num_coupling_surf):
                self.pb.constr_old[i] = self.pb.constr[i]

            # solve time for time step
            wte = time.time()
            wt = wte - wts

            # print time step info to screen
            self.pb.pbs.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)

            # write restart info - old and new quantities are the same at this stage
            self.pb.pbs.io.write_restart(self.pb, N)
            
        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()
