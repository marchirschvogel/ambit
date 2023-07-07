#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import utilities
import solver_nonlin
import expression
from projection import project
from mpiroutines import allgather_vec, allgather_vec_entry

from solid import SolidmechanicsProblem, SolidmechanicsSolverPrestr
from base import problem_base, solver_base


class SolidmechanicsConstraintProblem(problem_base):

    def __init__(self, io_params, time_params_solid, fem_params, constitutive_models, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None):
        super().__init__(io_params, time_params_solid, comm)

        self.problem_physics = 'solid_constraint'

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

        self.incompressible_2field = self.pbs.incompressible_2field

        self.set_variational_forms_and_jacobians()

        self.numdof = self.pbs.numdof + self.lm.getSize()

        self.localsolve = self.pbs.localsolve
        self.have_rom = self.pbs.have_rom

        self.sub_solve = False


    def get_problem_var_list(self):

        if self.pbs.incompressible_2field:
            is_ghosted = [1, 1, 0]
            return [self.pbs.u.vector, self.pbs.p.vector, self.lm], is_ghosted
        else:
            is_ghosted = [1, 0]
            return [self.pbs.u.vector, self.lm], is_ghosted


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

        self.work_coupling, self.work_coupling_old = ufl.as_ufl(0), ufl.as_ufl(0)

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
                    cq_ += self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                    cq_old_ += self.pbs.vf.volume(self.pbs.u_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                elif self.coupling_params['constraint_quantity'][n] == 'flux':
                    cq_ += self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                    cq_old_ += self.pbs.vf.flux(self.pbs.v_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                else:
                    raise NameError("Unknown constraint quantity! Choose either volume or flux!")

            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbs.u, self.pbs.du))

            df_ = ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):

                ds_p = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_p_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})
                df_ += self.pbs.timefac*self.pbs.vf.flux(self.pbs.var_u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_p)

                # add to solid rhs contributions
                self.work_coupling += self.pbs.vf.deltaW_ext_neumann_normal_cur(self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), self.coupfuncs[-1], ds_p)
                self.work_coupling_old += self.pbs.vf.deltaW_ext_neumann_normal_cur(self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), self.coupfuncs_old[-1], ds_p)

            self.dforce.append(fem.form(df_))

        # minus sign, since contribution to external work!
        self.pbs.weakform_u += -self.pbs.timefac * self.work_coupling - (1.-self.pbs.timefac) * self.work_coupling_old

        # add to solid Jacobian
        self.pbs.weakform_lin_uu += -self.pbs.timefac * ufl.derivative(self.work_coupling, self.pbs.u, self.pbs.du)


    def set_pressure_fem(self, var, p0Da):

        # set pressure functions
        for i in range(self.num_coupling_surf):
            self.pr0D.val = -allgather_vec_entry(var, i, self.comm)
            p0Da[i].interpolate(self.pr0D.evaluate)


    def set_problem_residual_jacobian_forms(self):

        self.pbs.set_problem_residual_jacobian_forms()


    def assemble_residual_stiffness(self, t, subsolver=None):

        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        K_list = [[None]*(2+off) for _ in range(2+off)]
        r_list = [None]*(2+off)

        # add to solid momentum equation
        self.set_pressure_fem(self.lm, self.coupfuncs)

        # solid main blocks
        r_list_solid, K_list_solid = self.pbs.assemble_residual_stiffness(t)

        K_list[0][0] = K_list_solid[0][0]
        r_list[0] = r_list_solid[0]
        if self.pbs.incompressible_2field:
            K_list[0][1] = K_list_solid[0][1]
            K_list[1][0] = K_list_solid[1][0]
            K_list[1][1] = K_list_solid[1][1] # should be only non-zero if we have stress-mediated growth...
            r_list[1] = r_list_solid[1]

        ls, le = self.lm.getOwnershipRange()

        for i in range(len(self.surface_p_ids)):
            cq = fem.assemble_scalar(fem.form(self.cq[i]))
            cq = self.comm.allgather(cq)
            self.constr[i] = sum(cq)*self.cq_factor[i]

        val, val_old = [], []
        for n in range(self.num_coupling_surf):
            curvenumber = self.prescribed_curve[n]
            val.append(self.pbs.ti.timecurves(curvenumber)(t)), val_old.append(self.pbs.ti.timecurves(curvenumber)(t-self.dt))

        # Lagrange multiplier coupling residual
        r_lm = PETSc.Vec().createMPI(size=self.num_coupling_surf)
        for i in range(ls,le):
            r_lm[i] = self.constr[i] - val[i]

        r_list[1+off] = r_lm

        # rows and columns for offdiagonal matrices
        row_ids = list(range(self.num_coupling_surf))
        col_ids = list(range(self.num_coupling_surf))

        # offdiagonal u-s columns
        k_us_cols=[]
        for i in range(len(col_ids)):
            k_us_cols.append(fem.petsc.assemble_vector(self.dforce[i])) # already multiplied by time-integration factor

        # offdiagonal s-u rows
        k_su_rows=[]
        for i in range(len(row_ids)):
            k_su_rows.append(fem.petsc.assemble_vector(fem.form(self.cq_factor[i]*self.dcq[i])))

        # apply dbcs to matrix entries - basically since these are offdiagonal we want a zero there!
        for i in range(len(col_ids)):

            fem.apply_lifting(k_us_cols[i], [self.pbs.jac_uu], [self.pbs.bc.dbcs], x0=[self.pbs.u.vector], scale=0.0)
            k_us_cols[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(k_us_cols[i], self.pbs.bc.dbcs, x0=self.pbs.u.vector, scale=0.0)

        # ghost update on k_su_rows
        for i in range(len(row_ids)):
            k_su_rows[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # setup offdiagonal matrices
        locmatsize = self.pbs.V_u.dofmap.index_map.size_local * self.pbs.V_u.dofmap.index_map_bs
        matsize = self.pbs.V_u.dofmap.index_map.size_global * self.pbs.V_u.dofmap.index_map_bs
        # row ownership range of uu block
        irs, ire = K_list[0][0].getOwnershipRange()

        # derivative of fluid residual w.r.t. 0D pressures
        K_us = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(self.num_coupling_surf)), bsize=None, nnz=None, csr=None, comm=self.comm)
        K_us.setUp()

        # set columns
        for i in range(len(col_ids)):
            K_us[irs:ire, col_ids[i]] = k_us_cols[i][irs:ire]

        K_us.assemble()

        # derivative of 0D residual w.r.t. solid displacements/fluid velocities
        K_su = PETSc.Mat().createAIJ(size=((self.num_coupling_surf),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.comm)
        K_su.setUp()

        # set rows
        for i in range(len(row_ids)):
            K_su[row_ids[i], irs:ire] = k_su_rows[i][irs:ire]

        K_su.assemble()

        K_list[0][1+off] = K_us
        K_list[1+off][0] = K_su

        # destroy PETSc vectors
        for i in range(len(row_ids)): k_su_rows[i].destroy()
        for i in range(len(col_ids)): k_us_cols[i].destroy()

        return r_list, K_list


    def get_index_sets(self, isoptions={}):

        if self.have_rom: # currently, ROM can only be on (subset of) first variable
            ured = PETSc.Vec().createMPI(size=(self.rom.V.getLocalSize()[1],self.rom.V.getSize()[1]), comm=self.comm)
            self.rom.V.multTranspose(self.pbs.u.vector, ured)
            uvec = ured
        else:
            uvec = self.pbs.u.vector

        offset_u = uvec.getOwnershipRange()[0] + self.lm.getOwnershipRange()[0]
        if self.pbs.incompressible_2field: offset_u += self.pbs.p.vector.getOwnershipRange()[0]
        iset_u = PETSc.IS().createStride(uvec.getLocalSize(), first=offset_u, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            offset_p = offset_u + uvec.getLocalSize()
            iset_p = PETSc.IS().createStride(self.pbs.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            offset_s = offset_p + self.pbs.p.vector.getLocalSize()
        else:
            offset_s = offset_u + uvec.getLocalSize()

        iset_s = PETSc.IS().createStride(self.lm.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            if isoptions['lms_to_p']:
                iset_p = iset_p.expand(iset_s) # add to pressure block
                return [iset_u, iset_p]
            elif isoptions['lms_to_v']:
                iset_u = iset_u.expand(iset_s) # add to displacement block (could be bad...)
                return [iset_u, iset_p]
            else:
                return [iset_u, iset_p, iset_s]
        else:
            return [iset_u, iset_s]


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # solid problem
        self.pbs.read_restart(sname, N)
        # LM data
        if self.pbs.restart_step > 0:
            restart_data = np.loadtxt(self.pbs.io.output_path+'/checkpoint_lm_'+str(N)+'.txt')
            self.lm[:], self.lm_old[:] = restart_data[:], restart_data[:]


    def evaluate_initial(self):

        self.pbs.evaluate_initial()

        self.set_pressure_fem(self.lm_old, self.coupfuncs_old)

        self.constr, self.constr_old = [], []
        for i in range(self.num_coupling_surf):
            lm_sq, lm_old_sq = allgather_vec(self.lm, self.comm), allgather_vec(self.lm_old, self.comm)
            con = fem.assemble_scalar(fem.form(self.cq[i]))
            con = self.comm.allgather(con)
            self.constr.append(sum(con))
            self.constr_old.append(sum(con))


    def write_output_ini(self):

        self.pbs.write_output_ini()


    def get_time_offset(self):
        return 0.


    def evaluate_pre_solve(self, t, N):

        self.pbs.evaluate_pre_solve(t, N)


    def evaluate_post_solve(self, t, N):

        self.pbs.evaluate_post_solve(t, N)


    def set_output_state(self, N):

        self.pbs.set_output_state(N)


    def write_output(self, N, t, mesh=False):

        self.pbs.write_output(N, t)


    def update(self):

        # update time step
        self.pbs.update()

        # update old pressures on solid
        self.lm_old.axpby(1.0, 0.0, self.lm)
        self.set_pressure_fem(self.lm_old, self.coupfuncs_old)
        # update old 3D constraint variable
        for i in range(self.num_coupling_surf):
            self.constr_old[i] = self.constr[i]


    def print_to_screen(self):

        self.pbs.print_to_screen()


    def induce_state_change(self):

        self.pbs.induce_state_change()


    def write_restart(self, sname, N):

        self.pbs.write_restart(sname, N)

        if self.pbs.io.write_restart_every > 0 and N % self.pbs.io.write_restart_every == 0:
            lm_sq = allgather_vec(self.lm, self.comm)
            if self.comm.rank == 0:
                f = open(self.pbs.io.output_path+'/checkpoint_lm_'+str(N)+'.txt', 'wt')
                for i in range(len(lm_sq)):
                    f.write('%.16E\n' % (lm_sq[i]))
                f.close()


    def check_abort(self, t):
        pass



class SolidmechanicsConstraintSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        # perform Proper Orthogonal Decomposition
        if self.pb.have_rom:
            self.pb.rom.prepare_rob()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], solver_params=self.solver_params)

        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            solver_params_prestr = copy.deepcopy(self.solver_params)
            # modify solver parameters in case user specified alternating ones for prestressing (should do, because it's a 2x2 problem maximum)
            try: solver_params_prestr['solve_type'] = self.solver_params['solve_type_prestr']
            except: pass
            try: solver_params_prestr['block_precond'] = self.solver_params['block_precond_prestr']
            except: pass
            try: solver_params_prestr['precond_fields'] = self.solver_params['precond_fields_prestr']
            except: pass
            # initialize solid mechanics solver
            self.solverprestr = SolidmechanicsSolverPrestr(self.pb.pbs, solver_params_prestr)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.ksp.destroy()

        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old - self.pb.work_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbs.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t, localdata=self.pb.pbs.localdata)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbs.ti.print_timestep(N, t, self.solnln.sepstring, ni=ni, li=li, wt=wt)
