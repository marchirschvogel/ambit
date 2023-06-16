#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
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

    def __init__(self, io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None, alevar={}):

        self.problem_physics = 'fluid_flow0d'

        self.comm = comm

        self.coupling_params = coupling_params

        self.surface_vq_ids = self.coupling_params['surface_ids']
        try: self.surface_p_ids = self.coupling_params['surface_p_ids']
        except: self.surface_p_ids = self.surface_vq_ids

        self.num_coupling_surf = len(self.surface_vq_ids)

        try: self.cq_factor = self.coupling_params['cq_factor']
        except: self.cq_factor = [1.]*self.num_coupling_surf

        try: self.eps_fd = self.coupling_params['eps_fd']
        except: self.eps_fd = 1.0e-5

        try: self.print_subiter = self.coupling_params['print_subiter']
        except: self.print_subiter = False

        try: self.restart_periodicref = self.coupling_params['restart_periodicref']
        except: self.restart_periodicref = 0

        try: self.Nmax_periodicref = self.coupling_params['Nmax_periodicref']
        except: self.Nmax_periodicref = 10

        # only option in fluid mechanics!
        self.coupling_type = 'monolithic_lagrange'

        # assert that we do not have conflicting timings
        time_params_flow0d['maxtime'] = time_params_fluid['maxtime']
        time_params_flow0d['numstep'] = time_params_fluid['numstep']

        # initialize problem instances (also sets the variational forms for the fluid problem)
        self.pbf = FluidmechanicsProblem(io_params, time_params_fluid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm, alevar=alevar)
        self.pb0 = Flow0DProblem(io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params, comm=self.comm)

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.set_variational_forms()

        self.numdof = self.pbf.numdof + self.pb0.numdof
        # fluid is 'master' problem - define problem variables based on its values
        self.simname = self.pbf.simname
        self.restart_step = self.pbf.restart_step
        self.numstep_stop = self.pbf.numstep_stop
        self.dt = self.pbf.dt
        self.localsolve = self.pbf.localsolve
        self.have_rom = self.pbf.have_rom

        self.sub_solve = True


    def get_problem_var_list(self):

        is_ghosted = [1, 1, 0]
        return [self.pbf.v.vector, self.pbf.p.vector, self.lm], is_ghosted


    # defines the monolithic coupling forms for 0D flow and fluid mechanics
    def set_variational_forms(self):

        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        self.coupfuncs, self.coupfuncs_old = [], []

        # Lagrange multipliers
        self.lm, self.lm_old = PETSc.Vec().createMPI(size=self.num_coupling_surf), PETSc.Vec().createMPI(size=self.num_coupling_surf)

        # 3D fluxes
        self.constr, self.constr_old = [], []

        self.power_coupling, self.power_coupling_old = ufl.as_ufl(0), ufl.as_ufl(0)

        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):

            self.pr0D = expression.template()

            self.coupfuncs.append(fem.Function(self.pbf.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbf.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)

            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):

                if self.pbf.alevar['w'] is None:
                    fluxvel, fluxvel_old = self.pbf.v, self.pbf.v_old
                else: # we need the relative velocity here
                    fluxvel, fluxvel_old = self.pbf.v - self.pbf.alevar['w'], self.pbf.v_old - self.pbf.alevar['w_old']

                ds_vq = ufl.ds(subdomain_data=self.pbf.io.mt_b1, subdomain_id=self.surface_vq_ids[n][i], metadata={'quadrature_degree': self.pbf.quad_degree})
                cq_ += self.pbf.vf.flux(fluxvel, ds_vq, w=self.pbf.alevar['w'], Fale=self.pbf.alevar['Fale'])
                cq_old_ += self.pbf.vf.flux(fluxvel_old, ds_vq, w=self.pbf.alevar['w_old'], Fale=self.pbf.alevar['Fale_old'])

            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbf.v, self.pbf.dv))

            df_ = ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):

                ds_p = ufl.ds(subdomain_data=self.pbf.io.mt_b1, subdomain_id=self.surface_p_ids[n][i], metadata={'quadrature_degree': self.pbf.quad_degree})
                df_ += self.pbf.timefac*self.pbf.vf.flux(self.pbf.var_v, ds_p, w=ufl.constantvalue.zero(self.pbf.ki.dim), Fale=self.pbf.alevar['Fale'])

                # add to fluid rhs contributions
                self.power_coupling += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs[-1], ds_p, Fale=self.pbf.alevar['Fale'])
                self.power_coupling_old += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs_old[-1], ds_p, Fale=self.pbf.alevar['Fale_old'])

            self.dforce.append(fem.form(df_))

        # minus sign, since contribution to external power!
        self.pbf.weakform_v += -self.pbf.timefac * self.power_coupling - (1.-self.pbf.timefac) * self.power_coupling_old

        # add to fluid Jacobian
        self.pbf.weakform_lin_vv += -self.pbf.timefac * ufl.derivative(self.power_coupling, self.pbf.v, self.pbf.dv)

        # old Lagrange multipliers - initialize with initial pressures
        self.pb0.cardvasc0D.initialize_lm(self.lm, self.pb0.initialconditions)
        self.pb0.cardvasc0D.initialize_lm(self.lm_old, self.pb0.initialconditions)


    def set_problem_residual_jacobian_forms(self):

        self.pbf.set_problem_residual_jacobian_forms()


    def assemble_residual_stiffness(self, t, subsolver=None):

        K_list = [[None]*3 for _ in range(3)]
        r_list = [None]*3

        for i in range(self.num_coupling_surf):
            cq = fem.assemble_scalar(fem.form(self.cq[i]))
            cq = self.comm.allgather(cq)
            self.constr[i] = sum(cq)*self.cq_factor[i]

        # Lagrange multipliers (pressures) to be passed to 0D model
        lm_sq = allgather_vec(self.lm, self.comm)

        for i in range(self.num_coupling_surf):
            self.pb0.c[i] = lm_sq[i]

        subsolver.newton(t, print_iter=self.print_subiter, sub=True)

        # add to fluid momentum equation
        self.pb0.cardvasc0D.set_pressure_fem(self.lm, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs)

        # fluid main blocks
        r_list_fluid, K_list_fluid = self.pbf.assemble_residual_stiffness(t)

        K_list[0][0] = K_list_fluid[0][0]
        K_list[0][1] = K_list_fluid[0][1]
        K_list[1][0] = K_list_fluid[1][0]
        K_list[1][1] = K_list_fluid[1][1] # should be only non-zero if we have stabilization...

        r_list[0] = r_list_fluid[0]
        r_list[1] = r_list_fluid[1]

        s_sq = allgather_vec(self.pb0.s, self.comm)

        ls, le = self.lm.getOwnershipRange()

        # Lagrange multiplier coupling residual
        r_lm = PETSc.Vec().createMPI(size=self.num_coupling_surf)
        for i in range(ls,le):
            r_lm[i] = self.constr[i] - s_sq[self.pb0.cardvasc0D.v_ids[i]]

        r_list[2] = r_lm

        # assemble 0D rhs contributions
        self.pb0.df_old.assemble()
        self.pb0.f_old.assemble()
        self.pb0.df.assemble()
        self.pb0.f.assemble()
        self.pb0.s.assemble()

        # now the LM matrix - via finite differencing
        # store df, f, and aux vectors prior to perturbation solves
        df_tmp, f_tmp, aux_tmp = self.pb0.K.createVecLeft(), self.pb0.K.createVecLeft(), np.zeros(self.pb0.numdof)
        df_tmp.axpby(1.0, 0.0, self.pb0.df)
        f_tmp.axpby(1.0, 0.0, self.pb0.f)
        aux_tmp[:] = self.pb0.aux[:]
        # store 0D state variable prior to perturbation solves
        s_tmp = self.pb0.K.createVecLeft()
        s_tmp.axpby(1.0, 0.0, self.pb0.s)

        # Lagrange multiplier stiffness matrix (currently treated with FD!)
        K_lm = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
        K_lm.setUp()

        # finite differencing for LM siffness matrix
        for i in range(self.num_coupling_surf):
            for j in range(self.num_coupling_surf):

                self.pb0.c[j] = lm_sq[j] + self.eps_fd # perturbed LM
                subsolver.newton(t, print_iter=False)
                s_pert_sq = allgather_vec(self.pb0.s, self.comm)
                K_lm[i,j] = -(s_pert_sq[self.pb0.cardvasc0D.v_ids[i]] - s_sq[self.pb0.cardvasc0D.v_ids[i]])/self.eps_fd
                self.pb0.c[j] = lm_sq[j] # restore LM

        # restore df, f, and aux vectors for correct time step update
        self.pb0.df.axpby(1.0, 0.0, df_tmp)
        self.pb0.f.axpby(1.0, 0.0, f_tmp)
        self.pb0.aux[:] = aux_tmp[:]
        # restore 0D state variable
        self.pb0.s.axpby(1.0, 0.0, s_tmp)

        df_tmp.destroy(), f_tmp.destroy(), s_tmp.destroy()
        del aux_tmp, lm_sq, s_sq, s_pert_sq

        K_lm.assemble()

        K_list[2][2] = K_lm

        # now the offdiagonal matrices
        row_ids = list(range(self.num_coupling_surf))
        col_ids = list(range(self.num_coupling_surf))

        # offdiagonal v-s columns
        k_vs_cols=[]
        for i in range(len(col_ids)):
            k_vs_cols.append(fem.petsc.assemble_vector(self.dforce[i])) # already multiplied by time-integration factor

        # offdiagonal s-v rows
        k_sv_rows=[]
        for i in range(len(row_ids)):
            k_sv_rows.append(fem.petsc.assemble_vector(fem.form(self.cq_factor[i]*self.dcq[i])))

        # apply velocity dbcs to matrix entries k_vs - basically since these are offdiagonal we want a zero there!
        for i in range(len(col_ids)):

            fem.apply_lifting(k_vs_cols[i], [self.pbf.jac_vv], [self.pbf.bc.dbcs], x0=[self.pbf.v.vector], scale=0.0)
            k_vs_cols[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(k_vs_cols[i], self.pbf.bc.dbcs, x0=self.pbf.v.vector, scale=0.0)

        # ghost update on k_sv_rows
        for i in range(len(row_ids)):
            k_sv_rows[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # setup offdiagonal matrices
        locmatsize = self.pbf.V_v.dofmap.index_map.size_local * self.pbf.V_v.dofmap.index_map_bs
        matsize = self.pbf.V_v.dofmap.index_map.size_global * self.pbf.V_v.dofmap.index_map_bs
        # row ownership range of vv block
        irs, ire = K_list[0][0].getOwnershipRange()

        # derivative of fluid residual w.r.t. 0D pressures
        K_vs = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(K_lm.getSize()[0])), bsize=None, nnz=None, csr=None, comm=self.comm)
        K_vs.setUp()

        # set columns
        for i in range(len(col_ids)):
            K_vs[irs:ire, col_ids[i]] = k_vs_cols[i][irs:ire]

        K_vs.assemble()

        # derivative of 0D residual w.r.t. fluid velocities
        K_sv = PETSc.Mat().createAIJ(size=((K_lm.getSize()[0]),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.comm)
        K_sv.setUp()

        # set rows
        for i in range(len(row_ids)):
            K_sv[row_ids[i], irs:ire] = k_sv_rows[i][irs:ire]

        K_sv.assemble()

        K_list[0][2] = K_vs
        K_list[2][0] = K_sv

        # destroy PETSc vectors
        for i in range(len(row_ids)): k_sv_rows[i].destroy()
        for i in range(len(col_ids)): k_vs_cols[i].destroy()

        return r_list, K_list


    def get_index_sets(self, isoptions={}):

        if self.have_rom: # currently, ROM can only be on (subset of) first variable
            vred = PETSc.Vec().createMPI(size=(self.rom.V.getLocalSize()[1],self.rom.V.getSize()[1]), comm=self.comm)
            self.rom.V.multTranspose(self.pbf.v.vector, vred)
            vvec = vred
        else:
            vvec = self.pbf.v.vector

        offset_v = vvec.getOwnershipRange()[0] + self.pbf.p.vector.getOwnershipRange()[0] + self.lm.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec.getLocalSize(), first=offset_v, step=1, comm=self.comm)

        offset_p = offset_v + vvec.getLocalSize()
        iset_p = PETSc.IS().createStride(self.pbf.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_s = offset_p + self.pbf.p.vector.getLocalSize()
        iset_s = PETSc.IS().createStride(self.lm.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        if isoptions['lms_to_p']:
            iset_p = iset_p.expand(iset_s) # add to pressure block
            return [iset_v, iset_p]
        elif isoptions['lms_to_v']:
            iset_v = iset_u.expand(iset_s) # add to velocity block (could be bad...)
            return [iset_v, iset_p]
        else:
            return [iset_v, iset_p, iset_s]


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # fluid + flow0d problem
        self.pbf.read_restart(sname, N)
        self.pb0.read_restart(sname, N)

        if self.pbf.restart_step > 0:
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm)
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm_old)


    def evaluate_initial(self):

        self.pbf.evaluate_initial()

        self.pb0.cardvasc0D.set_pressure_fem(self.lm_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)

        self.pb0.c, self.constr, self.constr_old = [], [], []
        for i in range(self.num_coupling_surf):
            lm_sq, lm_old_sq = allgather_vec(self.lm, self.comm), allgather_vec(self.lm_old, self.comm)
            self.pb0.c.append(lm_sq[i])
            con = fem.assemble_scalar(fem.form(self.cq_old[i]))
            con = self.comm.allgather(con)
            self.constr.append(sum(con)*self.cq_factor[i])
            self.constr_old.append(sum(con)*self.cq_factor[i])

        # length of c from 3D-0D coupling
        self.pb0.len_c_3d0d = len(self.pb0.c)

        if bool(self.pb0.chamber_models):
            self.pb0.y = []
            for ch in ['lv','rv','la','ra']:
                if self.pb0.chamber_models[ch]['type']=='0D_elast': self.pb0.y.append(self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['activation_curve'])(self.pbf.t_init))
                if self.pb0.chamber_models[ch]['type']=='0D_elast_prescr': self.pb0.y.append(self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['elastance_curve'])(self.pbf.t_init))
                if self.pb0.chamber_models[ch]['type']=='0D_prescr': self.pb0.c.append(self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['prescribed_curve'])(self.pbf.t_init))

        # if we have prescribed variable values over time
        if bool(self.pb0.prescribed_variables):
            for a in self.pb0.prescribed_variables:
                varindex = self.pb0.cardvasc0D.varmap[a]
                curvenumber = self.pb0.prescribed_variables[a]
                val = self.pb0.ti.timecurves(curvenumber)(self.pb0.t_init)
                self.pb0.s[varindex], self.pb0.s_old[varindex] = val, val

        # initially evaluate 0D model at old state
        self.pb0.cardvasc0D.evaluate(self.pb0.s_old, self.pbf.t_init, self.pb0.df_old, self.pb0.f_old, None, None, self.pb0.c, self.pb0.y, self.pb0.aux_old)
        self.pb0.auxTc_old[:] = self.pb0.aux_old[:]


    def write_output_ini(self):

        self.pbf.write_output_ini()


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t):

        self.pbf.evaluate_pre_solve(t)
        self.pb0.evaluate_pre_solve(t)


    def evaluate_post_solve(self, t, N):

        self.pbf.evaluate_post_solve(t, N)
        self.pb0.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbf.set_output_state()
        self.pb0.set_output_state()


    def write_output(self, N, t, mesh=False):

        self.pbf.write_output(N, t)
        self.pb0.write_output(N, t)


    def update(self):

        # update time step - fluid and 0D model
        self.pbf.update()
        self.pb0.update()

        # update old LMs
        self.lm_old.axpby(1.0, 0.0, self.lm)
        self.pb0.cardvasc0D.set_pressure_fem(self.lm_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)
        # update old 3D fluxes
        self.constr_old[:] = self.constr[:]


    def print_to_screen(self):

        self.pbf.print_to_screen()
        self.pb0.print_to_screen()


    def induce_state_change(self):

        self.pbf.induce_state_change()
        self.pb0.induce_state_change()


    def write_restart(self, sname, N):

        self.pbf.write_restart(sname, N)
        self.pb0.write_restart(sname, N)

        if self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0:
            self.pb0.cardvasc0D.write_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm)


    def check_abort(self, t):

        self.pb0.check_abort(t)



class FluidmechanicsFlow0DSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        # perform Proper Orthogonal Decomposition
        if self.pb.have_rom:
            self.pb.rom.prepare_rob()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], solver_params=self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbf.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old - self.pb.power_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.sepstring, self.pb.pbf.numstep, ni=ni, li=li, wt=wt)
