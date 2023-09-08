#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import expression, ioparams
from ..solver.projection import project
from ..mpiroutines import allgather_vec

from ..solid.solid_main import SolidmechanicsProblem, SolidmechanicsSolverPrestr
from ..flow0d.flow0d_main import Flow0DProblem
from ..base import problem_base, solver_base


class SolidmechanicsFlow0DProblem(problem_base):

    def __init__(self, io_params, time_params_solid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None):
        super().__init__(io_params, time_params_solid, comm)

        self.problem_physics = 'solid_flow0d'

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

        self.pbrom = self.pbs # ROM problem can only be solid

        self.incompressible_2field = self.pbs.incompressible_2field

        # for multiscale G&R analysis
        self.t_prev = 0
        self.t_gandr_setpoint = 0
        self.restart_multiscale = False

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        if self.pbs.problem_type == 'solid_flow0d_multiscale_gandr': self.have_multiscale_gandr = True
        else: self.have_multiscale_gandr = False

        self.set_variational_forms()

        if self.coupling_type == 'monolithic_direct':
            self.numdof = self.pbs.numdof + self.pb0.numdof
        elif self.coupling_type == 'monolithic_lagrange':
            self.numdof = self.pbs.numdof + self.lm.getSize()
        else:
            raise ValueError("Unknown coupling type!")

        self.localsolve = self.pbs.localsolve

        if self.coupling_type == 'monolithic_lagrange':
            self.sub_solve = True
        else:
            self.sub_solve = False

        self.print_enhanced_info = self.pbs.io.print_enhanced_info

        # 3D fluxes
        self.constr, self.constr_old = [[]]*self.num_coupling_surf, [[]]*self.num_coupling_surf

        # set 3D-0D coupling array
        self.pb0.c = [[]]*(self.num_coupling_surf)

        # number of fields involved
        if self.pbs.incompressible_2field: self.nfields=3
        else: self.nfields=2

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.coupling_type == 'monolithic_lagrange':
            if self.pbs.incompressible_2field:
                is_ghosted = [1, 1, 0]
                return [self.pbs.u.vector, self.pbs.p.vector, self.lm], is_ghosted
            else:
                is_ghosted = [1, 0]
                return [self.pbs.u.vector, self.lm], is_ghosted

        if self.coupling_type == 'monolithic_direct':
            if self.pbs.incompressible_2field:
                is_ghosted = [1, 1, 0]
                return [self.pbs.u.vector, self.pbs.p.vector, self.pb0.s], is_ghosted
            else:
                is_ghosted = [1, 0]
                return [self.pbs.u.vector, self.pb0.s], is_ghosted


    # defines the monolithic coupling forms for 0D flow and solid mechanics
    def set_variational_forms(self):

        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        self.coupfuncs, self.coupfuncs_old, coupfuncs_pre = [], [], []

        if self.coupling_type == 'monolithic_lagrange':

            # Lagrange multipliers
            self.lm, self.lm_old = PETSc.Vec().createMPI(size=self.num_coupling_surf), PETSc.Vec().createMPI(size=self.num_coupling_surf)

        self.work_coupling, self.work_coupling_old = ufl.as_ufl(0), ufl.as_ufl(0)

        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):

            try: coupling_quantity = self.coupling_params['coupling_quantity'][n]
            except: coupling_quantity = 'volume'

            try: variable_quantity = self.coupling_params['variable_quantity'][n]
            except: variable_quantity = 'pressure'

            self.pr0D = expression.template()

            self.coupfuncs.append(fem.Function(self.pbs.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbs.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)

            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):

                ds_vq = ufl.ds(subdomain_data=self.pbs.io.mt_b1, subdomain_id=self.surface_vq_ids[n][i], metadata={'quadrature_degree': self.pbs.quad_degree})

                if coupling_quantity == 'volume':
                    assert(self.coupling_type == 'monolithic_direct' and variable_quantity == 'pressure')
                    cq_ += self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                    cq_old_ += self.pbs.vf.volume(self.pbs.u_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                elif coupling_quantity == 'flux':
                    assert(self.coupling_type == 'monolithic_direct' and variable_quantity == 'pressure')
                    cq_ += self.pbs.vf.flux(self.pbs.vel, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                    cq_old_ += self.pbs.vf.flux(self.pbs.v_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                elif coupling_quantity == 'pressure':
                    assert(self.coupling_type == 'monolithic_lagrange')
                    if variable_quantity == 'volume':
                        cq_ += self.pbs.vf.volume(self.pbs.u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_vq)
                        cq_old_ += self.pbs.vf.volume(self.pbs.u_old, self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), ds_vq)
                    elif variable_quantity == 'flux':
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
                df_ += self.pbs.timefac*self.pbs.vf.flux(self.pbs.var_u, self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), ds_p)

                # add to solid rhs contributions
                self.work_coupling += self.pbs.vf.deltaW_ext_neumann_normal_cur(self.pbs.ki.J(self.pbs.u,ext=True), self.pbs.ki.F(self.pbs.u,ext=True), self.coupfuncs[-1], ds_p)
                self.work_coupling_old += self.pbs.vf.deltaW_ext_neumann_normal_cur(self.pbs.ki.J(self.pbs.u_old,ext=True), self.pbs.ki.F(self.pbs.u_old,ext=True), self.coupfuncs_old[-1], ds_p)

            self.dforce.append(df_)

        # minus sign, since contribution to external work!
        self.pbs.weakform_u += -self.pbs.timefac * self.work_coupling - (1.-self.pbs.timefac) * self.work_coupling_old

        # add to solid Jacobian
        self.pbs.weakform_lin_uu += -self.pbs.timefac * ufl.derivative(self.work_coupling, self.pbs.u, self.pbs.du)

        if self.coupling_type == 'monolithic_lagrange' and self.pbs.restart_step==0:
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

            growth_thres_proj = project(growth_thresolds, self.pbs.Vd_scalar, self.pbs.dx_, comm=self.comm)
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


    def set_problem_residual_jacobian_forms(self):

        self.pbs.set_problem_residual_jacobian_forms()
        self.set_problem_residual_jacobian_forms_coupling()


    def set_problem_residual_jacobian_forms_coupling(self):

        tes = time.time()
        if self.comm.rank == 0:
            print('FEM form compilation for solid-0D coupling...')
            sys.stdout.flush()

        self.cq_form, self.cq_old_form, self.dcq_form, self.dforce_form = [], [], [], []

        for i in range(self.num_coupling_surf):
            self.cq_form.append(fem.form(self.cq[i]))
            self.cq_old_form.append(fem.form(self.cq_old[i]))

            self.dcq_form.append(fem.form(self.cq_factor[i]*self.dcq[i]))
            self.dforce_form.append(fem.form(self.dforce[i]))

        tee = time.time() - tes
        if self.comm.rank == 0:
            print('FEM form compilation for solid-0D finished, te = %.2f s' % (tee))
            sys.stdout.flush()


    def set_problem_vector_matrix_structures(self):

        self.pbs.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()


    def set_problem_vector_matrix_structures_coupling(self):

        self.r_lm = PETSc.Vec().createMPI(size=self.num_coupling_surf)

        # Lagrange multiplier stiffness matrix (currently treated with FD!)
        if self.coupling_type == 'monolithic_lagrange':
            self.K_lm = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
            self.K_lm.setUp()
            sze_coup = self.num_coupling_surf
            self.row_ids = list(range(self.num_coupling_surf))
            self.col_ids = list(range(self.num_coupling_surf))

        if self.coupling_type == 'monolithic_direct':
            sze_coup = self.pb0.numdof
            self.row_ids = self.pb0.cardvasc0D.c_ids
            self.col_ids = self.pb0.cardvasc0D.v_ids

        # setup offdiagonal matrices
        locmatsize = self.pbs.V_u.dofmap.index_map.size_local * self.pbs.V_u.dofmap.index_map_bs
        matsize = self.pbs.V_u.dofmap.index_map.size_global * self.pbs.V_u.dofmap.index_map_bs

        # derivative of solid residual w.r.t. 0D pressures
        self.k_us_vec = []
        for i in range(len(self.col_ids)):
            self.k_us_vec.append(fem.petsc.create_vector(self.dforce_form[i]))

        self.K_us = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(sze_coup)), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_us.setUp()

        self.k_su_vec = []
        for i in range(len(self.row_ids)):
            self.k_su_vec.append(fem.petsc.create_vector(self.dcq_form[i]))

        # derivative of 0D residual w.r.t. solid displacements
        self.K_su = PETSc.Mat().createAIJ(size=((sze_coup),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_su.setUp()


    def assemble_residual(self, t, subsolver=None):

        if self.pbs.incompressible_2field: off = 1
        else: off = 0

        if self.coupling_type == 'monolithic_lagrange':

            for i in range(self.num_coupling_surf):
                cq = fem.assemble_scalar(self.cq_form[i])
                cq = self.comm.allgather(cq)
                self.constr[i] = sum(cq)*self.cq_factor[i]

            # Lagrange multipliers (pressures) to be passed to 0D model
            lm_sq = allgather_vec(self.lm, self.comm)

            for i in range(self.num_coupling_surf):
                self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = lm_sq[i]

            if subsolver is not None:
                subsolver.newton(t, print_iter=self.print_subiter, sub=True)

            # add to solid momentum equation
            self.pb0.cardvasc0D.set_pressure_fem(self.lm, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs)

        if self.coupling_type == 'monolithic_direct':

            # add to solid momentum equation
            self.pb0.cardvasc0D.set_pressure_fem(self.pb0.s, self.pb0.cardvasc0D.v_ids, self.pr0D, self.coupfuncs)

            # volumes/fluxes to be passed to 0D model
            for i in range(len(self.pb0.cardvasc0D.c_ids)):
                cq = fem.assemble_scalar(self.cq_form[i])
                cq = self.comm.allgather(cq)
                self.pb0.c[i] = sum(cq)*self.cq_factor[i]

            # 0D rhs vector
            self.pb0.assemble_residual(t)

            self.r_list[1+off] = self.pb0.r_list[0]

        # solid main blocks
        self.pbs.assemble_residual(t)

        self.r_list[0] = self.pbs.r_list[0]
        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbs.r_list[1]

        if self.coupling_type == 'monolithic_lagrange':

            s_sq = allgather_vec(self.pb0.s, self.comm)

            ls, le = self.lm.getOwnershipRange()

            # Lagrange multiplier coupling residual
            for i in range(ls,le):
                self.r_lm[i] = self.constr[i] - s_sq[self.pb0.cardvasc0D.v_ids[i]]

            self.r_lm.assemble()

            self.r_list[1+off] = self.r_lm

            del lm_sq, s_sq

        if bool(self.residual_scale):
            self.scale_residual_list([self.r_list[1+off]], [self.residual_scale[1+off]])


    def assemble_stiffness(self, t, subsolver=None):

        if self.pbs.incompressible_2field: off = 1
        else: off = 0

        if self.coupling_type == 'monolithic_lagrange':

            # Lagrange multipliers (pressures) to be passed to 0D model
            lm_sq = allgather_vec(self.lm, self.comm)

            for i in range(self.num_coupling_surf):
                self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = lm_sq[i]

        if self.coupling_type == 'monolithic_direct':

            # volumes/fluxes to be passed to 0D model
            for i in range(len(self.pb0.cardvasc0D.c_ids)):
                cq = fem.assemble_scalar(fem.form(self.cq[i]))
                cq = self.comm.allgather(cq)
                self.pb0.c[i] = sum(cq)*self.cq_factor[i]

            # 0D stiffness
            self.pb0.assemble_stiffness(t)

            self.K_list[1+off][1+off] = self.pb0.K_list[0][0]

        # solid main blocks
        self.pbs.assemble_stiffness(t)

        self.K_list[0][0] = self.pbs.K_list[0][0]
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbs.K_list[0][1]
            self.K_list[1][0] = self.pbs.K_list[1][0]
            self.K_list[1][1] = self.pbs.K_list[1][1] # should be only non-zero if we have stress-mediated growth...

        if self.coupling_type == 'monolithic_lagrange':

            s_sq = allgather_vec(self.pb0.s, self.comm)

            # assemble 0D rhs contributions
            self.pb0.df_old.assemble()
            self.pb0.f_old.assemble()
            self.pb0.df.assemble()
            self.pb0.f.assemble()
            self.pb0.s.assemble()

            # now the LM matrix - via finite differencing
            # store df, f, and aux vectors prior to perturbation solves
            self.pb0.df_tmp.axpby(1.0, 0.0, self.pb0.df)
            self.pb0.f_tmp.axpby(1.0, 0.0, self.pb0.f)
            self.pb0.aux_tmp[:] = self.pb0.aux[:]
            # store 0D state variable prior to perturbation solves
            self.pb0.s_tmp.axpby(1.0, 0.0, self.pb0.s)

            # finite differencing for LM siffness matrix
            if subsolver is not None:
                for i in range(self.num_coupling_surf):
                    for j in range(self.num_coupling_surf):

                        self.pb0.c[self.pb0.cardvasc0D.c_ids[j]] = lm_sq[j] + self.eps_fd # perturbed LM
                        subsolver.newton(t, print_iter=False)
                        s_pert_sq = allgather_vec(self.pb0.s, self.comm)
                        self.K_lm[i,j] = -(s_pert_sq[self.pb0.cardvasc0D.v_ids[i]] - s_sq[self.pb0.cardvasc0D.v_ids[i]])/self.eps_fd
                        self.pb0.c[self.pb0.cardvasc0D.c_ids[j]] = lm_sq[j] # restore LM

            # restore df, f, and aux vectors for correct time step update
            self.pb0.df.axpby(1.0, 0.0, self.pb0.df_tmp)
            self.pb0.f.axpby(1.0, 0.0, self.pb0.f_tmp)
            self.pb0.aux[:] = self.pb0.aux_tmp[:]
            # restore 0D state variable
            self.pb0.s.axpby(1.0, 0.0, self.pb0.s_tmp)

            self.K_lm.assemble()

            self.K_list[1+off][1+off] = self.K_lm

            del lm_sq, s_sq

        # offdiagonal s-u rows
        for i in range(len(self.row_ids)):

            # depending on if we have volumes, fluxes, or pressures passed in (latter for LM coupling)
            if self.pb0.cq[i] == 'volume':   timefac = 1./self.dt
            if self.pb0.cq[i] == 'flux':     timefac = -self.pb0.theta0d_timint(t) # 0D model time-integration factor
            if self.pb0.cq[i] == 'pressure': timefac = 1.

            with self.k_su_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_su_vec[i], self.dcq_form[i])
            # ghost update on k_su_rows - needs to be done prior to scale
            self.k_su_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.k_su_vec[i].scale(timefac)

        # offdiagonal u-s columns
        for i in range(len(self.col_ids)):
            with self.k_us_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_us_vec[i], self.dforce_form[i]) # already multiplied by time-integration factor
            # apply displacement dbcs to matrix entries k_us - basically since these are offdiagonal we want a zero there!
            fem.apply_lifting(self.k_us_vec[i], [self.pbs.jac_uu], [self.pbs.bc.dbcs], x0=[self.pbs.u.vector], scale=0.0)
            self.k_us_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(self.k_us_vec[i], self.pbs.bc.dbcs, x0=self.pbs.u.vector, scale=0.0)

        # row ownership range of uu block
        irs, ire = self.pbs.K_list[0][0].getOwnershipRange()

        # set columns
        for i in range(len(self.col_ids)):
            self.K_us[irs:ire, self.col_ids[i]] = self.k_us_vec[i][irs:ire]

        self.K_us.assemble()

        # set rows
        for i in range(len(self.row_ids)):
            self.K_su[self.row_ids[i], irs:ire] = self.k_su_vec[i][irs:ire]

        self.K_su.assemble()

        if bool(self.residual_scale):
            self.K_us.scale(self.residual_scale[0])
            self.K_su.scale(self.residual_scale[1+off])
            self.K_list[1+off][1+off].scale(self.residual_scale[1+off])

        self.K_list[0][1+off] = self.K_us
        self.K_list[1+off][0] = self.K_su


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            uvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            uvec_ls = self.rom.V.getLocalSize()[1]
        else:
            uvec_or0 = self.pbs.u.vector.getOwnershipRange()[0]
            uvec_ls = self.pbs.u.vector.getLocalSize()

        if self.coupling_type == 'monolithic_direct':   rvec = self.pb0.s
        if self.coupling_type == 'monolithic_lagrange': rvec = self.lm

        offset_u = uvec_or0 + rvec.getOwnershipRange()[0]
        if self.pbs.incompressible_2field: offset_u += self.pbs.p.vector.getOwnershipRange()[0]
        iset_u = PETSc.IS().createStride(uvec_ls, first=offset_u, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            offset_p = offset_u + uvec_ls
            iset_p = PETSc.IS().createStride(self.pbs.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            offset_s = offset_p + self.pbs.p.vector.getLocalSize()
        else:
            offset_s = offset_u + uvec_ls

        iset_s = PETSc.IS().createStride(rvec.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            if isoptions['lms_to_p']:
                iset_p = iset_p.expand(iset_s) # add to pressure block
                ilist = [iset_u, iset_p]
            elif isoptions['lms_to_v']:
                iset_u = iset_u.expand(iset_s) # add to displacement block (could be bad...)
                ilist = [iset_u, iset_p]
            else:
                ilist = [iset_u, iset_p, iset_s]
        else:
            ilist = [iset_u, iset_s]

        return ilist


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # solid + flow0d problem
        self.pbs.read_restart(sname, N)
        self.pb0.read_restart(sname, N)

        if self.pbs.restart_step > 0:
            if self.coupling_type == 'monolithic_lagrange':
                self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm)
                self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm_old)


    def evaluate_initial(self):

        self.pbs.evaluate_initial()

        # set pressure functions for old state - s_old already initialized by 0D flow problem
        if self.coupling_type == 'monolithic_direct':
            self.pb0.cardvasc0D.set_pressure_fem(self.pb0.s_old, self.pb0.cardvasc0D.v_ids, self.pr0D, self.coupfuncs_old)

        if self.coupling_type == 'monolithic_lagrange':
            self.pb0.cardvasc0D.set_pressure_fem(self.lm_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)

        if self.coupling_type == 'monolithic_direct':
            # old 3D coupling quantities (volumes or fluxes)
            for i in range(self.num_coupling_surf):
                cq = fem.assemble_scalar(self.cq_old_form[i])
                cq = self.comm.allgather(cq)
                self.pb0.c[i] = sum(cq)*self.cq_factor[i]

        if self.coupling_type == 'monolithic_lagrange':
            for i in range(self.num_coupling_surf):
                lm_sq, lm_old_sq = allgather_vec(self.lm, self.comm), allgather_vec(self.lm_old, self.comm)
                self.pb0.c[i] = lm_sq[i]
                con = fem.assemble_scalar(self.cq_old_form[i])
                con = self.comm.allgather(con)
                self.constr[i] = sum(con)*self.cq_factor[i]
                self.constr_old[i] = sum(con)*self.cq_factor[i]

        # length of c from 3D-0D coupling
        self.pb0.len_c_3d0d = len(self.pb0.c)

        if bool(self.pb0.chamber_models):
            for i, ch in enumerate(['lv','rv','la','ra']):
                if self.pb0.chamber_models[ch]['type']=='0D_elast': self.pb0.y[i] = self.pbs.ti.timecurves(self.pb0.chamber_models[ch]['activation_curve'])(self.pbs.t_init)
                if self.pb0.chamber_models[ch]['type']=='0D_elast_prescr': self.pb0.y[i] = self.pbs.ti.timecurves(self.pb0.chamber_models[ch]['elastance_curve'])(self.pbs.t_init)
                if self.pb0.chamber_models[ch]['type']=='0D_prescr': self.pb0.c.append(self.pbs.ti.timecurves(self.pb0.chamber_models[ch]['prescribed_curve'])(self.pbs.t_init))

        # if we have prescribed variable values over time
        if self.restart_step==0: # we read s and s_old in case of restart
            if bool(self.pb0.prescribed_variables):
                for a in self.pb0.prescribed_variables:
                    varindex = self.pb0.cardvasc0D.varmap[a]
                    prescr = self.pb0.prescribed_variables[a]
                    prtype = list(prescr.keys())[0]
                    if prtype=='val':
                        val = prescr['val']
                    elif prtype=='curve':
                        curvenumber = prescr['curve']
                        val = self.pb0.ti.timecurves(curvenumber)(self.pb0.t_init)
                    else:
                        raise ValueError("Unknown type to prescribe a variable.")
                    self.pb0.s[varindex], self.pb0.s_old[varindex] = val, val

        # initially evaluate 0D model at old state
        self.pb0.cardvasc0D.evaluate(self.pb0.s_old, self.pbs.t_init, self.pb0.df_old, self.pb0.f_old, None, None, self.pb0.c, self.pb0.y, self.pb0.aux_old)
        self.pb0.auxTc_old[:] = self.pb0.aux_old[:]


    def write_output_ini(self):

        self.pbs.write_output_ini()


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t, N):

        self.pbs.evaluate_pre_solve(t, N)
        self.pb0.evaluate_pre_solve(t, N)


    def evaluate_post_solve(self, t, N):

        self.pbs.evaluate_post_solve(t, N)
        self.pb0.evaluate_post_solve(t, N)

        if self.have_multiscale_gandr:
            self.set_homeostatic_threshold(t), self.set_growth_trigger(t-t_off)


    def set_output_state(self, t):

        self.pbs.set_output_state(t)
        self.pb0.set_output_state(t)


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

        self.pbs.write_restart(sname, N)
        self.pb0.write_restart(sname, N)

        if self.coupling_type == 'monolithic_lagrange':
            if self.pbs.io.write_restart_every > 0 and N % self.pbs.io.write_restart_every == 0:
                self.pb0.cardvasc0D.write_restart(self.pb0.output_path_0D, sname+'_lm', N, self.lm)


    def check_abort(self, t):

        self.pb0.check_abort(t)


    def destroy(self):

        self.pbs.destroy()
        self.pb0.destroy()

        for i in range(len(self.col_ids)): self.k_us_vec[i].destroy()
        for i in range(len(self.row_ids)): self.k_su_vec[i].destroy()



class SolidmechanicsFlow0DSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlin.solver_nonlinear_ode([self.pb.pb0], self.solver_params['subsolver_params'])
        else:
            self.subsol = None

        self.evaluate_assemble_system_initial(subsolver=self.subsol)

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params, subsolver=self.subsol)

        if (self.pb.pbs.prestress_initial or self.pb.pbs.prestress_initial_only) and self.pb.pbs.restart_step == 0:
            # initialize solid mechanics solver
            solver_params_prestr = copy.deepcopy(self.solver_params)
            # modify solver parameters in case user specified alternating ones for prestressing (should do, because it's a 2x2 problem maximum)
            try: solver_params_prestr['solve_type'] = self.solver_params['solve_type_prestr']
            except: pass
            try: solver_params_prestr['block_precond'] = self.solver_params['block_precond_prestr']
            except: pass
            try: solver_params_prestr['precond_fields'] = self.solver_params['precond_fields_prestr']
            except: pass
            self.solverprestr = SolidmechanicsSolverPrestr(self.pb.pbs, solver_params_prestr)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if (self.pb.pbs.prestress_initial or self.pb.pbs.prestress_initial_only) and self.pb.pbs.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.destroy()

        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0 and not self.pb.restart_multiscale:
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
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbs.numstep, ni=ni, li=li, wt=wt)
