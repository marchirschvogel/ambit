#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
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
from .. import utilities, expression, ioparams
from ..mpiroutines import allgather_vec, allgather_mat

from ..fluid.fluid_main import FluidmechanicsProblem
from ..flow0d.flow0d_main import Flow0DProblem
from ..base import problem_base, solver_base


class FluidmechanicsFlow0DProblem(problem_base):

    def __init__(self, pbase, io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, mor_params={}, alevar={}):

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        self.problem_physics = 'fluid_flow0d'

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

        try: self.condense_0d_model = self.coupling_params['condense_0d_model']
        except: self.condense_0d_model = False

        self.have_condensed_variables = False
        if self.condense_0d_model:
            self.have_condensed_variables = True

        # only option in fluid mechanics!
        self.coupling_type = 'monolithic_lagrange'

        # assert that we do not have conflicting timings
        time_params_flow0d['maxtime'] = time_params_fluid['maxtime']
        time_params_flow0d['numstep'] = time_params_fluid['numstep']

        # initialize problem instances (also sets the variational forms for the fluid problem)
        self.pbf = FluidmechanicsProblem(pbase, io_params, time_params_fluid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, alevar=alevar)
        self.pb0 = Flow0DProblem(pbase, io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params)

        self.pbrom = self.pbf # ROM problem can only be fluid
        self.pbrom_host = self

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.set_variational_forms()

        self.numdof = self.pbf.numdof + self.LM.getSize()

        self.localsolve = self.pbf.localsolve

        self.sub_solve = True
        self.print_enhanced_info = self.pbf.io.print_enhanced_info

        # 3D fluxes
        self.constr, self.constr_old = [[]]*self.num_coupling_surf, [[]]*self.num_coupling_surf

        # set 3D-0D coupling array
        self.offc = 0
        if bool(self.pb0.chamber_models):
            if self.pb0.chamber_models['lv']['type']=='3D_fluid' and self.pb0.chamber_models['lv']['num_outflows']==0 and self.pb0.cardvasc0D.cormodel:
                self.pb0.auxdata['p'], self.pb0.auxdata_old['p'] = {-1:0}, {-1:0} # dummy entries
                self.offc = 1

        # re-define coupling array
        self.pb0.c = [[]]*(self.num_coupling_surf+self.offc)

        # number of fields involved
        if not self.condense_0d_model:
            self.nfields = 3
        else:
            self.nfields = 2

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if not self.condense_0d_model:
            if self.pbf.num_dupl > 1: is_ghosted = [1, 2, 0]
            else:                     is_ghosted = [1, 1, 0]
            varlist = [self.pbf.v.vector, self.pbf.p.vector, self.LM]
        else:
            if self.pbf.num_dupl > 1: is_ghosted = [1, 2]
            else:                     is_ghosted = [1, 1]
            varlist = [self.pbf.v.vector, self.pbf.p.vector]

        return varlist, is_ghosted


    # defines the monolithic coupling forms for 0D flow and fluid mechanics
    def set_variational_forms(self):

        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        self.coupfuncs, self.coupfuncs_old, self.coupfuncs_mid = [], [], []

        # Lagrange multipliers
        self.LM, self.LM_old = PETSc.Vec().createMPI(size=self.num_coupling_surf), PETSc.Vec().createMPI(size=self.num_coupling_surf)

        self.power_coupling, self.power_coupling_old, self.power_coupling_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):

            self.pr0D = expression.template()

            self.coupfuncs.append(fem.Function(self.pbf.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbf.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)
            self.coupfuncs_mid.append(self.pbf.timefac * self.coupfuncs[-1] + (1.-self.pbf.timefac) * self.coupfuncs_old[-1])

            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):

                ds_vq = self.pbf.io.ds(self.surface_vq_ids[n][i])
                cq_ += self.pbf.vf.flux(self.pbf.v, ds_vq, w=self.pbf.alevar['w'], F=self.pbf.alevar['Fale'])
                cq_old_ += self.pbf.vf.flux(self.pbf.v_old, ds_vq, w=self.pbf.alevar['w_old'], F=self.pbf.alevar['Fale_old'])

            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbf.v, self.pbf.dv))

            df_, df_mid_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):

                ds_p = self.pbf.io.ds(self.surface_p_ids[n][i])
                df_ += self.pbf.timefac*self.pbf.vf.flux(self.pbf.var_v, ds_p, w=ufl.constantvalue.zero(self.pbf.ki.dim), F=self.pbf.alevar['Fale'])
                df_mid_ += self.pbf.timefac*self.pbf.vf.flux(self.pbf.var_v, ds_p, w=ufl.constantvalue.zero(self.pbf.ki.dim), F=self.pbf.alevar['Fale_mid'])

                # add to fluid rhs contributions
                self.power_coupling += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs[-1], ds_p, F=self.pbf.alevar['Fale'])
                self.power_coupling_old += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs_old[-1], ds_p, F=self.pbf.alevar['Fale_old'])
                self.power_coupling_mid += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs_mid[-1], ds_p, F=self.pbf.alevar['Fale_mid'])

            if self.pbf.ti.eval_nonlin_terms=='trapezoidal': self.dforce.append(df_)
            if self.pbf.ti.eval_nonlin_terms=='midpoint': self.dforce.append(df_mid_)

        if self.pbf.ti.eval_nonlin_terms=='trapezoidal':
            # minus sign, since contribution to external power!
            self.pbf.weakform_v += -self.pbf.timefac * self.power_coupling - (1.-self.pbf.timefac) * self.power_coupling_old
            # add to fluid Jacobian
            self.pbf.weakform_lin_vv += -self.pbf.timefac * ufl.derivative(self.power_coupling, self.pbf.v, self.pbf.dv)
        if self.pbf.ti.eval_nonlin_terms=='midpoint':
            # minus sign, since contribution to external power!
            self.pbf.weakform_v += -self.power_coupling_mid
            # add to fluid Jacobian
            self.pbf.weakform_lin_vv += -ufl.derivative(self.power_coupling_mid, self.pbf.v, self.pbf.dv)

        # old Lagrange multipliers - initialize with initial pressures
        if self.pbase.restart_step==0:
            self.pb0.cardvasc0D.initialize_lm(self.LM, self.pb0.initialconditions)
            self.pb0.cardvasc0D.initialize_lm(self.LM_old, self.pb0.initialconditions)


    def set_problem_residual_jacobian_forms(self, pre=False):

        self.pbf.set_problem_residual_jacobian_forms(pre=pre)
        self.set_problem_residual_jacobian_forms_coupling()


    def set_problem_residual_jacobian_forms_coupling(self):

        ts = time.time()
        utilities.print_status("FEM form compilation for fluid-0D coupling...", self.comm, e=" ")

        self.cq_form, self.cq_old_form, self.dcq_form, self.dforce_form = [], [], [], []

        for i in range(self.num_coupling_surf):
            if self.pbf.io.USE_MIXED_DOLFINX_BRANCH or self.pbf.io.USE_NEW_DOLFINX:
                self.cq_form.append(fem.form(self.cq[i], entity_maps=self.pbf.io.entity_maps))
                self.cq_old_form.append(fem.form(self.cq_old[i], entity_maps=self.pbf.io.entity_maps))

                self.dcq_form.append(fem.form(self.cq_factor[i]*self.dcq[i], entity_maps=self.pbf.io.entity_maps))
                self.dforce_form.append(fem.form(self.dforce[i], entity_maps=self.pbf.io.entity_maps))
            else:
                self.cq_form.append(fem.form(self.cq[i]))
                self.cq_old_form.append(fem.form(self.cq_old[i]))

                self.dcq_form.append(fem.form(self.cq_factor[i]*self.dcq[i]))
                self.dforce_form.append(fem.form(self.dforce[i]))

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)


    def set_problem_vector_matrix_structures(self):

        self.pbf.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()


    def set_problem_vector_matrix_structures_coupling(self):

        self.r_lm = PETSc.Vec().createMPI(size=self.num_coupling_surf)

        # Lagrange multiplier stiffness matrix (currently treated with FD!)
        self.K_lm = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_lm.setUp()
        self.row_ids = list(range(self.num_coupling_surf))
        self.col_ids = list(range(self.num_coupling_surf))

        # setup offdiagonal matrices
        locmatsize = self.pbf.V_v.dofmap.index_map.size_local * self.pbf.V_v.dofmap.index_map_bs
        matsize = self.pbf.V_v.dofmap.index_map.size_global * self.pbf.V_v.dofmap.index_map_bs

        self.k_vs_vec = []
        for i in range(len(self.col_ids)):
            self.k_vs_vec.append(fem.petsc.create_vector(self.dforce_form[i]))

        self.k_sv_vec = []
        for i in range(len(self.row_ids)):
            self.k_sv_vec.append(fem.petsc.create_vector(self.dcq_form[i]))

        self.dofs_coupling_vq, self.dofs_coupling_p = [[]]*self.num_coupling_surf, [[]]*self.num_coupling_surf

        self.k_vs_subvec, self.k_sv_subvec, sze_vs, sze_sv = [], [], [], []

        for n in range(self.num_coupling_surf):

            nds_vq_local = fem.locate_dofs_topological(self.pbf.V_v, self.pbf.io.mesh.topology.dim-1, self.pbf.io.mt_b1.indices[np.isin(self.pbf.io.mt_b1.values, self.surface_vq_ids[n])])
            nds_vq = np.array( self.pbf.V_v.dofmap.index_map.local_to_global(np.asarray(nds_vq_local, dtype=np.int32)), dtype=np.int32 )
            self.dofs_coupling_vq[n] = PETSc.IS().createBlock(self.pbf.V_v.dofmap.index_map_bs, nds_vq, comm=self.comm)

            self.k_sv_subvec.append( self.k_sv_vec[n].getSubVector(self.dofs_coupling_vq[n]) )

            sze_sv.append(self.k_sv_subvec[-1].getSize())

            nds_p_local = fem.locate_dofs_topological(self.pbf.V_v, self.pbf.io.mesh.topology.dim-1, self.pbf.io.mt_b1.indices[np.isin(self.pbf.io.mt_b1.values, self.surface_p_ids[n])])
            nds_p = np.array( self.pbf.V_v.dofmap.index_map.local_to_global(np.asarray(nds_p_local, dtype=np.int32)), dtype=np.int32 )
            self.dofs_coupling_p[n] = PETSc.IS().createBlock(self.pbf.V_v.dofmap.index_map_bs, nds_p, comm=self.comm)

            self.k_vs_subvec.append( self.k_vs_vec[n].getSubVector(self.dofs_coupling_p[n]) )

            sze_vs.append(self.k_vs_subvec[-1].getSize())

        # derivative of fluid residual w.r.t. multipliers
        self.K_vs = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(PETSc.DECIDE,self.num_coupling_surf)), bsize=None, nnz=self.num_coupling_surf, csr=None, comm=self.comm)
        self.K_vs.setUp()

        # derivative of multiplier constraints w.r.t. fluid velocities
        self.K_sv = PETSc.Mat().createAIJ(size=((PETSc.DECIDE,self.num_coupling_surf),(locmatsize,matsize)), bsize=None, nnz=max(sze_sv), csr=None, comm=self.comm)
        self.K_sv.setUp()
        self.K_sv.setOption(PETSc.Mat.Option.ROW_ORIENTED, False)

        # In case we might want to condense the 0D model into the fluid momentum residual block. This implementation is mainly for
        # testing and comparison purposes (to other approaches which do that...).
        # This will eliminate the multiplier from the system, hence the fluid momentum and Jacobian need to be updated accordingly.
        # If iterative solvers are used, this approach may severely compromise the sparsity pattern and break the solver performance!
        # Also, it is in genetral difficult to find a reason why this should be done (write me if you have one).
        if self.condense_0d_model:

            # inverse of LM stiffness matrix
            self.K_lm_inv = PETSc.Mat().createAIJ(size=(self.num_coupling_surf,self.num_coupling_surf), bsize=None, nnz=self.num_coupling_surf*self.num_coupling_surf, csr=None, comm=self.comm)
            self.K_lm_inv.setUp()

            # need to set K_vs here to get correct sparsity patterns for mat-mat products
            for i in range(len(self.col_ids)):
                # NOTE: only set the surface-subset of the k_vs vector entries to avoid placing unnecessary zeros!
                self.k_vs_vec[i].getSubVector(self.dofs_coupling_p[i], subvec=self.k_vs_subvec[i])
                self.K_vs.setValues(self.dofs_coupling_p[i], self.col_ids[i], self.k_vs_subvec[i].array, addv=PETSc.InsertMode.INSERT)
                self.k_vs_vec[i].restoreSubVector(self.dofs_coupling_p[i], subvec=self.k_vs_subvec[i])
            self.K_vs.assemble()

            # need to set K_sv here to get correct sparsity patterns for mat-mat products
            for i in range(len(self.row_ids)):
                # NOTE: only set the surface-subset of the k_sv vector entries to avoid placing unnecessary zeros!
                self.k_sv_vec[i].getSubVector(self.dofs_coupling_vq[i], subvec=self.k_sv_subvec[i])
                self.K_sv.setValues(self.row_ids[i], self.dofs_coupling_vq[i], self.k_sv_subvec[i].array, addv=PETSc.InsertMode.INSERT)
                self.k_sv_vec[i].restoreSubVector(self.dofs_coupling_vq[i], subvec=self.k_sv_subvec[i])
            self.K_sv.assemble()

            # set 1's to get correct allocation pattern
            islm = PETSc.IS().createStride(self.num_coupling_surf, first=0, step=1, comm=self.comm)
            self.K_lm_inv.setValuesBlocked(islm, islm, np.ones((self.num_coupling_surf,self.num_coupling_surf), dtype=np.int32), addv=PETSc.InsertMode.INSERT)
            self.K_lm_inv.assemble()
            # numpy array for serial inverse
            self.K_lm_array_inv = np.zeros((self.num_coupling_surf,self.num_coupling_surf))

            self.Kvs_Klminv = self.K_vs.matMult(self.K_lm_inv)
            self.Kvs_Klminv_Ksv = self.Kvs_Klminv.matMult(self.K_sv)

            self.r_v_kcondens = self.Kvs_Klminv.createVecLeft()
            self.ksv_v = self.K_lm.createVecLeft()

            self.r_lm_kcondens = self.K_lm.createVecLeft()


    def assemble_residual(self, t, subsolver=None):

        for i in range(self.num_coupling_surf):
            cq = fem.assemble_scalar(self.cq_form[i])
            cq = self.comm.allgather(cq)
            self.constr[i] = sum(cq)*self.cq_factor[i]

        # Lagrange multipliers (pressures) to be passed to 0D model
        LM_sq = allgather_vec(self.LM, self.comm)
        for i in range(self.num_coupling_surf):
            self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = LM_sq[i]

        # point auxdata dict to dict of integral evaluations (fluxes, pressures) in case needed by 0D
        self.pb0.auxdata['q'], self.pb0.auxdata['p'] = self.pbf.qv_, self.pbf.pu_

        # special case: append upstream pressure to coupling array in case we don't have an LM, but a monitored pressure value
        if bool(self.pb0.chamber_models):
            if self.pb0.chamber_models['lv']['type']=='3D_fluid' and self.pb0.chamber_models['lv']['num_outflows']==0 and self.pb0.cardvasc0D.cormodel:
                dp_id = self.pb0.chamber_models['lv']['dp_monitor_id']
                self.pb0.c[0] = self.pb0.auxdata['p'][dp_id]

        if subsolver is not None:
            # only have rank 0 solve the ODE, then broadcast solution
            err = -1
            if self.comm.rank==0:
                err = subsolver.newton(t, print_iter=self.print_subiter, sub=True)
            self.comm.Barrier()
            # need to broadcast to all cores
            err = self.comm.bcast(err, root=0)
            if err>0: subsolver.solver_error(self)
            self.pb0.s.array[:] = self.comm.bcast(self.pb0.s.array, root=0)
            self.pb0.df.array[:] = self.comm.bcast(self.pb0.df.array, root=0)
            self.pb0.f.array[:] = self.comm.bcast(self.pb0.f.array, root=0)
            self.pb0.aux[:] = self.comm.bcast(self.pb0.aux, root=0)

        # add to fluid momentum equation
        self.pb0.cardvasc0D.set_pressure_fem(self.LM, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs)

        # fluid main blocks
        self.pbf.assemble_residual(t)

        self.r_list[0] = self.pbf.r_list[0]
        self.r_list[1] = self.pbf.r_list[1]

        ls, le = self.LM.getOwnershipRange()

        # Lagrange multiplier coupling residual
        for i in range(ls,le):
            self.r_lm[i] = self.constr[i] - self.pb0.s[self.pb0.cardvasc0D.v_ids[i]]

        self.r_lm.assemble()

        if self.condense_0d_model:
            # call those parts of assemble_stiffness_3d0d that are needed for forming the residual
            self.assemble_stiffness_3d0d(t, LM_sq, subsolver=subsolver, condensed_res_action=True)
            self.Kvs_Klminv.mult(self.r_lm, self.r_v_kcondens)
            self.r_list[0].axpy(-1., self.r_v_kcondens)
        else:
            self.r_list[2] = self.r_lm

        del LM_sq


    def assemble_stiffness(self, t, subsolver=None):

        # Lagrange multipliers (pressures) to be passed to 0D model
        LM_sq = allgather_vec(self.LM, self.comm)

        for i in range(self.num_coupling_surf):
            self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = LM_sq[i]

        # point auxdata dict to dict of integral evaluations (fluxes, pressures) in case needed by 0D
        self.pb0.auxdata['q'], self.pb0.auxdata['p'] = self.pbf.qv_, self.pbf.pu_

        # special case: append upstream pressure to coupling array in case we don't have an LM, but a monitored pressure value
        if bool(self.pb0.chamber_models):
            if self.pb0.chamber_models['lv']['type']=='3D_fluid' and self.pb0.chamber_models['lv']['num_outflows']==0 and self.pb0.cardvasc0D.cormodel:
                dp_id = self.pb0.chamber_models['lv']['dp_monitor_id']
                self.pb0.c[0] = self.pb0.auxdata['p'][dp_id]

        # fluid main blocks
        self.pbf.assemble_stiffness(t)

        self.K_list[0][0] = self.pbf.K_list[0][0]
        self.K_list[0][1] = self.pbf.K_list[0][1]
        self.K_list[1][0] = self.pbf.K_list[1][0]
        self.K_list[1][1] = self.pbf.K_list[1][1] # should be only non-zero if we have stabilization...

        self.assemble_stiffness_3d0d(t, LM_sq, subsolver=subsolver)


    def assemble_stiffness_3d0d(self, t, LM_sq, subsolver=None, condensed_res_action=False):

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

        ls, le = self.K_lm.getOwnershipRange()

        # finite differencing for LM siffness matrix
        if subsolver is not None:
            for i in range(ls, le): # row-owning rank calls the ODE solver
                for j in range(self.num_coupling_surf):
                    self.pb0.c[self.pb0.cardvasc0D.c_ids[j]] = LM_sq[j] + self.eps_fd # perturbed LM
                    subsolver.newton(t, print_iter=False, sub=True)
                    val = -(self.pb0.s[self.pb0.cardvasc0D.v_ids[i]] - self.pb0.s_tmp[self.pb0.cardvasc0D.v_ids[i]])/self.eps_fd
                    self.K_lm.setValue(i, j, val, addv=PETSc.InsertMode.INSERT)
                    self.pb0.c[self.pb0.cardvasc0D.c_ids[j]] = LM_sq[j] # restore LM

        self.comm.Barrier() # do we need this here, since not all processes participate in the ODE solve?

        # restore df, f, and aux vectors for correct time step update
        self.pb0.df.axpby(1.0, 0.0, self.pb0.df_tmp)
        self.pb0.f.axpby(1.0, 0.0, self.pb0.f_tmp)
        self.pb0.aux[:] = self.pb0.aux_tmp[:]
        # restore 0D state variable
        self.pb0.s.axpby(1.0, 0.0, self.pb0.s_tmp)

        self.K_lm.assemble()

        if not self.condense_0d_model:
            self.K_list[2][2] = self.K_lm

        del LM_sq

        if not condensed_res_action:
            # offdiagonal s-v rows
            for i in range(len(self.row_ids)):
                with self.k_sv_vec[i].localForm() as r_local: r_local.set(0.0)
                fem.petsc.assemble_vector(self.k_sv_vec[i], self.dcq_form[i])
                self.k_sv_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # offdiagonal v-s columns
        for i in range(len(self.col_ids)):
            with self.k_vs_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_vs_vec[i], self.dforce_form[i]) # already multiplied by time-integration factor
            self.k_vs_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # set zeros at DBC entries
            fem.set_bc(self.k_vs_vec[i], self.pbf.bc.dbcs, x0=self.pbf.v.vector, scale=0.0)

        # set columns
        for i in range(len(self.col_ids)):
            # NOTE: only set the surface-subset of the k_vs vector entries to avoid placing unnecessary zeros!
            self.k_vs_vec[i].getSubVector(self.dofs_coupling_p[i], subvec=self.k_vs_subvec[i])
            self.K_vs.setValues(self.dofs_coupling_p[i], self.col_ids[i], self.k_vs_subvec[i].array, addv=PETSc.InsertMode.INSERT)
            self.k_vs_vec[i].restoreSubVector(self.dofs_coupling_p[i], subvec=self.k_vs_subvec[i])

        self.K_vs.assemble()

        if not condensed_res_action:
            # set rows
            for i in range(len(self.row_ids)):
                # NOTE: only set the surface-subset of the k_sv vector entries to avoid placing unnecessary zeros!
                self.k_sv_vec[i].getSubVector(self.dofs_coupling_vq[i], subvec=self.k_sv_subvec[i])
                self.K_sv.setValues(self.row_ids[i], self.dofs_coupling_vq[i], self.k_sv_subvec[i].array, addv=PETSc.InsertMode.INSERT)
                self.k_sv_vec[i].restoreSubVector(self.dofs_coupling_vq[i], subvec=self.k_sv_subvec[i])

            self.K_sv.assemble()

        if self.condense_0d_model:

            if subsolver is not None:
                # gather matrix and do a serial inverse with numpy (matrix is always super small!)
                K_lm_array = allgather_mat(self.K_lm, self.comm)
                # let only rank 0 do the inverse, then broadcast
                if self.comm.rank==0:
                    self.K_lm_array_inv[:] = np.linalg.inv(K_lm_array)
                self.comm.Barrier()
                self.K_lm_array_inv[:] = self.comm.bcast(self.K_lm_array_inv, root=0)

                # now set back to parallel K_lm_inv matrix (for efficient multiplications later on)
                for i in range(ls, le):
                    for j in range(self.num_coupling_surf):
                        self.K_lm_inv.setValue(i, j, self.K_lm_array_inv[i,j], addv=PETSc.InsertMode.INSERT)

                self.K_lm_inv.assemble()

                del K_lm_array

            self.K_vs.matMult(self.K_lm_inv, result=self.Kvs_Klminv)

            if not condensed_res_action:
                self.Kvs_Klminv.matMult(self.K_sv, result=self.Kvs_Klminv_Ksv)
                self.K_list[0][0].axpy(-1., self.Kvs_Klminv_Ksv)

        else:
            self.K_list[0][2] = self.K_vs
            self.K_list[2][0] = self.K_sv


    def get_index_sets(self, isoptions={}):

        if self.condense_0d_model:
            return self.pbf.get_index_sets(isoptions=isoptions)

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.vector.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.vector.getLocalSize()

        offset_v = vvec_or0 + self.pbf.p.vector.getOwnershipRange()[0] + self.LM.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.pbf.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_s = offset_p + self.pbf.p.vector.getLocalSize()
        iset_s = PETSc.IS().createStride(self.LM.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_s = iset_s.expand(iset_r) # add to 0D block
            iset_s.sort() # should be sorted, otherwise PETSc may struggle to extract block

        if isoptions['lms_to_p']:
            iset_p = iset_p.expand(iset_s) # add to pressure block
            ilist = [iset_v, iset_p]
        elif isoptions['lms_to_v']:
            iset_v = iset_v.expand(iset_s) # add to velocity block (could be bad...)
            ilist = [iset_v, iset_p]
        else:
            ilist = [iset_v, iset_p, iset_s]

        return ilist


    def update_condensed_vars(self, del_x):

        # compute LM
        self.K_sv.mult(del_x[0], self.ksv_v)

        self.r_lm_kcondens.axpby(-1., 0., self.r_lm)
        self.r_lm_kcondens.axpy(-1., self.ksv_v)

        self.K_lm_inv.multAdd(self.r_lm_kcondens, self.LM, self.LM)


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # fluid + flow0d problem
        self.pbf.read_restart(sname, N)
        self.pb0.read_restart(sname, N)

        if N > 0:
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.LM)
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.LM_old)


    def evaluate_initial(self):

        self.pbf.evaluate_initial()

        self.pb0.cardvasc0D.set_pressure_fem(self.LM_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)

        # special case: append upstream pressure to coupling array in case we don't have an LM, but a monitored pressure value
        if bool(self.pb0.chamber_models):
            if self.pb0.chamber_models['lv']['type']=='3D_fluid' and self.pb0.chamber_models['lv']['num_outflows']==0 and self.pb0.cardvasc0D.cormodel:
                if self.pbase.restart_step==0:
                    self.pb0.auxdata_old['p'] = copy.deepcopy(self.pbf.pu_old_) # copy since we write restart and update auxdata_old differently
                # for k in self.pb0.auxdata_old['p']: self.pb0.auxdata['p'][k] = self.pb0.auxdata_old['p'][k]
                dp_id = self.pb0.chamber_models['lv']['dp_monitor_id']
                self.pb0.c[0] = self.pb0.auxdata_old['p'][dp_id]

        for i in range(self.num_coupling_surf):
            LM_sq, lm_old_sq = allgather_vec(self.LM, self.comm), allgather_vec(self.LM_old, self.comm)
            self.pb0.c[i+self.offc] = LM_sq[i]
            con = fem.assemble_scalar(self.cq_old_form[i])
            con = self.comm.allgather(con)
            self.constr[i] = sum(con)*self.cq_factor[i]
            self.constr_old[i] = sum(con)*self.cq_factor[i]

        # length of c from 3D-0D coupling
        self.pb0.len_c_3d0d = len(self.pb0.c)

        if bool(self.pb0.chamber_models):
            for i, ch in enumerate(['lv','rv','la','ra']):
                if self.pb0.chamber_models[ch]['type']=='0D_elast': self.pb0.y[i] = self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['activation_curve'])(self.pbase.t_init)
                if self.pb0.chamber_models[ch]['type']=='0D_elast_prescr': self.pb0.y[i] = self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['elastance_curve'])(self.pbase.t_init)
                if self.pb0.chamber_models[ch]['type']=='0D_prescr': self.pb0.c.append(self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['prescribed_curve'])(self.pbase.t_init))

        # if we have prescribed variable values over time
        if self.pbase.restart_step==0: # we read s and s_old in case of restart
            if bool(self.pb0.prescribed_variables):
                for a in self.pb0.prescribed_variables:
                    varindex = self.pb0.cardvasc0D.varmap[a]
                    prescr = self.pb0.prescribed_variables[a]
                    prtype = list(prescr.keys())[0]
                    if prtype=='val':
                        val = prescr['val']
                    elif prtype=='curve':
                        curvenumber = prescr['curve']
                        val = self.pb0.ti.timecurves(curvenumber)(self.pbase.t_init)
                    elif prtype=='flux_monitor':
                        monid = prescr['flux_monitor']
                        val = self.pbf.qv_old_[monid]
                    else:
                        raise ValueError("Unknown type to prescribe a variable.")
                    self.pb0.s[varindex], self.pb0.s_old[varindex] = val, val

        # initially evaluate 0D model at old state
        self.pb0.cardvasc0D.evaluate(self.pb0.s_old, self.pbase.t_init, self.pb0.df_old, self.pb0.f_old, None, None, self.pb0.c, self.pb0.y, self.pb0.aux_old)
        self.pb0.auxTc_old[:] = self.pb0.aux_old[:]


    def write_output_ini(self):

        self.pbf.write_output_ini()


    def write_output_pre(self):

        self.pbf.write_output_pre()
        self.pb0.write_output_pre()


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t, N, dt):

        self.pbf.evaluate_pre_solve(t, N, dt)
        self.pb0.evaluate_pre_solve(t, N, dt)


    def evaluate_post_solve(self, t, N):

        self.pbf.evaluate_post_solve(t, N)
        self.pb0.evaluate_post_solve(t, N)


    def set_output_state(self, N):

        self.pbf.set_output_state(N)
        self.pb0.set_output_state(N)


    def write_output(self, N, t, mesh=False):

        self.pbf.write_output(N, t)
        self.pb0.write_output(N, t)


    def update(self):

        # update time step - fluid and 0D model
        self.pbf.update()
        self.pb0.update()

        # update old LMs
        self.LM_old.axpby(1.0, 0.0, self.LM)
        self.pb0.cardvasc0D.set_pressure_fem(self.LM_old, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs_old)
        # update old 3D fluxes
        self.constr_old[:] = self.constr[:]


    def print_to_screen(self):

        self.pbf.print_to_screen()
        self.pb0.print_to_screen()


    def induce_state_change(self):

        self.pbf.induce_state_change()
        self.pb0.induce_state_change()


    def write_restart(self, sname, N, force=False):

        self.pbf.write_restart(sname, N, force=force)
        self.pb0.write_restart(sname, N, force=force)

        if (self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0) or force:
            LM_sq = allgather_vec(self.LM, self.comm)
            if self.comm.rank == 0:
                f = open(self.pb0.output_path_0D+'/checkpoint_'+sname+'_lm_'+str(N)+'.txt', 'wt')
                for i in range(len(LM_sq)):
                    f.write('%.16E\n' % (LM_sq[i]))
                f.close()
            del LM_sq


    def check_abort(self, t):

        return self.pb0.check_abort(t)


    def destroy(self):

        self.pbf.destroy()
        self.pb0.destroy()

        for i in range(len(self.col_ids)): self.k_vs_vec[i].destroy()
        for i in range(len(self.row_ids)): self.k_sv_vec[i].destroy()



class FluidmechanicsFlow0DSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbf.pre)
        self.pb.set_problem_vector_matrix_structures()

        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlin.solver_nonlinear_ode([self.pb.pb0], self.solver_params['subsolver_params'])
        else:
            self.subsol = None

        self.evaluate_assemble_system_initial(subsolver=self.subsol)

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params, subsolver=self.subsol)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old - self.pb.power_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.pbf.io.USE_MIXED_DOLFINX_BRANCH or self.pb.pbf.io.USE_NEW_DOLFINX:
                res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.pbf.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.pbf.io.entity_maps)
            else:
                res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbase.numstep, ni=ni, li=li, wt=wt)
