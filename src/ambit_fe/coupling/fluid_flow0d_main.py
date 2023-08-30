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
from ..mpiroutines import allgather_vec

from ..fluid.fluid_main import FluidmechanicsProblem
from ..flow0d.flow0d_main import Flow0DProblem
from ..base import problem_base, solver_base


class FluidmechanicsFlow0DProblem(problem_base):

    def __init__(self, io_params, time_params_fluid, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, mor_params={}, comm=None, alevar={}):
        super().__init__(io_params, time_params_fluid, comm)

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

        # only option in fluid mechanics!
        self.coupling_type = 'monolithic_lagrange'

        # assert that we do not have conflicting timings
        time_params_flow0d['maxtime'] = time_params_fluid['maxtime']
        time_params_flow0d['numstep'] = time_params_fluid['numstep']

        # initialize problem instances (also sets the variational forms for the fluid problem)
        self.pbf = FluidmechanicsProblem(io_params, time_params_fluid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, comm=self.comm, alevar=alevar)
        self.pb0 = Flow0DProblem(io_params, time_params_flow0d, model_params_flow0d, time_curves, coupling_params, comm=self.comm)

        self.pbrom = self.pbf # ROM problem can only be fluid

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.set_variational_forms()

        self.numdof = self.pbf.numdof + self.lm.getSize()

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
        self.nfields = 3

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbf.num_dupl > 1: is_ghosted = [1, 2, 0]
        else:                     is_ghosted = [1, 1, 0]
        return [self.pbf.v.vector, self.pbf.p.vector, self.lm], is_ghosted


    # defines the monolithic coupling forms for 0D flow and fluid mechanics
    def set_variational_forms(self):

        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        self.coupfuncs, self.coupfuncs_old = [], []

        # Lagrange multipliers
        self.lm, self.lm_old = PETSc.Vec().createMPI(size=self.num_coupling_surf), PETSc.Vec().createMPI(size=self.num_coupling_surf)

        self.power_coupling, self.power_coupling_old = ufl.as_ufl(0), ufl.as_ufl(0)

        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):

            self.pr0D = expression.template()

            self.coupfuncs.append(fem.Function(self.pbf.Vd_scalar)), self.coupfuncs_old.append(fem.Function(self.pbf.Vd_scalar))
            self.coupfuncs[-1].interpolate(self.pr0D.evaluate), self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate)

            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):

                ds_vq = ufl.ds(subdomain_data=self.pbf.io.mt_b1, subdomain_id=self.surface_vq_ids[n][i], metadata={'quadrature_degree': self.pbf.quad_degree})
                cq_ += self.pbf.vf.flux(self.pbf.v, ds_vq, w=self.pbf.alevar['w'], Fale=self.pbf.alevar['Fale'])
                cq_old_ += self.pbf.vf.flux(self.pbf.v_old, ds_vq, w=self.pbf.alevar['w_old'], Fale=self.pbf.alevar['Fale_old'])

            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbf.v, self.pbf.dv))

            df_ = ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):

                ds_p = ufl.ds(subdomain_data=self.pbf.io.mt_b1, subdomain_id=self.surface_p_ids[n][i], metadata={'quadrature_degree': self.pbf.quad_degree})
                df_ += self.pbf.timefac*self.pbf.vf.flux(self.pbf.var_v, ds_p, w=ufl.constantvalue.zero(self.pbf.ki.dim), Fale=self.pbf.alevar['Fale'])

                # add to fluid rhs contributions
                self.power_coupling += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs[-1], ds_p, Fale=self.pbf.alevar['Fale'])
                self.power_coupling_old += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs_old[-1], ds_p, Fale=self.pbf.alevar['Fale_old'])

            self.dforce.append(df_)

        # minus sign, since contribution to external power!
        self.pbf.weakform_v += -self.pbf.timefac * self.power_coupling - (1.-self.pbf.timefac) * self.power_coupling_old

        # add to fluid Jacobian
        self.pbf.weakform_lin_vv += -self.pbf.timefac * ufl.derivative(self.power_coupling, self.pbf.v, self.pbf.dv)

        # old Lagrange multipliers - initialize with initial pressures
        if self.pbf.restart_step==0:
            self.pb0.cardvasc0D.initialize_lm(self.lm, self.pb0.initialconditions)
            self.pb0.cardvasc0D.initialize_lm(self.lm_old, self.pb0.initialconditions)


    def set_problem_residual_jacobian_forms(self):

        self.pbf.set_problem_residual_jacobian_forms()
        self.set_problem_residual_jacobian_forms_coupling()


    def set_problem_residual_jacobian_forms_coupling(self):

        tes = time.time()
        if self.comm.rank == 0:
            print('FEM form compilation for fluid-0D coupling...')
            sys.stdout.flush()

        self.cq_form, self.cq_old_form, self.dcq_form, self.dforce_form = [], [], [], []

        for i in range(self.num_coupling_surf):
            self.cq_form.append(fem.form(self.cq[i]))
            self.cq_old_form.append(fem.form(self.cq_old[i]))

            self.dcq_form.append(fem.form(self.cq_factor[i]*self.dcq[i]))
            self.dforce_form.append(fem.form(self.dforce[i]))

        tee = time.time() - tes
        if self.comm.rank == 0:
            print('FEM form compilation for fluid-0D finished, te = %.2f s' % (tee))
            sys.stdout.flush()


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

        # derivative of solid residual w.r.t. 0D pressures
        self.k_vs_vec = []
        for i in range(len(self.col_ids)):
            self.k_vs_vec.append(fem.petsc.create_vector(self.dforce_form[i]))

        self.K_vs = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(self.num_coupling_surf)), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_vs.setUp()

        self.k_sv_vec = []
        for i in range(len(self.row_ids)):
            self.k_sv_vec.append(fem.petsc.create_vector(self.dcq_form[i]))

        # derivative of 0D residual w.r.t. solid displacements
        self.K_sv = PETSc.Mat().createAIJ(size=((self.num_coupling_surf),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_sv.setUp()


    def assemble_residual(self, t, subsolver=None):

        for i in range(self.num_coupling_surf):
            cq = fem.assemble_scalar(self.cq_form[i])
            cq = self.comm.allgather(cq)
            self.constr[i] = sum(cq)*self.cq_factor[i]

        # Lagrange multipliers (pressures) to be passed to 0D model
        lm_sq = allgather_vec(self.lm, self.comm)
        for i in range(self.num_coupling_surf):
            self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = lm_sq[i]

        # point auxdata dict to dict of integral evaluations (fluxes, pressures) in case needed by 0D
        self.pb0.auxdata['q'], self.pb0.auxdata['p'] = self.pbf.qv_, self.pbf.pu_

        # special case: append upstream pressure to coupling array in case we don't have an LM, but a monitored pressure value
        if bool(self.pb0.chamber_models):
            if self.pb0.chamber_models['lv']['type']=='3D_fluid' and self.pb0.chamber_models['lv']['num_outflows']==0 and self.pb0.cardvasc0D.cormodel:
                dp_id = self.pb0.chamber_models['lv']['dp_monitor_id']
                self.pb0.c[0] = self.pb0.auxdata['p'][dp_id]

        if subsolver is not None:
            subsolver.newton(t, print_iter=self.print_subiter, sub=True)

        # add to fluid momentum equation
        self.pb0.cardvasc0D.set_pressure_fem(self.lm, list(range(self.num_coupling_surf)), self.pr0D, self.coupfuncs)

        # fluid main blocks
        self.pbf.assemble_residual(t)

        self.r_list[0] = self.pbf.r_list[0]
        self.r_list[1] = self.pbf.r_list[1]

        s_sq = allgather_vec(self.pb0.s, self.comm)

        ls, le = self.lm.getOwnershipRange()

        # Lagrange multiplier coupling residual
        for i in range(ls,le):
            self.r_lm[i] = self.constr[i] - s_sq[self.pb0.cardvasc0D.v_ids[i]]

        self.r_lm.assemble()

        self.r_list[2] = self.r_lm

        del lm_sq, s_sq

        if bool(self.residual_scale):
            self.scale_residual_list([self.r_lm], [self.residual_scale[2]])


    def assemble_stiffness(self, t, subsolver=None):

        K_list = [[None]*3 for _ in range(3)]

        # Lagrange multipliers (pressures) to be passed to 0D model
        lm_sq = allgather_vec(self.lm, self.comm)

        for i in range(self.num_coupling_surf):
            self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = lm_sq[i]

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

        self.K_list[2][2] = self.K_lm

        del lm_sq, s_sq

        # offdiagonal s-v rows
        for i in range(len(self.row_ids)):
            with self.k_sv_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_sv_vec[i], self.dcq_form[i])
            self.k_sv_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # offdiagonal v-s columns
        for i in range(len(self.col_ids)):
            with self.k_vs_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_vs_vec[i], self.dforce_form[i]) # already multiplied by time-integration factor
            # apply velocity dbcs to matrix entries k_vs - basically since these are offdiagonal we want a zero there!
            fem.apply_lifting(self.k_vs_vec[i], [self.pbf.jac_vv], [self.pbf.bc.dbcs], x0=[self.pbf.v.vector], scale=0.0)
            self.k_vs_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(self.k_vs_vec[i], self.pbf.bc.dbcs, x0=self.pbf.v.vector, scale=0.0)

        # row ownership range of vv block
        irs, ire = self.pbf.K_list[0][0].getOwnershipRange()

        # set columns
        for i in range(len(self.col_ids)):
            self.K_vs[irs:ire, self.col_ids[i]] = self.k_vs_vec[i][irs:ire]

        self.K_vs.assemble()

        # set rows
        for i in range(len(self.row_ids)):
            self.K_sv[self.row_ids[i], irs:ire] = self.k_sv_vec[i][irs:ire]

        self.K_sv.assemble()

        if bool(self.residual_scale):
            self.K_vs.scale(self.residual_scale[0])
            self.K_sv.scale(self.residual_scale[2])
            self.K_lm.scale(self.residual_scale[2])

        self.K_list[0][2] = self.K_vs
        self.K_list[2][0] = self.K_sv


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.vector.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.vector.getLocalSize()

        offset_v = vvec_or0 + self.pbf.p.vector.getOwnershipRange()[0] + self.lm.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.pbf.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_s = offset_p + self.pbf.p.vector.getLocalSize()
        iset_s = PETSc.IS().createStride(self.lm.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_s = iset_s.expand(iset_r) # add to 0D block
            iset_s.sort() # should be sorted, otherwise PETSc may struggle to extract block

        if isoptions['lms_to_p']:
            iset_p = iset_p.expand(iset_s) # add to pressure block
            ilist = [iset_v, iset_p]
        elif isoptions['lms_to_v']:
            iset_v = iset_u.expand(iset_s) # add to velocity block (could be bad...)
            ilist = [iset_v, iset_p]
        else:
            ilist = [iset_v, iset_p, iset_s]

        return ilist


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

        # special case: append upstream pressure to coupling array in case we don't have an LM, but a monitored pressure value
        if bool(self.pb0.chamber_models):
            if self.pb0.chamber_models['lv']['type']=='3D_fluid' and self.pb0.chamber_models['lv']['num_outflows']==0 and self.pb0.cardvasc0D.cormodel:
                if self.restart_step==0:
                    self.pb0.auxdata_old['p'] = copy.deepcopy(self.pbf.pu_old_) # copy since we write restart and update auxdata_old differently
                # for k in self.pb0.auxdata_old['p']: self.pb0.auxdata['p'][k] = self.pb0.auxdata_old['p'][k]
                dp_id = self.pb0.chamber_models['lv']['dp_monitor_id']
                self.pb0.c[0] = self.pb0.auxdata_old['p'][dp_id]

        for i in range(self.num_coupling_surf):
            lm_sq, lm_old_sq = allgather_vec(self.lm, self.comm), allgather_vec(self.lm_old, self.comm)
            self.pb0.c[i+self.offc] = lm_sq[i]
            con = fem.assemble_scalar(self.cq_old_form[i])
            con = self.comm.allgather(con)
            self.constr[i] = sum(con)*self.cq_factor[i]
            self.constr_old[i] = sum(con)*self.cq_factor[i]

        # length of c from 3D-0D coupling
        self.pb0.len_c_3d0d = len(self.pb0.c)

        if bool(self.pb0.chamber_models):
            for i, ch in enumerate(['lv','rv','la','ra']):
                if self.pb0.chamber_models[ch]['type']=='0D_elast': self.pb0.y[i] = self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['activation_curve'])(self.pbf.t_init)
                if self.pb0.chamber_models[ch]['type']=='0D_elast_prescr': self.pb0.y[i] = self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['elastance_curve'])(self.pbf.t_init)
                if self.pb0.chamber_models[ch]['type']=='0D_prescr': self.pb0.c.append(self.pb0.ti.timecurves(self.pb0.chamber_models[ch]['prescribed_curve'])(self.pbf.t_init))

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
                    elif prtype=='flux_monitor':
                        monid = prescr['flux_monitor']
                        val = self.pbf.qv_old_[monid]
                    else:
                        raise ValueError("Unknown type to prescribe a variable.")
                    self.pb0.s[varindex], self.pb0.s_old[varindex] = val, val

        # initially evaluate 0D model at old state
        self.pb0.cardvasc0D.evaluate(self.pb0.s_old, self.pbf.t_init, self.pb0.df_old, self.pb0.f_old, None, None, self.pb0.c, self.pb0.y, self.pb0.aux_old)
        self.pb0.auxTc_old[:] = self.pb0.aux_old[:]


    def write_output_ini(self):

        self.pbf.write_output_ini()


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t, N):

        self.pbf.evaluate_pre_solve(t, N)
        self.pb0.evaluate_pre_solve(t, N)


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


    def destroy(self):

        self.pbf.destroy()
        self.pb0.destroy()

        for i in range(len(self.col_ids)): self.k_vs_vec[i].destroy()
        for i in range(len(self.row_ids)): self.k_sv_vec[i].destroy()



class FluidmechanicsFlow0DSolver(solver_base):

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


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbf.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old - self.pb.power_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.pbf.io.USE_MIXED_DOLFINX_BRANCH:
                res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.pbf.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.pbf.io.entity_maps)
            else:
                res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbf.numstep, ni=ni, li=li, wt=wt)
