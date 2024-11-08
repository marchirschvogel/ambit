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
from ..mpiroutines import allgather_vec, allgather_vec_entry

from ..fluid.fluid_main import FluidmechanicsProblem
from ..base import problem_base, solver_base


class FluidmechanicsConstraintProblem(problem_base):

    def __init__(self, pbase, io_params, time_params_fluid, fem_params, constitutive_models, bc_dict, time_curves, coupling_params, io, mor_params={}, alevar={}):

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        self.problem_physics = 'fluid_constraint'

        self.coupling_params = coupling_params

        self.num_coupling_surf = len(self.coupling_params['constraint_physics'])
        assert(len(self.coupling_params['constraint_physics'])==len(self.coupling_params['multiplier_physics']))

        # store surfaces in lists for convenience
        self.surface_vq_ids, self.surface_lm_ids, self.vq_scales, self.btype_vq, self.btype_lm, self.on_subdomain = [], [], [], [], [], []
        for i in range(self.num_coupling_surf):
            self.surface_vq_ids.append(self.coupling_params['constraint_physics'][i]['id'])
            self.surface_lm_ids.append(self.coupling_params['multiplier_physics'][i]['id'])
            if 'scales' in self.coupling_params['constraint_physics'][i]:
                self.vq_scales.append(self.coupling_params['constraint_physics'][i]['scales'])
            else:
                self.vq_scales.append([1.]*len(self.coupling_params['constraint_physics'][i]['id']))
            if 'boundary_type' in self.coupling_params['constraint_physics'][i]:
                self.btype_vq.append(self.coupling_params['constraint_physics'][i]['boundary_type'])
            else:
                self.btype_vq.append(['ext']*len(self.coupling_params['constraint_physics'][i]['id']))
            if 'boundary_type' in self.coupling_params['multiplier_physics'][i]:
                self.btype_lm.append(self.coupling_params['multiplier_physics'][i]['boundary_type'])
            else:
                self.btype_lm.append(['ext']*len(self.coupling_params['multiplier_physics'][i]['id']))
            if 'on_subdomain' in self.coupling_params['constraint_physics'][i]:
                self.on_subdomain.append(self.coupling_params['constraint_physics'][i]['on_subdomain'])
            else:
                self.on_subdomain.append(False)

        # initialize problem instances (also sets the variational forms for the fluid problem)
        self.pbf = FluidmechanicsProblem(pbase, io_params, time_params_fluid, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params=mor_params, alevar=alevar)

        self.pbrom = self.pbf # ROM problem can only be fluid
        self.pbrom_host = self

        self.set_variational_forms()

        self.numdof = self.pbf.numdof + self.LM.getSize()

        self.localsolve = self.pbf.localsolve

        self.sub_solve = False
        self.print_subiter = False
        self.have_condensed_variables = False

        if 'regularization' in self.coupling_params:
            self.have_regularization = True
        else:
            self.have_regularization = False

        self.io = self.pbf.io

        # 3D fluxes
        self.constr, self.constr_old = [[]]*self.num_coupling_surf, [[]]*self.num_coupling_surf

        # number of fields involved
        self.nfields = 3

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbf.num_dupl > 1: is_ghosted = [1, 2, 0]
        else:                     is_ghosted = [1, 1, 0]
        varlist = [self.pbf.v.x.petsc_vec, self.pbf.p.x.petsc_vec, self.LM]

        return varlist, is_ghosted


    # defines the monolithic coupling forms for constraints and fluid mechanics
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

                if self.btype_vq[n][i]=='ext':
                    fct_side = None
                    bmi = 0 # to grep out ds from bmeasures
                elif self.btype_vq[n][i]=='int':
                    fct_side = '+'
                    bmi = 2 # to grep out dS from bmeasures
                else:
                    raise NameError("Unknown boundary type for constraint! Can only be 'ext' (external) or 'int' (internal).")

                if self.on_subdomain[n]:
                    assert(self.btype_vq[n][i]=='ext') # cannot do an internal intergral on a subdomain here...
                    dom_u = self.coupling_params['constraint_physics'][n]['domain']
                    ds_vq = ufl.ds(domain=self.pbf.io.submshes_emap[dom_u][0], subdomain_data=self.pbf.io.sub_mt_b1[dom_u], subdomain_id=self.surface_vq_ids[n][i], metadata={'quadrature_degree': self.pbf.io.quad_degree})
                else:
                    ds_vq = self.pbf.bmeasures[bmi](self.surface_vq_ids[n][i])
                cq_ += self.vq_scales[n][i] * self.pbf.vf.flux(self.pbf.v, ds_vq, w=self.pbf.alevar['w'], F=self.pbf.alevar['Fale'], fcts=fct_side)
                cq_old_ += self.vq_scales[n][i] * self.pbf.vf.flux(self.pbf.v_old, ds_vq, w=self.pbf.alevar['w_old'], F=self.pbf.alevar['Fale_old'], fcts=fct_side)

            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbf.v, self.pbf.dv))

            df_, df_mid_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_lm_ids[n])):

                if self.btype_lm[n][i]=='ext':
                    fct_side = None
                    bmi = 0 # to grep out ds from bmeasures
                elif self.btype_lm[n][i]=='int':
                    fct_side = '+'
                    bmi = 2 # to grep out dS from bmeasures
                else:
                    raise NameError("Unknown boundary type for multiplier! Can only be 'ext' (external) or 'int' (internal).")

                ds_p = self.pbf.bmeasures[bmi](self.surface_lm_ids[n][i])

                if self.coupling_params['multiplier_physics'][n]['type'] == 'pressure':

                    # add to fluid rhs contributions - external power, but positive, since multiplier acts against outward normal
                    self.power_coupling += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs[-1], ds_p, F=self.pbf.alevar['Fale'])
                    self.power_coupling_old += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs_old[-1], ds_p, F=self.pbf.alevar['Fale_old'])
                    self.power_coupling_mid += self.pbf.vf.deltaW_ext_neumann_normal_cur(self.coupfuncs_mid[-1], ds_p, F=self.pbf.alevar['Fale_mid'])

                    # derivative w.r.t. multiplier
                    df_ += self.pbf.timefac * self.pbf.vf.flux(self.pbf.var_v, ds_p, w=ufl.constantvalue.zero(self.pbf.ki.dim), F=self.pbf.alevar['Fale'])
                    df_mid_ += self.pbf.timefac * self.pbf.vf.flux(self.pbf.var_v, ds_p, w=ufl.constantvalue.zero(self.pbf.ki.dim), F=self.pbf.alevar['Fale_mid'])

                elif self.coupling_params['multiplier_physics'][n]['type'] == 'active_stress':

                    # for safety reasons, we require a membrane model on each of the surfaces we use for applying the active stress multiplier!
                    assert(self.surface_lm_ids[n][i] == self.pbf.bc_dict['membrane'][n]['id'][i])
                    # assert that there's no active stress model used in the membrane model at that boundary
                    assert('active_stress' not in self.pbf.bc_dict['membrane'][n]['params'])
                    # use the same parameters from the membrane model at that boundary
                    h0, memmodel = self.pbf.bc_dict['membrane'][n]['params']['h0'], self.pbf.bc_dict['membrane'][n]['params']['model']

                    if self.coupling_params['multiplier_physics'][n]['dir'] == 'iso':
                        params_ = {'h0' : h0, 'model' : memmodel, 'active_stress' : {'dir' : 'iso'}}
                    elif self.coupling_params['multiplier_physics'][n]['dir'] == 'cl':
                        assert(bool(self.pbf.io.fiber_data))
                        omega, iota, gamma = self.coupling_params['multiplier_physics'][n]['omega'], self.coupling_params['multiplier_physics'][n]['iota'], self.coupling_params['multiplier_physics'][n]['gamma']
                        params_ = {'h0' : h0, 'model' : memmodel, 'active_stress' : {'dir' : 'cl', 'omega' : omega, 'iota' : iota, 'gamma' : gamma}}
                    else:
                        raise NameError("Unknown active stress direction! Choose either iso or cl!")

                    ivar_ = {'tau_a' : self.coupfuncs[-1]}
                    ivar_old_ = {'tau_a' : self.coupfuncs_old[-1]}
                    ivar_mid_ = {'tau_a' : self.coupfuncs_mid[-1]}

                    if 'weight' in self.coupling_params['multiplier_physics'][n].keys():
                        # active stress weighting for reduced solid
                        wact_func = fem.Function(self.pbf.V_scalar)
                        self.pbf.io.readfunction(wact_func, self.coupling_params['multiplier_physics'][n]['weight'])
                        self.pbf.actweights.append(wact_func)
                    else:
                        self.pbf.actweights.append(None)

                    # add internal active stress power to fluid rhs contributions
                    self.power_coupling += self.pbf.vf.deltaW_ext_membrane(self.pbf.ki.F(self.pbf.ufluid), self.pbf.ki.Fdot(self.pbf.v), None, params_, ds_p, ivar=ivar_, fibfnc=self.pbf.fib_func, wallfield=self.pbf.wallfields[n], actweight=self.pbf.actweights[-1], returnquantity='active_stress_power')
                    self.power_coupling_old += self.pbf.vf.deltaW_ext_membrane(self.pbf.ki.F(self.pbf.uf_old), self.pbf.ki.Fdot(self.pbf.v_old), None, params_, ds_p, ivar=ivar_old_, fibfnc=self.pbf.fib_func, wallfield=self.pbf.wallfields[n], actweight=self.pbf.actweights[-1], returnquantity='active_stress_power')
                    self.power_coupling_mid += self.pbf.vf.deltaW_ext_membrane(self.pbf.ki.F(self.pbf.ufluid_mid), self.pbf.ki.Fdot(self.pbf.vel_mid), None, params_, ds_p, ivar=ivar_mid_, fibfnc=self.pbf.fib_func, wallfield=self.pbf.wallfields[n], actweight=self.pbf.actweights[-1], returnquantity='active_stress_power')

                    # derivative w.r.t. multiplier
                    df_ += self.pbf.timefac * self.pbf.vf.deltaW_ext_membrane(self.pbf.ki.F(self.pbf.ufluid), self.pbf.ki.Fdot(self.pbf.v), None, params_, ds_p, ivar=ivar_, fibfnc=self.pbf.fib_func, wallfield=self.pbf.wallfields[n], actweight=self.pbf.actweights[-1], returnquantity='active_stress_power_deriv')
                    df_mid_ += self.pbf.timefac * self.pbf.vf.deltaW_ext_membrane(self.pbf.ki.F(self.pbf.ufluid_mid), self.pbf.ki.Fdot(self.pbf.vel_mid), None, params_, ds_p, ivar=ivar_mid_, fibfnc=self.pbf.fib_func, wallfield=self.pbf.wallfields[n], actweight=self.pbf.actweights[-1], returnquantity='active_stress_power_deriv')

                elif self.coupling_params['multiplier_physics'][n]['type'] == 'valve_viscosity':

                    # add to fluid rhs contributions - external power; should be positive, hence minus sign
                    self.power_coupling += -self.pbf.vf.deltaW_ext_robin_valve(self.pbf.v, self.coupfuncs[-1], ds_p, fcts=fct_side, w=self.pbf.alevar['w'], F=self.pbf.alevar['Fale'])
                    self.power_coupling_old += -self.pbf.vf.deltaW_ext_robin_valve(self.pbf.v_old, self.coupfuncs_old[-1], ds_p, fcts=fct_side, w=self.pbf.alevar['w_old'], F=self.pbf.alevar['Fale_old'])
                    self.power_coupling_mid += -self.pbf.vf.deltaW_ext_robin_valve(self.pbf.vel_mid, self.coupfuncs_mid[-1], ds_p, fcts=fct_side, w=self.pbf.alevar['w_mid'], F=self.pbf.alevar['Fale_mid'])

                    # derivative w.r.t. multiplier
                    df_ += -self.pbf.timefac * self.pbf.vf.deltaW_ext_robin_valve_deriv_visc(self.pbf.v, ds_p, fcts=fct_side, w=self.pbf.alevar['w'], F=self.pbf.alevar['Fale'])
                    df_mid_ += -self.pbf.timefac * self.pbf.vf.deltaW_ext_robin_valve_deriv_visc(self.pbf.vel_mid, ds_p, fcts=fct_side, w=self.pbf.alevar['w_mid'], F=self.pbf.alevar['Fale_mid'])

                else:
                    raise NameError("Unknown multiplier physics type! Choose either pressure or active_stress!")

            if self.pbf.ti.eval_nonlin_terms=='trapezoidal': self.dforce.append(df_)
            if self.pbf.ti.eval_nonlin_terms=='midpoint': self.dforce.append(df_mid_)

        if self.pbf.ti.eval_nonlin_terms=='trapezoidal':
            # add to fluid rhs
            self.pbf.weakform_v += self.pbf.timefac * self.power_coupling + (1.-self.pbf.timefac) * self.power_coupling_old
            # add to fluid Jacobian
            self.pbf.weakform_lin_vv += self.pbf.timefac * ufl.derivative(self.power_coupling, self.pbf.v, self.pbf.dv)
        if self.pbf.ti.eval_nonlin_terms=='midpoint':
            # add to fluid rhs
            self.pbf.weakform_v += self.power_coupling_mid
            # add to fluid Jacobian
            self.pbf.weakform_lin_vv += ufl.derivative(self.power_coupling_mid, self.pbf.v, self.pbf.dv)


    def set_multiplier(self, var, p0Da):

        # set pressure functions
        for i in range(self.num_coupling_surf):
            self.pr0D.val = allgather_vec_entry(var, i, self.comm)
            p0Da[i].interpolate(self.pr0D.evaluate)


    def set_problem_residual_jacobian_forms(self, pre=False):

        self.pbf.set_problem_residual_jacobian_forms(pre=pre)
        self.set_problem_residual_jacobian_forms_coupling()


    def set_problem_residual_jacobian_forms_coupling(self):

        ts = time.time()
        utilities.print_status("FEM form compilation for fluid-constraint coupling...", self.comm, e=" ")

        self.cq_form, self.cq_old_form, self.dcq_form, self.dforce_form = [], [], [], []

        for i in range(self.num_coupling_surf):
            if self.on_subdomain[i]:
                # entity map child to parent
                em_u = {self.io.mesh : self.pbf.io.submshes_emap[self.coupling_params['constraint_physics'][i]['domain']][1]}
            else:
                em_u = self.pbf.io.entity_maps
            self.cq_form.append(fem.form(self.cq[i], entity_maps=em_u))
            self.cq_old_form.append(fem.form(self.cq_old[i], entity_maps=em_u))

            self.dcq_form.append(fem.form(self.dcq[i], entity_maps=em_u))
            self.dforce_form.append(fem.form(self.dforce[i], entity_maps=self.pbf.io.entity_maps))

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)


    def set_problem_vector_matrix_structures(self):

        self.pbf.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()


    def set_problem_vector_matrix_structures_coupling(self):

        self.r_lm = PETSc.Vec().createMPI(size=self.num_coupling_surf)

        # Lagrange multiplier stiffness matrix (could be non-zero for regularized constraints...)
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

            nds_p_local = fem.locate_dofs_topological(self.pbf.V_v, self.pbf.io.mesh.topology.dim-1, self.pbf.io.mt_b1.indices[np.isin(self.pbf.io.mt_b1.values, self.surface_lm_ids[n])])
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

        self.constr_val = list(range(self.num_coupling_surf))

        self.alpha_reg = list(range(self.num_coupling_surf))
        self.kp_reg = list(range(self.num_coupling_surf))
        if self.have_regularization:
            for n in range(self.num_coupling_surf):
                self.kp_reg[n] = self.coupling_params['regularization'][n]['kp']


    def assemble_residual(self, t, subsolver=None):

        # interpolate LM into function
        self.set_multiplier(self.LM, self.coupfuncs)

        # fluid main blocks
        self.pbf.assemble_residual(t)

        self.r_list[0] = self.pbf.r_list[0]
        self.r_list[1] = self.pbf.r_list[1]

        ls, le = self.LM.getOwnershipRange()

        for i in range(len(self.surface_lm_ids)):
            cq = fem.assemble_scalar(self.cq_form[i])
            cq = self.comm.allgather(cq)
            self.constr[i] = sum(cq)

        # Lagrange multiplier coupling residual
        for i in range(ls,le):
            if self.have_regularization:
                self.r_lm[i] = self.alpha_reg[i]*(self.constr[i] - self.constr_val[i]) + (1.-self.alpha_reg[i])*self.kp_reg[i]*self.LM[i]
            else:
                self.r_lm[i] = self.constr[i] - self.constr_val[i]

        self.r_lm.assemble()

        self.r_list[2] = self.r_lm


    def assemble_stiffness(self, t, subsolver=None):

        # fluid main blocks
        self.pbf.assemble_stiffness(t)

        self.K_list[0][0] = self.pbf.K_list[0][0]
        self.K_list[0][1] = self.pbf.K_list[0][1]
        self.K_list[1][0] = self.pbf.K_list[1][0]
        self.K_list[1][1] = self.pbf.K_list[1][1] # non-zero if we have stabilization

        # offdiagonal s-v rows
        for i in range(len(self.row_ids)):
            with self.k_sv_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_sv_vec[i], self.dcq_form[i])
            self.k_sv_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            if self.have_regularization:
                self.k_sv_vec[i].scale(self.alpha_reg[i])

        # offdiagonal v-s columns
        for i in range(len(self.col_ids)):
            with self.k_vs_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_vs_vec[i], self.dforce_form[i]) # already multiplied by time-integration factor
            self.k_vs_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # set zeros at DBC entries
            fem.set_bc(self.k_vs_vec[i], self.pbf.bc.dbcs, x0=self.pbf.v.x.petsc_vec, scale=0.0)

        # set columns
        for i in range(len(self.col_ids)):
            # NOTE: only set the surface-subset of the k_vs vector entries to avoid placing unnecessary zeros!
            self.k_vs_vec[i].getSubVector(self.dofs_coupling_p[i], subvec=self.k_vs_subvec[i])
            self.K_vs.setValues(self.dofs_coupling_p[i], self.col_ids[i], self.k_vs_subvec[i].array, addv=PETSc.InsertMode.INSERT)
            self.k_vs_vec[i].restoreSubVector(self.dofs_coupling_p[i], subvec=self.k_vs_subvec[i])

        self.K_vs.assemble()

        # set rows
        for i in range(len(self.row_ids)):
            # NOTE: only set the surface-subset of the k_sv vector entries to avoid placing unnecessary zeros!
            self.k_sv_vec[i].getSubVector(self.dofs_coupling_vq[i], subvec=self.k_sv_subvec[i])
            self.K_sv.setValues(self.row_ids[i], self.dofs_coupling_vq[i], self.k_sv_subvec[i].array, addv=PETSc.InsertMode.INSERT)
            self.k_sv_vec[i].restoreSubVector(self.dofs_coupling_vq[i], subvec=self.k_sv_subvec[i])

        self.K_sv.assemble()

        self.K_list[0][2] = self.K_vs
        self.K_list[2][0] = self.K_sv

        if self.have_regularization:
            ls, le = self.K_lm.getOwnershipRange()
            for i in range(ls,le):
                self.K_lm[i,i] = (1.-self.alpha_reg[i])*self.kp_reg[i]

            self.K_lm.assemble()

            self.K_list[2][2] = self.K_lm


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.x.petsc_vec.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.x.petsc_vec.getLocalSize()

        offset_v = vvec_or0 + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.LM.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.pbf.p.x.petsc_vec.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_s = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
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


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # fluid problem
        self.pbf.read_restart(sname, N)
        # LM data
        if N > 0:
            restart_data = np.loadtxt(self.pbf.io.output_path+'/checkpoint_'+sname+'_lm_'+str(N)+'.txt', ndmin=1)
            self.LM[:], self.LM_old[:] = restart_data[:], restart_data[:]


    def evaluate_initial(self):

        self.pbf.evaluate_initial()

        self.set_multiplier(self.LM_old, self.coupfuncs_old)

        for i in range(self.num_coupling_surf):
            con = fem.assemble_scalar(self.cq_form[i])
            con = self.comm.allgather(con)
            self.constr[i] = sum(con)
            self.constr_old[i] = sum(con)


    def write_output_ini(self):

        self.pbf.write_output_ini()


    def write_output_pre(self):

        self.pbf.write_output_pre()


    def evaluate_pre_solve(self, t, N, dt):

        self.pbf.evaluate_pre_solve(t, N, dt)

        for n in range(self.num_coupling_surf):
            self.constr_val[n] = self.pbf.ti.timecurves(self.coupling_params['constraint_physics'][n]['prescribed_curve'])(t)

        if self.have_regularization:
            for n in range(self.num_coupling_surf):
                self.alpha_reg[n] = self.pbf.ti.timecurves(self.coupling_params['regularization'][n]['curve'])(t)


    def evaluate_post_solve(self, t, N):

        self.pbf.evaluate_post_solve(t, N)


    def set_output_state(self, N):

        self.pbf.set_output_state(N)


    def write_output(self, N, t, mesh=False):

        self.pbf.write_output(N, t)

        if self.pbf.io.write_results_every > 0 and N % self.pbf.io.write_results_every == 0:
            if np.isclose(t,self.pbase.dt): mode = 'wt'
            else: mode = 'a'
            LM_sq = allgather_vec(self.LM, self.comm)
            if self.comm.rank == 0:
                for i in range(len(LM_sq)):
                    f = open(self.pbase.output_path+'/results_'+self.pbase.simname+'_LM'+str(i+1)+'.txt', mode)
                    f.write('%.16E %.16E\n' % (t,LM_sq[i]))
                    f.close()
            del LM_sq


    def update(self):

        # update time step
        self.pbf.update()

        # update old pressures on solid
        self.LM_old.axpby(1.0, 0.0, self.LM)
        self.set_multiplier(self.LM_old, self.coupfuncs_old)
        # update old 3D constraint variable
        self.constr_old[:] = self.constr[:]


    def print_to_screen(self):

        self.pbf.print_to_screen()

        LM_sq = allgather_vec(self.LM, self.comm)
        for i in range(self.num_coupling_surf):
            utilities.print_status("LM"+str(i+1)+" = %.4e" % (LM_sq[i]), self.comm)
        del LM_sq


    def induce_state_change(self):

        self.pbf.induce_state_change()


    def write_restart(self, sname, N, force=False):

        self.pbf.write_restart(sname, N, force=force)

        if (self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0) or force:
            LM_sq = allgather_vec(self.LM, self.comm)
            if self.comm.rank == 0:
                f = open(self.pbf.io.output_path+'/checkpoint_'+sname+'_lm_'+str(N)+'.txt', 'wt')
                for i in range(len(LM_sq)):
                    f.write('%.16E\n' % (LM_sq[i]))
                f.close()
            del LM_sq


    def check_abort(self, t):

        return False


    def destroy(self):

        self.pbf.destroy()

        for i in range(len(self.col_ids)): self.k_vs_vec[i].destroy()
        for i in range(len(self.row_ids)): self.k_sv_vec[i].destroy()



class FluidmechanicsConstraintSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbf.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old - self.pb.power_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.pbf.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.pbf.io.entity_maps)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
