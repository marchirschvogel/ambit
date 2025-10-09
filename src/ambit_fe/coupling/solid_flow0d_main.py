#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import copy
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import utilities, meshutils, expression
from ..solver.projection import project
from ..mpiroutines import allgather_vec

from ..solid.solid_main import (
    SolidmechanicsProblem,
    SolidmechanicsSolverPrestr,
)
from ..flow0d.flow0d_main import Flow0DProblem
from ..base import problem_base, solver_base


class SolidmechanicsFlow0DProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params_solid,
        time_params_flow0d,
        fem_params,
        constitutive_models,
        model_params_flow0d,
        bc_dict,
        time_curves,
        coupling_params,
        io,
        mor_params={},
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        self.problem_physics = "solid_flow0d"

        self.coupling_params = coupling_params

        self.surface_vq_ids = self.coupling_params["surface_ids"]
        self.surface_p_ids = self.coupling_params.get("surface_p_ids", self.surface_vq_ids)

        self.num_coupling_surf = len(self.surface_vq_ids)

        self.cq_factor = self.coupling_params.get("cq_factor", [1.0] * self.num_coupling_surf)

        self.coupling_type = self.coupling_params.get("coupling_type", "monolithic_direct")

        self.eps_fd = self.coupling_params.get("eps_fd", 1e-5)

        self.print_subiter = self.coupling_params.get("print_subiter", False)

        self.write_checkpoints_periodicref = self.coupling_params.get("write_checkpoints_periodicref", False)
        self.restart_periodicref = self.coupling_params.get("restart_periodicref", 0)
        self.Nmax_periodicref = self.coupling_params.get("Nmax_periodicref", 10)

        self.have_condensed_variables = False

        # initialize problem instances (also sets the variational forms for the solid problem)
        self.pbs = SolidmechanicsProblem(
            pbase,
            io_params,
            time_params_solid,
            fem_params,
            constitutive_models,
            bc_dict,
            time_curves,
            io,
            mor_params=mor_params,
        )
        self.pb0 = Flow0DProblem(
            pbase,
            io_params,
            time_params_flow0d,
            model_params_flow0d,
            time_curves,
            coupling_params,
        )

        self.pbrom = self.pbs  # ROM problem can only be solid
        self.pbrom_host = self

        self.incompressible_2field = self.pbs.incompressible_2field

        # for multiscale G&R analysis
        self.t_prev = 0
        self.t_gandr_setpoint = 0
        self.restart_multiscale = False

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        if self.pbase.problem_type == "solid_flow0d_multiscale_gandr":
            self.have_multiscale_gandr = True
        else:
            self.have_multiscale_gandr = False

        self.set_variational_forms()

        if self.coupling_type == "monolithic_direct":
            self.numdof = self.pbs.numdof + self.pb0.numdof
        elif self.coupling_type == "monolithic_lagrange":
            self.numdof = self.pbs.numdof + self.LM.getSize()
        else:
            raise ValueError("Unknown coupling type!")

        self.localsolve = self.pbs.localsolve

        if self.coupling_type == "monolithic_lagrange":
            self.sub_solve = True
        else:
            self.sub_solve = False

        self.io = self.pbs.io

        # 3D fluxes
        self.constr, self.constr_old = (
            [[]] * self.num_coupling_surf,
            [[]] * self.num_coupling_surf,
        )

        # set 3D-0D coupling array
        self.pb0.c = [[]] * (self.num_coupling_surf)

        # number of fields involved
        if self.pbs.incompressible_2field:
            self.nfields = 3
        else:
            self.nfields = 2

        # residual and matrix lists
        self.r_list, self.r_list_rom = (
            [None] * self.nfields,
            [None] * self.nfields,
        )
        self.K_list, self.K_list_rom = (
            [[None] * self.nfields for _ in range(self.nfields)],
            [[None] * self.nfields for _ in range(self.nfields)],
        )

    def get_problem_var_list(self):
        if self.coupling_type == "monolithic_lagrange":
            if self.pbs.incompressible_2field:
                is_ghosted = [1, 1, 0]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.LM,
                ], is_ghosted
            else:
                is_ghosted = [1, 0]
                return [self.pbs.u.x.petsc_vec, self.LM], is_ghosted

        if self.coupling_type == "monolithic_direct":
            if self.pbs.incompressible_2field:
                is_ghosted = [1, 1, 0]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pb0.s,
                ], is_ghosted
            else:
                is_ghosted = [1, 0]
                return [self.pbs.u.x.petsc_vec, self.pb0.s], is_ghosted

    # defines the monolithic coupling forms for 0D flow and solid mechanics
    def set_variational_forms(self):
        self.cq, self.cq_old, self.dcq, self.dforce = [], [], [], []
        (
            self.coupfuncs,
            self.coupfuncs_old,
            self.coupfuncs_mid,
            coupfuncs_pre,
        ) = (
            [],
            [],
            [],
            [],
        )

        if self.coupling_type == "monolithic_lagrange":
            # Lagrange multipliers
            self.LM, self.LM_old = (
                PETSc.Vec().createMPI(size=self.num_coupling_surf),
                PETSc.Vec().createMPI(size=self.num_coupling_surf),
            )

        self.work_coupling, self.work_coupling_old, self.work_coupling_mid = (
            ufl.as_ufl(0),
            ufl.as_ufl(0),
            ufl.as_ufl(0),
        )

        # coupling variational forms and Jacobian contributions
        for n in range(self.num_coupling_surf):
            try:
                coupling_quantity = self.coupling_params["coupling_quantity"][n]
            except:
                coupling_quantity = "volume"

            try:
                variable_quantity = self.coupling_params["variable_quantity"][n]
            except:
                variable_quantity = "pressure"

            self.pr0D = expression.template()

            (
                self.coupfuncs.append(fem.Function(self.pbs.Vd_scalar)),
                self.coupfuncs_old.append(fem.Function(self.pbs.Vd_scalar)),
            )
            (
                self.coupfuncs[-1].interpolate(self.pr0D.evaluate),
                self.coupfuncs_old[-1].interpolate(self.pr0D.evaluate),
            )
            self.coupfuncs_mid.append(
                self.pbs.timefac * self.coupfuncs[-1] + (1.0 - self.pbs.timefac) * self.coupfuncs_old[-1]
            )

            cq_, cq_old_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_vq_ids[n])):
                ds_vq = self.pbs.bmeasures[0](self.surface_vq_ids[n][i])

                if coupling_quantity == "volume":
                    assert self.coupling_type == "monolithic_direct" and variable_quantity == "pressure"
                    cq_ += self.pbs.vf.volume(
                        self.pbs.u,
                        ds_vq,
                        F=self.pbs.ki.F(self.pbs.u, ext=True),
                    )
                    cq_old_ += self.pbs.vf.volume(
                        self.pbs.u_old,
                        ds_vq,
                        F=self.pbs.ki.F(self.pbs.u_old, ext=True),
                    )
                elif coupling_quantity == "flux":
                    assert self.coupling_type == "monolithic_direct" and variable_quantity == "pressure"
                    cq_ += self.pbs.vf.flux(
                        self.pbs.vel,
                        ds_vq,
                        F=self.pbs.ki.F(self.pbs.u, ext=True),
                    )
                    cq_old_ += self.pbs.vf.flux(
                        self.pbs.v_old,
                        ds_vq,
                        F=self.pbs.ki.F(self.pbs.u_old, ext=True),
                    )
                elif coupling_quantity == "pressure":
                    assert self.coupling_type == "monolithic_lagrange"
                    if variable_quantity == "volume":
                        cq_ += self.pbs.vf.volume(
                            self.pbs.u,
                            ds_vq,
                            F=self.pbs.ki.F(self.pbs.u, ext=True),
                        )
                        cq_old_ += self.pbs.vf.volume(
                            self.pbs.u_old,
                            ds_vq,
                            F=self.pbs.ki.F(self.pbs.u_old, ext=True),
                        )
                    elif variable_quantity == "flux":
                        cq_ += self.pbs.vf.flux(
                            self.pbs.vel,
                            ds_vq,
                            F=self.pbs.ki.F(self.pbs.u, ext=True),
                        )
                        cq_old_ += self.pbs.vf.flux(
                            self.pbs.v_old,
                            ds_vq,
                            F=self.pbs.ki.F(self.pbs.u_old, ext=True),
                        )
                    else:
                        raise NameError("Unknown variable quantity! Choose either volume or flux!")
                else:
                    raise NameError("Unknown coupling quantity! Choose either volume, flux, or pressure!")

            self.cq.append(cq_), self.cq_old.append(cq_old_)
            self.dcq.append(ufl.derivative(self.cq[-1], self.pbs.u, self.pbs.du))

            df_, df_mid_ = ufl.as_ufl(0), ufl.as_ufl(0)
            for i in range(len(self.surface_p_ids[n])):
                ds_p = self.pbs.bmeasures[0](self.surface_p_ids[n][i])
                df_ += self.pbs.timefac * self.pbs.vf.flux(self.pbs.var_u, ds_p, F=self.pbs.ki.F(self.pbs.u, ext=True))
                df_mid_ += self.pbs.timefac * self.pbs.vf.flux(
                    self.pbs.var_u,
                    ds_p,
                    F=self.pbs.ki.F(self.pbs.us_mid, ext=True),
                )

                # add to solid rhs contributions
                self.work_coupling += self.pbs.vf.deltaW_ext_neumann_normal_cur(
                    self.coupfuncs[-1],
                    ds_p,
                    F=self.pbs.ki.F(self.pbs.u, ext=True),
                )
                self.work_coupling_old += self.pbs.vf.deltaW_ext_neumann_normal_cur(
                    self.coupfuncs_old[-1],
                    ds_p,
                    F=self.pbs.ki.F(self.pbs.u_old, ext=True),
                )
                self.work_coupling_mid += self.pbs.vf.deltaW_ext_neumann_normal_cur(
                    self.coupfuncs_mid[-1],
                    ds_p,
                    F=self.pbs.ki.F(self.pbs.us_mid, ext=True),
                )

            if self.pbs.ti.eval_nonlin_terms == "trapezoidal":
                self.dforce.append(df_)
            if self.pbs.ti.eval_nonlin_terms == "midpoint":
                self.dforce.append(df_mid_)

        if self.pbs.ti.eval_nonlin_terms == "trapezoidal":
            # minus sign, since contribution to external work!
            self.pbs.weakform_u += (
                -self.pbs.timefac * self.work_coupling - (1.0 - self.pbs.timefac) * self.work_coupling_old
            )
            # add to solid Jacobian
            self.pbs.weakform_lin_uu += -self.pbs.timefac * ufl.derivative(self.work_coupling, self.pbs.u, self.pbs.du)
        if self.pbs.ti.eval_nonlin_terms == "midpoint":
            # minus sign, since contribution to external work!
            self.pbs.weakform_u += -self.work_coupling_mid
            # add to solid Jacobian
            self.pbs.weakform_lin_uu += -ufl.derivative(self.work_coupling_mid, self.pbs.u, self.pbs.du)

        if self.coupling_type == "monolithic_lagrange" and self.pbase.restart_step == 0:
            # old Lagrange multipliers - initialize with initial pressures
            self.pb0.cardvasc0D.initialize_lm(self.LM, self.pb0.initialconditions)
            self.pb0.cardvasc0D.initialize_lm(self.LM_old, self.pb0.initialconditions)

    # for multiscale G&R analysis
    def set_homeostatic_threshold(self, t):
        # time is absolute time (should only be set in first cycle)
        eps = 1.0e-14
        if t >= self.t_gandr_setpoint - eps and t < self.t_gandr_setpoint + self.pbs.dt - eps:
            utilities.print_status("Set homeostatic growth thresholds...", self.comm)
            time.sleep(1)

            growth_thresolds = []
            for n in range(self.pbs.num_domains):
                if self.pbs.mat_growth[n]:
                    growth_settrig = self.pbs.constitutive_models["MAT" + str(n + 1) + ""]["growth"]["growth_settrig"]

                    if growth_settrig == "fibstretch":
                        growth_thresolds.append(
                            self.pbs.ma[n].fibstretch_e(
                                self.pbs.ki.C(self.pbs.u),
                                self.pbs.theta,
                                self.pbs.fib_func[0],
                            )
                        )
                    elif growth_settrig == "volstress":
                        growth_thresolds.append(
                            tr(
                                self.pbs.ma[n].M_e(
                                    self.pbs.u,
                                    self.pbs.p,
                                    self.pbs.ki.C(self.pbs.u),
                                    ivar=self.pbs.internalvars,
                                )
                            )
                        )
                    else:
                        raise NameError("Unknown growth trigger to be set as homeostatic threshold!")

                else:
                    growth_thresolds.append(ufl.as_ufl(0))

            growth_thres_proj = project(
                growth_thresolds,
                self.pbs.Vd_scalar,
                self.pbs.dx_,
                comm=self.comm,
            )
            self.pbs.growth_param_funcs["growth_thres"].x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            self.pbs.growth_param_funcs["growth_thres"].interpolate(growth_thres_proj)

    # for multiscale G&R analysis
    def set_growth_trigger(self, t):
        # time is relative time (w.r.t. heart cycle)
        eps = 1.0e-14
        if t >= self.t_gandr_setpoint - eps and t < self.t_gandr_setpoint + self.pbs.dt - eps:
            utilities.print_status("Set growth triggers...", self.comm)
            time.sleep(1)

            self.pbs.u_set.x.petsc_vec.axpby(1.0, 0.0, self.pbs.u.x.petsc_vec)
            self.pbs.u_set.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            if self.pbs.incompressible_2field:
                self.pbs.p_set.x.petsc_vec.axpby(1.0, 0.0, self.pbs.p.x.petsc_vec)
                self.pbs.p_set.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )

            self.pbs.tau_a_set.x.petsc_vec.axpby(1.0, 0.0, self.pbs.tau_a.x.petsc_vec)
            self.pbs.tau_a_set.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            if self.pbs.have_frank_starling:
                self.pbs.amp_old_set.x.petsc_vec.axpby(1.0, 0.0, self.pbs.amp_old.x.petsc_vec)
                self.pbs.amp_old_set.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )

            self.pb0.s_set.axpby(1.0, 0.0, self.pb0.s)

    def set_problem_residual_jacobian_forms(self, pre=False):
        self.pbs.set_problem_residual_jacobian_forms(pre=pre)
        self.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        ts = time.time()
        utilities.print_status("FEM form compilation for solid-0D coupling...", self.comm, e=" ")

        self.cq_form, self.cq_old_form, self.dcq_form, self.dforce_form = (
            [],
            [],
            [],
            [],
        )

        for i in range(self.num_coupling_surf):
            self.cq_form.append(fem.form(self.cq[i]))
            self.cq_old_form.append(fem.form(self.cq_old[i]))

            self.dcq_form.append(fem.form(self.cq_factor[i] * self.dcq[i]))
            self.dforce_form.append(fem.form(self.dforce[i]))

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        self.pbs.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        self.r_lm = PETSc.Vec().createMPI(size=self.num_coupling_surf)

        # Lagrange multiplier stiffness matrix (currently treated with FD!)
        if self.coupling_type == "monolithic_lagrange":
            self.K_lm = PETSc.Mat().createAIJ(
                size=(self.num_coupling_surf, self.num_coupling_surf),
                bsize=None,
                nnz=None,
                csr=None,
                comm=self.comm,
            )
            self.K_lm.setUp()
            sze_coup = self.num_coupling_surf
            self.row_ids = list(range(self.num_coupling_surf))
            self.col_ids = list(range(self.num_coupling_surf))

        if self.coupling_type == "monolithic_direct":
            sze_coup = self.pb0.numdof
            self.row_ids = self.pb0.cardvasc0D.c_ids
            self.col_ids = self.pb0.cardvasc0D.v_ids

        # setup offdiagonal matrices
        locmatsize = self.pbs.V_u.dofmap.index_map.size_local * self.pbs.V_u.dofmap.index_map_bs
        matsize = self.pbs.V_u.dofmap.index_map.size_global * self.pbs.V_u.dofmap.index_map_bs

        self.k_us_vec = []
        for i in range(len(self.col_ids)):
            self.k_us_vec.append(fem.petsc.create_vector(self.dforce_form[i]))

        self.k_su_vec = []
        for i in range(len(self.row_ids)):
            self.k_su_vec.append(fem.petsc.create_vector(self.dcq_form[i]))

        self.dofs_coupling_vq, self.dofs_coupling_p = (
            [[]] * self.num_coupling_surf,
            [[]] * self.num_coupling_surf,
        )

        self.k_us_subvec, self.k_su_subvec, sze_us, sze_su = [], [], [], []

        for n in range(self.num_coupling_surf):
            self.dofs_coupling_vq[n] = meshutils.get_index_set_id(self.pbs.io, self.pbs.V_u, self.surface_vq_ids[n], self.pbs.io.mesh.topology.dim-1, self.comm)

            self.k_su_subvec.append(self.k_su_vec[n].getSubVector(self.dofs_coupling_vq[n]))

            sze_su.append(self.k_su_subvec[-1].getSize())

            self.dofs_coupling_p[n] = meshutils.get_index_set_id(self.pbs.io, self.pbs.V_u, self.surface_p_ids[n], self.pbs.io.mesh.topology.dim-1, self.comm)

            self.k_us_subvec.append(self.k_us_vec[n].getSubVector(self.dofs_coupling_p[n]))

            sze_us.append(self.k_us_subvec[-1].getSize())

        # derivative of solid residual w.r.t. 0D pressures/multipliers
        self.K_us = PETSc.Mat().createAIJ(
            size=((locmatsize, matsize), (PETSc.DECIDE, sze_coup)),
            bsize=None,
            nnz=self.num_coupling_surf,
            csr=None,
            comm=self.comm,
        )
        self.K_us.setUp()

        # derivative of 0D residual/multiplier constraints w.r.t. solid displacements
        self.K_su = PETSc.Mat().createAIJ(
            size=((PETSc.DECIDE, sze_coup), (locmatsize, matsize)),
            bsize=None,
            nnz=max(sze_su),
            csr=None,
            comm=self.comm,
        )
        self.K_su.setUp()
        self.K_su.setOption(PETSc.Mat.Option.ROW_ORIENTED, False)

    def assemble_residual(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        if self.coupling_type == "monolithic_lagrange":
            for i in range(self.num_coupling_surf):
                cq = fem.assemble_scalar(self.cq_form[i])
                cq = self.comm.allgather(cq)
                self.constr[i] = sum(cq) * self.cq_factor[i]

            # Lagrange multipliers (pressures) to be passed to 0D model
            LM_sq = allgather_vec(self.LM, self.comm)

            for i in range(self.num_coupling_surf):
                self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = LM_sq[i]

            if subsolver is not None:
                # only have rank 0 solve the ODE, then broadcast solution
                err = -1
                if self.comm.rank == 0:
                    err = subsolver.newton(t, print_iter=self.print_subiter, sub=True)
                self.comm.Barrier()
                # need to broadcast to all cores
                err = self.comm.bcast(err, root=0)
                if err > 0:
                    subsolver.solver_error(self)
                self.pb0.s.array[:] = self.comm.bcast(self.pb0.s.array, root=0)
                self.pb0.df.array[:] = self.comm.bcast(self.pb0.df.array, root=0)
                self.pb0.f.array[:] = self.comm.bcast(self.pb0.f.array, root=0)
                self.pb0.aux[:] = self.comm.bcast(self.pb0.aux, root=0)

            # add to solid momentum equation
            self.pb0.cardvasc0D.set_pressure_fem(
                self.LM,
                list(range(self.num_coupling_surf)),
                self.pr0D,
                self.coupfuncs,
            )

        if self.coupling_type == "monolithic_direct":
            # add to solid momentum equation
            self.pb0.cardvasc0D.set_pressure_fem(
                self.pb0.s,
                self.pb0.cardvasc0D.v_ids,
                self.pr0D,
                self.coupfuncs,
            )

            # volumes/fluxes to be passed to 0D model
            for i in range(len(self.pb0.cardvasc0D.c_ids)):
                cq = fem.assemble_scalar(self.cq_form[i])
                cq = self.comm.allgather(cq)
                self.pb0.c[i] = sum(cq) * self.cq_factor[i]

            # 0D rhs vector
            self.pb0.assemble_residual(t)

            self.r_list[1 + off] = self.pb0.r_list[0]

        # solid main blocks
        self.pbs.assemble_residual(t)

        self.r_list[0] = self.pbs.r_list[0]
        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbs.r_list[1]

        if self.coupling_type == "monolithic_lagrange":
            ls, le = self.LM.getOwnershipRange()

            # Lagrange multiplier coupling residual
            for i in range(ls, le):
                self.r_lm[i] = self.constr[i] - self.pb0.s[self.pb0.cardvasc0D.v_ids[i]]

            self.r_lm.assemble()

            self.r_list[1 + off] = self.r_lm

            del LM_sq

    def assemble_stiffness(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        if self.coupling_type == "monolithic_lagrange":
            # Lagrange multipliers (pressures) to be passed to 0D model
            LM_sq = allgather_vec(self.LM, self.comm)

            for i in range(self.num_coupling_surf):
                self.pb0.c[self.pb0.cardvasc0D.c_ids[i]] = LM_sq[i]

        if self.coupling_type == "monolithic_direct":
            # volumes/fluxes to be passed to 0D model
            for i in range(len(self.pb0.cardvasc0D.c_ids)):
                cq = fem.assemble_scalar(fem.form(self.cq[i]))
                cq = self.comm.allgather(cq)
                self.pb0.c[i] = sum(cq) * self.cq_factor[i]

            # 0D stiffness
            self.pb0.assemble_stiffness(t)

            self.K_list[1 + off][1 + off] = self.pb0.K_list[0][0]

        # solid main blocks
        self.pbs.assemble_stiffness(t)

        self.K_list[0][0] = self.pbs.K_list[0][0]
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbs.K_list[0][1]
            self.K_list[1][0] = self.pbs.K_list[1][0]
            self.K_list[1][1] = self.pbs.K_list[1][1]  # should be only non-zero if we have stress-mediated growth...

        if self.coupling_type == "monolithic_lagrange":
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
                for i in range(ls, le):  # row-owning rank calls the ODE solver
                    for j in range(self.num_coupling_surf):
                        self.pb0.c[self.pb0.cardvasc0D.c_ids[j]] = LM_sq[j] + self.eps_fd  # perturbed LM
                        subsolver.newton(t, print_iter=False, sub=True)
                        val = (
                            -(self.pb0.s[self.pb0.cardvasc0D.v_ids[i]] - self.pb0.s_tmp[self.pb0.cardvasc0D.v_ids[i]])
                            / self.eps_fd
                        )
                        self.K_lm.setValue(i, j, val, addv=PETSc.InsertMode.INSERT)
                        self.pb0.c[self.pb0.cardvasc0D.c_ids[j]] = LM_sq[j]  # restore LM

            self.comm.Barrier()  # do we need this here, since not all processes participate in the ODE solve?

            # restore df, f, and aux vectors for correct time step update
            self.pb0.df.axpby(1.0, 0.0, self.pb0.df_tmp)
            self.pb0.f.axpby(1.0, 0.0, self.pb0.f_tmp)
            self.pb0.aux[:] = self.pb0.aux_tmp[:]
            # restore 0D state variable
            self.pb0.s.axpby(1.0, 0.0, self.pb0.s_tmp)

            self.K_lm.assemble()

            self.K_list[1 + off][1 + off] = self.K_lm

            del LM_sq

        # offdiagonal s-u rows
        for i in range(len(self.row_ids)):
            # depending on if we have volumes, fluxes, or pressures passed in (latter for LM coupling)
            if self.pb0.cq[i] == "volume":
                timefac = 1.0 / self.pbase.dt
            if self.pb0.cq[i] == "flux":
                timefac = -self.pb0.theta0d_timint(t)  # 0D model time-integration factor
            if self.pb0.cq[i] == "pressure":
                timefac = 1.0

            with self.k_su_vec[i].localForm() as r_local:
                r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_su_vec[i], self.dcq_form[i])
            # ghost update on k_su_rows - needs to be done prior to scale
            self.k_su_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.k_su_vec[i].scale(timefac)

        # offdiagonal u-s columns
        for i in range(len(self.col_ids)):
            with self.k_us_vec[i].localForm() as r_local:
                r_local.set(0.0)
            fem.petsc.assemble_vector(
                self.k_us_vec[i], self.dforce_form[i]
            )  # already multiplied by time-integration factor
            self.k_us_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # set zeros at DBC entries
            fem.set_bc(
                self.k_us_vec[i],
                self.pbs.bc.dbcs,
                x0=self.pbs.u.x.petsc_vec,
                alpha=0.0,
            )

        # set columns
        for i in range(len(self.col_ids)):
            # NOTE: only set the surface-subset of the k_us vector entries to avoid placing unnecessary zeros!
            self.k_us_vec[i].getSubVector(self.dofs_coupling_p[i], subvec=self.k_us_subvec[i])
            self.K_us.setValues(
                self.dofs_coupling_p[i],
                self.col_ids[i],
                self.k_us_subvec[i].array,
                addv=PETSc.InsertMode.INSERT,
            )
            self.k_us_vec[i].restoreSubVector(self.dofs_coupling_p[i], subvec=self.k_us_subvec[i])

        self.K_us.assemble()

        # set rows
        for i in range(len(self.row_ids)):
            # NOTE: only set the surface-subset of the k_su vector entries to avoid placing unnecessary zeros!
            self.k_su_vec[i].getSubVector(self.dofs_coupling_vq[i], subvec=self.k_su_subvec[i])
            self.K_su.setValues(
                self.row_ids[i],
                self.dofs_coupling_vq[i],
                self.k_su_subvec[i].array,
                addv=PETSc.InsertMode.INSERT,
            )
            self.k_su_vec[i].restoreSubVector(self.dofs_coupling_vq[i], subvec=self.k_su_subvec[i])

        self.K_su.assemble()

        self.K_list[0][1 + off] = self.K_us
        self.K_list[1 + off][0] = self.K_su

    def get_index_sets(self, isoptions={}):
        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            uvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            uvec_ls = self.rom.V.getLocalSize()[1]
        else:
            uvec_or0 = self.pbs.u.x.petsc_vec.getOwnershipRange()[0]
            uvec_ls = self.pbs.u.x.petsc_vec.getLocalSize()

        if self.coupling_type == "monolithic_direct":
            rvec = self.pb0.s
        if self.coupling_type == "monolithic_lagrange":
            rvec = self.LM

        offset_u = uvec_or0 + rvec.getOwnershipRange()[0]
        if self.pbs.incompressible_2field:
            offset_u += self.pbs.p.x.petsc_vec.getOwnershipRange()[0]
        iset_u = PETSc.IS().createStride(uvec_ls, first=offset_u, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            offset_p = offset_u + uvec_ls
            iset_p = PETSc.IS().createStride(
                self.pbs.p.x.petsc_vec.getLocalSize(),
                first=offset_p,
                step=1,
                comm=self.comm,
            )

        if self.pbs.incompressible_2field:
            offset_s = offset_p + self.pbs.p.x.petsc_vec.getLocalSize()
        else:
            offset_s = offset_u + uvec_ls

        iset_s = PETSc.IS().createStride(rvec.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            if isoptions["lms_to_p"]:
                iset_p = iset_p.expand(iset_s)  # add to pressure block
                ilist = [iset_u, iset_p]
            elif isoptions["lms_to_v"]:
                iset_u = iset_u.expand(iset_s)  # add to displacement block (could be bad...)
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

        if N > 0:
            if self.coupling_type == "monolithic_lagrange":
                self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname + "_lm", N, self.LM)
                self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname + "_lm", N, self.LM_old)

    def evaluate_initial(self):
        self.pbs.evaluate_initial()

        # set pressure functions for old state - s_old already initialized by 0D flow problem
        if self.coupling_type == "monolithic_direct":
            self.pb0.cardvasc0D.set_pressure_fem(
                self.pb0.s_old,
                self.pb0.cardvasc0D.v_ids,
                self.pr0D,
                self.coupfuncs_old,
            )

        if self.coupling_type == "monolithic_lagrange":
            self.pb0.cardvasc0D.set_pressure_fem(
                self.LM_old,
                list(range(self.num_coupling_surf)),
                self.pr0D,
                self.coupfuncs_old,
            )

        if self.coupling_type == "monolithic_direct":
            # old 3D coupling quantities (volumes or fluxes)
            for i in range(self.num_coupling_surf):
                cq = fem.assemble_scalar(self.cq_old_form[i])
                cq = self.comm.allgather(cq)
                self.pb0.c[i] = sum(cq) * self.cq_factor[i]

        if self.coupling_type == "monolithic_lagrange":
            for i in range(self.num_coupling_surf):
                LM_sq, lm_old_sq = (
                    allgather_vec(self.LM, self.comm),
                    allgather_vec(self.LM_old, self.comm),
                )
                self.pb0.c[i] = LM_sq[i]
                con = fem.assemble_scalar(self.cq_old_form[i])
                con = self.comm.allgather(con)
                self.constr[i] = sum(con) * self.cq_factor[i]
                self.constr_old[i] = sum(con) * self.cq_factor[i]

        # length of c from 3D-0D coupling
        self.pb0.len_c_3d0d = len(self.pb0.c)

        if bool(self.pb0.chamber_models):
            for i, ch in enumerate(["lv", "rv", "la", "ra"]):
                if self.pb0.chamber_models[ch]["type"] == "0D_elast":
                    self.pb0.y[i] = self.pbs.ti.timecurves(self.pb0.chamber_models[ch]["activation_curve"])(
                        self.pbase.t_init
                    )
                if self.pb0.chamber_models[ch]["type"] == "0D_elast_prescr":
                    self.pb0.y[i] = self.pbs.ti.timecurves(self.pb0.chamber_models[ch]["elastance_curve"])(
                        self.pbase.t_init
                    )
                if self.pb0.chamber_models[ch]["type"] == "0D_prescr":
                    self.pb0.c.append(
                        self.pbs.ti.timecurves(self.pb0.chamber_models[ch]["prescribed_curve"])(self.pbase.t_init)
                    )

        # if we have prescribed variable values over time
        if self.pbase.restart_step == 0:  # we read s and s_old in case of restart
            if bool(self.pb0.prescribed_variables):
                for a in self.pb0.prescribed_variables:
                    varindex = self.pb0.cardvasc0D.varmap[a]
                    prescr = self.pb0.prescribed_variables[a]
                    prtype = list(prescr.keys())[0]
                    if prtype == "val":
                        val = prescr["val"]
                    elif prtype == "curve":
                        curvenumber = prescr["curve"]
                        val = self.pb0.ti.timecurves(curvenumber)(self.pbase.t_init)
                    else:
                        raise ValueError("Unknown type to prescribe a variable.")
                    self.pb0.s[varindex], self.pb0.s_old[varindex] = val, val

        # initially evaluate 0D model at old state
        self.pb0.cardvasc0D.evaluate(
            self.pb0.s_old,
            self.pbase.t_init,
            self.pb0.df_old,
            self.pb0.f_old,
            None,
            None,
            self.pb0.c,
            self.pb0.y,
            self.pb0.aux_old,
        )
        self.pb0.auxTc_old[:] = self.pb0.aux_old[:]

    def write_output_ini(self):
        self.pbs.write_output_ini()

    def write_output_pre(self):
        self.pbs.write_output_pre()
        self.pb0.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbs.evaluate_pre_solve(t, N, dt)
        self.pb0.evaluate_pre_solve(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbs.evaluate_post_solve(t, N)
        self.pb0.evaluate_post_solve(t, N)

        if self.have_multiscale_gandr:
            self.set_homeostatic_threshold(t), self.set_growth_trigger(t)

    def set_output_state(self, t):
        self.pbs.set_output_state(t)
        self.pb0.set_output_state(t)

    def write_output(self, N, t, mesh=False):
        self.pbs.write_output(N, t)
        self.pb0.write_output(N, t)

        if self.coupling_type == "monolithic_lagrange":
            if self.pbs.io.write_results_every > 0 and N % self.pbs.io.write_results_every == 0:
                if np.isclose(t, self.pbase.dt):
                    mode = "wt"
                else:
                    mode = "a"
                LM_sq = allgather_vec(self.LM, self.comm)
                if self.comm.rank == 0:
                    for i in range(len(LM_sq)):
                        f = open(
                            self.pbase.output_path + "/results_" + self.pbase.simname + "_LM" + str(i + 1) + ".txt",
                            mode,
                        )
                        f.write("%.16E %.16E\n" % (t, LM_sq[i]))
                        f.close()
                del LM_sq

    def update(self):
        # update time step - solid and 0D model
        self.pbs.update()
        self.pb0.update()

        # update old pressures on solid
        if self.coupling_type == "monolithic_direct":
            self.pb0.cardvasc0D.set_pressure_fem(
                self.pb0.s_old,
                self.pb0.cardvasc0D.v_ids,
                self.pr0D,
                self.coupfuncs_old,
            )
        if self.coupling_type == "monolithic_lagrange":
            self.LM_old.axpby(1.0, 0.0, self.LM)
            self.pb0.cardvasc0D.set_pressure_fem(
                self.LM_old,
                list(range(self.num_coupling_surf)),
                self.pr0D,
                self.coupfuncs_old,
            )
            # update old 3D fluxes
            self.constr_old[:] = self.constr[:]

    def print_to_screen(self):
        self.pbs.print_to_screen()
        self.pb0.print_to_screen()

        if self.coupling_type == "monolithic_lagrange":
            LM_sq = allgather_vec(self.LM, self.comm)
            for i in range(self.num_coupling_surf):
                utilities.print_status("LM" + str(i + 1) + " = %.4e" % (LM_sq[i]), self.comm)
            del LM_sq

    def induce_state_change(self):
        self.pbs.induce_state_change()
        self.pb0.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.pbs.write_restart(sname, N, force=force)
        self.pb0.write_restart(sname, N, force=force)

        if self.coupling_type == "monolithic_lagrange":
            if (self.pbs.io.write_restart_every > 0 and N % self.pbs.io.write_restart_every == 0) or force:
                LM_sq = allgather_vec(self.LM, self.comm)
                if self.comm.rank == 0:
                    f = open(
                        self.pb0.output_path_0D + "/checkpoint_" + sname + "_lm_" + str(N) + ".txt",
                        "wt",
                    )
                    for i in range(len(LM_sq)):
                        f.write("%.16E\n" % (LM_sq[i]))
                    f.close()
                del LM_sq

    def check_abort(self, t):
        return self.pb0.check_abort(t)

    def destroy(self):
        self.pbs.destroy()
        self.pb0.destroy()

        for i in range(len(self.col_ids)):
            self.k_us_vec[i].destroy()
        for i in range(len(self.row_ids)):
            self.k_su_vec[i].destroy()


class SolidmechanicsFlow0DSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbs.pre)
        self.pb.set_problem_vector_matrix_structures()

        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlin.solver_nonlinear_ode([self.pb.pb0], self.solver_params["subsolver_params"])
        else:
            self.subsol = None

        self.evaluate_assemble_system_initial(subsolver=self.subsol)

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params, subsolver=self.subsol)

        if self.pb.pbs.prestress_initial or self.pb.pbs.prestress_initial_only:
            # initialize solid mechanics solver
            solver_params_prestr = copy.deepcopy(self.solver_params)
            # modify solver parameters in case user specified alternating ones for prestressing (should do, because it's a 2x2 problem maximum)
            try:
                solver_params_prestr["solve_type"] = self.solver_params["solve_type_prestr"]
            except:
                pass
            try:
                solver_params_prestr["block_precond"] = self.solver_params["block_precond_prestr"]
            except:
                pass
            try:
                solver_params_prestr["precond_fields"] = self.solver_params["precond_fields_prestr"]
            except:
                pass
            self.solverprestr = SolidmechanicsSolverPrestr(self.pb.pbs, solver_params_prestr)

    def solve_initial_state(self):
        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.pre:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()

        # consider consistent initial acceleration
        if self.pb.pbs.timint != "static" and self.pb.pbase.restart_step == 0 and not self.pb.restart_multiscale:
            ts = time.time()
            utilities.print_status(
                "Setting forms and solving for consistent initial acceleration...",
                self.pb.comm,
                e=" ",
            )

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = (
                self.pb.pbs.deltaW_kin_old
                + self.pb.pbs.deltaW_int_old
                - self.pb.pbs.deltaW_ext_old
                - self.pb.work_coupling_old
            )

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du)  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbs.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t, localdata=self.pb.pbs.localdata)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbase.numstep, ni=ni, li=li, wt=wt)
