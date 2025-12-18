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
from .. import utilities, ioparams, meshutils

from ..fluid.fluid_main import (
    FluidmechanicsProblem,
    FluidmechanicsSolverPrestr,
)
from ..phasefield.phasefield_main import PhasefieldProblem
from ..base import problem_base, solver_base


class FluidmechanicsPhasefieldProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params_fluid,
        time_params_phasefield,
        fem_params_fluid,
        fem_params_phasefield,
        constitutive_models_fluid,
        constitutive_models_phasefield,
        bc_dict_fluid,
        bc_dict_phasefield,
        time_curves,
        io,
        mor_params={},
        pbf=None,
        pbp=None,
        is_ale=False,
    ):
        self.pbase = pbase
        self.pbf = pbf
        self.pbp = pbp

        # pointer to communicator
        self.comm = self.pbase.comm

        self.problem_physics = "fluid_phasefield"

        # instantiate problem classes
        # fluid
        if pbf is None:
            self.pbf = FluidmechanicsProblem(
                pbase,
                io_params,
                time_params_fluid,
                fem_params_fluid,
                constitutive_models_fluid,
                bc_dict_fluid,
                time_curves,
                io,
                mor_params=mor_params,
                is_ale=is_ale,
                is_multiphase=True,
            )
        # phase field
        if pbp is None:
            self.pbp = PhasefieldProblem(
                pbase,
                io_params,
                time_params_phasefield,
                fem_params_phasefield,
                constitutive_models_phasefield,
                bc_dict_phasefield,
                time_curves,
                io,
                mor_params=mor_params,
                is_ale=is_ale,
                is_advected=True,
            )
        self.pbf.phasevar["phi"] = self.pbp.phi
        self.pbf.phasevar["phi_old"] = self.pbp.phi_old
        self.pbf.phasevar["phidot"] = self.pbp.phidot_expr
        self.pbf.phasevar["phidot_old"] = self.pbp.phidot_old

        self.pbp.fluidvar["v"] = self.pbf.v
        self.pbp.fluidvar["v_old"] = self.pbf.v_old

        self.set_coupling_parameters()

        self.pbrom = self.pbf  # ROM problem can only be fluid
        self.pbrom_host = self

        # modify results to write...
        self.pbf.results_to_write = io_params["results_to_write"][0]
        self.pbp.results_to_write = io_params["results_to_write"][1]

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.sub_solve = False
        self.print_subiter = False
        self.have_condensed_variables = False

        self.numdof = self.pbf.numdof + self.pbp.numdof

        self.io = self.pbf.io

        # number of fields involved
        self.nfields = 4

        # residual and matrix lists
        self.r_list, self.r_list_rom = (
            [None] * self.nfields,
            [None] * self.nfields,
        )
        self.K_list, self.K_list_rom = (
            [[None] * self.nfields for _ in range(self.nfields)],
            [[None] * self.nfields for _ in range(self.nfields)],
        )

    def set_coupling_parameters(self):
        pass # up to now, nothing to be set...

    def get_problem_var_list(self):
        if self.pbf.num_dupl > 1:
            is_ghosted = [1, 2, 1, 1]
        else:
            is_ghosted = [1, 1, 1, 1]
        return [
            self.pbf.v.x.petsc_vec,
            self.pbf.p.x.petsc_vec,
            self.pbp.phi.x.petsc_vec,
            self.pbp.mu.x.petsc_vec,
        ], is_ghosted

    # defines the monolithic coupling forms for fluid mechanics in ALE reference frame
    def set_variational_forms(self):
        self.pbf.set_variational_forms()
        self.pbp.set_variational_forms()
        self.set_variational_forms_coupling()

    def set_variational_forms_coupling(self):
        # add Korteweg force to fluid momentum
        self.korteweg_force, self.korteweg_force_old, self.korteweg_force_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

        for n, M in enumerate(self.pbf.domain_ids):
            self.korteweg_force += self.pbf.vf.korteweg_force1(self.pbp.phi, self.pbp.mu, self.pbf.dx(M), F=self.pbf.alevar["Fale"])
            self.korteweg_force_old += self.pbf.vf.korteweg_force1(self.pbp.phi_old, self.pbp.mu_old, self.pbf.dx(M), F=self.pbf.alevar["Fale_old"])
            self.korteweg_force_mid += self.pbf.vf.korteweg_force1(self.pbp.phi_mid, self.pbp.mu_mid, self.pbf.dx(M), F=self.pbf.alevar["Fale_mid"])

        # add to fluid momentum
        if self.pbf.ti.eval_nonlin_terms == "trapezoidal":
            self.pbf.weakform_v += self.pbf.timefac * self.korteweg_force + (1.0 - self.pbf.timefac) * self.korteweg_force_old
        if self.pbf.ti.eval_nonlin_terms == "midpoint":
            self.pbf.weakform_v += self.korteweg_force_mid

        # derivative of fluid momentum w.r.t. phase field
        self.weakform_lin_vphi = ufl.derivative(self.pbf.weakform_v, self.pbp.phi, self.pbp.dphi)
        # derivative of fluid momentum w.r.t. potential
        self.weakform_lin_vmu = ufl.derivative(self.pbf.weakform_v, self.pbp.mu, self.pbp.dmu)
        # derivative of fluid continuity w.r.t. phase field
        self.weakform_lin_pphi = []
        for n in range(self.pbf.num_domains):
            self.weakform_lin_pphi.append(ufl.derivative(self.pbf.weakform_p[n], self.pbp.phi, self.pbp.dphi))
        # derivative of phase field w.r.t. fluid velocity
        self.weakform_lin_phiv = ufl.derivative(self.pbp.weakform_phi, self.pbf.v, self.pbf.dv)

    def set_problem_residual_jacobian_forms(self, pre=False):
        # fluid + pahsefield
        self.pbf.set_problem_residual_jacobian_forms(pre=pre)
        self.pbp.set_problem_residual_jacobian_forms()
        self.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        ts = time.time()
        utilities.print_status(
            "FEM form compilation for fluid-phasefield coupling...",
            self.comm,
            e=" ",
        )

        self.jac_vphi = fem.form(self.weakform_lin_vphi, entity_maps=self.io.entity_maps)
        self.jac_vmu = fem.form(self.weakform_lin_vmu, entity_maps=self.io.entity_maps)
        self.jac_phiv = fem.form(self.weakform_lin_phiv, entity_maps=self.io.entity_maps)

        if not bool(self.pbf.io.duplicate_mesh_domains):
            self.weakform_lin_pphi = sum(self.weakform_lin_pphi)

        self.jac_pphi = fem.form(self.weakform_lin_pphi, entity_maps=self.io.entity_maps)
        if self.pbf.num_dupl > 1:
            self.jac_pphi_ = []
            for j in range(self.pbf.num_dupl):
                self.jac_pphi_.append([self.jac_pphi[j]])

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        self.pbf.set_problem_vector_matrix_structures()
        self.pbp.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        self.K_vphi = fem.petsc.assemble_matrix(self.jac_vphi)
        self.K_vphi.assemble()
        self.K_vmu = fem.petsc.assemble_matrix(self.jac_vmu)
        self.K_vmu.assemble()
        self.K_phiv = fem.petsc.assemble_matrix(self.jac_phiv)
        self.K_phiv.assemble()
        if self.pbf.num_dupl > 1:
            self.K_pphi = fem.petsc.assemble_matrix(self.jac_pphi_)
        else:
            self.K_pphi = fem.petsc.assemble_matrix(self.jac_pphi)
        self.K_pphi.assemble()

    def assemble_residual(self, t, subsolver=None):
        self.pbf.assemble_residual(t)
        self.pbp.assemble_residual(t)
        self.assemble_residual_coupling(t)

        self.r_list[0] = self.pbf.r_list[0]
        self.r_list[1] = self.pbf.r_list[1]
        self.r_list[2] = self.pbp.r_list[0]
        self.r_list[3] = self.pbp.r_list[1]

    def assemble_residual_coupling(self, t, subsolver=None):
        pass

    def assemble_stiffness(self, t, subsolver=None):
        self.assemble_stiffness_coupling(t)

        self.pbf.assemble_stiffness(t)
        self.pbp.assemble_stiffness(t)

        self.K_list[0][0] = self.pbf.K_list[0][0]
        self.K_list[0][1] = self.pbf.K_list[0][1]
        self.K_list[1][0] = self.pbf.K_list[1][0]
        self.K_list[1][1] = self.pbf.K_list[1][1]

        self.K_list[2][2] = self.pbp.K_list[0][0]
        self.K_list[2][3] = self.pbp.K_list[0][1]

        self.K_list[3][2] = self.pbp.K_list[1][0]
        self.K_list[3][3] = self.pbp.K_list[1][1]

    def assemble_stiffness_coupling(self, t):
        # derivative of fluid momentum w.r.t. phase field
        self.K_vphi.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vphi, self.jac_vphi, self.pbf.bc.dbcs)
        self.K_vphi.assemble()

        self.K_list[0][2] = self.K_vphi

        # derivative of fluid momentum w.r.t. potential
        self.K_vmu.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vmu, self.jac_vmu, self.pbf.bc.dbcs)
        self.K_vmu.assemble()

        self.K_list[0][3] = self.K_vmu

        # derivative of fluid continuity w.r.t. phase field
        self.K_pphi.zeroEntries()
        fem.petsc.assemble_matrix(self.K_pphi, self.jac_pphi, [])
        self.K_pphi.assemble()

        self.K_list[1][2] = self.K_pphi

        # derivative of phase field w.r.t. fluid velocity
        self.K_phiv.zeroEntries()
        fem.petsc.assemble_matrix(self.K_phiv, self.jac_phiv, self.pbp.bc.dbcs)
        self.K_phiv.assemble()

        self.K_list[2][0] = self.K_phiv

    def get_solver_index_sets(self, isoptions={}):
        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.x.petsc_vec.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.x.petsc_vec.getLocalSize()

        offset_v = (
            vvec_or0 + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
        )
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions["rom_to_new"]:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r)  # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(
            self.pbf.p.x.petsc_vec.getLocalSize(),
            first=offset_p,
            step=1,
            comm=self.comm,
        )

        offset_d = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_d = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_d,
            step=1,
            comm=self.comm,
        )

        if isoptions["ale_to_v"]:
            iset_v = iset_v.expand(iset_d)  # add ALE to velocity block

        if isoptions["rom_to_new"]:
            ilist = [iset_v, iset_p, iset_r, iset_d]
        else:
            ilist = [iset_v, iset_p, iset_d]

        if isoptions["ale_to_v"]:
            ilist.pop(-1)

        return ilist

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if N > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        self.pbf.evaluate_initial()
        self.pbp.evaluate_initial()

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

    def write_output_pre(self):
        self.pbf.write_output_pre()
        self.pbp.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbf.evaluate_pre_solve(t, N, dt)
        self.pbp.evaluate_pre_solve(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbf.evaluate_post_solve(t, N)
        self.pbp.evaluate_post_solve(t, N)

    def set_output_state(self, N):
        self.pbf.set_output_state(N)
        self.pbp.set_output_state(N)

    def write_output(self, N, t, mesh=False):
        self.io.write_output(self, N=N, t=t)

    def update(self):
        # update time step - fluid and ALE
        self.pbf.update()
        self.pbp.update()

    def print_to_screen(self):
        self.pbf.print_to_screen()
        self.pbp.print_to_screen()

    def induce_state_change(self):
        self.pbf.induce_state_change()
        self.pbp.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.io.write_restart(self, N, force=force)

    def check_abort(self, t):
        return False

    def destroy(self):
        self.pbf.destroy()
        self.pbp.destroy()


class FluidmechanicsPhasefieldSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbf.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

    def solve_initial_state(self):
        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the FrSI problem
        if self.pb.pbf.pre:
            # solve reduced-solid/FrSI prestress problem
            self.solverprestr.solve_initial_prestress()

        # consider consistent initial acceleration
        if (
            self.pb.pbf.fluid_governing_type == "navierstokes_transient"
            or self.pb.pbf.fluid_governing_type == "stokes_transient"
        ) and self.pb.pbase.restart_step == 0:
            ts = time.time()
            utilities.print_status(
                "Setting forms and solving for consistent initial acceleration...",
                self.pb.comm,
                e=" ",
            )

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv)  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa = (
                fem.form(weakform_a, entity_maps=self.pb.io.entity_maps),
                fem.form(weakform_lin_aa, entity_maps=self.pb.io.entity_maps),
            )
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
