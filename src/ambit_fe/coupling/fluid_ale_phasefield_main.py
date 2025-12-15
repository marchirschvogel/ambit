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
from .. import ioparams
from .. import utilities, meshutils
from ..mpiroutines import allgather_vec

from .fluid_ale_main import FluidmechanicsAleProblem
from .fluid_phasefield_main import FluidmechanicsPhasefieldProblem
from ..fluid.fluid_main import FluidmechanicsProblem
from ..fluid.fluid_main import FluidmechanicsSolverPrestr
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base


class FluidmechanicsAlePhasefieldProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params_fluid,
        time_params_phasefield,
        fem_params_fluid,
        fem_params_phasefield,
        fem_params_ale,
        constitutive_models_fluid,
        constitutive_models_phasefield,
        constitutive_models_ale,
        bc_dict_fluid,
        bc_dict_phasefield,
        bc_dict_ale,
        time_curves,
        coupling_params_fluid_ale,
        io,
        mor_params={},
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_coupling_fluid_ale(coupling_params_fluid_ale)

        self.problem_physics = "fluid_ale_phasefield"

        # instantiate problem classes
        # fluid-ALE
        self.pbfa = FluidmechanicsAleProblem(
            pbase,
            io_params,
            time_params_fluid,
            fem_params_fluid,
            fem_params_ale,
            constitutive_models_fluid,
            constitutive_models_ale,
            bc_dict_fluid,
            bc_dict_ale,
            time_curves,
            coupling_params_fluid_ale,
            io,
            mor_params=mor_params,
            is_multiphase=True,
        )

        # fluid-phasefield
        self.pbfp = FluidmechanicsPhasefieldProblem(
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
            mor_params=mor_params,
            is_ale=True,
            pbf=self.pbfa.pbf
        )

        self.pbf = self.pbfa.pbf
        self.pba = self.pbfa.pba
        self.pbp = self.pbfp.pbp

        self.pbrom = self.pbf  # ROM problem can only be fluid
        self.pbrom_host = self

        # modify results to write...
        self.pbf.results_to_write = io_params["results_to_write"][0]
        self.pba.results_to_write = io_params["results_to_write"][1]

        self.sub_solve = False
        self.print_subiter = False
        self.have_condensed_variables = False

        self.io = io

        self.set_coupling_parameters()

        self.numdof = self.pbf.numdof + self.pba.numdof + self.pbp.numdof

        self.localsolve = False

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
        pass

    def get_problem_var_list(self):
        if self.pbf.num_dupl > 1:
            is_ghosted = [1, 2, 0, 1]
        else:
            is_ghosted = [1, 1, 0, 1]
        return [
            self.pbf.v.x.petsc_vec,
            self.pbf.p.x.petsc_vec,
            self.pbfc.LM,
            self.pba.d.x.petsc_vec,
        ], is_ghosted

    def set_variational_forms(self):
        self.pbfa.set_variational_forms()
        self.pbfc.set_variational_forms_coupling()
        self.set_variational_forms_coupling()

    def set_variational_forms_coupling(self):
        pass

    def set_problem_residual_jacobian_forms(self, pre=False):
        # fluid + ALE
        self.pbfa.set_problem_residual_jacobian_forms(pre=pre)
        # fluid + constraint
        self.pbfc.set_problem_residual_jacobian_forms_coupling()
        # now the additional ALE-0D coupling terms (in case of moving in-/outflow boundaries from/to 0D model)
        self.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        pass

    def set_problem_vector_matrix_structures(self):
        self.pbfa.set_problem_vector_matrix_structures()
        self.pbfc.set_problem_vector_matrix_structures_coupling()

        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        if self.pbfa.coupling_strategy == "monolithic":
            # setup offdiagonal matrix
            locmatsize = self.pba.V_d.dofmap.index_map.size_local * self.pba.V_d.dofmap.index_map_bs
            matsize = self.pba.V_d.dofmap.index_map.size_global * self.pba.V_d.dofmap.index_map_bs

            self.k_sd_vec = []
            for i in range(len(self.pbfc.row_ids)):
                self.k_sd_vec.append(fem.petsc.assemble_vector(self.dcqd_form[i]))

            self.dofs_coupling_vq = [[]] * self.pbfc.num_coupling_surf

            self.k_sd_subvec, sze_sd = [], []

            for n in range(self.pbfc.num_coupling_surf):
                self.dofs_coupling_vq[n] = meshutils.get_index_set_id(self.pba.io, self.pba.V_d, self.pbfc.surface_vq_ids[n], self.pba.io.mesh.topology.dim-1, self.comm)

                self.k_sd_subvec.append(self.k_sd_vec[n].getSubVector(self.dofs_coupling_vq[n]))

                sze_sd.append(self.k_sd_subvec[-1].getSize())

            # derivative of multiplier constraint w.r.t. fluid velocities
            self.K_sd = PETSc.Mat().createAIJ(
                size=(
                    (PETSc.DECIDE, self.pbfc.num_coupling_surf),
                    (locmatsize, matsize),
                ),
                bsize=None,
                nnz=max(sze_sd),
                csr=None,
                comm=self.comm,
            )
            self.K_sd.setUp()
            self.K_sd.setOption(PETSc.Mat.Option.ROW_ORIENTED, False)

    def assemble_residual(self, t, subsolver=None):
        self.pbfc.assemble_residual_coupling(t)
        self.pbfa.assemble_residual(t)

        # fluid
        self.r_list[0] = self.pbfa.r_list[0]
        self.r_list[1] = self.pbfa.r_list[1]
        # constraints
        self.r_list[2] = self.pbfc.r_list[2]
        # ALE
        self.r_list[3] = self.pbfa.r_list[2]

    def assemble_stiffness(self, t, subsolver=None):
        self.pbfc.assemble_stiffness_coupling(t)
        self.pbfa.assemble_stiffness(t)

        # fluid - momentum
        self.K_list[0][0] = self.pbfa.K_list[0][0]
        self.K_list[0][1] = self.pbfa.K_list[0][1]
        self.K_list[0][2] = self.pbfc.K_vs
        self.K_list[0][3] = self.pbfa.K_list[0][2]
        # fluid - continuity
        self.K_list[1][0] = self.pbfa.K_list[1][0]
        self.K_list[1][1] = self.pbfa.K_list[1][1]
        self.K_list[1][3] = self.pbfa.K_list[1][2]
        # ...

        self.assemble_stiffness_coupling(t)

    def assemble_stiffness_coupling(self, t):
        pass

    def get_index_sets(self, isoptions={}):
        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.x.petsc_vec.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.x.petsc_vec.getLocalSize()

        offset_v = (
            vvec_or0
            + self.pbf.p.x.petsc_vec.getOwnershipRange()[0]
            + self.pbfc.LM.getOwnershipRange()[0]
            + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
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

        offset_s = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_s = PETSc.IS().createStride(self.pbfc.LM.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        offset_d = offset_s + self.pbfc.LM.getLocalSize()
        iset_d = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_d,
            step=1,
            comm=self.comm,
        )

        if isoptions["rom_to_new"]:
            iset_s = iset_s.expand(iset_r)  # add to 0D block
            iset_s.sort()  # should be sorted, otherwise PETSc may struggle to extract block

        if isoptions["ale_to_v"]:
            iset_v = iset_v.expand(iset_d)  # add ALE to velocity block

        if isoptions["lms_to_p"]:
            iset_p = iset_p.expand(
                iset_s
            )  # add to pressure block - attention: will merge ROM to this block too in case of 'rom_to_new' is True!
            ilist = [iset_v, iset_p, iset_d]
        elif isoptions["lms_to_v"]:
            iset_v = iset_v.expand(
                iset_s
            )  # add to velocity block (could be bad...) - attention: will merge ROM to this block too in case of 'rom_to_new' is True!
            ilist = [iset_v, iset_p, iset_d]
        else:
            ilist = [iset_v, iset_p, iset_s, iset_d]

        if isoptions["ale_to_v"]:
            ilist.pop(-1)

        return ilist

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # fluid-ALE + flow0d problem
        if N > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        self.pbfa.evaluate_initial()
        self.pbfc.evaluate_initial_coupling()

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

    def write_output_pre(self):
        self.pbfa.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbfa.evaluate_pre_solve(t, N, dt)
        self.pbfc.evaluate_pre_solve_coupling(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbfa.evaluate_post_solve(t, N)

    def set_output_state(self, N):
        self.pbfa.set_output_state(N)

    def write_output(self, N, t, mesh=False):
        self.io.write_output(self, N=N, t=t)  # combined fluid-ALE output routine

    def update(self):
        # update time step - fluid+flow0d and ALE
        self.pbfa.update()
        self.pbfc.update_coupling()

    def print_to_screen(self):
        self.pbfa.print_to_screen()
        self.pbfc.print_to_screen_coupling()

    def induce_state_change(self):
        self.pbfa.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.io.write_restart(self, N, force=force)

    def check_abort(self, t):
        return self.pbfc.check_abort(t)

    def destroy(self):
        self.pbfa.destroy()
        self.pbfc.destroy_coupling()
        self.destroy_coupling()

    def destroy_coupling(self):
        pass


class FluidmechanicsAlePhasefieldSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbf.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

        if self.pb.pbf.prestress_initial or self.pb.pbf.prestress_initial_only:
            solver_params_prestr = copy.deepcopy(self.solver_params)
            # modify solver parameters in case user specified alternating ones for prestressing (should do, because it's a 2x2 problem)
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
            # initialize fluid mechanics solver
            self.solverprestr = FluidmechanicsSolverPrestr(self.pb.pbf, solver_params_prestr)

    def solve_initial_state(self):
        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
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
            weakform_a = (
                self.pb.pbf.deltaW_kin_old
                + self.pb.pbf.deltaW_int_old
                - self.pb.pbf.deltaW_ext_old
            )

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
