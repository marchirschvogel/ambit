#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import ioparams
from .. import utilities
from .. import boundaryconditions
from ..mpiroutines import allgather_vec

from .fsi_main import FSIProblem
from .fluid_ale_main import FluidmechanicsAleProblem
from .fluid_flow0d_main import FluidmechanicsFlow0DProblem
from .fluid_ale_flow0d_main import FluidmechanicsAleFlow0DProblem
from ..fluid.fluid_main import FluidmechanicsProblem
from ..solid.solid_main import SolidmechanicsProblem
from ..flow0d.flow0d_main import Flow0DProblem
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base


class FSIFlow0DProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params_solid,
        time_params_fluid,
        time_params_flow0d,
        fem_params_solid,
        fem_params_fluid,
        fem_params_ale,
        constitutive_models_solid,
        constitutive_models_fluid_ale,
        model_params_flow0d,
        bc_dict_solid,
        bc_dict_fluid_ale,
        time_curves,
        coupling_params_fluid_ale,
        coupling_params_fluid_flow0d,
        io,
        ios,
        iof,
        mor_params={},
        is_multiphase=False,
    ):
        # problem_base.__init__(self, io_params, time_params_solid, comm=comm, comm_sq=comm_sq)

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_coupling_fluid_ale(coupling_params_fluid_ale)

        self.problem_physics = "fsi_flow0d"

        self.io = io
        self.ios, self.iof = ios, iof

        self.have_condensed_variables = False

        # instantiate problem classes
        # FSI - fluid-ALE-solid
        self.pbfas = FSIProblem(
            pbase,
            io_params,
            time_params_solid,
            time_params_fluid,
            fem_params_solid,
            fem_params_fluid,
            fem_params_ale,
            constitutive_models_solid,
            constitutive_models_fluid_ale,
            bc_dict_solid,
            bc_dict_fluid_ale,
            time_curves,
            coupling_params_fluid_ale,
            io,
            ios,
            iof,
            mor_params=mor_params,
            is_multiphase=is_multiphase,
        )
        # fluid-ALE-0D
        self.pbfa0 = FluidmechanicsAleFlow0DProblem(
            pbase,
            io_params,
            time_params_fluid,
            time_params_flow0d,
            fem_params_fluid,
            fem_params_ale,
            constitutive_models_fluid_ale[0],
            constitutive_models_fluid_ale[1],
            model_params_flow0d,
            bc_dict_fluid_ale[0],
            bc_dict_fluid_ale[1],
            time_curves,
            coupling_params_fluid_ale,
            coupling_params_fluid_flow0d,
            iof,
            mor_params=mor_params,
            is_multiphase=is_multiphase,
            pbfa=self.pbfas.pbfa,
        )

        self.pbs = self.pbfas.pbs
        self.pbf = self.pbfas.pbfa.pbf
        self.pba = self.pbfas.pbfa.pba

        self.pbf0 = self.pbfa0.pbf0
        self.pb0 = self.pbfa0.pb0

        self.pbrom = self.pbs  # ROM problem can only be solid
        self.pbrom_host = self

        # modify results to write...
        self.pbs.results_to_write = io_params["results_to_write"][0]
        self.pbf.results_to_write = io_params["results_to_write"][1][0]
        self.pba.results_to_write = io_params["results_to_write"][1][1]

        self.incompressible_2field = self.pbs.incompressible_2field
        self.fsi_system = self.pbfas.fsi_system

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.print_subiter = self.pbf0.print_subiter

        self.numdof = self.pbs.numdof + self.pbf.numdof + self.pba.numdof + self.pbf0.LM.getSize()
        if self.pbfas.fsi_system == "neumann_neumann":
            self.numdof += self.pbfas.lm.x.petsc_vec.getSize()

        self.sub_solve = True

        # number of fields involved
        if self.pbfas.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                self.nfields = 7
            else:
                self.nfields = 6
        else:
            if self.pbs.incompressible_2field:
                self.nfields = 6
            else:
                self.nfields = 5

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
        if self.pbfas.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 1, 2, 1, 0, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1, 0, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbfas.lm.x.petsc_vec,
                    self.pbf0.LM,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
            else:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 2, 1, 0, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 0, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbfas.lm.x.petsc_vec,
                    self.pbf0.LM,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
        else:
            if self.pbs.incompressible_2field:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 1, 2, 0, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 0, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbf0.LM,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
            else:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 2, 0, 1]
                else:
                    is_ghosted = [1, 1, 1, 0, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbf0.LM,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted

    def set_variational_forms(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfas.set_variational_forms()
        # fluid-0D, ALE-0D coup
        self.pbf0.set_variational_forms_coupling()
        self.pbfa0.set_variational_forms_coupling()

    def set_variational_forms_coupling(self):
        pass # no additional coupling forms needed

    def set_problem_residual_jacobian_forms(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfas.set_problem_residual_jacobian_forms()
        # fluid-0D, ALE-0D coup
        self.pbf0.set_problem_residual_jacobian_forms_coupling()
        self.pbfa0.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        pass

    def set_problem_vector_matrix_structures(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfas.set_problem_vector_matrix_structures()
        # fluid-0D, ALE-0D coup
        self.pbf0.set_problem_vector_matrix_structures_coupling()
        self.pbfa0.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        pass

    def assemble_residual(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            ofs = 1
        else:
            ofs = 0
        if self.pbfas.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        # fluid-0D coup - prior to fluid assemble!
        self.pbf0.assemble_residual_coupling(t, subsolver=subsolver)
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfas.assemble_residual(t)

        # solid
        self.r_list[0] = self.pbfas.r_list[0]
        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbfas.r_list[1]
        # fluid
        self.r_list[1 + ofs] = self.pbfas.r_list[1 + ofs]
        self.r_list[2 + ofs] = self.pbfas.r_list[2 + ofs]
        if self.pbfas.fsi_system == "neumann_neumann":
            self.r_list[3 + ofs] = self.pbfas.r_list[3 + ofs]
        # 3D-0D constraint
        self.r_list[3 + ofc + ofs] = self.pbf0.r_list[2]
        # ALE
        self.r_list[4 + ofc + ofs] = self.pbfas.r_list[3 + ofc + ofs]

    def assemble_stiffness(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            ofs = 1
        else:
            ofs = 0
        if self.pbfas.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        # fluid-0D coup - prior to fluid assemble!
        self.pbf0.assemble_stiffness_coupling(t, subsolver=subsolver)
        # ALE-0D
        self.pbfa0.assemble_stiffness_coupling(t)
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfas.assemble_stiffness(t)

        # solid momentum
        self.K_list[0][0] = self.pbfas.K_list[0][0]
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbfas.K_list[0][1]
        if self.pbfas.fsi_system == "neumann_neumann":
            self.K_list[0][3 + ofs] = self.pbfas.K_list[0][3 + ofs]
        if self.pbfas.fsi_system == "neumann_dirichlet":
            self.K_list[0][1 + ofs] = self.pbfas.K_list[0][1 + ofs]
        # solid incompressibility
        if self.pbs.incompressible_2field:
            self.K_list[1][0] = self.pbfas.K_list[1][0]
            self.K_list[1][1] = self.pbfas.K_list[1][1]
            if self.pbfas.fsi_system == "neumann_dirichlet":
                self.K_list[1 + ofs][1] = self.pbfas.K_list[1 + ofs][1]
        # fluid momentum
        self.K_list[1 + ofs][1 + ofs] = self.pbfas.K_list[1 + ofs][1 + ofs]
        self.K_list[1 + ofs][2 + ofs] = self.pbfas.K_list[1 + ofs][2 + ofs]
        if self.pbfas.fsi_system == "neumann_neumann":
            self.K_list[1 + ofs][3 + ofs] = self.pbfas.K_list[1 + ofs][3 + ofs]
        if self.pbfas.fsi_system == "neumann_dirichlet":
            self.K_list[1 + ofs][0] = self.pbfas.K_list[1 + ofs][0]
        self.K_list[1 + ofs][3 + ofc + ofs] = self.pbf0.K_vs
        self.K_list[1 + ofs][4 + ofc + ofs] = self.pbfas.K_list[1 + ofs][3 + ofc + ofs]
        # fluid continuity
        self.K_list[2 + ofs][1 + ofs] = self.pbfas.K_list[2 + ofs][1 + ofs]
        self.K_list[2 + ofs][2 + ofs] = self.pbfas.K_list[2 + ofs][2 + ofs]
        self.K_list[2 + ofs][4 + ofc + ofs] = self.pbfas.K_list[2 + ofs][3 + ofc + ofs]
        # FSI coupling constraint
        if self.pbfas.fsi_system == "neumann_neumann":
            self.K_list[3 + ofs][0] = self.pbfas.K_list[3 + ofs][0]
            self.K_list[3 + ofs][1 + ofs] = self.pbfas.K_list[3 + ofs][1 + ofs]
        # 3D-0D
        self.K_list[3 + ofc + ofs][1 + ofs] = self.pbf0.K_sv
        self.K_list[3 + ofc + ofs][3 + ofc + ofs] = self.pbf0.K_lm
        self.K_list[3 + ofc + ofs][4 + ofc + ofs] = self.pbfa0.K_sd
        # ALE
        self.K_list[4 + ofc + ofs][4 + ofc + ofs] = self.pbfas.K_list[3 + ofc + ofs][3 + ofc + ofs]
        self.K_list[4 + ofc + ofs][1 + ofs] = self.pbfas.K_list[3 + ofc + ofs][1 + ofs]


    def get_solver_index_sets(self, isoptions={}):
        # iterative solvers here are only implemented for neumann_dirichlet system!
        assert(self.pbfas.fsi_system == "neumann_dirichlet")

        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            uvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            uvec_ls = self.rom.V.getLocalSize()[1]
        else:
            uvec_or0 = self.pbs.u.x.petsc_vec.getOwnershipRange()[0]
            uvec_ls = self.pbs.u.x.petsc_vec.getLocalSize()

        offset_u = uvec_or0 + self.pbf.v.x.petsc_vec.getOwnershipRange()[0] + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.pbf0.LM.getOwnershipRange()[0] + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
        if self.pbs.incompressible_2field:
            offset_u += self.pbs.p.x.petsc_vec.getOwnershipRange()[0]
        iset_u = PETSc.IS().createStride(uvec_ls, first=offset_u, step=1, comm=self.comm)

        if self.pbs.incompressible_2field:
            offset_ps = offset_u + uvec_ls
            iset_ps = PETSc.IS().createStride(
                self.pbs.p.x.petsc_vec.getLocalSize(),
                first=offset_ps,
                step=1,
                comm=self.comm,
            )

        if self.pbs.incompressible_2field:
            offset_v = offset_ps + self.pbs.p.x.petsc_vec.getLocalSize()
        else:
            offset_v = offset_u + uvec_ls

        iset_v = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_v,
            step=1,
            comm=self.comm)

        offset_p = offset_v + self.pbf.v.x.petsc_vec.getLocalSize()
        iset_p = PETSc.IS().createStride(
            self.pbf.p.x.petsc_vec.getLocalSize(),
            first=offset_p,
            step=1,
            comm=self.comm,
       )

        offset_s = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_s = PETSc.IS().createStride(self.pbf0.LM.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        offset_d = offset_s + self.pbf0.LM.getLocalSize()
        iset_d = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_d,
            step=1,
            comm=self.comm,
       )

        if self.pbs.incompressible_2field:
            ilist = [iset_u, iset_ps, iset_v, iset_p, iset_s, iset_d]
        else:
            ilist = [iset_u, iset_v, iset_p, iset_s, iset_d]

        return ilist


    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # fluid-ALE + flow0d problem
        if N > 0:
            self.io.readcheckpoint(self, N)

        self.pb0.read_restart(sname, N)

        if self.pbase.restart_step > 0:
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname + "_lm", N, self.pbf0.LM)
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname + "_lm", N, self.pbf0.LM_old)

    def evaluate_initial(self):
        self.pbfas.evaluate_initial()
        self.pbf0.evaluate_initial_coupling()

    def write_output_ini(self):
        # self.io.write_output(self, writemesh=True)
        self.pbfas.write_output_ini()
        #self.pbf0.write_output_ini()

    def write_output_pre(self):
        self.pbfas.write_output_pre()
        #self.pbf0.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbfas.evaluate_pre_solve(t, N, dt)
        self.pbf0.evaluate_pre_solve(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbfas.evaluate_post_solve(t, N)
        self.pbf0.evaluate_post_solve(t, N)

    def set_output_state(self, N):
        self.pbfas.set_output_state(N)
        self.pbf0.set_output_state(N)

    def write_output(self, N, t, mesh=False):
        self.pbfas.write_output(N, t)
        self.pbf0.write_output_coupling(N, t)  # writes LMs of 3D-0D

    def update(self):
        # update time step - solid,fluid,ALE, fluid-flow0d coup
        self.pbfas.update()
        self.pbf0.update_coupling()

    def print_to_screen(self):
        self.pbfas.print_to_screen()
        self.pbf0.print_to_screen_coupling()

    def induce_state_change(self):
        self.pbfas.induce_state_change()
        self.pbf0.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.pbfas.write_restart(self, N, force=force)
        self.pb0.write_restart(sname, N, force=force)

        if (self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0) or force:
            lm_sq = allgather_vec(self.pbf0.LM, self.comm)
            if self.comm.rank == 0:
                f = open(
                    self.pb0.output_path_0D + "/checkpoint_" + sname + "_lm_" + str(N) + ".txt",
                    "wt",
                )
                for i in range(len(lm_sq)):
                    f.write("%.16E\n" % (lm_sq[i]))
                f.close()
            del lm_sq

    def check_abort(self, t):
        return self.pbfa0.check_abort(t)

    def destroy(self):
        self.pbfas.destroy()
        self.pbfa0.destroy()


class FSIFlow0DSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlin.solver_nonlinear_ode([self.pb.pb0], self.solver_params["subsolver_params"])
        else:
            self.subsol = None

        self.evaluate_assemble_system_initial(subsolver=self.subsol)

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params, subsolver=self.subsol)

    def solve_initial_state(self):
        # consider consistent initial acceleration of solid
        if self.pb.pbs.timint != "static" and self.pb.pbase.restart_step == 0:
            ts = time.time()
            utilities.print_status(
                "Setting forms and solving for consistent initial solid acceleration...",
                self.pb.comm,
                e=" ",
            )

            # weak form at initial state for consistent initial acceleration solve
            if self.pb.pbfas.fsi_system=="neumann_dirichlet":
                weakform_a_solid = (
                    self.pb.pbs.deltaW_kin_old
                    + self.pb.pbs.deltaW_int_old
                    - self.pb.pbs.deltaW_ext_old
                )
            else:
                weakform_a_solid = (
                    self.pb.pbs.deltaW_kin_old
                    + self.pb.pbs.deltaW_int_old
                    - self.pb.pbs.deltaW_ext_old
                    + self.pb.pbfas.work_coupling_solid_old
                )

            weakform_lin_aa_solid = ufl.derivative(
                weakform_a_solid, self.pb.pbs.a_old, self.pb.pbs.du
            )  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a_solid, jac_aa_solid = (
                fem.form(weakform_a_solid, entity_maps=self.pb.io.entity_maps),
                fem.form(weakform_lin_aa_solid, entity_maps=self.pb.io.entity_maps),
            )
            self.solnln.solve_consistent_ini_acc(res_a_solid, jac_aa_solid, self.pb.pbs.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

        # consider consistent initial acceleration of fluid
        if (
            self.pb.pbf.fluid_governing_type == "navierstokes_transient"
            or self.pb.pbf.fluid_governing_type == "stokes_transient"
        ) and self.pb.pbase.restart_step == 0:
            ts = time.time()
            utilities.print_status(
                "Setting forms and solving for consistent initial fluid acceleration...",
                self.pb.comm,
                e=" ",
            )

            # weak form at initial state for consistent initial acceleration solve
            if self.pb.pbfas.fsi_system=="neumann_dirichlet":
                weakform_a_fluid = (
                    self.pb.pbf.deltaW_kin_old
                    + self.pb.pbf.deltaW_int_old
                    - self.pb.pbf.deltaW_ext_old
                    - self.pb.pbf0.power_coupling_old
                )
            else:
                weakform_a_fluid = (
                    self.pb.pbf.deltaW_kin_old
                    + self.pb.pbf.deltaW_int_old
                    - self.pb.pbf.deltaW_ext_old
                    - self.pb.pbfas.power_coupling_fluid_old
                    - self.pb.pbf0.power_coupling_old
                )

            weakform_lin_aa_fluid = ufl.derivative(
                weakform_a_fluid, self.pb.pbf.a_old, self.pb.pbf.dv
            )  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a_fluid, jac_aa_fluid = (
                fem.form(weakform_a_fluid, entity_maps=self.pb.io.entity_maps),
                fem.form(weakform_lin_aa_fluid, entity_maps=self.pb.io.entity_maps),
            )
            self.solnln.solve_consistent_ini_acc(res_a_fluid, jac_aa_fluid, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbase.numstep, ni=ni, li=li, wt=wt)
