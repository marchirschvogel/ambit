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
from .fluid_ale_multiphase_main import FluidmechanicsAleMultiphaseProblem
from .fluid_ale_main import FluidmechanicsAleProblem
from ..fluid.fluid_main import FluidmechanicsProblem
from ..solid.solid_main import SolidmechanicsProblem
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base


class FSIMultiphaseProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params_solid,
        time_params_fluid,
        time_params_phasefield,
        fem_params_solid,
        fem_params_fluid,
        fem_params_phasefield,
        fem_params_ale,
        constitutive_models_solid,
        constitutive_models_fluid,
        constitutive_models_phasefield,
        constitutive_models_ale,
        bc_dict_solid,
        bc_dict_fluid,
        bc_dict_phasefield,
        bc_dict_ale,
        bc_dict_lm,
        time_curves,
        coupling_params_fluid_ale,
        io,
        mor_params={},
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_coupling_fluid_ale(coupling_params_fluid_ale)

        self.problem_physics = "fsi_multiphase"

        self.io = io

        self.have_condensed_variables = False

        # instantiate problem classes
        # FSI - fluid-ALE-solid
        self.pbfsi = FSIProblem(
            pbase,
            io_params,
            time_params_solid,
            time_params_fluid,
            fem_params_solid,
            fem_params_fluid,
            fem_params_ale,
            constitutive_models_solid,
            constitutive_models_fluid,
            constitutive_models_ale,
            bc_dict_solid,
            bc_dict_fluid,
            bc_dict_ale,
            bc_dict_lm,
            time_curves,
            coupling_params_fluid_ale,
            io,
            mor_params=mor_params,
            is_multiphase=True,
        )

        # fluid+ALE+phasefield
        self.pbfap = FluidmechanicsAleMultiphaseProblem(
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
            mor_params=mor_params,
            pbfa=self.pbfsi.pbfa,
        )

        self.pbs = self.pbfsi.pbs
        self.pbf = self.pbfsi.pbfa.pbf
        self.pba = self.pbfsi.pbfa.pba
        self.pbp = self.pbfap.pbp
        self.pbfp = self.pbfap.pbfp

        self.pbrom = self.pbs  # ROM problem can only be solid
        self.pbrom_host = self

        # modify results to write...
        self.pbs.results_to_write = io_params["results_to_write"][0]
        self.pbf.results_to_write = io_params["results_to_write"][1]
        self.pbp.results_to_write = io_params["results_to_write"][2]
        self.pba.results_to_write = io_params["results_to_write"][3]

        self.incompressible_2field = self.pbs.incompressible_2field
        self.fsi_system = self.pbfsi.fsi_system

        self.localsolve = False
        self.print_subiter = False
        self.sub_solve = False

        self.numdof = self.pbfsi.numdof + self.pbp.numdof

        # number of fields involved
        if self.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                self.nfields = 8
            else:
                self.nfields = 7
        else:
            if self.pbs.incompressible_2field:
                self.nfields = 7
            else:
                self.nfields = 6

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
        if self.pbfsi.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 1, 2, 1, 1, 1, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbp.phi.x.petsc_vec,
                    self.pbp.mu.x.petsc_vec,
                    self.pbfsi.lm.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
            else:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 2, 1, 1, 1, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbp.phi.x.petsc_vec,
                    self.pbp.mu.x.petsc_vec,
                    self.pbfsi.lm.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
        else:
            if self.pbs.incompressible_2field:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 1, 2, 1, 1, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbp.phi.x.petsc_vec,
                    self.pbp.mu.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
            else:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 2, 1, 1, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pbp.phi.x.petsc_vec,
                    self.pbp.mu.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted

    def set_variational_forms(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_variational_forms()
        # phasefield
        self.pbp.set_variational_forms()
        # fluid-phasefield
        self.pbfp.set_variational_forms_coupling()
        # ALE-phasefield
        self.pbfap.set_variational_forms_coupling()

    def set_variational_forms_coupling(self):
        pass # no additional coupling forms needed

    def set_problem_residual_jacobian_forms(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_problem_residual_jacobian_forms()
        # phasefield
        self.pbp.set_problem_residual_jacobian_forms()
        # fluid-phasefield
        self.pbfp.set_problem_residual_jacobian_forms_coupling()
        # ALE-phasefield
        self.pbfap.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        pass

    def set_problem_vector_matrix_structures(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_problem_vector_matrix_structures()
        # phasefield
        self.pbp.set_problem_vector_matrix_structures()
        # fluid-phasefield
        self.pbfp.set_problem_vector_matrix_structures_coupling()
        # ALE-phasefield
        self.pbfap.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        pass

    def assemble_residual(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            ofs = 1
        else:
            ofs = 0
        if self.pbfsi.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.assemble_residual(t)
        self.pbp.assemble_residual(t)

        # solid
        self.r_list[0] = self.pbfsi.r_list[0]
        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbfsi.r_list[1]
        # fluid
        self.r_list[1 + ofs] = self.pbfsi.r_list[1 + ofs]
        self.r_list[2 + ofs] = self.pbfsi.r_list[2 + ofs]
        # Cahn-Hilliard
        self.r_list[3 + ofs] = self.pbp.r_list[0]
        self.r_list[4 + ofs] = self.pbp.r_list[1]
        # coupling constraint
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.r_list[5 + ofs] = self.pbfsi.r_list[3 + ofs]
        # ALE
        self.r_list[5 + ofc + ofs] = self.pbfsi.r_list[3 + ofc + ofs]

    def assemble_stiffness(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            ofs = 1
        else:
            ofs = 0
        if self.pbfsi.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.assemble_stiffness(t)
        self.pbp.assemble_stiffness(t)
        self.pbfp.assemble_stiffness_coupling(t)
        self.pbfap.assemble_stiffness_coupling(t)

        # solid momentum
        self.K_list[0][0] = self.pbfsi.K_list[0][0]                  # w.r.t. solid displacement
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbfsi.K_list[0][1]              # w.r.t. solid pressure
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.K_list[0][5 + ofs] = self.pbfsi.K_list[0][3 + ofs]  # w.r.t. Lagrange multiplier
        if self.pbfsi.fsi_system == "neumann_dirichlet":
            self.K_list[0][1 + ofs] = self.pbfsi.K_list[0][1 + ofs]  # w.r.t. fluid velcocity

        # solid incompressibility
        if self.pbs.incompressible_2field:
            self.K_list[1][0] = self.pbfsi.K_list[1][0]  # w.r.t. solid displacement
            self.K_list[1][1] = self.pbfsi.K_list[1][1]  # w.r.t. solid pressure

        # fluid momentum
        self.K_list[1 + ofs][1 + ofs] = self.pbfsi.K_list[1 + ofs][1 + ofs]              # w.r.t. fluid velcocity
        self.K_list[1 + ofs][2 + ofs] = self.pbfsi.K_list[1 + ofs][2 + ofs]              # w.r.t. fluid pressure
        if self.pbfsi.fsi_system == "neumann_dirichlet":
            self.K_list[1 + ofs][0] = self.pbfsi.K_list[1 + ofs][0]                      # w.r.t. solid displacement
            if self.pbs.incompressible_2field:
                self.K_list[1 + ofs][1] = self.pbfsi.K_list[1 + ofs][1]                  # w.r.t. solid pressure
        self.K_list[1 + ofs][3 + ofs] = self.pbfp.K_vphi                                 # w.r.t. phase
        self.K_list[1 + ofs][4 + ofs] = self.pbfp.K_vmu                                  # w.r.t. potential
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.K_list[1 + ofs][5 + ofs] = self.pbfsi.K_list[1 + ofs][3 + ofs]          # w.r.t. Lagrange multiplier
        self.K_list[1 + ofs][5 + ofc + ofs] = self.pbfsi.K_list[1 + ofs][3 + ofc + ofs]  # w.r.t. ALE displacement

        # fluid continuity
        self.K_list[2 + ofs][1 + ofs] = self.pbfsi.K_list[2 + ofs][1 + ofs]              # w.r.t. fluid velcocity
        self.K_list[2 + ofs][2 + ofs] = self.pbfsi.K_list[2 + ofs][2 + ofs]              # w.r.t. fluid pressure
        self.K_list[2 + ofs][3 + ofs] = self.pbfp.K_pphi                                 # w.r.t. phase
        self.K_list[2 + ofs][5 + ofc + ofs] = self.pbfsi.K_list[2 + ofs][3 + ofc + ofs]  # w.r.t. ALE displacement

        # phase field
        self.K_list[3 + ofs][1 + ofs] = self.pbfp.K_phiv         # w.r.t. fluid velocity
        self.K_list[3 + ofs][2 + ofs] = self.pbfp.K_phip         # w.r.t. fluid pressure
        self.K_list[3 + ofs][3 + ofs] = self.pbp.K_list[0][0]    # w.r.t. phase
        self.K_list[3 + ofs][4 + ofs] = self.pbp.K_list[0][1]    # w.r.t. potential
        self.K_list[3 + ofs][5 + ofc + ofs] = self.pbfap.K_phid  # w.r.t. ALE displacement

        # potential
        self.K_list[4 + ofs][3 + ofs] = self.pbp.K_list[1][0]   # w.r.t. phase
        self.K_list[4 + ofs][4 + ofs] = self.pbp.K_list[1][1]   # w.r.t. potential
        self.K_list[4 + ofs][5 + ofc + ofs] = self.pbfap.K_mud  # w.r.t. ALE displacement

        # FSI coupling constraint
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.K_list[5 + ofs][0] = self.pbfsi.K_list[3 + ofs][0]              # w.r.t. solid displacement
            self.K_list[5 + ofs][1 + ofs] = self.pbfsi.K_list[3 + ofs][1 + ofs]  # w.r.t. fluid velocity
            self.K_list[5 + ofs][5 + ofs] = self.pbfsi.K_list[3 + ofs][3 + ofs]  # w.r.t. Lagrange multiplier (carries only DBCs)

        # ALE
        self.K_list[5 + ofc + ofs][5 + ofc + ofs] = self.pbfsi.K_list[3 + ofc + ofs][3 + ofc + ofs]  # w.r.t. ALE displacement
        self.K_list[5 + ofc + ofs][1 + ofs] = self.pbfsi.K_list[3 + ofc + ofs][1 + ofs]              # w.r.t. fluid velocity

    def get_solver_index_sets(self, isoptions={}):
        # iterative solvers here are only implemented for neumann_dirichlet system!
        assert(self.pbfsi.fsi_system == "neumann_dirichlet")
        raise RuntimeError("Iterative solver not yet implemented for multiphase FSI!")

        return ilist


    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # FSI + phasefield problem
        if N > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        self.pbfsi.evaluate_initial()
        self.pbp.evaluate_initial()

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

    def write_output_pre(self):
        self.pbfsi.write_output_pre()
        self.pbp.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbfsi.evaluate_pre_solve(t, N, dt)
        self.pbp.evaluate_pre_solve(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbfsi.evaluate_post_solve(t, N)
        self.pbp.evaluate_post_solve(t, N)

    def set_output_state(self, N):
        self.pbfsi.set_output_state(N)
        self.pbp.set_output_state(N)

    def write_output(self, N, t, mesh=False):
        self.io.write_output(self, N=N, t=t) # combined multiphase FSI output routine

    def update(self):
        # update time step - solid,fluid,ALE, phasefield
        self.pbfsi.update()
        self.pbp.update()

    def print_to_screen(self):
        self.pbfsi.print_to_screen()
        self.pbp.print_to_screen()

    def induce_state_change(self):
        self.pbfsi.induce_state_change()
        self.pbp.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.io.write_restart(self, N, force=force)

    def check_abort(self, t):
        return False

    def destroy(self):
        self.pbfsi.destroy()
        self.pbp.destroy()


class FSIMultiphaseSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

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
            if self.pb.pbfsi.fsi_system=="neumann_dirichlet":
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
                    + self.pb.pbfsi.work_coupling_solid_old
                )

            weakform_lin_aa_solid = ufl.derivative(
                weakform_a_solid, self.pb.pbs.a_old, self.pb.pbs.du
            )  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a_solid, jac_aa_solid = (
                fem.form(weakform_a_solid, entity_maps=self.pb.io.entity_maps),
                fem.form(weakform_lin_aa_solid, entity_maps=self.pb.io.entity_maps),
            )
            self.solnln.solve_consistent_init(res_a_solid, jac_aa_solid, self.pb.pbs.a_old)

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
            if self.pb.pbfsi.fsi_system=="neumann_dirichlet":
                weakform_a_fluid = (
                    self.pb.pbf.deltaW_kin_old
                    + self.pb.pbf.deltaW_int_old
                    - self.pb.pbf.deltaW_ext_old
                )
            else:
                weakform_a_fluid = (
                    self.pb.pbf.deltaW_kin_old
                    + self.pb.pbf.deltaW_int_old
                    - self.pb.pbf.deltaW_ext_old
                    - self.pb.pbfsi.power_coupling_fluid_old
                )

            weakform_lin_aa_fluid = ufl.derivative(
                weakform_a_fluid, self.pb.pbf.a_old, self.pb.pbf.dv
            )  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a_fluid, jac_aa_fluid = (
                fem.form(weakform_a_fluid, entity_maps=self.pb.io.entity_maps),
                fem.form(weakform_lin_aa_fluid, entity_maps=self.pb.io.entity_maps),
            )
            self.solnln.solve_consistent_init(res_a_fluid, jac_aa_fluid, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
