#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, copy
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
from ..solid.solid_main import SolidmechanicsSolverPrestr
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
        io,
        coupling_params={},
        mor_params={},
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

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
            io,
            coupling_params=coupling_params[0],
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
            io,
            coupling_params=coupling_params,
            mor_params=mor_params,
            pbfa=self.pbfsi.pbfa,
        )

        self.pbs = self.pbfsi.pbs
        self.pbf = self.pbfsi.pbfa.pbf
        self.pba = self.pbfsi.pbfa.pba
        self.pbp = self.pbfap.pbp
        self.pbfp = self.pbfap.pbfp

        self.coupling_params = coupling_params[0]
        self.set_coupling_parameters()  # any additional ones not set by FSI (e.g. phase-scatra...)

        if self.coupling_phase_solidscatra:
            self.V_lmss = fem.functionspace(self.io.msh_emap_lm[0], ("Lagrange", self.pbs.pbscat.order_conc))
            # Lagrange multiplier for phasefield-scatra coupling
            self.lmss = fem.Function(self.V_lmss)
            self.lmss_old = fem.Function(self.V_lmss)

            self.dlmss = ufl.TrialFunction(self.V_lmss)  # incremental lm
            self.var_lmss = ufl.TestFunction(self.V_lmss)  # lm test function

        # in order to get correct contributions of the capillary stress on the (FSI) boundary, we should use this option...
        assert(self.pbfp.capillary_force_from_korteweg_stress)

        self.pbrom = self.pbs  # ROM problem can only be solid
        self.pbrom_host = self

        self.localsolve = False
        self.print_subiter = False
        self.sub_solve = False

        self.numdof = self.pbfsi.numdof + self.pbp.numdof

        # number of fields involved
        self.nfields = 6
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.nfields += 1
        if self.coupling_phase_solidscatra:
            self.nfields += 1

        # any offsets from solid mechanics (hydrostatic pressure, pore pressure, ...)
        self.nfields += self.pbs.offs

        # store some info on variable and equation names (used e.g. in solver print)
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.var_names = self.pbs.var_names + self.pbf.var_names + self.pbp.var_names + ["lm"] + self.pba.var_names
            self.eq_names = self.pbs.eq_names + self.pbf.eq_names + self.pbp.eq_names + ["FSI coup constraint"] + self.pba.eq_names
        else:
            self.var_names = self.pbs.var_names + self.pbf.var_names + self.pbp.var_names + self.pba.var_names
            self.eq_names = self.pbs.eq_names + self.pbf.eq_names + self.pbp.eq_names + self.pba.eq_names
        if self.coupling_phase_solidscatra:
            self.var_names += ["lmss"]
            self.eq_names += ["scatra-phase constr"]

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
        self.coupling_phase_solidscatra = self.coupling_params.get("coupling_phase_solidscatra", False)

    def get_problem_var_list(self):
        vlist_, is_ghosted = self.pbs.get_problem_var_list()
        vlist_f, is_ghosted_f = self.pbf.get_problem_var_list()
        vlist_ += vlist_f
        is_ghosted += is_ghosted_f
        vlist_p, is_ghosted_p = self.pbp.get_problem_var_list()
        vlist_ += vlist_p
        is_ghosted += is_ghosted_p
        if self.pbfsi.fsi_system == "neumann_neumann":
            vlist_.append(self.pbfsi.lm.x.petsc_vec)
            is_ghosted.append(1)
        vlist_a, is_ghosted_a = self.pba.get_problem_var_list()
        vlist_ += vlist_a
        is_ghosted += is_ghosted_a
        if self.coupling_phase_solidscatra:
            vlist_.append(self.lmss.x.petsc_vec)
            is_ghosted.append(1)
        return vlist_, is_ghosted

    def set_variational_forms(self):
        self.set_variational_forms_residual()
        self.set_variational_forms_jacobian()

    def set_variational_forms_residual(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_variational_forms_residual()
        # need to set these here - after fluid has done its job and phasefield is about to come...
        self.pbp.fluidvar["alpha"], self.pbp.fluidvar["alpha_old"], self.pbp.fluidvar["alpha_mid"] = [None]*self.pbf.num_domains, [None]*self.pbf.num_domains, [None]*self.pbf.num_domains
        for n, M in enumerate(self.pbf.domain_ids):
            self.pbp.fluidvar["alpha"][n] = self.pbf.alpha[n]
            self.pbp.fluidvar["alpha_old"][n] = self.pbf.alpha_old[n]
            self.pbp.fluidvar["alpha_mid"][n] = self.pbf.alpha_mid[n]
        # phasefield
        self.pbp.set_variational_forms_residual()
        # fluid-phasefield
        self.pbfp.set_variational_forms_residual_coupling()
        # ALE-phasefield
        self.pbfap.set_variational_forms_residual_coupling()
        # create additional coupling forms (e.g. interface wetting conditions)
        self.set_variational_forms_residual_coupling()

    def set_variational_forms_jacobian(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_variational_forms_jacobian()
        # phasefield
        self.pbp.set_variational_forms_jacobian()
        # fluid-phasefield
        self.pbfp.set_variational_forms_jacobian_coupling()
        # ALE-phasefield
        self.pbfap.set_variational_forms_jacobian_coupling()
        # create additional coupling forms
        self.set_variational_forms_jacobian_coupling()

    def set_variational_forms_residual_coupling(self):
        # wetting (or other) conditions imposed at interface
        if bool(self.pbfsi.wetting_interface):
            wetting = self.pbp.vf.weakform_robin_wetting(self.pbp.phi, self.pbp.phidot, self.pbfsi.wetting_interface["c1"], self.io.ds(self.io.interface_id_f), F=self.pbf.alevar["Fale"])
            wetting_old = self.pbp.vf.weakform_robin_wetting(self.pbp.phi_old, self.pbp.phidot_old, self.pbfsi.wetting_interface["c1"], self.io.ds(self.io.interface_id_f), F=self.pbf.alevar["Fale_old"])
            wetting_mid = self.pbp.vf.weakform_robin_wetting(self.pbp.phi_mid, self.pbp.phidot_mid, self.pbfsi.wetting_interface["c1"], self.io.ds(self.io.interface_id_f), F=self.pbf.alevar["Fale_mid"])

            if self.pbp.ti.res_eval == "trap":
                if not self.pbp.ti.potential_at_midpoint:
                    self.pbp.weakform_mu += wetting
                else:
                    self.pbp.weakform_mu += (self.pbp.timefac * wetting + (1.-self.pbp.timefac) * wetting_old)
            if self.pbp.ti.res_eval == "midp":
                if not self.pbp.ti.potential_at_midpoint:
                    self.pbp.weakform_mu += wetting
                else:
                    self.pbp.weakform_mu += wetting_mid
            if self.pbp.ti.res_eval == "back":
                self.pbp.weakform_mu += wetting

        # phasefield coupling to solid scalar transport
        if self.coupling_phase_solidscatra:
            c_lo = self.coupling_params["c_lo"]
            c_up = self.coupling_params["c_up"]

            c_gamma = c_lo * (1.0 - self.pbf.phasevar["chi"]) + c_up * self.pbf.phasevar["chi"]
            self.weakform_lmss = (self.pbs.pbscat.c[0] * self.var_lmss) * self.io.ds(self.io.interface_id_s) - (c_gamma * self.var_lmss) * self.io.ds(self.io.interface_id_f)

            self.pbs.pbscat.weakform_c[0] += (self.lmss * self.pbs.pbscat.var_c[0]) * self.io.ds(self.io.interface_id_s)

    def set_variational_forms_jacobian_coupling(self):
        if self.coupling_phase_solidscatra:
            self.weakform_lin_lmssc = ufl.derivative(self.weakform_lmss, self.pbs.pbscat.c[0], self.pbs.pbscat.dc[0])
            self.weakform_lin_lmssphi = ufl.derivative(self.weakform_lmss, self.pbp.phi, self.pbp.dphi)
            self.weakform_lin_clmss = ufl.derivative(self.pbs.pbscat.weakform_c[0], self.lmss, self.dlmss)

    def set_problem_residual_jacobian_forms(self, pre=False):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_problem_residual_jacobian_forms(pre=pre)
        # phasefield
        self.pbp.set_problem_residual_jacobian_forms()
        # fluid-phasefield
        self.pbfp.set_problem_residual_jacobian_forms_coupling()
        # ALE-phasefield
        self.pbfap.set_problem_residual_jacobian_forms_coupling()
        # any additional coupling
        self.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        if self.coupling_phase_solidscatra:
            self.res_ls = fem.form(self.weakform_lmss, entity_maps=self.io.entity_maps)
            self.jac_lsc = fem.form(self.weakform_lin_lmssc, entity_maps=self.io.entity_maps)
            self.jac_lsphi = fem.form(self.weakform_lin_lmssphi, entity_maps=self.io.entity_maps)
            self.jac_cls = fem.form(self.weakform_lin_clmss, entity_maps=self.io.entity_maps)

    def set_problem_vector_matrix_structures(self):
        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.set_problem_vector_matrix_structures()
        # phasefield
        self.pbp.set_problem_vector_matrix_structures()
        # fluid-phasefield
        self.pbfp.set_problem_vector_matrix_structures_coupling()
        # ALE-phasefield
        self.pbfap.set_problem_vector_matrix_structures_coupling()
        # any additional stuff
        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        if self.coupling_phase_solidscatra:
            self.r_ls = fem.petsc.assemble_vector(self.res_ls)

            self.K_lsc = fem.petsc.assemble_matrix(self.jac_lsc, [])
            self.K_lsc.assemble()
            self.K_lsphi = fem.petsc.assemble_matrix(self.jac_lsphi, [])
            self.K_lsphi.assemble()
            self.K_cls = fem.petsc.assemble_matrix(self.jac_cls, self.pbs.pbscat.dbcs[0])
            self.K_cls.assemble()

    def assemble_residual(self, t, subsolver=None):
        if self.pbfsi.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        self.assemble_residual_coupling(t)

        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.assemble_residual(t)
        self.pbp.assemble_residual(t)

        # solid
        self.r_list[0 : self.pbs.offs+1] = self.pbfsi.r_list[0 : self.pbs.offs+1]
        # fluid
        self.r_list[1 + self.pbs.offs] = self.pbfsi.r_list[1 + self.pbs.offs]
        self.r_list[2 + self.pbs.offs] = self.pbfsi.r_list[2 + self.pbs.offs]
        # Cahn-Hilliard
        self.r_list[3 + self.pbs.offs] = self.pbp.r_list[0]
        self.r_list[4 + self.pbs.offs] = self.pbp.r_list[1]
        # coupling constraint
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.r_list[5 + self.pbs.offs] = self.pbfsi.r_list[3 + self.pbs.offs]
        # ALE
        self.r_list[5 + ofc + self.pbs.offs] = self.pbfsi.r_list[3 + ofc + self.pbs.offs]

        if self.coupling_phase_solidscatra:
            self.r_list[6 + ofc + self.pbs.offs] = self.r_ls

    def assemble_residual_coupling(self, t):
        if self.coupling_phase_solidscatra:
            with self.r_ls.localForm() as r_local:
                r_local.set(0.0)
            fem.petsc.assemble_vector(self.r_ls, self.res_ls)
            # fem.apply_lifting(
            #     self.r_ls,
            #     [self.jac_lsls],
            #     [self.dbcs_lmss],
            #     x0=[self.lmss.x.petsc_vec],
            #     alpha=-1.0,
            # )
            self.r_ls.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # fem.set_bc(self.r_ls, self.dbcs_lmss, x0=self.lmss.x.petsc_vec, alpha=-1.0)

    def assemble_stiffness(self, t, subsolver=None):
        if self.pbfsi.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        self.assemble_stiffness_coupling(t)

        # FSI - fluid, solid, ALE, + FSI coup
        self.pbfsi.assemble_stiffness(t)
        self.pbp.assemble_stiffness(t)
        self.pbfp.assemble_stiffness_coupling(t)
        self.pbfap.assemble_stiffness_coupling(t)

        # solid momentum
        self.K_list[0][0 : self.pbs.offs+1] = self.pbfsi.K_list[0][0 : self.pbs.offs+1]  # w.r.t. solid displacement (+ pressure, ...)
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.K_list[0][5 + self.pbs.offs] = self.pbfsi.K_list[0][3 + self.pbs.offs]  # w.r.t. Lagrange multiplier
        if self.pbfsi.fsi_system == "neumann_dirichlet":
            self.K_list[0][1 + self.pbs.offs] = self.pbfsi.K_list[0][1 + self.pbs.offs]  # w.r.t. fluid velcocity

        # solid incompressibility
        off=0
        if self.pbs.incompressible_2field:
            off+=1
            self.K_list[1][0] = self.pbfsi.K_list[1][0]  # w.r.t. solid displacement
            self.K_list[1][1] = self.pbfsi.K_list[1][1]  # w.r.t. solid pressure
        if self.pbs.have_diffusion:
            off+=1
            self.K_list[off][0] = self.pbs.K_list[off][0]
            self.K_list[0][off] = self.pbs.K_list[0][off]
            self.K_list[off][off] = self.pbs.K_list[off][off]

        # fluid momentum
        self.K_list[1 + self.pbs.offs][1 + self.pbs.offs] = self.pbfsi.K_list[1 + self.pbs.offs][1 + self.pbs.offs]              # w.r.t. fluid velcocity
        self.K_list[1 + self.pbs.offs][2 + self.pbs.offs] = self.pbfsi.K_list[1 + self.pbs.offs][2 + self.pbs.offs]              # w.r.t. fluid pressure
        if self.pbfsi.fsi_system == "neumann_dirichlet":
            self.K_list[1 + self.pbs.offs][0] = self.pbfsi.K_list[1 + self.pbs.offs][0]                      # w.r.t. solid displacement
            if self.pbs.incompressible_2field:
                self.K_list[1 + self.pbs.offs][1] = self.pbfsi.K_list[1 + self.pbs.offs][1]                  # w.r.t. solid pressure
        self.K_list[1 + self.pbs.offs][3 + self.pbs.offs] = self.pbfp.K_vphi                                 # w.r.t. phase
        self.K_list[1 + self.pbs.offs][4 + self.pbs.offs] = self.pbfp.K_vmu                                  # w.r.t. potential
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.K_list[1 + self.pbs.offs][5 + self.pbs.offs] = self.pbfsi.K_list[1 + self.pbs.offs][3 + self.pbs.offs]          # w.r.t. Lagrange multiplier
        self.K_list[1 + self.pbs.offs][5 + ofc + self.pbs.offs] = self.pbfsi.K_list[1 + self.pbs.offs][3 + ofc + self.pbs.offs]  # w.r.t. ALE displacement

        # fluid continuity
        self.K_list[2 + self.pbs.offs][1 + self.pbs.offs] = self.pbfsi.K_list[2 + self.pbs.offs][1 + self.pbs.offs]              # w.r.t. fluid velcocity
        self.K_list[2 + self.pbs.offs][2 + self.pbs.offs] = self.pbfsi.K_list[2 + self.pbs.offs][2 + self.pbs.offs]              # w.r.t. fluid pressure
        self.K_list[2 + self.pbs.offs][3 + self.pbs.offs] = self.pbfp.K_pphi                                 # w.r.t. phase
        self.K_list[2 + self.pbs.offs][4 + self.pbs.offs] = self.pbfp.K_pmu                                  # w.r.t. potential
        self.K_list[2 + self.pbs.offs][5 + ofc + self.pbs.offs] = self.pbfsi.K_list[2 + self.pbs.offs][3 + ofc + self.pbs.offs]  # w.r.t. ALE displacement

        # phase field
        self.K_list[3 + self.pbs.offs][1 + self.pbs.offs] = self.pbfp.K_phiv         # w.r.t. fluid velocity
        self.K_list[3 + self.pbs.offs][2 + self.pbs.offs] = self.pbfp.K_phip         # w.r.t. fluid pressure
        self.K_list[3 + self.pbs.offs][3 + self.pbs.offs] = self.pbp.K_list[0][0]    # w.r.t. phase
        self.K_list[3 + self.pbs.offs][4 + self.pbs.offs] = self.pbp.K_list[0][1]    # w.r.t. potential
        self.K_list[3 + self.pbs.offs][5 + ofc + self.pbs.offs] = self.pbfap.K_phid  # w.r.t. ALE displacement

        # potential
        self.K_list[4 + self.pbs.offs][3 + self.pbs.offs] = self.pbp.K_list[1][0]   # w.r.t. phase
        self.K_list[4 + self.pbs.offs][4 + self.pbs.offs] = self.pbp.K_list[1][1]   # w.r.t. potential
        self.K_list[4 + self.pbs.offs][5 + ofc + self.pbs.offs] = self.pbfap.K_mud  # w.r.t. ALE displacement

        # FSI coupling constraint
        if self.pbfsi.fsi_system == "neumann_neumann":
            self.K_list[5 + self.pbs.offs][0] = self.pbfsi.K_list[3 + self.pbs.offs][0]              # w.r.t. solid displacement
            self.K_list[5 + self.pbs.offs][1 + self.pbs.offs] = self.pbfsi.K_list[3 + self.pbs.offs][1 + self.pbs.offs]  # w.r.t. fluid velocity
            self.K_list[5 + self.pbs.offs][5 + self.pbs.offs] = self.pbfsi.K_list[3 + self.pbs.offs][3 + self.pbs.offs]  # w.r.t. Lagrange multiplier (carries only DBCs)

        # ALE
        self.K_list[5 + ofc + self.pbs.offs][5 + ofc + self.pbs.offs] = self.pbfsi.K_list[3 + ofc + self.pbs.offs][3 + ofc + self.pbs.offs]  # w.r.t. ALE displacement
        self.K_list[5 + ofc + self.pbs.offs][1 + self.pbs.offs] = self.pbfsi.K_list[3 + ofc + self.pbs.offs][1 + self.pbs.offs]              # w.r.t. fluid velocity

        # coupling to solid-scatra
        if self.coupling_phase_solidscatra:
            self.K_list[6 + ofc + self.pbs.offs][3 + self.pbs.offs] = self.K_lsphi   # w.r.t. phase
            self.K_list[6 + ofc + self.pbs.offs][self.pbs.offs] = self.K_lsc  # w.r.t. concentration
            self.K_list[self.pbs.offs][6 + ofc + self.pbs.offs] = self.K_cls  # w.r.t. lmss

    def assemble_stiffness_coupling(self, t, subsolver=None):
        if self.coupling_phase_solidscatra:
            self.K_lsc.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lsc, self.jac_lsc, [])
            self.K_lsc.assemble()
            self.K_lsphi.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lsphi, self.jac_lsphi, [])
            self.K_lsphi.assemble()
            self.K_cls.zeroEntries()
            fem.petsc.assemble_matrix(self.K_cls, self.jac_cls, self.pbs.pbscat.dbcs[0])
            self.K_cls.assemble()

    def get_solver_index_sets(self, isoptions={}, blocked=False):
        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            uvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            uvec_ls = self.rom.V.getLocalSize()[1]
        else:
            uvec_or0 = self.pbs.u.x.petsc_vec.getOwnershipRange()[0]
            uvec_ls = self.pbs.u.x.petsc_vec.getLocalSize()

        if blocked:
            offset_u = uvec_or0
        else:
            offset_u = uvec_or0 + self.pbf.v.x.petsc_vec.getOwnershipRange()[0] + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.pbp.phi.x.petsc_vec.getOwnershipRange()[0] + self.pbp.mu.x.petsc_vec.getOwnershipRange()[0] + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
        if self.pbs.incompressible_2field:
            offset_u += self.pbs.p.x.petsc_vec.getOwnershipRange()[0]
        if not blocked:
            if self.pbfsi.fsi_system == "neumann_neumann":
                offset_u += self.pbfsi.lm.x.petsc_vec.getOwnershipRange()[0]
        iset_u = PETSc.IS().createStride(uvec_ls, first=offset_u, step=1, comm=self.comm)
        iset_u.setBlockSize(self.pbs.u.x.petsc_vec.getBlockSize())

        if self.pbs.incompressible_2field:
            offset_ps = offset_u + uvec_ls
            iset_ps = PETSc.IS().createStride(
                self.pbs.p.x.petsc_vec.getLocalSize(),
                first=offset_ps,
                step=1,
                comm=self.comm,
            )
            iset_ps.setBlockSize(self.pbs.p.x.petsc_vec.getBlockSize())

        if blocked:
            offset_v = self.pbf.v.x.petsc_vec.getOwnershipRange()[0] + self.pbf.p.x.petsc_vec.getOwnershipRange()[0]
            if self.pbfsi.fsi_system == "neumann_neumann":
                offset_v += self.pbfsi.lm.x.petsc_vec.getOwnershipRange()[0]
        else:
            if self.pbs.incompressible_2field:
                offset_v = offset_ps + self.pbs.p.x.petsc_vec.getLocalSize()
            else:
                offset_v = offset_u + uvec_ls

        iset_v = PETSc.IS().createStride(
            self.pbf.v.x.petsc_vec.getLocalSize(),
            first=offset_v,
            step=1,
            comm=self.comm
        )
        iset_v.setBlockSize(self.pbf.v.x.petsc_vec.getBlockSize())

        offset_p = offset_v + self.pbf.v.x.petsc_vec.getLocalSize()
        iset_p = PETSc.IS().createStride(
            self.pbf.p.x.petsc_vec.getLocalSize(),
            first=offset_p,
            step=1,
            comm=self.comm,
        )
        iset_p.setBlockSize(self.pbf.p.x.petsc_vec.getBlockSize())

        if blocked:
            offset_phi = self.pbp.phi.x.petsc_vec.getOwnershipRange()[0] + self.pbp.mu.x.petsc_vec.getOwnershipRange()[0]
        else:
            offset_phi = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_phi = PETSc.IS().createStride(
            self.pbp.phi.x.petsc_vec.getLocalSize(),
            first=offset_phi,
            step=1,
            comm=self.comm,
        )
        iset_phi.setBlockSize(self.pbp.phi.x.petsc_vec.getBlockSize())
        offset_mu = offset_phi + self.pbp.phi.x.petsc_vec.getLocalSize()
        iset_mu = PETSc.IS().createStride(
            self.pbp.mu.x.petsc_vec.getLocalSize(),
            first=offset_mu,
            step=1,
            comm=self.comm,
        )
        iset_mu.setBlockSize(self.pbp.mu.x.petsc_vec.getBlockSize())

        if self.pbfsi.fsi_system == "neumann_neumann":
            if blocked:  # to v,p block!
                offset_l = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
            else:
                offset_l = offset_mu + self.pbp.mu.x.petsc_vec.getLocalSize()
            iset_l = PETSc.IS().createStride(
                self.pbfsi.lm.x.petsc_vec.getLocalSize(),
                first=offset_l,
                step=1,
                comm=self.comm,
            )
            iset_l.setBlockSize(self.pbfsi.lm.x.petsc_vec.getBlockSize())

        if blocked:
            offset_d = self.pba.d.x.petsc_vec.getOwnershipRange()[0]
        else:
            if self.pbfsi.fsi_system == "neumann_neumann":
                offset_d = offset_l + self.pbfsi.lm.x.petsc_vec.getLocalSize()
            else:
                offset_d = offset_mu + self.pbp.mu.x.petsc_vec.getLocalSize()

        iset_d = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_d,
            step=1,
            comm=self.comm,
        )
        iset_d.setBlockSize(self.pba.d.x.petsc_vec.getBlockSize())

        if self.pbfsi.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                ilist = [iset_u, iset_ps, iset_v, iset_p, iset_phi, iset_mu, iset_l, iset_d]
            else:
                ilist = [iset_u, iset_v, iset_p, iset_phi, iset_mu, iset_l, iset_d]
        else:
            if self.pbs.incompressible_2field:
                ilist = [iset_u, iset_ps, iset_v, iset_p, iset_phi, iset_mu, iset_d]
            else:
                ilist = [iset_u, iset_v, iset_p, iset_phi, iset_mu, iset_d]

        return ilist


    ### now the base routines for this problem

    def read_restart(self, sname, N):
        self.pbfsi.read_restart(sname, N)
        self.pbp.read_restart(sname, N)

    def evaluate_initial(self):
        self.pbfsi.evaluate_initial()
        self.pbp.evaluate_initial()

    def write_output_ini(self):
        self.pbfsi.write_output_ini()
        self.pbp.write_output_ini()

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

    def write_output(self, N, t, msh=False):
        self.pbfsi.write_output(N=N, t=t)
        self.pbp.write_output(N=N, t=t)

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
        self.pbfsi.write_restart(sname, N, force=force)
        self.pbp.write_restart(sname, N, force=force)

    def check_abort(self, t):
        return False

    def destroy(self):
        self.pbfsi.destroy()
        self.pbp.destroy()


class FSIMultiphaseSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbs.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

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
        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the multiphase FSI problem
        if self.pb.pbs.pre:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()

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

    def solve_nonlinear_problem(self, t, N):
        self.solnln.newton(t, N)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
