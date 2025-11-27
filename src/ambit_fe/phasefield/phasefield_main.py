#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from . import phasefield_constitutive
from . import phasefield_variationalform
from .. import timeintegration
from .. import utilities
from ..solver import solver_nonlin
from .. import boundaryconditions
from .. import ioparams

from ..base import problem_base, solver_base

"""
Phase field problem class: Cahn-Hilliard
"""

class PhasefieldProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params,
        fem_params,
        constitutive_models,
        bc_dict,
        time_curves,
        io,
        mor_params={},
        is_advected=False,
        is_ale=False,
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_fem_phasefield(fem_params)
        ioparams.check_params_time_phasefield(time_params)

        self.problem_physics = "phasefield"

        self.results_to_write = io_params["results_to_write"]

        self.io = io

        self.is_ale = is_ale
        self.is_advected = is_advected

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        # # for FSI, we want to specify the subdomains
        self.domain_ids = self.io.io_params.get("domain_ids_fluid", np.arange(1, self.num_domains + 1))

        # TODO: Find nicer solution here...
        if self.pbase.problem_type == "fsi" or self.pbase.problem_type == "fsi_flow0d":
            self.dx, self.bmeasures = self.io.dx, self.io.bmeasures
        else:
            self.dx, self.bmeasures = self.io.create_integration_measures(
                self.io.mesh, [self.io.mt_d, self.io.mt_b, self.io.mt_sb]
            )

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_phi = fem_params["order_phi"]
        self.order_mu = fem_params["order_mu"]
        self.quad_degree = fem_params["quad_degree"]

        # collect domain data
        self.lam, self.m = [], []
        for n, M in enumerate(self.domain_ids):
            self.lam.append(self.constitutive_models["MAT" + str(n + 1)]["params_cahnhilliard"]["lambda"])
            self.m.append(self.constitutive_models["MAT" + str(n + 1)]["params_cahnhilliard"]["M"])

        self.localsolve = False  # no idea what might have to be solved locally...
        self.prestress_initial = False  # guess prestressing in ALE is somehow senseless...
        self.incompressible_2field = False  # always False here...
        self.have_condensed_variables = False  # always False here...
        self.sub_solve = False
        self.print_subiter = False
        self.dim = self.io.mesh.geometry.dim

        # model order reduction
        self.mor_params = mor_params
        if bool(self.mor_params):
            self.have_rom = True
        else:
            self.have_rom = False
        # will be set by solver base class
        self.rom = None

        # ALE problem variables
        self.alevar = {}
        # fluid variables
        self.fluidvar = {}

        # function spaces for phi and mu
        self.V_phi = fem.functionspace(
            self.io.mesh,
            ("Lagrange", self.order_phi),
        )

        self.V_mu = fem.functionspace(
            self.io.mesh,
            ("Lagrange", self.order_mu),
        )

        # for output writing - function spaces on the degree of the mesh
        self.mesh_degree = self.io.mesh._ufl_domain._ufl_coordinate_element._degree
        self.V_out_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree))

        # functions phase field
        self.dphi = ufl.TrialFunction(self.V_phi)  # Incremental phase field
        self.var_phi = ufl.TestFunction(self.V_phi)  # Test function
        self.phi = fem.Function(self.V_phi, name="PhaseField")
        self.phidot = fem.Function(self.V_phi)
        # functions potential
        self.dmu = ufl.TrialFunction(self.V_mu)  # Incremental potential
        self.var_mu = ufl.TestFunction(self.V_mu)  # Test function
        self.mu = fem.Function(self.V_mu, name="Potential")
        # values of previous time step
        self.phi_old = fem.Function(self.V_phi)
        self.phidot_old = fem.Function(self.V_phi)
        self.mu_old = fem.Function(self.V_mu)

        self.numdof = self.phi.x.petsc_vec.getSize() + self.mu.x.petsc_vec.getSize()

        # initialize phase field time-integration class
        self.ti = timeintegration.timeintegration_phasefield(
            time_params,
            self.pbase.dt,
            self.pbase.numstep,
            time_curves=time_curves,
            t_init=self.pbase.t_init,
            dim=self.dim,
            comm=self.comm,
        )

        self.timefac_m, self.timefac = self.ti.timefactors()

        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(
                phasefield_constitutive.constitutive(self.constitutive_models["MAT" + str(n + 1)])
            )

        # initialize pahse field (Cahn-Hilliard) variational form class
        if not bool(self.alevar):
            self.vf = phasefield_variationalform.variationalform(self.var_phi, self.var_mu)
        else:
            # mid-point representation of ALE velocity
            self.alevar["w_mid"] = self.timefac * self.alevar["w"] + (1.0 - self.timefac) * self.alevar["w_old"]
            # mid-point representation of ALE deformation gradient - linear in ALE displacement, hence we can combine it like this
            self.alevar["Fale_mid"] = (
                self.timefac * self.alevar["Fale"] + (1.0 - self.timefac) * self.alevar["Fale_old"]
            )

            self.vf = phasefield_variationalform.variationalform_ale(self.var_phi, self.var_mu)

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond(
            self.io,
            fem_params=fem_params,
            vf=self.vf,
            ti=self.ti,
            V_field=self.V_phi,
        )
        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if "dirichlet" in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.bc_dict["dirichlet"])

        if "dirichlet_vol" in self.bc_dict.keys():
            self.bc.dirichlet_vol(self.bc_dict["dirichlet_vol"])

        # number of fields involved
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
        is_ghosted = [1, 1]
        return [self.phi.x.petsc_vec, self.mu.x.petsc_vec], is_ghosted

    # the main function that defines the Cahn-Hilliard problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):
        # set form for phidot
        self.phidot_expr = self.ti.set_phidot(self.phi, self.phi_old, self.phidot_old)

        self.phase_field, self.potential = ufl.as_ufl(0), ufl.as_ufl(0)
        self.phase_field_old = ufl.as_ufl(0)

        for n, M in enumerate(self.domain_ids):
            self.phase_field += self.vf.cahnhilliard_phase(self.phidot_expr, self.phi, self.mu, self.m[n], self.dx(M))
            self.phase_field_old += self.vf.cahnhilliard_phase(self.phidot_old, self.phi_old, self.mu_old, self.m[n], self.dx(M))
            self.potential += self.vf.cahnhilliard_potential(self.phi, self.mu, self.ma[n].driv_force(self.phi), self.lam[n], self.dx(M))

        self.weakform_phi = self.timefac * self.phase_field + (1.-self.timefac) * self.phase_field_old # at n+\theta
        self.weakform_mu = self.potential # always at n+1
        self.weakform_lin_phiphi = ufl.derivative(self.weakform_phi, self.phi, self.dphi)
        self.weakform_lin_phimu = ufl.derivative(self.weakform_phi, self.mu, self.dmu)
        self.weakform_lin_mumu = ufl.derivative(self.weakform_mu, self.mu, self.dmu)
        self.weakform_lin_muphi = ufl.derivative(self.weakform_mu, self.phi, self.dphi)

    def set_problem_residual_jacobian_forms(self):
        ts = time.time()
        utilities.print_status("FEM form compilation for phase field (Cahn-Hilliard)...", self.comm, e=" ")

        self.res_phi = fem.form(self.weakform_phi, entity_maps=self.io.entity_maps)
        self.res_mu = fem.form(self.weakform_mu, entity_maps=self.io.entity_maps)
        self.jac_phiphi = fem.form(self.weakform_lin_phiphi, entity_maps=self.io.entity_maps)
        self.jac_phimu = fem.form(self.weakform_lin_phimu, entity_maps=self.io.entity_maps)
        self.jac_mumu = fem.form(self.weakform_lin_mumu, entity_maps=self.io.entity_maps)
        self.jac_muphi = fem.form(self.weakform_lin_muphi, entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        self.r_phi = fem.petsc.assemble_vector(self.res_phi)
        self.r_mu = fem.petsc.assemble_vector(self.res_mu)
        self.K_phiphi = fem.petsc.assemble_matrix(self.jac_phiphi)
        self.K_phiphi.assemble()
        self.K_phimu = fem.petsc.assemble_matrix(self.jac_phimu)
        self.K_phimu.assemble()
        self.K_mumu = fem.petsc.assemble_matrix(self.jac_mumu)
        self.K_mumu.assemble()
        self.K_muphi = fem.petsc.assemble_matrix(self.jac_muphi)
        self.K_muphi.assemble()

        self.r_list[0] = self.r_phi
        self.r_list[1] = self.r_mu
        self.K_list[0][0] = self.K_phiphi
        self.K_list[0][1] = self.K_phimu
        self.K_list[1][1] = self.K_mumu
        self.K_list[1][0] = self.K_muphi

    def assemble_residual(self, t, subsolver=None):
        # assemble rhs vector
        with self.r_phi.localForm() as r_local:
            r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_phi, self.res_phi)
        fem.apply_lifting(
            self.r_phi,
            [self.jac_phiphi],
            [self.bc.dbcs],
            x0=[self.phi.x.petsc_vec],
            alpha=-1.0,
        )
        self.r_phi.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.r_phi, self.bc.dbcs, x0=self.phi.x.petsc_vec, alpha=-1.0)

        with self.r_mu.localForm() as r_local:
            r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_mu, self.res_mu)
        self.r_mu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.r_list[0] = self.r_phi
        self.r_list[1] = self.r_mu

    def assemble_stiffness(self, t, subsolver=None):
        # assemble system matrix
        self.K_phiphi.zeroEntries()
        fem.petsc.assemble_matrix(self.K_phiphi, self.jac_phiphi, self.bc.dbcs)
        self.K_phiphi.assemble()

        self.K_phimu.zeroEntries()
        fem.petsc.assemble_matrix(self.K_phimu, self.jac_phimu, self.bc.dbcs)
        self.K_phimu.assemble()

        self.K_mumu.zeroEntries()
        fem.petsc.assemble_matrix(self.K_mumu, self.jac_mumu, [])
        self.K_mumu.assemble()

        self.K_muphi.zeroEntries()
        fem.petsc.assemble_matrix(self.K_muphi, self.jac_muphi, [])
        self.K_muphi.assemble()

        self.K_list[0][0] = self.K_phiphi
        self.K_list[0][1] = self.K_phimu
        self.K_list[1][1] = self.K_mumu
        self.K_list[1][0] = self.K_muphi

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if self.pbase.restart_step > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        # read initial conditions from file
        if self.pbase.restart_step == 0: # TODO: Not so nice, find better solution...
            if self.pbase.initial_fields is not None:
                for n, fld in enumerate([self.phi_old, self.mu_old]):
                    if self.pbase.initial_fields[n] is not None:
                        # can only be a path to a file (str) or an expression
                        if isinstance(self.pbase.initial_fields[n], str):
                            self.io.readfunction(fld, self.pbase.initial_fields[n])
                        else:
                            expr = self.pbase.initial_fields[n]()
                            fld.interpolate(expr.evaluate)
                            fld.x.petsc_vec.ghostUpdate(
                                addv=PETSc.InsertMode.INSERT,
                                mode=PETSc.ScatterMode.FORWARD,
                            )

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

    def write_output_pre(self):
        pass

    def evaluate_pre_solve(self, t, N, dt):
        # set time-dependent functions
        self.ti.set_time_funcs(t, dt)

        # DBC from files
        if self.bc.have_dirichlet_file:
            for m in self.ti.funcs_data:
                file = list(m.values())[0].replace("*", str(N))
                func = list(m.keys())[0]
                self.io.readfunction(func, file)
                sc = m["scale"]
                if sc != 1.0:
                    func.x.petsc_vec.scale(sc)

    def evaluate_post_solve(self, t, N):
        pass

    def set_output_state(self, N):
        pass

    def write_output(self, N, t, mesh=False):
        self.io.write_output(self, N=N, t=t)

    def update(self):
        self.ti.update_timestep(self.phi, self.phi_old, self.phidot, self.phidot_old, self.mu, self.mu_old)

    def print_to_screen(self):
        pass

    def induce_state_change(self):
        pass

    def write_restart(self, sname, N):
        self.io.write_restart(self, N)

    def check_abort(self, t):
        pass

    def destroy(self):
        self.io.close_output_files(self)


class PhasefieldSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

    def solve_initial_state(self):
        pass

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
