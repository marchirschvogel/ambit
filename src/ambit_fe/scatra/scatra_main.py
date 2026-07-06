#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from . import scatra_constitutive
from . import scatra_variationalform
from .. import timeintegration
from .. import utilities
from ..solver import solver_nonlin
from .. import boundaryconditions
from .. import ioparams

from ..base import problem_base, solver_base

"""
Scalar transport problem - can be coupled to solids or fluids, ALE-capable
"""


class ScatraProblem(problem_base):
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

        self.time_params = time_params[0]
        self.fem_params = fem_params[0]
        self.bc_dict = bc_dict[0]

        ioparams.check_params_fem_scatra(self.fem_params)
        ioparams.check_params_time_fluid(self.time_params)

        self.problem_physics = "scatra"

        self.results_to_write = io_params["results_to_write"]

        self.io = io
        self.write_restart_every = self.io.write_restart_every

        self.is_ale = is_ale
        self.is_advected = is_advected

        self.order_conc = self.fem_params["order_conc"]
        self.quad_degree = self.fem_params["quad_degree"]

        # collect relevant domain data and mesh
        self.domain_ids = self.io.domain_ids[self.io.m_id_scatra]
        self.num_domains = self.io.num_domains[self.io.m_id_scatra]
        self.mesh = self.io.mesh_[self.io.m_id_scatra]
        # mesh tags for DBCs
        self.mt_d, self.mt_b, self.mt_sb = self.io.mt_d_[self.io.m_id_scatra], self.io.mt_b_[self.io.m_id_scatra], self.io.mt_sb_[self.io.m_id_scatra]
        # global measures for weak BCs
        self.dx, self.bmeasures = self.io.dx, self.io.bmeasures
        # results files dictionary for I/O
        self.resultsfiles = {}

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models[0], self.mesh)

        self.localsolve = False
        self.prestress_initial = False
        self.incompressible_2field = False
        self.have_condensed_variables = False

        self.sub_solve = False
        self.print_subiter = False

        self.dim = self.mesh.geometry.dim

        # number of species that are transported - currently hard-wired to 1 - TODO: Make general, also for norms in solver!!
        self.num_species = 1

        # type of discontinuous function spaces
        if (
            str(self.mesh.ufl_cell()) == "tetrahedron"
            or str(self.mesh.ufl_cell()) == "triangle"
            or str(self.mesh.ufl_cell()) == "triangle3D"
        ):
            dg_type = "DG"
            if (self.order_conc > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif (
            str(self.mesh.ufl_cell()) == "hexahedron"
            or str(self.mesh.ufl_cell()) == "quadrilateral"
            or str(self.mesh.ufl_cell()) == "quadrilateral3D"
        ):
            dg_type = "DQ"
            if (self.order_conc > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")

        self.Vex = self.mesh.ufl_domain().ufl_coordinate_element()

        # model order reduction
        self.mor_params = mor_params
        if bool(self.mor_params):
            self.have_rom = True
        else:
            self.have_rom = False
        # will be set by solver base class
        self.rom = None

        # function space for c
        self.V_c = fem.functionspace(self.mesh, ("Lagrange", self.order_conc))

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.functionspace(
            self.mesh,
            (
                dg_type,
                self.order_conc - 1,
                (self.mesh.geometry.dim, self.mesh.geometry.dim),
            ),
        )
        self.Vd_vector = fem.functionspace(
            self.mesh,
            (dg_type, self.order_conc - 1, (self.mesh.geometry.dim,)),
        )
        self.Vd_scalar = fem.functionspace(self.mesh, (dg_type, self.order_conc - 1))

        # for output writing - function spaces on the degree of the mesh
        self.mesh_degree = self.mesh._ufl_domain._ufl_coordinate_element._degree
        self.V_out_tensor = fem.functionspace(
            self.mesh,
            (
                "Lagrange",
                self.mesh_degree,
                (self.mesh.geometry.dim, self.mesh.geometry.dim),
            ),
        )
        self.V_out_vector = fem.functionspace(
            self.mesh,
            ("Lagrange", self.mesh_degree, (self.mesh.geometry.dim,)),
        )
        self.V_out_scalar = fem.functionspace(self.mesh, ("Lagrange", self.mesh_degree))

        # coordinate element function space - based on input mesh
        self.Vcoord = fem.functionspace(self.mesh, self.Vex)

        # functions
        self.dc, self.var_c, self.c, self.cdot, self.c_old, self.c_veryold, self.cdot_old = [], [], [], [], [], [], []
        for i in range(self.num_species):
            self.dc.append(ufl.TrialFunction(self.V_c))  # Incremental concentrations
            self.var_c.append(ufl.TestFunction(self.V_c))  # Test function
            self.c.append(fem.Function(self.V_c, name="Concentration"+str(i+1)))
            self.cdot.append(fem.Function(self.V_c))
            # values of previous time step(s)
            self.c_old.append(fem.Function(self.V_c))
            self.c_veryold.append(fem.Function(self.V_c))
            self.cdot_old.append(fem.Function(self.V_c))

        # reference coordinates
        self.x_ref = ufl.SpatialCoordinate(self.mesh)

        self.numdof = 0
        for i in range(self.num_species):
            self.numdof += self.c[i].x.petsc_vec.getSize()

        # initialize scatra time-integration class
        self.ti = timeintegration.timeintegration_scatra(
            self.time_params,
            self.pbase.dt,
            self.pbase.numstep,
            time_curves=time_curves,
            t_init=self.pbase.t_init,
            dim=self.dim,
            comm=self.comm,
        )

        self.timefac_m, self.timefac = self.ti.timefactors()

        # initialize material/constitutive classes (one per domain)
        self.ma = [[]] * self.num_species
        for i in range(self.num_species):
            for n in range(self.num_domains):
                self.ma[i].append(
                    scatra_constitutive.constitutive(self.constitutive_models["MAT" + str(n + 1)])
                )

        # initialize scatra variational form classes
        self.vf = []
        for i in range(self.num_species):
            if not self.is_ale:
                self.vf.append(scatra_variationalform.variationalform(tstfncs=[self.var_c[i]], n0=self.io.n0))
            else:
                self.vf.append(scatra_variationalform.variationalform_ale(tstfncs=[self.var_c[i]], n0=self.io.n0))

        # set form for cdot
        self.cdot_expr = []
        for i in range(self.num_species):
            self.cdot_expr.append(self.ti.set_cdot(self.c[i], self.c_old[i], self.c_veryold[i], self.cdot_old[i]))

        # set mid-point representations
        self.c_mid, self.cdot_mid = [], []
        for i in range(self.num_species):
            self.c_mid.append(self.timefac * self.c[i] + (1.0 - self.timefac) * self.c_old[i])
            self.cdot_mid.append(self.timefac * self.cdot_expr[i] + (1.0 - self.timefac) * self.cdot_old[i])

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_scatra(
            self,
            V_field=self.V_c,
            Vdisc_scalar=self.Vd_scalar,
        )

        self.dbcs = [[]] * self.num_species
        # Dirichlet boundary conditions
        if "dirichlet" in self.bc_dict.keys():
            for i in range(self.num_species):  # TODO: For now, same for all - but may vary per c-field!
                self.bc.dirichlet_bcs(self.bc_dict["dirichlet"], self.dbcs[i])

        # number of fields involved
        self.nfields = 1 * self.num_species

        self.var_names = []
        for i in range(self.num_species):
            self.var_names.append("c" + str(i+1))

        # store some info on variable and equation names (used e.g. in solver print)
        self.var_names = []
        for i in range(self.num_species):
            self.var_names.append("c" + str(i+1))
        self.eq_names = ["concentration"]

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
        is_ghosted = [1] * self.num_species
        vlist_ = []
        for i in range(self.num_species):
            vlist_.append(self.c[i].x.petsc_vec)
        return vlist_, is_ghosted

    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):
        self.set_variational_forms_residual()
        self.set_variational_forms_jacobian()

    def set_variational_forms_residual(self):
        self.variational_form = [ufl.as_ufl(0)] * self.num_species
        self.variational_form_old = [ufl.as_ufl(0)] * self.num_species
        self.variational_form_mid = [ufl.as_ufl(0)] * self.num_species

        for i in range(self.num_species):
            for n, M in enumerate(self.domain_ids):
                self.variational_form[i] += self.vf[i].diffusion(self.cdot_expr[i], self.c[i], self.ma[i][n].diffusive_flux(self.c[i], self.cdot_expr[i]), self.dx(M))
                self.variational_form_old[i] += self.vf[i].diffusion(self.cdot_old[i], self.c_old[i], self.ma[i][n].diffusive_flux(self.c_old[i], self.cdot_old[i]), self.dx(M))
                self.variational_form_mid[i] += self.vf[i].diffusion(self.cdot_mid[i], self.c_mid[i], self.ma[i][n].diffusive_flux(self.c_mid[i], self.cdot_mid[i]), self.dx(M))

        w_neumann, w_neumann_old, w_neumann_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

        if "neumann" in self.bc_dict.keys():
            w_neumann = self.bc.neumann_bcs(
                self.bc_dict["neumann"],
                self.bmeasures,
                funcs_to_update=self.ti.funcs_to_update,
                funcsexpr_to_update=self.ti.funcsexpr_to_update,
            )
            w_neumann_old = self.bc.neumann_bcs(
                self.bc_dict["neumann"],
                self.bmeasures,
                funcs_to_update=self.ti.funcs_to_update_old,
                funcsexpr_to_update=self.ti.funcsexpr_to_update_old,
            )
            w_neumann_mid = self.bc.neumann_bcs(
                self.bc_dict["neumann"],
                self.bmeasures,
                funcs_to_update=self.ti.funcs_to_update_mid,
                funcsexpr_to_update=self.ti.funcsexpr_to_update_mid,
            )

        for i in range(self.num_species):
            self.variational_form[i] += -w_neumann
            self.variational_form_old[i] += -w_neumann_old
            self.variational_form_mid[i] += -w_neumann_mid

        self.weakform_c = [None] * self.num_species
        for i in range(self.num_species):
            if self.ti.res_eval == "trap":
                self.weakform_c[i] = self.timefac * self.variational_form[i] + (1.-self.timefac) * self.variational_form_old[i]
            if self.ti.res_eval == "midp":
                self.weakform_c[i] = self.variational_form_mid[i]
            if self.ti.res_eval == "back":
                self.weakform_c[i] = self.variational_form[i]

    def set_variational_forms_jacobian(self):
        self.weakform_lin_cc = [[None] * self.num_species for _ in range(self.num_species)]
        for i in range(self.num_species):
            for j in range(self.num_species):
                self.weakform_lin_cc[i][j] = ufl.derivative(self.weakform_c[i], self.c[j], self.dc[j])

    def set_problem_residual_jacobian_forms(self):
        ts = time.time()
        utilities.print_status("FEM form compilation for scalar transport...", self.comm, e=" ")

        self.res_c, self.jac_cc = [None] * self.num_species, [[None] * self.num_species for _ in range(self.num_species)]
        for i in range(self.num_species):
            self.res_c[i] = fem.form(self.weakform_c[i], entity_maps=self.io.entity_maps)
            for j in range(self.num_species):
                self.jac_cc[i][j] = fem.form(self.weakform_lin_cc[i][j], entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        ts = time.time()
        utilities.print_status("Creating vector and matrix data structures for scalar transport...", self.pbase.comm, e=" ")

        self.r_c, self.K_cc = [None] * self.num_species, [[None] * self.num_species for _ in range(self.num_species)]
        for i in range(self.num_species):
            self.r_c[i] = fem.petsc.assemble_vector(self.res_c[i])
            for j in range(self.num_species):
                self.K_cc[i][j] = fem.petsc.assemble_matrix(self.jac_cc[i][j], self.dbcs[i])
                self.K_cc[i][j].assemble()

        for i in range(self.num_species):
            self.r_list[i] = self.r_c[i]
            for j in range(self.num_species):
                self.K_list[i][j] = self.K_cc[i][j]

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def assemble_residual(self, t, subsolver=None):
        for i in range(self.num_species):
            # assemble rhs vector
            with self.r_c[i].localForm() as r_local:
                r_local.set(0.0)
            fem.petsc.assemble_vector(self.r_c[i], self.res_c[i])
            fem.apply_lifting(
                self.r_c[i],
                [self.jac_cc[i][i]],
                [self.dbcs[i]],
                x0=[self.c[i].x.petsc_vec],
                alpha=-1.0,
            )
            self.r_c[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(self.r_c[i], self.dbcs[i], x0=self.c[i].x.petsc_vec, alpha=-1.0)

            self.r_list[i] = self.r_c[i]

    def assemble_stiffness(self, t, subsolver=None):
        for i in range(self.num_species):
            for j in range(self.num_species):
                # assemble system matrix
                self.K_cc[i][j].zeroEntries()
                fem.petsc.assemble_matrix(self.K_cc[i][j], self.jac_cc[i][j], self.dbcs[i])
                self.K_cc[i][j].assemble()

                self.K_list[i][j] = self.K_cc[i][j]

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if self.pbase.restart_step > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        pass

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

    def write_output_pre(self):
        pass

    def evaluate_pre_solve(self, t, N, dt):
        # set time-dependent functions
        self.ti.set_time_funcs(t, dt)

        # DBC from files
        if self.bc.have_dirichlet_fileseries:
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
        for i in range(self.num_species):
            self.ti.update_timestep(self.c[i], self.c_old[i], self.c_veryold[i], self.cdot[i], self.cdot_old[i])

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


class ScatraSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

    def solve_initial_state(self):
        pass

    def solve_nonlinear_problem(self, t, N):
        self.solnln.newton(t, N)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
