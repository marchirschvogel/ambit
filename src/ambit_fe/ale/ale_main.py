#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from . import ale_kinematics_constitutive
from . import ale_variationalform
from .. import timeintegration
from .. import utilities
from ..solver import solver_nonlin
from .. import boundaryconditions
from .. import ioparams

from ..base import problem_base, solver_base

"""
Arbitrary Lagrangian Eulerian (ALE) mechanics problem
"""


class AleProblem(problem_base):
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
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_fem_ale(fem_params)
        ioparams.check_params_time_fluid(time_params)

        self.problem_physics = "ale"

        self.results_to_write = io_params["results_to_write"]

        self.io = io

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        # for FSI, we want to specify the subdomains
        self.domain_ids = self.io.io_params.get("domain_ids_fluid", np.arange(1, self.num_domains + 1))

        # TODO: Find nicer solution here...
        if self.pbase.problem_type == "fsi" or self.pbase.problem_type == "fsi_flow0d":
            self.dx, self.bmeasures = self.io.dx, self.io.bmeasures
        else:
            self.dx, self.bmeasures = self.io.create_integration_measures(
                self.io.mesh, [self.io.mt_d, self.io.mt_b1, self.io.mt_b2]
            )

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_disp = fem_params["order_disp"]
        self.quad_degree = fem_params["quad_degree"]

        self.localsolve = False  # no idea what might have to be solved locally...
        self.prestress_initial = False  # guess prestressing in ALE is somehow senseless...
        self.incompressible_2field = False  # always False here...
        self.have_condensed_variables = False  # always False here...

        self.sub_solve = False
        self.print_subiter = False

        self.dim = self.io.mesh.geometry.dim

        # type of discontinuous function spaces
        if (
            str(self.io.mesh.ufl_cell()) == "tetrahedron"
            or str(self.io.mesh.ufl_cell()) == "triangle"
            or str(self.io.mesh.ufl_cell()) == "triangle3D"
        ):
            dg_type = "DG"
            if (self.order_disp > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif (
            str(self.io.mesh.ufl_cell()) == "hexahedron"
            or str(self.io.mesh.ufl_cell()) == "quadrilateral"
            or str(self.io.mesh.ufl_cell()) == "quadrilateral3D"
        ):
            dg_type = "DQ"
            if (self.order_disp > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")

        self.Vex = self.io.mesh.ufl_domain().ufl_coordinate_element()

        # model order reduction
        self.mor_params = mor_params
        if bool(self.mor_params):
            self.have_rom = True
        else:
            self.have_rom = False
        # will be set by solver base class
        self.rom = None

        # function space for d
        self.V_d = fem.functionspace(
            self.io.mesh,
            ("Lagrange", self.order_disp, (self.io.mesh.geometry.dim,)),
        )

        # continuous tensor and scalar function spaces of order order_disp
        self.V_tensor = fem.functionspace(
            self.io.mesh,
            (
                "Lagrange",
                self.order_disp,
                (self.io.mesh.geometry.dim, self.io.mesh.geometry.dim),
            ),
        )
        self.V_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.order_disp))

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.functionspace(
            self.io.mesh,
            (
                dg_type,
                self.order_disp - 1,
                (self.io.mesh.geometry.dim, self.io.mesh.geometry.dim),
            ),
        )
        self.Vd_vector = fem.functionspace(
            self.io.mesh,
            (dg_type, self.order_disp - 1, (self.io.mesh.geometry.dim,)),
        )
        self.Vd_scalar = fem.functionspace(self.io.mesh, (dg_type, self.order_disp - 1))

        # for output writing - function spaces on the degree of the mesh
        self.mesh_degree = self.io.mesh._ufl_domain._ufl_coordinate_element._degree
        self.V_out_tensor = fem.functionspace(
            self.io.mesh,
            (
                "Lagrange",
                self.mesh_degree,
                (self.io.mesh.geometry.dim, self.io.mesh.geometry.dim),
            ),
        )
        self.V_out_vector = fem.functionspace(
            self.io.mesh,
            ("Lagrange", self.mesh_degree, (self.io.mesh.geometry.dim,)),
        )
        self.V_out_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree))

        # coordinate element function space - based on input mesh
        self.Vcoord = fem.functionspace(self.io.mesh, self.Vex)

        # functions
        self.dd = ufl.TrialFunction(self.V_d)  # Incremental displacement
        self.var_d = ufl.TestFunction(self.V_d)  # Test function
        self.d = fem.Function(self.V_d, name="AleDisplacement")
        # auxiliary domain velocity vector
        self.w = fem.Function(self.V_d, name="AleVelocity")
        # values of previous time step
        self.d_old = fem.Function(self.V_d)
        self.w_old = fem.Function(self.V_d)

        # reference coordinates
        self.x_ref = ufl.SpatialCoordinate(self.io.mesh)

        self.numdof = self.d.x.petsc_vec.getSize()

        # initialize ALE time-integration class
        self.ti = timeintegration.timeintegration_ale(
            time_params,
            self.pbase.dt,
            self.pbase.numstep,
            time_curves=time_curves,
            t_init=self.pbase.t_init,
            dim=self.dim,
            comm=self.comm,
        )

        # initialize kinematics_constitutive class
        self.ki = ale_kinematics_constitutive.kinematics(self.dim, elem_metrics={"jac_det": self.io.detj0})

        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(
                ale_kinematics_constitutive.constitutive(self.ki, self.constitutive_models["MAT" + str(n + 1)])
            )

        # initialize ALE variational form class
        self.vf = ale_variationalform.variationalform(self.var_d, n0=self.io.n0, ro0=self.io.ro0)

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond(
            self.io,
            fem_params=fem_params,
            vf=self.vf,
            ti=self.ti,
            V_field=self.V_d,
            Vdisc_scalar=self.Vd_scalar,
        )
        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if "dirichlet" in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.bc_dict["dirichlet"])

        if "dirichlet_vol" in self.bc_dict.keys():
            self.bc.dirichlet_vol(self.bc_dict["dirichlet_vol"])

        self.set_variational_forms()

        # number of fields involved
        self.nfields = 1

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
        is_ghosted = [1]
        return [self.d.x.petsc_vec], is_ghosted

    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):
        # set form for domain velocity
        self.wel = self.ti.set_wel(self.d, self.d_old, self.w_old)

        # internal virtual work
        self.deltaW_int = ufl.as_ufl(0)

        for n, M in enumerate(self.domain_ids):
            # internal virtual work
            self.deltaW_int += self.vf.deltaW_int(self.ma[n].stress(self.d, self.wel), self.dx(M))

        # external virtual work (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_body, w_robin = (
            ufl.as_ufl(0),
            ufl.as_ufl(0),
            ufl.as_ufl(0),
        )
        if "neumann" in self.bc_dict.keys():
            w_neumann = self.bc.neumann_bcs(
                self.bc_dict["neumann"],
                self.bmeasures,
                funcs_to_update=self.ti.funcs_to_update,
                funcs_to_update_vec=self.ti.funcs_to_update_vec,
                funcsexpr_to_update=self.ti.funcsexpr_to_update,
                funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec,
            )
        if "bodyforce" in self.bc_dict.keys():
            w_body = self.bc.bodyforce(
                self.bc_dict["bodyforce"],
                self.dx,
                funcs_to_update=self.ti.funcs_to_update,
                funcsexpr_to_update=self.ti.funcsexpr_to_update,
            )
        if "robin" in self.bc_dict.keys():
            w_robin = self.bc.robin_bcs(self.bc_dict["robin"], self.d, self.wel, self.bmeasures)

        self.deltaW_ext = w_neumann + w_body + w_robin

        # internal minus external virtual work
        self.weakform_d = self.deltaW_int - self.deltaW_ext
        self.weakform_lin_dd = ufl.derivative(self.weakform_d, self.d, self.dd)

    def set_problem_residual_jacobian_forms(self):
        ts = time.time()
        utilities.print_status("FEM form compilation for ALE...", self.comm, e=" ")

        self.res_d = fem.form(self.weakform_d, entity_maps=self.io.entity_maps)
        self.jac_dd = fem.form(self.weakform_lin_dd, entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        self.r_d = fem.petsc.create_vector(self.res_d)
        self.K_dd = fem.petsc.create_matrix(self.jac_dd)

        self.r_list[0] = self.r_d
        self.K_list[0][0] = self.K_dd

    def assemble_residual(self, t, subsolver=None):
        # assemble rhs vector
        with self.r_d.localForm() as r_local:
            r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_d, self.res_d)
        fem.apply_lifting(
            self.r_d,
            [self.jac_dd],
            [self.bc.dbcs],
            x0=[self.d.x.petsc_vec],
            alpha=-1.0,
        )
        self.r_d.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.r_d, self.bc.dbcs, x0=self.d.x.petsc_vec, alpha=-1.0)

        self.r_list[0] = self.r_d

    def assemble_stiffness(self, t, subsolver=None):
        # assemble system matrix
        self.K_dd.zeroEntries()
        fem.petsc.assemble_matrix(self.K_dd, self.jac_dd, self.bc.dbcs)
        self.K_dd.assemble()

        self.K_list[0][0] = self.K_dd

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
        self.ti.update_timestep(self.d, self.d_old, self.w, self.w_old)

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


class AleSolver(solver_base):
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
