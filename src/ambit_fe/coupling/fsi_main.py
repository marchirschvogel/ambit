#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import numpy as np
from dolfinx import fem, mesh
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import utilities
from .. import boundaryconditions

from ..solid.solid_main import SolidmechanicsProblem
from .fluid_ale_main import FluidmechanicsAleProblem

from ..base import problem_base, solver_base

"""
FSI problem class
"""


class FSIProblem(problem_base):
    def __init__(
        self,
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
        coupling_params,
        io,
        ios,
        iof,
        mor_params={},
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        self.problem_physics = "fsi"

        self.io = io
        self.ios, self.iof = ios, iof

        # initialize problem instances (also sets the variational forms for the solid and fluid problem)
        self.pbs = SolidmechanicsProblem(
            pbase,
            io_params,
            time_params_solid,
            fem_params_solid,
            constitutive_models_solid,
            bc_dict_solid,
            time_curves,
            ios,
            mor_params=mor_params,
        )
        self.pbfa = FluidmechanicsAleProblem(
            pbase,
            io_params,
            time_params_fluid,
            fem_params_fluid,
            fem_params_ale,
            constitutive_models_fluid_ale[0],
            constitutive_models_fluid_ale[1],
            bc_dict_fluid_ale[0],
            bc_dict_fluid_ale[1],
            time_curves,
            coupling_params,
            iof,
            mor_params=mor_params,
        )

        self.pbrom = self.pbs  # ROM problem can only be solid so far...
        self.pbrom_host = self

        self.pbf = self.pbfa.pbf
        self.pba = self.pbfa.pba

        # fluid displacement, but defined within solid function space
        self.ufs = fem.Function(self.pbs.V_u)

        # modify results to write...
        self.pbs.results_to_write = io_params["results_to_write"][0]
        self.pbf.results_to_write = io_params["results_to_write"][1][0]
        self.pba.results_to_write = io_params["results_to_write"][1][1]

        self.incompressible_2field = self.pbs.incompressible_2field

        self.fsi_governing_type = coupling_params.get("fsi_governing_type", "solid_governed")

        self.fsi_system = coupling_params.get("fsi_system", "neumann_neumann")

        self.zero_lm_boundary = coupling_params.get("zero_lm_boundary", False)

        self.have_condensed_variables = False

        # Lagrange multiplier function space
        self.V_lm = fem.functionspace(
            self.io.msh_emap_lm[0],
            (
                "Lagrange",
                self.pbs.order_disp,
                (self.io.msh_emap_lm[0].geometry.dim,),
            ),
        )

        # Lagrange multiplier
        self.lm = fem.Function(self.V_lm)
        self.lm_old = fem.Function(self.V_lm)

        self.zero_lm = fem.Function(self.V_lm)

        if self.fsi_system == "neumann_neumann":
            self.dlm = ufl.TrialFunction(self.V_lm)  # incremental lm
            self.var_lm = ufl.TestFunction(self.V_lm)  # lm test function

            self.bclm = boundaryconditions.boundary_cond(self.io, dim=self.io.msh_emap_lm[0].topology.dim)
            # set the whole boundary of the LM subspace to zero (beneficial when we have solid and fluid with overlapping DBCs)
            if self.zero_lm_boundary:  # TODO: Seems to not work properly - investigate!

                num_vertices_s = self.pbs.V_u.dofmap.index_map.size_local + self.pbs.V_u.dofmap.index_map.num_ghosts
                bs = self.pbs.V_u.dofmap.bs
                dofs_solid_ = np.empty(bs*num_vertices_s, dtype=np.int32)
                for i, vert in enumerate(self.io.msh_emap_solid[3]):
                    for j in range(bs):
                        dofs_solid_[bs*i+j] = bs*vert+j

                num_vertices_f = self.pbf.V_v.dofmap.index_map.size_local + self.pbf.V_v.dofmap.index_map.num_ghosts
                bs = self.pbf.V_v.dofmap.bs
                dofs_fluid_ = np.empty(bs*num_vertices_f, dtype=np.int32)
                for i, vert in enumerate(self.io.msh_emap_fluid[3]):
                    for j in range(bs):
                        dofs_fluid_[bs*i+j] = bs*vert+j

                dbcs_dofs_solid_all, dbcs_dofs_fluid_all = [], []
                for i in range(len(self.pbs.bc.dbcs)):
                    dbcs_dofs_solid_all.append(self.pbs.bc.dbcs[i].dof_indices()[0])
                for i in range(len(self.pbf.bc.dbcs)):
                    dbcs_dofs_fluid_all.append(self.pbf.bc.dbcs[i].dof_indices()[0])

                dbcs_dofs_solid_all_con = np.sort(np.concatenate(dbcs_dofs_solid_all))
                dbcs_dofs_fluid_all_con = np.sort(np.concatenate(dbcs_dofs_fluid_all))

                dofs_solid_ = np.sort(dofs_solid_)
                dofs_fluid_ = np.sort(dofs_fluid_)

                dbcs_dofs_solid_all_glob = dofs_solid_[dbcs_dofs_solid_all_con]
                dbcs_dofs_fluid_all_glob = dofs_fluid_[dbcs_dofs_fluid_all_con]

                common_dbcs_glob = np.intersect1d(dbcs_dofs_solid_all_glob, dbcs_dofs_fluid_all_glob)

                raise ValueError("Under development!")

        self.set_variational_forms()

        self.numdof = self.pbs.numdof + self.pbfa.numdof

        self.localsolve = False
        self.sub_solve = False

        # number of fields involved
        if self.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                self.nfields = 6
            else:
                self.nfields = 5
        else:
            if self.pbs.incompressible_2field:
                self.nfields = 5
            else:
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

    def get_problem_var_list(self):
        if self.fsi_system == "neumann_neumann":
            if self.pbs.incompressible_2field:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 1, 2, 1, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.lm.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
            else:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 2, 1, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.lm.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
        else:
            if self.pbs.incompressible_2field:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 1, 2, 1]
                else:
                    is_ghosted = [1, 1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbs.p.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted
            else:
                if self.pbf.num_dupl > 1:
                    is_ghosted = [1, 1, 2, 1]
                else:
                    is_ghosted = [1, 1, 1, 1]
                return [
                    self.pbs.u.x.petsc_vec,
                    self.pbf.v.x.petsc_vec,
                    self.pbf.p.x.petsc_vec,
                    self.pba.d.x.petsc_vec,
                ], is_ghosted

    # defines the monolithic coupling forms for FSI
    def set_variational_forms(self):
        self.work_coupling_solid = ufl.dot(self.lm, self.pbs.var_u) * self.io.ds(self.io.interface_id_s)
        self.work_coupling_solid_old = ufl.dot(self.lm_old, self.pbs.var_u) * self.io.ds(self.io.interface_id_s)
        self.power_coupling_fluid = ufl.dot(self.lm, self.pbf.var_v) * self.io.ds(self.io.interface_id_f)
        self.power_coupling_fluid_old = ufl.dot(self.lm_old, self.pbf.var_v) * self.io.ds(self.io.interface_id_f)

        # add to solid and fluid virtual work/power (no contribution to Jacobian, since lambda is a PK1 traction)
        self.pbs.weakform_u += (
            self.pbs.timefac * self.work_coupling_solid + (1.0 - self.pbs.timefac) * self.work_coupling_solid_old
        )
        self.pbf.weakform_v += (
            -self.pbf.timefac * self.power_coupling_fluid - (1.0 - self.pbf.timefac) * self.power_coupling_fluid_old
        )

        if self.fsi_system == "neumann_neumann":
            if self.fsi_governing_type == "solid_governed":
                self.weakform_l = ufl.dot(self.pbs.u, self.var_lm) * self.io.ds(self.io.interface_id_s) - ufl.dot(
                    self.pbf.ufluid, self.var_lm
                ) * self.io.ds(self.io.interface_id_f)
            elif self.fsi_governing_type == "fluid_governed":
                self.weakform_l = ufl.dot(self.pbf.v, self.var_lm) * self.io.ds(self.io.interface_id_f) - ufl.dot(
                    self.pbs.vel, self.var_lm
                ) * self.io.ds(self.io.interface_id_s)
            else:
                raise ValueError("Unknown FSI governing type.")

            self.weakform_lin_lu = ufl.derivative(self.weakform_l, self.pbs.u, self.pbs.du)
            self.weakform_lin_lv = ufl.derivative(self.weakform_l, self.pbf.v, self.pbf.dv)

            self.weakform_lin_ul = self.pbs.timefac * ufl.derivative(self.work_coupling_solid, self.lm, self.dlm)
            self.weakform_lin_vl = -self.pbf.timefac * ufl.derivative(self.power_coupling_fluid, self.lm, self.dlm)

            # even though this is zero, we still want to explicitly form and create the matrix for DBC application
            self.weakform_lin_ll = ufl.derivative(self.weakform_l, self.lm, self.dlm)
            # dummy form to initially get a sparsity pattern for LM DBC application
            self.from_ll_diag_dummy = ufl.inner(self.dlm, self.var_lm) * self.io.ds(self.io.interface_id_s) - ufl.inner(
                self.dlm, self.var_lm
            ) * self.io.ds(self.io.interface_id_f)

        elif self.fsi_system == "neumann_dirichlet":
            dbcs_coup_fluid_solid = []
            dbcs_coup_fluid_solid.append(
                fem.dirichletbc(
                    self.ufs,
                    fem.locate_dofs_topological(
                        self.pbs.V_u,
                        self.pbs.io.mesh.topology.dim - 1,
                        self.pbs.io.mt_b1.indices[np.isin(self.pbs.io.mt_b1.values, self.io.surf_interf)],
                    ),
                )
            )

            # get surface dofs for dr_solid/dv matrix entry
            fnode_indices_local = fem.locate_dofs_topological(
                self.pbs.V_u,
                self.pbs.io.mesh.topology.dim - 1,
                self.pbs.io.mt_b1.indices[np.isin(self.pbs.io.mt_b1.values, self.io.surf_interf)],
            )
            fnode_indices_all = np.array(
                self.pbs.V_u.dofmap.index_map.local_to_global(np.asarray(fnode_indices_local, dtype=np.int32)),
                dtype=np.int32,
            )
            self.fdofs_solid = PETSc.IS().createBlock(
                self.pbs.V_u.dofmap.index_map_bs,
                fnode_indices_all,
                comm=self.comm,
            )

            fnode_indices_local = fem.locate_dofs_topological(
                self.pbf.V_v,
                self.pbf.io.mesh.topology.dim - 1,
                self.pbf.io.mt_b1.indices[np.isin(self.pbf.io.mt_b1.values, self.io.surf_interf)],
            )
            fnode_indices_all = np.array(
                self.pbf.V_v.dofmap.index_map.local_to_global(np.asarray(fnode_indices_local, dtype=np.int32)),
                dtype=np.int32,
            )
            self.fdofs_fluid = PETSc.IS().createBlock(
                self.pbf.V_v.dofmap.index_map_bs,
                fnode_indices_all,
                comm=self.comm,
            )

            # now add the DBCs: pay attention to order... first u=uf, then the others... hence re-set!
            if bool(dbcs_coup_fluid_solid):
                # store DBCs without those from fluid
                self.pbs.bc.dbcs_nofluid = []
                for k in self.pbs.bc.dbcs:
                    self.pbs.bc.dbcs_nofluid.append(k)
                self.pbs.bc.dbcs = []
                self.pbs.bc.dbcs += dbcs_coup_fluid_solid
                # Dirichlet boundary conditions
                if "dirichlet" in self.pbs.bc_dict.keys():
                    self.pbs.bc.dirichlet_bcs(self.pbs.bc_dict["dirichlet"])

            # self.fluid_solid_disp_data = fem.create_nonmatching_meshes_interpolation_data(
            #         self.pbs.u.function_space.mesh._cpp_object,
            #         self.pbs.u.function_space.element,
            #         self.ufs.function_space.mesh._cpp_object)

    def set_problem_residual_jacobian_forms(self):
        # solid + ALE-fluid
        self.pbs.set_problem_residual_jacobian_forms()
        self.pbfa.set_problem_residual_jacobian_forms()

        ts = time.time()
        utilities.print_status("FEM form compilation for FSI coupling...", self.comm, e=" ")

        if self.fsi_system == "neumann_neumann":
            self.res_l = fem.form(self.weakform_l, entity_maps=self.io.entity_maps)
            self.jac_lu = fem.form(self.weakform_lin_lu, entity_maps=self.io.entity_maps)
            self.jac_lv = fem.form(self.weakform_lin_lv, entity_maps=self.io.entity_maps)

            self.jac_ul = fem.form(self.weakform_lin_ul, entity_maps=self.io.entity_maps)
            self.jac_vl = fem.form(self.weakform_lin_vl, entity_maps=self.io.entity_maps)

            # even though this is zero, we still want to explicitly form and create the matrix for DBC application
            self.jac_ll = fem.form(self.weakform_lin_ll, entity_maps=self.io.entity_maps)
            self.jac_ll_dummy = fem.form(self.from_ll_diag_dummy, entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        # solid + ALE-fluid
        self.pbs.set_problem_vector_matrix_structures()
        self.pbfa.set_problem_vector_matrix_structures()

        if self.fsi_system == "neumann_neumann":
            self.r_l = fem.petsc.create_vector(self.res_l)

            self.K_ul = fem.petsc.create_matrix(self.jac_ul)
            self.K_vl = fem.petsc.create_matrix(self.jac_vl)

            self.K_lu = fem.petsc.create_matrix(self.jac_lu)
            self.K_lv = fem.petsc.create_matrix(self.jac_lv)

            self.K_ll = fem.petsc.create_matrix(self.jac_ll_dummy)

        elif self.fsi_system == "neumann_dirichlet":
            # create unity vector with 1's on surface dofs and zeros elsewhere
            self.Isol = self.pbs.K_uu.createVecLeft()
            self.Isol.setValues(
                self.fdofs_solid,
                np.ones(self.fdofs_solid.getLocalSize()),
                addv=PETSc.InsertMode.INSERT,
            )
            self.Isol.assemble()
            # create diagonal matrix
            self.Diag_sol = PETSc.Mat().createAIJ(
                self.pbs.K_uu.getSizes(),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.comm,
            )
            self.Diag_sol.setUp()
            self.Diag_sol.assemble()
            # set 1's to get correct allocation pattern
            self.Diag_sol.shift(1.0)
            # now only set the 1's at surface dofs
            self.Diag_sol.setDiagonal(self.Isol, addv=PETSc.InsertMode.INSERT)
            self.Diag_sol.assemble()
            # create from ALE matrix and only keep the necessary columns
            # need to assemble here to get correct sparsity pattern when doing the column product
            self.K_uv_ = fem.petsc.assemble_matrix(self.pbs.jac_uu, [])
            self.K_uv_.assemble()
            # now multiply to grep out the correct columns
            self.K_uv = self.K_uv_.matMult(self.Diag_sol)
            self.K_uv.setOption(
                PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True
            )  # needed so that zeroRows does not change it!
            # get interface solid rhs vector
            self.r_u_interface = self.pbs.r_u.getSubVector(self.fdofs_solid)
            self.r_v_solid = self.pbf.K_vv.createVecLeft()

            # Jacobian contributions
            self.K_uu_nodbc = fem.petsc.create_matrix(self.pbs.jac_uu)
            self.K_uu_nodbc.assemble()
            self.K_uu_nodbc_interface = self.K_uu_nodbc.createSubMatrix(self.fdofs_solid, self.fdofs_solid)

            self.K_vv_solid = fem.petsc.create_matrix(self.pbf.jac_vv)
            self.K_vv_solid.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

            self.K_vu_solid = fem.petsc.create_matrix(self.pbf.jac_vv)
            self.K_vu_solid.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    def assemble_residual(self, t, subsolver=None):
        if self.fsi_system == "neumann_dirichlet":
            self.evaluate_residual_dbc_coupling()

        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        self.pbs.assemble_residual(t)
        self.pbfa.assemble_residual(t)

        if self.fsi_system == "neumann_neumann":
            with self.r_l.localForm() as r_local:
                r_local.set(0.0)
            fem.petsc.assemble_vector(self.r_l, self.res_l)
            fem.apply_lifting(
                self.r_l,
                [self.jac_ll],
                [self.bclm.dbcs],
                x0=[self.lm.x.petsc_vec],
                alpha=-1.0,
            )
            self.r_l.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(self.r_l, self.bclm.dbcs, x0=self.lm.x.petsc_vec, alpha=-1.0)
        elif self.fsi_system == "neumann_dirichlet":
            self.evaluate_residual_forces_interface()

        self.r_list[0] = self.pbs.r_list[0]

        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbs.r_list[1]

        self.r_list[1 + off] = self.pbfa.r_list[0]
        self.r_list[2 + off] = self.pbfa.r_list[1]

        if self.fsi_system == "neumann_neumann":
            self.r_list[3 + off] = self.r_l
            self.r_list[4 + off] = self.pbfa.r_list[2]
        else:
            self.r_list[3 + off] = self.pbfa.r_list[2]

    def assemble_stiffness(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        self.pbs.assemble_stiffness(t)
        self.pbfa.assemble_stiffness(t)

        # solid displacement
        self.K_list[0][0] = self.pbs.K_list[0][0]
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbs.K_list[0][1]

        if self.fsi_system == "neumann_neumann":
            self.K_ul.zeroEntries()
            fem.petsc.assemble_matrix(self.K_ul, self.jac_ul, self.pbs.bc.dbcs)
            self.K_ul.assemble()
            self.K_list[0][3 + off] = self.K_ul

        # solid pressure
        if self.pbs.incompressible_2field:
            self.K_list[1][0] = self.pbs.K_list[1][0]
            self.K_list[1][1] = self.pbs.K_list[1][1]

        # fluid velocity
        self.K_list[1 + off][1 + off] = self.pbfa.K_list[0][0]
        self.K_list[1 + off][2 + off] = self.pbfa.K_list[0][1]

        if self.fsi_system == "neumann_neumann":
            self.K_vl.zeroEntries()
            fem.petsc.assemble_matrix(self.K_vl, self.jac_vl, self.pbf.bc.dbcs)
            self.K_vl.assemble()
            self.K_list[1 + off][3 + off] = self.K_vl
            self.K_list[1 + off][4 + off] = self.pbfa.K_list[0][2]

        # fluid pressure
        self.K_list[2 + off][1 + off] = self.pbfa.K_list[1][0]
        self.K_list[2 + off][2 + off] = self.pbfa.K_list[1][1]
        if self.fsi_system == "neumann_neumann":
            self.K_list[2 + off][4 + off] = self.pbfa.K_list[1][2]
        else:
            self.K_list[2 + off][3 + off] = self.pbfa.K_list[1][2]

        # LM
        if self.fsi_system == "neumann_neumann":
            self.K_lu.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lu, self.jac_lu, self.bclm.dbcs)
            self.K_lu.assemble()
            self.K_list[3 + off][0] = self.K_lu
            self.K_lv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lv, self.jac_lv, self.bclm.dbcs)
            self.K_lv.assemble()
            self.K_list[3 + off][1 + off] = self.K_lv
            self.K_ll.zeroEntries()
            fem.petsc.assemble_matrix(self.K_ll, self.jac_ll, self.bclm.dbcs)
            self.K_ll.assemble()
            self.K_list[3 + off][3 + off] = self.K_ll

            # ALE displacement
            self.K_list[4 + off][4 + off] = self.pbfa.K_list[2][2]
            self.K_list[4 + off][1 + off] = self.pbfa.K_list[2][0]
        elif self.fsi_system == "neumann_dirichlet":
            self.K_uv_.zeroEntries()  # TODO!!!
            fem.petsc.assemble_matrix(self.K_uv_, self.pbs.jac_uu, self.pbs.bc.dbcs_nofluid)  # need DBCs w/o fluid here
            self.K_uv_.assemble()
            # multiply to get the relevant columns only
            self.K_uv_.matMult(self.Diag_sol, result=self.K_uv)
            # zero rows where DBC is applied and set diagonal entry to -1
            self.K_uv.zeroRows(self.fdofs_solid, diag=-1.0)
            # we apply u_fluid to solid, hence get du_fluid/dv
            fac = self.pbf.ti.get_factor_deriv_varint(self.pbase.dt)
            self.K_uv.scale(fac)

            self.evaluate_stiffness_forces_interface()

            self.K_list[0][1 + off] = self.K_uv

            # ALE displacement
            self.K_list[3 + off][3 + off] = self.pbfa.K_list[2][2]
            self.K_list[3 + off][1 + off] = self.pbfa.K_list[2][0]

    def evaluate_residual_dbc_coupling(self):
        # we need a vector representation of ufluid to apply in solid DBCs
        self.pbf.ti.update_varint(
            self.pbf.v.x.petsc_vec,
            self.pbf.v_old.x.petsc_vec,
            self.pbf.uf_old.x.petsc_vec,
            self.pbase.dt,
            varintout=self.pbf.uf.x.petsc_vec,
            uflform=False,
        )
        # self.ufs.x.petsc_vec.axpby(1.0, 0.0, self.pbf.uf.x.petsc_vec)
        # self.ufs.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.ufs.interpolate(self.pbf.uf)
        self.ufs.x.scatter_forward()

    def evaluate_residual_forces_interface(self):
        # get solid reaction forces on interface
        self.pbs.r_u.getSubVector(self.fdofs_solid, subvec=self.r_u_interface)  # TODO: Before or after set_bc??
        self.r_v_solid.setValues(self.fdofs_fluid, self.r_u_interface.array)
        self.pbs.r_u.restoreSubVector(self.fdofs_solid, subvec=self.r_u_interface)
        # update fluid residual
        self.pbf.r_v.axpy(-1.0, self.r_v_solid)
        self.pbf.r_v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def evaluate_stiffness_forces_interface(self):
        self.K_uu_nodbc.zeroEntries()
        fem.petsc.assemble_matrix(self.K_uu_nodbc, self.pbs.jac_uu, [])  # self.pbs.bc.dbcs
        self.K_uu_nodbc.assemble()

        self.K_uu_nodbc.createSubMatrix(
            self.fdofs_solid,
            self.fdofs_solid,
            submat=self.K_uu_nodbc_interface,
        )

        # need to update K_vv
        self.K_vv_solid.zeroEntries()
        self.K_vv_solid.setValues(
            self.fdofs_fluid,
            self.fdofs_fluid,
            self.K_uu_nodbc_interface[:, :],
            addv=PETSc.InsertMode.INSERT,
        )
        self.K_vv_solid.assemble()
        np.set_printoptions(threshold=sys.maxsize)
        print(self.K_uu_nodbc[:, :])

        self.pbf.K_vv.axpy(-1.0, self.K_vv_solid)

        # self.K_uu_nodbc.matMult(self.Diag_sol, result=self.K_uu_nodbc_interface)

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if N > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        self.pbs.evaluate_initial()
        self.pbfa.evaluate_initial()

    def write_output_ini(self):
        # self.io.write_output(self, writemesh=True)
        self.pbs.write_output_ini()
        self.pbfa.write_output_ini()

    def write_output_pre(self):
        self.pbs.write_output_pre()
        self.pbfa.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbs.evaluate_pre_solve(t, N, dt)
        self.pbfa.evaluate_pre_solve(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbs.evaluate_post_solve(t, N)
        self.pbfa.evaluate_post_solve(t, N)

    def set_output_state(self, N):
        self.pbs.set_output_state(N)
        self.pbfa.set_output_state(N)

    def write_output(self, N, t, mesh=False):
        # self.io.write_output(self, N=N, t=t) # combined FSI output routine
        self.pbs.write_output(N, t)
        self.pbfa.write_output(N, t)

    def update(self):
        # update time step - solid and 0D model
        self.pbs.update()
        self.pbfa.update()

        # update Lagrange multiplier
        self.lm_old.x.petsc_vec.axpby(1.0, 0.0, self.lm.x.petsc_vec)
        self.lm_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def print_to_screen(self):
        self.pbs.print_to_screen()
        self.pbfa.print_to_screen()

    def induce_state_change(self):
        self.pbs.induce_state_change()
        self.pbfa.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.io.write_restart(self, N, force=force)

    def check_abort(self, t):
        return False

    def destroy(self):
        self.pbs.destroy()
        self.pbfa.destroy()


class FSISolver(solver_base):
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
            weakform_a_solid = (
                self.pb.pbs.deltaW_kin_old
                + self.pb.pbs.deltaW_int_old
                - self.pb.pbs.deltaW_ext_old
                + self.pb.work_coupling_solid_old
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
            weakform_a_fluid = (
                self.pb.pbf.deltaW_kin_old
                + self.pb.pbf.deltaW_int_old
                - self.pb.pbf.deltaW_ext_old
                - self.pb.power_coupling_fluid_old
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
        self.solnln.newton(t, localdata=self.pb.pbs.localdata)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
