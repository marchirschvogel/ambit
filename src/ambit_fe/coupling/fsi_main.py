#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys, math
import numpy as np
from dolfinx import fem, mesh, io
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import utilities, meshutils
from .. import boundaryconditions

from ..solid.solid_main import SolidmechanicsProblem
from ..ale.ale_main import AleProblem
from ..fluid.fluid_main import FluidmechanicsProblem
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
        constitutive_models_fluid,
        constitutive_models_ale,
        bc_dict_solid,
        bc_dict_fluid,
        bc_dict_ale,
        bc_dict_lm,
        time_curves,
        io,
        coupling_params={},
        mor_params={},
        is_multiphase=False,
        pbs=None,
        pbf=None,
        pba=None,
        pbfa=None,
    ):
        self.pbase = pbase
        self.pbs = pbs
        self.pbf = pbf
        self.pba = pba
        self.pbfa = pbfa

        # pointer to communicator
        self.comm = self.pbase.comm

        self.problem_physics = "fsi"

        self.io = io

        # instantiate problem classes
        # solid
        if pbs is None:
            self.pbs = SolidmechanicsProblem(
                pbase,
                io_params,
                time_params_solid,
                fem_params_solid,
                constitutive_models_solid,
                bc_dict_solid,
                time_curves,
                io,
                mor_params=mor_params,
            )
        # fluid-ALE
        if pbfa is None:
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
                io,
                coupling_params=coupling_params,
                mor_params=mor_params,
                is_multiphase=is_multiphase,
            )

        self.pbf = self.pbfa.pbf
        self.pba = self.pbfa.pba

        self.coupling_params = coupling_params
        self.set_coupling_parameters()

        self.pbrom = self.pbs  # ROM problem can only be solid so far...
        self.pbrom_host = self

        # modify results to write...
        self.pbs.results_to_write = io_params["results_to_write"][0]
        self.pbf.results_to_write = io_params["results_to_write"][1]
        self.pba.results_to_write = io_params["results_to_write"][2]

        # currently no meshtags on interface mesh supported...
        self.mt_d, self.mt_b, self.mt_sb = None, None, None

        self.incompressible_2field = self.pbs.incompressible_2field
        self.have_condensed_variables = False

        if self.fsi_system == "neumann_neumann":
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

            self.dlm = ufl.TrialFunction(self.V_lm)  # incremental lm
            self.var_lm = ufl.TestFunction(self.V_lm)  # lm test function

            # Dirichlet boundary conditions for LM - if given
            self.dbcs_lm = []
            if bc_dict_lm is not None:
                bc = boundaryconditions.boundary_cond(self, V_field=self.V_lm)
                if "dirichlet" in bc_dict_lm.keys():
                    bc.dirichlet_bcs(bc_dict_lm["dirichlet"], self.dbcs_lm)

        self.numdof = self.pbs.numdof + self.pbfa.numdof
        if self.fsi_system == "neumann_neumann":
            self.numdof += self.lm.x.petsc_vec.getSize()

        self.localsolve = False
        self.sub_solve = False
        self.print_subiter = False

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

    def set_coupling_parameters(self):
        self.fsi_governing_type = self.coupling_params.get("fsi_governing_type", "solid_governed")
        self.fsi_system = self.coupling_params.get("fsi_system", "neumann_neumann") # neumann_neumann, neumann_dirichlet
        self.wetting_interface = self.coupling_params.get("wetting_condition_interface", {}) # only for multiphase FSI

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
        self.set_variational_forms_residual()
        self.set_variational_forms_jacobian()

    def set_variational_forms_residual(self):
        # solid + ALE-fluid
        self.pbs.set_variational_forms_residual()
        self.pbfa.set_variational_forms_residual()
        self.set_variational_forms_residual_coupling()

    def set_variational_forms_jacobian(self):
        # solid + ALE-fluid
        self.pbs.set_variational_forms_jacobian()
        self.pbfa.set_variational_forms_jacobian()
        self.set_variational_forms_jacobian_coupling()

    def set_variational_forms_residual_coupling(self):
        # fluid displacement, but defined on solid domain
        self.ufs = fem.Function(self.pbs.V_u)

        # establish dof mappings from fluid to solid
        if self.fsi_system=="neumann_dirichlet":
            # get global correspondence array of solid and fluid interface nodes
            self.fluid_to_solid_mapping()
            # get vector block sizes
            bs_s = self.pbs.u.x.petsc_vec.getBlockSize()
            bs_f = self.pbf.v.x.petsc_vec.getBlockSize()
            assert bs_s == bs_f # has to be the same... different dimensions of solid and fluid not supported at the moment...
            bs = bs_s

            s0, s1 = self.pbs.u.x.petsc_vec.getOwnershipRange()

            # convert to owned block/node range
            assert s0 % bs == 0 and s1 % bs == 0
            s0b, s1b = s0 // bs, s1 // bs

            # distribute by owned solid nodes!
            mask = (self.solid_nodes_glob >= s0b) & (self.solid_nodes_glob < s1b)

            solid_nodes_loc = self.solid_nodes_glob[mask]
            fluid_nodes_loc = self.fluid_nodes_glob[mask]

            assert solid_nodes_loc.size == fluid_nodes_loc.size

            # now build the index sets according to owned solid nodes distribution!
            self.fdofs_solid_global_sub = PETSc.IS().createBlock(bs, solid_nodes_loc, comm=self.comm)
            self.fdofs_fluid_global_sub = PETSc.IS().createBlock(bs, fluid_nodes_loc, comm=self.comm)

            # get indices and store for later...
            self.rows_fs = self.fdofs_solid_global_sub.getIndices()
            self.cols_fs = self.fdofs_fluid_global_sub.getIndices()

        if self.fsi_system == "neumann_neumann":
            self.work_coupling_solid = ufl.dot(self.lm, self.pbs.var_u) * self.io.ds(self.io.interface_id_s)
            self.work_coupling_solid_old = ufl.dot(self.lm_old, self.pbs.var_u) * self.io.ds(self.io.interface_id_s)
            self.power_coupling_fluid = ufl.dot(self.lm, self.pbf.var_v) * self.io.ds(self.io.interface_id_f)
            self.power_coupling_fluid_old = ufl.dot(self.lm_old, self.pbf.var_v) * self.io.ds(self.io.interface_id_f)

            # add to solid and fluid virtual work/power (no contribution to Jacobian, since lambda is a PK1 traction)
            # NOTE: We rather want the loads always at t_{n+1} to be consistent to the Neumann-Dirichlet scheme!
            # self.pbs.weakform_u += self.pbs.timefac * self.work_coupling_solid + (1.0 - self.pbs.timefac) * self.work_coupling_solid_old
            # self.pbf.weakform_v += -self.pbf.timefac * self.power_coupling_fluid - (1.0 - self.pbf.timefac) * self.power_coupling_fluid_old
            self.pbs.weakform_u += self.work_coupling_solid
            self.pbf.weakform_v += -self.power_coupling_fluid

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


        elif self.fsi_system == "neumann_dirichlet":

            self.dbcs_coup_fluid_solid = []

            if all(isinstance(x, int) for x in self.io.surf_interf):
                nodes_dbcs_fs = fem.locate_dofs_topological(self.pbs.V_u, self.pbs.mesh.topology.dim - 1, self.pbs.mt_b.indices[np.isin(self.pbs.mt_b.values, self.io.surf_interf)])
            else: # can only be locator function otherwise...
                nodes_dbcs_fs_ = []
                for lc in self.io.surf_interf:
                    nodes_dbcs_fs_.append(fem.locate_dofs_geometrical(self.pbs.V_u, lc.evaluate))
                nodes_dbcs_fs = np.concatenate(nodes_dbcs_fs_).ravel()

            self.dbcs_coup_fluid_solid.append(fem.dirichletbc(self.ufs, nodes_dbcs_fs))

            # now add the DBCs: pay attention to order... first u=uf, then the others... hence re-set!
            if bool(self.dbcs_coup_fluid_solid):
                # store DBCs without those from fluid
                self.pbs.dbcs_nofluid = []
                for k in self.pbs.dbcs:
                    self.pbs.dbcs_nofluid.append(k)
                self.pbs.dbcs = []
                self.pbs.dbcs += self.dbcs_coup_fluid_solid
                # Dirichlet boundary conditions
                if "dirichlet" in self.pbs.bc_dict.keys():
                    self.pbs.bc.dirichlet_bcs(self.pbs.bc_dict["dirichlet"], self.pbs.dbcs)

                # Here, we remove any DBC indices from the interface
                # NOTE: Do not use PETSc's index set "difference" method as it will not preserve the specific ordering needed!

                self.dbc_dofs_solid_global = []
                for k in range(len(self.pbs.dbcs_nofluid)):
                    if self.pbs.bc_dict["dirichlet"][k]["dir"]=="all": sub=None
                    if self.pbs.bc_dict["dirichlet"][k]["dir"]=="x": sub=0
                    if self.pbs.bc_dict["dirichlet"][k]["dir"]=="y": sub=1
                    if self.pbs.bc_dict["dirichlet"][k]["dir"]=="z": sub=2
                    self.dbc_dofs_solid_global.append( meshutils.get_index_set(self.pbs.V_u, self.comm, pb=self.pbs, identifier=self.pbs.bc_dict["dirichlet"][k]["id"], codim=self.pbs.bc_dict["dirichlet"][k].get("codimension", self.pbs.mesh.topology.dim-1), sub=sub, mask_owned=False) )
                dbcs_dofs_solid_all = []
                for k in range(len(self.dbc_dofs_solid_global)):
                    dbcs_dofs_solid_all.append( self.dbc_dofs_solid_global[k].allGather().array )

                dbcs_dofs_solid_all_flat = [item for sublist in dbcs_dofs_solid_all for item in sublist]

                idxs_d = set(dbcs_dofs_solid_all_flat)
                idxs_i = self.fdofs_solid_global_sub.getIndices()
                diff = [i for i in idxs_i if i not in idxs_d]

                self.fdofs_solid_global_sub = PETSc.IS().createGeneral(diff, comm=self.comm)

                self.dbc_dofs_fluid_global = []
                for k in range(len(self.pbf.dbcs)):
                    if self.pbf.bc_dict["dirichlet"][k]["dir"]=="all": sub=None
                    if self.pbf.bc_dict["dirichlet"][k]["dir"]=="x": sub=0
                    if self.pbf.bc_dict["dirichlet"][k]["dir"]=="y": sub=1
                    if self.pbf.bc_dict["dirichlet"][k]["dir"]=="z": sub=2
                    self.dbc_dofs_fluid_global.append( meshutils.get_index_set(self.pbf.V_v, self.comm, pb=self.pbf, identifier=self.pbf.bc_dict["dirichlet"][k]["id"], codim=self.pbf.bc_dict["dirichlet"][k].get("codimension", self.pbf.mesh.topology.dim-1), sub=sub, mask_owned=False) )
                dbcs_dofs_fluid_all = []
                for k in range(len(self.dbc_dofs_fluid_global)):
                    dbcs_dofs_fluid_all.append( self.dbc_dofs_fluid_global[k].allGather().array )

                dbcs_dofs_fluid_all_flat = [item for sublist in dbcs_dofs_fluid_all for item in sublist]

                idxs_d = set(dbcs_dofs_fluid_all_flat)
                idxs_i = self.fdofs_fluid_global_sub.getIndices()
                diff = [i for i in idxs_i if i not in idxs_d]

                self.fdofs_fluid_global_sub = PETSc.IS().createGeneral(diff, comm=self.comm)

            self.ufs_subvec = self.pbf.uf.x.petsc_vec.getSubVector(self.fdofs_fluid_global_sub)

        else:
            raise ValueError("Unknown value for 'fsi_system'. Choose 'neumann_neumann' or 'neumann_dirichlet'.")

    def set_variational_forms_jacobian_coupling(self):
        if self.fsi_system == "neumann_neumann":
            self.weakform_lin_lu = ufl.derivative(self.weakform_l, self.pbs.u, self.pbs.du)
            self.weakform_lin_lv = ufl.derivative(self.weakform_l, self.pbf.v, self.pbf.dv)

            # self.weakform_lin_ul = self.pbs.timefac * ufl.derivative(self.work_coupling_solid, self.lm, self.dlm)
            # self.weakform_lin_vl = -self.pbf.timefac * ufl.derivative(self.power_coupling_fluid, self.lm, self.dlm)
            self.weakform_lin_ul = ufl.derivative(self.work_coupling_solid, self.lm, self.dlm)
            self.weakform_lin_vl = -ufl.derivative(self.power_coupling_fluid, self.lm, self.dlm)

            # for DBC application to LM, even if zero...
            self.weakform_lin_ll = ufl.derivative(self.weakform_l, self.lm, self.dlm)

        elif self.fsi_system == "neumann_dirichlet":
            pass

        else:
            raise ValueError("Unknown value for 'fsi_system'. Choose 'neumann_neumann' or 'neumann_dirichlet'.")

    def set_problem_residual_jacobian_forms(self):
        # solid + ALE-fluid
        self.pbs.set_problem_residual_jacobian_forms()
        self.pbfa.set_problem_residual_jacobian_forms()
        self.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        ts = time.time()
        utilities.print_status("FEM form compilation for FSI coupling...", self.comm, e=" ")

        if self.fsi_system == "neumann_neumann":
            self.res_l = fem.form(self.weakform_l, entity_maps=self.io.entity_maps)
            self.jac_lu = fem.form(self.weakform_lin_lu, entity_maps=self.io.entity_maps)
            self.jac_lv = fem.form(self.weakform_lin_lv, entity_maps=self.io.entity_maps)

            self.jac_ul = fem.form(self.weakform_lin_ul, entity_maps=self.io.entity_maps)
            self.jac_vl = fem.form(self.weakform_lin_vl, entity_maps=self.io.entity_maps)
            # needed for DBC application to LM
            self.jac_ll = fem.form(self.weakform_lin_ll, entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        # solid + ALE-fluid
        self.pbs.set_problem_vector_matrix_structures()
        self.pbfa.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        ts = time.time()
        utilities.print_status("Creating vector and matrix data structures for FSI coupling...", self.pbase.comm, e=" ")

        if self.fsi_system == "neumann_neumann":
            self.r_l = fem.petsc.assemble_vector(self.res_l)

            self.K_ul = fem.petsc.assemble_matrix(self.jac_ul, self.pbs.dbcs)
            self.K_ul.assemble()
            self.K_vl = fem.petsc.assemble_matrix(self.jac_vl, self.pbf.dbcs)
            self.K_vl.assemble()

            self.K_lu = fem.petsc.assemble_matrix(self.jac_lu, self.dbcs_lm)
            self.K_lu.assemble()
            self.K_lv = fem.petsc.assemble_matrix(self.jac_lv, self.dbcs_lm)
            self.K_lv.assemble()

            self.K_ll = fem.petsc.assemble_matrix(self.jac_ll, self.dbcs_lm)
            if bool(self.dbcs_lm):
                self.K_ll.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            self.K_ll.assemble()

        if self.fsi_system == "neumann_dirichlet":
            # solid reaction forces
            self.r_reac_sol = fem.petsc.assemble_vector(self.pbs.res_u)
            # reaction forces on fluid side
            self.r_reac_on_fluid = fem.petsc.assemble_vector(self.pbf.res_v)

            # get interface solid rhs vector
            self.r_u_interface = self.r_reac_sol.getSubVector(self.fdofs_solid_global_sub)

            # set up matrix that contains 1's at distinct row-column pairs
            self.Diag_sol = PETSc.Mat().createAIJ(
                (self.pbs.K_uu.getSizes()[0],self.pbf.K_vv.getSizes()[0]),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.comm,
            )
            self.Diag_sol.setUp()

            rstart, rend = self.Diag_sol.getOwnershipRange()
            # now only set the 1's at surface dofs
            for i, j in zip(self.rows_fs, self.cols_fs):
                self.Diag_sol.setValue(i, j, 1.0, addv=PETSc.InsertMode.INSERT)
            self.Diag_sol.assemble()

            # create from solid matrix and only keep the necessary columns
            # need to assemble here to get correct sparsity pattern when doing the column product
            self.K_uu_work = fem.petsc.assemble_matrix(self.pbs.jac_uu, self.pbs.dbcs_nofluid)
            self.K_uu_work.assemble()
            # now multiply to grep out the correct columns / rows
            self.K_uv = self.K_uu_work.matMult(self.Diag_sol)
            self.K_vu = self.Diag_sol.transposeMatMult(self.K_uu_work)

            if self.pbs.incompressible_2field:
                self.K_up_work = fem.petsc.assemble_matrix(self.pbs.jac_up, self.pbs.dbcs_nofluid)
                self.K_up_work.assemble()
                self.K_vps = self.Diag_sol.transposeMatMult(self.K_up_work)

            # contributions to fluid main block on interface
            self.K_vv_i = self.Diag_sol.transposeMatMult(self.K_uv)

            self.K_uv.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)  # needed so that zeroRows does not change it!

            # create unity vector with 1's everywhere but on surface dofs
            self.Ifo = self.K_vu.createVecRight()
            self.Ifo.array[:] = 1.0
            self.Ifo.setValues(
                self.fdofs_solid_global_sub,
                np.zeros(self.fdofs_solid_global_sub.getLocalSize()),
                addv=PETSc.InsertMode.INSERT,
            )
            self.Ifo.assemble()

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def assemble_residual(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            ofs = 1
        else:
            ofs = 0
        if self.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        self.assemble_residual_coupling(t)
        self.pbs.assemble_residual(t)
        self.pbfa.assemble_residual(t)
        # update of fluid residual - to be done after fluid residual!
        if self.fsi_system == "neumann_dirichlet":
            self.evaluate_residual_forces_interface()
        # solid momentum
        self.r_list[0] = self.pbs.r_list[0]
        if self.pbs.incompressible_2field:
            # solid incompressibility
            self.r_list[1] = self.pbs.r_list[1]
        # fluid momentum
        self.r_list[1 + ofs] = self.pbfa.r_list[0]
        # fluid continuity
        self.r_list[2 + ofs] = self.pbfa.r_list[1]
        # FSI coupling constraint
        if self.fsi_system == "neumann_neumann":
            self.r_list[3 + ofs] = self.r_l
        # ALE
        self.r_list[3 + ofc + ofs] = self.pbfa.r_list[2]

    def assemble_residual_coupling(self, t, subsolver=None):
        if self.fsi_system == "neumann_dirichlet":
            self.evaluate_residual_dbc_coupling() # prior to solid residual assemble!

        if self.fsi_system == "neumann_neumann":
            with self.r_l.localForm() as r_local:
                r_local.set(0.0)
            fem.petsc.assemble_vector(self.r_l, self.res_l)
            fem.apply_lifting(
                self.r_l,
                [self.jac_ll],
                [self.dbcs_lm],
                x0=[self.lm.x.petsc_vec],
                alpha=-1.0,
            )
            self.r_l.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(self.r_l, self.dbcs_lm, x0=self.lm.x.petsc_vec, alpha=-1.0)

    def assemble_stiffness(self, t, subsolver=None):
        if self.pbs.incompressible_2field:
            ofs = 1
        else:
            ofs = 0
        if self.fsi_system == "neumann_neumann":
            ofc = 1
        else:
            ofc = 0

        self.pbs.assemble_stiffness(t)
        self.pbfa.assemble_stiffness(t)
        self.assemble_stiffness_coupling(t)

        # solid momentum
        self.K_list[0][0] = self.pbs.K_list[0][0]  # w.r.t. solid displacement
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbs.K_list[0][1]  # w.r.t. solid pressure
        if self.fsi_system == "neumann_neumann":
            self.K_list[0][3 + ofs] = self.K_ul  # w.r.t. Lagrange multiplier
        if self.fsi_system == "neumann_dirichlet":
            self.K_list[0][1 + ofs] = self.K_uv  # w.r.t. fluid velocity

        # solid incompressibility
        if self.pbs.incompressible_2field:
            self.K_list[1][0] = self.pbs.K_list[1][0]  # w.r.t. solid displacement
            self.K_list[1][1] = self.pbs.K_list[1][1]  # w.r.t. solid pressure

        # fluid momentum
        self.K_list[1 + ofs][1 + ofs] = self.pbfa.K_list[0][0]  # w.r.t. fluid velocity
        self.K_list[1 + ofs][2 + ofs] = self.pbfa.K_list[0][1]  # w.r.t. fluid pressure
        if self.fsi_system == "neumann_neumann":
            self.K_list[1 + ofs][3 + ofs] = self.K_vl  # w.r.t. Lagrange multiplier
        if self.fsi_system == "neumann_dirichlet":
            self.K_list[1 + ofs][0] = self.K_vu  # w.r.t. solid displacement
            if self.pbs.incompressible_2field:
                self.K_list[1 + ofs][1] = self.K_vps  # w.r.t. solid pressure
        self.K_list[1 + ofs][3 + ofc + ofs] = self.pbfa.K_list[0][2]  # w.r.t. ALE displacement

        # fluid continuity
        self.K_list[2 + ofs][1 + ofs] = self.pbf.K_list[1][0]  # w.r.t. fluid velocity
        self.K_list[2 + ofs][2 + ofs] = self.pbf.K_list[1][1]  # w.r.t. fluid pressure
        self.K_list[2 + ofs][3 + ofc + ofs] = self.pbfa.K_list[1][2]  # w.r.t. ALE displacement

        # FSI coupling constraint
        if self.fsi_system == "neumann_neumann":
            self.K_list[3 + ofs][0] = self.K_lu  # w.r.t. solid displacement
            self.K_list[3 + ofs][1 + ofs] = self.K_lv  # w.r.t. fluid velocity
            self.K_list[3 + ofs][3 + ofs] = self.K_ll  # w.r.t. Lagrange multiplier (zero, only LM DBCs)

        # ALE
        self.K_list[3 + ofc + ofs][3 + ofc + ofs] = self.pbfa.K_list[2][2]   # w.r.t. ALE displacement
        self.K_list[3 + ofc + ofs][1 + ofs] = self.pbfa.K_list[2][0]  # w.r.t. fluid velocity

    def assemble_stiffness_coupling(self, t, subsolver=None):
        if self.fsi_system == "neumann_neumann":
            self.K_ul.zeroEntries()
            fem.petsc.assemble_matrix(self.K_ul, self.jac_ul, self.pbs.dbcs)
            self.K_ul.assemble()
            self.K_vl.zeroEntries()
            fem.petsc.assemble_matrix(self.K_vl, self.jac_vl, self.pbf.dbcs)
            self.K_vl.assemble()
            # LM
            self.K_lu.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lu, self.jac_lu, self.dbcs_lm)
            self.K_lu.assemble()
            self.K_lv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lv, self.jac_lv, self.dbcs_lm)
            self.K_lv.assemble()
            # zero matrix, but can have entries if DBCs are given on LM
            self.K_ll.zeroEntries()
            fem.petsc.assemble_matrix(self.K_ll, self.jac_ll, self.dbcs_lm)
            self.K_ll.assemble()

        if self.fsi_system == "neumann_dirichlet":
            # first do stiffness from Dirichlet conditions fluid-to-solid
            self.K_uu_work.zeroEntries()
            fem.petsc.assemble_matrix(self.K_uu_work, self.pbs.jac_uu, self.pbs.dbcs_nofluid)  # need DBCs w/o fluid here
            self.K_uu_work.assemble()

            # we apply u_fluid to solid, hence get du_fluid/dv
            fac = self.pbf.ti.get_factor_deriv_varint(self.pbase.dt)

            # multiply to get the relevant columns only
            self.K_uu_work.matMult(self.Diag_sol, result=self.K_uv)

            # get vv interface contribution prior to modification
            self.Diag_sol.transposeMatMult(self.K_uv, result=self.K_vv_i)
            # scale with du_fluid/dv and update fluid main block
            self.pbf.K_vv.axpy(fac, self.K_vv_i)

            # zero rows where DBC is applied and set interface entries to -1
            self.K_uv.zeroRows(self.rows_fs, diag=0.0)

            for i, j in zip(self.rows_fs, self.cols_fs):
                self.K_uv.setValue(i, j, -1.0, addv=PETSc.InsertMode.INSERT)
            self.K_uv.assemble()

            # scale with time integration factor
            self.K_uv.scale(fac)

            # multiply to get the relevant rows only
            self.Diag_sol.transposeMatMult(self.K_uu_work, result=self.K_vu)

            # eliminate interface!
            self.K_vu.diagonalScale(R=self.Ifo)
            self.K_vu.assemble()

            if self.pbs.incompressible_2field:
                self.K_up_work.zeroEntries()
                fem.petsc.assemble_matrix(self.K_up_work, self.pbs.jac_up, self.pbs.dbcs_nofluid)  # need DBCs w/o fluid here
                self.K_up_work.assemble()

                self.Diag_sol.transposeMatMult(self.K_up_work, result=self.K_vps)
                self.K_vps.assemble()


    def evaluate_residual_dbc_coupling(self):
        # we need a vector representation of ufluid to apply in solid DBCs
        self.pbf.ti.update_varint(
            self.pbf.v.x.petsc_vec,
            self.pbf.v_old.x.petsc_vec,
            self.pbf.uf_old.x.petsc_vec,
            self.pbase.dt,
            varint_veryold=self.pbf.uf_veryold.x.petsc_vec,
            varintout=self.pbf.uf.x.petsc_vec,
            uflform=False,
        )

        # now overwrite interface dofs with uf
        self.pbf.uf.x.petsc_vec.getSubVector(self.fdofs_fluid_global_sub, subvec=self.ufs_subvec)
        self.ufs.x.petsc_vec.setValues(self.fdofs_solid_global_sub, self.ufs_subvec.array)
        self.pbf.uf.x.petsc_vec.restoreSubVector(self.fdofs_fluid_global_sub, subvec=self.ufs_subvec)
        self.ufs.x.petsc_vec.assemble()

        self.ufs.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def evaluate_residual_forces_interface(self):
        with self.r_reac_sol.localForm() as r_local: r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_reac_sol, self.pbs.res_u)
        fem.apply_lifting(
            self.r_reac_sol,
            [self.pbs.jac_uu],
            [self.dbcs_coup_fluid_solid],
            x0=[self.pbs.u.x.petsc_vec],
            alpha=-1.0,
        )
        self.r_reac_sol.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # get solid reaction forces on interface
        self.r_reac_sol.getSubVector(self.fdofs_solid_global_sub, subvec=self.r_u_interface)
        self.r_reac_on_fluid.setValues(self.fdofs_fluid_global_sub, self.r_u_interface.array, addv=PETSc.InsertMode.INSERT)
        self.r_reac_sol.restoreSubVector(self.fdofs_solid_global_sub, subvec=self.r_u_interface)
        self.r_reac_on_fluid.assemble()

        self.r_reac_on_fluid.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) # needed?!

        # add solid residual to fluid
        self.pbf.r_v.axpy(1.0, self.r_reac_on_fluid)

    def get_solver_index_sets(self, isoptions={}):
        # iterative solvers here are only implemented for neumann_dirichlet system!
        assert(self.fsi_system == "neumann_dirichlet")

        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            uvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            uvec_ls = self.rom.V.getLocalSize()[1]
        else:
            uvec_or0 = self.pbs.u.x.petsc_vec.getOwnershipRange()[0]
            uvec_ls = self.pbs.u.x.petsc_vec.getLocalSize()

        offset_u = uvec_or0 + self.pbf.v.x.petsc_vec.getOwnershipRange()[0] + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
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

        offset_d = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_d = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_d,
            step=1,
            comm=self.comm,
       )

        if self.pbs.incompressible_2field:
            ilist = [iset_u, iset_ps, iset_v, iset_p, iset_d]
        else:
            ilist = [iset_u, iset_v, iset_p, iset_d]

        return ilist

    def fluid_to_solid_mapping(self):
        """
        This function perfroms a mapping of fluid and solid interface nodes and builds a global correspondence array. This is necessary for
        the Neumann-Dirichlet FSI routine, since, in general, the interface nodes - though coming from the same parent mesh - are not equally
        distributed across ranks in a 1-to-1 fashion. It can happen that one rank owns more interface nodes from the fluid than from the solid,
        and hence also has a different number of ghost solid vs. fluid nodes. Hence, we need to gather the interface nodes of both solid and fluid
        and do a correspondence mapping using the cKDTree distance function, which here is the only (known) way of establishing this correspondence.
        """

        ts = time.time()
        utilities.print_status(
            "Getting dof mappings from fluid to solid...",
            self.comm,
            e=" ",
        )

        from scipy.spatial import cKDTree

        dofs_s = self.pbs.V_u.tabulate_dof_coordinates()
        dofs_f = self.pbf.V_v.tabulate_dof_coordinates()
        if self.pbs.incompressible_2field:
            dofs_sp = self.pbs.V_p.tabulate_dof_coordinates()

        if all(isinstance(x, int) for x in self.io.surf_interf):
            fnodes_s_loc = fem.locate_dofs_topological(self.pbs.V_u, self.pbs.mesh.topology.dim-1, self.pbs.mt_b.indices[np.isin(self.pbs.mt_b.values, self.io.surf_interf)])
        else: # can only be locator function otherwise...
            fnodes_s_loc_ = []
            for lc in self.io.surf_interf:
                fnodes_s_loc_.append(fem.locate_dofs_geometrical(self.pbs.V_u, lc.evaluate))
            fnodes_s_loc = np.concatenate(fnodes_s_loc_).ravel()
        fnodes_s_glb = np.array(self.pbs.V_u.dofmap.index_map.local_to_global(np.asarray(fnodes_s_loc, dtype=np.int32)), dtype=np.int32)

        if all(isinstance(x, int) for x in self.io.surf_interf):
            fnodes_f_loc = fem.locate_dofs_topological(self.pbf.V_v, self.pbf.mesh.topology.dim-1, self.pbf.mt_b.indices[np.isin(self.pbf.mt_b.values, self.io.surf_interf)])
        else: # can only be locator function otherwise...
            fnodes_f_loc_ = []
            for lc in self.io.surf_interf:
                fnodes_f_loc_.append(fem.locate_dofs_geometrical(self.pbf.V_v, lc.evaluate))
            fnodes_f_loc = np.concatenate(fnodes_f_loc_).ravel()
        fnodes_f_glb = np.array(self.pbf.V_v.dofmap.index_map.local_to_global(np.asarray(fnodes_f_loc, dtype=np.int32)), dtype=np.int32)

        # restrict to owned global
        Istart_s, Iend_s = self.pbs.V_u.dofmap.index_map.local_range
        mask_s = np.logical_and(fnodes_s_glb >= Istart_s, fnodes_s_glb < Iend_s)
        fnodes_s_glb = fnodes_s_glb[mask_s]

        # restrict to owned global
        Istart_f, Iend_f = self.pbf.V_v.dofmap.index_map.local_range
        mask_f = np.logical_and(fnodes_f_glb >= Istart_f, fnodes_f_glb < Iend_f)
        fnodes_f_glb = fnodes_f_glb[mask_f]

        # back to owned local
        fnodes_s_loc = np.array(self.pbs.V_u.dofmap.index_map.global_to_local(np.asarray(fnodes_s_glb, dtype=np.int32)), dtype=np.int32)
        fnodes_f_loc = np.array(self.pbf.V_v.dofmap.index_map.global_to_local(np.asarray(fnodes_f_glb, dtype=np.int32)), dtype=np.int32)

        # assert that they have the same size...
        assert(fnodes_s_loc.size==fnodes_s_glb.size)
        assert(fnodes_f_loc.size==fnodes_f_glb.size)

        # get local node coordinates
        xs_loc = dofs_s[fnodes_s_loc]
        xf_loc = dofs_f[fnodes_f_loc]

        # now gather: we need a global correspondence
        all_xs = self.comm.allgather(xs_loc)
        all_gs = self.comm.allgather(fnodes_s_glb)
        all_xf = self.comm.allgather(xf_loc)
        all_gf = self.comm.allgather(fnodes_f_glb)

        xs = np.vstack(all_xs)
        gs = np.concatenate(all_gs)
        xf = np.vstack(all_xf)
        gf = np.concatenate(all_gf)

        tree = cKDTree(xs)
        dist, idx = tree.query(xf, distance_upper_bound=1e-6)
        if np.isinf(dist).any():
            raise RuntimeError("Unmatched fluid interface dofs")

        solid_of_fluid = gs[idx]
        fluid_glob = gf

        # global gathered node arrays for later index set construction
        self.solid_nodes_glob = np.asarray(solid_of_fluid, dtype=PETSc.IntType)
        self.fluid_nodes_glob = np.asarray(fluid_glob, dtype=PETSc.IntType)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)


    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if N > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        self.pbs.evaluate_initial()
        self.pbfa.evaluate_initial()

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

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

    def write_output(self, N, t, msh=False):
        self.io.write_output(self, N=N, t=t) # combined FSI output routine

    def update(self):
        # update time step - solid and ALE fluid
        self.pbs.update()
        self.pbfa.update()
        self.update_coupling()

    def update_coupling(self):
        if self.fsi_system == "neumann_neumann":
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
            if self.pb.fsi_system=="neumann_dirichlet":
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
                    + self.pb.work_coupling_solid_old # TODO: check sign!
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
            if self.pb.fsi_system=="neumann_dirichlet":
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
                    - self.pb.power_coupling_fluid_old # TODO: check sign!
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
        self.solnln.newton(t, localdata=self.pb.pbs.localdata)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
