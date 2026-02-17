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
        constitutive_models_fluid_ale,
        bc_dict_solid,
        bc_dict_fluid_ale,
        time_curves,
        coupling_params,
        io,
        ios,
        iof,
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
        self.ios, self.iof = ios, iof

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
                ios,
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
                constitutive_models_fluid_ale[0],
                constitutive_models_fluid_ale[1],
                bc_dict_fluid_ale[0],
                bc_dict_fluid_ale[1],
                time_curves,
                coupling_params,
                iof,
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
        self.pbf.results_to_write = io_params["results_to_write"][1][0]
        self.pba.results_to_write = io_params["results_to_write"][1][1]

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

            if self.remove_mutual_solid_fluid_bcs:  # TODO: Seems to not work properly - investigate!

                ndbc_solid, ndbc_fluid = len(self.pbs.dbcs), len(self.pbf.dbcs)

                dbcs_dofs_solid_all, dbcs_dofs_fluid_all = [], []
                for k in range(ndbc_solid):
                    dbcs_dofs_solid_all.append(self.pbs.dbcs[k].dof_indices()[0])
                for k in range(ndbc_fluid):
                    dbcs_dofs_fluid_all.append(self.pbf.dbcs[k].dof_indices()[0])

                dbcs_dofs_solid_all_glob, dbcs_dofs_fluid_all_glob = [], []
                for k in range(ndbc_solid):
                    dbcs_dofs_solid_all_glob.append(self.dofs_solid_main[dbcs_dofs_solid_all[k]])
                # dbcs_dofs_fluid_all_con = np.sort(np.concatenate(dbcs_dofs_fluid_all))
                dbcs_dofs_fluid_all_con = np.concatenate(dbcs_dofs_fluid_all)
                dbcs_dofs_fluid_all_glob = self.dofs_fluid_main[dbcs_dofs_fluid_all_con]

                common_dbcs_glob = []
                for k in range(ndbc_solid):
                    common_dbcs_glob.append( np.intersect1d(dbcs_dofs_solid_all_glob[k], dbcs_dofs_fluid_all_glob) )

                dbc_dofs_solid_all_glob_new = [[] for _ in range(ndbc_solid)]
                for k in range(ndbc_solid):
                    if common_dbcs_glob[k].size > 0:
                        dbc_dofs_solid_all_glob_new[k] = np.setdiff1d(dbcs_dofs_solid_all_glob[k], dbcs_dofs_fluid_all_glob)
                    else:
                        dbc_dofs_solid_all_glob_new[k] = dbcs_dofs_solid_all_glob[k]

                dbcs_dofs_solid_all_new = [[] for _ in range(ndbc_solid)]
                for k in range(ndbc_solid):
                    dbcs_dofs_solid_all_new[k] = dbcs_dofs_solid_all[k][np.argsort(dbc_dofs_solid_all_glob_new[k])]

                dbcs_dofs_solid_all_new_tmp = [[] for _ in range(ndbc_solid)]
                for k in range(ndbc_solid):
                    if self.pbs.bc_dict["dirichlet"][k]['dir']=='all':
                        dbcs_dofs_solid_all_new_tmp[k] = dbcs_dofs_solid_all_new[k]
                    elif self.pbs.bc_dict["dirichlet"][k]['dir']=='x' or self.pbs.bc_dict["dirichlet"][k]['dir']=='y' or self.pbs.bc_dict["dirichlet"][k]['dir']=='z':
                        dbcs_dofs_solid_all_new_tmp[k] = np.empty(bs*len(dbcs_dofs_solid_all_new[k]), dtype=np.int32)
                        for i in range(len(dbcs_dofs_solid_all_new[k])):
                            for j in range(bs):
                                dbcs_dofs_solid_all_new_tmp[k][bs*i+j] = dbcs_dofs_solid_all_new[k][i]

                dbcs_verts_solid_all_new = [[] for _ in range(ndbc_solid)]
                for k in range(ndbc_solid):
                    dbcs_verts_solid_all_new[k] = np.empty(int(len(dbcs_dofs_solid_all_new[k]/bs)), dtype=np.int32)

                for k in range(ndbc_solid):
                    for i in range(len(dbcs_verts_solid_all_new[k])):
                        dbcs_verts_solid_all_new[k][i] = int(dbcs_dofs_solid_all_new_tmp[k][bs*i] / bs)

                self.pbs.dbcs = []
                for k in range(ndbc_solid):
                    # if common_dbcs_glob[k].size > 0: # ...
                    self.pbs.bc_dict["dirichlet"][k]['dir'] += '_by_dofs'
                    self.pbs.bc_dict["dirichlet"][k]['dofs'] = dbcs_verts_solid_all_new[k] # vertices!
                self.pbs.bc.dirichlet_bcs(self.pbs.bc_dict["dirichlet"], self.pbs.dbcs)

                raise RuntimeError("Under development!")

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
        self.remove_mutual_solid_fluid_bcs = self.coupling_params.get("remove_mutual_solid_fluid_bcs", False)

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
        # solid + ALE-fluid
        self.pbs.set_variational_forms()
        self.pbfa.set_variational_forms()
        self.set_variational_forms_coupling()

    def set_variational_forms_coupling(self):
        # fluid displacement, but defined on solid domain
        self.ufs = fem.Function(self.pbs.V_u)

        # establish dof mappings from fluid to solid
        if self.fsi_system=="neumann_dirichlet":
            self.fluid_to_solid_mapping()
            # solid
            self.fdofs_solid_global_sub = meshutils.get_index_set(self.pbs.V_u, self.comm, io=self.pbs.io, idlist=self.io.surf_interf, codim=self.pbs.io.mesh.topology.dim-1, mapper=self.map_s, mask_owned=True)
            if self.pbs.incompressible_2field:
                self.fdofs_solidp_global_sub = meshutils.get_index_set(self.pbs.V_p, self.comm, io=self.pbs.io, idlist=self.io.surf_interf, codim=self.pbs.io.mesh.topology.dim-1, mask_owned=True)
            # fluid
            self.fdofs_fluid_global_sub = meshutils.get_index_set(self.pbf.V_v, self.comm, io=self.pbf.io, idlist=self.io.surf_interf, codim=self.pbf.io.mesh.topology.dim-1, mapper=self.map_f2s, mask_owned=True)
            # check consistency of local size - TODO: There can be partitions where the number of owned dofs per core differes for solid and fluid! Weird, but currently, we have to exclude these cases...
            assert(self.fdofs_solid_global_sub.getSize()==self.fdofs_fluid_global_sub.getSize())
            assert(self.fdofs_solid_global_sub.getLocalSize()==self.fdofs_fluid_global_sub.getLocalSize())

        if self.fsi_system == "neumann_neumann":
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

        elif self.fsi_system == "neumann_dirichlet":

            self.dbcs_coup_fluid_solid = []
            self.dbcs_coup_fluid_solid.append(
                fem.dirichletbc(
                    self.ufs,
                    fem.locate_dofs_topological(
                        self.pbs.V_u,
                        self.pbs.io.mesh.topology.dim - 1,
                        self.pbs.io.mt_b.indices[np.isin(self.pbs.io.mt_b.values, self.io.surf_interf)],
                    ),
                )
            )

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
                    self.dbc_dofs_solid_global.append( meshutils.get_index_set(self.pbs.V_u, self.comm, io=self.pbs.io, idlist=self.pbs.bc_dict["dirichlet"][k]["id"], codim=self.pbs.bc_dict["dirichlet"][k].get("codimension", self.pbs.io.mesh.topology.dim-1), sub=sub, mask_owned=True) )
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
                    self.dbc_dofs_fluid_global.append( meshutils.get_index_set(self.pbf.V_v, self.comm, io=self.pbf.io, idlist=self.pbf.bc_dict["dirichlet"][k]["id"], codim=self.pbf.bc_dict["dirichlet"][k].get("codimension", self.pbf.io.mesh.topology.dim-1), sub=sub, mask_owned=True) )
                dbcs_dofs_fluid_all = []
                for k in range(len(self.dbc_dofs_fluid_global)):
                    dbcs_dofs_fluid_all.append( self.dbc_dofs_fluid_global[k].allGather().array )

                dbcs_dofs_fluid_all_flat = [item for sublist in dbcs_dofs_fluid_all for item in sublist]

                idxs_d = set(dbcs_dofs_fluid_all_flat)
                idxs_i = self.fdofs_fluid_global_sub.getIndices()
                diff = [i for i in idxs_i if i not in idxs_d]

                self.fdofs_fluid_global_sub = PETSc.IS().createGeneral(diff, comm=self.comm)

            # NOTE: If a solid dof of the interface is subject to a DBC, the respective fluid dof necessarily needs that DBC set, too (and vice versa)
            # again check consistency of local size - TODO: There can be partitions where the number of owned dofs per core differes for solid and fluid! Weird, but currently, we have to exclude these cases...
            assert(self.fdofs_solid_global_sub.getSize()==self.fdofs_fluid_global_sub.getSize())
            assert(self.fdofs_solid_global_sub.getLocalSize()==self.fdofs_fluid_global_sub.getLocalSize())

            # further consistency checks (should go eventually...)
            dofs_s = self.pbs.V_u.tabulate_dof_coordinates()[:,:self.pbf.io.mesh.topology.dim].flatten()
            dofs_f = self.pbf.V_v.tabulate_dof_coordinates()[:,:self.pbf.io.mesh.topology.dim].flatten()
            tmp_s = self.pbs.u.x.petsc_vec.copy()
            tmp_f = self.pbf.v.x.petsc_vec.copy()
            tmp_s.zeroEntries()
            tmp_f.zeroEntries()
            ls, le = tmp_s.getOwnershipRange()
            tmp_s[ls:le] = dofs_s[:tmp_s.getLocalSize()]
            ls, le = tmp_f.getOwnershipRange()
            tmp_f[ls:le] = dofs_f[:tmp_f.getLocalSize()]
            tmp_s.assemble()
            tmp_f.assemble()

            tmp_s_new = tmp_s.copy()
            tmp_f_new = tmp_f.copy()
            tmp_s_new.zeroEntries()
            tmp_f_new.zeroEntries()

            sub_s = tmp_s.getSubVector(self.fdofs_solid_global_sub)
            sub_f = tmp_f.getSubVector(self.fdofs_fluid_global_sub)
            tmp_s.zeroEntries()
            tmp_f.zeroEntries()

            # own setting
            tmp_s.setValues(self.fdofs_solid_global_sub, sub_s.array, addv=PETSc.InsertMode.INSERT)
            tmp_f.setValues(self.fdofs_fluid_global_sub, sub_f.array, addv=PETSc.InsertMode.INSERT)

            # cross setting
            tmp_s_new.setValues(self.fdofs_solid_global_sub, sub_f.array, addv=PETSc.InsertMode.INSERT)
            tmp_f_new.setValues(self.fdofs_fluid_global_sub, sub_s.array, addv=PETSc.InsertMode.INSERT)

            assert(np.isclose(tmp_s_new.array, tmp_s.array).all())
            assert(np.isclose(tmp_f_new.array, tmp_f.array).all())

            self.ufs_subvec = self.pbf.uf.x.petsc_vec.getSubVector(self.fdofs_fluid_global_sub)

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

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        # solid + ALE-fluid
        self.pbs.set_problem_vector_matrix_structures()
        self.pbfa.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        if self.fsi_system == "neumann_neumann":
            self.r_l = fem.petsc.assemble_vector(self.res_l)

            self.K_ul = fem.petsc.assemble_matrix(self.jac_ul, self.pbs.dbcs)
            self.K_ul.assemble()
            self.K_vl = fem.petsc.assemble_matrix(self.jac_vl, self.pbf.dbcs)
            self.K_vl.assemble()

            self.K_lu = fem.petsc.assemble_matrix(self.jac_lu, [])
            self.K_lu.assemble()
            self.K_lv = fem.petsc.assemble_matrix(self.jac_lv, [])
            self.K_lv.assemble()

        if self.fsi_system == "neumann_dirichlet":
            # solid reaction forces
            self.r_reac_sol = fem.petsc.assemble_vector(self.pbs.res_u)
            # reaction forces on fluid side
            self.r_reac_on_fluid = fem.petsc.assemble_vector(self.pbf.res_v)

            # get interface solid rhs vector
            self.r_u_interface = self.r_reac_sol.getSubVector(self.fdofs_solid_global_sub)

            # identity and zero numpy arrays
            self.I_loc = np.zeros((self.fdofs_solid_global_sub.getLocalSize(),self.fdofs_fluid_global_sub.getLocalSize()))
            np.fill_diagonal(self.I_loc, 1.0)

            self.Diag_sol = PETSc.Mat().createAIJ(
                (self.pbs.K_uu.getSizes()[0],self.pbf.K_vv.getSizes()[0]),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.comm,
            )
            self.Diag_sol.setUp()

            # now only set the 1's at surface dofs
            self.Diag_sol.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            self.Diag_sol.setValues(self.fdofs_solid_global_sub, self.fdofs_fluid_global_sub, self.I_loc, addv=PETSc.InsertMode.INSERT)
            self.Diag_sol.assemble()

            if self.pbs.incompressible_2field:
                self.I_locp = np.zeros((self.fdofs_solidp_global_sub.getLocalSize(),self.fdofs_fluid_global_sub.getLocalSize()))
                np.fill_diagonal(self.I_locp, 1.0)

                self.Diag_solp = PETSc.Mat().createAIJ(
                    (self.pbs.K_up.getSizes()[0],self.pbf.K_vv.getSizes()[0]),
                    bsize=None,
                    nnz=(1, 1),
                    csr=None,
                    comm=self.comm,
                )
                self.Diag_solp.setUp()

                # now only set the 1's at surface dofs
                self.Diag_solp.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                self.Diag_solp.setValues(self.fdofs_solidp_global_sub, self.fdofs_fluid_global_sub, self.I_locp, addv=PETSc.InsertMode.INSERT)
                self.Diag_solp.assemble()

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
                self.K_vps = self.Diag_solp.transposeMatMult(self.K_up_work)
                # self.K_vps.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

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
            self.r_l.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

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
        self.K_list[0][0] = self.pbs.K_list[0][0]
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbs.K_list[0][1]
        if self.fsi_system == "neumann_neumann":
            self.K_list[0][3 + ofs] = self.K_ul
        if self.fsi_system == "neumann_dirichlet":
            self.K_list[0][1 + ofs] = self.K_uv
        # solid incompressibility
        if self.pbs.incompressible_2field:
            self.K_list[1][0] = self.pbs.K_list[1][0]
            self.K_list[1][1] = self.pbs.K_list[1][1]
            if self.fsi_system == "neumann_dirichlet":
                self.K_list[1 + ofs][1] = self.K_vps
        # fluid momentum
        self.K_list[1 + ofs][1 + ofs] = self.pbfa.K_list[0][0]
        self.K_list[1 + ofs][2 + ofs] = self.pbfa.K_list[0][1]
        if self.fsi_system == "neumann_neumann":
            self.K_list[1 + ofs][3 + ofs] = self.K_vl
        if self.fsi_system == "neumann_dirichlet":
            self.K_list[1 + ofs][0] = self.K_vu
        self.K_list[1 + ofs][3 + ofc + ofs] = self.pbfa.K_list[0][2]
        # fluid continuity
        self.K_list[2 + ofs][1 + ofs] = self.pbf.K_list[1][0]
        self.K_list[2 + ofs][2 + ofs] = self.pbf.K_list[1][1]
        self.K_list[2 + ofs][3 + ofc + ofs] = self.pbfa.K_list[1][2]
        # FSI coupling constraint
        if self.fsi_system == "neumann_neumann":
            self.K_list[3 + ofs][0] = self.K_lu
            self.K_list[3 + ofs][1 + ofs] = self.K_lv
        # ALE
        self.K_list[3 + ofc + ofs][3 + ofc + ofs] = self.pbfa.K_list[2][2]
        self.K_list[3 + ofc + ofs][1 + ofs] = self.pbfa.K_list[2][0]

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
            fem.petsc.assemble_matrix(self.K_lu, self.jac_lu, [])
            self.K_lu.assemble()
            self.K_lv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_lv, self.jac_lv, [])
            self.K_lv.assemble()

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
            self.K_uv.zeroRows(self.fdofs_solid_global_sub, diag=0.0)
            self.K_uv.setValues(self.fdofs_solid_global_sub, self.fdofs_fluid_global_sub, -self.I_loc, addv=PETSc.InsertMode.INSERT)
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

                self.Diag_solp.transposeMatMult(self.K_up_work, result=self.K_vps)
                self.K_vps.assemble()

                for i in range(len(self.dbc_dofs_fluid_global)):
                    self.K_vps.zeroRows(self.dbc_dofs_fluid_global[i], diag=0.0)

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
        ts = time.time()
        utilities.print_status(
            "Getting dof mappings from fluid to solid...",
            self.comm,
            e=" ",
        )

        from scipy.spatial import cKDTree

        dofs_s = self.pbs.V_u.tabulate_dof_coordinates()
        dofs_f = self.pbf.V_v.tabulate_dof_coordinates()

        fnodes_s_loc = fem.locate_dofs_topological(self.pbs.V_u, self.pbs.io.mesh.topology.dim-1, self.pbs.io.mt_b.indices[np.isin(self.pbs.io.mt_b.values, self.io.surf_interf)])
        fnodes_s_glb = np.array(self.pbs.V_u.dofmap.index_map.local_to_global(np.asarray(fnodes_s_loc, dtype=np.int32)), dtype=np.int32)

        fnodes_f_loc = fem.locate_dofs_topological(self.pbf.V_v, self.pbf.io.mesh.topology.dim-1, self.pbf.io.mt_b.indices[np.isin(self.pbf.io.mt_b.values, self.io.surf_interf)])
        fnodes_f_glb = np.array(self.pbf.V_v.dofmap.index_map.local_to_global(np.asarray(fnodes_f_loc, dtype=np.int32)), dtype=np.int32)

        # fluid to solid mapping
        tree = cKDTree(dofs_s[fnodes_s_loc])
        _, self.map_s = tree.query(dofs_s[fnodes_s_loc], distance_upper_bound=1e-6) # identity map!
        _, self.map_f2s = tree.query(dofs_f[fnodes_f_loc], distance_upper_bound=1e-6)
        self.map_s = np.argsort(self.map_s)
        self.map_f2s = np.argsort(self.map_f2s)

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
        # self.pba.write_output(N, t)

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
