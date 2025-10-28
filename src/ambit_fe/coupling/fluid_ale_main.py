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
from .. import utilities, ioparams

from ..fluid.fluid_main import (
    FluidmechanicsProblem,
    FluidmechanicsSolverPrestr,
)
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base


class FluidmechanicsAleProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params,
        fem_params_fluid,
        fem_params_ale,
        constitutive_models_fluid,
        constitutive_models_ale,
        bc_dict_fluid,
        bc_dict_ale,
        time_curves,
        coupling_params,
        io,
        mor_params={},
        pbf=None,
        pba=None,
    ):
        self.pbase = pbase
        self.pbf = pbf
        self.pba = pba

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_coupling_fluid_ale(coupling_params)

        self.problem_physics = "fluid_ale"

        (
            self.have_dbc_fluid_ale,
            self.have_weak_dirichlet_fluid_ale,
            self.have_dbc_ale_fluid,
            self.have_robin_ale_fluid,
        ) = False, False, False, False

        # instantiate problem classes
        # ALE
        if pba is None:
            self.pba = AleProblem(
                pbase,
                io_params,
                time_params,
                fem_params_ale,
                constitutive_models_ale,
                bc_dict_ale,
                time_curves,
                io,
                mor_params=mor_params,
            )
            # ALE variables that are handed to fluid problem
            alevariables = {
                "Fale": self.pba.ki.F(self.pba.d),
                "Fale_old": self.pba.ki.F(self.pba.d_old),
                "w": self.pba.wel,
                "w_old": self.pba.w_old,
            }
        # fluid
        if pbf is None:
            self.pbf = FluidmechanicsProblem(
                pbase,
                io_params,
                time_params,
                fem_params_fluid,
                constitutive_models_fluid,
                bc_dict_fluid,
                time_curves,
                io,
                mor_params=mor_params,
                alevar=alevariables,
            )

        self.coupling_params = coupling_params
        self.set_coupling_parameters()

        self.pbrom = self.pbf  # ROM problem can only be fluid
        self.pbrom_host = self

        # modify results to write...
        self.pbf.results_to_write = io_params["results_to_write"][0]
        self.pba.results_to_write = io_params["results_to_write"][1]

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False

        # NOTE: Fluid and ALE function spaces should be of the same type, but are different objects.
        # For some reason, when applying a function from one funtion space as DBC to another function space,
        # errors occur. Therefore, we define these auxiliary variables and interpolate respectively...

        # fluid displacement, but defined within ALE function space
        self.ufa = fem.Function(self.pba.V_d)
        # ALE velocity, but defined within fluid function space
        self.wf = fem.Function(self.pbf.V_v)

        self.set_variational_forms()

        if self.coupling_strategy == "monolithic":
            self.numdof = self.pbf.numdof + self.pba.numdof
        else:
            self.numdof = [self.pbf.numdof, self.pba.numdof]

        self.sub_solve = False
        self.print_subiter = False
        self.have_condensed_variables = False

        self.io = self.pbf.io

        # number of fields involved
        self.nfields = 3

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
        self.coupling_fluid_ale = self.coupling_params.get("coupling_fluid_ale", {})
        self.coupling_ale_fluid = self.coupling_params.get("coupling_ale_fluid", {})
        self.coupling_strategy = self.coupling_params.get("coupling_strategy", "monolithic")

    def get_problem_var_list(self):
        if self.pbf.num_dupl > 1:
            is_ghosted = [1, 2, 1]
        else:
            is_ghosted = [1, 1, 1]
        return [
            self.pbf.v.x.petsc_vec,
            self.pbf.p.x.petsc_vec,
            self.pba.d.x.petsc_vec,
        ], is_ghosted

    # defines the monolithic coupling forms for fluid mechanics in ALE reference frame
    def set_variational_forms(self):
        # any DBC conditions that we want to set from fluid to ALE (mandatory for FSI or FrSI)
        if bool(self.coupling_fluid_ale):
            dbcs_coup_fluid_ale, work_weak_dirichlet_fluid_ale = (
                [],
                ufl.as_ufl(0),
            )

            for j in range(len(self.coupling_fluid_ale)):
                ids_fluid_ale = self.coupling_fluid_ale[j]["surface_ids"]

                if self.coupling_fluid_ale[j]["type"] == "strong_dirichlet":
                    dbcs_coup_fluid_ale.append(
                        fem.dirichletbc(
                            self.ufa,
                            fem.locate_dofs_topological(
                                self.pba.V_d,
                                self.io.mesh.topology.dim - 1,
                                self.io.mt_b1.indices[np.isin(self.io.mt_b1.values, ids_fluid_ale)],
                            ),
                        )
                    )

                    # get surface dofs for dr_ALE/dv matrix entry
                    fnode_indices_local = fem.locate_dofs_topological(
                        self.pba.V_d,
                        self.pba.io.mesh.topology.dim - 1,
                        self.pba.io.mt_b1.indices[np.isin(self.pba.io.mt_b1.values, ids_fluid_ale)],
                    )
                    fnode_indices_all = np.array(
                        self.pba.V_d.dofmap.index_map.local_to_global(np.asarray(fnode_indices_local, dtype=np.int32)),
                        dtype=np.int32,
                    )
                    self.fdofs = PETSc.IS().createBlock(
                        self.pba.V_d.dofmap.index_map_bs,
                        fnode_indices_all,
                        comm=self.comm,
                    )

                elif self.coupling_fluid_ale[j]["type"] == "weak_dirichlet":
                    beta = self.coupling_fluid_ale[j]["beta"]
                    hscale = self.coupling_fluid_ale[j].get("hscale", False)

                    for i in range(len(ids_fluid_ale)):
                        db_ = self.pba.bmeasures[0](ids_fluid_ale[i])

                        for n in range(self.pba.num_domains):
                            work_weak_dirichlet_fluid_ale += self.pba.vf.deltaW_int_nitsche_dirichlet(
                                self.pba.d,
                                self.pbf.ufluid,
                                self.pba.ma[n].stress(self.pba.var_d, self.pba.var_d),
                                beta,
                                db_,
                                hscale=hscale,
                            )  # here, ufluid as form is used!

                else:
                    raise ValueError("Unknown coupling_fluid_ale option for fluid to ALE!")

            # now add the DBCs: pay attention to order... first u=uf, then the others... hence re-set!
            if bool(dbcs_coup_fluid_ale):
                # store DBCs without those from fluid
                self.pba.bc.dbcs_nofluid = []
                for k in self.pba.bc.dbcs:
                    self.pba.bc.dbcs_nofluid.append(k)
                self.pba.bc.dbcs = []
                self.pba.bc.dbcs += dbcs_coup_fluid_ale
                # Dirichlet boundary conditions
                if "dirichlet" in self.pba.bc_dict.keys():
                    self.pba.bc.dirichlet_bcs(self.pba.bc_dict["dirichlet"])
                self.have_dbc_fluid_ale = True

            if not isinstance(work_weak_dirichlet_fluid_ale, ufl.constantvalue.Zero):
                # add to ALE internal virtual work
                self.pba.weakform_d += work_weak_dirichlet_fluid_ale
                # add to ALE jacobian form and define offdiagonal derivative w.r.t. fluid
                self.pba.weakform_lin_dd += ufl.derivative(work_weak_dirichlet_fluid_ale, self.pba.d, self.pba.dd)
                self.weakform_lin_dv = ufl.derivative(
                    work_weak_dirichlet_fluid_ale, self.pbf.v, self.pbf.dv
                )  # only contribution is from weak DBC here!
                self.have_weak_dirichlet_fluid_ale = True

        # any DBC conditions that we want to set from ALE to fluid
        if bool(self.coupling_ale_fluid):
            (
                dbcs_coup_ale_fluid,
                work_robin_ale_fluid,
                work_robin_ale_fluid_old,
                work_robin_ale_fluid_mid,
            ) = [], ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

            for j in range(len(self.coupling_ale_fluid)):
                ids_ale_fluid = self.coupling_ale_fluid[j]["surface_ids"]

                if self.coupling_ale_fluid[j]["type"] == "strong_dirichlet":
                    dbcs_coup_ale_fluid.append(
                        fem.dirichletbc(
                            self.wf,
                            fem.locate_dofs_topological(
                                self.pbf.V_v,
                                self.io.mesh.topology.dim - 1,
                                self.io.mt_b1.indices[np.isin(self.io.mt_b1.values, ids_ale_fluid)],
                            ),
                        )
                    )

                    # NOTE: linearization entries due to strong DBCs of ALE on fluid are currently not considered in the monolithic block matrix!

                elif self.coupling_ale_fluid[j]["type"] == "robin":
                    for i in range(len(ids_ale_fluid)):
                        if self.coupling_ale_fluid[j]["type"] == "robin":
                            beta = self.coupling_ale_fluid[j]["beta"]
                            db_ = self.pbf.bmeasures[0](ids_ale_fluid[i])
                            work_robin_ale_fluid += self.pbf.vf.deltaW_int_robin_cur(
                                self.pbf.v,
                                self.pba.wel,
                                beta,
                                db_,
                                Fale=self.pba.ki.F(self.pba.d),
                            )  # here, wel as form is used!
                            work_robin_ale_fluid_old += self.pbf.vf.deltaW_int_robin_cur(
                                self.pbf.v_old,
                                self.pba.w_old,
                                beta,
                                db_,
                                Fale=self.pba.ki.F(self.pba.d_old),
                            )
                            work_robin_ale_fluid_mid += self.pbf.vf.deltaW_int_robin_cur(
                                self.pbf.vel_mid,
                                self.pbf.timefac * self.pba.wel + (1.0 - self.pbf.timefac) * self.pba.w_old,
                                beta,
                                db_,
                                Fale=self.pba.ki.F(
                                    self.pbf.timefac * self.pba.d + (1.0 - self.pbf.timefac) * self.pba.d_old
                                ),
                            )

                else:
                    raise ValueError("Unknown coupling_ale_fluid option for ALE to fluid!")

            if bool(dbcs_coup_ale_fluid):
                # now add the DBCs: pay attention to order... first v=w, then the others... hence re-set!
                self.pbf.bc.dbcs = []
                self.pbf.bc.dbcs += dbcs_coup_ale_fluid
                # Dirichlet boundary conditions
                if "dirichlet" in self.pbf.bc_dict.keys():
                    self.pbf.bc.dirichlet_bcs(self.pbf.bc_dict["dirichlet"])
                self.have_dbc_ale_fluid = True

            if not isinstance(work_robin_ale_fluid, ufl.constantvalue.Zero):
                if self.pbf.ti.eval_nonlin_terms == "trapezoidal":
                    # add to fluid internal virtual power
                    self.pbf.weakform_v += (
                        self.pbf.timefac * work_robin_ale_fluid + (1.0 - self.pbf.timefac) * work_robin_ale_fluid_old
                    )
                    # add to fluid jacobian form
                    self.pbf.weakform_lin_vv += self.pbf.timefac * ufl.derivative(
                        work_robin_ale_fluid, self.pbf.v, self.pbf.dv
                    )
                if self.pbf.ti.eval_nonlin_terms == "midpoint":
                    # add to fluid internal virtual power
                    self.pbf.weakform_v += work_robin_ale_fluid_mid
                    # add to fluid jacobian form
                    self.pbf.weakform_lin_vv += ufl.derivative(work_robin_ale_fluid_mid, self.pbf.v, self.pbf.dv)
                self.have_robin_ale_fluid = True

        # derivative of fluid momentum w.r.t. ALE displacement - also includes potential weak Dirichlet or Robin BCs from ALE to fluid!
        self.weakform_lin_vd = ufl.derivative(self.pbf.weakform_v, self.pba.d, self.pba.dd)

        # derivative of fluid continuity w.r.t. ALE displacement
        self.weakform_lin_pd = []
        for n in range(self.pbf.num_domains):
            self.weakform_lin_pd.append(ufl.derivative(self.pbf.weakform_p[n], self.pba.d, self.pba.dd))

    def set_problem_residual_jacobian_forms(self, pre=False):
        # fluid + ALE
        self.pbf.set_problem_residual_jacobian_forms(pre=pre)
        self.pba.set_problem_residual_jacobian_forms()
        self.set_problem_residual_jacobian_forms_coupling()

    def set_problem_residual_jacobian_forms_coupling(self):
        if self.coupling_strategy == "monolithic":
            ts = time.time()
            utilities.print_status(
                "FEM form compilation for fluid-ALE coupling...",
                self.comm,
                e=" ",
            )

            if not bool(self.pbf.io.duplicate_mesh_domains):
                self.weakform_lin_pd = sum(self.weakform_lin_pd)

            # coupling
            self.jac_vd = fem.form(self.weakform_lin_vd, entity_maps=self.io.entity_maps)
            self.jac_pd = fem.form(self.weakform_lin_pd, entity_maps=self.io.entity_maps)
            if self.pbf.num_dupl > 1:
                self.jac_pd_ = []
                for j in range(self.pbf.num_dupl):
                    self.jac_pd_.append([self.jac_pd[j]])
            if self.have_weak_dirichlet_fluid_ale:
                self.jac_dv = fem.form(self.weakform_lin_dv, entity_maps=self.io.entity_maps)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.comm)

    def set_problem_vector_matrix_structures(self):
        self.pbf.set_problem_vector_matrix_structures()
        self.pba.set_problem_vector_matrix_structures()
        self.set_problem_vector_matrix_structures_coupling()

    def set_problem_vector_matrix_structures_coupling(self):
        if self.coupling_strategy == "monolithic":
            self.K_vd = fem.petsc.create_matrix(self.jac_vd)
            if self.have_weak_dirichlet_fluid_ale:
                self.K_dv = fem.petsc.create_matrix(self.jac_dv)
            elif self.have_dbc_fluid_ale:
                # create unity vector with 1's on surface dofs and zeros elsewhere
                self.Iale = self.pba.K_dd.createVecLeft()
                self.Iale.setValues(
                    self.fdofs,
                    np.ones(self.fdofs.getLocalSize()),
                    addv=PETSc.InsertMode.INSERT,
                )
                self.Iale.assemble()
                # create diagonal matrix
                self.Diag_ale = PETSc.Mat().createAIJ(
                    self.pba.K_dd.getSizes(),
                    bsize=None,
                    nnz=(1, 1),
                    csr=None,
                    comm=self.comm,
                )
                self.Diag_ale.setUp()
                self.Diag_ale.assemble()
                # set 1's to get correct allocation pattern
                self.Diag_ale.shift(1.0)
                # now only set the 1's at surface dofs
                self.Diag_ale.setDiagonal(self.Iale, addv=PETSc.InsertMode.INSERT)
                self.Diag_ale.assemble()
                # create from ALE matrix and only keep the necessary columns
                # need to assemble here to get correct sparsity pattern when doing the column product
                self.K_dv_ = fem.petsc.assemble_matrix(self.pba.jac_dd, [])
                self.K_dv_.assemble()
                # now multiply to grep out the correct columns
                self.K_dv = self.K_dv_.matMult(self.Diag_ale)
                self.K_dv.setOption(
                    PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True
                )  # needed so that zeroRows does not change it!
            else:
                self.K_dv = None

            if self.pbf.num_dupl > 1:
                self.K_pd = fem.petsc.create_matrix(self.jac_pd_)
            else:
                self.K_pd = fem.petsc.create_matrix(self.jac_pd)

    def assemble_residual(self, t, subsolver=None):
        # prior to ALE residual assemble!
        self.assemble_residual_coupling(t)

        self.pbf.assemble_residual(t)
        self.pba.assemble_residual(t)

        self.r_list[0] = self.pbf.r_list[0]
        self.r_list[1] = self.pbf.r_list[1]
        self.r_list[2] = self.pba.r_list[0]

    def assemble_residual_coupling(self, t, subsolver=None):
        self.evaluate_residual_dbc_coupling()

    def assemble_stiffness(self, t, subsolver=None):
        self.assemble_stiffness_coupling(t)

        self.pbf.assemble_stiffness(t)
        self.pba.assemble_stiffness(t)

        self.K_list[0][0] = self.pbf.K_list[0][0]
        self.K_list[0][1] = self.pbf.K_list[0][1]
        self.K_list[1][0] = self.pbf.K_list[1][0]
        self.K_list[1][1] = self.pbf.K_list[1][1]

        self.K_list[2][2] = self.pba.K_list[0][0]

    def assemble_stiffness_coupling(self, t):
        if self.have_weak_dirichlet_fluid_ale:
            self.K_dv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_dv, self.jac_dv, self.pba.bc.dbcs)
            self.K_dv.assemble()
        elif self.have_dbc_fluid_ale:
            self.K_dv_.zeroEntries()
            fem.petsc.assemble_matrix(self.K_dv_, self.pba.jac_dd, self.pba.bc.dbcs_nofluid)  # need DBCs w/o fluid here
            self.K_dv_.assemble()
            # multiply to get the relevant columns only
            self.K_dv_.matMult(self.Diag_ale, result=self.K_dv)
            # zero rows where DBC is applied and set diagonal entry to -1
            self.K_dv.zeroRows(self.fdofs, diag=-1.0)
            # we apply u_fluid to ALE, hence get du_fluid/dv
            fac = self.pbf.ti.get_factor_deriv_varint(self.pbase.dt)
            self.K_dv.scale(fac)

        self.K_list[2][0] = self.K_dv

        # derivative of fluid momentum w.r.t. ALE displacement
        self.K_vd.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vd, self.jac_vd, self.pbf.bc.dbcs)
        self.K_vd.assemble()
        self.K_list[0][2] = self.K_vd

        # derivative of fluid continuity w.r.t. ALE displacement
        self.K_pd.zeroEntries()
        if self.pbf.num_dupl > 1:
            fem.petsc.assemble_matrix(self.K_pd, self.jac_pd_, [])
        else:
            fem.petsc.assemble_matrix(self.K_pd, self.jac_pd, [])
        self.K_pd.assemble()
        self.K_list[1][2] = self.K_pd

    def evaluate_residual_dbc_coupling(self):
        if self.have_dbc_fluid_ale:
            # we need a vector representation of ufluid to apply in ALE DBCs
            self.pbf.ti.update_varint(
                self.pbf.v.x.petsc_vec,
                self.pbf.v_old.x.petsc_vec,
                self.pbf.uf_old.x.petsc_vec,
                self.pbase.dt,
                varintout=self.pbf.uf.x.petsc_vec,
                uflform=False,
            )
            self.ufa.x.petsc_vec.axpby(1.0, 0.0, self.pbf.uf.x.petsc_vec)
            self.ufa.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if self.have_dbc_ale_fluid:
            # we need a vector representation of w to apply in fluid DBCs
            self.pba.ti.update_dvar(
                self.pba.d.x.petsc_vec,
                self.pba.d_old.x.petsc_vec,
                self.pba.w_old.x.petsc_vec,
                self.pbase.dt,
                dvarout=self.pba.w.x.petsc_vec,
                uflform=False,
            )
            self.wf.x.petsc_vec.axpby(1.0, 0.0, self.pba.w.x.petsc_vec)
            self.wf.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def get_index_sets(self, isoptions={}):
        if self.rom is not None:  # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.x.petsc_vec.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.x.petsc_vec.getLocalSize()

        offset_v = (
            vvec_or0 + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
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

        offset_d = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_d = PETSc.IS().createStride(
            self.pba.d.x.petsc_vec.getLocalSize(),
            first=offset_d,
            step=1,
            comm=self.comm,
        )

        if isoptions["ale_to_v"]:
            iset_v = iset_v.expand(iset_d)  # add ALE to velocity block

        if isoptions["rom_to_new"]:
            ilist = [iset_v, iset_p, iset_r, iset_d]
        else:
            ilist = [iset_v, iset_p, iset_d]

        if isoptions["ale_to_v"]:
            ilist.pop(-1)

        return ilist

    # DEPRECATED: This is something we should actually not do! It will mess with gradients we need w.r.t. the reference (e.g. for FrSI)
    # Instead of moving the mesh, we formulate Navier-Stokes w.r.t. a reference state using the ALE kinematics
    def move_mesh(self):
        d = fem.Function(self.pba.Vcoord)
        d.interpolate(self.pba.d)
        self.io.mesh.geometry.x[:, : self.pba.dim] += d.x.array.reshape((-1, self.pba.dim))
        utilities.print_status("Updating mesh...", self.comm)

    def print_warning_ale(self):
        utilities.print_status(" ", self.comm)
        utilities.print_status(
            "*********************************************************************************************************************",
            self.comm,
        )
        utilities.print_status(
            "*** Warning: You are solving Navier-Stokes by only updating the frame after each time step! This is inconsistent! ***",
            self.comm,
        )
        utilities.print_status(
            "*********************************************************************************************************************",
            self.comm,
        )
        utilities.print_status(" ", self.comm)

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if N > 0:
            self.io.readcheckpoint(self, N)

    def evaluate_initial(self):
        self.pbf.evaluate_initial()

    def write_output_ini(self):
        self.io.write_output(self, writemesh=True)

    def write_output_pre(self):
        self.pbf.write_output_pre()
        self.pba.write_output_pre()

    def evaluate_pre_solve(self, t, N, dt):
        self.pbf.evaluate_pre_solve(t, N, dt)
        self.pba.evaluate_pre_solve(t, N, dt)

    def evaluate_post_solve(self, t, N):
        self.pbf.evaluate_post_solve(t, N)
        self.pba.evaluate_post_solve(t, N)

    def set_output_state(self, N):
        self.pbf.set_output_state(N)
        self.pba.set_output_state(N)

    def write_output(self, N, t, mesh=False):
        self.io.write_output(self, N=N, t=t)

    def update(self):
        # update time step - fluid and ALE
        self.pbf.update()
        self.pba.update()

    def print_to_screen(self):
        self.pbf.print_to_screen()
        self.pba.print_to_screen()

    def induce_state_change(self):
        self.pbf.induce_state_change()
        self.pba.induce_state_change()

    def write_restart(self, sname, N, force=False):
        self.io.write_restart(self, N, force=force)

    def check_abort(self, t):
        return False

    def destroy(self):
        self.pbf.destroy()
        self.pba.destroy()


class FluidmechanicsAleSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbf.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        if self.pb.coupling_strategy == "monolithic":
            self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)
        elif self.pb.coupling_strategy == "partitioned":
            self.solnln = solver_nonlin.solver_nonlinear([self.pb.pbf, self.pb.pba], self.solver_params, cp=self.pb)
        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")

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
        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the FrSI problem
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
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv)  # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa = (
                fem.form(weakform_a, entity_maps=self.pb.io.entity_maps),
                fem.form(weakform_lin_aa, entity_maps=self.pb.io.entity_maps),
            )
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    # we overload this function here in order to take care of the partitioned solve,
    # where the ROM needs to be an object of the fluid, not the coupled problem
    def evaluate_assemble_system_initial(self, subsolver=None):
        # evaluate old initial state of model
        self.evaluate_system_initial()

        if self.pb.coupling_strategy == "monolithic":
            self.pb.assemble_residual(self.pb.pbase.t_init)
            self.pb.assemble_stiffness(self.pb.pbase.t_init)

            # create ROM matrix structures
            if self.pb.rom:
                self.pb.rom.set_reduced_data_structures_residual(self.pb.r_list, self.pb.r_list_rom)
                self.pb.K_list_tmp = [[None]]
                self.pb.rom.set_reduced_data_structures_matrix(self.pb.K_list, self.pb.K_list_rom, self.pb.K_list_tmp)

                if self.pb.pbf.pre:
                    self.pb.pbf.rom = self.pb.rom
                    self.pb.pbf.rom.set_reduced_data_structures_residual(self.pb.pbf.r_list, self.pb.pbf.r_list_rom)
                    self.pb.pbf.K_list_tmp = [[None]]
                    self.pb.pbf.rom.set_reduced_data_structures_matrix(
                        self.pb.pbf.K_list,
                        self.pb.pbf.K_list_rom,
                        self.pb.pbf.K_list_tmp,
                    )

        elif self.pb.coupling_strategy == "partitioned":
            self.pb.pbf.rom = self.pb.rom
            self.pb.pbrom_host = self.pb.pbf  # overridden

            self.pb.assemble_residual(self.pb.pbase.t_init)
            self.pb.pbf.assemble_stiffness(self.pb.pbase.t_init)
            self.pb.pba.assemble_stiffness(self.pb.pbase.t_init)

            # create ROM matrix structures
            if self.pb.pbf.rom:
                self.pb.pbf.rom.set_reduced_data_structures_residual(self.pb.pbf.r_list, self.pb.pbf.r_list_rom)
                self.pb.pbf.K_list_tmp = [[None]]
                self.pb.pbf.rom.set_reduced_data_structures_matrix(
                    self.pb.pbf.K_list,
                    self.pb.pbf.K_list_rom,
                    self.pb.pbf.K_list_tmp,
                )

        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
