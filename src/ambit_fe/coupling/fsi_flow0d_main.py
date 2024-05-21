#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
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
from ..solid.solid_main import SolidmechanicsProblem
from .fluid_ale_flow0d_main import FluidmechanicsAleFlow0DProblem
from ..base import problem_base, solver_base


class FSIFlow0DProblem(FSIProblem,problem_base):

    def __init__(self, pbase, io_params, time_params_solid, time_params_fluid, time_params_flow0d, fem_params_solid, fem_params_fluid, fem_params_ale, constitutive_models_solid, constitutive_models_fluid_ale, model_params_flow0d, bc_dict_solid, bc_dict_fluid_ale, time_curves, coupling_params_fluid_ale, coupling_params_fluid_flow0d, io, ios, iof, mor_params={}):
        # problem_base.__init__(self, io_params, time_params_solid, comm=comm, comm_sq=comm_sq)

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_coupling_fluid_ale(coupling_params_fluid_ale)

        self.problem_physics = 'fsi_flow0d'

        self.io = io
        self.ios, self.iof = ios, iof

        # assert that we do not have conflicting timings - TODO: Find better solution by moving these to global control parameters...
        assert(time_params_fluid['maxtime'] == time_params_solid['maxtime'])
        if 'numstep' in time_params_solid.keys() or 'numstep' in time_params_fluid.keys():
            assert(time_params_fluid['numstep'] == time_params_solid['numstep'])
        if 'dt' in time_params_solid.keys() or 'dt' in time_params_fluid.keys():
            assert(time_params_fluid['dt'] == time_params_solid['dt'])

        try: self.fsi_governing_type = self.coupling_params['fsi_governing_type']
        except: self.fsi_governing_type = 'solid_governed'

        try: self.zero_lm_boundary = self.coupling_params['zero_lm_boundary']
        except: self.zero_lm_boundary = False

        self.have_condensed_variables = False

        self.have_dbc_fluid_ale, self.have_weak_dirichlet_fluid_ale, self.have_dbc_ale_fluid, self.have_robin_ale_fluid = False, False, False, False

        # initialize problem instances (also sets the variational forms for the fluid flow0d problem)
        self.pbs   = SolidmechanicsProblem(pbase, io_params, time_params_solid, fem_params_solid, constitutive_models_solid, bc_dict_solid, time_curves, ios, mor_params=mor_params)
        self.pbfa0 = FluidmechanicsAleFlow0DProblem(pbase, io_params, time_params_fluid, time_params_flow0d, fem_params_fluid, fem_params_ale, constitutive_models_fluid_ale[0], constitutive_models_fluid_ale[1], model_params_flow0d, bc_dict_fluid_ale[0], bc_dict_fluid_ale[1], time_curves, coupling_params_fluid_ale, coupling_params_fluid_flow0d, iof, mor_params=mor_params)

        self.pbf = self.pbfa0.pbf
        self.pbf0 = self.pbfa0.pbf0
        self.pb0 = self.pbfa0.pb0
        self.pba = self.pbfa0.pba

        self.pbrom = self.pbs # ROM problem can only be solid
        self.pbrom_host = self

        # modify results to write...
        self.pbs.results_to_write = io_params['results_to_write'][0]
        self.pbf.results_to_write = io_params['results_to_write'][1][0]
        self.pba.results_to_write = io_params['results_to_write'][1][1]

        self.incompressible_2field = self.pbs.incompressible_2field

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.print_subiter = self.pbf0.print_subiter

        P_lm = ufl.VectorElement("CG", self.io.msh_emap_lm[0].ufl_cell(), self.pbs.order_disp)
        self.V_lm = fem.FunctionSpace(self.io.msh_emap_lm[0], P_lm)

        # Lagrange multiplier
        self.lm = fem.Function(self.V_lm)
        self.lm_old = fem.Function(self.V_lm)

        self.dlm = ufl.TrialFunction(self.V_lm)    # incremental LM
        self.var_lm = ufl.TestFunction(self.V_lm)  # LM test function

        self.bclm = boundaryconditions.boundary_cond(self.io, dim=self.io.msh_emap_lm[0].topology.dim)
        # set the whole boundary of the LM subspace to zero (beneficial when we have solid and fluid with overlapping DBCs)
        if self.zero_lm_boundary: # TODO: Seems to not work properly!
            self.io.msh_emap_lm[0].topology.create_connectivity(self.io.msh_emap_lm[0].topology.dim-1, self.io.msh_emap_lm[0].topology.dim)
            boundary_facets_lm = mesh.exterior_facet_indices(self.io.msh_emap_lm[0].topology)
            self.bclm.dbcs.append( fem.dirichletbc(self.lm, boundary_facets_lm) )

        self.set_variational_forms()

        self.numdof = self.pbs.numdof + self.pbfa0.numdof + self.lm.vector.getSize()

        self.sub_solve = True
        self.print_enhanced_info = self.pbf.io.print_enhanced_info

        # number of fields involved
        if self.pbs.incompressible_2field: self.nfields=7
        else: self.nfields=6

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbs.incompressible_2field:
            if self.pbf.num_dupl > 1: is_ghosted = [1, 1, 1, 2, 1, 0, 1]
            else:                     is_ghosted = [1, 1, 1, 1, 1, 0, 1]
            return [self.pbs.u.vector, self.pbs.p.vector, self.pbf.v.vector, self.pbf.p.vector, self.lm.vector, self.pbf0.LM, self.pba.d.vector], is_ghosted
        else:
            if self.pbf.num_dupl > 1: is_ghosted = [1, 1, 2, 1, 0, 1]
            else:                     is_ghosted = [1, 1, 1, 1, 0, 1]
            return [self.pbs.u.vector, self.pbf.v.vector, self.pbf.p.vector, self.lm.vector, self.pbf0.LM, self.pba.d.vector], is_ghosted


    def set_variational_forms(self):
        super().set_variational_forms()

        self.dcqd = []
        for n in range(self.pbf0.num_coupling_surf):
            self.dcqd.append(ufl.derivative(self.pbf0.cq[n], self.pba.d, self.pba.dd))


    def set_problem_residual_jacobian_forms(self):

        # solid, ALE-fluid, 3D-0D coupling
        self.pbs.set_problem_residual_jacobian_forms()
        self.pbfa0.set_problem_residual_jacobian_forms()

        ts = time.time()
        utilities.print_status("FEM form compilation for FSI coupling...", self.comm, e=" ")

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

        # solid, ALE-fluid, 3D-0D coupling
        self.pbs.set_problem_vector_matrix_structures()
        self.pbfa0.set_problem_vector_matrix_structures()

        self.r_l = fem.petsc.create_vector(self.res_l)

        self.K_ul = fem.petsc.create_matrix(self.jac_ul)
        self.K_vl = fem.petsc.create_matrix(self.jac_vl)

        self.K_lu = fem.petsc.create_matrix(self.jac_lu)
        self.K_lv = fem.petsc.create_matrix(self.jac_lv)

        self.K_ll = fem.petsc.create_matrix(self.jac_ll_dummy)


    def assemble_residual(self, t, subsolver=None):

        if self.pbs.incompressible_2field: off = 1
        else: off = 0

        self.pbfa0.evaluate_residual_dbc_coupling()

        self.pbs.assemble_residual(t)
        self.pbfa0.assemble_residual(t, subsolver=subsolver)

        # solid
        self.r_list[0] = self.pbs.r_list[0]
        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbs.r_list[1]

        # fluid
        self.r_list[1+off] = self.pbf.r_list[0]
        self.r_list[2+off] = self.pbf.r_list[1]

        # FSI coupling
        with self.r_l.localForm() as r_local: r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_l, self.res_l)
        fem.apply_lifting(self.r_l, [self.jac_ll], [self.bclm.dbcs], x0=[self.lm.vector], scale=-1.0)
        self.r_l.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.r_l, self.bclm.dbcs, x0=self.lm.vector, scale=-1.0)

        self.r_list[3+off] = self.r_l

        # flow0d
        self.r_list[4+off] = self.pbf0.r_list[2]

        # ALE
        self.r_list[5+off] = self.pba.r_list[0]


    def assemble_stiffness(self, t, subsolver=None):

        if self.pbs.incompressible_2field: off = 1
        else: off = 0

        self.pbs.assemble_stiffness(t)
        self.pbfa0.assemble_stiffness(t, subsolver=subsolver)

        # solid displacement
        self.K_list[0][0] = self.pbs.K_list[0][0]
        if self.pbs.incompressible_2field:
            self.K_list[0][1] = self.pbs.K_list[0][1]

        self.K_ul.zeroEntries()
        fem.petsc.assemble_matrix(self.K_ul, self.jac_ul, self.pbs.bc.dbcs)
        self.K_ul.assemble()
        self.K_list[0][3+off] = self.K_ul

        # solid pressure
        if self.pbs.incompressible_2field:
            self.K_list[1][0] = self.pbs.K_list[1][0]
            self.K_list[1][1] = self.pbs.K_list[1][1]

        # fluid velocity
        self.K_list[1+off][1+off] = self.pbf.K_list[0][0]
        self.K_list[1+off][2+off] = self.pbf.K_list[0][1]

        self.K_vl.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vl, self.jac_vl, self.pbf.bc.dbcs)
        self.K_vl.assemble()
        self.K_list[1+off][3+off] = self.K_vl

        self.K_list[1+off][4+off] = self.pbfa0.K_list[0][2]
        self.K_list[1+off][5+off] = self.pbfa0.K_list[0][3]

        # fluid pressure
        self.K_list[2+off][1+off] = self.pbf.K_list[1][0]
        self.K_list[2+off][2+off] = self.pbf.K_list[1][1]
        self.K_list[2+off][5+off] = self.pbfa0.K_list[1][3]

        # FSI LM
        self.K_lu.zeroEntries()
        fem.petsc.assemble_matrix(self.K_lu, self.jac_lu, self.bclm.dbcs)
        self.K_lu.assemble()
        self.K_list[3+off][0] = self.K_lu
        self.K_lv.zeroEntries()
        fem.petsc.assemble_matrix(self.K_lv, self.jac_lv, self.bclm.dbcs)
        self.K_lv.assemble()
        self.K_list[3+off][1+off] = self.K_lv
        self.K_ll.zeroEntries()
        fem.petsc.assemble_matrix(self.K_ll, self.jac_ll, self.bclm.dbcs)
        self.K_ll.assemble()
        self.K_list[3+off][3+off] = self.K_ll

        # 3D-0D LM
        self.K_list[4+off][1+off] = self.pbfa0.K_list[2][0]
        self.K_list[4+off][4+off] = self.pbfa0.K_list[2][2]
        self.K_list[4+off][5+off] = self.pbfa0.K_list[2][3]

        # ALE displacement
        self.K_list[5+off][5+off] = self.pbfa0.K_list[3][3]
        self.K_list[5+off][1+off] = self.pbfa0.K_list[3][0]


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # fluid-ALE + flow0d problem
        if N > 0:
            self.io.readcheckpoint(self, N)

        self.pb0.read_restart(sname, N)

        if self.pbase.restart_step > 0:
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.pbf0.LM)
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.pbf0.LM_old)


    def evaluate_initial(self):

        self.pbs.evaluate_initial()
        self.pbfa0.evaluate_initial()


    def write_output_ini(self):

        # self.io.write_output(self, writemesh=True)
        self.pbs.write_output_ini()
        self.pbfa0.write_output_ini()


    def write_output_pre(self):

        self.pbs.write_output_pre()
        self.pbfa0.write_output_pre()


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t, N, dt):

        self.pbs.evaluate_pre_solve(t, N, dt)
        self.pbfa0.evaluate_pre_solve(t, N, dt)


    def evaluate_post_solve(self, t, N):

        self.pbs.evaluate_post_solve(t, N)
        self.pbfa0.evaluate_post_solve(t, N)


    def set_output_state(self, N):

        self.pbs.set_output_state(N)
        self.pbfa0.set_output_state(N)


    def write_output(self, N, t, mesh=False):

        # self.io.write_output(self, N=N, t=t) # combined FSI output routine
        self.pbs.write_output(N, t)
        self.pbfa0.write_output(N, t)


    def update(self):

        # update time step - solid,fluid+flow0d and ALE
        self.pbs.update()
        self.pbfa0.update()


    def print_to_screen(self):

        self.pbs.print_to_screen()
        self.pbfa0.print_to_screen()


    def induce_state_change(self):

        self.pbs.induce_state_change()
        self.pbfa0.induce_state_change()


    def write_restart(self, sname, N, force=False):

        self.io.write_restart(self, N, force=force)
        self.pb0.write_restart(sname, N, force=force)

        if (self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0) or force:
            lm_sq = allgather_vec(self.pbf0.LM, self.comm)
            if self.comm.rank == 0:
                f = open(self.pb0.output_path_0D+'/checkpoint_'+sname+'_lm_'+str(N)+'.txt', 'wt')
                for i in range(len(lm_sq)):
                    f.write('%.16E\n' % (lm_sq[i]))
                f.close()
            del lm_sq


    def check_abort(self, t):

        return self.pbfa0.check_abort(t)


    def destroy(self):

        self.pbs.destroy()
        self.pbfa0.destroy()



class FSIFlow0DSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlin.solver_nonlinear_ode([self.pb.pb0], self.solver_params['subsolver_params'])
        else:
            self.subsol = None

        self.evaluate_assemble_system_initial(subsolver=self.subsol)

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params, subsolver=self.subsol)


    def solve_initial_state(self):

        # consider consistent initial acceleration of solid
        if self.pb.pbs.timint != 'static' and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial solid acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a_solid = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old + self.pb.work_coupling_solid_old

            weakform_lin_aa_solid = ufl.derivative(weakform_a_solid, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a_solid, jac_aa_solid = fem.form(weakform_a_solid, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa_solid, entity_maps=self.pb.io.entity_maps)
            self.solnln.solve_consistent_ini_acc(res_a_solid, jac_aa_solid, self.pb.pbs.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

        # consider consistent initial acceleration of fluid
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial fluid acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a_fluid = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old - self.pb.power_coupling_fluid_old - self.pb.pbf0.power_coupling_old

            weakform_lin_aa_fluid = ufl.derivative(weakform_a_fluid, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a_fluid, jac_aa_fluid = fem.form(weakform_a_fluid, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa_fluid, entity_maps=self.pb.io.entity_maps)
            self.solnln.solve_consistent_ini_acc(res_a_fluid, jac_aa_fluid, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbase.numstep, ni=ni, li=li, wt=wt)
