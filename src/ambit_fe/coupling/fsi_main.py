#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np
from dolfinx import fem, mesh
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import ioparams, utilities
from .. import boundaryconditions

from ..solid.solid_main import SolidmechanicsProblem, SolidmechanicsSolverPrestr
from .fluid_ale_main import FluidmechanicsAleProblem

from ..base import problem_base, solver_base

"""
FSI problem class
"""

class FSIProblem(problem_base):

    def __init__(self, io_params, time_params_solid, time_params_fluid, fem_params_solid, fem_params_fluid, fem_params_ale, constitutive_models_solid, constitutive_models_fluid_ale, bc_dict_solid, bc_dict_fluid_ale, bc_dict_lm, time_curves, coupling_params, io, ios, iof, mor_params={}, comm=None):
        super().__init__(io_params, time_params_solid, comm=comm)

        self.problem_physics = 'fsi'

        self.coupling_params = coupling_params

        self.io = io
        self.ios, self.iof = ios, iof

        # assert that we do not have conflicting timings - TODO: Find better solution by moving these to global control parameters...
        assert(time_params_fluid['maxtime'] == time_params_solid['maxtime'])
        assert(time_params_fluid['numstep'] == time_params_solid['numstep'])

        # initialize problem instances (also sets the variational forms for the solid and fluid problem)
        self.pbs  = SolidmechanicsProblem(io_params, time_params_solid, fem_params_solid, constitutive_models_solid, bc_dict_solid, time_curves, ios, mor_params=mor_params, comm=self.comm)
        self.pbfa = FluidmechanicsAleProblem(io_params, time_params_fluid, fem_params_fluid, fem_params_ale, constitutive_models_fluid_ale[0], constitutive_models_fluid_ale[1], bc_dict_fluid_ale[0], bc_dict_fluid_ale[1], time_curves, coupling_params, iof, mor_params=mor_params, comm=self.comm)

        self.pbrom = self.pbs # ROM problem can only be solid so far...

        self.pbf = self.pbfa.pbf
        self.pba = self.pbfa.pba

        # modify results to write...
        self.pbs.results_to_write = io_params['results_to_write'][0]
        self.pbf.results_to_write = io_params['results_to_write'][1][0]
        self.pba.results_to_write = io_params['results_to_write'][1][1]

        self.incompressible_2field = self.pbs.incompressible_2field

        try: self.fsi_governing_type = self.coupling_params['fsi_governing_type']
        except: self.fsi_governing_type = 'solid_governed'

        P_lm = ufl.VectorElement("CG", self.io.msh_emap_lm[0].ufl_cell(), self.pbs.order_disp)
        self.V_lm = fem.FunctionSpace(self.io.msh_emap_lm[0], P_lm)

        # Lagrange multiplier
        self.LM = fem.Function(self.V_lm)
        self.LM_old = fem.Function(self.V_lm)

        self.dLM = ufl.TrialFunction(self.V_lm)    # incremental LM
        self.var_LM = ufl.TestFunction(self.V_lm)  # LM test function

        self.bclm = boundaryconditions.boundary_cond(self.io, dim=self.io.msh_emap_lm[0].topology.dim)
        # TODO: Application of DBCs to LM space not yet working, but might be needed when fluid and solid share mutual DBC nodes!
        #self.bclm.dirichlet_bcs(bc_dict_lm['dirichlet'], self.V_lm)

        self.set_variational_forms()

        self.numdof = self.pbs.numdof + self.pbfa.numdof

        self.localsolve = False
        self.sub_solve = False
        self.print_enhanced_info = self.io.print_enhanced_info

        # number of fields involved
        if self.pbs.incompressible_2field: self.nfields=6
        else: self.nfields=5

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)],  [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbs.incompressible_2field:
            if self.pbf.num_dupl > 1: is_ghosted = [1, 1, 1, 2, 1, 1]
            else:                     is_ghosted = [1, 1, 1, 1, 1, 1]
            return [self.pbs.u.vector, self.pbs.p.vector, self.pbf.v.vector, self.pbf.p.vector, self.LM.vector, self.pba.d.vector], is_ghosted
        else:
            if self.pbf.num_dupl > 1: is_ghosted = [1, 1, 2, 1, 1]
            else:                     is_ghosted = [1, 1, 1, 1, 1]
            return [self.pbs.u.vector, self.pbf.v.vector, self.pbf.p.vector, self.LM.vector, self.pba.d.vector], is_ghosted


    # defines the monolithic coupling forms for FSI
    def set_variational_forms(self):

        work_coupling_solid = ufl.dot(self.LM, self.pbs.var_u)*self.io.ds(self.io.interface_id_s)
        work_coupling_solid_old = ufl.dot(self.LM_old, self.pbs.var_u)*self.io.ds(self.io.interface_id_s)
        work_coupling_fluid = ufl.dot(self.LM, self.pbf.var_v)*self.io.ds(self.io.interface_id_f)
        work_coupling_fluid_old = ufl.dot(self.LM_old, self.pbf.var_v)*self.io.ds(self.io.interface_id_f)

        # add to solid and fluid virtual work/power
        self.pbs.weakform_u += self.pbs.timefac * work_coupling_solid + (1.-self.pbs.timefac) * work_coupling_solid_old
        self.pbf.weakform_v += -self.pbf.timefac * work_coupling_fluid - (1.-self.pbf.timefac) * work_coupling_fluid_old

        # add to solid and fluid Jacobian
        self.pbs.weakform_lin_uu += self.pbs.timefac * ufl.derivative(work_coupling_solid, self.pbs.u, self.pbs.du)
        self.pbf.weakform_lin_vv += -self.pbf.timefac * ufl.derivative(work_coupling_fluid, self.pbf.v, self.pbf.dv)

        if self.fsi_governing_type=='solid_governed':
            self.weakform_l = ufl.dot(self.pbs.u, self.var_LM)*self.io.ds(self.io.interface_id_s) - ufl.dot(self.pbf.ufluid, self.var_LM)*self.io.ds(self.io.interface_id_f)
        elif self.fsi_governing_type=='fluid_governed':
            self.weakform_l = ufl.dot(self.pbf.v, self.var_LM)*self.io.ds(self.io.interface_id_f) - ufl.dot(self.pbs.vel, self.var_LM)*self.io.ds(self.io.interface_id_s)
        else:
            raise ValueError("Unknown FSI governing type.")

        self.weakform_lin_lu = ufl.derivative(self.weakform_l, self.pbs.u, self.pbs.du)
        self.weakform_lin_lv = ufl.derivative(self.weakform_l, self.pbf.v, self.pbf.dv)

        self.weakform_lin_ul = self.pbs.timefac * ufl.derivative(work_coupling_solid, self.LM, self.dLM)
        self.weakform_lin_vl = -self.pbf.timefac * ufl.derivative(work_coupling_fluid, self.LM, self.dLM)

        # even though this is zero, we still want to explicitly form and create the matrix for DBC application
        self.weakform_lin_ll = ufl.derivative(self.weakform_l, self.LM, self.dLM)


    def set_problem_residual_jacobian_forms(self):

        # solid + ALE-fluid
        self.pbs.set_problem_residual_jacobian_forms()
        self.pbfa.set_problem_residual_jacobian_forms()

        ts = time.time()
        utilities.print_status("FEM form compilation for FSI coupling...", self.comm, e=" ")

        self.res_l = fem.form(self.weakform_l, entity_maps=self.io.entity_maps)
        self.jac_lu = fem.form(self.weakform_lin_lu, entity_maps=self.io.entity_maps)
        self.jac_lv = fem.form(self.weakform_lin_lv, entity_maps=self.io.entity_maps)

        self.jac_ul = fem.form(self.weakform_lin_ul, entity_maps=self.io.entity_maps)
        self.jac_vl = fem.form(self.weakform_lin_vl, entity_maps=self.io.entity_maps)

        # even though this is zero, we still want to explicitly form and create the matrix for DBC application
        self.jac_ll = fem.form(self.weakform_lin_ll, entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)


    def set_problem_vector_matrix_structures(self):

        # solid + ALE-fluid
        self.pbs.set_problem_vector_matrix_structures()
        self.pbfa.set_problem_vector_matrix_structures()

        self.r_l = fem.petsc.create_vector(self.res_l)

        self.K_ul = fem.petsc.create_matrix(self.jac_ul)
        self.K_vl = fem.petsc.create_matrix(self.jac_vl)

        self.K_lu = fem.petsc.create_matrix(self.jac_lu)
        self.K_lv = fem.petsc.create_matrix(self.jac_lv)

        self.K_ll = fem.petsc.create_matrix(self.jac_ll)


    def assemble_residual(self, t, subsolver=None):

        if self.pbs.incompressible_2field: off = 1
        else: off = 0

        self.pbs.assemble_residual(t)
        self.pbfa.assemble_residual(t)

        with self.r_l.localForm() as r_local: r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_l, self.res_l)
        self.r_l.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.r_list[0] = self.pbs.r_list[0]

        if self.pbs.incompressible_2field:
            self.r_list[1] = self.pbs.r_list[1]

        self.r_list[1+off] = self.pbfa.r_list[0]
        self.r_list[2+off] = self.pbfa.r_list[1]

        self.r_list[3+off] = self.r_l
        self.r_list[4+off] = self.pbfa.r_list[2]

        if bool(self.residual_scale):
            self.scale_residual_list([self.r_l], [self.residual_scale[3+off]])


    def assemble_stiffness(self, t, subsolver=None):

        if self.pbs.incompressible_2field: off = 1
        else: off = 0

        self.pbs.assemble_stiffness(t)
        self.pbfa.assemble_stiffness(t)

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
        self.K_list[1+off][1+off] = self.pbfa.K_list[0][0]
        self.K_list[1+off][2+off] = self.pbfa.K_list[0][1]

        self.K_vl.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vl, self.jac_vl, self.pbf.bc.dbcs)
        self.K_vl.assemble()
        self.K_list[1+off][3+off] = self.K_vl
        self.K_list[1+off][4+off] = self.pbfa.K_list[0][2]

        # fluid pressure
        self.K_list[2+off][1+off] = self.pbfa.K_list[1][0]
        self.K_list[2+off][2+off] = self.pbfa.K_list[1][1]
        self.K_list[2+off][4+off] = self.pbfa.K_list[1][2]

        # LM
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

        # ALE displacement
        self.K_list[4+off][4+off] = self.pbfa.K_list[2][2]

        if bool(self.residual_scale):
            self.K_ul.scale(self.residual_scale[0])
            self.K_lu.scale(self.residual_scale[3+off])
            self.K_vl.scale(self.residual_scale[1+off])
            self.K_lv.scale(self.residual_scale[3+off])


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # solid + fluid-ALE problem
        self.pbs.read_restart(sname, N)
        self.pbfa.read_restart(sname, N)

        if self.pbs.restart_step > 0:
            if self.coupling_type == 'monolithic_lagrange':
                self.pbf.cardvasc0D.read_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm)
                self.pbf.cardvasc0D.read_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm_old)


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


    def get_time_offset(self):

        return 0.


    def evaluate_pre_solve(self, t, N):

        self.pbs.evaluate_pre_solve(t, N)
        self.pbfa.evaluate_pre_solve(t, N)


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
        self.LM_old.vector.axpby(1.0, 0.0, self.LM.vector)


    def print_to_screen(self):

        self.pbs.print_to_screen()
        self.pbfa.print_to_screen()


    def induce_state_change(self):

        self.pbs.induce_state_change()
        self.pbfa.induce_state_change()


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)

        # self.write_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm)


    def check_abort(self, t):

        return False


    def destroy(self):
        pass



class FSISolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)

        if (self.pb.pbs.prestress_initial or self.pb.pbs.prestress_initial_only) and self.pb.pbs.restart_step == 0:
            # initialize fluid mechanics solver
            self.solverprestr = SolidmechanicsSolverPrestr(self.pb.pbs, self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration of solid
        if self.pb.pbs.timint != 'static' and self.pb.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial solid acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a_solid = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old

            weakform_lin_aa_solid = ufl.derivative(weakform_a_solid, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a_solid, jac_aa_solid = fem.form(weakform_a_solid, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa_solid, entity_maps=self.pb.io.entity_maps)
            else:
                res_a_solid, jac_aa_solid = fem.form(weakform_a_solid), fem.form(weakform_lin_aa_solid)
            self.solnln.solve_consistent_ini_acc(res_a_solid, jac_aa_solid, self.pb.pbs.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

        # consider consistent initial acceleration of fluid
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial fluid acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a_fluid = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            weakform_lin_aa_fluid = ufl.derivative(weakform_a_fluid, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a_fluid, jac_aa_fluid = fem.form(weakform_a_fluid, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa_fluid, entity_maps=self.pb.io.entity_maps)
            else:
                res_a_fluid, jac_aa_fluid = fem.form(weakform_a_fluid), fem.form(weakform_lin_aa_fluid)
            self.solnln.solve_consistent_ini_acc(res_a_fluid, jac_aa_fluid, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t, localdata=self.pb.pbs.localdata)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
