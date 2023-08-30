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
import ..ioparams

from ..solid import SolidmechanicsProblem, SolidmechanicsSolverPrestr
from ..fluid_ale import FluidmechanicsAleProblem

from ..base import problem_base, solver_base


class FSIProblem(problem_base):

    def __init__(self, io_params, time_params_solid, time_params_fluid, fem_params_solid, fem_params_fluid, constitutive_models_solid, constitutive_models_fluid_ale, bc_dict_solid, bc_dict_fluid_ale, time_curves, coupling_params, io, ios, iof, mor_params={}, comm=None):
        super().__init__(io_params, time_params_solid, comm)

        self.problem_physics = 'fsi'

        self.coupling_params = coupling_params
        # self.coupling_surface = self.coupling_params['coupling_fluid_ale']['surface_ids']

        self.io = io
        self.ios, self.iof = ios, iof

        # assert that we do not have conflicting timings
        time_params_fluid['maxtime'] = time_params_solid['maxtime']
        time_params_fluid['numstep'] = time_params_solid['numstep']

        # initialize problem instances (also sets the variational forms for the solid and fluid problem)
        self.pbs  = SolidmechanicsProblem(io_params, time_params_solid, fem_params_solid, constitutive_models_solid, bc_dict_solid, time_curves, ios, mor_params=mor_params, comm=self.comm)
        self.pbfa = FluidmechanicsAleProblem(io_params, time_params_fluid, fem_params_fluid, constitutive_models_fluid_ale[0], constitutive_models_fluid_ale[1], bc_dict_fluid_ale[0], bc_dict_fluid_ale[1], time_curves, coupling_params, iof, mor_params=mor_params, comm=self.comm)

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

        self.set_variational_forms()

        self.numdof = self.pbs.numdof + self.pbfa.numdof

        self.localsolve = False
        self.sub_solve = False
        self.print_enhanced_info = self.io.print_enhanced_info


    def get_problem_var_list(self):

        if self.pbs.incompressible_2field:
            is_ghosted = [1, 1, 1, 1, 1, 1]
            return [self.pbs.u.vector, self.pbs.p.vector, self.pbf.v.vector, self.pbf.p.vector, self.LM.vector, self.pba.d.vector], is_ghosted
        else:
            is_ghosted = [1, 1, 1, 1, 1]
            return [self.pbs.u.vector, self.pbf.v.vector, self.pbf.p.vector, self.LM.vector, self.pba.d.vector], is_ghosted


    # defines the monolithic coupling forms for FSI
    def set_variational_forms(self):

        P_lm = ufl.VectorElement("CG", self.io.msh_emap_lm[0].ufl_cell(), self.pbs.order_disp)
        self.V_lm = fem.FunctionSpace(self.io.msh_emap_lm[0], P_lm)
        # self.V_lm = fem.VectorFunctionSpace(self.io.msh_emap_lm[0], ("CG", self.pbs.order_disp))

        # Lagrange multiplier
        self.LM = fem.Function(self.V_lm)
        self.LM_old = fem.Function(self.V_lm)

        self.dLM = ufl.TrialFunction(self.V_lm)    # incremental LM
        self.var_LM = ufl.TestFunction(self.V_lm)  # LM test function

        dssolid = ufl.dS(domain=self.io.mesh, subdomain_data=self.io.mt_b1, subdomain_id=self.io.surf_interf[0], metadata={'quadrature_degree': self.pbs.quad_degree})
        dsfluid = ufl.dS(domain=self.pbf.io.mesh, subdomain_data=self.pbf.io.mt_b1, subdomain_id=self.io.surf_interf[0], metadata={'quadrature_degree': self.pbf.quad_degree})

        interface_facets = self.io.mt_b1.indices[self.io.mt_b1.values == self.io.surf_interf[0]]
        solid_cells = self.io.mt_d.indices[self.io.mt_d.values == self.io.dom_solid[0]]

        tdim=3
        fdim = tdim - 1
        integration_entities_s = []
        integration_entities_f = []
        self.io.mesh.topology.create_connectivity(tdim, fdim)
        self.io.mesh.topology.create_connectivity(fdim, tdim)
        c_to_f = self.io.mesh.topology.connectivity(tdim, fdim)
        f_to_c = self.io.mesh.topology.connectivity(fdim, tdim)
        facet_imap = self.io.mesh.topology.index_map(fdim)
        # Loop over facets on interface
        for facet in interface_facets:
            # check if this facet is owned
            if facet < facet_imap.size_local:
                # get cells connected to the facet
                cells = f_to_c.links(facet)
                local_facets = [c_to_f.links(cells[0]).tolist().index(facet),
                                c_to_f.links(cells[1]).tolist().index(facet)]

                # add (cell, local_facet_index) pairs to correct side
                if cells[0] in solid_cells:
                    integration_entities_s.extend((cells[0], local_facets[0]))
                    integration_entities_f.extend((cells[1], local_facets[1]))
                else:
                    integration_entities_s.extend((cells[1], local_facets[1]))
                    integration_entities_f.extend((cells[0], local_facets[0]))

        # create a measure, passing the data we just created so we can integrate
        # over the correct entities
        interface_id_s = self.io.dom_solid[0]
        interface_id_f = self.io.dom_fluid[0]
        integration_entities = [(interface_id_s, integration_entities_s),
                                (interface_id_f, integration_entities_f)]

        dS_fsi = ufl.Measure("ds", subdomain_data=integration_entities, domain=self.io.mesh)

        # work_coupling_solid = self.pbs.vf.deltaW_ext_neumann_cur(self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), self.LM, db_s_)
        # work_coupling_solid = self.pbs.vf.deltaW_ext_neumann_ref(self.LM, ds_fsi)
        # work_coupling_solid_old = self.pbs.vf.deltaW_ext_neumann_cur(self.pbs.ki.J(self.pbs.u_old), self.pbs.ki.F(self.pbs.u_old), self.LM_old, ds_fsi)
        #
        # # work_coupling_fluid = self.pbf.vf.deltaW_ext_neumann_cur(self.LM, db_f_, Fale=self.pba.ki.F(self.pba.d))
        # work_coupling_fluid = self.pbf.vf.deltaW_ext_neumann_ref(self.LM, ds_fsi)
        # work_coupling_fluid_old = self.pbf.vf.deltaW_ext_neumann_cur(self.LM_old, ds_fsi, Fale=self.pba.ki.F(self.pba.d_old))

        work_coupling_solid = (ufl.dot(self.LM, self.pbs.var_u))('+')*dssolid# dS_fsi(interface_id_s)
        work_coupling_fluid = (ufl.dot(self.LM, self.pbf.var_v))('+')*dsfluid

        # dbtest = ufl.ds(domain=self.pbs.io.mesh_master, metadata={'quadrature_degree': self.pbs.quad_degree})
        # # (ufl.dot(self.pbs.u, self.pbs.var_u)) * dbtest
        # testform = (ufl.dot(self.pbs.u, self.pbs.var_u)) * self.pbs.dx_[0] - (ufl.dot(self.LM, self.pbs.var_u)) * dS_fsi(interface_id_s)
        # gg = fem.form(testform, entity_maps=self.io.entity_maps)
        # sys.exit()

        # add to solid and fluid virtual work/power
        self.pbs.weakform_u += self.pbs.timefac * work_coupling_solid #+ (1.-self.pbs.timefac) * work_coupling_solid_old
        self.pbf.weakform_v += self.pbf.timefac * work_coupling_fluid #+ (1.-self.pbf.timefac) * work_coupling_fluid_old

        # add to solid and fluid Jacobian
        self.pbs.weakform_lin_uu += self.pbs.timefac * ufl.derivative(work_coupling_solid, self.pbs.u, self.pbs.du)
        self.pbf.weakform_lin_vv += self.pbf.timefac * ufl.derivative(work_coupling_fluid, self.pbf.v, self.pbf.dv)

        self.pbfa.weakform_lin_vd += self.pbf.timefac * ufl.derivative(work_coupling_fluid, self.pba.d, self.pba.dd)

        # now the LM problem
        # db_x_fsi_ = ufl.dx(domain=self.io.msh_emap_lm[0], metadata={'quadrature_degree': self.pbs.quad_degree})

        if self.fsi_governing_type=='solid_governed':
            self.weakform_l = (ufl.dot((self.pbs.u - self.pbf.ufluid), self.var_LM))*dS_fsi(interface_id_s)#db_x_fsi_
            # self.weakform_l = (ufl.dot((self.pbs.vel - self.pbf.v), self.var_LM))*db_s_#db_x_fsi_
        elif self.fsi_governing_type=='fluid_governed':
            self.weakform_l = (ufl.dot((self.pbf.v - self.pbs.vel), self.var_LM))('+')*dS_fsi#db_x_fsi_
        else:
            raise ValueError("Unknown FSI governing type.")

        self.weakform_lin_lu = ufl.derivative(self.weakform_l, self.pbs.u, self.pbs.du)
        self.weakform_lin_lv = ufl.derivative(self.weakform_l, self.pbf.v, self.pbf.dv)

        self.weakform_lin_ul = ufl.derivative(self.pbs.weakform_u, self.LM, self.dLM)
        self.weakform_lin_vl = ufl.derivative(self.pbf.weakform_v, self.LM, self.dLM)


    def set_problem_residual_jacobian_forms(self):

        # solid + ALE-fluid
        self.pbs.set_problem_residual_jacobian_forms()
        self.pbfa.set_problem_residual_jacobian_forms()

        tes = time.time()
        if self.comm.rank == 0:
            print('FEM form compilation for FSI coupling...')
            sys.stdout.flush()

        self.res_l = fem.form(self.weakform_l, entity_maps=self.io.entity_maps)

        self.jac_lu = fem.form(self.weakform_lin_lu, entity_maps=self.io.entity_maps)
        self.jac_lv = fem.form(self.weakform_lin_lv, entity_maps=self.io.entity_maps)

        self.jac_ul = fem.form(self.weakform_lin_ul, entity_maps=self.io.entity_maps)
        self.jac_vl = fem.form(self.weakform_lin_vl, entity_maps=self.io.entity_maps)

        tee = time.time() - tes
        if self.comm.rank == 0:
            print('FEM form compilation for FSI coupling finished, te = %.2f s' % (tee))
            sys.stdout.flush()


    def set_problem_vector_matrix_structures():

        # solid + ALE-fluid
        self.pbs.set_problem_vector_matrix_structures()
        self.pbfa.set_problem_vector_matrix_structures()


    def assemble_residual(self, t, subsolver=None):

        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        r_list = [None]*(5+off)

        r_list_solid = self.pbs.assemble_residual(t)
        r_list_fluid_ale = self.pbfa.assemble_residual(t)

        r_list[0] = r_list_solid[0]
        if self.pbs.incompressible_2field:
            r_list[1] = r_list_solid[1]
        r_list[1+off] = r_list_fluid_ale[0]
        r_list[2+off] = r_list_fluid_ale[1]
        r_l = fem.petsc.assemble_vector(self.res_l)
        r_l.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        r_list[3+off] = r_l
        r_list[4+off] = r_list_fluid_ale[2]

        return r_list


    def assemble_stiffness(self, t, subsolver=None):

        if self.pbs.incompressible_2field:
            off = 1
        else:
            off = 0

        K_list = [[None]*(5+off) for _ in range(5+off)]

        K_list_solid = self.pbs.assemble_stiffness(t)
        K_list_fluid_ale = self.pbfa.assemble_stiffness(t)

        # solid displacement
        K_list[0][0] = K_list_solid[0][0]
        if self.pbs.incompressible_2field:
            K_list[0][1] = K_list_solid[0][1]
        K_ul = fem.petsc.assemble_matrix(self.jac_ul, self.pbs.bc.dbcs)
        K_ul.assemble()
        K_list[0][3+off] = K_ul

        # solid pressure
        if self.pbs.incompressible_2field:
            K_list[1][0] = K_list_solid[1][0]
            K_list[1][1] = K_list_solid[1][1]

        # fluid velocity
        K_list[1+off][1+off] = K_list_fluid_ale[0][0]
        K_list[1+off][2+off] = K_list_fluid_ale[0][1]
        K_vl = fem.petsc.assemble_matrix(self.jac_vl, self.pbf.bc.dbcs)
        K_vl.assemble()
        K_list[1+off][3+off] = K_vl
        K_list[1+off][4+off] = K_list_fluid_ale[0][2]

        # fluid pressure
        K_list[2+off][1+off] = K_list_fluid_ale[1][0]
        K_list[2+off][2+off] = K_list_fluid_ale[1][1]
        K_list[2+off][4+off] = K_list_fluid_ale[1][2]

        # LM
        K_lu = fem.petsc.assemble_matrix(self.jac_lu, [])
        K_lu.assemble()
        K_list[3+off][0] = K_lu
        K_lv = fem.petsc.assemble_matrix(self.jac_lv, [])
        K_lv.assemble()
        K_list[3+off][1+off] = K_lv

        # ALE displacement
        K_list[4+off][4+off] = K_list_fluid_ale[2][2]

        return K_list


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

        self.pbs.write_output_ini()
        self.pbfa.write_output_ini()


    def get_time_offset(self):
        return 0.
        # return (self.pbf.ti.cycle[0]-1) * self.pbf.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


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

        self.pbs.check_abort(t)
        self.pbfa.check_abort(t)


    def destroy(self):

        super().destroy()



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
            # weak form at initial state for consistent initial acceleration solve
            weakform_a_solid = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old

            weakform_lin_aa_solid = ufl.derivative(weakform_a_solid, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a_solid, jac_aa_solid = fem.form(weakform_a_solid, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa_solid, entity_maps=self.pb.io.entity_maps)
            else:
                res_a_solid, jac_aa_solid = fem.form(weakform_a_solid), fem.form(weakform_lin_aa_solid)
            self.solnln.solve_consistent_ini_acc(res_a_solid, jac_aa_solid, self.pb.pbs.a_old)

        # consider consistent initial acceleration of fluid
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a_fluid = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            weakform_lin_aa_fluid = ufl.derivative(weakform_a_fluid, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a_fluid, jac_aa_fluid = fem.form(weakform_a_fluid, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa_fluid, entity_maps=self.pb.io.entity_maps)
            else:
                res_a_fluid, jac_aa_fluid = fem.form(weakform_a_fluid), fem.form(weakform_lin_aa_fluid)
            self.solnln.solve_consistent_ini_acc(res_a_fluid, jac_aa_fluid, self.pb.pbf.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t, localdata=self.pb.pbs.localdata)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
