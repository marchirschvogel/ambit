#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np
from dolfinx import fem, mesh
import ufl
from petsc4py import PETSc

import utilities
import solver_nonlin
import expression
from projection import project
from mpiroutines import allgather_vec

from solid import SolidmechanicsProblem, SolidmechanicsSolver
from fluid_ale import FluidmechanicsAleProblem

from base import solver_base


class FSIProblem():

    def __init__(self, io_params, time_params_solid, time_params_fluid, fem_params_solid, fem_params_fluid, constitutive_models_solid, constitutive_models_fluid_ale, bc_dict_solid, bc_dict_fluid_ale, time_curves, coupling_params, ios, iof, mor_params={}, comm=None):

        self.problem_physics = 'fsi'

        self.comm = comm

        self.coupling_params = coupling_params
        self.coupling_surface = self.coupling_params['coupling_fluid_ale']['surface_ids']

        self.ios, self.iof = ios, iof

        # assert that we do not have conflicting timings
        time_params_fluid['maxtime'] = time_params_solid['maxtime']
        time_params_fluid['numstep'] = time_params_solid['numstep']

        # initialize problem instances (also sets the variational forms for the solid and fluid problem)
        self.pbs  = SolidmechanicsProblem(io_params['io_solid'], time_params_solid, fem_params_solid, constitutive_models_solid, bc_dict_solid, time_curves, ios, mor_params=mor_params, comm=self.comm)
        self.pbfa = FluidmechanicsAleProblem(io_params['io_fluid'], time_params_fluid, fem_params_fluid, constitutive_models_fluid_ale[0], constitutive_models_fluid_ale[1], bc_dict_fluid_ale[0], bc_dict_fluid_ale[1], time_curves, coupling_params, iof, mor_params=mor_params, comm=self.comm)

        try: self.fluid_solid_interface = self.coupling_params['fluid_solid_interface']
        except: self.fluid_solid_interface = 'solid_governed'

        self.set_variational_forms()

        self.numdof = self.pbs.numdof + self.pbfa.numdof
        # solid is 'master' problem - define problem variables based on its values
        self.simname = self.pbs.simname
        self.restart_step = self.pbs.restart_step
        self.numstep_stop = self.pbs.numstep_stop
        self.dt = self.pbs.dt
        self.have_rom = self.pbs.have_rom
        if self.have_rom: self.rom = self.pbs.rom


    def get_problem_var_list(self):

        if self.pbs.incompressible_2field:
            is_ghosted = [1, 1, 1, 1, 1, 1]
            return [self.pbs.u.vector, self.pbs.p.vector, self.pbfa.pbf.v.vector, self.pbfa.pbf.p.vector, self.pbfa.pba.d.vector, self.LMs], is_ghosted
        else:
            is_ghosted = [1, 1, 1, 1, 1]
            return [self.pbs.u.vector, self.pbfa.pbf.v.vector, self.pbfa.pbf.p.vector, self.pbfa.pba.d.vector, self.LMs], is_ghosted


    # defines the monolithic coupling forms for FSI
    def set_variational_forms(self):

        self.pbs.set_variational_forms()
        self.pbfa.set_variational_forms()

        # solid-sided interface
        # submsh_entities_solid = fem.locate_dofs_topological(self.pbs.V_u, self.ios.mesh.topology.dim-1, self.ios.mt_b1.indices[self.ios.mt_b1.values == self.coupling_surface[0]])
        submsh_entities_solid = self.ios.mt_b1.indices[self.ios.mt_b1.values == self.coupling_surface[0]]
        self.fsi_interface_solid, entity_map_solid, vertex_map_solid, geom_map_solid = mesh.create_submesh(self.ios.mesh, self.ios.mesh.topology.dim-1, submsh_entities_solid)#[0:2]

        facet_imap_solid = self.ios.mesh.topology.index_map(self.ios.mesh.topology.dim-1)

        facet_integration_entities_solid = []
        self.ios.mesh.topology.create_connectivity(self.ios.mesh.topology.dim, self.ios.mesh.topology.dim-1)
        self.ios.mesh.topology.create_connectivity(self.ios.mesh.topology.dim-1, self.ios.mesh.topology.dim)
        c_to_f_solid = self.ios.mesh.topology.connectivity(self.ios.mesh.topology.dim, self.ios.mesh.topology.dim-1)
        f_to_c_solid = self.ios.mesh.topology.connectivity(self.ios.mesh.topology.dim-1, self.ios.mesh.topology.dim)

        interface_facets_solid = self.ios.mt_b1.indices[self.ios.mt_b1.values == self.coupling_surface[0]]

        num_facets_solid = facet_imap_solid.size_local + facet_imap_solid.num_ghosts
        inv_entity_map_solid = np.full(num_facets_solid, -1)
        inv_entity_map_solid[entity_map_solid] = np.arange(len(entity_map_solid))
        self.entity_maps_solid = {self.fsi_interface_solid: inv_entity_map_solid}

        print(self.comm.rank,self.entity_maps_solid)
        sys.exit()

        mshdomain_solid = self.ios.mesh

        for facet in interface_facets_solid:
            # Check if this facet is owned
            if facet < facet_imap_solid.size_local:
                # Get a cell connected to the facet
                cell = f_to_c_solid.links(facet)[0]
                local_facet = c_to_f_solid.links(cell).tolist().index(facet)
                facet_integration_entities_solid.extend([cell, local_facet])

        ds_fsi_solid = ufl.Measure("ds", subdomain_data=[(self.coupling_surface[0], facet_integration_entities_solid)], domain=mshdomain_solid)
        sys.exit()
        # fluid-sided interface
        # submsh_entities_fluid = fem.locate_dofs_topological(self.pbfa.pbf.V_v, self.iof.mesh.topology.dim-1, self.iof.mt_b1.indices[self.iof.mt_b1.values == self.coupling_surface[0]])
        submsh_entities_fluid = self.iof.mt_b1.indices[self.iof.mt_b1.values == self.coupling_surface[0]]
        self.fsi_interface_fluid, entity_map_fluid, vertex_map_fluid, geom_map_fluid = mesh.create_submesh(self.iof.mesh, self.iof.mesh.topology.dim-1, submsh_entities_fluid)#[0:2]

        facet_imap_fluid = self.iof.mesh.topology.index_map(self.iof.mesh.topology.dim-1)

        facet_integration_entities_fluid = []
        self.iof.mesh.topology.create_connectivity(self.iof.mesh.topology.dim, self.iof.mesh.topology.dim-1)
        self.iof.mesh.topology.create_connectivity(self.iof.mesh.topology.dim-1, self.iof.mesh.topology.dim)
        c_to_f_fluid = self.iof.mesh.topology.connectivity(self.iof.mesh.topology.dim, self.iof.mesh.topology.dim-1)
        f_to_c_fluid = self.iof.mesh.topology.connectivity(self.iof.mesh.topology.dim-1, self.iof.mesh.topology.dim)

        interface_facets_fluid = self.iof.mt_b1.indices[self.iof.mt_b1.values == self.coupling_surface[0]]

        num_facets_fluid = facet_imap_fluid.size_local + facet_imap_fluid.num_ghosts
        inv_entity_map_fluid = np.full(num_facets_fluid, -1)
        inv_entity_map_fluid[entity_map_fluid] = np.arange(len(entity_map_fluid))
        self.entity_maps_fluid = {self.fsi_interface_fluid: inv_entity_map_fluid}

        mshdomain_fluid = self.iof.mesh

        for facet in interface_facets_fluid:
            # Check if this facet is owned
            if facet < facet_imap_fluid.size_local:
                # Get a cell connected to the facet
                cell = f_to_c_fluid.links(facet)[0]
                local_facet = c_to_f_fluid.links(cell).tolist().index(facet)
                facet_integration_entities_fluid.extend([cell, local_facet])

        ds_fsi_fluid = ufl.Measure("ds", subdomain_data=[(self.coupling_surface[0], facet_integration_entities_fluid)], domain=mshdomain_fluid)

        P_lm_solid = ufl.VectorElement("CG", self.fsi_interface_solid.ufl_cell(), self.pbs.order_disp)
        self.V_lm_solid = fem.FunctionSpace(self.fsi_interface_solid, P_lm_solid)

        P_lm_fluid = ufl.VectorElement("CG", self.fsi_interface_fluid.ufl_cell(), self.pbfa.pbf.order_vel)
        self.V_lm_fluid = fem.FunctionSpace(self.fsi_interface_fluid, P_lm_fluid)

        # cf. https://github.com/jpdean/mixed_domain_demos/blob/main/lagrange_multiplier.py

        # Lagrange multiplier
        self.LMs = fem.Function(self.V_lm_solid)
        self.LMs_old = fem.Function(self.V_lm_solid)

        self.LMf = fem.Function(self.V_lm_fluid)
        self.LMf_old = fem.Function(self.V_lm_fluid)

        if self.fluid_solid_interface=='solid_governed':
            self.dLM = ufl.TrialFunction(self.V_lm_solid) # incremental LM
            self.var_LM = ufl.TestFunction(self.V_lm_solid)  # LM test function
            dx_fsi = ufl.dx(metadata={'quadrature_degree': self.pbs.quad_degree})
        elif self.fluid_solid_interface=='fluid_governed':
            self.dLM = ufl.TrialFunction(self.V_lm_fluid) # incremental LM
            self.var_LM = ufl.TestFunction(self.V_lm_fluid)  # LM test function
            dx_fsi = ufl.dx(metadata={'quadrature_degree': self.pbfa.pbf.quad_degree})
        else:
            raise ValueError("Unknown fluid_solid_interface setting! Choose 'solid_governed' or 'fluid_governed'.")

        work_coupling_solid = self.pbs.vf.deltaW_ext_neumann_cur(self.pbs.ki.J(self.pbs.u), self.pbs.ki.F(self.pbs.u), self.LMs, ds_fsi_solid)
        work_coupling_fluid = self.pbfa.pbf.vf.deltaW_ext_neumann_cur(self.LMf, ds_fsi_fluid, Fale=self.pbfa.pba.ki.F(self.pbfa.pba.d))

        # add to solid and fluid virtual work/power
        self.pbs.weakform_u += work_coupling_solid
        # self.pbfa.pbf.weakform_v += work_coupling_fluid

        if self.fluid_solid_interface=='solid_governed':
            self.res_LM = ufl.dot((self.pbs.u - self.pbfa.pba.d), self.var_LM) * ds_fsi_solid(self.coupling_surface[0])

        if self.fluid_solid_interface=='fluid_governed':
            self.res_LM = ufl.dot((self.pbs.vel - self.pbfa.pbf.v), self.var_LM) * ds_fsi_fluid(self.coupling_surface[0])



    def set_problem_residual_jacobian_forms(self):

        self.pbs.set_problem_residual_jacobian_forms() # TODO: Seems that we need to pass entitiy maps here for form creation
        self.pbfa.set_problem_residual_jacobian_forms() # TODO: Seems that we need to pass entitiy maps here for form creation

        if self.fluid_solid_interface=='solid_governed':
            ff = fem.form(self.res_LM, entity_maps=self.entity_maps_solid)

        if self.fluid_solid_interface=='fluid_governed':
            ff = fem.form(self.res_LM, entity_maps=self.entity_maps_fluid)

        # for testing purposes...
        tmp1, tmp2 = self.assemble_residual_stiffness(0.)


    def assemble_residual_stiffness(self, t, subsolver=None):

        r_list_solid, K_list_solid = self.pbs.assemble_residual_stiffness(t)
        r_list_fluid_ale, K_list_fluid_ale = self.pbfa.assemble_residual_stiffness(t)

        sys.exit()

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

        return (self.pbf.ti.cycle[0]-1) * self.pbf.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t):

        self.pbs.evaluate_pre_solve(t)
        self.pbfa.evaluate_pre_solve(t)


    def evaluate_post_solve(self, t, N):

        self.pbs.evaluate_post_solve(t, N)
        self.pbfa.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbs.set_output_state()
        self.pbfa.set_output_state()


    def write_output(self, N, t, mesh=False):

        self.pbs.write_output(N, t)
        self.pbfa.write_output(N, t)


    def update(self):

        # update time step - solid and 0D model
        self.pbs.update()
        self.pbfa.update()

        # update Lagrange multiplier
        self.lm_old.axpby(1.0, 0.0, self.lm)


    def print_to_screen(self):

        self.pbs.print_to_screen()
        self.pbfa.print_to_screen()


    def induce_state_change(self):

        self.pbs.induce_state_change()
        self.pbfa.induce_state_change()


    def write_restart(self, sname, N):

        self.pbs.io.write_restart(self.pbs, N)
        self.pbfa.io.write_restart(self.pbs, N)

        self.pb.write_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm)


    def check_abort(self, t):

        self.pbs.check_abort(t)
        self.pbfa.check_abort(t)



class FSISolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        sys.exit()

        # perform Proper Orthogonal Decomposition
        if self.pb.have_rom:
            self.pb.rom.prepare_rob()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], solver_params=self.solver_params)

        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            # initialize fluid mechanics solver
            self.solverprestr = SolidmechanicsSolver(self.pb.pbs, self.solver_params)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
        if self.pb.pbs.prestress_initial and self.pb.pbs.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.ksp.destroy()
        else:
            # set flag definitely to False if we're restarting
            self.pb.pbs.prestress_initial = False

        # consider consistent initial acceleration
        if self.pb.pbs.timint != 'static' and self.pb.pbs.restart_step == 0 and not self.pb.restart_multiscale:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbs.deltaW_kin_old + self.pb.pbs.deltaW_int_old - self.pb.pbs.deltaW_ext_old - self.pb.work_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbs.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t, localdata=self.pb.pbs.localdata)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.sepstring, ni=ni, li=li, wt=wt)
