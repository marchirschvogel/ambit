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
from fluid import FluidmechanicsProblem

from base import solver_base


class FSIProblem():

    def __init__(self, io_params, time_params_solid, time_params_fluid, fem_params_solid, fem_params_fluid, constitutive_models_solid, constitutive_models_fluid, bc_dict_solid, bc_dict_fluid, time_curves, coupling_params, ios, iof, mor_params={}, comm=None):

        self.problem_physics = 'fsi'

        self.comm = comm

        self.coupling_params = coupling_params
        self.coupling_surface = self.coupling_params['surface_ids']

        self.ios, self.iof = ios, iof

        # assert that we do not have conflicting timings
        time_params_fluid['maxtime'] = time_params_solid['maxtime']
        time_params_fluid['numstep'] = time_params_solid['numstep']

        # initialize problem instances (also sets the variational forms for the solid and fluid problem)
        self.pbs = SolidmechanicsProblem(io_params, time_params_solid, fem_params_solid, constitutive_models_solid, bc_dict_solid, time_curves, ios, mor_params=mor_params, comm=self.comm)
        self.pbf = FluidmechanicsProblem(io_params, time_params_fluid, fem_params_fluid, constitutive_models_fluid, bc_dict_fluid, time_curves, ios, mor_params=mor_params, comm=self.comm)

        self.incompressible_2field = self.pbs.incompressible_2field

        self.set_variational_forms_and_jacobians()

        self.numdof = self.pbs.numdof + self.pbf.numdof
        # solid is 'master' problem - define problem variables based on its values
        self.simname = self.pbs.simname
        self.restart_step = self.pbs.restart_step
        self.numstep_stop = self.pbs.numstep_stop
        self.dt = self.pbs.dt

        print("made it through IO....")
        sys.exit()


    # defines the monolithic coupling forms for FSI
    def set_variational_forms_and_jacobians(self):

        # create FSI interface
        submsh_entities = fem.locate_dofs_topological(self.pbs.V_u, self.ios.mesh.topology.dim-1, self.ios.mt_b1.indices[self.ios.mt_b1.values == self.coupling_surface[0]])
        self.fsi_interface, entity_map = mesh.create_submesh(self.ios.mesh, self.ios.mesh.topology.dim, submsh_entities)[0:2]
        #submsh_entities = fem.locate_dofs_topological(self.pbf.V_v, self.iof.mesh.topology.dim-1, self.iof.mt_b1.indices[self.iof.mt_b1.values == self.coupling_surface[0]])
        #self.fsi_interface, entity_map = mesh.create_submesh(self.iof.mesh, self.iof.mesh.topology.dim, submsh_entities)[0:2]

        # cf. https://github.com/jpdean/mixed_domain_demos/blob/main/lagrange_multiplier.py

        facet_imap = self.ios.mesh.topology.index_map(self.ios.mesh.topology.dim-1)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        inv_entity_map = np.full(num_facets, -1)
        inv_entity_map[entity_map] = np.arange(len(entity_map))
        entity_maps = {self.fsi_interface: inv_entity_map}

        dx_fsi = ufl.dx(metadata={'quadrature_degree': self.pbs.quad_degree})

        # Lagrange multiplier function space
        P_lm = ufl.VectorElement("CG", self.fsi_interface.ufl_cell(), self.pbs.order_disp)
        self.V_lm = fem.FunctionSpace(self.fsi_interface, P_lm)

        # Lagrange multiplier
        self.LM = fem.Function(self.V_lm)
        self.LM_old = fem.Function(self.V_lm)

        self.dLM    = ufl.TrialFunction(self.V_lm) # incremental LM
        self.var_LM = ufl.TestFunction(self.V_lm)  # LM test function

        facet_integration_entities = {self.coupling_surface[0]: []}
        self.ios.mesh.topology.create_connectivity(self.ios.mesh.topology.dim, self.ios.mesh.topology.dim-1)
        self.ios.mesh.topology.create_connectivity(self.ios.mesh.topology.dim-1, self.ios.mesh.topology.dim)
        c_to_f = self.ios.mesh.topology.connectivity(self.ios.mesh.topology.dim, self.ios.mesh.topology.dim-1)
        f_to_c = self.ios.mesh.topology.connectivity(self.ios.mesh.topology.dim-1, self.ios.mesh.topology.dim)

        interface_facets_solid = self.ios.mt_b1.indices[self.ios.mt_b1.values == self.coupling_surface[0]]

        for facet in interface_facets_solid:
            # Check if this facet is owned
            if facet < facet_imap.size_local:
                # Get a cell connected to the facet
                cell = f_to_c.links(facet)[0]
                local_facet = c_to_f.links(cell).tolist().index(facet)
                facet_integration_entities[self.coupling_surface[0]].extend([cell, local_facet])
        ds_fsi = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=self.ios.mesh)

        # weak form of kinematic FSI coupling condition, v_solid = v_fluid
        self.res_LM = ufl.dot((self.pbs.vel - self.pbf.v), self.var_LM) * ds_fsi(self.coupling_surface[0])

        # TODO: failing...
        form = fem.form(self.res_LM, entity_maps=entity_maps)
        #r_lm = fem.petsc.assemble_vector(fem.form(self.res_LM)) # test...


    ### now the base routines for this problem

    def pre_timestep_routines(self):

        self.pbs.pre_timestep_routines()
        self.pbf.pre_timestep_routines()


    def read_restart(self, sname, N):

        # solid + flow0d problem
        self.pbs.read_restart(sname, N)
        self.pbf.read_restart(sname, N)

        if self.pbs.restart_step > 0:
            if self.coupling_type == 'monolithic_lagrange':
                self.pbf.cardvasc0D.read_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm)
                self.pbf.cardvasc0D.read_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm_old)


    def evaluate_initial(self):

        pass


    def write_output_ini(self):

        self.pbs.write_output_ini()
        self.pbf.write_output_ini()


    def get_time_offset(self):

        return (self.pbf.ti.cycle[0]-1) * self.pbf.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t):

        self.pbs.evaluate_pre_solve(t)
        self.pbf.evaluate_pre_solve(t)


    def evaluate_post_solve(self, t, N):

        self.pbs.evaluate_post_solve(t, N)
        self.pbf.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbs.set_output_state()
        self.pbf.set_output_state()


    def write_output(self, N, t, mesh=False):

        self.pbs.write_output(N, t)
        self.pbf.write_output(N, t)


    def update(self):

        # update time step - solid and 0D model
        self.pbs.update()
        self.pbf.update()

        # update Lagrange multiplier
        self.lm_old.axpby(1.0, 0.0, self.lm)


    def print_to_screen(self):

        self.pbs.print_to_screen()
        self.pbf.print_to_screen()


    def induce_state_change(self):

        self.pbs.induce_state_change()
        self.pbf.induce_state_change()


    def write_restart(self, sname, N):

        self.pbs.io.write_restart(self.pbs, N)
        self.pbf.io.write_restart(self.pbs, N)

        self.pb.write_restart(self.pbf.output_path_0D, sname+'_lm', N, self.lm)


    def check_abort(self, t):

        self.pbs.check_abort(t)
        self.pbf.check_abort(t)



class FSISolver(solver_base):

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):

        self.pb = problem

        self.solver_params_solid = solver_params_solid
        self.solver_params_flow0d = solver_params_flow0d

        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_constraint_monolithic(self.pb, self.pb.pbs.V_u, self.pb.pbs.V_p, self.solver_params_solid, self.solver_params_flow0d)

        if self.pb.pbs.prestress_initial:
            # add coupling work to prestress weak form
            self.pb.pbs.weakform_prestress_u -= self.pb.work_coupling_prestr
            # initialize solid mechanics solver
            self.solverprestr = SolidmechanicsSolver(self.pb.pbs, self.solver_params_solid)


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

            jac_a = ufl.derivative(weakform_a, self.pb.pbs.a_old, self.pb.pbs.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.pbs.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(self.pb.pbs.u, self.pb.pbs.p, self.pb.pbf.s, t, localdata=self.pb.pbs.localdata)


    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.sepstring, self.pb.pbs.numstep, wt=wt)
