#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, math
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import utilities
import solver_nonlin
import expression

from fluid import FluidmechanicsProblem, FluidmechanicsSolver
from ale import AleProblem
from base import solver_base
from meshutils import gather_surface_dof_indices
from projection import project


class FluidmechanicsAleProblem():

    def __init__(self, io_params, time_params, fem_params, constitutive_models_fluid, constitutive_models_ale, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params, io, mor_params={}, comm=None):

        self.problem_physics = 'fluid_ale'

        self.comm = comm

        self.coupling_params = coupling_params

        try: self.coupling_fluid_ale = self.coupling_params['coupling_fluid_ale']
        except: self.coupling_fluid_ale = {}

        try: self.coupling_ale_fluid = self.coupling_params['coupling_ale_fluid']
        except: self.coupling_ale_fluid = {}

        try: self.fluid_on_deformed = self.coupling_params['fluid_on_deformed']
        except: self.fluid_on_deformed = 'consistent'

        # initialize problem instances (also sets the variational forms for the fluid and ALE problem)
        self.pba = AleProblem(io_params, time_params, fem_params, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params, comm=self.comm)
        # ALE variables that are handed to fluid problem
        alevariables = {'Fale' : self.pba.ki.F(self.pba.d), 'Fale_old' : self.pba.ki.F(self.pba.d_old), 'w' : self.pba.wel, 'w_old' : self.pba.w_old, 'fluid_on_deformed' : self.fluid_on_deformed}
        self.pbf = FluidmechanicsProblem(io_params, time_params, fem_params, constitutive_models_fluid, bc_dict_fluid, time_curves, io, mor_params=mor_params, comm=self.comm, alevar=alevariables)

        # modify results to write...
        self.pbf.results_to_write = io_params['results_to_write'][0]
        self.pba.results_to_write = io_params['results_to_write'][1]

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.have_rom = False

        # NOTE: Fluid and ALE function spaces should be of the same type, but are different objects.
        # For some reason, when applying a function from one funtion space as DBC to another function space,
        # errors occur. Therefore, we define these auxiliary variables and interpolate respectively...

        # fluid displacement, but defined within ALE function space
        self.ufa = fem.Function(self.pba.V_d)
        # ALE velocity, but defined within fluid function space
        self.wf = fem.Function(self.pbf.V_v)

        self.set_variational_forms()

        self.numdof = self.pbf.numdof + self.pba.numdof
        # fluid is 'master' problem - define problem variables based on its values
        self.simname = self.pbf.simname
        self.restart_step = self.pbf.restart_step
        self.numstep_stop = self.pbf.numstep_stop
        self.dt = self.pbf.dt
        self.have_rom = self.pbf.have_rom
        if self.have_rom: self.rom = self.pbf.rom

        self.sub_solve = False


    def get_problem_var_list(self):

        is_ghosted = [True]*3
        return [self.pbf.v.vector, self.pbf.p.vector, self.pba.d.vector], is_ghosted


    # defines the monolithic coupling forms for fluid mechanics in ALE reference frame
    def set_variational_forms(self):

        # any DBC conditions that we want to set from fluid to ALE (mandatory for FSI or FrSI)
        if bool(self.coupling_fluid_ale):

            ids_fluid_ale = self.coupling_fluid_ale['surface_ids']

            if self.coupling_fluid_ale['type'] == 'strong_dirichlet':

                dbcs_coup_fluid_ale = []
                for i in range(len(ids_fluid_ale)):
                    dbcs_coup_fluid_ale.append( fem.dirichletbc(self.ufa, fem.locate_dofs_topological(self.pba.V_d, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == ids_fluid_ale[i]])) )

                # pay attention to order... first u=uf, then the others... hence re-set!
                self.pba.bc.dbcs = []
                self.pba.bc.dbcs += dbcs_coup_fluid_ale
                # Dirichlet boundary conditions
                if 'dirichlet' in self.pba.bc_dict.keys():
                    self.pba.bc.dirichlet_bcs(self.pba.bc_dict['dirichlet'], self.pba.V_d)

                # NOTE: linearization entries due to strong DBCs of fluid on ALE are currently not considered in the monolithic block matrix!

                #fdi = set(gather_surface_dof_indices(self.pba, self.pba.V_d, ids_fluid_ale, self.comm))

                ## fluid and ALE actually should have same sizes...
                #locmatsize_d = self.pba.V_d.dofmap.index_map.size_local * self.pba.V_d.dofmap.index_map_bs
                #matsize_d = self.pba.V_d.dofmap.index_map.size_global * self.pba.V_d.dofmap.index_map_bs

                #locmatsize_v = self.pbf.V_v.dofmap.index_map.size_local * self.pbf.V_v.dofmap.index_map_bs
                #matsize_v = self.pbf.V_v.dofmap.index_map.size_global * self.pbf.V_v.dofmap.index_map_bs

                ## now we have to assemble the offdiagonal stiffness due to the DBCs w=v set on the ALE surface - cannot be treated with "derivative" since DBCs are not present in form
                #self.K_dv = PETSc.Mat().createAIJ(size=((locmatsize_d,matsize_d),(locmatsize_v,matsize_v)), bsize=None, nnz=None, csr=None, comm=self.comm)
                #self.K_dv.setUp()

                #for i in range(matsize_d):
                    #if i in fdi:
                        #self.K_dv[i,i] = -1.0/(self.pbf.timefac*self.pbf.dt)

                #self.K_dv.assemble()

            elif self.coupling_fluid_ale['type'] == 'robin':

                beta = self.coupling_fluid_ale['beta']

                work_dbc_robin_fluid_ale = ufl.as_ufl(0)
                for i in range(len(ids_fluid_ale)):
                    db_ = ufl.ds(subdomain_data=self.pba.io.mt_b1, subdomain_id=ids_fluid_ale[i], metadata={'quadrature_degree': self.pba.quad_degree})
                    work_dbc_robin_fluid_ale += self.pba.vf.deltaW_int_robin_cur(self.pba.d, self.pbf.ufluid, self.pba.ki.F(self.pba.d), beta, db_) # here, ufluid as form is used!

                # add to ALE internal virtual work
                self.pba.weakform_d += work_dbc_robin_fluid_ale
                # add to ALE jacobian form and define offdiagonal derivative w.r.t. fluid
                self.pba.weakform_lin_dd += ufl.derivative(work_dbc_robin_fluid_ale, self.pba.d, self.pba.dd)
                self.weakform_lin_dv = ufl.derivative(work_dbc_robin_fluid_ale, self.pbf.v, self.pbf.dv) # only contribution is from weak DBC here!

            else:
                raise ValueError("Unknown coupling_fluid_ale option for fluid to ALE!")

        # any DBC conditions that we want to set from ALE to fluid
        if bool(self.coupling_ale_fluid):

            ids_ale_fluid = self.coupling_ale_fluid['surface_ids']

            if self.coupling_ale_fluid['type'] == 'strong_dirichlet':

                dbcs_coup_ale_fluid = []
                for i in range(len(ids_ale_fluid)):
                    dbcs_coup_ale_fluid.append( fem.dirichletbc(self.wf, fem.locate_dofs_topological(self.pbf.V_v, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == ids_ale_fluid[i]])) )

                # pay attention to order... first v=w, then the others... hence re-set!
                self.pbf.bc.dbcs = []
                self.pbf.bc.dbcs += dbcs_coup_ale_fluid
                # Dirichlet boundary conditions
                if 'dirichlet' in self.pbf.bc_dict.keys():
                    self.pbf.bc.dirichlet_bcs(self.pbf.bc_dict['dirichlet'], self.pbf.V_v)

                #NOTE: linearization entries due to strong DBCs of fluid on ALE are currently not considered in the monolithic block matrix!

            elif self.coupling_ale_fluid['type'] == 'robin' or self.coupling_ale_fluid['type'] == 'robin_internal':

                beta = self.coupling_ale_fluid['beta']

                work_robin_ale_fluid = ufl.as_ufl(0)
                for i in range(len(ids_ale_fluid)):
                    if self.coupling_ale_fluid['type'] == 'robin':
                        db_ = ufl.ds(subdomain_data=self.pbf.io.mt_b1, subdomain_id=ids_ale_fluid[i], metadata={'quadrature_degree': self.pbf.quad_degree})
                        work_robin_ale_fluid += self.pbf.vf.deltaW_int_robin_cur(self.pbf.v, self.pba.wel, beta, db_, Fale=self.pba.ki.F(self.pba.d)) # here, wel as form is used!
                    # if we have an internal surface, we need to use ufl's dS instead of ds
                    if self.coupling_ale_fluid['type'] == 'robin_internal':
                        db_ = ufl.dS(subdomain_data=self.pbf.io.mt_b1, subdomain_id=ids_ale_fluid[i], metadata={'quadrature_degree': self.pbf.quad_degree})
                        work_robin_ale_fluid += self.pbf.vf.deltaW_int_robin_cur(self.pbf.v, self.pba.wel, beta, db_, Fale=self.pba.ki.F(self.pba.d), fcts='+') # here, wel as form is used!

                # add to fluid internal virtual power
                self.pbf.weakform_v += work_robin_ale_fluid
                # add to fluid jacobian form
                self.pbf.weakform_lin_vv += ufl.derivative(work_robin_ale_fluid, self.pbf.v, self.pbf.dv)

            else:
                raise ValueError("Unknown coupling_ale_fluid option for ALE to fluid!")

        # derivative of fluid momentum w.r.t. ALE displacement - also includes potential weak Dirichlet or Robin BCs from ALE to fluid!
        self.weakform_lin_vd = ufl.derivative(self.pbf.weakform_v, self.pba.d, self.pba.dd)

        # derivative of fluid continuity w.r.t. ALE displacement
        self.weakform_lin_pd = ufl.derivative(self.pbf.weakform_p, self.pba.d, self.pba.dd)


    def set_problem_residual_jacobian_forms(self):

        # fluid + ALE
        self.pbf.set_problem_residual_jacobian_forms()
        self.pba.set_problem_residual_jacobian_forms()

        tes = time.time()
        if self.comm.rank == 0:
            print('FEM form compilation for coupling...')
            sys.stdout.flush()

        # coupling
        if self.io.USE_MIXED_DOLFINX_BRANCH:
            self.jac_vd = fem.form(self.weakform_lin_vd, entity_maps=self.pbf.entity_maps)
            self.jac_pd = fem.form(self.weakform_lin_pd, entity_maps=self.pbf.entity_maps)
        else:
            self.jac_vd = fem.form(self.weakform_lin_vd)
            self.jac_pd = fem.form(self.weakform_lin_pd)
        if bool(self.coupling_fluid_ale):
            if self.coupling_fluid_ale['type'] == 'robin':
                self.jac_dv = fem.form(self.weakform_lin_dv)

        tee = time.time() - tes
        if self.comm.rank == 0:
            print('FEM form compilation for coupling finished, te = %.2f s' % (tee))
            sys.stdout.flush()


    def assemble_residual_stiffness(self, t, subsolver=None):

        K_list = [[None]*3 for _ in range(3)]
        r_list = [None]*3

        if bool(self.coupling_fluid_ale):
            if self.coupling_fluid_ale['type'] == 'strong_dirichlet':
                # we need a vector representation of ufluid to apply in ALE DBCs
                uf_vec = self.pbf.ti.update_uf_ost(self.pbf.v.vector, self.pbf.v_old.vector, self.pbf.uf_old.vector, ufl=False)
                self.ufa.vector.axpby(1.0, 0.0, uf_vec)
                self.ufa.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                #K_list[2][0] = self.K_uv
                uf_vec.destroy()
            if self.coupling_fluid_ale['type'] == 'robin':
                K_uv = fem.petsc.assemble_matrix(self.jac_dv, self.pba.bc.dbcs)
                K_uv.assemble()
                K_list[2][0] = K_uv

        if bool(self.coupling_ale_fluid):
            if self.coupling_ale_fluid['type'] == 'strong_dirichlet':
                # we need a vector representation of w to apply in fluid DBCs
                w_vec = self.pba.ti.update_w_ost(self.pba.d.vector, self.pba.d_old.vector, self.pba.w_old.vector, ufl=False)
                self.wf.vector.axpby(1.0, 0.0, w_vec)
                self.wf.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                w_vec.destroy()

        r_list_fluid, K_list_fluid = self.pbf.assemble_residual_stiffness(t)

        r_list_ale, K_list_ale = self.pba.assemble_residual_stiffness(t)

        K_list[0][0] = K_list_fluid[0][0]
        K_list[0][1] = K_list_fluid[0][1]

        # derivative of fluid momentum w.r.t. ALE displacement
        K_vd = fem.petsc.assemble_matrix(self.jac_vd, self.pbf.bc.dbcs)
        K_vd.assemble()
        K_list[0][2] = K_vd

        K_list[1][0] = K_list_fluid[1][0]
        K_list[1][1] = K_list_fluid[1][1]

        # derivative of fluid continuity w.r.t. ALE displacement
        K_pd = fem.petsc.assemble_matrix(self.jac_pd, [])
        K_pd.assemble()
        K_list[1][2] = K_pd

        K_list[2][2] = K_list_ale[0][0]

        r_list[0] = r_list_fluid[0]
        r_list[1] = r_list_fluid[1]
        r_list[2] = r_list_ale[0]

        return r_list, K_list


    def get_index_sets(self, isoptions={}):

        if self.have_rom: # currently, ROM can only be on (subset of) first variable
            vred = PETSc.Vec().createMPI(size=(self.rom.V.getLocalSize()[1],self.rom.V.getSize()[1]), comm=self.comm)
            self.rom.V.multTranspose(self.pbf.v.vector, vred)
            vvec = vred
        else:
            vvec = self.pbf.v.vector

        offset_v = vvec.getOwnershipRange()[0] + self.pbf.p.vector.getOwnershipRange()[0] + self.pba.d.vector.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec.getLocalSize(), first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createStride(len(self.rom.im_rom_r), first=offset_v, step=1, comm=self.comm) # same offset, since contained in v
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec.getLocalSize()
        iset_p = PETSc.IS().createStride(self.pbf.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_d = offset_p + self.pbf.p.vector.getLocalSize()
        iset_d = PETSc.IS().createStride(self.pba.d.vector.getLocalSize(), first=offset_d, step=1, comm=self.comm)

        # for convenience, add ALE as last in list (since we might want to address this with a decoupled block solve)
        if isoptions['rom_to_new']:
            return [iset_v, iset_p, iset_r, iset_d]
        else:
            return [iset_v, iset_p, iset_d]


    # DEPRECATED: This is something we should actually not do! It will mess with gradients we need w.r.t. the reference (e.g. for FrSI)
    # Instead of moving the mesh, we formulate Navier-Stokes w.r.t. a reference state using the ALE kinematics
    def move_mesh(self):

        d = fem.Function(self.pba.Vcoord)
        d.interpolate(self.pba.d)
        self.io.mesh.geometry.x[:,:self.pba.dim] += d.x.array.reshape((-1, self.pba.dim))
        if self.comm.rank == 0:
            print('Updating mesh...')
            sys.stdout.flush()


    def print_warning_ale(self):
        if self.comm.rank == 0:
            print(' ')
            print('*********************************************************************************************************************')
            print('*** Warning: You are solving Navier-Stokes by only updating the frame after each time step! This is inconsistent! ***')
            print('*********************************************************************************************************************')
            print(' ')
            sys.stdout.flush()


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # fluid + ALE problem
        self.pbf.read_restart(sname, N)
        self.pba.read_restart(sname, N)


    def evaluate_initial(self):

        # issue a warning to the user in case of inconsistent fluid-ALE coupling
        # (might though be wanted in some cases for efficiency increases...)
        if self.fluid_on_deformed=='from_last_step' or self.fluid_on_deformed=='mesh_move':
            self.print_warning_ale()


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):

        return 0.


    def evaluate_pre_solve(self, t):

        self.pbf.evaluate_pre_solve(t)
        self.pba.evaluate_pre_solve(t)


    def evaluate_post_solve(self, t, N):

        self.pbf.evaluate_post_solve(t, N)
        self.pba.evaluate_post_solve(t, N)


    def set_output_state(self):

        self.pbf.set_output_state()
        self.pba.set_output_state()


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t)


    def update(self):

        # update time step - fluid and ALE
        self.pbf.update()
        self.pba.update()

        if self.fluid_on_deformed=='mesh_move':
            self.move_mesh()


    def print_to_screen(self):

        self.pbf.print_to_screen()
        self.pba.print_to_screen()


    def induce_state_change(self):

        self.pbf.induce_state_change()
        self.pba.induce_state_change()


    def write_restart(self, sname, N):

        self.pbf.write_restart(sname, N)
        self.pba.write_restart(sname, N)


    def check_abort(self, t):

        self.pbf.check_abort(t)
        self.pba.check_abort(t)



class FluidmechanicsAleSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        # perform Proper Orthogonal Decomposition
        if self.pb.have_rom:
            self.pb.rom.prepare_rob()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)

        if self.pb.pbf.prestress_initial and self.pb.pbf.restart_step == 0:
            # initialize fluid mechanics solver
            self.solverprestr = FluidmechanicsSolver(self.pb.pbf, self.solver_params)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the FrSI problem
        if self.pb.pbf.prestress_initial and self.pb.pbf.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.ksp.destroy()

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbf.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.pbf.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.pbf.entity_maps)
            else:
                res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)
