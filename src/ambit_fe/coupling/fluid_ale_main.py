#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, math
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from ..solver import solver_nonlin
from .. import expression, ioparams

from ..fluid.fluid_main import FluidmechanicsProblem, FluidmechanicsSolverPrestr
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base
from ..meshutils import gather_surface_dof_indices


class FluidmechanicsAleProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params_fluid, fem_params_ale, constitutive_models_fluid, constitutive_models_ale, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params, io, mor_params={}, comm=None):
        super().__init__(io_params, time_params, comm)

        ioparams.check_params_coupling_fluid_ale(coupling_params)

        self.problem_physics = 'fluid_ale'

        self.coupling_params = coupling_params

        try: self.coupling_fluid_ale = self.coupling_params['coupling_fluid_ale']
        except: self.coupling_fluid_ale = {}

        try: self.coupling_ale_fluid = self.coupling_params['coupling_ale_fluid']
        except: self.coupling_ale_fluid = {}

        try: self.fluid_on_deformed = self.coupling_params['fluid_on_deformed']
        except: self.fluid_on_deformed = 'consistent'

        try: self.coupling_strategy = self.coupling_params['coupling_strategy']
        except: self.coupling_strategy = 'monolithic'

        self.have_dbc_fluid_ale, self.have_weak_dirichlet_fluid_ale, self.have_dbc_ale_fluid, self.have_robin_ale_fluid = False, False, False, False

        # initialize problem instances (also sets the variational forms for the fluid and ALE problem)
        self.pba = AleProblem(io_params, time_params, fem_params_ale, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params, comm=self.comm)
        # ALE variables that are handed to fluid problem
        alevariables = {'Fale' : self.pba.ki.F(self.pba.d), 'Fale_old' : self.pba.ki.F(self.pba.d_old), 'w' : self.pba.wel, 'w_old' : self.pba.w_old, 'fluid_on_deformed' : self.fluid_on_deformed}
        self.pbf = FluidmechanicsProblem(io_params, time_params, fem_params_fluid, constitutive_models_fluid, bc_dict_fluid, time_curves, io, mor_params=mor_params, comm=self.comm, alevar=alevariables)

        self.pbrom = self.pbf # ROM problem can only be fluid

        # modify results to write...
        self.pbf.results_to_write = io_params['results_to_write'][0]
        self.pba.results_to_write = io_params['results_to_write'][1]

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

        if self.coupling_strategy == 'monolithic':
            self.numdof = self.pbf.numdof + self.pba.numdof
        else:
            self.numdof = [self.pbf.numdof, self.pba.numdof]

        self.sub_solve = False
        self.print_enhanced_info = self.pbf.io.print_enhanced_info

        # number of fields involved
        self.nfields = 3

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)],  [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbf.num_dupl > 1: is_ghosted = [1, 2, 1]
        else:                     is_ghosted = [1, 1, 1]
        return [self.pbf.v.vector, self.pbf.p.vector, self.pba.d.vector], is_ghosted


    # defines the monolithic coupling forms for fluid mechanics in ALE reference frame
    def set_variational_forms(self):

        # any DBC conditions that we want to set from fluid to ALE (mandatory for FSI or FrSI)
        if bool(self.coupling_fluid_ale):

            dbcs_coup_fluid_ale, work_weak_dirichlet_fluid_ale, work_weak_dirichlet_fluid_ale_old = [], ufl.as_ufl(0), ufl.as_ufl(0)

            for j in range(len(self.coupling_fluid_ale)):

                ids_fluid_ale = self.coupling_fluid_ale[j]['surface_ids']

                if self.coupling_fluid_ale[j]['type'] == 'strong_dirichlet':

                    for i in range(len(ids_fluid_ale)):
                        dbcs_coup_fluid_ale.append( fem.dirichletbc(self.ufa, fem.locate_dofs_topological(self.pba.V_d, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == ids_fluid_ale[i]])) )

                    # NOTE: linearization entries due to strong DBCs of fluid on ALE are currently not considered in the monolithic block matrix!

                    # fdi = set(gather_surface_dof_indices(self.pba.io, self.pba.V_d, ids_fluid_ale, self.comm))

                    # # fluid and ALE actually should have same sizes...
                    # locmatsize_d = self.pba.V_d.dofmap.index_map.size_local * self.pba.V_d.dofmap.index_map_bs
                    # matsize_d = self.pba.V_d.dofmap.index_map.size_global * self.pba.V_d.dofmap.index_map_bs
                    #
                    # locmatsize_v = self.pbf.V_v.dofmap.index_map.size_local * self.pbf.V_v.dofmap.index_map_bs
                    # matsize_v = self.pbf.V_v.dofmap.index_map.size_global * self.pbf.V_v.dofmap.index_map_bs
                    #
                    # # now we have to assemble the offdiagonal stiffness due to the DBCs w=v set on the ALE surface - cannot be treated with "derivative" since DBCs are not present in form
                    # self.K_dv = PETSc.Mat().createAIJ(size=((locmatsize_d,matsize_d),(locmatsize_v,matsize_v)), bsize=None, nnz=None, csr=None, comm=self.comm)
                    # self.K_dv.setUp()
                    #
                    # for i in range(matsize_d):
                    #     if i in fdi:
                    #         self.K_dv[i,i] = -self.pbf.timefac*self.pbf.dt
                    #
                    # self.K_dv.assemble()

                elif self.coupling_fluid_ale[j]['type'] == 'weak_dirichlet':

                    beta = self.coupling_fluid_ale[j]['beta']

                    for i in range(len(ids_fluid_ale)):
                        db_ = ufl.ds(domain=self.pba.io.mesh, subdomain_data=self.pba.io.mt_b1, subdomain_id=ids_fluid_ale[i], metadata={'quadrature_degree': self.pba.quad_degree})

                        for n in range(self.pba.num_domains):
                            work_weak_dirichlet_fluid_ale += self.pba.vf.deltaW_int_nitsche_dirichlet(self.pba.d, self.pbf.ufluid, self.pba.ma[n].stress(self.pba.var_d), beta, db_) # here, ufluid as form is used!
                            work_weak_dirichlet_fluid_ale_old += self.pba.vf.deltaW_int_nitsche_dirichlet(self.pba.d_old, self.pbf.uf_old, self.pba.ma[n].stress(self.pba.var_d), beta, db_)

                else:
                    raise ValueError("Unknown coupling_fluid_ale option for fluid to ALE!")

            # now add the DBCs: pay attention to order... first u=uf, then the others... hence re-set!
            if bool(dbcs_coup_fluid_ale):
                self.pba.bc.dbcs = []
                self.pba.bc.dbcs += dbcs_coup_fluid_ale
                # Dirichlet boundary conditions
                if 'dirichlet' in self.pba.bc_dict.keys():
                    self.pba.bc.dirichlet_bcs(self.pba.bc_dict['dirichlet'], self.pba.V_d)
                self.have_dbc_fluid_ale = True

            if not isinstance(work_weak_dirichlet_fluid_ale, ufl.constantvalue.Zero):
                # add to ALE internal virtual work
                self.pba.weakform_d += self.pbf.timefac * work_weak_dirichlet_fluid_ale + (1.-self.pbf.timefac) * work_weak_dirichlet_fluid_ale_old
                # add to ALE jacobian form and define offdiagonal derivative w.r.t. fluid
                self.pba.weakform_lin_dd += self.pbf.timefac * ufl.derivative(work_weak_dirichlet_fluid_ale, self.pba.d, self.pba.dd)
                self.weakform_lin_dv = self.pbf.timefac * ufl.derivative(work_weak_dirichlet_fluid_ale, self.pbf.v, self.pbf.dv) # only contribution is from weak DBC here!
                self.have_weak_dirichlet_fluid_ale = True

        # any DBC conditions that we want to set from ALE to fluid
        if bool(self.coupling_ale_fluid):

            dbcs_coup_ale_fluid, work_robin_ale_fluid, work_robin_ale_fluid_old = [], ufl.as_ufl(0), ufl.as_ufl(0)

            for j in range(len(self.coupling_ale_fluid)):

                ids_ale_fluid = self.coupling_ale_fluid[j]['surface_ids']

                if self.coupling_ale_fluid[j]['type'] == 'strong_dirichlet':

                    for i in range(len(ids_ale_fluid)):
                        dbcs_coup_ale_fluid.append( fem.dirichletbc(self.wf, fem.locate_dofs_topological(self.pbf.V_v, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[self.io.mt_b1.values == ids_ale_fluid[i]])) )

                    #NOTE: linearization entries due to strong DBCs of fluid on ALE are currently not considered in the monolithic block matrix!

                elif self.coupling_ale_fluid[j]['type'] == 'robin':

                    for i in range(len(ids_ale_fluid)):
                        if self.coupling_ale_fluid[j]['type'] == 'robin':
                            beta = self.coupling_ale_fluid[j]['beta']
                            db_ = ufl.ds(domain=self.pbf.io.mesh, subdomain_data=self.pbf.io.mt_b1, subdomain_id=ids_ale_fluid[i], metadata={'quadrature_degree': self.pbf.quad_degree})
                            work_robin_ale_fluid += self.pbf.vf.deltaW_int_robin_cur(self.pbf.v, self.pba.wel, beta, db_, Fale=self.pba.ki.F(self.pba.d)) # here, wel as form is used!
                            work_robin_ale_fluid_old += self.pbf.vf.deltaW_int_robin_cur(self.pbf.v_old, self.pba.w_old, beta, db_, Fale=self.pba.ki.F(self.pba.d_old))

                else:
                    raise ValueError("Unknown coupling_ale_fluid option for ALE to fluid!")

            if bool(dbcs_coup_ale_fluid):
                # now add the DBCs: pay attention to order... first v=w, then the others... hence re-set!
                self.pbf.bc.dbcs = []
                self.pbf.bc.dbcs += dbcs_coup_ale_fluid
                # Dirichlet boundary conditions
                if 'dirichlet' in self.pbf.bc_dict.keys():
                    self.pbf.bc.dirichlet_bcs(self.pbf.bc_dict['dirichlet'], self.pbf.V_v)
                self.have_dbc_ale_fluid = True

            if not isinstance(work_robin_ale_fluid, ufl.constantvalue.Zero):
                # add to fluid internal virtual power
                self.pbf.weakform_v += self.pbf.timefac * work_robin_ale_fluid + (1.-self.pbf.timefac) * work_robin_ale_fluid_old
                # add to fluid jacobian form
                self.pbf.weakform_lin_vv += self.pbf.timefac * ufl.derivative(work_robin_ale_fluid, self.pbf.v, self.pbf.dv)
                self.have_robin_ale_fluid = True

        # derivative of fluid momentum w.r.t. ALE displacement - also includes potential weak Dirichlet or Robin BCs from ALE to fluid!
        self.weakform_lin_vd = ufl.derivative(self.pbf.weakform_v, self.pba.d, self.pba.dd)

        # derivative of fluid continuity w.r.t. ALE displacement
        self.weakform_lin_pd = []
        for n in range(self.pbf.num_domains):
            self.weakform_lin_pd.append( ufl.derivative(self.pbf.weakform_p[n], self.pba.d, self.pba.dd) )


    def set_problem_residual_jacobian_forms(self):

        # fluid + ALE
        self.pbf.set_problem_residual_jacobian_forms()
        self.pba.set_problem_residual_jacobian_forms()

        if self.coupling_strategy=='monolithic':

            tes = time.time()
            if self.comm.rank == 0:
                print('FEM form compilation for fluid-ALE coupling...')
                sys.stdout.flush()

            if not bool(self.pbf.io.duplicate_mesh_domains):
                self.weakform_lin_pd = sum(self.weakform_lin_pd)

            # coupling
            if self.io.USE_MIXED_DOLFINX_BRANCH:
                self.jac_vd = fem.form(self.weakform_lin_vd, entity_maps=self.pbf.io.entity_maps)
                self.jac_pd = fem.form(self.weakform_lin_pd, entity_maps=self.pbf.io.entity_maps)
                if self.pbf.num_dupl > 1:
                    self.jac_pd_ = []
                    for j in range(self.pbf.num_dupl):
                        self.jac_pd_.append([self.jac_pd[j]])
                if self.have_weak_dirichlet_fluid_ale:
                    self.jac_dv = fem.form(self.weakform_lin_dv, entity_maps=self.pbf.io.entity_maps)
            else:
                self.jac_vd = fem.form(self.weakform_lin_vd)
                self.jac_pd = fem.form(self.weakform_lin_pd)
                if self.have_weak_dirichlet_fluid_ale:
                    self.jac_dv = fem.form(self.weakform_lin_dv)

            tee = time.time() - tes
            if self.comm.rank == 0:
                print('FEM form compilation for fluid-ALE coupling finished, te = %.2f s' % (tee))
                sys.stdout.flush()


    def set_problem_vector_matrix_structures(self):

        self.pbf.set_problem_vector_matrix_structures()
        self.pba.set_problem_vector_matrix_structures()

        if self.coupling_strategy=='monolithic':

            self.K_vd = fem.petsc.create_matrix(self.jac_vd)
            if self.have_weak_dirichlet_fluid_ale:
                self.K_dv = fem.petsc.create_matrix(self.jac_dv)
            else:
                self.K_dv = None

            if self.pbf.num_dupl > 1:
                self.K_pd = fem.petsc.create_matrix_block(self.jac_pd_)
            else:
                self.K_pd = fem.petsc.create_matrix(self.jac_pd)


    def assemble_residual(self, t, subsolver=None):

        self.evaluate_residual_dbc_coupling()

        self.pbf.assemble_residual(t)
        self.pba.assemble_residual(t)

        self.r_list[0] = self.pbf.r_list[0]
        self.r_list[1] = self.pbf.r_list[1]
        self.r_list[2] = self.pba.r_list[0]


    def assemble_stiffness(self, t, subsolver=None):

        # if self.have_dbc_fluid_ale:
            #K_list[2][0] = self.K_dv
        if self.have_weak_dirichlet_fluid_ale:
            self.K_dv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_dv, self.jac_dv, self.pba.bc.dbcs)
            self.K_dv.assemble()
            self.K_list[2][0] = self.K_dv

        self.pbf.assemble_stiffness(t)
        self.pba.assemble_stiffness(t)

        self.K_list[0][0] = self.pbf.K_list[0][0]
        self.K_list[0][1] = self.pbf.K_list[0][1]

        # derivative of fluid momentum w.r.t. ALE displacement
        self.K_vd.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vd, self.jac_vd, self.pbf.bc.dbcs)
        self.K_vd.assemble()
        self.K_list[0][2] = self.K_vd

        self.K_list[1][0] = self.pbf.K_list[1][0]
        self.K_list[1][1] = self.pbf.K_list[1][1]

        # derivative of fluid continuity w.r.t. ALE displacement
        self.K_pd.zeroEntries()
        if self.pbf.num_dupl > 1:
            fem.petsc.assemble_matrix_block(self.K_pd, self.jac_pd_, [])
        else:
            fem.petsc.assemble_matrix(self.K_pd, self.jac_pd, [])
        self.K_pd.assemble()
        self.K_list[1][2] = self.K_pd

        self.K_list[2][2] = self.pba.K_list[0][0]


    def evaluate_residual_dbc_coupling(self):

        if self.have_dbc_fluid_ale:
            # we need a vector representation of ufluid to apply in ALE DBCs
            self.pbf.ti.update_uf_ost(self.pbf.v.vector, self.pbf.v_old.vector, self.pbf.uf_old.vector, ufout=self.pbf.uf.vector, ufl=False)
            self.ufa.vector.axpby(1.0, 0.0, self.pbf.uf.vector)
            self.ufa.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if self.have_dbc_ale_fluid:
            # we need a vector representation of w to apply in fluid DBCs
            self.pba.ti.update_w_ost(self.pba.d.vector, self.pba.d_old.vector, self.pba.w_old.vector, wout=self.pba.w.vector, ufl=False)
            self.wf.vector.axpby(1.0, 0.0, self.pba.w.vector)
            self.wf.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.vector.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.vector.getLocalSize()

        offset_v = vvec_or0 + self.pbf.p.vector.getOwnershipRange()[0] + self.pba.d.vector.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.pbf.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_d = offset_p + self.pbf.p.vector.getLocalSize()
        iset_d = PETSc.IS().createStride(self.pba.d.vector.getLocalSize(), first=offset_d, step=1, comm=self.comm)

        if isoptions['ale_to_v']:
            iset_v = iset_v.expand(iset_d) # add ALE to velocity block

        if isoptions['rom_to_new']:
            ilist = [iset_v, iset_p, iset_r, iset_d]
        else:
            ilist = [iset_v, iset_p, iset_d]

        if isoptions['ale_to_v']: ilist.pop(-1)

        return ilist


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

        # read restart information
        if self.restart_step > 0:
            self.io.readcheckpoint(self, N)
            self.simname += '_r'+str(N)
            # TODO: quick-fix - simname variables of single field problems need to be addressed, too
            # but this should be handled by one variable, however neeeds revamp of I/O
            self.pbf.simname += '_r'+str(N)
            self.pba.simname += '_r'+str(N)


    def evaluate_initial(self):

        self.pbf.evaluate_initial()

        # issue a warning to the user in case of inconsistent fluid-ALE coupling
        # (might though be wanted in some cases for efficiency increases...)
        if self.fluid_on_deformed=='from_last_step' or self.fluid_on_deformed=='mesh_move':
            self.print_warning_ale()


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):

        return 0.


    def evaluate_pre_solve(self, t, N):

        self.pbf.evaluate_pre_solve(t, N)
        self.pba.evaluate_pre_solve(t, N)


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

        if self.fluid_on_deformed=='mesh_move':
            self.move_mesh()


    def print_to_screen(self):

        self.pbf.print_to_screen()
        self.pba.print_to_screen()


    def induce_state_change(self):

        self.pbf.induce_state_change()
        self.pba.induce_state_change()


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)


    def check_abort(self, t):

        self.pbf.check_abort(t)
        self.pba.check_abort(t)


    def destroy(self):

        self.pbf.destroy()
        self.pba.destroy()



class FluidmechanicsAleSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        if self.pb.coupling_strategy=='monolithic':
            self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)
        elif self.pb.coupling_strategy=='partitioned':
            self.solnln = solver_nonlin.solver_nonlinear([self.pb.pbf,self.pb.pba], self.solver_params, cp=self.pb)
        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")

        if (self.pb.pbf.prestress_initial or self.pb.pbf.prestress_initial_only) and self.pb.pbf.restart_step == 0:
            solver_params_prestr = copy.deepcopy(self.solver_params)
            # modify solver parameters in case user specified alternating ones for prestressing (should do, because it's a 2x2 problem)
            try: solver_params_prestr['solve_type'] = self.solver_params['solve_type_prestr']
            except: pass
            try: solver_params_prestr['block_precond'] = self.solver_params['block_precond_prestr']
            except: pass
            try: solver_params_prestr['precond_fields'] = self.solver_params['precond_fields_prestr']
            except: pass
            # initialize fluid mechanics solver
            self.solverprestr = FluidmechanicsSolverPrestr(self.pb.pbf, self.solver_params)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the FrSI problem
        if (self.pb.pbf.prestress_initial or self.pb.pbf.prestress_initial_only) and self.pb.pbf.restart_step == 0:
            # solve solid prestress problem
            self.solverprestr.solve_initial_prestress()
            self.solverprestr.solnln.destroy()

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbf.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.pbf.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.pbf.io.entity_maps)
            else:
                res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)


    # we overload this function here in order to take care of the partitioned solve,
    # where the ROM needs to be an object of the fluid, not the coupled problem
    def evaluate_assemble_system_initial(self, subsolver=None):

        # evaluate old initial state of model
        self.evaluate_system_initial()

        if self.pb.coupling_strategy=='monolithic':

            self.pb.assemble_residual(self.pb.t_init)
            self.pb.assemble_stiffness(self.pb.t_init)

            # create ROM matrix structures
            if self.pb.rom:
                self.pb.rom.set_reduced_data_structures_residual(self.pb.r_list, self.pb.r_list_rom)
                self.pb.K_list_tmp = [[None]]
                self.pb.rom.set_reduced_data_structures_matrix(self.pb.K_list, self.pb.K_list_rom, self.pb.K_list_tmp)

        elif self.pb.coupling_strategy=='partitioned':

            self.pb.pbf.rom = self.pb.rom

            self.pb.assemble_residual(self.pb.t_init)
            self.pb.pbf.assemble_stiffness(self.pb.t_init)
            self.pb.pba.assemble_stiffness(self.pb.t_init)

            # create ROM matrix structures
            if self.pb.pbf.rom:
                self.pb.pbf.rom.set_reduced_data_structures_residual(self.pb.pbf.r_list, self.pb.pbf.r_list_rom)
                self.pb.pbf.K_list_tmp = [[None]]
                self.pb.pbf.rom.set_reduced_data_structures_matrix(self.pb.pbf.K_list, self.pb.pbf.K_list_rom, self.pb.pbf.K_list_tmp)

        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
