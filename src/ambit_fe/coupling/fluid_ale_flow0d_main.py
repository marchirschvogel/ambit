#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
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

from .fluid_ale_main import FluidmechanicsAleProblem
from .fluid_flow0d_main import FluidmechanicsFlow0DProblem
from ..fluid.fluid_main import FluidmechanicsSolverPrestr
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base
from ..meshutils import gather_surface_dof_indices


class FluidmechanicsAleFlow0DProblem(FluidmechanicsAleProblem,problem_base):

    def __init__(self, io_params, time_params_fluid, time_params_flow0d, fem_params_fluid, fem_params_ale, constitutive_models_fluid, constitutive_models_ale, model_params_flow0d, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params_fluid_ale, coupling_params_fluid_flow0d, io, mor_params={}, comm=None):
        problem_base.__init__(self, io_params, time_params_fluid, comm)

        ioparams.check_params_coupling_fluid_ale(coupling_params_fluid_ale)

        self.problem_physics = 'fluid_ale_flow0d'

        try: self.coupling_fluid_ale = coupling_params_fluid_ale['coupling_fluid_ale']
        except: self.coupling_fluid_ale = {}

        try: self.coupling_ale_fluid = coupling_params_fluid_ale['coupling_ale_fluid']
        except: self.coupling_ale_fluid = {}

        try: self.fluid_on_deformed = coupling_params_fluid_ale['fluid_on_deformed']
        except: self.fluid_on_deformed = 'consistent'

        try: self.coupling_strategy = coupling_params_fluid_ale['coupling_strategy']
        except: self.coupling_strategy = 'monolithic'

        self.have_dbc_fluid_ale, self.have_weak_dirichlet_fluid_ale, self.have_dbc_ale_fluid, self.have_robin_ale_fluid = False, False, False, False

        # initialize problem instances (also sets the variational forms for the fluid flow0d problem)
        self.pba  = AleProblem(io_params, time_params_fluid, fem_params_ale, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params, comm=self.comm)
        # ALE variables that are handed to fluid problem
        alevariables = {'Fale' : self.pba.ki.F(self.pba.d), 'Fale_old' : self.pba.ki.F(self.pba.d_old), 'w' : self.pba.wel, 'w_old' : self.pba.w_old, 'fluid_on_deformed' : self.fluid_on_deformed}
        self.pbf0 = FluidmechanicsFlow0DProblem(io_params, time_params_fluid, time_params_flow0d, fem_params_fluid, constitutive_models_fluid, model_params_flow0d, bc_dict_fluid, time_curves, coupling_params_fluid_flow0d, io, mor_params=mor_params, comm=self.comm, alevar=alevariables)

        self.pbf = self.pbf0.pbf
        self.pb0 = self.pbf0.pb0

        self.pbrom = self.pbf # ROM problem can only be fluid

        # modify results to write...
        self.pbf.results_to_write = io_params['results_to_write'][0]
        self.pba.results_to_write = io_params['results_to_write'][1]

        self.io = io

        # indicator for no periodic reference state estimation
        self.noperiodicref = 1

        self.localsolve = False
        self.print_subiter = self.pbf0.print_subiter

        # NOTE: Fluid and ALE function spaces should be of the same type, but are different objects.
        # For some reason, when applying a function from one funtion space as DBC to another function space,
        # errors occur. Therefore, we define these auxiliary variables and interpolate respectively...

        # fluid displacement, but defined within ALE function space
        self.ufa = fem.Function(self.pba.V_d)
        # ALE velocity, but defined within fluid function space
        self.wf = fem.Function(self.pbf.V_v)

        self.set_variational_forms()

        if self.coupling_strategy == 'monolithic':
            self.numdof = self.pbf.numdof + self.pbf0.lm.getSize() + self.pba.numdof
        else:
            self.numdof = [self.pbf.numdof + self.pbf0.lm.getSize(), self.pba.numdof]

        self.sub_solve = True
        self.print_enhanced_info = self.pbf.io.print_enhanced_info

        # number of fields involved
        self.nfields = 4

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbf0.pbf.num_dupl > 1: is_ghosted = [1, 2, 0, 1]
        else:                          is_ghosted = [1, 1, 0, 1]
        return [self.pbf.v.vector, self.pbf.p.vector, self.pbf0.lm, self.pba.d.vector], is_ghosted


    def set_variational_forms(self):
        super().set_variational_forms()

        self.dcqd = []
        for n in range(self.pbf0.num_coupling_surf):
            self.dcqd.append(ufl.derivative(self.pbf0.cq[n], self.pba.d, self.pba.dd))


    def set_problem_residual_jacobian_forms(self):

        super().set_problem_residual_jacobian_forms()
        self.pbf0.set_problem_residual_jacobian_forms_coupling()

        if self.coupling_strategy=='monolithic':

            tes = time.time()
            if self.comm.rank == 0:
                print('FEM form compilation for ALE-0D coupling...')
                sys.stdout.flush()

            self.dcqd_form = []

            for i in range(self.pbf0.num_coupling_surf):
                self.dcqd_form.append(fem.form(self.pbf0.cq_factor[i]*self.dcqd[i]))

            tee = time.time() - tes
            if self.comm.rank == 0:
                print('FEM form compilation for ALE-0D finished, te = %.2f s' % (tee))
                sys.stdout.flush()


    def set_problem_vector_matrix_structures(self):

        super().set_problem_vector_matrix_structures()
        self.pbf0.set_problem_vector_matrix_structures_coupling()

        if self.coupling_strategy=='monolithic':

            self.k_sd_vec = []
            for i in range(len(self.pbf0.row_ids)):
                self.k_sd_vec.append(fem.petsc.create_vector(self.dcqd_form[i]))

            # setup offdiagonal matrix
            locmatsize = self.pba.V_d.dofmap.index_map.size_local * self.pba.V_d.dofmap.index_map_bs
            matsize = self.pba.V_d.dofmap.index_map.size_global * self.pba.V_d.dofmap.index_map_bs

            # derivative of 0D residual w.r.t. ALE displacements
            self.K_sd = PETSc.Mat().createAIJ(size=((self.pbf0.num_coupling_surf),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.comm)
            self.K_sd.setUp()


    def assemble_residual(self, t, subsolver=None):

        self.evaluate_residual_dbc_coupling()

        self.pbf0.assemble_residual(t, subsolver=subsolver)
        self.pba.assemble_residual(t)

        # fluid
        self.r_list[0] = self.pbf0.r_list[0]
        self.r_list[1] = self.pbf0.r_list[1]
        # flow0d
        self.r_list[2] = self.pbf0.r_list[2]
        # ALE
        self.r_list[3] = self.pba.r_list[0]


    def assemble_stiffness(self, t, subsolver=None):

        # if self.have_dbc_fluid_ale:
            # self.K_list[3][0] = self.K_dv
        if self.have_weak_dirichlet_fluid_ale:
            self.K_dv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_dv, self.jac_dv, self.pba.bc.dbcs)
            self.K_dv.assemble()
            self.K_list[3][0] = self.K_dv

        self.pbf0.assemble_stiffness(t, subsolver=subsolver)
        self.pba.assemble_stiffness(t)

        self.K_list[0][0] = self.pbf0.K_list[0][0]
        self.K_list[0][1] = self.pbf0.K_list[0][1]
        self.K_list[0][2] = self.pbf0.K_list[0][2]

        self.K_list[1][0] = self.pbf0.K_list[1][0]
        self.K_list[1][1] = self.pbf0.K_list[1][1]
        self.K_list[1][2] = self.pbf0.K_list[1][2]

        self.K_list[2][0] = self.pbf0.K_list[2][0]
        self.K_list[2][1] = self.pbf0.K_list[2][1]
        self.K_list[2][2] = self.pbf0.K_list[2][2]

        # derivative of fluid momentum w.r.t. ALE displacement
        self.K_vd.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vd, self.jac_vd, self.pbf.bc.dbcs)
        self.K_vd.assemble()
        self.K_list[0][3] = self.K_vd

        # derivative of fluid continuity w.r.t. ALE velocity
        self.K_pd.zeroEntries()
        if self.pbf.num_dupl > 1:
            fem.petsc.assemble_matrix_block(self.K_pd, self.jac_pd_, [])
        else:
            fem.petsc.assemble_matrix(self.K_pd, self.jac_pd, [])
        self.K_pd.assemble()
        self.K_list[1][3] = self.K_pd

        # offdiagonal s-d rows: derivative of flux constraint w.r.t. ALE displacement (in case of moving coupling boundaries)
        for i in range(len(self.pbf0.row_ids)):
            with self.k_sd_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_sd_vec[i], self.dcqd_form[i])
            self.k_sd_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        irs, ire = self.pbf.K_list[0][0].getOwnershipRange()

        # set rows
        for i in range(len(self.pbf0.row_ids)):
            self.K_sd[self.pbf0.row_ids[i], irs:ire] = self.k_sd_vec[i][irs:ire]

        self.K_sd.assemble()

        if bool(self.residual_scale):
            self.K_sd.scale(self.residual_scale[2])

        self.K_list[2][3] = self.K_sd
        self.K_list[3][3] = self.pba.K_list[0][0]


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.vector.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.vector.getLocalSize()

        offset_v = vvec_or0 + self.pbf.p.vector.getOwnershipRange()[0] + self.pbf0.lm.getOwnershipRange()[0] + self.pba.d.vector.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.pbf.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_s = offset_p + self.pbf.p.vector.getLocalSize()
        iset_s = PETSc.IS().createStride(self.pbf0.lm.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        offset_d = offset_s + self.pbf0.lm.getLocalSize()
        iset_d = PETSc.IS().createStride(self.pba.d.vector.getLocalSize(), first=offset_d, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_s = iset_s.expand(iset_r) # add to 0D block
            iset_s.sort() # should be sorted, otherwise PETSc may struggle to extract block

        if isoptions['ale_to_v']:
            iset_v = iset_v.expand(iset_d) # add ALE to velocity block

        if isoptions['lms_to_p']:
            iset_p = iset_p.expand(iset_s) # add to pressure block - attention: will merge ROM to this block too in case of 'rom_to_new' is True!
            ilist = [iset_v, iset_p, iset_d]
        elif isoptions['lms_to_v']:
            iset_v = iset_v.expand(iset_s) # add to velocity block (could be bad...) - attention: will merge ROM to this block too in case of 'rom_to_new' is True!
            ilist = [iset_v, iset_p, iset_d]
        else:
            ilist = [iset_v, iset_p, iset_s, iset_d]

        if isoptions['ale_to_v']: ilist.pop(-1)

        return ilist


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # fluid-ALE + flow0d problem
        if self.restart_step > 0:
            self.io.readcheckpoint(self, N)
            self.simname += '_r'+str(N)
            # TODO: quick-fix - simname variables of single field problems need to be addressed, too
            # but this should be handled by one variable, however neeeds revamp of I/O
            self.pbf.simname += '_r'+str(N)
            self.pba.simname += '_r'+str(N)

        self.pb0.read_restart(sname, N)

        if self.restart_step > 0:
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.pbf0.lm)
            self.pb0.cardvasc0D.read_restart(self.pb0.output_path_0D, sname+'_lm', N, self.pbf0.lm_old)


    def evaluate_initial(self):

        self.pbf0.evaluate_initial()
        self.pba.evaluate_initial()


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):

        return (self.pb0.ti.cycle[0]-1) * self.pb0.cardvasc0D.T_cycl * self.noperiodicref # zero if T_cycl variable is not specified


    def evaluate_pre_solve(self, t, N):

        self.pbf0.evaluate_pre_solve(t, N)
        self.pba.evaluate_pre_solve(t, N)


    def evaluate_post_solve(self, t, N):

        self.pbf0.evaluate_post_solve(t, N)
        self.pba.evaluate_post_solve(t, N)


    def set_output_state(self, N):

        self.pbf0.set_output_state(N)
        self.pba.set_output_state(N)


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t) # combined fluid-ALE output routine
        self.pb0.write_output(N, t)


    def update(self):

        # update time step - fluid+flow0d and ALE
        self.pbf0.update()
        self.pba.update()


    def print_to_screen(self):

        self.pbf0.print_to_screen()
        self.pba.print_to_screen()


    def induce_state_change(self):

        self.pbf0.induce_state_change()
        self.pba.induce_state_change()


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)

        self.pb0.write_restart(sname, N)

        if self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0:
            self.pb0.cardvasc0D.write_restart(self.pb0.output_path_0D, sname+'_lm', N, self.pbf0.lm)


    def check_abort(self, t):

        self.pbf0.check_abort(t)
        self.pba.check_abort(t)


    def destroy(self):

        self.pbf0.destroy()
        self.pba.destroy()

        if self.coupling_strategy=='monolithic':
            for i in range(len(self.pbf0.row_ids)): self.k_sd_vec[i].destroy()



class FluidmechanicsAleFlow0DSolver(solver_base):

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
        if self.pb.coupling_strategy=='monolithic':
            self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params, subsolver=self.subsol)
        elif self.pb.coupling_strategy=='partitioned':
            self.solnln = solver_nonlin.solver_nonlinear([self.pb.pbf0,self.pb.pba], self.solver_params, subsolver=self.subsol, cp=self.pb)
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
            self.solverprestr = FluidmechanicsSolverPrestr(self.pb.pbf, solver_params_prestr)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the 3D-0D problem
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

            self.pb.assemble_residual(self.pb.t_init, subsolver=None) # note: subsolver only passed to stiffness eval to get correct sparsity pattern)
            self.pb.assemble_stiffness(self.pb.t_init, subsolver=subsolver)

            # create ROM matrix structures
            if self.pb.rom:
                self.pb.rom.set_reduced_data_structures_residual(self.pb.r_list, self.pb.r_list_rom)
                self.pb.K_list_tmp = [[None]]
                self.pb.rom.set_reduced_data_structures_matrix(self.pb.K_list, self.pb.K_list_rom, self.pb.K_list_tmp)

        elif self.pb.coupling_strategy=='partitioned':

            self.pb.pbf0.rom = self.pb.rom

            self.pb.assemble_residual(self.pb.t_init, subsolver=None)
            self.pb.pbf0.assemble_stiffness(self.pb.t_init, subsolver=subsolver)
            self.pb.pba.assemble_stiffness(self.pb.t_init)

            # create ROM matrix structures
            if self.pb.pbf0.rom:
                self.pb.pbf0.rom.set_reduced_data_structures_residual(self.pb.pbf0.r_list, self.pb.pbf0.r_list_rom)
                self.pb.pbf0.K_list_tmp = [[None]]
                self.pb.pbf0.rom.set_reduced_data_structures_matrix(self.pb.pbf0.K_list, self.pb.pbf0.K_list_rom, self.pb.pbf0.K_list_tmp)

        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pb0.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbf.numstep, ni=ni, li=li, wt=wt)
