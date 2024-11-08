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
from ..mpiroutines import allgather_vec

from .fluid_ale_main import FluidmechanicsAleProblem
from .fluid_constraint_main import FluidmechanicsConstraintProblem
from ..fluid.fluid_main import FluidmechanicsSolverPrestr
from ..ale.ale_main import AleProblem
from ..base import problem_base, solver_base


class FluidmechanicsAleConstraintProblem(FluidmechanicsAleProblem,problem_base):

    def __init__(self, pbase, io_params, time_params, fem_params_fluid, fem_params_ale, constitutive_models_fluid, constitutive_models_ale, bc_dict_fluid, bc_dict_ale, time_curves, coupling_params_fluid_ale, coupling_params_fluid_constr, io, mor_params={}):

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_coupling_fluid_ale(coupling_params_fluid_ale)

        self.problem_physics = 'fluid_ale_constraint'

        try: self.coupling_fluid_ale = coupling_params_fluid_ale['coupling_fluid_ale']
        except: self.coupling_fluid_ale = {}

        try: self.coupling_ale_fluid = coupling_params_fluid_ale['coupling_ale_fluid']
        except: self.coupling_ale_fluid = {}

        try: self.coupling_strategy = coupling_params_fluid_ale['coupling_strategy']
        except: self.coupling_strategy = 'monolithic'

        self.have_dbc_fluid_ale, self.have_weak_dirichlet_fluid_ale, self.have_dbc_ale_fluid, self.have_robin_ale_fluid = False, False, False, False

        # initialize problem instances (also sets the variational forms for the fluid flow0d problem)
        self.pba  = AleProblem(pbase, io_params, time_params, fem_params_ale, constitutive_models_ale, bc_dict_ale, time_curves, io, mor_params=mor_params)
        # ALE variables that are handed to fluid problem
        alevariables = {'Fale' : self.pba.ki.F(self.pba.d), 'Fale_old' : self.pba.ki.F(self.pba.d_old), 'w' : self.pba.wel, 'w_old' : self.pba.w_old}
        self.pbfc = FluidmechanicsConstraintProblem(pbase, io_params, time_params, fem_params_fluid, constitutive_models_fluid, bc_dict_fluid, time_curves, coupling_params_fluid_constr, io, mor_params=mor_params, alevar=alevariables)

        self.pbf = self.pbfc.pbf

        self.pbrom = self.pbf # ROM problem can only be fluid
        self.pbrom_host = self

        # modify results to write...
        self.pbf.results_to_write = io_params['results_to_write'][0]
        self.pba.results_to_write = io_params['results_to_write'][1]

        self.sub_solve = False
        self.print_subiter = False
        self.have_condensed_variables = False

        self.io = io

        # NOTE: Fluid and ALE function spaces should be of the same type, but are different objects.
        # For some reason, when applying a function from one funtion space as DBC to another function space,
        # errors occur. Therefore, we define these auxiliary variables and interpolate respectively...

        # fluid displacement, but defined within ALE function space
        self.ufa = fem.Function(self.pba.V_d)
        # ALE velocity, but defined within fluid function space
        self.wf = fem.Function(self.pbf.V_v)

        self.set_variational_forms()

        if self.coupling_strategy == 'monolithic':
            self.numdof = self.pbf.numdof + self.pbfc.LM.getSize() + self.pba.numdof
        else:
            self.numdof = [self.pbf.numdof + self.pbfc.LM.getSize(), self.pba.numdof]

        self.localsolve = False

        self.io = self.pbf.io

        # number of fields involved
        self.nfields = 4

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)], [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.pbfc.pbf.num_dupl > 1: is_ghosted = [1, 2, 0, 1]
        else:                          is_ghosted = [1, 1, 0, 1]
        return [self.pbf.v.x.petsc_vec, self.pbf.p.x.petsc_vec, self.pbfc.LM, self.pba.d.x.petsc_vec], is_ghosted


    def set_variational_forms(self):
        super().set_variational_forms()

        self.dcqd = []
        for n in range(self.pbfc.num_coupling_surf):
            self.dcqd.append(ufl.derivative(self.pbfc.cq[n], self.pba.d, self.pba.dd))


    def set_problem_residual_jacobian_forms(self, pre=False):

        super().set_problem_residual_jacobian_forms(pre=pre)
        self.pbfc.set_problem_residual_jacobian_forms_coupling()

        if self.coupling_strategy=='monolithic':

            ts = time.time()
            utilities.print_status("FEM form compilation for ALE-constraint coupling...", self.comm, e=" ")

            self.dcqd_form = []

            for i in range(self.pbfc.num_coupling_surf):
                if self.pbfc.on_subdomain[i]:
                    # entity map child to parent
                    em_u = {self.io.mesh : self.pbf.io.submshes_emap[self.pbfc.coupling_params['constraint_physics'][i]['domain']][1]}
                else:
                    em_u = self.pbf.io.entity_maps
                self.dcqd_form.append(fem.form(self.dcqd[i], entity_maps=em_u))

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.comm)


    def set_problem_vector_matrix_structures(self):

        super().set_problem_vector_matrix_structures()
        self.pbfc.set_problem_vector_matrix_structures_coupling()

        if self.coupling_strategy=='monolithic':

            # setup offdiagonal matrix
            locmatsize = self.pba.V_d.dofmap.index_map.size_local * self.pba.V_d.dofmap.index_map_bs
            matsize = self.pba.V_d.dofmap.index_map.size_global * self.pba.V_d.dofmap.index_map_bs

            self.k_sd_vec = []
            for i in range(len(self.pbfc.row_ids)):
                self.k_sd_vec.append(fem.petsc.create_vector(self.dcqd_form[i]))

            self.dofs_coupling_vq = [[]]*self.pbfc.num_coupling_surf

            self.k_sd_subvec, sze_sd = [], []

            for n in range(self.pbfc.num_coupling_surf):

                nds_c_local = fem.locate_dofs_topological(self.pba.V_d, self.pba.io.mesh.topology.dim-1, self.pba.io.mt_b1.indices[np.isin(self.pba.io.mt_b1.values, self.pbfc.surface_vq_ids[n])])
                nds_c = np.array( self.pbf.V_v.dofmap.index_map.local_to_global(np.asarray(nds_c_local, dtype=np.int32)), dtype=np.int32 )
                self.dofs_coupling_vq[n] = PETSc.IS().createBlock(self.pba.V_d.dofmap.index_map_bs, nds_c, comm=self.comm)

                self.k_sd_subvec.append( self.k_sd_vec[n].getSubVector(self.dofs_coupling_vq[n]) )

                sze_sd.append(self.k_sd_subvec[-1].getSize())

            # derivative of multiplier constraint w.r.t. fluid velocities
            self.K_sd = PETSc.Mat().createAIJ(size=((PETSc.DECIDE,self.pbfc.num_coupling_surf),(locmatsize,matsize)), bsize=None, nnz=max(sze_sd), csr=None, comm=self.comm)
            self.K_sd.setUp()
            self.K_sd.setOption(PETSc.Mat.Option.ROW_ORIENTED, False)



    def assemble_residual(self, t, subsolver=None):

        self.evaluate_residual_dbc_coupling()

        self.pbfc.assemble_residual(t)
        self.pba.assemble_residual(t)

        # fluid
        self.r_list[0] = self.pbfc.r_list[0]
        self.r_list[1] = self.pbfc.r_list[1]
        # flow0d
        self.r_list[2] = self.pbfc.r_list[2]
        # ALE
        self.r_list[3] = self.pba.r_list[0]


    def assemble_stiffness(self, t, subsolver=None):

        if self.have_weak_dirichlet_fluid_ale:
            self.K_dv.zeroEntries()
            fem.petsc.assemble_matrix(self.K_dv, self.jac_dv, self.pba.bc.dbcs)
            self.K_dv.assemble()
        elif self.have_dbc_fluid_ale:
            self.K_dv_.zeroEntries()
            fem.petsc.assemble_matrix(self.K_dv_, self.pba.jac_dd, self.pba.bc.dbcs_nofluid) # need DBCs w/o fluid here
            self.K_dv_.assemble()
            # multiply to get the relevant columns only
            self.K_dv_.matMult(self.Diag_ale, result=self.K_dv)
            # zero rows where DBC is applied and set diagonal entry to -1
            self.K_dv.zeroRows(self.fdofs, diag=-1.)
            # we apply u_fluid to ALE, hence get du_fluid/dv
            fac = self.pbf.ti.get_factor_deriv_varint(self.pbase.dt)
            self.K_dv.scale(fac)

        self.K_list[3][0] = self.K_dv

        self.pbfc.assemble_stiffness(t)
        self.pba.assemble_stiffness(t)

        self.K_list[0][0] = self.pbfc.K_list[0][0]
        self.K_list[0][1] = self.pbfc.K_list[0][1]
        self.K_list[0][2] = self.pbfc.K_list[0][2]

        self.K_list[1][0] = self.pbfc.K_list[1][0]
        self.K_list[1][1] = self.pbfc.K_list[1][1]
        self.K_list[1][2] = self.pbfc.K_list[1][2]

        self.K_list[2][0] = self.pbfc.K_list[2][0]
        self.K_list[2][1] = self.pbfc.K_list[2][1]
        self.K_list[2][2] = self.pbfc.K_list[2][2]

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
        for i in range(len(self.pbfc.row_ids)):
            with self.k_sd_vec[i].localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.k_sd_vec[i], self.dcqd_form[i])
            self.k_sd_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            if self.pbfc.have_regularization:
                self.k_sd_vec[i].scale(self.pbfc.alpha_reg[i])

        # set rows
        for i in range(len(self.pbfc.row_ids)):
            # NOTE: only set the surface-subset of the k_sd vector entries to avoid placing unnecessary zeros!
            self.k_sd_vec[i].getSubVector(self.dofs_coupling_vq[i], subvec=self.k_sd_subvec[i])
            self.K_sd.setValues(self.pbfc.row_ids[i], self.dofs_coupling_vq[i], self.k_sd_subvec[i].array, addv=PETSc.InsertMode.INSERT)
            self.k_sd_vec[i].restoreSubVector(self.dofs_coupling_vq[i], subvec=self.k_sd_subvec[i])

        self.K_sd.assemble()

        self.K_list[2][3] = self.K_sd
        self.K_list[3][3] = self.pba.K_list[0][0]


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.pbf.v.x.petsc_vec.getOwnershipRange()[0]
            vvec_ls = self.pbf.v.x.petsc_vec.getLocalSize()

        offset_v = vvec_or0 + self.pbf.p.x.petsc_vec.getOwnershipRange()[0] + self.pbfc.LM.getOwnershipRange()[0] + self.pba.d.x.petsc_vec.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.pbf.p.x.petsc_vec.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        offset_s = offset_p + self.pbf.p.x.petsc_vec.getLocalSize()
        iset_s = PETSc.IS().createStride(self.pbfc.LM.getLocalSize(), first=offset_s, step=1, comm=self.comm)

        offset_d = offset_s + self.pbfc.LM.getLocalSize()
        iset_d = PETSc.IS().createStride(self.pba.d.x.petsc_vec.getLocalSize(), first=offset_d, step=1, comm=self.comm)

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
        if N > 0:
            self.io.readcheckpoint(self, N)

        # LM data
        if N > 0:
            restart_data = np.loadtxt(self.pbf.io.output_path+'/checkpoint_'+sname+'_lm_'+str(N)+'.txt', ndmin=1)
            self.pbfc.LM[:], self.pbfc.LM_old[:] = restart_data[:], restart_data[:]


    def evaluate_initial(self):

        self.pbfc.evaluate_initial()
        self.pba.evaluate_initial()


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def write_output_pre(self):

        self.pbfc.write_output_pre()
        self.pba.write_output_pre()


    def evaluate_pre_solve(self, t, N, dt):

        self.pbfc.evaluate_pre_solve(t, N, dt)
        self.pba.evaluate_pre_solve(t, N, dt)


    def evaluate_post_solve(self, t, N):

        self.pbfc.evaluate_post_solve(t, N)
        self.pba.evaluate_post_solve(t, N)


    def set_output_state(self, N):

        self.pbfc.set_output_state(N)
        self.pba.set_output_state(N)


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t) # combined fluid-ALE output routine

        if self.pbf.io.write_results_every > 0 and N % self.pbf.io.write_results_every == 0:
            if np.isclose(t,self.pbase.dt): mode = 'wt'
            else: mode = 'a'
            LM_sq = allgather_vec(self.pbfc.LM, self.comm)
            if self.comm.rank == 0:
                for i in range(len(LM_sq)):
                    f = open(self.pbase.output_path+'/results_'+self.pbase.simname+'_LM'+str(i+1)+'.txt', mode)
                    f.write('%.16E %.16E\n' % (t,LM_sq[i]))
                    f.close()
            del LM_sq


    def update(self):

        # update time step - fluid+flow0d and ALE
        self.pbfc.update()
        self.pba.update()


    def print_to_screen(self):

        self.pbfc.print_to_screen()
        self.pba.print_to_screen()


    def induce_state_change(self):

        self.pbfc.induce_state_change()
        self.pba.induce_state_change()


    def write_restart(self, sname, N, force=False):

        self.io.write_restart(self, N, force=force)

        if (self.pbf.io.write_restart_every > 0 and N % self.pbf.io.write_restart_every == 0) or force:
            LM_sq = allgather_vec(self.pbfc.LM, self.comm)
            if self.comm.rank == 0:
                f = open(self.pbf.io.output_path+'/checkpoint_'+sname+'_lm_'+str(N)+'.txt', 'wt')
                for i in range(len(LM_sq)):
                    f.write('%.16E\n' % (LM_sq[i]))
                f.close()
            del LM_sq


    def check_abort(self, t):

        return self.pbfc.check_abort(t)


    def destroy(self):

        self.pbfc.destroy()
        self.pba.destroy()

        if self.coupling_strategy=='monolithic':
            for i in range(len(self.pbfc.row_ids)): self.k_sd_vec[i].destroy()



class FluidmechanicsAleConstraintSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pbf.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        if self.pb.coupling_strategy=='monolithic':
            self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)
        elif self.pb.coupling_strategy=='partitioned':
            self.solnln = solver_nonlin.solver_nonlinear([self.pb.pbfc,self.pb.pba], self.solver_params, cp=self.pb)
        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")

        if self.pb.pbf.prestress_initial or self.pb.pbf.prestress_initial_only:
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
        if self.pb.pbf.pre:
            # solve reduced-solid/FrSI prestress problem
            self.solverprestr.solve_initial_prestress()

        # consider consistent initial acceleration
        if (self.pb.pbf.fluid_governing_type == 'navierstokes_transient' or self.pb.pbf.fluid_governing_type == 'stokes_transient') and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.pbf.deltaW_kin_old + self.pb.pbf.deltaW_int_old - self.pb.pbf.deltaW_ext_old - self.pb.pbfc.power_coupling_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.pbf.a_old, self.pb.pbf.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.io.entity_maps)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.pbf.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)


    # we overload this function here in order to take care of the partitioned solve,
    # where the ROM needs to be an object of the fluid, not the coupled problem
    def evaluate_assemble_system_initial(self, subsolver=None):

        # evaluate old initial state of model
        self.evaluate_system_initial()

        if self.pb.coupling_strategy=='monolithic':

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
                    self.pb.pbf.rom.set_reduced_data_structures_matrix(self.pb.pbf.K_list, self.pb.pbf.K_list_rom, self.pb.pbf.K_list_tmp)

        elif self.pb.coupling_strategy=='partitioned':

            self.pb.pbfc.rom = self.pb.rom
            self.pb.pbrom_host = self.pb.pbfc # overridden

            self.pb.assemble_residual(self.pb.pbase.t_init)
            self.pb.pbfc.assemble_stiffness(self.pb.pbase.t_init)
            self.pb.pba.assemble_stiffness(self.pb.pbase.t_init)

            # create ROM matrix structures
            if self.pb.pbfc.rom:
                self.pb.pbfc.rom.set_reduced_data_structures_residual(self.pb.pbfc.r_list, self.pb.pbfc.r_list_rom)
                self.pb.pbfc.K_list_tmp = [[None]]
                self.pb.pbfc.rom.set_reduced_data_structures_matrix(self.pb.pbfc.K_list, self.pb.pbfc.K_list_rom, self.pb.pbfc.K_list_tmp)

        else:
            raise ValueError("Unknown fluid-ALE coupling strategy! Choose either 'monolithic' or 'partitioned'.")


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.pbf.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)
