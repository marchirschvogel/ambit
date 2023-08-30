#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

from . import utilities
from ..solid import SolidmechanicsProblem, SolidmechanicsSolver
from ..solid_flow0d import SolidmechanicsFlow0DProblem, SolidmechanicsFlow0DSolver


class SolidmechanicsFlow0DMultiscaleGrowthRemodelingProblem():

    def __init__(self, io_params, time_params_solid_small, time_params_solid_large, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, multiscale_params, io, comm=None):

        self.comm = comm

        self.problem_physics = 'solid_flow0d_multiscale_gandr'

        gandr_trigger_phase = multiscale_params['gandr_trigger_phase']

        self.N_cycles = multiscale_params['numcycles']

        try: self.write_checkpoints = multiscale_params['write_checkpoints']
        except: self.write_checkpoints = False

        try: self.restart_cycle = multiscale_params['restart_cycle']
        except: self.restart_cycle = 0

        try: self.restart_from_small = multiscale_params['restart_from_small']
        except: self.restart_from_small = False

        constitutive_models_large = copy.deepcopy(constitutive_models)

        # set growth for small dynamic scale
        for n in range(len(constitutive_models)):
            try:
                growth_trig = constitutive_models['MAT'+str(n+1)+'']['growth']['growth_trig']
                constitutive_models['MAT'+str(n+1)+'']['growth']['growth_trig'] = 'prescribed_multiscale'
                constitutive_models['MAT'+str(n+1)+'']['growth']['growth_settrig'] = growth_trig
            except:
                pass

        # remove any dynamics from large scale constitutive models dict
        for n in range(len(constitutive_models_large)):
            try:
                constitutive_models_large['MAT'+str(n+1)+''].pop('inertia')
                constitutive_models_large['MAT'+str(n+1)+''].pop('visco_green')
            except:
                pass

            # set active stress to prescribed on large scale
            try:
                constitutive_models_large['MAT'+str(n+1)+'']['active_fiber']['prescribed_multiscale'] = True
                constitutive_models_large['MAT'+str(n+1)+'']['active_iso']['prescribed_multiscale'] = True
            except:
                pass

        # we have to be quasi-static on the large scale!
        assert(time_params_solid_large['timint'] == 'static')

        # initialize problem instances
        self.pbsmall = SolidmechanicsFlow0DProblem(io_params, time_params_solid_small, time_params_flow0d, fem_params, constitutive_models, model_params_flow0d, bc_dict, time_curves, coupling_params, io, comm=self.comm)
        self.pblarge = SolidmechanicsProblem(io_params, time_params_solid_large, fem_params, constitutive_models_large, bc_dict, time_curves, io, comm=self.comm)

        # we must have a growth law in at least one material
        assert(self.pbsmall.pbs.have_growth)
        assert(self.pblarge.have_growth)

        self.tol_small = multiscale_params['tol_small']
        self.tol_large = multiscale_params['tol_large']
        self.tol_outer = multiscale_params['tol_outer']

        # override by tol_small
        self.pbsmall.pbf.eps_periodic = self.tol_small
        self.pblarge.tol_stop_large = self.tol_large

        # store to ensure prestressed state is kept throughout the whole cycle (small scale prestress_initial gets set to False after initial prestress)
        self.prestress_initial = self.pbsmall.pbs.prestress_initial
        # set large scale prestress to False (only u_pre is added on the large scale if we have prestress, but no extra prestressing phase is undergone)
        self.pblarge.prestress_initial = False

        self.simname_small = self.pbsmall.pbs.simname + '_small'
        self.simname_large = self.pblarge.simname + '_large'

        if gandr_trigger_phase == 'end_diastole':
            self.pbsmall.t_gandr_setpoint = self.pbsmall.pbf.cardvasc0D.t_ed
        elif gandr_trigger_phase == 'end_systole':
            self.pbsmall.t_gandr_setpoint = self.pbsmall.pbf.cardvasc0D.t_es
        else:
            raise NameError("Unknown growth multiscale_trigger_phase")

        self.set_variational_forms_and_jacobians()


    # defines the solid and monolithic coupling forms for 0D flow and solid mechanics
    def set_variational_forms_and_jacobians(self):

        # add constant Neumann terms for large scale problem (trigger pressures)
        self.neumann_funcs = []
        w_neumann = ufl.as_ufl(0)
        for n in range(len(self.pbsmall.surface_p_ids)):

            self.neumann_funcs.append(fem.Function(self.pblarge.Vd_scalar))

            for i in range(len(self.pbsmall.surface_p_ids[n])):

                ds_ = ufl.ds(subdomain_data=self.pblarge.io.mt_b1, subdomain_id=self.pbsmall.surface_p_ids[n][i], metadata={'quadrature_degree': self.pblarge.quad_degree})

                # we apply the pressure onto a fixed configuration of the G&R trigger point, determined by the displacement field u_set
                # in the last G&R cycle, we assure that growth falls below a tolerance and hence the current and the set configuration coincide
                w_neumann += self.pblarge.vf.deltaW_ext_neumann_normal_cur(self.pblarge.ki.J(self.pblarge.u_set,ext=True), self.pblarge.ki.F(self.pblarge.u_set,ext=True), self.neumann_funcs[-1], ds_)

        self.pblarge.weakform_u -= w_neumann
        # linearization not needed (only if we applied the trigger load on the current state)
        #self.pblarge.jac_uu -= ufl.derivative(w_neumann, self.pblarge.u, self.pblarge.du)



class SolidmechanicsFlow0DMultiscaleGrowthRemodelingSolver():

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):

        self.pb = problem

        # initialize solver instances
        self.solversmall = SolidmechanicsFlow0DSolver(self.pb.pbsmall, solver_params_solid, solver_params_flow0d)
        self.solverlarge = SolidmechanicsSolver(self.pb.pblarge, solver_params_solid)

        # read restart information
        if self.pb.restart_cycle > 0:
            self.pb.pbsmall.pbs.simname = self.pb.simname_small + str(self.pb.restart_cycle)
            self.pb.pblarge.simname = self.pb.simname_large + str(self.pb.restart_cycle)
            self.pb.pbsmall.pbs.io.readcheckpoint(self.pb.pbsmall.pbs, self.pb.restart_cycle)
            self.pb.pblarge.io.readcheckpoint(self.pb.pblarge, self.pb.restart_cycle)
            self.pb.pbsmall.pbf.readrestart(self.pb.pbsmall.pbs.simname, self.pb.restart_cycle)
            # no need to do after restart
            self.pb.pbsmall.pbs.prestress_initial = False
            # induce the perturbation
            self.pb.pbsmall.induce_perturbation()
            # next small scale run is a resumption of a previous one
            self.pb.pbsmall.restart_multiscale = True


    def solve_problem(self):

        start = time.time()

        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.comm)

        # multiscale growth and remodeling solid 0D flow main time loop
        for N in range(self.pb.restart_cycle+1, self.pb.N_cycles+1):

            wts = time.time()

            # time offset from previous small scale times
            self.pb.pbsmall.t_prev = (self.pb.pbsmall.pbf.ti.cycle[0]-1) * self.pb.pbsmall.pbf.cardvasc0D.T_cycl

            # change output names
            self.pb.pbsmall.pbs.simname = self.pb.simname_small + str(N)
            self.pb.pblarge.simname = self.pb.simname_large + str(N)

            self.set_state_small()

            if not self.pb.restart_from_small:

                if self.pb.comm.rank == 0:
                    print("Solving small scale 3D-0D coupled solid-flow0d problem:")
                    sys.stdout.flush()

                # solve small scale 3D-0D coupled solid-flow0d problem with fixed growth
                self.solversmall.solve_problem()

                if self.pb.write_checkpoints:
                    # write checkpoint for potential restarts
                    self.pb.pbsmall.pbs.io.writecheckpoint(self.pb.pbsmall.pbs, N)
                    self.pb.pbsmall.pbf.write_restart(self.pb.pbsmall.pbs.simname, N, ms=True)

            else:

                # read small scale checkpoint if we restart from this scale
                self.pb.pbsmall.pbs.io.readcheckpoint(self.pb.pbsmall.pbs, self.pb.restart_cycle+1)
                self.pb.pbsmall.pbf.readrestart(self.pb.pbsmall.pbs.simname, self.pb.restart_cycle+1, ms=True)
                # induce the perturbation
                if not self.pb.pbsmall.pbf.have_induced_pert: self.pb.pbsmall.induce_perturbation()
                # no need to do after restart
                self.pb.pbsmall.pbs.prestress_initial = False
                # set flag to False again
                self.pb.restart_from_small = False

            # next small scale run is a resumption of a previous one
            self.pb.pbsmall.restart_multiscale = True

            # set large scale state
            self.set_state_large(N)

            # compute volume prior to G&R
            vol_prior = self.compute_volume_large()

            if self.pb.comm.rank == 0:
                print("Solving large scale solid growth and remodeling problem:")
                sys.stdout.flush()

            # solve large scale static G&R solid problem with fixed loads
            self.solverlarge.solve_problem()

            # compute volume after G&R
            vol_after = self.compute_volume_large()

            if self.pb.write_checkpoints:
                # write checkpoint for potential restarts
                self.pb.pblarge.io.writecheckpoint(self.pb.pblarge, N)

            # relative volume increase over large scale run
            volchange = (vol_after - vol_prior)/vol_prior
            if self.pb.comm.rank == 0:
                print('Volume change due to growth: %.4e' % (volchange))
                sys.stdout.flush()

            # check if below tolerance
            if abs(volchange) <= self.pb.tol_outer:
                break

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for full multiscale computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()


    def set_state_small(self):

        # set delta small to large
        u_delta = PETSc.Vec().createMPI((self.pb.pblarge.u.vector.getLocalSize(),self.pb.pblarge.u.vector.getSize()), bsize=self.pb.pblarge.u.vector.getBlockSize(), comm=self.pb.comm)
        u_delta.waxpy(-1.0, self.pb.pbsmall.pbs.u_set.vector, self.pb.pblarge.u.vector)
        if self.pb.pbsmall.pbs.incompressible_2field:
            p_delta = PETSc.Vec().createMPI((self.pb.pblarge.p.vector.getLocalSize(),self.pb.pblarge.p.vector.getSize()), bsize=self.pb.pblarge.p.vector.getBlockSize(), comm=self.pb.comm)
            p_delta.waxpy(-1.0, self.pb.pbsmall.pbs.p_set.vector, self.pb.pblarge.p.vector)

        # update small scale variables - add delta from growth to last small scale displacement
        self.pb.pbsmall.pbs.u.vector.axpy(1.0, u_delta)
        self.pb.pbsmall.pbs.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.pbsmall.pbs.u_old.vector.axpy(1.0, u_delta)
        self.pb.pbsmall.pbs.u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if self.pb.pbsmall.pbs.incompressible_2field:
            self.pb.pbsmall.pbs.p.vector.axpy(1.0, p_delta)
            self.pb.pbsmall.pbs.p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            self.pb.pbsmall.pbs.p_old.vector.axpy(1.0, p_delta)
            self.pb.pbsmall.pbs.p_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # we come from a quasi-static simulation - old v and a from previous small scale run have to be set to zero
        self.pb.pbsmall.pbs.v_old.vector.set(0.0)
        self.pb.pbsmall.pbs.v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.pbsmall.pbs.a_old.vector.set(0.0)
        self.pb.pbsmall.pbs.a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # 0D variables s and s_old are already correctly set from the previous small scale run (end values)

        # set constant prescribed growth stretch for subsequent small scale
        self.pb.pbsmall.pbs.theta.vector.axpby(1.0, 0.0, self.pb.pblarge.theta.vector)
        self.pb.pbsmall.pbs.theta.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.pbsmall.pbs.theta_old.vector.axpby(1.0, 0.0, self.pb.pblarge.theta.vector)
        self.pb.pbsmall.pbs.theta_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def set_state_large(self, N):

        # update large scale variables
        # only needed once - set prestress displacement from small scale
        if self.pb.prestress_initial and N == 1:
            self.pb.pblarge.u_pre.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.u_pre.vector)
            self.pb.pblarge.u_pre.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.pb.pblarge.u_set.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.u_set.vector)
        self.pb.pblarge.u_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.pb.pblarge.u.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.u_set.vector)
        self.pb.pblarge.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if self.pb.pblarge.incompressible_2field:
            self.pb.pblarge.p.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.p_set.vector)
            self.pb.pblarge.p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # constant large scale active tension
        self.pb.pblarge.tau_a.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.tau_a_set.vector)
        self.pb.pblarge.tau_a.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if self.pb.pblarge.have_frank_starling:
            self.pb.pblarge.amp_old.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.amp_old_set.vector)
            self.pb.pblarge.amp_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # pressures from growth set point
        self.pb.pbsmall.pbf.cardvasc0D.set_pressure_fem(self.pb.pbsmall.pbf.s_set, self.pb.pbsmall.pbf.cardvasc0D.v_ids, self.pb.pbsmall.pr0D, self.pb.neumann_funcs)

        # growth thresholds from set point
        self.pb.pblarge.growth_thres.vector.axpby(1.0, 0.0, self.pb.pbsmall.pbs.growth_thres.vector)
        self.pb.pblarge.growth_thres.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def compute_volume_large(self):

        J_all = ufl.as_ufl(0)
        for n in range(self.pb.pblarge.num_domains):

            J_all += self.pb.pblarge.ki.J(self.pb.pblarge.u,ext=True) * self.pb.pblarge.dx_[n]

        vol = fem.assemble_scalar(J_all)
        vol = self.pb.comm.allgather(vol)
        volume_large = sum(vol)

        if self.pb.comm.rank == 0:
            print('Volume of myocardium: %.4e' % (volume_large))
            sys.stdout.flush()

        return volume_large
