#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
from petsc4py import PETSc

from .. import utilities
from ..mpiroutines import allgather_vec_entry
from .solid_flow0d_main import SolidmechanicsFlow0DSolver


class SolidmechanicsFlow0DPeriodicRefSolver():

    def __init__(self, problem, solver_params):

        self.pb = problem

        # set indicator to zero (no temporal offsets)
        self.pb.noperiodicref = 0

        if self.pb.pbase.restart_step > 0:
            self.pb.pbase.simname += str(self.pb.restart_periodicref+1)

        if self.pb.restart_periodicref > 0:
            raise RuntimeError("Outer restart of this problem currently broken!")

        # initialize solver instance
        self.solver = SolidmechanicsFlow0DSolver(self.pb, solver_params)

        # store simname
        self.simname = self.pb.pbase.simname

        # read restart information
        if self.pb.restart_periodicref > 0:
            self.pb.pbase.simname = self.simname + str(self.pb.restart_periodicref)
            self.pb.read_restart(self.pb.pbase.simname, self.pb.restart_periodicref)
            if self.pb.pbs.prestress_initial:
                self.set_prestress_state()


    def solve_problem(self):

        start = time.time()

        # outer heart cycle loop
        for N in range(self.pb.restart_periodicref+1, self.pb.Nmax_periodicref+1):

            wts = time.time()

            # change output name
            self.pb.pbase.simname = self.simname + str(N)

            # solve one heart cycle
            self.solver.time_loop()

            # for the next loop, set back
            self.pb.pbase.restart_step = 0

            # set back state
            self.reset_state_initial()

            # set prestress for next loop
            if self.pb.pbs.prestress_initial:
                self.set_prestress_state()

            if self.pb.write_checkpoints_periodicref:
                self.pb.write_restart(self.pb.pbase.simname, N, force=True)

            # check if below tolerance
            if abs(self.pb.pb0.ti.cycleerror[0]) <= self.pb.pb0.eps_periodic:
                break

        self.solver.destroy()


    def set_prestress_state(self):

        self.pb.pbs.pre = True

        for i, m in enumerate(self.pb.pbs.ti.funcsexpr_to_update_pre):

            # we need to have the class variable 'val' in our expression!
            assert('val' in dir(self.pb.pbs.ti.funcsexpr_to_update_pre[m]))

            if self.pb.coupling_type == 'monolithic_direct':
                self.pb.pbs.ti.funcsexpr_to_update_pre[m].val = allgather_vec_entry(self.pb.pb0.s_old, self.pb.pb0.cardvasc0D.v_ids[i], self.pb.comm)
            if self.pb.coupling_type == 'monolithic_lagrange':
                self.pb.pbs.ti.funcsexpr_to_update_pre[m].val = allgather_vec_entry(self.pb.LM_old, list(range(self.pb.num_coupling_surf))[i], self.pb.comm)

            m.interpolate(self.pb.pbs.ti.funcsexpr_to_update_pre[m].evaluate)
            m.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # we need to invoke the prestress forms here
        self.pb.pbs.set_problem_residual_jacobian_forms(pre=True)


    # set state to zero
    def reset_state_initial(self):

        self.pb.pbs.u.vector.set(0.0)
        self.pb.pbs.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.pbs.u_old.vector.set(0.0)
        self.pb.pbs.u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.pb.pbs.v_old.vector.set(0.0)
        self.pb.pbs.v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.pbs.a_old.vector.set(0.0)
        self.pb.pbs.a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if self.pb.pbs.incompressible_2field:
            self.pb.pbs.p.vector.set(0.0)
            self.pb.pbs.p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            self.pb.pbs.p_old.vector.set(0.0)
            self.pb.pbs.p_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if self.pb.pbs.prestress_initial:
            self.pb.pbs.u_pre.vector.set(0.0)
            self.pb.pbs.u_pre.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # reset internal variables
        for i in range(len(self.pb.pbs.internalvars)):
            list(self.pb.pbs.internalvars.values())[i].vector.set(0.0)
            list(self.pb.pbs.internalvars.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            list(self.pb.pbs.internalvars_old.values())[i].vector.set(0.0)
            list(self.pb.pbs.internalvars_old.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # 0D variables s and s_old are already correctly set from the previous run (end values) and should serve as new initial conditions
