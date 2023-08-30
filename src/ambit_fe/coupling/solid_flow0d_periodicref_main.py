#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
from petsc4py import PETSc

from ..solid_flow0d import SolidmechanicsFlow0DSolver


class SolidmechanicsFlow0DPeriodicRefSolver():

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):

        self.pb = problem

        # set indicator to zero (no temporal offsets)
        self.pb.noperiodicref = 0

        # initialize solver instance
        self.solver = SolidmechanicsFlow0DSolver(self.pb, solver_params_solid, solver_params_flow0d)

        # store prestress flag (because flag is set to False after one prestress run)
        self.prestress_initial = self.pb.pbs.prestress_initial

        # store simname
        self.simname = self.pb.pbs.simname

        # read restart information
        if self.pb.restart_periodicref > 0:
            self.pb.pbs.simname = self.simname + str(self.pb.restart_periodicref)
            self.pb.readrestart(self.pb.pbs.simname, self.pb.restart_periodicref)


    def solve_problem(self):

        start = time.time()

        # outer heart cycle loop
        for N in range(self.pb.restart_periodicref+1, self.pb.Nmax_periodicref+1):

            wts = time.time()

            # change output names
            self.pb.pbs.simname = self.simname + str(N)

            self.reset_state_initial()

            # solve one heart cycle
            self.solver.solve_problem()

            # set prestress and re-initialize solid petsc solver
            if self.prestress_initial:
                self.pb.pbs.prestress_initial = True
                self.solver.solverprestr.solnln.initialize_petsc_solver()

            if self.pb.write_checkpoints_periodicref:
                self.pb.write_restart(self.pb.pbs.simname, N)

            # check if below tolerance
            if abs(self.pb.pbf.ti.cycleerror[0]) <= self.pb.pbf.eps_periodic:
                if self.pb.comm.rank == 0:
                    print("Periodicity on reference configuration reached after %i heart cycles with cycle error %.4f! Finished. :-)" % (self.pb.pbf.ti.cycle[0]-1,self.pb.pbf.ti.cycleerror[0]))
                    sys.stdout.flush()
                break

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for full computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()


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

        if self.prestress_initial:
            self.pb.pbs.u_pre.vector.set(0.0)
            self.pb.pbs.u_pre.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # reset internal variables
        for i in range(len(self.pb.pbs.internalvars)):
            list(self.pb.pbs.internalvars.values())[i].vector.set(0.0)
            list(self.pb.pbs.internalvars.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            list(self.pb.pbs.internalvars_old.values())[i].vector.set(0.0)
            list(self.pb.pbs.internalvars_old.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # 0D variables s and s_old are already correctly set from the previous run (end values) and should serve as new initial conditions
