#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import utilities
import expression
from projection import project
from mpiroutines import allgather_vec

from solid_flow0d import SolidmechanicsFlow0DSolver


class SolidmechanicsFlow0DPeriodicRefSolver():

    def __init__(self, problem, solver_params_solid, solver_params_flow0d):
    
        self.pb = problem
        
        # set indicator to zero (no temporal offsets)
        self.pb.noperiodicref = 0
        
        # initialize solver instances
        self.solver = SolidmechanicsFlow0DSolver(self.pb, solver_params_solid, solver_params_flow0d)
        
        # store prestress flag (because flag is set to False after one prestress run)
        self.prestress_initial = self.pb.pbs.prestress_initial

        # read restart information
        if self.pb.restart_periodicref > 0:
            self.pb.pbs.io.readcheckpoint(self.pb.pbs, self.pb.restart_periodicref)
            self.pb.pbf.readrestart(self.pb.pbs.simname + str(self.pb.restart_periodicref), self.pb.restart_periodicref)


    def solve_problem(self):
        
        start = time.time()
        
        # outer heart cycle main time loop
        for N in range(self.pb.restart_periodicref+1, self.pb.Nmax_periodicref+1):

            wts = time.time()

            # change output names
            self.pb.pbs.simname += str(N)

            self.reset_state_initial()

            # solve one heart cycle
            self.solver.solve_problem()
            
            # set prestress and re-initialize solid petsc solver
            if self.prestress_initial:
                self.pb.pbs.prestress_initial = self.prestress_initial
                self.solver.solverprestr.solnln.initialize_petsc_solver()
            
            # check if below tolerance
            if abs(self.pb.pbf.ti.cycleerror[0]) <= self.pb.pbf.eps_periodic:
                break

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for full computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()


    def reset_state_initial(self):
        # set state to zero
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

        # 0D variables s and s_old are already correctly set from the previous run (end values) and should serve as new initial conditions
