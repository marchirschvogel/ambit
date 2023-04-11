#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time
import utilities

class problem_base():

    def __init__(self, io_params, time_params, comm):

        self.comm = comm

        self.problem_type = io_params['problem_type']

        try: self.timint = time_params['timint']
        except: self.timint = 'static'

        if 'maxtime' in time_params.keys(): self.maxtime = time_params['maxtime']
        if 'numstep' in time_params.keys(): self.numstep = time_params['numstep']
        if 'maxtime' in time_params.keys(): self.dt = self.maxtime/self.numstep

        try: self.restart_step = io_params['restart_step']
        except: self.restart_step = 0

        try: self.numstep_stop = time_params['numstep_stop']
        except: self.numstep_stop = self.numstep

        self.t_init = self.restart_step * self.dt


    # routines that should be implemented by derived model problem
    def pre_timestep_routines(self):
        raise RuntimeError("Problem misses function implementation!")


    def read_restart(self):
        raise RuntimeError("Problem misses function implementation!")


    def evaluate_initial(self):
        raise RuntimeError("Problem misses function implementation!")


    def write_output_ini(self):
        raise RuntimeError("Problem misses function implementation!")


    def get_time_offset(self):
        raise RuntimeError("Problem misses function implementation!")


    def evaluate_pre_solve(self, t):
        raise RuntimeError("Problem misses function implementation!")


    def evaluate_post_solve(self, t, N):
        raise RuntimeError("Problem misses function implementation!")


    def set_output_state(self):
        raise RuntimeError("Problem misses function implementation!")


    def write_output(self, N, t):
        raise RuntimeError("Problem misses function implementation!")


    def update(self):
        raise RuntimeError("Problem misses function implementation!")


    def print_to_screen(self):
        raise RuntimeError("Problem misses function implementation!")


    def induce_state_change(self):
        raise RuntimeError("Problem misses function implementation!")


    def write_restart(self, sname, N):
        raise RuntimeError("Problem misses function implementation!")


    def check_abort(self, t):
        raise RuntimeError("Problem misses function implementation!")



class solver_base():

    def __init__(self, problem, solver_params):

        self.pb = problem

        self.solver_params = solver_params

        self.initialize_nonlinear_solver()


    # routines that should be implemented by derived model solver
    def solve_initial_state(self):
        raise RuntimeError("Problem solver misses function implementation!")


    def print_timestep_info(self):
        raise RuntimeError("Problem solver misses function implementation!")


    def solve_nonlinear_problem(self, t):
        raise RuntimeError("Problem solver misses function implementation!")


    def solve_problem(self):

        start = time.time()

        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.comm, self.pb.numdof)

        # anything that should be performed before the time loop (e.g. model reduction offline phase)
        self.pb.pre_timestep_routines()

        # read restart information if requested
        self.pb.read_restart(self.pb.simname, self.pb.restart_step)

        # evaluate old initial state of model
        self.pb.evaluate_initial()

        # any pre-solve that has to be done (e.g. prestress or consistent initial accelerations)
        self.solve_initial_state()

        # any output that needs to be written initially (e.g. mesh data)
        self.pb.write_output_ini()

        # ambit main time loop
        for N in range(self.pb.restart_step+1, self.pb.numstep_stop+1):

            wts = time.time()

            # current time
            t = N * self.pb.dt

            # offset time (for cyclic problems)
            t_off = self.pb.get_time_offset()

            # evaluate any (solution-independent) time curves or other functions
            self.pb.evaluate_pre_solve(t-t_off)

            # solve the nonlinear problem
            self.solve_nonlinear_problem(t-t_off)

            # any post-solve evaluates
            self.pb.evaluate_post_solve(t-t_off, N)

            # anything that has to be set prior to writing output
            self.pb.set_output_state()

            # write output of solutions
            self.pb.write_output(N, t)

            # update time step - old and new quantities are the same at this point
            self.pb.update()

            # print any post-solution state to screen
            self.pb.print_to_screen()

            # solution time for time step
            wte = time.time()
            wt = wte - wts

            # print timestep info
            self.print_timestep_info(N, t, wt)

            # apply any "on-the-fly" changes of model state/parameters
            self.pb.induce_state_change()

            # write restart information if desired
            self.pb.write_restart(self.pb.simname, N)

            # check any abort criterion
            if self.pb.check_abort(t-t_off):
                break

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()
