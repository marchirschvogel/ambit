#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time
import numpy as np
import utilities

class problem_base():

    def __init__(self, io_params, time_params, comm):

        self.comm = comm

        self.problem_type = io_params['problem_type']

        self.simname = io_params['simname']

        try: self.timint = time_params['timint']
        except: self.timint = 'static'

        if 'maxtime' in time_params.keys(): self.maxtime = time_params['maxtime']
        if 'numstep' in time_params.keys(): self.numstep = time_params['numstep']
        if 'maxtime' in time_params.keys(): self.dt = self.maxtime/self.numstep

        try: self.restart_step = io_params['restart_step']
        except: self.restart_step = 0

        try: self.numstep_stop = time_params['numstep_stop']
        except: self.numstep_stop = self.numstep

        try: self.results_to_write = io_params['results_to_write']
        except: self.results_to_write = []

        try: self.residual_scale_dt = time_params['residual_scale_dt']
        except: self.residual_scale_dt = False

        self.print_subiter = False

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


    def evaluate_pre_solve(self, t, N):
        raise RuntimeError("Problem misses function implementation!")


    def evaluate_post_solve(self, t, N):
        raise RuntimeError("Problem misses function implementation!")


    def set_output_state(self, t):
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


    def scale_residual_list(self, rlist, fac):

        for i in range(len(rlist)):
            rlist[i].scale(fac)


    def scale_jacobian_list(self, Klist, fac):

        for i in range(len(Klist)):
            if Klist[i] is not None: Klist[i].scale(fac)


class solver_base():

    def __init__(self, problem, solver_params):

        self.pb = problem

        self.solver_params = solver_params

        self.initialize_nonlinear_solver()

        self.wt, self.ni, self.li = 0., 0, 0
        self.wt_, self.ni_, self.li_ = [], [], []


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
            self.pb.evaluate_pre_solve(t-t_off, N)

            # solve the nonlinear problem
            self.solve_nonlinear_problem(t-t_off)

            # any post-solve evaluates
            self.pb.evaluate_post_solve(t-t_off, N)

            # anything that has to be set prior to writing output
            self.pb.set_output_state(t)

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
            self.print_timestep_info(N, t, self.solnln.ni, self.solnln.li, wt)

            # apply any "on-the-fly" changes of model state/parameters
            self.pb.induce_state_change()

            # write restart information if desired
            self.pb.write_restart(self.pb.simname, N)

            # update nonlinear and linear iteration counters
            self.update_counters(wt, t)

            # check any abort criterion
            if self.pb.check_abort(t-t_off):
                break

        # destroy stuff
        self.destroy()

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for computation: %.4f s (= %.2f min)\n' % ( time.time()-start, (time.time()-start)/60. ))

            print('{:<55s}{:<1.8f}'.format('Total solution time of all steps: ',self.wt))
            print('{:<55s}{:<1.8f}'.format('Average solution time per time step: ',self.wt/N))
            print('{:<55s}{:<1.8f}'.format('Maximum solution time of a time step: ',max(self.wt_)))
            print(' ')
            print('{:<55s}{:<1d}'.format('Total number of nonlinear iterations: ',self.ni))
            print('{:<55s}{:<1.1f}'.format('Average number of nonlinear iterations per time step: ',self.ni/N))
            print('{:<55s}{:<1d}'.format('Maximum number of nonlinear iterations in a time step: ',max(self.ni_)))

            if self.solnln.solvetype=='iterative':
                print(' ')
                print('{:<55s}{:<1d}'.format('Total number of linear iterations: ',self.li))
                print('{:<55s}{:<1.1f}'.format('Average number of linear iterations per time step: ',self.li/N))
                print('{:<55s}{:<1d}'.format('Maximum number of linear iterations in a time step: ',max(self.li_)))
                print('{:<55s}{:<1.1f}'.format('Average number of linear iterations per solve: ',self.li/self.ni))
                print('{:<55s}{:<1d}'.format('Maximum number of linear iterations in a solve: ',max(self.solnln.li_s)))

            print(self.solnln.sepstring)

            sys.stdout.flush()


    def update_counters(self, wt, t):

        self.wt += wt
        self.ni += self.solnln.ni
        self.li += self.solnln.li

        # for determination of max values
        self.wt_.append(wt)
        self.ni_.append(self.solnln.ni)
        self.li_.append(self.solnln.li)

        # write file for counters if requested
        if 'counters' in self.pb.results_to_write:

            # mode: 'wt' generates new file, 'a' appends to existing one
            if np.isclose(t,self.pb.dt): mode = 'wt'
            else: mode = 'a'

            if self.pb.comm.rank == 0:

                f = open(self.pb.io.output_path+'/results_'+self.pb.simname+'_wt_ni_li_counters.txt', mode)
                f.write('%.16E %.16E %i %i\n' % (t, wt, self.solnln.ni, self.solnln.li))
                f.close()


    def destroy(self):

        self.solnln.ksp.destroy()
