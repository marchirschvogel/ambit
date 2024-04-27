#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time
import numpy as np
from . import utilities

"""
Ambit problem and solver base classes
"""

class problem_base():

    def __init__(self, io_params, time_params, comm=None, comm_sq=None):

        self.comm = comm
        self.comm_sq = comm_sq

        self.problem_type = io_params['problem_type']

        self.simname = io_params['simname']
        self.output_path = io_params['output_path']

        self.maxtime = time_params['maxtime']
        if 'numstep' in time_params.keys():
            assert('dt' not in time_params.keys())
            self.numstep = time_params['numstep']
            self.dt = self.maxtime/self.numstep
        elif 'dt' in time_params.keys():
            assert('numstep' not in time_params.keys())
            self.dt = time_params['dt']
            self.numstep = int(self.maxtime/self.dt)
        else:
            raise RuntimeError("Need to specify either 'numstep' or 'dt' in time parameters!")

        try: self.restart_step = io_params['restart_step']
        except: self.restart_step = 0

        try: self.numstep_stop = time_params['numstep_stop']
        except: self.numstep_stop = self.numstep

        try: self.results_to_write = io_params['results_to_write']
        except: self.results_to_write = []

        try: self.residual_scale = time_params['residual_scale']
        except: self.residual_scale = []

        self.t_init = self.restart_step * self.dt

        self.have_rom = False


    # routines that should be implemented by derived model problem
    def pre_timestep_routines(self):
        raise RuntimeError("Problem misses function implementation!")


    def read_restart(self):
        raise RuntimeError("Problem misses function implementation!")


    def evaluate_initial(self):
        raise RuntimeError("Problem misses function implementation!")


    def write_output_ini(self):
        raise RuntimeError("Problem misses function implementation!")


    def write_output_pre(self):
        raise RuntimeError("Problem misses function implementation!")


    def get_time_offset(self):
        raise RuntimeError("Problem misses function implementation!")


    def evaluate_pre_solve(self, t, N, dt):
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


    def destroy(self):
        raise RuntimeError("Problem misses function implementation!")


    def scale_residual_list(self, rlist):

        for n in range(len(rlist)):
            rlist[n].scale(self.pbase.residual_scale[n])


    def scale_jacobian_list(self, Klist):

        for n in range(len(Klist)):
            for m in range(len(Klist)):
                if Klist[n][m] is not None: Klist[n][m].scale(self.pbase.residual_scale[n])



class solver_base():

    def __init__(self, problem, solver_params):

        self.pb = problem

        self.solver_params = solver_params

        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.pbase.simname, self.pb.comm, self.pb.numdof)

        # model order reduction stuff
        if self.pb.pbase.have_rom:
            from .mor import mor_main
            self.pb.rom = mor_main.ModelOrderReduction(self.pb.pbrom)
            # prepare reduced-order basis (offline phase): perform Proper Orthogonal Decomposition, or read in pre-computed modes
            self.pb.rom.prepare_rob()
        else:
            self.pb.rom = None

        # read restart information if requested
        self.pb.read_restart(self.pb.pbase.simname, self.pb.pbase.restart_step)
        # update simname
        if self.pb.pbase.restart_step > 0:
            self.pb.pbase.simname += '_r'+str(self.pb.pbase.restart_step)

        self.initialize_nonlinear_solver()

        self.wt, self.ni, self.li = 0., 0, 0
        self.wt_, self.ni_, self.li_ = [], [], []


    def evaluate_system_initial(self):

        # evaluate old initial state of model
        self.pb.evaluate_initial()


    def evaluate_assemble_system_initial(self, subsolver=None):

        self.evaluate_system_initial()

        self.pb.assemble_residual(self.pb.pbase.t_init, subsolver=None) # note: subsolver only passed to stiffness eval to get correct sparsity pattern of lm-lm block
        self.pb.assemble_stiffness(self.pb.pbase.t_init, subsolver=subsolver)

        # create ROM matrix structures
        if self.pb.rom:
            self.pb.rom.set_reduced_data_structures_residual(self.pb.r_list, self.pb.r_list_rom)
            self.pb.K_list_tmp = [[None]]
            self.pb.rom.set_reduced_data_structures_matrix(self.pb.K_list, self.pb.K_list_rom, self.pb.K_list_tmp)


    # routines that should be implemented by derived model solver
    def solve_initial_state(self):
        raise RuntimeError("Problem solver misses function implementation!")


    def print_timestep_info(self):
        raise RuntimeError("Problem solver misses function implementation!")


    def solve_nonlinear_problem(self, t):
        raise RuntimeError("Problem solver misses function implementation!")


    def solve_problem(self):

        # execute time loop
        self.time_loop()

        # destroy stuff
        self.destroy()


    def time_loop(self):

        self.starttime = time.time()

        # any pre-solve that has to be done (e.g. prestress or consistent initial accelerations)
        self.solve_initial_state()

        # initial mesh data output
        self.pb.write_output_ini()

        # any other output that needs to be written at the beginning (time-invariant fields, e.g. fibers)
        self.pb.write_output_pre()

        utilities.print_sep(self.pb.comm)

        # Ambit main time loop
        for N in range(self.pb.pbase.restart_step+1, self.pb.pbase.numstep_stop+1):

            wts = time.time()

            # current time
            t = N * self.pb.pbase.dt

            # offset time (for cyclic problems)
            t_off = self.pb.get_time_offset()

            # evaluate any (solution-independent) time curves or other functions
            self.pb.evaluate_pre_solve(t-t_off, N, self.pb.pbase.dt)

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
            wt = time.time() - wts

            # print timestep info
            self.print_timestep_info(N, t, self.solnln.ni, self.solnln.li, wt)

            # apply any "on-the-fly" changes of model state/parameters
            self.pb.induce_state_change()

            # write restart information if desired
            self.pb.write_restart(self.pb.pbase.simname, N)

            # update nonlinear and linear iteration counters
            self.update_counters(wt, t)

            # set final step
            self.Nfinal = N - self.pb.pbase.restart_step

            # check any abort criterion
            if self.pb.check_abort(t-t_off):
                break

        # print final info
        self.print_final()


    def print_final(self):

        utilities.print_status("Program complete. Time for computation: %.4f s (= %.2f min)\n" % ( time.time()-self.starttime, (time.time()-self.starttime)/60. ), self.pb.comm)

        utilities.print_status("{:<55s}{:<1.4f}{:<2s}".format("Total solution time of all steps: ",self.wt," s"), self.pb.comm)
        utilities.print_status("{:<55s}{:<1.4f}{:<2s}".format("Average solution time per time step: ",self.wt/self.Nfinal," s"), self.pb.comm)
        utilities.print_status("{:<55s}{:<1.4f}{:<2s}".format("Maximum solution time of a time step: ",max(self.wt_)," s"), self.pb.comm)
        utilities.print_status(" ", self.pb.comm)
        utilities.print_status("{:<55s}{:<1d}".format("Total number of nonlinear iterations: ",self.ni), self.pb.comm)
        utilities.print_status("{:<55s}{:<1.1f}".format("Average number of nonlinear iterations per time step: ",self.ni/self.Nfinal), self.pb.comm)
        utilities.print_status("{:<55s}{:<1d}".format("Maximum number of nonlinear iterations in a time step: ",max(self.ni_)), self.pb.comm)

        if self.solnln.solvetype[0]=="iterative":

            utilities.print_status(" ", self.pb.comm)
            utilities.print_status("{:<55s}{:<1d}".format("Total number of linear iterations: ",self.li), self.pb.comm)
            utilities.print_status("{:<55s}{:<1.1f}".format("Average number of linear iterations per time step: ",self.li/self.Nfinal), self.pb.comm)
            utilities.print_status("{:<55s}{:<1d}".format("Maximum number of linear iterations in a time step: ",max(self.li_)), self.pb.comm)
            utilities.print_status("{:<55s}{:<1.1f}".format("Average number of linear iterations per solve: ",self.li/self.ni), self.pb.comm)
            utilities.print_status("{:<55s}{:<1d}".format("Maximum number of linear iterations in a solve: ",max(self.solnln.li_s)), self.pb.comm)

        utilities.print_status("-"*63, self.pb.comm)
        self.reset_counters()


    def update_counters(self, wt, t):

        self.wt += wt
        self.ni += self.solnln.ni
        self.li += self.solnln.li

        # for determination of max values
        self.wt_.append(wt)
        self.ni_.append(self.solnln.ni)
        self.li_.append(self.solnln.li)

        # write file for counters if requested
        if 'counters' in self.pb.pbase.results_to_write:

            # mode: 'wt' generates new file, 'a' appends to existing one
            if np.isclose(t,self.pb.pbase.dt): mode = 'wt'
            else: mode = 'a'

            if self.pb.comm.rank == 0:

                f = open(self.pb.pbase.output_path+'/results_'+self.pb.pbase.simname+'_wt_ni_li_counters.txt', mode)
                f.write('%.16E %.16E %i %i\n' % (t, wt, self.solnln.ni, self.solnln.li))
                f.close()


    def reset_counters(self):

        self.wt, self.ni, self.li = 0., 0, 0
        self.wt_, self.ni_, self.li_ = [], [], []


    def destroy(self):

        # destroy problem-specific stuff first
        self.pb.destroy()

        # now destroy the residuals and jacobian vectors and matrices
        for n in range(self.pb.nfields):
            self.pb.r_list[n].destroy()
            for m in range(self.pb.nfields):
                if self.pb.K_list[n][m] is not None:
                    self.pb.K_list[n][m].destroy()

        # destroy ROM-specific ones
        if self.pb.pbase.have_rom:
            if self.pb.rom:
                self.pb.rom.destroy()
                self.pb.pbrom_host.K_list_tmp[0][0].destroy()
                for n in range(self.pb.pbrom_host.nfields):
                    self.pb.pbrom_host.r_list_rom[n].destroy()
                    for m in range(self.pb.pbrom_host.nfields):
                        if self.pb.pbrom_host.K_list_rom[n][m] is not None:
                            self.pb.pbrom_host.K_list_rom[n][m].destroy()

        # destroy solver data structures
        self.solnln.destroy()
        try: self.solverprestr.destroy()
        except: pass
