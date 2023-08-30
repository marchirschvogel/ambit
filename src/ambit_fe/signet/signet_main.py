#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from petsc4py import PETSc

from .. import timeintegration
from ..solver import solver_nonlin
from .. import ioparams

from ..base import problem_base, solver_base

# framework of signalling network models

class SignallingNetworkProblem(problem_base):

    def __init__(self, io_params, time_params, model_params, time_curves, coupling_params={}, comm=None):
        super().__init__(io_params, time_params, comm)

        self.problem_physics = 'signet'

        self.time_params = time_params

        try: initial_file = time_params['initial_file']
        except: initial_file = ''

        # could use extra write frequency setting for signet model (i.e. for coupled problem)
        try: self.write_results_every_signet = io_params['write_results_every_signet']
        except: self.write_results_every_signet = io_params['write_results_every']

        # for restart
        try: self.write_restart_every = io_params['write_restart_every']
        except: self.write_restart_every = -1

        # could use extra output path setting for signet model (i.e. for coupled problem)
        try: self.output_path_signet = io_params['output_path_signet']
        except: self.output_path_signet = io_params['output_path']

        # whether to output midpoint (t_{n+theta}) of state variables or endpoint (t_{n+1}) - for post-processing
        try: self.output_midpoint = io_params['output_midpoint_0D']
        except: self.output_midpoint = False

        try: self.prescribed_variables = model_params['prescribed_variables']
        except: self.prescribed_variables = {}

        try: self.initial_backwardeuler = time_params['initial_backwardeuler']
        except: self.initial_backwardeuler = False

        # initialize signet model class
        if model_params['modeltype'] == 'hypertrophy':
            from signet_hypertrophy import signethypertrophy
            self.signet = signethypertrophy(model_params['parameters'], comm=self.comm)
        else:
            raise NameError("Unknown signet modeltype!")

        self.numdof = self.signet.numdof

        # vectors and matrices
        self.dK_ = PETSc.Mat().createAIJ(size=(self.numdof,self.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.dK_.setUp()

        self.K_ = PETSc.Mat().createAIJ(size=(self.numdof,self.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K_.setUp()

        self.K = PETSc.Mat().createAIJ(size=(self.numdof,self.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K.setUp()

        self.r = self.K.createVecLeft()

        self.s, self.s_old, self.s_mid = self.K.createVecLeft(), self.K.createVecLeft(), self.K.createVecLeft()
        self.sTc, self.sTc_old = self.K.createVecLeft(), self.K.createVecLeft()

        self.df, self.df_old = self.K.createVecLeft(), self.K.createVecLeft()
        self.f, self.f_old   = self.K.createVecLeft(), self.K.createVecLeft()

        self.aux, self.aux_old, self.aux_mid = np.zeros(self.numdof), np.zeros(self.numdof), np.zeros(self.numdof)

        self.s_set = self.K.createVecLeft() # set point for multisale analysis

        self.c, self.y = [], []

        # initialize signet time-integration class
        self.ti = timeintegration.timeintegration_signet(time_params, time_curves, self.t_init, comm=self.comm)

        if initial_file:
            initialconditions = self.signet.set_initial_from_file(initial_file)
        else:
            initialconditions = time_params['initial_conditions']

        self.signet.initialize(self.s, initialconditions)
        self.signet.initialize(self.s_old, initialconditions)
        self.signet.initialize(self.sTc_old, initialconditions)

        self.theta_ost = time_params['theta_ost']

        self.odemodel = self.signet

        # number of fields involved
        self.nfields=1

        # residual and matrix lists
        self.r_list = [None]*self.nfields
        self.K_list = [[None]*self.nfields for _ in range(self.nfields)]


    def assemble_residual(self, t):

        self.signet.evaluate(self.s, t, self.df, self.f, None, None, self.c, self.y, self.aux)

        theta = self.thetasn_timint(t)

        self.df.assemble(), self.df_old.assemble()
        self.f.assemble(), self.f_old.assemble()

        # signet rhs vector: r = (df - df_old)/dt + theta * f + (1-theta) * f_old
        self.r.zeroEntries()

        self.r.axpy(1./self.dt, self.df)
        self.r.axpy(-1./self.dt, self.df_old)

        self.r.axpy(theta, self.f)
        self.r.axpy(1.-theta, self.f_old)

        self.r_list[0] = self.r


    def assemble_stiffness(self, t):

        self.signet.evaluate(self.s, t, None, None, self.dK_, self.K_, self.c, self.y, self.aux)

        theta = self.thetasn_timint(t)

        self.dK_.assemble()
        self.K_.assemble()
        self.K.assemble()

        self.K.zeroEntries()
        self.K.axpy(1./self.dt, self.dK_)
        self.K.axpy(theta, self.K_)

        self.K_list[0][0] = self.K


    def thetasn_timint(self, t):

        if self.initial_backwardeuler:
            if np.isclose(t,self.dt):
                theta = 1.0
            else:
                theta = self.theta_ost
        else:
            theta = self.theta_ost

        return theta


    def writerestart(self, sname, N, ms=False):

        self.signet.write_restart(self.output_path_signet, sname+'_s', N, self.s)
        self.signet.write_restart(self.output_path_signet, sname+'_aux', N, self.aux)
        self.signet.write_restart(self.output_path_signet, sname+'_sTc_old', N, self.sTc_old)
        if ms: self.signet.write_restart(self.output_path_signet, sname+'_s_set', N, self.s_set)

        if self.signet.T_cycl > 0: # write heart cycle info
            if self.comm.rank == 0:
                filename = self.output_path_signet+'/checkpoint_'+sname+'_cycledata_'+str(N)+'.txt'
                f = open(filename, 'wt')
                f.write('%i %.8f' % (self.ti.cycle[0],self.ti.cycleerror[0]))
                f.close()


    def readrestart(self, sname, rst, ms=False):

        self.signet.read_restart(self.output_path_signet, sname+'_s', rst, self.s)
        self.signet.read_restart(self.output_path_signet, sname+'_s', rst, self.s_old)
        self.signet.read_restart(self.output_path_signet, sname+'_aux', rst, self.aux)
        self.signet.read_restart(self.output_path_signet, sname+'_aux', rst, self.aux_old)
        self.signet.read_restart(self.output_path_signet, sname+'_sTc_old', rst, self.sTc_old)
        if ms: self.signet.read_restart(self.output_path_signet, sname+'_s_set', rst, self.s_set)

        if self.signet.T_cycl > 0: # read heart cycle info
            self.ti.cycle[0] = np.loadtxt(self.output_path_signet+'/checkpoint_'+sname+'_cycledata_'+str(rst)+'.txt', usecols=(0), dtype=int)
            self.ti.cycleerror[0] = np.loadtxt(self.output_path_signet+'/checkpoint_'+sname+'_cycledata_'+str(rst)+'.txt', usecols=(1), dtype=float)
            self.t_init -= (self.ti.cycle[0]-1) * self.signet.T_cycl


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # read restart information
        if self.restart_step > 0:
            self.readrestart(sname, N)
            self.simname += '_r'+str(N)


    def evaluate_initial(self):

        self.signet.evaluate(self.s_old, self.t_init, self.df_old, self.f_old, None, None, self.c, self.y, self.aux_old)


    def write_output_ini(self):
        pass


    def get_time_offset(self):
        return 0.


    def evaluate_pre_solve(self, t, N):
        pass


    def evaluate_post_solve(self, t, N):
        pass


    def set_output_state(self, t):

        # get midpoint dof values for post-processing (has to be called before update!)
        self.s.assemble(), self.s_old.assemble(), self.s_mid.assemble()
        self.signet.set_output_state(self.s, self.s_old, self.s_mid, self.thetasn_timint(t), midpoint=self.output_midpoint)
        self.signet.set_output_state(self.aux, self.aux_old, self.aux_mid, self.thetasn_timint(t), midpoint=self.output_midpoint)


    def write_output(self, N, t):

        # raw txt file output of 0D model quantities
        if self.write_results_every_signet > 0 and N % self.write_results_every_signet == 0:
            self.signet.write_output(self.output_path_signet, t, self.s_mid, self.aux_mid, self.simname)


    def update(self):

        # update timestep
        self.signet.update(self.s, self.df, self.f, self.s_old, self.df_old, self.f_old, self.aux, self.aux_old)


    def print_to_screen(self):
        pass


    def induce_state_change(self):
        pass


    def write_restart(self, sname, N):

        # write 0D restart info - old and new quantities are the same at this stage (except cycle values sTc)
        if self.write_restart_every > 0 and N % self.write_restart_every == 0:
            self.writerestart(sname, N)


    def check_abort(self, t):
        pass


    def destroy(self):
        pass



class SignallingNetworkSolver(solver_base):


    def initialize_nonlinear_solver(self):

        self.evaluate_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_ode([self.pb], self.solver_params)


    def solve_initial_state(self):
        pass


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, self.pb.numstep, ni=ni, li=li, wt=wt)
