#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np

from petsc4py import PETSc
import timeintegration
import utilities
import solver_nonlin

from base import problem_base

# framework of signalling network models

class SignallingNetworkProblem(problem_base):

    def __init__(self, io_params, time_params, model_params, time_curves, coupling_params={}, comm=None):
        problem_base.__init__(self, io_params, time_params, comm)
        
        self.problem_physics = 'signet'
        
        self.simname = io_params['simname']
        
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
        try: self.output_path_0D = io_params['output_path_signet']
        except: self.output_path_signet = io_params['output_path']
        
        # whether to output midpoint (t_{n+theta}) of state variables or endpoint (t_{n+1}) - for post-processing
        try: self.output_midpoint = io_params['output_midpoint_0D']
        except: self.output_midpoint = True
        
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

        # vectors and matrices
        self.dK = PETSc.Mat().createAIJ(size=(self.signet.numdof,self.signet.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.dK.setUp()
        
        self.K = PETSc.Mat().createAIJ(size=(self.signet.numdof,self.signet.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K.setUp()

        self.s, self.s_old, self.s_mid = self.K.createVecLeft(), self.K.createVecLeft(), self.K.createVecLeft()
        self.sTc, self.sTc_old = self.K.createVecLeft(), self.K.createVecLeft()
        
        self.df, self.df_old = self.K.createVecLeft(), self.K.createVecLeft()
        self.f, self.f_old   = self.K.createVecLeft(), self.K.createVecLeft()

        self.aux, self.aux_old, self.aux_mid = np.zeros(self.signet.numdof), np.zeros(self.signet.numdof), np.zeros(self.signet.numdof)
        
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


    def assemble_residual_stiffness(self, t):

        theta = self.thetasn_timint(t)

        K = PETSc.Mat().createAIJ(size=(self.signet.numdof,self.signet.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        K.setUp()
        
        # signet rhs vector: r = (df - df_old)/dt + theta * f + (1-theta) * f_old
        r = K.createVecLeft()

        r.axpy(1./self.dt, self.df)
        r.axpy(-1./self.dt, self.df_old)
        
        r.axpy(theta, self.f)
        r.axpy(1.-theta, self.f_old)     

        self.dK.assemble()
        self.K.assemble()
        K.assemble()

        K.axpy(1./self.dt, self.dK)
        K.axpy(theta, self.K)

        return r, K


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



class SignallingNetworkSolver():

    def __init__(self, problem, solver_params):
    
        self.pb = problem

        self.solver_params = solver_params

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_ode(self.pb, self.solver_params)
        

    def solve_problem(self):
        
        start = time.time()
        
        # print header
        utilities.print_problem(self.pb.problem_type, self.pb.comm, self.pb.signet.numdof)

        # read restart information
        if self.pb.restart_step > 0:
            self.pb.readrestart(self.pb.simname, self.pb.restart_step)
            self.pb.simname += '_r'+str(self.pb.restart_step)
        
        ## evaluate old state
        #if self.pb.excitation_curve is not None:
            #self.pb.c = []
            #self.pb.c.append(self.pb.ti.timecurves(self.pb.excitation_curve)(self.pb.t_init))

        self.pb.signet.evaluate(self.pb.s_old, self.pb.t_init, self.pb.df_old, self.pb.f_old, None, None, self.pb.c, self.pb.y, self.pb.aux_old)

        # flow 0d main time loop
        for N in range(self.pb.restart_step+1, self.pb.numstep_stop+1):
            
            wts = time.time()
            
            # current time
            t = N * self.pb.dt

            ## offset time for multiple cardiac cycles
            #t_off = (self.pb.ti.cycle[0]-1) * self.pb.signet.T_cycl # zero if T_cycl variable is not specified
            t_off = 0.

            ## external volume/flux from time curve
            #if self.pb.excitation_curve is not None:
                #self.pb.c[0] = self.pb.ti.timecurves(self.pb.excitation_curve)(t-t_off)
            ## activation curves
            #self.pb.evaluate_activation(t-t_off)

            # solve
            self.solnln.newton(self.pb.s, t-t_off)

            # get midpoint dof values for post-processing (has to be called before update!)
            self.pb.signet.set_output_state(self.pb.s, self.pb.s_old, self.pb.s_mid, self.pb.theta_ost, midpoint=self.output_midpoint), self.pb.signet.set_output_state(self.pb.aux, self.pb.aux_old, self.pb.aux_mid, self.pb.theta_ost, midpoint=self.output_midpoint)

            # raw txt file output of signet model quantities
            if self.pb.write_results_every_signet > 0 and N % self.pb.write_results_every_signet == 0:
                self.pb.signet.write_output(self.pb.output_path_signet, t, self.pb.s_mid, self.pb.aux_mid, self.pb.simname)

            # update timestep
            self.pb.signet.update(self.pb.s, self.pb.df, self.pb.f, self.pb.s_old, self.pb.df_old, self.pb.f_old, self.pb.aux, self.pb.aux_old)
            
            # print to screen
            self.pb.signet.print_to_screen(self.pb.s_mid,self.pb.aux_mid)

            # solve time for time step
            wte = time.time()
            wt = wte - wts

            # print time step info to screen
            self.pb.ti.print_timestep(N, t, self.solnln.sepstring, self.pb.numstep, wt=wt)
                        
            # write signet restart info - old and new quantities are the same at this stage (except cycle values sTc)
            if self.pb.write_restart_every > 0 and N % self.pb.write_restart_every == 0:
                self.pb.writerestart(self.pb.simname, N)

            #if is_periodic:
                #if self.pb.comm.rank == 0:
                    #print("Periodicity reached after %i heart cycles with cycle error %.4f! Finished. :-)" % (self.pb.ti.cycle[0]-1,self.pb.ti.cycleerror[0]))
                    #sys.stdout.flush()
                #break
            
        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Program complete. Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()
