#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, math
import numpy as np

from petsc4py import PETSc
import timeintegration
import utilities
import solver_nonlin

from base import problem_base

# framework of 0D flow models, relating pressure p (and its derivative) to fluxes q

class Flow0DProblem(problem_base):

    def __init__(self, io_params, time_params, model_params, time_curves, coupling_params={}, comm=None):
        problem_base.__init__(self, io_params, time_params, comm)
        
        self.problem_physics = 'flow0d'
        
        self.simname = io_params['simname']
        
        self.time_params = time_params
        
        # only relevant to syspul* models
        try:
            self.chamber_models = model_params['chamber_models']
            if 'ao' not in self.chamber_models.keys(): self.chamber_models['ao'] = {'type' : '0D_rigid'} # add aortic root model
        except: self.chamber_models = {}
        
        try: self.coronary_model = model_params['coronary_model']
        except: self.coronary_model = None
        
        try: self.excitation_curve = model_params['excitation_curve']
        except: self.excitation_curve = None

        try: initial_file = time_params['initial_file']
        except: initial_file = ''

        # could use extra write frequency setting for 0D model (i.e. for coupled problem)
        try: self.write_results_every_0D = io_params['write_results_every_0D']
        except: self.write_results_every_0D = io_params['write_results_every']

        # for restart
        try: self.write_restart_every = io_params['write_restart_every']
        except: self.write_restart_every = -1

        # could use extra output path setting for 0D model (i.e. for coupled problem)
        try: self.output_path_0D = io_params['output_path_0D']
        except: self.output_path_0D = io_params['output_path']
        
        try: valvelaws = model_params['valvelaws']
        except: valvelaws = {'av' : ['pwlin_pres',0], 'mv' : ['pwlin_pres',0], 'pv' : ['pwlin_pres',0], 'tv' : ['pwlin_pres',0]}

        try: self.cq = coupling_params['coupling_quantity']
        except: self.cq = ['volume']
        
        try: self.vq = coupling_params['variable_quantity']
        except: self.vq = ['pressure']
        
        try: self.eps_periodic = time_params['eps_periodic']
        except: self.eps_periodic = 1.0e-20
        
        try: self.periodic_checktype = time_params['periodic_checktype']
        except: self.periodic_checktype = 'allvar'
        
        try: self.prescribed_variables = model_params['prescribed_variables']
        except: self.prescribed_variables = {}
        
        try: self.perturb_type = model_params['perturb_type'][0]
        except: self.perturb_type = None

        try: self.perturb_factor = model_params['perturb_type'][1]
        except: self.perturb_factor = 1.

        try: self.perturb_id = model_params['perturb_type'][2]
        except: self.perturb_id = -1
        
        try: self.perturb_after_cylce = model_params['perturb_after_cylce']
        except: self.perturb_after_cylce = -1
        # definitely set to -1 if we don't have a perturb type
        if self.perturb_type is None: self.perturb_after_cylce = -1
        
        self.have_induced_pert = False

        # initialize 0D model class
        if model_params['modeltype'] == '2elwindkessel':
            from cardiovascular0D_2elwindkessel import cardiovascular0D2elwindkessel
            self.cardvasc0D = cardiovascular0D2elwindkessel(model_params['parameters'], self.cq, self.vq, comm=self.comm)
        elif model_params['modeltype'] == '4elwindkesselLsZ':
            from cardiovascular0D_4elwindkesselLsZ import cardiovascular0D4elwindkesselLsZ
            self.cardvasc0D = cardiovascular0D4elwindkesselLsZ(model_params['parameters'], self.cq, self.vq, comm=self.comm)
        elif model_params['modeltype'] == '4elwindkesselLpZ':
            from cardiovascular0D_4elwindkesselLpZ import cardiovascular0D4elwindkesselLpZ
            self.cardvasc0D = cardiovascular0D4elwindkesselLpZ(model_params['parameters'], self.cq, self.vq, comm=self.comm)
        elif model_params['modeltype'] == 'syspul':
            from cardiovascular0D_syspul import cardiovascular0Dsyspul
            self.cardvasc0D = cardiovascular0Dsyspul(model_params['parameters'], self.chamber_models, self.coronary_model, self.cq, self.vq, valvelaws=valvelaws, comm=self.comm)
        elif model_params['modeltype'] == 'syspulcap':
            from cardiovascular0D_syspulcap import cardiovascular0Dsyspulcap
            self.cardvasc0D = cardiovascular0Dsyspulcap(model_params['parameters'], self.chamber_models, self.coronary_model, self.cq, self.vq, valvelaws=valvelaws, comm=self.comm)
        elif model_params['modeltype'] == 'syspulcapcor':
            from cardiovascular0D_syspulcap import cardiovascular0Dsyspulcapcor
            self.cardvasc0D = cardiovascular0Dsyspulcapcor(model_params['parameters'], self.chamber_models, self.coronary_model, self.cq, self.vq, valvelaws=valvelaws, comm=self.comm)
        elif model_params['modeltype'] == 'syspulcaprespir':
            from cardiovascular0D_syspulcaprespir import cardiovascular0Dsyspulcaprespir
            self.cardvasc0D = cardiovascular0Dsyspulcaprespir(model_params['parameters'], self.chamber_models, self.coronary_model, self.cq, self.vq, valvelaws=valvelaws, comm=self.comm)
        else:
            raise NameError("Unknown 0D modeltype!")

        # vectors and matrices
        self.dK = PETSc.Mat().createAIJ(size=(self.cardvasc0D.numdof,self.cardvasc0D.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.dK.setUp()
        
        self.K = PETSc.Mat().createAIJ(size=(self.cardvasc0D.numdof,self.cardvasc0D.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.K.setUp()

        self.s, self.s_old, self.s_mid = self.K.createVecLeft(), self.K.createVecLeft(), self.K.createVecLeft()
        self.sTc, self.sTc_old = self.K.createVecLeft(), self.K.createVecLeft()
        
        self.df, self.df_old = self.K.createVecLeft(), self.K.createVecLeft()
        self.f, self.f_old   = self.K.createVecLeft(), self.K.createVecLeft()

        self.aux, self.aux_old, self.aux_mid = np.zeros(self.cardvasc0D.numdof), np.zeros(self.cardvasc0D.numdof), np.zeros(self.cardvasc0D.numdof)
        
        self.s_set = self.K.createVecLeft() # set point for multisale analysis
        
        self.c, self.y = [], []

        # initialize flow0d time-integration class
        self.ti = timeintegration.timeintegration_flow0d(time_params, time_curves, self.t_init, comm=self.comm)

        if initial_file:
            initialconditions = self.cardvasc0D.set_initial_from_file(initial_file)
        else:
            initialconditions = time_params['initial_conditions']

        self.cardvasc0D.initialize(self.s, initialconditions)
        self.cardvasc0D.initialize(self.s_old, initialconditions)
        self.cardvasc0D.initialize(self.sTc_old, initialconditions)

        self.theta_ost = time_params['theta_ost']


    def assemble_residual_stiffness(self):
        
        K = PETSc.Mat().createAIJ(size=(self.cardvasc0D.numdof,self.cardvasc0D.numdof), bsize=None, nnz=None, csr=None, comm=self.comm)
        K.setUp()
        
        # 0D rhs vector: r = (df - df_old)/dt + theta * f + (1-theta) * f_old
        r = K.createVecLeft()

        r.axpy(1./self.dt, self.df)
        r.axpy(-1./self.dt, self.df_old)
        
        r.axpy(self.theta_ost, self.f)
        r.axpy(1.-self.theta_ost, self.f_old)     

        self.dK.assemble()
        self.K.assemble()
        K.assemble()

        K.axpy(1./self.dt, self.dK)
        K.axpy(self.theta_ost, self.K)

        return r, K


    def writerestart(self, sname, N, ms=False):
        
        self.cardvasc0D.write_restart(self.output_path_0D, sname+'_s', N, self.s)
        self.cardvasc0D.write_restart(self.output_path_0D, sname+'_aux', N, self.aux)
        self.cardvasc0D.write_restart(self.output_path_0D, sname+'_sTc_old', N, self.sTc_old)
        if ms: self.cardvasc0D.write_restart(self.output_path_0D, sname+'_s_set', N, self.s_set)
        
        if self.cardvasc0D.T_cycl > 0: # write heart cycle info
            if self.comm.rank == 0:
                filename = self.output_path_0D+'/checkpoint_'+sname+'_cycledata_'+str(N)+'.txt'
                f = open(filename, 'wt')
                f.write('%i %.8f' % (self.ti.cycle[0],self.ti.cycleerror[0]))
                f.close()


    def readrestart(self, sname, rst, ms=False):
        
        self.cardvasc0D.read_restart(self.output_path_0D, sname+'_s', rst, self.s)
        self.cardvasc0D.read_restart(self.output_path_0D, sname+'_s', rst, self.s_old)
        self.cardvasc0D.read_restart(self.output_path_0D, sname+'_aux', rst, self.aux)
        self.cardvasc0D.read_restart(self.output_path_0D, sname+'_aux', rst, self.aux_old)
        self.cardvasc0D.read_restart(self.output_path_0D, sname+'_sTc_old', rst, self.sTc_old)
        if ms: self.cardvasc0D.read_restart(self.output_path_0D, sname+'_s_set', rst, self.s_set)

        if self.cardvasc0D.T_cycl > 0: # read heart cycle info
            self.ti.cycle[0] = np.loadtxt(self.output_path_0D+'/checkpoint_'+sname+'_cycledata_'+str(rst)+'.txt', usecols=(0), dtype=int)
            self.ti.cycleerror[0] = np.loadtxt(self.output_path_0D+'/checkpoint_'+sname+'_cycledata_'+str(rst)+'.txt', usecols=(1), dtype=float)
            self.t_init -= (self.ti.cycle[0]-1) * self.cardvasc0D.T_cycl


    def evaluate_activation(self, t):
        
        # activation curves
        if bool(self.chamber_models):
            ci=0
            for ch in ['lv','rv','la','ra']:
                if self.chamber_models[ch]['type']=='0D_elast': 
                    self.y[ci] = self.ti.timecurves(self.chamber_models[ch]['activation_curve'])(t)
                    ci+=1
                if self.chamber_models[ch]['type']=='0D_elast_prescr': 
                    self.y[ci] = self.ti.timecurves(self.chamber_models[ch]['elastance_curve'])(t)
                    ci+=1


    def induce_perturbation(self):
        
        if self.perturb_after_cylce > 0: # at least run through one healthy cycle
            
            if self.ti.cycle[0] > self.perturb_after_cylce:

                if self.comm.rank == 0:
                    print(">>> Induced cardiovascular disease type: %s" % (self.perturb_type))
                    sys.stdout.flush()
        
                self.cardvasc0D.induce_perturbation(self.perturb_type, self.perturb_factor)
                self.have_induced_pert = True



class Flow0DSolver():

    def __init__(self, problem, solver_params):
    
        self.pb = problem

        self.solver_params = solver_params

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_0D(self.pb, self.solver_params)
        

    def solve_problem(self):
        
        start = time.time()
        
        # print header
        utilities.print_problem(self.pb.problem_type, self.pb.comm, self.pb.cardvasc0D.numdof)

        # read restart information
        if self.pb.restart_step > 0:
            self.pb.readrestart(self.pb.simname, self.pb.restart_step)
            self.pb.simname += '_r'+str(self.pb.restart_step)
        
        # evaluate old state
        if self.pb.excitation_curve is not None:
            self.pb.c = []
            self.pb.c.append(self.pb.ti.timecurves(self.pb.excitation_curve)(self.pb.t_init))
        if bool(self.pb.chamber_models):
            self.pb.y = []
            for ch in ['lv','rv','la','ra']:
                if self.pb.chamber_models[ch]['type']=='0D_elast': self.pb.y.append(self.pb.ti.timecurves(self.pb.chamber_models[ch]['activation_curve'])(self.pb.t_init))
                if self.pb.chamber_models[ch]['type']=='0D_elast_prescr': self.pb.y.append(self.pb.ti.timecurves(self.pb.chamber_models[ch]['elastance_curve'])(self.pb.t_init))
                if self.pb.chamber_models[ch]['type']=='0D_prescr': self.pb.c.append(self.pb.ti.timecurves(self.pb.chamber_models[ch]['prescribed_curve'])(self.pb.t_init))

        self.pb.cardvasc0D.evaluate(self.pb.s_old, self.pb.t_init, self.pb.df_old, self.pb.f_old, None, None, self.pb.c, self.pb.y, self.pb.aux_old)


        # flow 0d main time loop
        for N in range(self.pb.restart_step+1, self.pb.numstep_stop+1):
            
            wts = time.time()
            
            # current time
            t = N * self.pb.dt

            # offset time for multiple cardiac cycles
            t_off = (self.pb.ti.cycle[0]-1) * self.pb.cardvasc0D.T_cycl # zero if T_cycl variable is not specified

            # external volume/flux from time curve
            if self.pb.excitation_curve is not None:
                self.pb.c[0] = self.pb.ti.timecurves(self.pb.excitation_curve)(t-t_off)
            # activation curves
            self.pb.evaluate_activation(t-t_off)

            # solve
            self.solnln.newton(self.pb.s, t-t_off)

            # get midpoint dof values for post-processing (has to be called before update!)
            self.pb.cardvasc0D.midpoint_avg(self.pb.s, self.pb.s_old, self.pb.s_mid, self.pb.theta_ost), self.pb.cardvasc0D.midpoint_avg(self.pb.aux, self.pb.aux_old, self.pb.aux_mid, self.pb.theta_ost)

            # raw txt file output of 0D model quantities
            if self.pb.write_results_every_0D > 0 and N % self.pb.write_results_every_0D == 0:
                self.pb.cardvasc0D.write_output(self.pb.output_path_0D, t, self.pb.s_mid, self.pb.aux_mid, self.pb.simname)

            # update timestep
            self.pb.cardvasc0D.update(self.pb.s, self.pb.df, self.pb.f, self.pb.s_old, self.pb.df_old, self.pb.f_old, self.pb.aux, self.pb.aux_old)
            
            # print to screen
            self.pb.cardvasc0D.print_to_screen(self.pb.s_mid,self.pb.aux_mid)

            # solve time for time step
            wte = time.time()
            wt = wte - wts

            # print time step info to screen
            self.pb.ti.print_timestep(N, t, self.pb.numstep, wt=wt)
            
            # check for periodicity in cardiac cycle and stop if reached (only for syspul* models - cycle counter gets updated here)
            is_periodic = self.pb.cardvasc0D.cycle_check(self.pb.s, self.pb.sTc, self.pb.sTc_old, t-t_off, self.pb.ti.cycle, self.pb.ti.cycleerror, self.pb.eps_periodic, check=self.pb.periodic_checktype, inioutpath=self.pb.output_path_0D, nm=self.pb.simname, induce_pert_after_cycl=self.pb.perturb_after_cylce)

            # induce some disease/perturbation for cardiac cycle (i.e. valve stenosis or leakage)
            if self.pb.perturb_type is not None and not self.pb.have_induced_pert: self.pb.induce_perturbation()
            
            # write 0D restart info - old and new quantities are the same at this stage (except cycle values sTc)
            if self.pb.write_restart_every > 0 and N % self.pb.write_restart_every == 0:
                self.pb.writerestart(self.pb.simname, N)

            if is_periodic:
                if self.pb.comm.rank == 0:
                    print("Periodicity reached after %i heart cycles with cycle error %.4f! Finished. :-)" % (self.pb.ti.cycle[0]-1,self.pb.ti.cycleerror[0]))
                    sys.stdout.flush()
                break
            

        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()
