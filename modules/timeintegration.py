#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from petsc4py import PETSc
import ufl

from projection import project
import expression

class timeintegration():
    
    def __init__(self, time_params, time_curves, t_init, domain=[], comm=None):
        
        try: self.timint = time_params['timint']
        except: self.timint = 'static'
        
        if 'numstep' in time_params.keys(): self.numstep = time_params['numstep']
        if 'maxtime' in time_params.keys(): self.maxtime = time_params['maxtime'] 
        
        if 'maxtime' in time_params.keys(): self.dt = self.maxtime/self.numstep
        
        self.time_curves = time_curves
        self.t_init = t_init
        
        self.domain = domain

        self.comm = comm
        
        # time-dependent functions to update
        self.funcs_to_update, self.funcs_to_update_old, self.funcs_to_update_vec, self.funcs_to_update_vec_old = [], [], [], []


    # print timestep info
    def print_timestep(self, N, t, separator, wt=0):
        
        if self.comm.rank == 0:

            print("### TIME STEP %i / %i successfully completed | TIME: %.4f | wt = %.2e" % (N,self.numstep,t,wt))
            print(separator)
            sys.stdout.flush()


    def set_time_funcs(self, funcs, funcs_vec, t):

        for m in funcs_vec:
            load = expression.template_vector()
            load.val_x, load.val_y, load.val_z = list(m.values())[0][0](t), list(m.values())[0][1](t), list(m.values())[0][2](t)
            list(m.keys())[0].interpolate(load.evaluate)
            list(m.keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        for m in funcs:
            load = expression.template()
            load.val = list(m.values())[0](t)
            list(m.keys())[0].interpolate(load.evaluate)
            list(m.keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    
    def update_time_funcs(self, funcs, funcs_old, funcsvec, funcsvec_old):
        
        # update time-dependent functions
        for m in range(len(funcs_old)):
            list(funcs_old[m].keys())[0].vector.axpby(1.0, 0.0, list(funcs[m].keys())[0].vector)
            list(funcs_old[m].keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        for m in range(len(funcsvec_old)):
            list(funcsvec_old[m].keys())[0].vector.axpby(1.0, 0.0, list(funcsvec[m].keys())[0].vector)
            list(funcsvec_old[m].keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # zero
    def zero(self, t):
        return 0.0

    def timecurves(self, cnum):
        
        if cnum==0: return self.zero
        if cnum==1: return self.time_curves.tc1
        if cnum==2: return self.time_curves.tc2
        if cnum==3: return self.time_curves.tc3
        if cnum==4: return self.time_curves.tc4
        if cnum==5: return self.time_curves.tc5
        if cnum==6: return self.time_curves.tc6
        if cnum==7: return self.time_curves.tc7
        if cnum==8: return self.time_curves.tc8
        if cnum==9: return self.time_curves.tc9


# Solid mechanics time integration class
class timeintegration_solid(timeintegration):
    
    def __init__(self, time_params, fem_params, time_curves, t_init, dx_, comm):
        timeintegration.__init__(self, time_params, time_curves, t_init, dx_, comm)
        
        if self.timint == 'genalpha':
            
            # if the spectral radius, rho_inf_genalpha, is specified, the parameters are computed from it
            try:
                self.rho_inf_genalpha = time_params['rho_inf_genalpha']
                self.alpha_m, self.alpha_f, self.beta, self.gamma = self.compute_genalpha_params(self.rho_inf_genalpha)
            # otherwise, user can specify each parameter individually
            except:
                try: self.alpha_m = time_params['alpha_m']
                except: raise AttributeError("Need to specify alpha_m if rho_inf_genalpha is not sepcified!")

                try: self.alpha_f = time_params['alpha_f']
                except: raise AttributeError("Need to specify alpha_f if rho_inf_genalpha is not sepcified!")
            
                try: self.beta = time_params['beta']
                except: raise AttributeError("Need to specify beta if rho_inf_genalpha is not sepcified!")
            
                try: self.gamma = time_params['gamma']
                except: raise AttributeError("Need to specify gamma if rho_inf_genalpha is not sepcified!")

            
        if self.timint == 'ost':
            
            self.theta_ost = time_params['theta_ost']
        
        self.incompressible_2field = fem_params['incompressible_2field']


    def set_acc_vel(self, u, u_old, v_old, a_old):
        
        # set forms for acc and vel
        if self.timint == 'genalpha':
            acc = self.update_a_newmark(u, u_old, v_old, a_old, ufl=True)
            vel = self.update_v_newmark(acc, u, u_old, v_old, a_old, ufl=True)
        elif self.timint == 'ost':
            acc = self.update_a_ost(u, u_old, v_old, a_old, ufl=True)
            vel = self.update_v_ost(acc, u, u_old, v_old, a_old, ufl=True)
        elif self.timint == 'static':
            acc = ufl.constantvalue.zero(3)
            vel = ufl.constantvalue.zero(3)
        else:
            raise NameError("Unknown time-integration algorithm for solid mechanics!")
        
        return acc, vel


    def timefactors(self):
        
        if self.timint=='genalpha': timefac_m, timefac = 1.-self.alpha_m, 1.-self.alpha_f
        if self.timint=='ost':      timefac_m, timefac = self.theta_ost, self.theta_ost
        if self.timint=='static':   timefac_m, timefac = 1., 1.
        
        return timefac_m, timefac


    def update_timestep(self, u, u_old, v_old, a_old, p, p_old, internalvars, internalvars_old, funcs, funcs_old, funcsvec, funcsvec_old):
    
        # now update old kinematic fields with new quantities
        if self.timint == 'genalpha':
            self.update_fields_newmark(u, u_old, v_old, a_old)
        if self.timint == 'ost':
            self.update_fields_ost(u, u_old, v_old, a_old)
        
        # update pressure variable
        if self.incompressible_2field:
            p_old.vector.axpby(1.0, 0.0, p.vector)
            p_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update internal variables (e.g. active stress, growth stretch, plastic strains, ...)
        for i in range(len(internalvars_old)):
            list(internalvars_old.values())[i].vector.axpby(1.0, 0.0, list(internalvars.values())[i].vector)
            list(internalvars_old.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update time dependent load curves
        self.update_time_funcs(funcs, funcs_old, funcsvec, funcsvec_old)


    def update_a_newmark(self, u, u_old, v_old, a_old, ufl=True):
        # update formula for acceleration
        if ufl:
            dt_ = self.dt
            beta_ = self.beta
        else:
            dt_ = float(self.dt)
            beta_ = float(self.beta)
        return 1./(beta_*dt_*dt_) * (u - u_old) - 1./(beta_*dt_) * v_old - (1.-2.*beta_)/(2.*beta_) * a_old


    def update_a_ost(self, u, u_old, v_old, a_old, ufl=True):
        # update formula for acceleration
        if ufl:
            dt_ = self.dt
            theta_ = self.theta_ost
        else:
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)
        return 1./(theta_*theta_*dt_*dt_) * (u - u_old) - 1./(theta_*theta_*dt_) * v_old - (1.-theta_)/theta_ * a_old


    def update_v_newmark(self, a, u, u_old, v_old, a_old, ufl=True):
        # update formula for velocity
        if ufl:
            dt_ = self.dt
            gamma_ = self.gamma
            beta_ = self.beta
        else:
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)
            beta_ = float(self.beta)
        return gamma_/(beta_*dt_) * (u - u_old) - (gamma_ - beta_)/beta_ * v_old - (gamma_-2.*beta_)/(2.*beta_) * dt_*a_old


    def update_v_ost(self, a, u, u_old, v_old, a_old, ufl=True):
        # update formula for velocity
        if ufl:
            dt_ = self.dt
            theta_ = self.theta_ost
        else:
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)
        return 1./(theta_*dt_) * (u - u_old) - (1. - theta_)/theta_ * v_old


    def update_fields_newmark(self, u, u_old, v_old, a_old):
        # update fields at the end of each time step 
        # get vectors (references)
        u_vec, u0_vec  = u.vector, u_old.vector
        v0_vec, a0_vec = v_old.vector, a_old.vector
        u_vec.assemble(), u0_vec.assemble(), v0_vec.assemble(), a0_vec.assemble()
        
        # use update functions using vector arguments
        a_vec = self.update_a_newmark(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = self.update_v_newmark(a_vec, u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        
        # update acceleration: a_old <- a
        a_old.vector.axpby(1.0, 0.0, a_vec)
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update velocity: v_old <- v
        v_old.vector.axpby(1.0, 0.0, v_vec)
        v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update displacement: u_old <- u
        u_old.vector.axpby(1.0, 0.0, u_vec)
        u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def update_fields_ost(self, u, u_old, v_old, a_old):
        # update fields at the end of each time step 
        # get vectors (references)
        u_vec, u0_vec  = u.vector, u_old.vector
        v0_vec, a0_vec = v_old.vector, a_old.vector
        u_vec.assemble(), u0_vec.assemble(), v0_vec.assemble(), a0_vec.assemble()
        
        # use update functions using vector arguments
        a_vec = self.update_a_ost(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = self.update_v_ost(a_vec, u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        
        # update acceleration: a_old <- a
        a_old.vector.axpby(1.0, 0.0, a_vec)
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update velocity: v_old <- v
        v_old.vector.axpby(1.0, 0.0, v_vec)
        v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update displacement: u_old <- u
        u_old.vector.axpby(1.0, 0.0, u_vec)
        u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def compute_genalpha_params(self, rho_inf):
        
        alpha_m = (2.*rho_inf-1.)/(rho_inf+1.)
        alpha_f = rho_inf/(rho_inf+1.)
        beta    = 0.25*(1.-alpha_m+alpha_f)**2.
        gamma   = 0.5-alpha_m+alpha_f

        return alpha_m, alpha_f, beta, gamma



# Fluid mechanics time integration class
class timeintegration_fluid(timeintegration):
    
    def __init__(self, time_params, fem_params, time_curves, t_init, comm):
        timeintegration.__init__(self, time_params, time_curves, t_init, comm=comm)

        self.theta_ost = time_params['theta_ost']


    def set_acc(self, v, v_old, a_old):
        
        # set forms for acc and vel
        if self.timint == 'ost':
            acc = self.update_a_ost(v, v_old, a_old, ufl=True)
        elif self.timint == 'static':
            acc = ufl.constantvalue.zero(3)
        else:
            raise NameError("Unknown time-integration algorithm for fluid mechanics!")
        
        return acc


    def set_uf(self, v, v_old, uf_old):
        
        # set forms for acc and vel
        if self.timint == 'ost':
            uf = self.update_uf_ost(v, v_old, uf_old, ufl=True)
        elif self.timint == 'static':
            uf = ufl.constantvalue.zero(3)
        else:
            raise NameError("Unknown time-integration algorithm for fluid mechanics!")
        
        return uf


    def timefactors(self):
        
        if self.timint=='ost':    timefac_m, timefac = self.theta_ost, self.theta_ost
        if self.timint=='static': timefac_m, timefac = 1., 1.
        
        return timefac_m, timefac


    def update_timestep(self, v, v_old, a_old, p, p_old, funcs, funcs_old, funcsvec, funcsvec_old, uf_old=None):
    
        # update old fields with new quantities
        self.update_fields_ost(v, v_old, a_old, uf_old=uf_old)
        
        # update pressure variable
        p_old.vector.axpby(1.0, 0.0, p.vector)
        p_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update time dependent load curves
        self.update_time_funcs(funcs, funcs_old, funcsvec, funcsvec_old)


    def update_a_ost(self, v, v_old, a_old, ufl=True):
        # update formula for acceleration
        if ufl:
            dt_ = self.dt
            theta_ = self.theta_ost
        else:
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)
        return 1./(theta_*dt_) * (v - v_old) - (1.-theta_)/theta_ * a_old


    def update_uf_ost(self, v, v_old, uf_old, ufl=True):
        # update formula for integrated fluid displacement uf
        if ufl:
            dt_ = self.dt
            theta_ = self.theta_ost
        else:
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)
        return theta_*dt_ * v + (1.-theta_)*dt_ * v_old + uf_old


    def update_fields_ost(self, v, v_old, a_old, uf_old=None):
        # update fields at the end of each time step 
        # get vectors (references)
        v_vec, v0_vec  = v.vector, v_old.vector
        a0_vec = a_old.vector
        v_vec.assemble(), v0_vec.assemble(), a0_vec.assemble()
        
        # use update functions using vector arguments
        a_vec = self.update_a_ost(v_vec, v0_vec, a0_vec, ufl=False)
        
        # update acceleration: a_old <- a
        a_old.vector.axpby(1.0, 0.0, a_vec)
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # update velocity: v_old <- v
        v_old.vector.axpby(1.0, 0.0, v_vec)
        v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        if uf_old is not None:

            uf0_vec = uf_old.vector
            # use update functions using vector arguments
            uf_vec = self.update_uf_ost(v_vec, v0_vec, uf0_vec, ufl=False)

            # update fluid displacement: uf_old <- uf
            uf_old.vector.axpby(1.0, 0.0, uf_vec)
            uf_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            


# ALE time integration class
class timeintegration_ale(timeintegration):

    def update_timestep(self, u, u_old):
        # update state variable
        u_old.vector.axpby(1.0, 0.0, u.vector)
        u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# Flow0d time integration class
class timeintegration_flow0d(timeintegration):
    
    # initialize base class
    def __init__(self, time_params, time_curves, t_init, comm, cycle=[1], cycleerror=[1]):
        timeintegration.__init__(self, time_params, time_curves, t_init, comm=comm)
    
        self.cycle = cycle
        self.cycleerror = cycleerror
        

    # print time step info
    def print_timestep(self, N, t, separator, Nmax, wt=0):

        if self.comm.rank == 0:

            if self.cycle[0]==1: # cycle error does not make sense in first cycle
                print("### TIME STEP %i / %i successfully completed | TIME: %.4f | CYCLE: %i | CYCLE ERROR: - | wt = %.2e" % (N,Nmax,t,self.cycle[0],wt))
            else:
                print("### TIME STEP %i / %i successfully completed | TIME: %.4f | CYCLE: %i | CYCLE ERROR: %.4f | wt = %.2e" % (N,Nmax,t,self.cycle[0],self.cycleerror[0],wt))
            print(separator)
            sys.stdout.flush()



# SignallingNetwork time integration class
class timeintegration_signet(timeintegration):
    
    # initialize base class
    def __init__(self, time_params, time_curves, t_init, comm):
        timeintegration.__init__(self, time_params, time_curves, t_init, comm=comm)


    # print time step info
    def print_timestep(self, N, t, separator, Nmax, wt=0):

        if self.comm.rank == 0:

            print("### TIME STEP %i / %i successfully completed | TIME: %.4f | wt = %.2e" % (N,Nmax,t,wt))
            print(separator)
            sys.stdout.flush()
