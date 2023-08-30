#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from petsc4py import PETSc
import ufl

from . import expression


class timeintegration():

    def __init__(self, time_params, time_curves=None, t_init=0., comm=None):

        try: self.timint = time_params['timint']
        except: self.timint = 'static'

        if 'numstep' in time_params.keys(): self.numstep = time_params['numstep']
        if 'maxtime' in time_params.keys(): self.maxtime = time_params['maxtime']

        if 'maxtime' in time_params.keys(): self.dt = self.maxtime/self.numstep

        self.time_curves = time_curves
        self.t_init = t_init

        self.comm = comm

        # time-dependent functions to update
        self.funcs_to_update, self.funcs_to_update_old, self.funcs_to_update_vec, self.funcs_to_update_vec_old = [], [], [], []
        # functions which are fed with external data
        self.funcs_data = []


    # print timestep info
    def print_timestep(self, N, t, lsp, ni=0, li=0, wt=0):

        if self.comm.rank == 0:
            msg = "### TIME STEP %i / %i completed | TIME: %.4f | ni = %i | li = %i | wt = %.2e" % (N,self.numstep,t,ni,li,wt)
            print(msg)
            if lsp > len(msg): lensep = lsp
            else: lensep = len(msg)
            print("-"*lensep)
            sys.stdout.flush()


    # print prestress step info
    def print_prestress_step(self, N, t, Nmax, lsp, ni=0, li=0, wt=0):

        if self.comm.rank == 0:
            msg = "### PRESTRESS STEP %i / %i completed | PSEUDO TIME: %.4f | ni = %i | li = %i | wt = %.2e" % (N,Nmax,t,ni,li,wt)
            print(msg)
            if lsp > len(msg): lensep = lsp
            else: lensep = len(msg)
            print("-"*lensep)
            sys.stdout.flush()


    def set_time_funcs(self, t, funcs, funcs_vec):

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


    def update_time_funcs(self):

        # update time-dependent functions
        for m in range(len(self.funcs_to_update_old)):
            list(self.funcs_to_update_old[m].keys())[0].vector.axpby(1.0, 0.0, list(self.funcs_to_update[m].keys())[0].vector)
            list(self.funcs_to_update_old[m].keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        for m in range(len(self.funcs_to_update_vec_old)):
            list(self.funcs_to_update_vec_old[m].keys())[0].vector.axpby(1.0, 0.0, list(self.funcs_to_update_vec[m].keys())[0].vector)
            list(self.funcs_to_update_vec_old[m].keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # zero
    def zero(self, t):
        return 0.0

    def timecurves(self, cnum):

        if cnum==0:  return self.zero
        if cnum==1:  return self.time_curves.tc1
        if cnum==2:  return self.time_curves.tc2
        if cnum==3:  return self.time_curves.tc3
        if cnum==4:  return self.time_curves.tc4
        if cnum==5:  return self.time_curves.tc5
        if cnum==6:  return self.time_curves.tc6
        if cnum==7:  return self.time_curves.tc7
        if cnum==8:  return self.time_curves.tc8
        if cnum==9:  return self.time_curves.tc9
        if cnum==10: return self.time_curves.tc10
        if cnum==11: return self.time_curves.tc11
        if cnum==12: return self.time_curves.tc12
        if cnum==13: return self.time_curves.tc13
        if cnum==14: return self.time_curves.tc14
        if cnum==15: return self.time_curves.tc15
        if cnum==16: return self.time_curves.tc16
        if cnum==17: return self.time_curves.tc17
        if cnum==18: return self.time_curves.tc18
        if cnum==19: return self.time_curves.tc19
        if cnum==20: return self.time_curves.tc20


# Solid mechanics time integration class
class timeintegration_solid(timeintegration):

    def __init__(self, time_params, fem_params, time_curves=None, t_init=0., comm=None):
        timeintegration.__init__(self, time_params, time_curves=time_curves, t_init=t_init, comm=comm)

        if self.timint == 'genalpha':

            # if the spectral radius, rho_inf_genalpha, is specified, the parameters are computed from it
            try:
                self.rho_inf_genalpha = time_params['rho_inf_genalpha']
                self.alpha_m, self.alpha_f, self.beta, self.gamma = self.compute_genalpha_params(self.rho_inf_genalpha)
            # otherwise, user can specify each parameter individually
            except:
                self.alpha_m = time_params['alpha_m']
                self.alpha_f = time_params['alpha_f']
                self.beta = time_params['beta']
                self.gamma = time_params['gamma']

        if self.timint == 'ost':

            self.theta_ost = time_params['theta_ost']

        try: self.incompressible_2field = fem_params['incompressible_2field']
        except: self.incompressible_2field = False


    def set_acc_vel(self, u, u_old, v_old, a_old):

        # set forms for acc and vel
        if self.timint == 'genalpha':
            acc = self.update_a_newmark(u, u_old, v_old, a_old, ufl=True)
            vel = self.update_v_newmark(u, u_old, v_old, a_old, ufl=True)
        elif self.timint == 'ost':
            acc = self.update_a_ost(u, u_old, v_old, a_old, ufl=True)
            vel = self.update_v_ost(u, u_old, v_old, a_old, ufl=True)
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


    def update_timestep(self, u, u_old, v, v_old, a, a_old, p, p_old, internalvars, internalvars_old):

        # now update old kinematic fields with new quantities
        if self.timint == 'genalpha':
            self.update_fields_newmark(u, u_old, v, v_old, a, a_old)
        if self.timint == 'ost':
            self.update_fields_ost(u, u_old, v, v_old, a, a_old)

        # update pressure variable
        if self.incompressible_2field:
            p_old.vector.axpby(1.0, 0.0, p.vector)
            p_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update internal variables (e.g. active stress, growth stretch, plastic strains, ...)
        for i in range(len(internalvars_old)):
            list(internalvars_old.values())[i].vector.axpby(1.0, 0.0, list(internalvars.values())[i].vector)
            list(internalvars_old.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update time dependent load curves
        self.update_time_funcs()


    def update_a_newmark(self, u, u_old, v_old, a_old, aout=None, ufl=True):
        # update formula for acceleration
        if ufl: # ufl form
            dt_ = self.dt
            beta_ = self.beta
            return 1./(beta_*dt_*dt_) * (u - u_old) - 1./(beta_*dt_) * v_old - (1.-2.*beta_)/(2.*beta_) * a_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            beta_ = float(self.beta)

            aout.axpby(-(1.-2.*beta_)/(2.*beta_), 0.0, a_old)
            aout.axpy(-1./(beta_*dt_), v_old)
            aout.axpy(1./(beta_*dt_*dt_), u)
            aout.axpy(-1./(beta_*dt_*dt_), u_old)


    def update_a_ost(self, u, u_old, v_old, a_old, aout=None, ufl=True):
        # update formula for acceleration
        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return 1./(theta_*theta_*dt_*dt_) * (u - u_old) - 1./(theta_*theta_*dt_) * v_old - (1.-theta_)/theta_ * a_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            aout.axpby(-(1.-theta_)/theta_, 0.0, a_old)
            aout.axpy(-1./(theta_*theta_*dt_), v_old)
            aout.axpy(1./(theta_*theta_*dt_*dt_), u)
            aout.axpy(-1./(theta_*theta_*dt_*dt_), u_old)


    def update_v_newmark(self, u, u_old, v_old, a_old, vout=None, ufl=True):
        # update formula for velocity
        if ufl: # ufl form
            dt_ = self.dt
            gamma_ = self.gamma
            beta_ = self.beta
            return gamma_/(beta_*dt_) * (u - u_old) - (gamma_ - beta_)/beta_ * v_old - (gamma_-2.*beta_)/(2.*beta_) * dt_*a_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)
            beta_ = float(self.beta)

            vout.axpby(-(gamma_-beta_)/beta_, 0.0, v_old)
            vout.axpy(-(gamma_-2.*beta_)/(2.*beta_) * dt_, a_old)
            vout.axpy(gamma_/(beta_*dt_), u)
            vout.axpy(-gamma_/(beta_*dt_), u_old)


    def update_v_ost(self, u, u_old, v_old, a_old, vout=None, ufl=True):
        # update formula for velocity
        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return 1./(theta_*dt_) * (u - u_old) - (1. - theta_)/theta_ * v_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            vout.axpby(-(1.-theta_)/theta_, 0.0, v_old)
            vout.axpy(1./(theta_*dt_), u)
            vout.axpy(-1./(theta_*dt_), u_old)


    def update_fields_newmark(self, u, u_old, v, v_old, a, a_old):

        # use update functions using vector arguments
        self.update_a_newmark(u.vector, u_old.vector, v_old.vector, a_old.vector, aout=a.vector, ufl=False)
        self.update_v_newmark(u.vector, u_old.vector, v_old.vector, a_old.vector, vout=v.vector, ufl=False)

        self.update_a_v_u(a_old, v_old, u_old, a, v, u)


    def update_fields_ost(self, u, u_old, v, v_old, a, a_old):

        # use update functions using vector arguments
        self.update_a_ost(u.vector, u_old.vector, v_old.vector, a_old.vector, aout=a.vector, ufl=False)
        self.update_v_ost(u.vector, u_old.vector, v_old.vector, a_old.vector, vout=v.vector, ufl=False)

        self.update_a_v_u(a_old, v_old, u_old, a, v, u)


    def update_a_v_u(self, a_old, v_old, u_old, a, v, u):

        # update acceleration: a_old <- a
        a_old.vector.axpby(1.0, 0.0, a.vector)
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update velocity: v_old <- v
        v_old.vector.axpby(1.0, 0.0, v.vector)
        v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update displacement: u_old <- u
        u_old.vector.axpby(1.0, 0.0, u.vector)
        u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def compute_genalpha_params(self, rho_inf): # cf. Chung and Hulbert (1993)

        alpha_m = (2.*rho_inf-1.)/(rho_inf+1.)
        alpha_f = rho_inf/(rho_inf+1.)
        beta    = 0.25*(1.-alpha_m+alpha_f)**2.
        gamma   = 0.5-alpha_m+alpha_f

        return alpha_m, alpha_f, beta, gamma



# Fluid mechanics time integration class
class timeintegration_fluid(timeintegration):

    def __init__(self, time_params, fem_params, time_curves=None, t_init=0., comm=None):
        timeintegration.__init__(self, time_params, time_curves=time_curves, t_init=t_init, comm=comm)

        if self.timint == 'ost':

            self.theta_ost = time_params['theta_ost']

        if self.timint == 'genalpha':

            # if the spectral radius, rho_inf_genalpha, is specified, the parameters are computed from it
            try:
                self.rho_inf_genalpha = time_params['rho_inf_genalpha']
                self.alpha_m, self.alpha_f, self.gamma = self.compute_genalpha_params(self.rho_inf_genalpha)
            # otherwise, user can specify each parameter individually
            except:
                self.alpha_m = time_params['alpha_m']
                self.alpha_f = time_params['alpha_f']
                self.gamma = time_params['gamma']


    def set_acc(self, v, v_old, a_old):

        # set forms for acc and vel
        if self.timint == 'ost':
            acc = self.update_a_ost(v, v_old, a_old, ufl=True)
        elif self.timint == 'genalpha':
            acc = self.update_a_genalpha(v, v_old, a_old, ufl=True)
        else:
            raise NameError("Unknown time-integration algorithm for fluid mechanics!")

        return acc


    def set_uf(self, v, v_old, uf_old):

        # set forms for acc and vel
        if self.timint == 'ost':
            uf = self.update_uf_ost(v, v_old, uf_old, ufl=True)
        elif self.timint == 'genalpha':
            uf = self.update_uf_genalpha(v, v_old, uf_old, ufl=True)
        else:
            raise NameError("Unknown time-integration algorithm for fluid mechanics!")

        return uf


    def timefactors(self):

        if self.timint=='ost':      timefac_m, timefac = self.theta_ost, self.theta_ost
        if self.timint=='genalpha': timefac_m, timefac = self.alpha_m, self.alpha_f # note the different definition compared to solid mechanics, where alpha_(.) <- 1.-alpha_(.)

        return timefac_m, timefac


    def update_timestep(self, v, v_old, a, a_old, p, p_old, internalvars, internalvars_old, uf=None, uf_old=None):

        # update old fields with new quantities
        if self.timint == 'ost':
            self.update_fields_ost(v, v_old, a, a_old, uf=uf, uf_old=uf_old)
        if self.timint == 'genalpha':
            self.update_fields_genalpha(v, v_old, a, a_old, uf=uf, uf_old=uf_old)

        # update pressure variable
        p_old.vector.axpby(1.0, 0.0, p.vector)
        # for duplicate p-nodes, our combined (nested) pressure is not ghosted, but the subvecs
        try:
            p_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        except:
            subvecs = p_old.vector.getNestSubVecs()
            for j in range(len(subvecs)): subvecs[j].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update internal variables (e.g. active stress for reduced solid)
        for i in range(len(internalvars_old)):
            list(internalvars_old.values())[i].vector.axpby(1.0, 0.0, list(internalvars.values())[i].vector)
            list(internalvars_old.values())[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update time dependent load curves
        self.update_time_funcs()


    def update_a_ost(self, v, v_old, a_old, aout=None, ufl=True):
        # update formula for acceleration
        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return 1./(theta_*dt_) * (v - v_old) - (1.-theta_)/theta_ * a_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            aout.axpby(-(1.-theta_)/theta_, 0.0, a_old)
            aout.axpy(1./(theta_*dt_), v)
            aout.axpy(-1./(theta_*dt_), v_old)


    def update_a_genalpha(self, v, v_old, a_old, aout=None, ufl=True):
        # update formula for acceleration
        if ufl: # ufl form
            dt_ = self.dt
            gamma_ = self.gamma
            return 1./(gamma_*dt_) * (v - v_old) - (1.-gamma_)/gamma_ * a_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)

            aout.axpby(-(1.-gamma_)/gamma_, 0.0, a_old)
            aout.axpy(1./(gamma_*dt_), v)
            aout.axpy(-1./(gamma_*dt_), v_old)


    def update_uf_ost(self, v, v_old, uf_old, ufout=None, ufl=True):
        # update formula for integrated fluid displacement uf
        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return theta_*dt_ * v + (1.-theta_)*dt_ * v_old + uf_old
        else:
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            ufout.axpby(1., 0.0, uf_old)
            ufout.axpy(theta_*dt_, v)
            ufout.axpy((1.-theta_)*dt_, v_old)


    def update_uf_genalpha(self, v, v_old, uf_old, ufout=None, ufl=True):
        # update formula for integrated fluid displacement uf
        if ufl: # ufl form
            dt_ = self.dt
            gamma_ = self.gamma
            return gamma_*dt_ * v + (1.-gamma_)*dt_ * v_old + uf_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)

            ufout.axpby(1., 0.0, uf_old)
            ufout.axpy(gamma_*dt_, v)
            ufout.axpy((1.-gamma_)*dt_, v_old)


    def update_fields_ost(self, v, v_old, a, a_old, uf=None, uf_old=None):

        # use update functions using vector arguments
        self.update_a_ost(v.vector, v_old.vector, a_old.vector, aout=a.vector, ufl=False)

        if uf_old is not None:

            # use update functions using vector arguments
            self.update_uf_ost(v.vector, v_old.vector, uf_old.vector, ufout=uf.vector, ufl=False)

        self.update_a_v(a_old, v_old, a, v, uf_old=uf_old, uf=uf)


    def update_fields_genalpha(self, v, v_old, w, a_old, uf_old=None):

        # use update functions using vector arguments
        self.update_a_genalpha(v.vector, v_old.vector, a_old.vector, aout=a.vector, ufl=False)

        if uf_old is not None:

            # use update functions using vector arguments
            self.update_uf_genalpha(v.vector, v_old.vector, uf_old.vector, ufout=uf.vector, ufl=False)

        self.update_a_v(a_old, v_old, a_vec, v, uf_old=uf_old, uf=uf)


    def update_a_v(self, a_old, v_old, a, v, uf_old=None, uf=None):
        # update acceleration: a_old <- a
        a_old.vector.axpby(1.0, 0.0, a.vector)
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update velocity: v_old <- v
        v_old.vector.axpby(1.0, 0.0, v.vector)
        v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if uf_old is not None:

            # update fluid displacement: uf_old <- uf
            uf_old.vector.axpby(1.0, 0.0, uf.vector)
            uf_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def compute_genalpha_params(self, rho_inf): # cf. Jansen et al. (2000)

        # note that for solid, the alphas are differently defined: alpha_(.) <- 1.-alpha_(.)
        alpha_m = 0.5*(3.-rho_inf)/(1.+rho_inf)
        alpha_f = 1./(1.+rho_inf)
        gamma   = 0.5+alpha_m-alpha_f

        return alpha_m, alpha_f, gamma


# ALE time integration class
class timeintegration_ale(timeintegration_fluid):

    def update_timestep(self, d, d_old, w, w_old):

        # update old fields with new quantities
        self.update_fields(d, d_old, w, w_old)

        # update time dependent load curves
        self.update_time_funcs()


    def set_wel(self, d, d_old, w_old):

        # set form for domain velocity wel
        if self.timint == 'ost':
            wel = self.update_w_ost(d, d_old, w_old, ufl=True)
        else:
            raise NameError("Unknown time scheme for ALE mechanics!")

        return wel


    def update_w_ost(self, d, d_old, w_old, wout=None, ufl=True):
        # update formula for domain velocity
        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return 1./(theta_*dt_) * (d - d_old) - (1.-theta_)/theta_ * w_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            wout.axpby(-(1.-theta_)/theta_, 0.0, w_old)
            wout.axpy(1./(theta_*dt_), d)
            wout.axpy(-1./(theta_*dt_), d_old)


    def update_fields(self, d, d_old, w, w_old):

        # use update functions using vector arguments
        self.update_w_ost(d.vector, d_old.vector, w_old.vector, wout=w.vector, ufl=False)

        # update velocity: w_old <- w
        w_old.vector.axpby(1.0, 0.0, w.vector)
        w_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update displacement: d_old <- d
        d_old.vector.axpby(1.0, 0.0, d.vector)
        d_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)



# Flow0d time integration class
class timeintegration_flow0d(timeintegration):

    # initialize base class
    def __init__(self, time_params, time_curves=None, t_init=0., comm=None, cycle=[1], cycleerror=[1]):
        timeintegration.__init__(self, time_params, time_curves=time_curves, t_init=t_init, comm=comm)

        self.cycle = cycle
        self.cycleerror = cycleerror


    # print time step info
    def print_timestep(self, N, t, lsp, Nmax, ni=0, li=0, wt=0):

        if self.comm.rank == 0:

            if self.cycle[0]==1: # cycle error does not make sense in first cycle
                msg = "### TIME STEP %i / %i completed | TIME: %.4f | CYCLE: %i | CYCLE ERROR: - | ni = %i | li = %i | wt = %.2e" % (N,Nmax,t,self.cycle[0],ni,li,wt)
            else:
                msg = "### TIME STEP %i / %i completed | TIME: %.4f | CYCLE: %i | CYCLE ERROR: %.4f | ni = %i | li = %i | wt = %.2e" % (N,Nmax,t,self.cycle[0],self.cycleerror[0],ni,li,wt)
            print(msg)
            if lsp > len(msg): lensep = lsp
            else: lensep = len(msg)
            print("-"*lensep)
            sys.stdout.flush()



# SignallingNetwork time integration class
class timeintegration_signet(timeintegration):

    # initialize base class
    def __init__(self, time_params, time_curves=None, t_init=0., comm=None):
        timeintegration.__init__(self, time_params, time_curves=time_curves, t_init=t_init, comm=comm)


    # print time step info
    def print_timestep(self, N, t, lsp, Nmax, ni=0, li=0, wt=0):

        if self.comm.rank == 0:

            msg = "### TIME STEP %i / %i completed | TIME: %.4f | ni = %i | wt = %.2e" % (N,Nmax,t,ni,wt)
            print(msg)
            if lsp > len(msg): lensep = lsp
            else: lensep = len(msg)
            print("-"*lensep)
            sys.stdout.flush()
