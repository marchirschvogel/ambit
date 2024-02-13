#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from petsc4py import PETSc
import ufl

from . import expression, utilities

"""
Time-integration classes for all problems
"""

class timeintegration():

    def __init__(self, time_params, dt, Nmax, time_curves=None, t_init=0., dim=3, comm=None):

        try: self.timint = time_params['timint']
        except: self.timint = 'static'

        self.dt = dt
        self.numstep = Nmax

        self.time_curves = time_curves
        self.t_init = t_init

        try: self.eval_nonlin_terms = time_params['eval_nonlin_terms']
        except: self.eval_nonlin_terms = 'trapezoidal'

        self.dim = dim

        self.comm = comm

        # time-dependent functions to update
        self.funcs_to_update, self.funcs_to_update_old, self.funcs_to_update_mid = [], [], []
        self.funcs_to_update_vec, self.funcs_to_update_vec_old, self.funcs_to_update_vec_mid = [], [], []
        # time- and potentially space-dependent expressions to update
        self.funcsexpr_to_update, self.funcsexpr_to_update_old, self.funcsexpr_to_update_mid = {}, {}, {}
        self.funcsexpr_to_update_vec, self.funcsexpr_to_update_vec_old, self.funcsexpr_to_update_vec_mid = {}, {}, {}

        # functions which are fed with external data
        self.funcs_data = []


    # print timestep info
    def print_timestep(self, N, t, lsp, ni=0, li=0, wt=0):

        msg = "### TIME STEP %i / %i completed | TIME: %.4f | ni = %i | li = %i | wt = %.2e" % (N,self.numstep,t,ni,li,wt)
        utilities.print_status(msg, self.comm)
        if lsp > len(msg): lensep = lsp
        else: lensep = len(msg)
        utilities.print_status("-"*lensep, self.comm)


    # print prestress step info
    def print_prestress_step(self, N, t, Nmax, lsp, ni=0, li=0, wt=0):

        msg = "### PRESTRESS STEP %i / %i completed | PSEUDO TIME: %.4f | ni = %i | li = %i | wt = %.2e" % (N,Nmax,t,ni,li,wt)
        utilities.print_status(msg, self.comm)
        if lsp > len(msg): lensep = lsp
        else: lensep = len(msg)
        utilities.print_status("-"*lensep, self.comm)


    def set_time_funcs(self, t, dt, funcs, funcs_vec, funcs_mid=None, funcs_vec_mid=None):

        for m in funcs_vec:
            load = expression.template_vector(dim=self.dim)
            load.val_x, load.val_y, load.val_z = list(m.values())[0][0](t), list(m.values())[0][1](t), list(m.values())[0][2](t)
            list(m.keys())[0].interpolate(load.evaluate)
            list(m.keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        for m in funcs:
            load = expression.template()
            load.val = list(m.values())[0](t)
            list(m.keys())[0].interpolate(load.evaluate)
            list(m.keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        _, timefac = self.timefactors()
        tmid = timefac*t + (1.-timefac)*(t-dt)

        if funcs_vec_mid is not None:
            for m in funcs_vec_mid:
                load = expression.template_vector(dim=self.dim)
                load.val_x, load.val_y, load.val_z = list(m.values())[0][0](tmid), list(m.values())[0][1](tmid), list(m.values())[0][2](tmid)
                list(m.keys())[0].interpolate(load.evaluate)
                list(m.keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if funcs_mid is not None:
            for m in funcs_mid:
                load = expression.template()
                load.val = list(m.values())[0](tmid)
                list(m.keys())[0].interpolate(load.evaluate)
                list(m.keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # set the time in user expressions - note that they hence must have a class variable self.t
        for m in self.funcsexpr_to_update_vec:
            self.funcsexpr_to_update_vec[m].t = t
            m.interpolate(self.funcsexpr_to_update_vec[m].evaluate)
            m.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # update time-dependent functions
    def update_time_funcs(self):

        for m in range(len(self.funcs_to_update_old)):
            list(self.funcs_to_update_old[m].keys())[0].vector.axpby(1.0, 0.0, list(self.funcs_to_update[m].keys())[0].vector)
            list(self.funcs_to_update_old[m].keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        for m in range(len(self.funcs_to_update_vec_old)):
            list(self.funcs_to_update_vec_old[m].keys())[0].vector.axpby(1.0, 0.0, list(self.funcs_to_update_vec[m].keys())[0].vector)
            list(self.funcs_to_update_vec_old[m].keys())[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # OST update formula for the time derivative of a variable
    def update_dvar_ost(self, var, var_old, dvar_old, dvarout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return 1./(theta_*dt_) * (var - var_old) - (1.-theta_)/theta_ * dvar_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            dvarout.axpby(-(1.-theta_)/theta_, 0.0, dvar_old)
            dvarout.axpy(1./(theta_*dt_), var)
            dvarout.axpy(-1./(theta_*dt_), var_old)


    # OST update formula for the second time derivative of a variable
    def update_d2var_ost(self, var, var_old, dvar_old, d2var_old, d2varout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return 1./(theta_*theta_*dt_*dt_) * (var - var_old) - 1./(theta_*theta_*dt_) * dvar_old - (1.-theta_)/theta_ * d2var_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            d2varout.axpby(-(1.-theta_)/theta_, 0.0, d2var_old)
            d2varout.axpy(-1./(theta_*theta_*dt_), dvar_old)
            d2varout.axpy(1./(theta_*theta_*dt_*dt_), var)
            d2varout.axpy(-1./(theta_*theta_*dt_*dt_), var_old)


    # Newmark update formula for the time derivative of a variable: 1st order scheme
    def update_dvar_newmark_1st(self, var, var_old, dvar_old, dvarout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            gamma_ = self.gamma
            return 1./(gamma_*dt_) * (var - var_old) - (1.-gamma_)/gamma_ * dvar_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)

            dvarout.axpby(-(1.-gamma_)/gamma_, 0.0, dvar_old)
            dvarout.axpy(1./(gamma_*dt_), var)
            dvarout.axpy(-1./(gamma_*dt_), var_old)


    # Newmark update formula for the time derivative of a variable: 2nd order scheme
    def update_dvar_newmark_2nd(self, var, var_old, dvar_old, d2var_old, dvarout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            gamma_ = self.gamma
            beta_ = self.beta
            return gamma_/(beta_*dt_) * (var - var_old) - (gamma_ - beta_)/beta_ * dvar_old - (gamma_-2.*beta_)/(2.*beta_) * dt_*d2var_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)
            beta_ = float(self.beta)

            dvarout.axpby(-(gamma_-beta_)/beta_, 0.0, dvar_old)
            dvarout.axpy(-(gamma_-2.*beta_)/(2.*beta_) * dt_, d2var_old)
            dvarout.axpy(gamma_/(beta_*dt_), var)
            dvarout.axpy(-gamma_/(beta_*dt_), var_old)


    # Newmark update formula for the second time derivative of a variable
    def update_d2var_newmark(self, var, var_old, dvar_old, d2var_old, d2varout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            beta_ = self.beta
            return 1./(beta_*dt_*dt_) * (var - var_old) - 1./(beta_*dt_) * dvar_old - (1.-2.*beta_)/(2.*beta_) * d2var_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            beta_ = float(self.beta)

            d2varout.axpby(-(1.-2.*beta_)/(2.*beta_), 0.0, d2var_old)
            d2varout.axpy(-1./(beta_*dt_), dvar_old)
            d2varout.axpy(1./(beta_*dt_*dt_), var)
            d2varout.axpy(-1./(beta_*dt_*dt_), var_old)


    # OST update formula for the first integration of a variable
    def update_varint_ost(self, var, var_old, varint_old, varintout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            theta_ = self.theta_ost
            return theta_*dt_ * var + (1.-theta_)*dt_ * var_old + varint_old
        else:
            dt_ = float(self.dt)
            theta_ = float(self.theta_ost)

            varintout.axpby(1., 0.0, varint_old)
            varintout.axpy(theta_*dt_, var)
            varintout.axpy((1.-theta_)*dt_, var_old)


    # Newmark update formula for the first integration of a variable: 1st order scheme
    def update_varint_newmark_1st(self, var, var_old, varint_old, varintout=None, ufl=True):

        if ufl: # ufl form
            dt_ = self.dt
            gamma_ = self.gamma
            return gamma_*dt_ * var + (1.-gamma_)*dt_ * var_old + varint_old
        else: # PETSc vector update
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)

            varintout.axpby(1., 0.0, varint_old)
            varintout.axpy(gamma_*dt_, var)
            varintout.axpy((1.-gamma_)*dt_, var_old)


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

    def __init__(self, time_params, dt, Nmax, fem_params, time_curves=None, t_init=0., dim=3, comm=None):
        timeintegration.__init__(self, time_params, dt, Nmax, time_curves=time_curves, t_init=t_init, dim=dim, comm=comm)

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
            acc = self.update_d2var_newmark(u, u_old, v_old, a_old, ufl=True)
            vel = self.update_dvar_newmark_2nd(u, u_old, v_old, a_old, ufl=True)
        elif self.timint == 'ost':
            acc = self.update_d2var_ost(u, u_old, v_old, a_old, ufl=True)
            vel = self.update_dvar_ost(u, u_old, v_old, a_old, ufl=True)
        elif self.timint == 'static':
            acc = ufl.constantvalue.zero(self.dim)
            vel = ufl.constantvalue.zero(self.dim)
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


    def update_fields_newmark(self, u, u_old, v, v_old, a, a_old):

        # use update functions using vector arguments
        self.update_d2var_newmark(u.vector, u_old.vector, v_old.vector, a_old.vector, d2varout=a.vector, ufl=False)
        self.update_dvar_newmark_2nd(u.vector, u_old.vector, v_old.vector, a_old.vector, dvarout=v.vector, ufl=False)

        self.update_a_v_u(a_old, v_old, u_old, a, v, u)


    def update_fields_ost(self, u, u_old, v, v_old, a, a_old):

        # use update functions using vector arguments
        self.update_d2var_ost(u.vector, u_old.vector, v_old.vector, a_old.vector, d2varout=a.vector, ufl=False)
        self.update_dvar_ost(u.vector, u_old.vector, v_old.vector, dvarout=v.vector, ufl=False)

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

    def __init__(self, time_params, dt, Nmax, fem_params, time_curves=None, t_init=0., dim=3, comm=None):
        timeintegration.__init__(self, time_params, dt, Nmax, time_curves=time_curves, t_init=t_init, dim=dim, comm=comm)

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
            acc = self.update_dvar_ost(v, v_old, a_old, ufl=True)
        elif self.timint == 'genalpha':
            acc = self.update_dvar_newmark_1st(v, v_old, a_old, ufl=True)
        else:
            raise NameError("Unknown time-integration algorithm for fluid mechanics!")

        return acc


    def set_uf(self, v, v_old, uf_old):

        # set forms for acc and vel
        if self.timint == 'ost':
            uf = self.update_varint_ost(v, v_old, uf_old, ufl=True)
        elif self.timint == 'genalpha':
            uf = self.update_varint_newmark_1st(v, v_old, uf_old, ufl=True)
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


    def update_fields_ost(self, v, v_old, a, a_old, uf=None, uf_old=None):

        # use update functions using vector arguments
        self.update_dvar_ost(v.vector, v_old.vector, a_old.vector, dvarout=a.vector, ufl=False)

        if uf_old is not None:

            # use update functions using vector arguments
            self.update_varint_ost(v.vector, v_old.vector, uf_old.vector, varintout=uf.vector, ufl=False)

        self.update_a_v(a_old, v_old, a, v, uf_old=uf_old, uf=uf)


    def update_fields_genalpha(self, v, v_old, a, a_old, uf=None, uf_old=None):

        # use update functions using vector arguments
        self.update_dvar_newmark_1st(v.vector, v_old.vector, a_old.vector, dvarout=a.vector, ufl=False)

        if uf_old is not None:

            # use update functions using vector arguments
            self.update_varint_newmark_1st(v.vector, v_old.vector, uf_old.vector, varintout=uf.vector, ufl=False)

        self.update_a_v(a_old, v_old, a, v, uf_old=uf_old, uf=uf)


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
        if self.timint == 'ost':
            self.update_fields_ost(d, d_old, w, w_old)
        if self.timint == 'genalpha':
            self.update_fields_genalpha(d, d_old, w, w_old)

        # update time dependent load curves
        self.update_time_funcs()


    def set_wel(self, d, d_old, w_old):

        # set form for domain velocity wel
        if self.timint == 'ost':
            wel = self.update_dvar_ost(d, d_old, w_old, ufl=True)
        elif self.timint == 'genalpha':
            wel = self.update_dvar_newmark_1st(d, d_old, w_old, ufl=True)
        else:
            raise NameError("Unknown time scheme for ALE mechanics!")

        return wel


    def update_fields_ost(self, d, d_old, w, w_old):

        # use update functions using vector arguments
        self.update_dvar_ost(d.vector, d_old.vector, w_old.vector, dvarout=w.vector, ufl=False)

        self.update_w_d(w_old, d_old, w, d)


    def update_fields_genalpha(self, d, d_old, w, w_old, uf=None, uf_old=None):

        # use update functions using vector arguments
        self.update_dvar_newmark_1st(d.vector, d_old.vector, w_old.vector, dvarout=w.vector, ufl=False)

        self.update_w_d(w_old, d_old, w, d)


    def update_w_d(self, w_old, d_old, w, d):
        # update ALE velocity: w_old <- w
        w_old.vector.axpby(1.0, 0.0, w.vector)
        w_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update ALE displacement: d_old <- d
        d_old.vector.axpby(1.0, 0.0, d.vector)
        d_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)



# Electrophysiology time integration class
class timeintegration_electrophysiology(timeintegration):

    def update_timestep(self, phi, phi_old, phidot, phidot_old):

        # update old fields with new quantities
        self.update_fields_ost(phi, phi_old, phidot, phidot_old)

        # update time dependent load curves
        self.update_time_funcs()


    def set_phidot(self, phi, phi_old, phidot_old):

        return self.update_dvar_ost(phi, phi_old, phidot_old, ufl=True)


    def update_fields_ost(self, phi, phi_old, phidot, phidot_old):

        # use update functions using vector arguments
        self.update_dvar_ost(phi.vector, phi_old.vector, phidot_old.vector, dvarout=phidot.vector, ufl=False)

        self.update_phidot_phi(phidot_old, phi_old, phidot, phi)


    def update_phidot_phi(self, w_old, d_old, w, d):
        # update time derivative of potential: phidot_old <- phidot
        phidot_old.vector.axpby(1.0, 0.0, phidot.vector)
        phidot_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # update potential: phi_old <- phi
        phi_old.vector.axpby(1.0, 0.0, phi.vector)
        phi_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)



# Flow0d time integration class
class timeintegration_flow0d(timeintegration):

    # initialize base class
    def __init__(self, time_params, dt, Nmax, time_curves=None, t_init=0., comm=None):
        timeintegration.__init__(self, time_params, dt, Nmax, time_curves=time_curves, t_init=t_init, comm=comm)

        self.cycle = [1]
        self.cycleerror = [1.]


    # print time step info
    def print_timestep(self, N, t, lsp, Nmax, ni=0, li=0, wt=0):

        if self.cycle[0]==1: # cycle error does not make sense in first cycle
            msg = "### TIME STEP %i / %i completed | TIME: %.4f | CYCLE: %i | CYCLE ERROR: - | ni = %i | li = %i | wt = %.2e" % (N,Nmax,t,self.cycle[0],ni,li,wt)
        else:
            msg = "### TIME STEP %i / %i completed | TIME: %.4f | CYCLE: %i | CYCLE ERROR: %.4f | ni = %i | li = %i | wt = %.2e" % (N,Nmax,t,self.cycle[0],self.cycleerror[0],ni,li,wt)
        utilities.print_status(msg, self.comm)
        if lsp > len(msg): lensep = lsp
        else: lensep = len(msg)
        utilities.print_status("-"*lensep, self.comm)



# SignallingNetwork time integration class
class timeintegration_signet(timeintegration):

    # initialize base class
    def __init__(self, time_params, dt, Nmax, time_curves=None, t_init=0., comm=None):
        timeintegration.__init__(self, time_params, dt, Nmax, time_curves=time_curves, t_init=t_init, comm=comm)


    # print time step info
    def print_timestep(self, N, t, lsp, Nmax, ni=0, li=0, wt=0):

        msg = "### TIME STEP %i / %i completed | TIME: %.4f | ni = %i | wt = %.2e" % (N,Nmax,t,ni,wt)
        utilities.print_status(msg, self.comm)
        if lsp > len(msg): lensep = lsp
        else: lensep = len(msg)
        utilities.print_status("-"*lensep, self.comm)
