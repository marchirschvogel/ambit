#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import sympy as sp

from .cardiovascular0D import cardiovascular0Dbase
from ..mpiroutines import allgather_vec
from .. import utilities

"""
This implements a 4-element windkessel model with an inertance L parallel to an impedance Z, all in series
to a compliance C parallel to a resistance R, with the original equation
LC/Z * d2p/dt2 + (L/(RZ) + C)*dp/dt + (p-p_ref)/R - Q - (L/R + L/Z)*dQ/dt - LC*d2Q/dt2 = 0, with Q:=-dV/dt
This is implemented as four first order ODEs:
LC/Z * dg/dt + (L/(RZ) + C)*g + (p-p_ref)/R + q + (L/R + L/Z)*s + LC*ds/dt = 0
dp/dt - g = 0
dV/dt - q = 0
dq/dt - s = 0
"""

class cardiovascular0D4elwindkesselLpZ(cardiovascular0Dbase):

    def __init__(self, params, cq, vq, init=True, ode_par=False, comm=None):
        # initialize base class
        super().__init__(init=init, ode_par=ode_par, comm=comm)

        # number of (independent) models
        try: self.num_models = params['num_models']
        except: self.num_models = 1

        self.R, self.C, self.Z, self.L, self.p_ref = [], [], [], [], []

        # parameters
        for n in range(self.num_models):
            # resistance
            try: self.R.append(params['R'+str(n+1)])
            except: self.R.append(params['R'])
            # compliance
            try: self.C.append(params['C'+str(n+1)])
            except: self.C.append(params['C'])
            # impedance
            try: self.Z.append(params['Z'+str(n+1)])
            except: self.Z.append(params['Z'])
            # inertance
            try: self.L.append(params['L'+str(n+1)])
            except: self.L.append(params['L'])
            # downstream reference pressure
            try: self.p_ref.append(params['p_ref'+str(n+1)])
            except: self.p_ref.append(params['p_ref'])

        self.cq = cq
        self.vq = vq

        # set up arrays
        self.setup_arrays()

        # set up symbolic equations
        self.equation_map()

        # symbolic stiffness matrix
        self.set_stiffness()

        # make Lambda functions out of symbolic expressions
        self.lambdify_expressions()


    def setup_arrays(self):

        # number of degrees of freedom - 4 per model
        self.numdof = 4*self.num_models

        self.v_ids, self.c_ids = [], []
        self.switch_V, self.cname, self.vname = [], [], []

        for n in range(self.num_models):

            self.v_ids.append(4*n+0)
            if self.cq[n] == 'volume':
                self.c_ids.append(4*n+2)
                assert(self.vq[0]=='pressure')
                self.switch_V.append(1), self.cname.append('V'), self.vname.append('p')
            elif self.cq[n] == 'flux':
                self.c_ids.append(4*n+2)
                assert(self.vq[n]=='pressure')
                self.switch_V.append(0), self.cname.append('Q'), self.vname.append('p')
            elif self.cq[n] == 'pressure':
                self.c_ids.append(n)
                if self.vq[n] == 'flux':
                    self.switch_V.append(0), self.cname.append('p'), self.vname.append('Q')
                elif self.vq[n] == 'volume':
                    self.switch_V.append(1), self.cname.append('p'), self.vname.append('V')
                else:
                    raise ValueError("Unknown variable quantity!")
            else:
                raise NameError("Unknown coupling quantity!")

        self.set_solve_arrays()


    def equation_map(self):

        self.varmap, self.auxmap = {}, {}
        for n in range(self.num_models):
            self.varmap[self.vname[n]] = 4*n+0
            self.varmap['g'] = 4*n+1
            self.varmap['q'] = 4*n+2
            self.varmap['s'] = 4*n+3
            self.auxmap[self.cname[n]] = 4*n+2

        self.t_ = sp.Symbol('t_')

        p_, g_, q_, s_, VQ_ = [], [], [], [], []
        for n in range(self.num_models):
            p_.append(sp.Symbol('p_'+str(n+1)))
            g_.append(sp.Symbol('g_'+str(n+1)))
            q_.append(sp.Symbol('q_'+str(n+1)))
            s_.append(sp.Symbol('s_'+str(n+1)))
            VQ_.append(sp.Symbol('VQ_'+str(n+1)))

        # dofs to differentiate w.r.t.
        for n in range(self.num_models):
            self.x_[4*n+0] = p_[n]
            self.x_[4*n+1] = g_[n]
            self.x_[4*n+2] = q_[n]
            self.x_[4*n+3] = s_[n]
            # coupling variables
            self.c_.append(VQ_[n])
        for n in range(self.num_models):
            if self.cq[n] == 'pressure': # switch Q <--> p for pressure coupling
                self.x_[4*n+0] = VQ_[n]
                self.c_[n] = p_[n]

        for n in range(self.num_models):

            # df part of rhs contribution (df - df_old)/dt
            self.df_[4*n+0] = self.L[n]*self.C[n]/self.Z[n] * g_[n] + self.L[n]*self.C[n] * s_[n]
            self.df_[4*n+1] = p_[n]
            self.df_[4*n+2] = VQ_[n] * self.switch_V[n]
            self.df_[4*n+3] = q_[n]

            # f part of rhs contribution theta * f + (1-theta) * f_old
            self.f_[4*n+0] = (self.L[n]/(self.R[n]*self.Z[n]) + self.C[n]) * g_[n] + (p_[n]-self.p_ref[n])/self.R[n] + q_[n] + (self.L[n]/self.R[n] + self.L[n]/self.Z[n]) * s_[n]
            self.f_[4*n+1] = -g_[n]
            self.f_[4*n+2] = -q_[n] - (1-self.switch_V[n]) * VQ_[n]
            self.f_[4*n+3] = -s_[n]

            # populate auxiliary variable vector
            self.a_[4*n+0] = self.c_[n]


    def initialize(self, var, iniparam):

        for n in range(self.num_models):
            try: var[4*n+0] = iniparam[self.vname[n]+str(n+1)+'_0']
            except: var[4*n+0] = iniparam[self.vname[n]+'_0']
            try: var[4*n+1] = iniparam['g'+str(n+1)+'_0']
            except: var[4*n+1] = iniparam['g_0']
            try: var[4*n+2] = iniparam['q'+str(n+1)+'_0']
            except: var[4*n+2] = iniparam['q_0']
            try: var[4*n+3] = iniparam['s'+str(n+1)+'_0']
            except: var[4*n+3] = iniparam['s_0']


    def initialize_lm(self, var, iniparam):

        for n in range(self.num_models):
            if 'p_0' in iniparam.keys(): var[n] = iniparam['p_0']
            if 'p'+str(n+1)+'_0' in iniparam.keys(): var[n] = iniparam['p'+str(n+1)+'_0']


    def print_to_screen(self, var, aux):

        if self.ode_parallel: var_arr = allgather_vec(var, self.comm)
        else: var_arr = var.array

        for n in range(self.num_models):
            utilities.print_status("Output of 0D model (4elwindkesselLpZ) "+str(n+1)+":", self.comm)

            utilities.print_status('{:<1s}{:<3s}{:<10.3f}'.format(self.cname[n],' = ',aux[4*n+0]), self.comm)

            utilities.print_status('{:<1s}{:<3s}{:<10.3f}'.format(self.vname[n],' = ',var_arr[4*n+0]), self.comm)
            utilities.print_status('{:<1s}{:<3s}{:<10.3f}'.format('g',' = ',var_arr[4*n+1]), self.comm)
            utilities.print_status('{:<1s}{:<3s}{:<10.3f}'.format('q',' = ',var_arr[4*n+2]), self.comm)
