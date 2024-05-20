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
2-element windkessel: C dp/dt + (p-p_ref)/R = Q, with Q:=-dV/dt (Poiseuille's flow for C=0)
(can be reproduced with 4elwindkesselLsZ by setting Z, L = 0)
"""

class cardiovascular0D2elwindkessel(cardiovascular0Dbase):

    def __init__(self, params, cq, vq, init=True, ode_par=False, comm=None):
        # initialize base class
        super().__init__(init=init, ode_par=ode_par, comm=comm)

        # number of (independent) models
        try: self.num_models = params['num_models']
        except: self.num_models = 1

        self.C, self.R, self.p_ref = [], [], []

        # parameters
        for n in range(self.num_models):
            # compliance
            try: self.C.append(params['C'+str(n+1)])
            except: self.C.append(params['C'])
            # resistance
            try: self.R.append(params['R'+str(n+1)])
            except: self.R.append(params['R'])
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

        # number of degrees of freedom - 1 per model
        self.numdof = self.num_models

        self.v_ids, self.c_ids = [], []
        self.switch_V, self.cname, self.vname = [], [], []

        for n in range(self.num_models):

            self.v_ids.append(n)
            self.c_ids.append(n)
            if self.cq[n] == 'volume':
                assert(self.vq[n]=='pressure')
                self.switch_V.append(1), self.cname.append('V'), self.vname.append('p')
            elif self.cq[n] == 'flux':
                assert(self.vq[n]=='pressure')
                self.switch_V.append(0), self.cname.append('Q'), self.vname.append('p')
            elif self.cq[n] == 'pressure':
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
            self.varmap[self.vname[n]] = n
            self.auxmap[self.cname[n]] = n

        self.t_ = sp.Symbol('t_')

        p_, VQ_ = [], []
        for n in range(self.num_models):
            p_.append(sp.Symbol('p_'+str(n+1)))
            VQ_.append(sp.Symbol('VQ_'+str(n+1)))

        # dofs to differentiate w.r.t.
        for n in range(self.num_models):
            self.x_[n] = p_[n]
            # coupling variables
            self.c_.append(VQ_[n])
        for n in range(self.num_models):
            if self.cq[n] == 'pressure': # switch Q <--> p for pressure coupling
                self.x_[n] = VQ_[n]
                self.c_[n] = p_[n]

        for n in range(self.num_models):

            # df part of rhs contribution (df - df_old)/dt
            self.df_[n] = self.C[n] * p_[n] + VQ_[n] * self.switch_V[n]

            # f part of rhs contribution theta * f + (1-theta) * f_old
            self.f_[n] = (p_[n]-self.p_ref[n])/self.R[n] - (1-self.switch_V[n]) * VQ_[n]

            # populate auxiliary variable vector
            self.a_[n] = self.c_[n]


    def initialize(self, var, iniparam):

        for n in range(self.num_models):
            try: var[n] = iniparam[self.vname[n]+str(n+1)+'_0']
            except: var[n] = iniparam[self.vname[n]+'_0']

    def initialize_lm(self, var, iniparam):

        for n in range(self.num_models):
            if 'p_0' in iniparam.keys(): var[n] = iniparam['p_0']
            if 'p'+str(n+1)+'_0' in iniparam.keys(): var[n] = iniparam['p'+str(n+1)+'_0']

    def print_to_screen(self, var, aux):

        if self.ode_parallel: var_arr = allgather_vec(var, self.comm)
        else: var_arr = var.array

        for n in range(self.num_models):
            utilities.print_status("Output of 0D model (2elwindkessel) "+str(n+1)+":", self.comm)

            utilities.print_status('{:<1s}{:<3s}{:<10.3f}'.format(self.cname[n],' = ',aux[n]), self.comm)
            utilities.print_status('{:<1s}{:<3s}{:<10.3f}'.format(self.vname[n],' = ',var_arr[n]), self.comm)
