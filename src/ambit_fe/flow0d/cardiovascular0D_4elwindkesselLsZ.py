#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import sympy as sp

from .cardiovascular0D import cardiovascular0Dbase
from ..mpiroutines import allgather_vec

# here we make use of sympy for symbolic differention...

### this implements a 4-element windkessel model with an inertance L in series to an impedance Z, all in series
# to a compliance C parallel to a resistance R, with the original equation
# C dp/dt + (p-p_ref)/R - (1+Z/R)*Q - (CZ + L/R)*dQ/dt - LC*d2Q/dt2 = 0, with Q:=-dV/dt
### this is implemented as three first order ODEs:
# C dp/dt + (p-p_ref)/R + (1+Z/R)*q + (CZ + L/R)*s + LC*ds/dt = 0
# dV/dt - q = 0
# dq/dt - s = 0

class cardiovascular0D4elwindkesselLsZ(cardiovascular0Dbase):

    def __init__(self, params, cq, vq, init=True, comm=None):
        # initialize base class
        super().__init__(init=init, comm=comm)

        # number of degrees of freedom
        self.numdof = 3

        # parameters
        self.R = params['R']
        self.C = params['C']
        self.Z = params['Z']
        self.L = params['L']
        self.p_ref = params['p_ref']

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

        # number of degrees of freedom
        self.numdof = 3

        self.v_ids = [0]
        self.c_ids = [1]

        if self.cq[0] == 'volume':
            assert(self.vq[0]=='pressure')
            self.switch_V, self.cname, self.vname = 1, 'V', 'p'
        elif self.cq[0] == 'flux':
            assert(self.vq[0]=='pressure')
            self.switch_V, self.cname, self.vname = 0, 'Q', 'p'
        elif self.cq[0] == 'pressure':
            if self.vq[0] == 'flux':
                self.switch_V, self.cname, self.vname = 0, 'p', 'Q'
            elif self.vq[0] == 'volume':
                self.switch_V, self.cname, self.vname = 1, 'p', 'V'
            else:
                raise ValueError("Unknown variable quantity!")
        else:
            raise NameError("Unknown coupling quantity!")

        self.set_solve_arrays()


    def equation_map(self):

        self.varmap = {self.vname : 0, 'q' : 1, 's' : 2}
        self.auxmap = {self.cname : 1}

        self.t_ = sp.Symbol('t_')
        p_ = sp.Symbol('p_')
        q_ = sp.Symbol('q_')
        s_ = sp.Symbol('s_')
        VQ_ = sp.Symbol('VQ_')

        # dofs to differentiate w.r.t.
        self.x_[0] = p_
        self.x_[1] = q_
        self.x_[2] = s_
        # coupling variables
        self.c_.append(VQ_)
        if self.cq[0] == 'pressure': # switch Q <--> p for pressure coupling
            self.x_[0] = VQ_
            self.c_[0] = p_

        # df part of rhs contribution (df - df_old)/dt
        self.df_[0] = self.C * p_ + self.L*self.C * s_
        self.df_[1] = VQ_ * self.switch_V
        self.df_[2] = q_

        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0] = (p_-self.p_ref)/self.R + (1.+self.Z/self.R) * q_ + (self.C*self.Z + self.L/self.R) * s_
        self.f_[1] = -q_ - (1-self.switch_V) * VQ_
        self.f_[2] = -s_

        # populate auxiliary variable vector
        self.a_[0] = self.c_[0]


    def initialize(self, var, iniparam):

        var[0] = iniparam[self.vname+'_0']
        var[1] = iniparam['q_0']
        var[2] = iniparam['s_0']


    def initialize_lm(self, var, iniparam):

        if 'p_0' in iniparam.keys(): var[0] = iniparam['p_0']


    def print_to_screen(self, var, aux):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        if self.comm.rank == 0:

            print("Output of 0D model (4elwindkesselLsZ):")

            print('{:<1s}{:<3s}{:<10.3f}'.format(self.cname,' = ',aux[0]))

            print('{:<1s}{:<3s}{:<10.3f}'.format(self.vname,' = ',var_sq[0]))
            print('{:<1s}{:<3s}{:<10.3f}'.format('q',' = ',var_sq[1]))

            sys.stdout.flush()

        if not isinstance(var, np.ndarray): del var_sq
