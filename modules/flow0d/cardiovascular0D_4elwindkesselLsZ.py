#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys, os, subprocess, time
import math
import numpy as np
import sympy as sp

from cardiovascular0D import cardiovascular0Dbase
from mpiroutines import allgather_vec

# here we make use of sympy for symbolic differention...

### this implements a 4-element windkessel model with an inertance L in series to an impedance Z, all in series
# to a compliance C parallel to a resistance R, with the original equation
# C dp/dt + (p-p_ref)/R - (1+Z/R)*Q - (CZ + L/R)*dQ/dt - LC*d2Q/dt2 = 0, with Q:=-dV/dt
### this is implemented as three first order ODEs:
# C dp/dt + (p-p_ref)/R + (1+Z/R)*q + (CZ + L/R)*s + LC*ds/dt = 0
# dV/dt - q = 0
# dq/dt - s = 0

class cardiovascular0D4elwindkesselLsZ(cardiovascular0Dbase):

    def __init__(self, params, cq=['volume'], comm=None):
        # initialize base class
        cardiovascular0Dbase.__init__(self, comm=comm)
        
        # number of degrees of freedom
        self.numdof = 3
        
        # parameters
        self.R = params['R']
        self.C = params['C']
        self.Z = params['Z']
        self.L = params['L']
        self.p_ref = params['p_ref']

        self.cq = cq
        self.switch_V, self.switch_Q = 1, 0
        
        self.cname = 'V'

        self.v_ids = [0]
        self.c_ids = [1]

        if self.cq[0] == 'volume':
            pass # 'volume' is default
        elif self.cq[0] == 'flux':
            self.switch_V, self.switch_Q = 0, 1
            self.cname  = 'Q'
        else:
            raise NameError("Unknown coupling quantity!")

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
        
        self.set_solve_arrays()
    
    
    def equation_map(self):
        
        self.varmap = {'p' : 0, 'q' : 1, 's' : 2}
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
        
        # df part of rhs contribution (df - df_old)/dt
        self.df_[0] = self.C * p_ + self.L*self.C * s_
        self.df_[1] = VQ_ * self.switch_V
        self.df_[2] = q_

        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0] = (p_-self.p_ref)/self.R + (1.+self.Z/self.R) * q_ + (self.C*self.Z + self.L/self.R) * s_
        self.f_[1] = -q_ - (1-self.switch_V) * VQ_
        self.f_[2] = -s_
        
        # auxiliary vector
        self.a_[0] = 0
        self.a_[1] = VQ_
        self.a_[2] = 0
    

    def initialize(self, x, iniparam):
        
        x[0] = iniparam['p_0']
        x[1] = iniparam['q_0']
        x[2] = iniparam['s_0']


    def initialize_lm(self, var, iniparam):

        if 'p_0' in iniparam.keys(): var[0] = iniparam['p_0']


    def print_to_screen(self, x, a):
        
        if isinstance(x, np.ndarray): x_sq = x
        else: x_sq = allgather_vec(x, self.comm)

        if self.comm.rank == 0:
            
            print("Output of 0D model (4elwindkesselLsZ):")
            
            print('{:<1s}{:<3s}{:<10.3f}'.format(self.cname,' = ',a[0]))
            
            print('{:<1s}{:<3s}{:<10.3f}'.format('p',' = ',x_sq[0]))
            print('{:<1s}{:<3s}{:<10.3f}'.format('q',' = ',x_sq[1]))

            sys.stdout.flush()
