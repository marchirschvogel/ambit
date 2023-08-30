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

# two RC models (2-element Windkessels) in series linking an in- to an outflow

class cardiovascular0DCRLinoutlink(cardiovascular0Dbase):

    def __init__(self, params, cq, vq, init=True, comm=None):
        # initialize base class
        super().__init__(init=init, comm=comm)

        # only these options allowed for this model
        assert(all(i=='pressure' for i in cq))
        assert(all(i=='flux' for i in vq))

        # parameters
        self.C_in = params['C_in']
        self.R_in = params['R_in']
        self.L_in = params['L_in']

        self.C_out = params['C_out']
        self.R_out = params['R_out']
        self.L_out = params['L_out']

        self.v_ids = [0,3]
        self.c_ids = [0,1]

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
        self.numdof = 4

        self.vname, self.cname = ['q_in','q_d','p_d','q_out'], ['p_in','p_out']

        self.set_solve_arrays()


    def equation_map(self):

        self.varmap['q_in']  = 0
        self.varmap['q_d']   = 1
        self.varmap['p_d']   = 2
        self.varmap['q_out'] = 3

        self.auxmap['p_in']  = 0
        self.auxmap['p_out'] = 1

        self.t_ = sp.Symbol('t_')
        q_in_   = sp.Symbol('q_in_')
        q_d_    = sp.Symbol('q_d_')
        p_d_    = sp.Symbol('p_d_')
        q_out_  = sp.Symbol('q_out_')

        # dofs to differentiate w.r.t.
        self.x_[self.varmap['q_in']]  = q_in_
        self.x_[self.varmap['q_d']]   = q_d_
        self.x_[self.varmap['p_d']]   = p_d_
        self.x_[self.varmap['q_out']] = q_out_

        p_in_ = sp.Symbol('p_in_')
        p_out_ = sp.Symbol('p_out_')

        # coupling variables
        self.c_.append(p_in_)
        self.c_.append(p_out_)

        # df part of rhs contribution (df - df_old)/dt
        self.df_[0] = self.C_in * p_in_
        self.df_[1] = (self.L_in/self.R_in) * q_d_
        self.df_[2] = self.C_out * p_d_
        self.df_[3] = (self.L_out/self.R_out) * q_out_

        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0] = q_d_ - q_in_
        self.f_[1] = (p_d_ - p_in_)/self.R_in + q_d_
        self.f_[2] = q_out_ - q_d_
        self.f_[3] = (p_out_ - p_d_)/self.R_out + q_out_

        # populate auxiliary variable vector
        self.a_[0] = self.c_[0]
        self.a_[1] = self.c_[1]


    def initialize(self, var, iniparam):

        for i in range(len(self.vname)):
            var[i] = iniparam[self.vname[i]+'_0']


    def initialize_lm(self, var, iniparam):

        if 'p_in_0' in iniparam.keys(): var[0] = iniparam['p_in_0']
        if 'p_out_0' in iniparam.keys(): var[1] = iniparam['p_out_0']


    def print_to_screen(self, var, aux):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        if self.comm.rank == 0:

            print("Output of 0D model (CRLinoutlink):")

            for i in range(len(self.cname)):
                print('{:<5s}{:<3s}{:<10.3f}'.format(self.cname[i],' = ',aux[self.auxmap[self.cname[i]]]))
            for i in range(len(self.vname)):
                print('{:<5s}{:<3s}{:<10.3f}'.format(self.vname[i],' = ',var_sq[self.varmap[self.vname[i]]]))

            sys.stdout.flush()

        if not isinstance(var, np.ndarray): del var_sq
