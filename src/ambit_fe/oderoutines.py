#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time
import numpy as np
import sympy as sp

from .mpiroutines import allgather_vec, allgather_vec_entry


class ode:

    def __init__(self, init=True, comm=None):
        self.init = init # for output
        self.varmap, self.auxmap = {}, {} # maps for primary and auxiliary variables
        if comm is not None: self.comm = comm # MPI communicator


    # evaluate model at current nonlinear iteration
    def evaluate(self, x, t, df=None, f=None, dK=None, K=None, c=[], y=[], a=None, fnc=[]):

        if isinstance(x, np.ndarray): x_sq = x
        else: x_sq = allgather_vec(x, self.comm)

        # ODE lhs (time derivative) residual part df
        if df is not None:

            for i in range(self.numdof):
                df[i] = self.df__[i](x_sq, c, t, fnc)

        # ODE rhs residual part f
        if f is not None:

            for i in range(self.numdof):
                f[i] = self.f__[i](x_sq, c, t, fnc)

        # ODE lhs (time derivative) stiffness part dK (ddf/dx)
        if dK is not None:

            for i in range(self.numdof):
                for j in range(self.numdof):
                    dK[i,j] = self.dK__[i][j](x_sq, c, t, fnc)

        # ODE rhs stiffness part K (df/dx)
        if K is not None:

            for i in range(self.numdof):
                for j in range(self.numdof):
                    K[i,j] = self.K__[i][j](x_sq, c, t, fnc)

        # auxiliary variable vector a (for post-processing or periodic state check)
        if a is not None:

            for i in range(self.numdof):
                a[i] = self.a__[i](x_sq, c, t, fnc)

        # deallocate sequential array if x was a PETSc vector that has been gathered
        if not isinstance(x, np.ndarray): del x_sq


    # symbolic stiffness matrix contributions ddf_/dx, df_/dx
    def set_stiffness(self):

        for i in range(self.numdof):
            for j in range(self.numdof):

                self.dK_[i][j] = sp.diff(self.df_[i],self.x_[j])
                self.K_[i][j]  = sp.diff(self.f_[i],self.x_[j])


    # make Lambda functions out of symbolic Sympy expressions
    def lambdify_expressions(self):

        if self.comm.rank == 0:
            print("ODE model: Calling lambdify for expressions...")
            sys.stdout.flush()

        ts = time.time()

        for i in range(self.numdof):
            self.df__[i] = sp.lambdify([self.x_, self.c_, self.t_, self.fnc_], self.df_[i], 'numpy')
            self.f__[i] = sp.lambdify([self.x_, self.c_, self.t_, self.fnc_], self.f_[i], 'numpy')
            self.a__[i] = sp.lambdify([self.x_, self.c_, self.t_, self.fnc_], self.a_[i], 'numpy')

        te = time.time() - ts

        if self.comm.rank == 0:
            print("ODE model: Finished lambdify for residual expressions, %.4f s" % (te))
            sys.stdout.flush()

        ts = time.time()

        for i in range(self.numdof):
            for j in range(self.numdof):
                if self.dK_[i][j] is not sp.S.Zero: self.dK__[i][j] = sp.lambdify([self.x_, self.c_, self.t_, self.fnc_], self.dK_[i][j], 'numpy')
                else:                               self.dK__[i][j] = lambda a, b, c, d : 0
                if self.K_[i][j] is not sp.S.Zero:  self.K__[i][j] = sp.lambdify([self.x_, self.c_, self.t_, self.fnc_], self.K_[i][j], 'numpy')
                else:                               self.K__[i][j] = lambda a, b, c, d : 0

        te = time.time() - ts

        if self.comm.rank == 0:
            print("ODE model: Finished lambdify for stiffness expressions, %.4f s" % (te))
            sys.stdout.flush()


    # set prescribed variable values for residual
    def set_prescribed_variables_residual(self, x, r, val, index_prescribed):

        if isinstance(x, np.ndarray): xs, xe = 0, len(x)
        else: xs, xe = x.getOwnershipRange()

        # modification of rhs entry
        if index_prescribed in range(xs,xe):
            r[index_prescribed] = x[index_prescribed] - val

        r.assemble()


    # set stiffness entries for prescribed variable values
    def set_prescribed_variables_stiffness(self, K, index_prescribed):

        # modification of stiffness matrix - all off-columns associated to index_prescribed = 0
        # diagonal entry associated to index_prescribed = 1
        K[index_prescribed,index_prescribed] = 1.
        for j in range(self.numdof):
            if j!=index_prescribed: K[index_prescribed,j] = 0.

        K.assemble()


    # time step update
    def update(self, var, df, f, var_old, df_old, f_old, aux, aux_old):

        if isinstance(var, np.ndarray): vs, ve = 0, len(var)
        else: vs, ve = var.getOwnershipRange()

        var_old[vs:ve] = var[vs:ve]
        df_old[vs:ve]  = df[vs:ve]
        f_old[vs:ve]   = f[vs:ve]

        # aux vector is always a numpy array
        aux_old[:] = aux[:]


    # midpoint-averaging of state variables (for post-processing)
    def set_output_state(self, var, var_old, var_out, theta, midpoint=True):

        if isinstance(var, np.ndarray): vs, ve = 0, len(var)
        else: vs, ve = var.getOwnershipRange()

        if midpoint:
            var_out[vs:ve] = theta*var[vs:ve] + (1.-theta)*var_old[vs:ve]
        else:
            var_out[vs:ve] = var[vs:ve]


    # set up the dof, coupling quantity, rhs, and stiffness arrays
    def set_solve_arrays(self):

        self.x_, self.a_, self.a__ = [0]*self.numdof, [0]*self.numdof, [0]*self.numdof
        self.c_, self.fnc_ = [], []

        self.df_, self.f_, self.df__, self.f__ = [0]*self.numdof, [0]*self.numdof, [0]*self.numdof, [0]*self.numdof
        self.dK_,  self.K_  = [[0]*self.numdof for _ in range(self.numdof)], [[0]*self.numdof for _ in range(self.numdof)]
        self.dK__, self.K__ = [[0]*self.numdof for _ in range(self.numdof)], [[0]*self.numdof for _ in range(self.numdof)]


    # output routine for ODE models
    def write_output(self, path, t, var, aux, nm=''):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        # mode: 'wt' generates new file, 'a' appends to existing one
        if self.init: mode = 'wt'
        else: mode = 'a'

        self.init = False

        if self.comm.rank == 0:

            for i in range(len(self.varmap)):

                filename = path+'/results_'+nm+'_'+list(self.varmap.keys())[i]+'.txt'
                f = open(filename, mode)

                f.write('%.16E %.16E\n' % (t,var_sq[list(self.varmap.values())[i]]))

                f.close()

            for i in range(len(self.auxmap)):

                filename = path+'/results_'+nm+'_'+list(self.auxmap.keys())[i]+'.txt'
                f = open(filename, mode)

                f.write('%.16E %.16E\n' % (t,aux[list(self.auxmap.values())[i]]))

                f.close()

        if not isinstance(var, np.ndarray): del var_sq


    # write restart routine for ODE models
    def write_restart(self, path, nm, N, var):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        if self.comm.rank == 0:

            filename = path+'/checkpoint_'+nm+'_'+str(N)+'.txt'
            f = open(filename, 'wt')

            for i in range(len(var_sq)):

                f.write('%.16E\n' % (var_sq[i]))

            f.close()

        if not isinstance(var, np.ndarray): del var_sq


    # read restart routine for ODE models
    def read_restart(self, path, nm, rstep, var):

        restart_data = np.loadtxt(path+'/checkpoint_'+nm+'_'+str(rstep)+'.txt', ndmin=1)

        if not isinstance(var, np.ndarray): var.assemble()

        var[:] = restart_data[:]


    # to write initial conditions (i.e. after a model has reached periodicity, so we may want to export these if we want to use
    # them in a new simulation starting from a homeostatic state)
    def write_initial(self, path, nm, varTc_old, varTc, auxTc_old, auxTc):

        if isinstance(varTc_old, np.ndarray): varTc_old_sq, varTc_sq = varTc_old, varTc
        else: varTc_old_sq, varTc_sq = allgather_vec(varTc_old, self.comm), allgather_vec(varTc, self.comm)

        if self.comm.rank == 0:

            filename1 = path+'/results_'+nm+'_initial_data_Tstart.txt' # conditions at beginning of cycle
            f1 = open(filename1, 'wt')
            filename2 = path+'/results_'+nm+'_initial_data_Tend.txt' # conditions at end of cycle
            f2 = open(filename2, 'wt')

            for i in range(len(self.varmap)):

                f1.write('%s %.16E\n' % (list(self.varmap.keys())[i]+'_0',varTc_old_sq[list(self.varmap.values())[i]]))
                f2.write('%s %.16E\n' % (list(self.varmap.keys())[i]+'_0',varTc_sq[list(self.varmap.values())[i]]))

            for i in range(len(self.auxmap)):

                f1.write('%s %.16E\n' % (list(self.auxmap.keys())[i]+'_0',auxTc_old[list(self.auxmap.values())[i]]))
                f2.write('%s %.16E\n' % (list(self.auxmap.keys())[i]+'_0',auxTc[list(self.auxmap.values())[i]]))

            f1.close()
            f2.close()

        if not isinstance(varTc_old, np.ndarray): del varTc_old_sq, varTc_sq


    # if we want to set the initial conditions from a txt file
    def set_initial_from_file(self, initialdata):

        pini0D = {}
        with open(initialdata) as fh:
            for line in fh:
                (key, val) = line.split()
                pini0D[key] = float(val)

        return pini0D
