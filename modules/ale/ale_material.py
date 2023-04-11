#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

class materiallaw:

    def __init__(self, w, F):
        self.w = w
        self.F = F

        self.dim = len(self.w)
        self.I = ufl.Identity(self.dim)


    def neohooke(self, params):

        # shear modulus and Poisson's ratio
        mu, nu = params['mu'], params['nu']

        beta = nu/(1.-2.*nu)

        # first invariant and determinant of right Cauchy-Green tensor
        Ic = ufl.tr(self.F.T*self.F)
        J = ufl.det(self.F)

        # compressible NeoHookean material (Holzapfel eq. 6.148)
        Psi = (mu/2.) * (Ic - 3.) + mu/(2.*beta)*(J**(-2.*beta) - 1.)

        # s_grad (tensor - PK1), s_div (scalar), s_ident (vector)
        return ufl.diff(Psi,self.F), 0, ufl.constantvalue.zero(self.dim)


    def helmholtz(self, params):

        k = params['k']

        # s_grad (tensor), s_div (scalar), s_ident (vector)
        return (k**2.)*ufl.grad(self.w), 0, self.w


    def linelast(self, params):

        Emod = params['Emod']
        kappa = params['kappa']

        # s_grad (tensor), s_div (scalar), s_ident (vector)
        #return Emod*ufl.sym(ufl.grad(self.w)), kappa*ufl.div(self.w), ufl.constantvalue.zero(self.dim)
        return Emod*ufl.sym(ufl.grad(self.w)) + kappa*ufl.div(self.w)*self.I, 0, ufl.constantvalue.zero(self.dim)


    def element_dependent_stiffness(self, params, metric):

        alpha = params['alpha']
        kappa = metric

        # s_grad (tensor), s_div (scalar), s_ident (vector)
        return kappa*ufl.sym(ufl.grad(self.w)), kappa*alpha*ufl.div(self.w), ufl.constantvalue.zero(self.dim)
