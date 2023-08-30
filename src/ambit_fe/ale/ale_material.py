#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

class materiallaw:

    def __init__(self, d, F):
        self.d = d
        self.F = F

        self.dim = len(self.d)
        self.I = ufl.Identity(self.dim)


    def linelast(self, params):

        Emod = params['Emod']
        kappa = params['kappa']

        # stress
        return Emod*ufl.sym(ufl.grad(self.d)) + kappa*ufl.div(self.d)*self.I


    def neohooke(self, params):

        # shear modulus and Poisson's ratio
        mu, nu = params['mu'], params['nu']

        beta = nu/(1.-2.*nu)

        # first invariant and determinant of right Cauchy-Green tensor
        Ic = ufl.tr(self.F.T*self.F)
        J = ufl.det(self.F)

        # compressible NeoHookean material (Holzapfel eq. 6.148)
        Psi = (mu/2.) * (Ic - 3.) + mu/(2.*beta)*(J**(-2.*beta) - 1.)

        # stress
        return ufl.diff(Psi,self.F)
