#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

class materiallaw:

    def __init__(self, d, F, elem_metrics):
        self.d = d
        self.F = F
        self.jac_det = elem_metrics['jac_det']

        self.dim = len(self.d)
        self.I = ufl.Identity(self.dim)


    def diffusion(self, params):

        D = params['D']
        try: scale_det = params['scale_det']
        except: scale_det = False

        if scale_det: fac = 1./self.jac_det
        else:         fac = 1.

        return fac * D*ufl.grad(self.d)


    def diffusion_sym(self, params):

        D = params['D']
        try: scale_det = params['scale_det']
        except: scale_det = False

        if scale_det: fac = 1./self.jac_det
        else:         fac = 1.

        return fac * D*ufl.sym(ufl.grad(self.d))


    def linelast(self, params):

        Emod = params['Emod']
        nu = params['nu']
        try: scale_det = params['scale_det']
        except: scale_det = False

        if scale_det: fac = 1./self.jac_det
        else:         fac = 1.

        mu = Emod/(2.*(1.+nu))
        lam = nu*Emod/((1.+nu)*(1.-2.*nu))

        epsilon = ufl.sym(ufl.grad(self.d))

        # stress
        return fac * (2.*mu*epsilon + lam*ufl.tr(epsilon)*self.I)


    def neohooke(self, params):

        # shear modulus and Poisson's ratio
        mu, nu = params['mu'], params['nu']
        try: scale_det = params['scale_det']
        except: scale_det = False

        if scale_det: fac = 1./self.jac_det
        else:         fac = 1.

        beta = nu/(1.-2.*nu)

        # first invariant and determinant of right Cauchy-Green tensor
        Ic = ufl.tr(self.F.T*self.F)
        J = ufl.det(self.F)

        # compressible NeoHookean material (Holzapfel eq. 6.148)
        Psi = (mu/2.) * (Ic - self.dim) + mu/(2.*beta)*(J**(-2.*beta) - 1.)

        # stress
        return fac * ufl.diff(Psi,self.F)


    def exponential(self, params):

        a_0, b_0, kappa = params['a_0'], params['b_0'], params['kappa']
        try: scale_det = params['scale_det']
        except: scale_det = False

        if scale_det: fac = 1./self.jac_det
        else:         fac = 1.

        J = ufl.det(self.F)
        Ic_bar = J**(-2./self.dim) * ufl.tr(self.F.T*self.F)

        # exponential law: can be soft in the small deformation realm and stiffer for larger deformations
        Psi = a_0/(2.*b_0)*(ufl.exp(b_0*(Ic_bar-self.dim)) - 1.) + (kappa/2.) * (J - 1.)**2.

        # stress
        return fac * ufl.diff(Psi,self.F)
