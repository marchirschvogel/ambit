#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

class variationalform():
    def __init__(self, var_phi, var_mu):
        self.var_phi = var_phi  # phase field test functions
        self.var_mu = var_mu  # potential test functions

    def cahnhilliard_phase(self, phidot, phi, mu, Jflux, ddomain, v=None, w=None, F=None):
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v, ufl.grad(phi))"
            advec = ufl.div(phi*v)
        else:
            advec = ufl.as_ufl(0)
        """ TeX:
        \int\limits_{\Omega} \left(\frac{\partial \phi}{\partial t} + \nabla\cdot(\phi\boldsymbol{v})\right) \delta \phi \, \mathrm{d}v - \int\limits_{\Omega} \boldsymbol{J} \cdot \nabla \delta \phi \, \mathrm{d}v = 0
        """
        return ( ufl.inner(phidot, self.var_phi) + ufl.inner(advec, self.var_phi) - ufl.inner(Jflux, ufl.grad(self.var_phi)) ) * ddomain

    def cahnhilliard_potential(self, phi, mu, driv_force, lmbda, ddomain, F=None):
        """ TeX:
        \int\limits_{\Omega} \mu \,\delta\mu \, \mathrm{d}v - \int\limits_{\Omega} \frac{\mathrm{d}f}{\mathrm{d}\phi} \delta\mu \,\mathrm{d}v - \int\limits_{\Omega} \lambda \nabla \phi \cdot \nabla \delta\mu \, \mathrm{d}v = 0
        """
        return ( ufl.inner(mu, self.var_mu) - ufl.inner(driv_force, self.var_mu) - lmbda*ufl.inner(ufl.grad(phi), ufl.grad(self.var_mu)) ) * ddomain


# gradients of a scalar field transform according to:
# grad(phi) = F^(-T) * Grad(phi)

class variationalform_ale(variationalform):
    def cahnhilliard_phase(self, phidot, phi, mu, Jflux, ddomain, v=None, w=None, F=None):
        J = ufl.det(F)
        Jdot = ufl.div(J*ufl.inv(F)*w)
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v-w, ufl.inv(F).T*ufl.grad(phi))"
            advec = ufl.div(J*ufl.inv(F)*phi*(v-w))
        else:
            advec = ufl.as_ufl(0)
        """ TeX:
        \int\limits_{\Omega_0} \left(\left.\frac{\partial (\widehat{J}\phi)}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot(\widehat{J}\boldsymbol{F}^{-1}\phi\,(\boldsymbol{v}-\widehat{\boldsymbol{w}}))\right) \delta \phi \, \mathrm{d}V - \int\limits_{\Omega_0} \boldsymbol{J} \cdot \boldsymbol{F}^{-\mathrm{T}}\nabla_0 \delta \phi \, \mathrm{d}V = 0
        """
        return ( ufl.inner(J*phidot + phi*Jdot, self.var_phi) + ufl.inner(advec, self.var_phi) - J*ufl.inner(Jflux, ufl.inv(F).T*ufl.grad(self.var_phi)) ) * ddomain

    def cahnhilliard_potential(self, phi, mu, driv_force, lmbda, ddomain, F=None):
        J = ufl.det(F)
        """ TeX:
        \int\limits_{\Omega_0} \widehat{J}\mu \,\delta\mu \, \mathrm{d}V - \int\limits_{\Omega_0} \widehat{J}\frac{\mathrm{d}f}{\mathrm{d}\phi} \delta\mu \,\mathrm{d}V - \int\limits_{\Omega_0} \widehat{J}\lambda \boldsymbol{F}^{-\mathrm{T}}\nabla_0 \phi \cdot \boldsymbol{F}^{-\mathrm{T}}\nabla_0 \delta\mu \, \mathrm{d}V = 0
        """
        return ( J*ufl.inner(mu, self.var_mu) - J*ufl.inner(driv_force, self.var_mu) - J*lmbda*ufl.inner(ufl.inv(F).T*ufl.grad(phi), ufl.inv(F).T*ufl.grad(self.var_mu)) ) * ddomain
