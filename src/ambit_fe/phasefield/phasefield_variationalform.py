#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

"""
Cahn-Hilliard variational forms class
\int_{\Omega} \frac{\partial \phi}{\partial t} \delta \phi \, \mathrm{d}x +
    \int_{\Omega} M \nabla\mu \cdot \nabla \delta \phi \, \mathrm{d}x
    &= 0 \quad \forall \ \delta \phi \in V,  \\
\int_{\Omega} \mu \,\delta\mu \, \mathrm{d}x - \int_{\Omega} \frac{\mathrm{d}f}{\mathrm{d}\phi} \delta\mu \,
  \mathrm{d}x - \int_{\Omega} \lambda \nabla \phi \cdot \nabla \delta\mu \, \mathrm{d}x
   &= 0 \quad \forall \ \delta\mu \in V.
"""

class variationalform():
    def __init__(self, var_phi, var_mu):
        self.var_phi = var_phi  # phase field test functions
        self.var_mu = var_mu  # potential test functions

    def cahnhilliard_phase(self, phidot, phi, mu, M, ddomain, v=None, w=None, F=None):
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v, ufl.grad(phi))"
            advec = ufl.div(phi*v)
        else:
            advec = ufl.as_ufl(0)
        return ( ufl.inner(phidot, self.var_phi) + ufl.inner(advec, self.var_phi) + M*ufl.inner(ufl.grad(mu), ufl.grad(self.var_phi)) ) * ddomain

    def cahnhilliard_potential(self, phi, mu, driv_force, lmbda, ddomain, F=None):
        return ( ufl.inner(mu, self.var_mu) - ufl.inner(driv_force, self.var_mu) - lmbda*ufl.inner(ufl.grad(phi), ufl.grad(self.var_mu)) ) * ddomain


# gradients of a scalar field transform according to:
# grad(phi) = F^(-T) * Grad(phi)

class variationalform_ale(variationalform):
    def cahnhilliard_phase(self, phidot, phi, mu, M, ddomain, v=None, w=None, F=None):
        J = ufl.det(F)
        Jdot = ufl.div(J*ufl.inv(F)*w)
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v-w, ufl.inv(F).T*ufl.grad(phi))"
            advec = ufl.div(J*ufl.inv(F)*phi*(v-w))
        else:
            advec = ufl.as_ufl(0)
        return ( ufl.inner(J*phidot + phi*Jdot, self.var_phi) + ufl.inner(advec, self.var_phi) + J*M*ufl.inner(ufl.inv(F).T*ufl.grad(mu), ufl.inv(F).T*ufl.grad(self.var_phi)) ) * ddomain

    def cahnhilliard_potential(self, phi, mu, driv_force, lmbda, ddomain, F=None):
        J = ufl.det(F)
        return ( J*ufl.inner(mu, self.var_mu) - J*ufl.inner(driv_force, self.var_mu) - J*lmbda*ufl.inner(ufl.inv(F).T*ufl.grad(phi), ufl.inv(F).T*ufl.grad(self.var_mu)) ) * ddomain
