#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base

class variationalform(variationalform_base):
    def __init__(
        self,
        tstfnc1=None,
        tstfnc2=None,
        trlfnc1=None,
        trlfnc2=None,
        n0=None,
        x_ref=None,
        formulation=None,
        ro0=None,
    ):
        self.var_phi = tstfnc1
        self.var_mu = tstfnc2
        variationalform_base.__init__(self, tstfnc1=tstfnc1, tstfnc2=tstfnc2, trlfnc1=trlfnc1, trlfnc2=trlfnc2, n0=n0, x_ref=x_ref, formulation=formulation, ro0=ro0)

    def cahnhilliard_phase(self, phidot, phi, mu, Jflux, ddomain, v=None, w=None, F=None):
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v, ufl.grad(phi))"
            advec = ufl.div(phi*v)
        else:
            advec = ufl.as_ufl(0)
        """ TeX:
        \int\limits_{\mathit{\Omega}} \left(\frac{\partial \phi}{\partial t} + \nabla\cdot(\phi\boldsymbol{v})\right) \delta \phi \, \mathrm{d}V - \int\limits_{\mathit{\Omega}} \boldsymbol{J} \cdot \nabla \delta \phi \, \mathrm{d}V = 0
        """
        return ( ufl.inner(phidot, self.var_phi) + ufl.inner(advec, self.var_phi) - ufl.inner(Jflux, ufl.grad(self.var_phi)) ) * ddomain

    def cahnhilliard_potential(self, phi, mu, driv_force, kappa, ddomain, F=None):
        """ TeX:
        \int\limits_{\mathit{\Omega}} \mu \,\delta\mu \, \mathrm{d}V - \int\limits_{\mathit{\Omega}} \frac{\mathrm{d}\psi}{\mathrm{d}\phi} \delta\mu \,\mathrm{d}V - \int\limits_{\mathit{\Omega}} \kappa \nabla \phi \cdot \nabla \delta\mu \, \mathrm{d}V = 0
        """
        return ( ufl.inner(mu, self.var_mu) - ufl.inner(driv_force, self.var_mu) - kappa*ufl.inner(ufl.grad(phi), ufl.grad(self.var_mu)) ) * ddomain

    def weakform_neumann_wetting(self, coeff, dboundary, F=None):
        return coeff * self.var_mu * dboundary

    def weakform_robin_wetting(self, phi, coeff, dboundary, F=None):
        return ((3.0/4.0) * (phi**2.0 - 1.0) * coeff * self.var_mu) * dboundary


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
        \int\limits_{\mathit{\Omega}_0} \left(\left.\frac{\partial (\widehat{J}\phi)}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot(\widehat{J}\widehat{\boldsymbol{F}}^{-1}\phi\,(\boldsymbol{v}-\widehat{\boldsymbol{w}}))\right) \delta \phi \, \mathrm{d}V - \int\limits_{\mathit{\Omega}_0} \widehat{J}\widehat{\boldsymbol{F}}^{-1}\boldsymbol{J} \cdot \nabla_0 \delta \phi \, \mathrm{d}V = 0
        """
        return ( ufl.inner(J*phidot + phi*Jdot, self.var_phi) + ufl.inner(advec, self.var_phi) - J*ufl.inner(ufl.inv(F)*Jflux, ufl.grad(self.var_phi)) ) * ddomain

    def cahnhilliard_potential(self, phi, mu, driv_force, kappa, ddomain, F=None):
        J = ufl.det(F)
        """ TeX:
        \int\limits_{\mathit{\Omega}_0} \widehat{J}\mu \,\delta\mu \, \mathrm{d}V - \int\limits_{\mathit{\Omega}_0} \widehat{J}\frac{\mathrm{d}\psi}{\mathrm{d}\phi} \delta\mu \,\mathrm{d}V - \int\limits_{\mathit{\Omega}_0} \kappa\,\widehat{J} \widehat{\boldsymbol{F}}^{-1}\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_0 \phi \cdot \nabla_0 \delta\mu \, \mathrm{d}V = 0
        """
        return ( J*ufl.inner(mu, self.var_mu) - J*ufl.inner(driv_force, self.var_mu) - J*kappa*ufl.inner(ufl.inv(F)*ufl.inv(F).T*ufl.grad(phi), ufl.grad(self.var_mu)) ) * ddomain
