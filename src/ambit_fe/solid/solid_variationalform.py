#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base

"""
Solid mechanics variational form class
Principle of Virtual Work
\delta \mathcal{W} = \delta \mathcal{W}_{\mathrm{kin}} + \delta \mathcal{W}_{\mathrm{int}} - \delta \mathcal{W}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{u}
"""

class variationalform(variationalform_base):

    ### Kinetic virtual work

    # TeX: \delta \mathcal{W}_{\mathrm{kin}} := \int\limits_{\Omega_{0}} \rho_{0}\boldsymbol{a} \cdot \delta\boldsymbol{u} \,\mathrm{d}V
    def deltaW_kin(self, a, rho0, ddomain):

        return rho0*ufl.dot(a, self.var_u)*ddomain

    # TeX: \delta \mathcal{W}_{\mathrm{kin,mass}} := \int\limits_{\Omega_{0}} \dot{\rho}_{0}\boldsymbol{v} \cdot \delta\boldsymbol{u} \,\mathrm{d}V
    def deltaW_kin_masschange(self, v, drho0, ddomain):

        return drho0*ufl.dot(v, self.var_u)*ddomain

    ### Internal virtual work

    # TeX: \delta \mathcal{W}_{\mathrm{int}} := \int\limits_{\Omega_{0}} \boldsymbol{S}(\boldsymbol{E}(\boldsymbol{u})) : \delta\boldsymbol{E} \,\mathrm{d}V = \int\limits_{\Omega_{0}} \boldsymbol{S}(\boldsymbol{C}(\boldsymbol{u})) : \frac{1}{2}\delta\boldsymbol{C} \,\mathrm{d}V
    def deltaW_int(self, S, F, ddomain):

        # TeX: \int\limits_{\Omega_0}\boldsymbol{S} : \frac{1}{2}\delta \boldsymbol{C}\mathrm{d}V
        var_C = ufl.grad(self.var_u).T * F + F.T * ufl.grad(self.var_u)
        return ufl.inner(S, 0.5*var_C)*ddomain

    def deltaW_int_pres(self, J, ddomain):
        # TeX: \int\limits_{\Omega_0}\left(J(\boldsymbol{u})-1\right)\delta p\,\mathrm{d}V
        return (J-1.)*self.var_p*ddomain


    # linearization of internal virtual work
    # we could use ufl to compute the derivative directly via ufl.derivative(...), however then, no material tangents from nonlinear consitutive laws
    # at the integration point level can be introduced, so we use a more explicit expression where Ctang can be included
    # TeX: D_{\Delta \boldsymbol{u}}\delta \mathcal{W}_{\mathrm{int}} = \int\limits_{\Omega_{0}} \left(\mathrm{Grad}\delta\boldsymbol{u}:\mathrm{Grad}\Delta\boldsymbol{u}\,\boldsymbol{S} + \boldsymbol{F}^{\mathrm{T}}\mathrm{Grad}\delta\boldsymbol{u} : \mathbb{C} : \boldsymbol{F}^{\mathrm{T}}\mathrm{Grad}\Delta\boldsymbol{u}\right)\mathrm{d}V
    # (Holzapfel 2000, eq. (8.81)); or, including the viscous material tangent:
    #      D_{\Delta \boldsymbol{u}}\delta \mathcal{W}_{\mathrm{int}} =
    #    = D_{\Delta \boldsymbol{u}}\int\limits_{\Omega_{0}} \boldsymbol{S}(\boldsymbol{C},\dot{\boldsymbol{C}}):\frac{1}{2}\delta\boldsymbol{C}\,\mathrm{d}V =
    #    = \frac{1}{2}\int\limits_{\Omega_{0}} \left(\left[\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{C}} : D_{\Delta \boldsymbol{u}} \boldsymbol{C} + \frac{\partial\boldsymbol{S}}{\partial\dot{\boldsymbol{C}}} : D_{\Delta \boldsymbol{u}} \dot{\boldsymbol{C}}\right] : \delta\boldsymbol{C} + \boldsymbol{S}:D_{\Delta \boldsymbol{u}}\delta\boldsymbol{C}\right)\mathrm{d}V =
    #    = \frac{1}{2}\int\limits_{\Omega_{0}} \left(\left[\frac{1}{2}\mathbb{C} : D_{\Delta \boldsymbol{u}} \boldsymbol{C} + \frac{1}{2}\mathbb{C}_{\mathrm{v}} : D_{\Delta \boldsymbol{u}} \dot{\boldsymbol{C}}\right] : \delta\boldsymbol{C} + \boldsymbol{S}:D_{\Delta \boldsymbol{u}}\delta\boldsymbol{C}\right)\mathrm{d}V
    def Lin_deltaW_int_du(self, S, F, Fdot, u, Ctang, Cmat_v, ddomain):

        C, Cdot = F.T*F, Fdot.T*F + F.T*Fdot
        var_C = ufl.grad(self.var_u).T * F + F.T * ufl.grad(self.var_u)
        dim = len(u)

        i, j, k, l, m, n = ufl.indices(6)
        Ctang_DuC = ufl.as_tensor(Ctang[i,j,k,l]*ufl.derivative(C, u, self.du)[k,l], (i,j))

        if not isinstance(Cmat_v, ufl.constantvalue.Zero):
            Ctangv_DudC = ufl.as_tensor(Cmat_v[i,j,k,l]*ufl.derivative(Cdot, u, self.du)[k,l], (i,j))
        else:
            Ctangv_DudC = ufl.constantvalue.zero((dim,dim))

        return (ufl.inner(0.5*(Ctang_DuC+Ctangv_DudC),0.5*var_C) + ufl.inner(S,ufl.derivative(0.5*var_C, u, self.du)))*ddomain


    # TeX: \int\limits_{\Omega_0} J(\boldsymbol{u})\Delta p \,\mathrm{div}\delta\boldsymbol{u}\,\mathrm{d}V =
    #      \int\limits_{\Omega_0} \frac{\partial\boldsymbol{S}}{\partial p} \Delta p : \frac{1}{2}\delta \boldsymbol{C}\,\mathrm{d}V
    def Lin_deltaW_int_dp(self, F, Ctang_p, ddomain):

        var_C = ufl.grad(self.var_u).T * F + F.T * ufl.grad(self.var_u)

        return ufl.inner(Ctang_p*self.dp, 0.5*var_C)*ddomain


    # TeX: \int\limits_{\Omega_0} J(\boldsymbol{u})\,\mathrm{div}\Delta\boldsymbol{u}\,\delta p\,\mathrm{d}V =
    #      \int\limits_{\Omega_0} \left(\frac{\partial J}{\partial\boldsymbol{C}} : D_{\Delta \boldsymbol{u}} \boldsymbol{C}\right)\delta p\,\mathrm{d}V
    def Lin_deltaW_int_pres_du(self, F, Jtang, u, ddomain):

        C = F.T*F
        return ufl.inner(Jtang, ufl.derivative(C, u, self.du)) * self.var_p*ddomain


    ### Volume / flux coupling conditions

    # volume
    def volume(self, u, dboundary, F=None):
        J = ufl.det(F)
        return -(1./3.)*J*ufl.dot(ufl.inv(F).T*self.n0, self.x_ref + u)*dboundary

    # flux: Q = -dV/dt
    # TeX: \int\limits_{\Gamma_{0}} J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\boldsymbol{v}\,\mathrm{d}A
    def flux(self, v, dboundary, F=None):
        J = ufl.det(F)
        return J*ufl.dot(ufl.inv(F).T*self.n0, v)*dboundary
