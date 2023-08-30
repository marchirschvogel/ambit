#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base


# ALE variational forms class
# Principle of Virtual Work
# TeX: \delta \mathcal{W} = \delta \mathcal{W}_{\mathrm{int}} - \delta \mathcal{W}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{u}
class variationalform(variationalform_base):

    def __init__(self, var_d, n0=None):
        self.var_d = var_d

        self.n0 = n0


    ### Internal virtual work

    # TeX: \delta \mathcal{W}_{\mathrm{int}} := \int\limits_{\Omega_0} \boldsymbol{\sigma} : \delta\boldsymbol{\epsilon} \,\mathrm{d}V
    def deltaW_int(self, stress, ddomain):

        return ufl.inner(stress,ufl.grad(self.var_d)) * ddomain

    # Robin term for weak imposition of Dirichlet condition
    # TeX: \int\limits_{\Gamma_0} J\beta\,(\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\cdot\delta\boldsymbol{u}\sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A

    # Nitsche term for weak imposition of Dirichlet condition
    # TeX: \int\limits_{\Gamma_0} \beta\,(\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\cdot\delta\boldsymbol{u}\,\mathrm{d}A - \int\limits_{\Gamma_0} \boldsymbol{\sigma}(\delta\boldsymbol{u})\boldsymbol{n}_{0}\cdot (\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\,\mathrm{d}A
    def deltaW_int_nitsche_dirichlet(self, u, uD, var_stress, beta, dboundary):

        return ( beta*ufl.dot((u-uD), self.var_d) - ufl.dot(ufl.dot(var_stress,self.n0),(u-uD)) )*dboundary


    ### External virtual work

    # Neumann load
    # TeX: \int\limits_{\Gamma_0} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{d} \,\mathrm{d}A
    def deltaW_ext_neumann_ref(self, func, dboundary):

        return ufl.dot(func, self.var_d)*dboundary


    # body force external virtual work
    # TeX: \int\limits_{\Omega_{0}} \hat{\boldsymbol{b}}\cdot\delta\boldsymbol{d}\,\mathrm{d}V
    def deltaW_ext_bodyforce(self, func, funcdir, ddomain):

        return func*ufl.dot(funcdir, self.var_d)*ddomain


    # Robin condition
    # TeX: \int\limits_{\Gamma_0} k\,\boldsymbol{d}\cdot\delta\boldsymbol{d}\,\mathrm{d}A
    def deltaW_ext_robin_spring(self, u, k, dboundary, upre=None):

        return -k*(ufl.dot(u, self.var_d)*dboundary)

    # Robin condition in normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}\otimes \boldsymbol{n})\,k\,\boldsymbol{d}\cdot\delta\boldsymbol{d}\,\mathrm{d}A
    def deltaW_ext_robin_spring_normal_ref(self, u, k_n, dboundary, upre=None):

        return -k_n*(ufl.dot(ufl.outer(self.n0,self.n0)*u, self.var_d)*dboundary)
