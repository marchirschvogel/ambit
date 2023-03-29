#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

# ALE variational forms class
# Principle of Virtual Work
# TeX: \delta \mathcal{W} = \delta \mathcal{W}_{\mathrm{int}} - \delta \mathcal{W}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{u}
class variationalform:
    
    def __init__(self, var_u, n=None):
        self.var_u = var_u
        
        self.n = n
    

    ### Internal virtual work

    # TeX: \delta \mathcal{W}_{\mathrm{int}} := \int\limits_{\Omega_0} \boldsymbol{\sigma} : \delta\boldsymbol{\epsilon} \,\mathrm{d}V
    def deltaW_int(self, stress, ddomain):
        
        s_grad, s_div, s_ident = stress
        
        # TeX: \int\limits_{\Omega_0} \kappa\,[\mathrm{sym}(\nabla_{0}\boldsymbol{u}) : \nabla_{0}\delta\boldsymbol{u} + \alpha (\nabla\cdot\boldsymbol{u}) \,(\nabla_0\cdot\delta\boldsymbol{u})] \,\mathrm{d}V
        return ( ufl.inner(s_grad,ufl.grad(self.var_u)) + ufl.dot(s_ident,self.var_u) + s_div*ufl.div(self.var_u) ) * ddomain

    # Nitsche term for weak imposition of Dirichlet condition
    # TeX: \int\limits_{\Gamma_0} \beta\,(\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\cdot\delta\boldsymbol{u}\,\mathrm{d}A - \int\limits_{\Gamma_0} \boldsymbol{P}(\delta\boldsymbol{u})\boldsymbol{n}_{0}\cdot (\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\,\mathrm{d}A
    def deltaW_int_nitsche_dirichlet(self, u, uD, var_stress, beta, dboundary):

        # TODO: Check out why the latter term does not work for nonlinear problems (e.g. a NeoHookean solid)
        return ( beta*ufl.dot((u-uD), self.var_u) - ufl.dot(var_stress*self.n,(u-uD)) )*dboundary


    ### External virtual work
    
    # Neumann load
    # TeX: \int\limits_{\Gamma_0} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{u} \,\mathrm{d}A
    def deltaW_ext_neumann_ref(self, func, dboundary):

        return ufl.dot(func, self.var_u)*dboundary

    
    # Robin condition
    # TeX: \int\limits_{\Gamma_0} k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_robin_spring(self, u, k, dboundary, upre=None):

        return -k*(ufl.dot(u, self.var_u)*dboundary)
    
    # Robin condition in normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}\otimes \boldsymbol{n})\,k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_robin_spring_normal(self, u, k_n, dboundary, upre=None):

        return -k_n*(ufl.dot(u, self.n)*ufl.dot(self.n, self.var_u)*dboundary)

