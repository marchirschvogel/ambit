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


    ### External virtual work
    
    # Neumann load
    # TeX: \int\limits_{\Gamma_0} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{u} \,\mathrm{d}A
    def deltaW_ext_neumann(self, func, dboundary):

        return ufl.dot(func, self.var_u)*dboundary

    
    # Robin condition
    # TeX: \int\limits_{\Gamma_0} k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_spring(self, u, k, dboundary):

        return -k*(ufl.dot(u, self.var_u)*dboundary)
    
    # Robin condition in normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}\otimes \boldsymbol{n})\,k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_spring_normal(self, u, k_n, dboundary):

        return -k_n*(ufl.dot(u, self.n)*ufl.dot(self.n, self.var_u)*dboundary)
