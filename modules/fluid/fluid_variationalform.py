#!/usr/bin/env python3

import numpy as np
from ufl import dot, inner, inv, grad, div, det, as_tensor, indices, derivative, diff, dev

# fluid mechanics variational forms class
# Principle of Virtual Power
# TeX: \delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}
class variationalform:
    
    def __init__(self, var_v, dv, var_p, dp, n=None):
        self.var_v = var_v
        self.var_p = var_p
        self.dv = dv
        self.dp = dp
        
        self.n = n
    
    ### Kinetic virtual power
    
    # TeX: \delta \mathcal{P}_{\mathrm{kin}} := \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\otimes\boldsymbol{v})^{\mathrm{T}}\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v
    def deltaP_kin(self, a, v, rho, ddomain, v_old=None):
        
        if v_old is None:
            return rho*dot(a + grad(v) * v, self.var_v)*ddomain
        else:
            return rho*dot(a + grad(v) * v_old, self.var_v)*ddomain

    ### Internal virtual power

    # TeX: \delta \mathcal{P}_{\mathrm{int}} := \int\limits_{\Omega} \boldsymbol{\sigma} : \delta\boldsymbol{\gamma} \,\mathrm{d}v
    def deltaP_int(self, sig, ddomain):
        
        # TeX: \int\limits_{\Omega}\boldsymbol{\sigma} : \delta \boldsymbol{\gamma}\,\mathrm{d}v
        var_gamma = 0.5*(grad(self.var_v).T + grad(self.var_v))
        return inner(sig, var_gamma)*ddomain

    def deltaP_int_pres(self, v, ddomain):
        # TeX: \int\limits_{\Omega}\mathrm{div}\boldsymbol{v}\,\delta p\,\mathrm{d}v
        return div(v)*self.var_p*ddomain

    def residual_v_strong(self, a, v, rho, sig):
        
        return rho*(a + grad(v) * v) - div(sig)
    
    def residual_p_strong(self, v):
        
        return div(v)
    
    def f_inert(self, a, v, rho):
        
        return rho*(a + grad(v) * v)
    
    def f_viscous(self, sig):
        
        return div(dev(sig))

    ### External virtual power
    
    # Neumann load (Cauchy traction)
    # TeX: \int\limits_{\Gamma} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}a
    def deltaP_ext_neumann(self, func, dboundary):

        return dot(func, self.var_v)*dboundary
    
    # Neumann load in normal direction (Cauchy traction)
    # TeX: \int\limits_{\Gamma} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def deltaP_ext_neumann_normal(self, func, dboundary):

        return func*dot(self.n, self.var_v)*dboundary
    
    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma} c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def deltaP_ext_robin_dashpot(self, v, c, dboundary):

        return -c*(dot(v, self.var_v)*dboundary)
    
    # Robin condition (dashpot) in normal direction
    # TeX: \int\limits_{\Gamma} (\boldsymbol{n}\otimes \boldsymbol{n})\,c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def deltaP_ext_robin_dashpot_normal(self, v, c_n, dboundary):

        return -c_n*(dot(v, self.n)*dot(self.n, self.var_v)*dboundary)



    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{v}\;\mathrm{d}a
    def flux(self, v, dboundary):
        
        return dot(self.n, v)*dboundary
        
    # surface - derivative of pressure load w.r.t. pressure
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def surface(self, dboundary):
        
        return dot(self.n, self.var_v)*dboundary
