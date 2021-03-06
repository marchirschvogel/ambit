#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ufl import dot, inner, inv, grad, div, det, as_tensor, indices, derivative, diff, sym, constantvalue, as_ufl

# solid mechanics variational base class
# Principle of Virtual Work
# TeX: \delta \mathcal{W} = \delta \mathcal{W}_{\mathrm{kin}} + \delta \mathcal{W}_{\mathrm{int}} - \delta \mathcal{W}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{u}
class variationalform:
    
    def __init__(self, var_u, du, var_p=None, dp=None, n0=None, x_ref=None):
        self.var_u = var_u
        self.var_p = var_p
        self.du = du
        self.dp = dp
        
        self.n0 = n0
        self.x_ref = x_ref
    
    ### Kinetic virtual work
    
    # TeX: \delta \mathcal{W}_{\mathrm{kin}} := \int\limits_{\Omega_{0}} \rho_{0}\boldsymbol{a} \cdot \delta\boldsymbol{u} \,\mathrm{d}V
    def deltaW_kin(self, a, rho0, ddomain):
        
        return rho0*dot(a, self.var_u)*ddomain

    # TeX: \delta \mathcal{W}_{\mathrm{kin,mass}} := \int\limits_{\Omega_{0}} \dot{\rho}_{0}\boldsymbol{v} \cdot \delta\boldsymbol{u} \,\mathrm{d}V
    def deltaW_kin_masschange(self, v, drho0, ddomain):
        
        return drho0*dot(v, self.var_u)*ddomain

    ### Internal virtual work

    # TeX: \delta \mathcal{W}_{\mathrm{int}} := \int\limits_{\Omega_{0}} \boldsymbol{S}(\boldsymbol{E}(\boldsymbol{u})) : \delta\boldsymbol{E} \,\mathrm{d}V = \int\limits_{\Omega_{0}} \boldsymbol{S}(\boldsymbol{C}(\boldsymbol{u})) : \frac{1}{2}\delta\boldsymbol{C} \,\mathrm{d}V
    def deltaW_int(self, S, F, ddomain):
        
        # TeX: \int\limits_{\Omega_0}\boldsymbol{S} : \frac{1}{2}\delta \boldsymbol{C}\mathrm{d}V
        var_C = grad(self.var_u).T * F + F.T * grad(self.var_u)
        return inner(S, 0.5*var_C)*ddomain

    def deltaW_int_pres(self, J, ddomain):
        # TeX: \int\limits_{\Omega_0}\left(J(\boldsymbol{u})-1\right)\delta p\,\mathrm{d}V
        return (J-1.)*self.var_p*ddomain

    # Rayleigh damping linear form: in the linear elastic case, we would have eta_m * mass_form + eta_k * stiffness_form,
    # the stiffness damping form would read 
    # \int\limits_{\Omega_{0}} \left[\mathbb{C} : \mathrm{sym}(\mathrm{Grad}\boldsymbol{v})\right] : \delta\boldsymbol{\varepsilon} \,\mathrm{d}V
    # rendering K * v in the assembled FE system
    # now, in the nonlinear case, the stiffness is deformation-dependent, and has a geometric as well as a material contribution
    # at first, we should only include material, not geometric stiffness to the stiffness part (see Charney 2008, "Unintended Consequences of Modeling Damping in Structures")
    # secondly, here, we only use the stiffness evaluated in the initial configuration
    # so, in the nonlinear realm, we would have the form
    # TeX: \int\limits_{\Omega_{0}} \left[\mathbb{C}(\boldsymbol{C}(\boldsymbol{u}_{0})) : \mathrm{sym}(\mathrm{Grad}\boldsymbol{v})\right] : \mathrm{sym}(\mathrm{Grad}\delta\boldsymbol{u}) \,\mathrm{d}V
    # hence, we get a K_mat(u_0) * v contribution in the assembled FE system
    def deltaW_damp(self, eta_m, eta_k, rho0, Cmat, v, ddomain):
        
        i, j, k, l, m, n = indices(6)

        Cmat_gradv = as_tensor(Cmat[i,j,k,l]*sym(grad(v))[k,l], (i,j))

        return (eta_m * rho0*dot(v, self.var_u) + eta_k * inner(Cmat_gradv,sym(grad(self.var_u)))) * ddomain


    # linearization of internal virtual work
    # we could use ufl to compute the derivative directly, via "derivative(self.deltaW_int(S,F,ddomain), u, self.du)",
    # however then, no material tangents from nonlinear consitutive laws at the integration point level can be introduced
    # so we use a more explicit expression where Ctang can be included
    # TeX: D_{\Delta \boldsymbol{u}}\delta \mathcal{W}_{\mathrm{int}} = \int\limits_{\Omega_{0}} \left(\mathrm{Grad}\delta\boldsymbol{u}:\mathrm{Grad}\Delta\boldsymbol{u}\,\boldsymbol{S} + \boldsymbol{F}^{\mathrm{T}}\mathrm{Grad}\delta\boldsymbol{u} : \mathbb{C} : \boldsymbol{F}^{\mathrm{T}}\mathrm{Grad}\Delta\boldsymbol{u}\right)\mathrm{d}V
    # (Holzapfel 2000, formula 8.81)
    # or:  D_{\Delta \boldsymbol{u}}\delta \mathcal{W}_{\mathrm{int}} = 
    #    = D_{\Delta \boldsymbol{u}}\int\limits_{\Omega_{0}} \boldsymbol{S}:\frac{1}{2}\delta\boldsymbol{C}\,\mathrm{d}V = 
    #    = \frac{1}{2}\int\limits_{\Omega_{0}} \left(\left[\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{C}} : D_{\Delta \boldsymbol{u}} \boldsymbol{C}\right] : \delta\boldsymbol{C} + \boldsymbol{S}:D_{\Delta \boldsymbol{u}}\delta\boldsymbol{C}\right)\mathrm{d}V = 
    #    = \frac{1}{2}\int\limits_{\Omega_{0}} \left(\left[\frac{1}{2}\mathbb{C} : D_{\Delta \boldsymbol{u}} \boldsymbol{C}\right] : \delta\boldsymbol{C} + \boldsymbol{S}:D_{\Delta \boldsymbol{u}}\delta\boldsymbol{C}\right)\mathrm{d}V
    def Lin_deltaW_int_du(self, S, F, u, Ctang, ddomain):
        
        C = F.T*F
        var_C = grad(self.var_u).T * F + F.T * grad(self.var_u)

        i, j, k, l, m, n = indices(6)
        Ctang_DuC = as_tensor(Ctang[i,j,k,l]*derivative(C, u, self.du)[k,l], (i,j))
        return (inner(0.5*Ctang_DuC,0.5*var_C) + inner(S,derivative(0.5*var_C, u, self.du)))*ddomain

    ## 1:1 from Holzapfel 2000, formula 8.81 - not working! The material stiffness does not work, no idea why... so see Lin_deltaW_int_du above for a correct form
    #def Lin_deltaW_int_du(self, S, F, u, Ctang, ddomain):

        #FT_graddu = F.T * grad(self.du)
        #FT_gradvaru = F.T * grad(self.var_u)
        
        #i, j, k, l = indices(4)
        #FT_graddu_Ctang = as_tensor(Ctang[i,j,k,l]*FT_graddu[k,l], (i,j)) 
        #stiff_material = inner(FT_gradvaru,FT_graddu_Ctang) # seems to be the problematic one
        #stiff_geometric = inner(grad(self.var_u), grad(self.du)*S) # seems ok
        
        #return (stiff_geometric + stiff_material)*ddomain


    # TeX: \int\limits_{\Omega_0} J(\boldsymbol{u})\Delta p \,\mathrm{div}\delta\boldsymbol{u}\,\mathrm{d}V = 
    #      \int\limits_{\Omega_0} \frac{\partial\boldsymbol{S}}{\partial p} \Delta p : \frac{1}{2}\delta \boldsymbol{C}\,\mathrm{d}V
    def Lin_deltaW_int_dp(self, F, Ctang_p, ddomain):

        var_C = grad(self.var_u).T * F + F.T * grad(self.var_u)
    
        return inner(Ctang_p*self.dp, 0.5*var_C)*ddomain


    # TeX: \int\limits_{\Omega_0} J(\boldsymbol{u})\,\mathrm{div}\Delta\boldsymbol{u}\,\delta p\,\mathrm{d}V = 
    #      \int\limits_{\Omega_0} \left(\frac{\partial J}{\partial\boldsymbol{C}} : D_{\Delta \boldsymbol{u}} \boldsymbol{C}\right)\delta p\,\mathrm{d}V
    def Lin_deltaW_int_pres_du(self, F, Jtang, u, ddomain):
        
        C = F.T*F
        return inner(Jtang, derivative(C, u, self.du)) * self.var_p*ddomain

    ### External virtual work

    # Neumann follower load
    # TeX: \int\limits_{\Gamma_{0}} p\,J \boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_neumann_true(self, J, F, func, dboundary):

        return func*J*dot(dot(inv(F).T,self.n0), self.var_u)*dboundary
    
    # Neumann load on reference configuration (1st Piola-Kirchhoff traction)
    # TeX: \int\limits_{\Gamma_{0}} \hat{\boldsymbol{t}}_{0} \cdot \delta\boldsymbol{u} \,\mathrm{d}A
    def deltaW_ext_neumann_ref(self, func, dboundary):

        return dot(func, self.var_u)*dboundary
    
    # Neumann load in reference normal (1st Piola-Kirchhoff traction)
    # TeX: \int\limits_{\Gamma_{0}} p\,\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_neumann_refnormal(self, func, dboundary):

        return func*dot(self.n0, self.var_u)*dboundary
    
    # Robin condition (spring)
    # TeX: \int\limits_{\Gamma_0} k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_spring(self, u, k, dboundary, u_prestr=None):
        
        if u_prestr is not None:
            return -k*(dot(u + u_prestr, self.var_u)*dboundary)
        else:
            return -k*(dot(u, self.var_u)*dboundary)
    
    # Robin condition (spring) in reference normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}_{0}\otimes \boldsymbol{n}_{0})\,k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_spring_normal(self, u, k_n, dboundary, u_prestr=None):

        if u_prestr is not None:
            return -k_n*(dot(u + u_prestr, self.n0)*dot(self.n0, self.var_u)*dboundary)
        else:
            return -k_n*(dot(u, self.n0)*dot(self.n0, self.var_u)*dboundary)
    
    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma_0} c\,\dot{\boldsymbol{u}}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_dashpot(self, v, c, dboundary):
        
        if not isinstance(v, constantvalue.Zero):
            return -c*(dot(v, self.var_u)*dboundary)
        else:
            return as_ufl(0)
    
    # Robin condition (dashpot) in reference normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}_{0}\otimes \boldsymbol{n}_{0})\,c\,\dot{\boldsymbol{u}}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_dashpot_normal(self, v, c_n, dboundary):
        
        if not isinstance(v, constantvalue.Zero):
            return -c_n*(dot(v, self.n0)*dot(self.n0, self.var_u)*dboundary)
        else:
            return as_ufl(0)



    ### Volume / flux coupling conditions
            
    # volume
    def volume(self, u, J, F, dboundary):
        
        return -(1./3.)*J*dot(dot(inv(F).T,self.n0), self.x_ref + u)*dboundary
        
    # flux: Q = -dV/dt
    # TeX: \int\limits_{\Gamma_{0}} J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\boldsymbol{v}\;\mathrm{d}A
    def flux(self, v, J, F, dboundary):
        
        return J*dot(dot(inv(F).T,self.n0), v)*dboundary
        
    # surface - derivative of pressure load w.r.t. pressure
    # TeX: \int\limits_{\Gamma_{0}} J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def surface(self, J, F, dboundary):
        
        return J*dot(dot(inv(F).T,self.n0), self.var_u)*dboundary
