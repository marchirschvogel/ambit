#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

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
    # (Holzapfel 2000, formula 8.81); or, including the viscous material tangent:
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

        if Cmat_v != ufl.constantvalue.zero((dim,dim)):
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

    ### External virtual work

    # Neumann follower load
    # TeX: \int\limits_{\Gamma_{0}} p\,J \boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_neumann_true(self, J, F, func, dboundary):

        return func*J*ufl.dot(ufl.dot(ufl.inv(F).T,self.n0), self.var_u)*dboundary
    
    # Neumann load on reference configuration (1st Piola-Kirchhoff traction)
    # TeX: \int\limits_{\Gamma_{0}} \hat{\boldsymbol{t}}_{0} \cdot \delta\boldsymbol{u} \,\mathrm{d}A
    def deltaW_ext_neumann_ref(self, func, dboundary):

        return ufl.dot(func, self.var_u)*dboundary
    
    # Neumann load in reference normal (1st Piola-Kirchhoff traction)
    # TeX: \int\limits_{\Gamma_{0}} p\,\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_neumann_refnormal(self, func, dboundary):

        return func*ufl.dot(self.n0, self.var_u)*dboundary
    
    # Robin condition (spring)
    # TeX: \int\limits_{\Gamma_0} k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_spring(self, u, k, dboundary, u_prestr=None):
        
        if u_prestr is not None:
            return -k*(ufl.dot(u + u_prestr, self.var_u)*dboundary)
        else:
            return -k*(ufl.dot(u, self.var_u)*dboundary)
    
    # Robin condition (spring) in reference normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}_{0}\otimes \boldsymbol{n}_{0})\,k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_spring_normal(self, u, k_n, dboundary, u_prestr=None):

        if u_prestr is not None:
            return -k_n*(ufl.dot(u + u_prestr, self.n0)*ufl.dot(self.n0, self.var_u)*dboundary)
        else:
            return -k_n*(ufl.dot(u, self.n0)*ufl.dot(self.n0, self.var_u)*dboundary)
    
    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma_0} c\,\dot{\boldsymbol{u}}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_dashpot(self, v, c, dboundary):
        
        if not isinstance(v, ufl.constantvalue.Zero):
            return -c*(ufl.dot(v, self.var_u)*dboundary)
        else:
            return ufl.as_ufl(0)
    
    # Robin condition (dashpot) in reference normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}_{0}\otimes \boldsymbol{n}_{0})\,c\,\dot{\boldsymbol{u}}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def deltaW_ext_robin_dashpot_normal(self, v, c_n, dboundary):
        
        if not isinstance(v, ufl.constantvalue.Zero):
            return -c_n*(ufl.dot(v, self.n0)*ufl.dot(self.n0, self.var_u)*dboundary)
        else:
            return ufl.as_ufl(0)

    # Elastic membrane potential on surface
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{S}(\tilde{\boldsymbol{C}}) : \frac{1}{2}\delta\tilde{\boldsymbol{C}}\;\mathrm{d}A
    def deltaW_ext_membrane(self, F, params, dboundary):
        
        C = F.T*F
        
        n0n0 = ufl.outer(self.n0,self.n0)
        
        I = ufl.Identity(3)
        
        model = params['model']
        
        # wall thickness
        h0 = params['h0']
        
        if model=='membrane_f':
            # deformation tensor with only normal components (C_nn, C_t1n = C_nt1, C_t2n = C_nt2)
            Cn = ufl.dot(C, n0n0) + ufl.dot(n0n0, C) - ufl.dot(self.n0,ufl.dot(C,self.n0)) * n0n0
            # plane strain deformation tensor where deformation is "1" in normal direction
            Cplane = C - Cn + n0n0
            # determinant: corresponds to product of in-plane stretches lambda_t1^2 * lambda_t2^2
            IIIplane = ufl.det(Cplane)
            # deformation tensor where normal stretch is dependent on in-plane stretches
            Cmod = C - Cn + (1./IIIplane) * n0n0
            # TODO: Need to recover an Fmod corresponding to Cmod!
            # Can this be done in ufl? See e.g. https://fenicsproject.org/qa/13600/possible-perform-spectral-decomposition-current-operators
            Fmod = F
        elif model=='membrane_transverse': # WARNING: NOT objective to large rotations!
            # only components in normal direction (F_nn, F_t1n, F_t2n)
            Fn = ufl.dot(F, n0n0)
            # plane deformation gradient: without components F_t1n, F_t2n, but with constant F_nn
            Fplane = F - Fn + n0n0
            # third invariant
            IIIplane = ufl.det(Fplane)**2.0
            # modified deformation gradient: without components F_t1n, F_t2n, and with F_nn dependent on F_t1n, F_t2n
            Fmod = F - Fn + (1./ufl.sqrt(IIIplane)) * n0n0
            # modified right Cauchy-Green tensor
            Cmod = Fmod.T*Fmod
        else:
            raise NameError("Unkown membrane model type!")
        
        # first invariant
        Ic = ufl.tr(Cmod)
        # declare variable for diff
        Ic_ = ufl.variable(Ic)
        
        a_0, b_0 = params['a_0'], params['b_0']
        
        # exponential isotropic strain energy
        Psi = a_0/(2.*b_0)*(ufl.exp(b_0*(Ic_-3.)) - 1.)
        
        dPsi_dIc = ufl.diff(Psi,Ic_)
        
        # 2nd PK stress
        S = 2.*dPsi_dIc * I
        
        # pressure contribution of plane stress model: -p C^(-1), with p = 2 (1/(lambda_t1^2 lambda_t2^2) dW/dIc - lambda_t1^2 lambda_t2^2 dW/dIIc) (cf. Holzapfel eq. (6.75) - we don't have an IIc term here)
        S += -2.*dPsi_dIc/(IIIplane) * ufl.inv(Cmod).T
        
        # 1st PK stress P = FS
        P = Fmod * S
        
        # only in-plane components of test function derivatives should be used!
        var_F = ufl.grad(self.var_u) - ufl.dot(ufl.grad(self.var_u),n0n0)
        
        # boundary virtual work
        return -h0*ufl.inner(P,var_F)*dboundary


    ### Volume / flux coupling conditions
            
    # volume
    def volume(self, u, J, F, dboundary):
        
        return -(1./3.)*J*ufl.dot(ufl.dot(ufl.inv(F).T,self.n0), self.x_ref + u)*dboundary
        
    # flux: Q = -dV/dt
    # TeX: \int\limits_{\Gamma_{0}} J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\boldsymbol{v}\;\mathrm{d}A
    def flux(self, v, J, F, dboundary):
        
        return J*ufl.dot(ufl.dot(ufl.inv(F).T,self.n0), v)*dboundary
        
    # surface - derivative of pressure load w.r.t. pressure
    # TeX: \int\limits_{\Gamma_{0}} J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\;\mathrm{d}A
    def surface(self, J, F, dboundary):
        
        return J*ufl.dot(ufl.dot(ufl.inv(F).T,self.n0), self.var_u)*dboundary
