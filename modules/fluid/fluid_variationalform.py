#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

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
            return rho*ufl.dot(a + ufl.grad(v) * v, self.var_v)*ddomain
        else:
            return rho*ufl.dot(a + ufl.grad(v) * v_old, self.var_v)*ddomain

    ### Internal virtual power

    # TeX: \delta \mathcal{P}_{\mathrm{int}} := \int\limits_{\Omega} \boldsymbol{\sigma} : \delta\boldsymbol{\gamma} \,\mathrm{d}v
    def deltaP_int(self, sig, ddomain):
        
        # TeX: \int\limits_{\Omega}\boldsymbol{\sigma} : \delta \boldsymbol{\gamma}\,\mathrm{d}v
        var_gamma = 0.5*(ufl.grad(self.var_v).T + ufl.grad(self.var_v))
        return ufl.inner(sig, var_gamma)*ddomain

    def deltaP_int_pres(self, v, ddomain):
        # TeX: \int\limits_{\Omega}\mathrm{div}\boldsymbol{v}\,\delta p\,\mathrm{d}v
        return ufl.div(v)*self.var_p*ddomain

    def residual_v_strong(self, a, v, rho, sig):
        
        return rho*(a + ufl.grad(v) * v) - ufl.div(sig)
    
    def residual_p_strong(self, v):
        
        return ufl.div(v)
    
    def f_inert(self, a, v, rho):
        
        return rho*(a + ufl.grad(v) * v)
    
    def f_viscous(self, sig):
        
        return ufl.div(ufl.dev(sig))

    ### External virtual power
    
    # Neumann load (Cauchy traction)
    # TeX: \int\limits_{\Gamma} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}a
    def deltaP_ext_neumann(self, func, dboundary):

        return ufl.dot(func, self.var_v)*dboundary
    
    # Neumann load in normal direction (Cauchy traction)
    # TeX: \int\limits_{\Gamma} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def deltaP_ext_neumann_normal(self, func, dboundary):

        return func*ufl.dot(self.n, self.var_v)*dboundary
    
    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma} c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def deltaP_ext_robin_dashpot(self, v, c, dboundary):

        return -c*(ufl.dot(v, self.var_v)*dboundary)
    
    # Robin condition (dashpot) in normal direction
    # TeX: \int\limits_{\Gamma} (\boldsymbol{n}\otimes \boldsymbol{n})\,c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def deltaP_ext_robin_dashpot_normal(self, v, c_n, dboundary):

        return -c_n*(ufl.dot(v, self.n)*ufl.dot(self.n, self.var_v)*dboundary)


    # Visco-elastic membrane potential on surface
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{S}(\tilde{\boldsymbol{C}},\dot{\tilde{\boldsymbol{C}}}) : \frac{1}{2}\delta\tilde{\boldsymbol{C}}\;\mathrm{d}A
    def deltaP_ext_membrane(self, F, Fdot, a, params, dboundary):
        
        # use n0 naming for convenience
        self.n0 = self.n
        
        C = F.T*F
        
        n0n0 = ufl.outer(self.n0,self.n0)
        
        I = ufl.Identity(3)
        
        model = params['model']
        
        # wall thickness
        h0 = params['h0']
        
        if model=='membrane_f':
            # only components in normal direction (F_nn, F_t1n, F_t2n)
            Fn = F*n0n0
            Fdotn = Fdot*n0n0
            # rank-deficient deformation gradient and Cauchy-Green tensor (phased-out normal components)
            F0 = F - Fn
            C0 = F0.T*F0
            # plane strain deformation tensor where deformation is "1" in normal direction
            Cplane = C0 + n0n0
            # determinant: corresponds to product of in-plane stretches lambda_t1^2 * lambda_t2^2
            IIIplane = ufl.det(Cplane)
            # deformation tensor where normal stretch is dependent on in-plane stretches
            Cmod = C0 + (1./IIIplane) * n0n0
            # rates of deformation: in-plane time derivatives of deformation gradient and Cauchy-Green tensor
            Fdotmod = Fdot - Fdotn
            Cplanedot = Fdotmod.T*F0 + F0.T*Fdotmod
            # Jacobi's formula: d(detA)/dt = detA * tr(A^-1 * dA/dt)
            IIIplanedot = IIIplane * ufl.tr(ufl.inv(Cplane) * Cplanedot)
            # time derivative of Cmod
            Cmoddot = Fdotmod.T*F0 + F0.T*Fdotmod - (IIIplanedot/(IIIplane*IIIplane)) * n0n0
            # TODO: Need to recover an Fmod corresponding to Cmod!
            # Can this be done in ufl? See e.g. https://fenicsproject.org/qa/13600/possible-perform-spectral-decomposition-current-operators
            Fmod = F
        else:
            raise NameError("Unkown membrane model type!")
        
        # first and second invariant
        Ic = ufl.tr(Cmod)
        IIc  = 0.5*(ufl.tr(Cmod)**2. - ufl.tr(Cmod*Cmod))
        # declare variables for diff
        Ic_ = ufl.variable(Ic)
        IIc_ = ufl.variable(IIc)
        Cmoddot_ = ufl.variable(Cmoddot)
        
        a_0, b_0 = params['a_0'], params['b_0']
        try: eta = params['eta']
        except: eta = 0.
        try: rho0 = params['rho0']
        except: rho0 = 0.

        # exponential isotropic strain energy
        Psi = a_0/(2.*b_0)*(ufl.exp(b_0*(Ic_-3.)) - 1.)
        # viscous pseudo-potential
        Psi_v = (eta/8.) * eta * ufl.tr(Cmoddot_*Cmoddot_)
        
        dPsi_dIc = ufl.diff(Psi,Ic_)
        dPsi_dIIc = ufl.diff(Psi,IIc_)
        
        # elastic 2nd PK stress
        S = 2.*(dPsi_dIc + Ic*dPsi_dIIc) * I - 2.*dPsi_dIIc * Cmod
        # viscous 2nd PK stress
        S += 2.*ufl.diff(Psi_v,Cmoddot_)
        
        # pressure contribution of plane stress model: -p C^(-1), with p = 2 (1/(lambda_t1^2 lambda_t2^2) dW/dIc - lambda_t1^2 lambda_t2^2 dW/dIIc) (cf. Holzapfel eq. (6.75) - we don't have an IIc term here)
        p = 2.*(dPsi_dIc/(IIIplane) - IIIplane*dPsi_dIIc)
        S += -p * ufl.inv(Cmod).T
        
        # 1st PK stress P = FS
        P = Fmod * S
        
        # only in-plane components of test function derivatives should be used!
        var_F = ufl.grad(self.var_v) - ufl.dot(ufl.grad(self.var_v),n0n0)

        # boundary inner virtual power
        dPb_int = h0*ufl.inner(P,var_F)*dboundary

        # boundary kinetic virtual work
        if not isinstance(a, ufl.constantvalue.Zero):
            dPb_kin = h0*rho0*ufl.dot(a,self.var_v)*dboundary
        else:
            dPb_kin = ufl.as_ufl(0)

        # minus signs, since this sums into external virtual work!
        return -dPb_int - dPb_kin


    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{v}\;\mathrm{d}a
    def flux(self, v, dboundary):
        
        return ufl.dot(self.n, v)*dboundary
        
    # surface - derivative of pressure load w.r.t. pressure
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\delta\boldsymbol{v}\;\mathrm{d}a
    def surface(self, dboundary):
        
        return ufl.dot(self.n, self.var_v)*dboundary
