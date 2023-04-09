#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from mathutils import spectral_decomposition_3x3

# fluid mechanics variational forms class
# Principle of Virtual Power
# TeX: \delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}
class variationalform:
    
    def __init__(self, var_v, dv, var_p, dp, n, formulation='nonconservative'):
        self.var_v = var_v
        self.var_p = var_p
        self.dv = dv
        self.dp = dp
        
        self.n = n
        self.formulation = formulation
        # for convenience, use naming n0 for ALE (it's a normal determined by the initial mesh!)
        self.n0 = self.n
    
    ### Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}
    
    def deltaW_kin_navierstokes(self, a, v, rho, ddomain, w=None, Fale=None):
        # standard Eulerian fluid
        if self.formulation=='nonconservative':
            # non-conservative form for Navier-Stokes:
            # TeX: \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v
            return rho*ufl.dot(a + ufl.grad(v) * v, self.var_v)*ddomain
        
        elif self.formulation=='conservative':
            # conservative form for Navier-Stokes
            # TeX: \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}\cdot(\boldsymbol{v}\otimes\boldsymbol{v})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v

            # note that we have div(v o v) = (grad v) v + v (div v), where the latter term is the divergence condition (Holzapfel eq. (1.292))
            return rho*ufl.dot(a + ufl.div(ufl.outer(v,v)), self.var_v)*ddomain
        
        else:
            raise ValueError("Unkown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, Fale=None):
        
        return rho*ufl.dot(ufl.grad(v) * v, self.var_v)*ddomain

    def deltaW_kin_transient_stokes(self, a, v, rho, ddomain, w=None, Fale=None):

        return rho*ufl.dot(a, self.var_v) * J*ddomain

    ### Internal virtual power \delta \mathcal{P}_{\mathrm{int}}

    # TeX: \int\limits_{\Omega}\boldsymbol{\sigma} : \boldsymbol{\nabla}(\delta\boldsymbol{v})\,\mathrm{d}v
    def deltaW_int(self, sig, ddomain, Fale=None):
        return ufl.inner(sig, ufl.grad(self.var_v))*ddomain

    # TeX: \int\limits_{\Omega}\boldsymbol{\nabla}\cdot\boldsymbol{v}\,\delta p\,\mathrm{d}v
    def deltaW_int_pres(self, v, ddomain, w=None, Fale=None):
        return ufl.div(v)*self.var_p*ddomain

    def residual_v_strong(self, a, v, rho, sig):
        
        return rho*(a + ufl.grad(v) * v) - ufl.div(sig)
    
    def residual_p_strong(self, v):
        
        return ufl.div(v)
    
    def f_inert(self, a, v, rho):
        
        return rho*(a + ufl.grad(v) * v)
    
    def f_viscous(self, sig):
        
        return ufl.div(ufl.dev(sig))

    ### External virtual power \delta \mathcal{P}_{\mathrm{ext}}
    
    # Neumann load (Cauchy traction)
    # TeX: \int\limits_{\Gamma} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}a
    def deltaW_ext_neumann_cur(self, func, dboundary, Fale=None):

        return ufl.dot(func, self.var_v)*dboundary

    # Neumann load in current normal direction (Cauchy traction) - coincides with reference normal in Eulerian fluid mechanics
    # TeX: \int\limits_{\Gamma} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
    def deltaW_ext_neumann_normal_cur(self, func, dboundary, Fale=None):

        return func*ufl.dot(self.n, self.var_v)*dboundary
    
    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann_cur(self, v, par1, par2, dboundary, w=None, Fale=None):

        vn = ufl.dot(v,self.n)
        return par1*(vn**2.)/(vn**2. + 0.01*par2**2.) * ufl.min_value(vn,0.) * ufl.dot(v,self.var_v)*dboundary # version from Esmaily Moghadam et al. 2011 if param2 = 0

    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma} c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
    def deltaW_ext_robin_dashpot(self, v, c, dboundary, Fale=None):

        return -c*(ufl.dot(v, self.var_v)*dboundary)

    # Robin condition (dashpot) in normal direction
    # TeX: \int\limits_{\Gamma} c\,(\boldsymbol{n}\otimes \boldsymbol{n})\boldsymbol{v}\cdot\delta\boldsymbol{v}\,\mathrm{d}a = 
    #       \int\limits_{\Gamma} c\,(\boldsymbol{v}\cdot \boldsymbol{n})\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
    def deltaW_ext_robin_dashpot_normal_cur(self, v, c_n, dboundary, Fale=None):

        return -c_n*(ufl.dot(v, self.n)*ufl.dot(self.n, self.var_v)*dboundary)


    # Visco-elastic membrane potential on surface
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{S}(\tilde{\boldsymbol{C}},\dot{\tilde{\boldsymbol{C}}}) : \frac{1}{2}\delta\tilde{\boldsymbol{C}}\,\mathrm{d}A
    def deltaW_ext_membrane(self, F, Fdot, a, params, dboundary, ivar=None, fibfnc=None):
        
        C = F.T*F
        
        n0n0 = ufl.outer(self.n0,self.n0)
        
        I = ufl.Identity(3)
        
        model = params['model']
        
        try: active = params['active_stress']
        except: active = None
        
        if active is not None:
            tau = ivar['tau_a']
            c0, l0 = fibfnc[0], fibfnc[1]
            omega, iota, gamma = params['active_stress']['omega'], params['active_stress']['iota'], params['active_stress']['gamma']

        # wall thickness
        h0 = params['h0']

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

        if model=='membrane_f':
            Fmod = F
        elif model=='membrane':
            # get eigenvalues and eigenvectors of C
            evalC, EprojC = spectral_decomposition_3x3(C)
            U = ufl.sqrt(evalC[0])*EprojC[0] + ufl.sqrt(evalC[1])*EprojC[1] + ufl.sqrt(evalC[2])*EprojC[2]
            R = F*ufl.inv(U)
            # get eigenvalues and eigenvectors of modified C
            evalCmod, EprojCmod = spectral_decomposition_3x3(Cmod)
            Umod = ufl.sqrt(evalCmod[0])*EprojCmod[0] + ufl.sqrt(evalCmod[1])*EprojCmod[1] + ufl.sqrt(evalCmod[2])*EprojCmod[2]
            Fmod = R*Umod
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
        Psi_v = (eta/8.) * ufl.tr(Cmoddot_*Cmoddot_)
        
        dPsi_dIc = ufl.diff(Psi,Ic_)
        dPsi_dIIc = ufl.diff(Psi,IIc_)
        
        # elastic 2nd PK stress
        S = 2.*(dPsi_dIc + Ic*dPsi_dIIc) * I - 2.*dPsi_dIIc * Cmod
        # viscous 2nd PK stress
        S += 2.*ufl.diff(Psi_v,Cmoddot_)
        
        # pressure contribution of plane stress model: -p C^(-1), with p = 2 (1/(lambda_t1^2 lambda_t2^2) dW/dIc - lambda_t1^2 lambda_t2^2 dW/dIIc) (cf. Holzapfel eq. (6.75) - we don't have an IIc term here)
        p = 2.*(dPsi_dIc/(IIIplane) - IIIplane*dPsi_dIIc)
        S += -p * ufl.inv(Cmod).T
        
        # add active stress
        if active is not None:
            S += tau * ( omega*ufl.outer(c0,c0) + iota*ufl.outer(l0,l0) + 2.*gamma*ufl.sym(ufl.outer(c0,l0)) )
        
        # 1st PK stress P = FS
        P = Fmod * S
        
        # only in-plane components of test function derivatives should be used!
        var_F = ufl.grad(self.var_v) - ufl.grad(self.var_v)*n0n0

        # boundary inner virtual power
        dPb_int = (h0*ufl.inner(P,var_F))*dboundary

        # boundary kinetic virtual power
        if not isinstance(a, ufl.constantvalue.Zero):
            dPb_kin = rho0*(h0*ufl.dot(a,self.var_v)*dboundary)
        else:
            dPb_kin = ufl.as_ufl(0)

        # minus signs, since this sums into external virtual power!
        return -dPb_int - dPb_kin


    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}a
    def flux(self, v, dboundary, w=None, Fale=None):
        
        return ufl.dot(self.n, v)*dboundary



# ALE fluid mechanics variational forms class (cf. https://w3.onera.fr/erc-aeroflex/project/strategies-for-coupling-the-fluid-and-solid-dynamics)
# Principle of Virtual Power
# TeX: \delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}
# all infinitesimal volume elements transform according to
# \mathrm{d}v = J\,\mathrm{d}V
# a normal vector times surface element transforms according to Nanson's formula:
# \boldsymbol{n}\,\mathrm{d}a = J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
# hence, all infinitesimal surface elements transform according to
# \mathrm{d}a = J\sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A

class variationalform_ale(variationalform):
    
    ### Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}
    
    def deltaW_kin_navierstokes(self, a, v, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)

        i, j, k = ufl.indices(3)

        if self.formulation=='nonconservative':
            # non-conservative form for ALE Navier-Stokes:
            # TeX:
            # \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})(\boldsymbol{v}-\boldsymbol{w})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v = 
            # \int\limits_{\Omega_0} J\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_{0}\boldsymbol{v}\,\boldsymbol{F}^{-1})(\boldsymbol{v}-\boldsymbol{w})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            return rho*ufl.dot(a + ufl.grad(v)*ufl.inv(Fale) * (v - w), self.var_v) * J*ddomain
        
        elif self.formulation=='conservative':
            # conservative form for ALE Navier-Stokes
            # TeX:
            # \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}\cdot(\boldsymbol{v}\otimes(\boldsymbol{v}-\boldsymbol{w}))\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v = 
            # \int\limits_{\Omega_0} J\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}_{0}(\boldsymbol{v}\otimes(\boldsymbol{v}-\boldsymbol{w})) : \boldsymbol{F}^{-\mathrm{T}}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V

            # note that we have div(v o (v-w)) = (grad v) (v-w) + v (div (v-w)) (Holzapfel eq. (1.292))
            # then use Holzapfel eq. (2.56)
            return rho*ufl.dot(a + ufl.as_tensor(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(Fale).T[j,k], i), self.var_v) *J*ddomain
        
        else:
            raise ValueError("Unkown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)
        return rho*ufl.dot(ufl.grad(v)*ufl.inv(Fale) * (v - w), self.var_v) * J*ddomain

    def deltaW_kin_transient_stokes(self, a, v, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)
        return rho*ufl.dot(a + ufl.grad(v)*ufl.inv(Fale) * (-w), self.var_v) * J*ddomain

    ### Internal virtual power \delta \mathcal{P}_{\mathrm{int}}

    # TeX:
    # \int\limits_{\Omega}\boldsymbol{\sigma} : \boldsymbol{\nabla}(\delta\boldsymbol{v})\,\mathrm{d}v = 
    # \int\limits_{\Omega_0}J\boldsymbol{\sigma} : \boldsymbol{\nabla}_{0}(\delta\boldsymbol{v})\boldsymbol{F}^{-1}\,\mathrm{d}V (Holzapfel eq. (8.43))
    def deltaW_int(self, sig, ddomain, Fale=None):
        J = ufl.det(Fale)
        return ufl.inner(sig, ufl.grad(self.var_v)*ufl.inv(Fale)) * J*ddomain

    # TeX:
    # \int\limits_{\Omega}\boldsymbol{\nabla}\cdot\boldsymbol{v}\,\delta p\,\mathrm{d}v = 
    # \int\limits_{\Omega_0}\boldsymbol{\nabla}_0\cdot(J\boldsymbol{F}^{-1}\boldsymbol{v})\,\delta p\,\mathrm{d}V
    # \int\limits_{\Omega_0}J\,\boldsymbol{\nabla}_0\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}}\,\delta p\,\mathrm{d}V (cf. Holzapfel eq. (2.56))
    def deltaW_int_pres(self, v, ddomain, Fale=None):
        J = ufl.det(Fale)
        return ufl.inner(ufl.grad(v), ufl.inv(Fale).T)*self.var_p * J*ddomain
    
    # Robin term for weak imposition of Dirichlet condition
    # TeX:
    # \int\limits_{\Gamma} \beta\,(\boldsymbol{v}-\boldsymbol{v}_{\mathrm{D}})\cdot\delta\boldsymbol{v}\,\mathrm{d}a = 
    # \int\limits_{\Gamma_0} J\beta\,(\boldsymbol{v}-\boldsymbol{v}_{\mathrm{D}})\cdot\delta\boldsymbol{v}\sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A
    def deltaW_int_robin_cur(self, v, vD, beta, dboundary, Fale=None, fcts=None):
        J = ufl.det(Fale)
        if fcts is None:
            return beta*ufl.dot((v-vD), self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0))*dboundary
        else:
            return (beta*ufl.dot((v-vD), self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0)))(fcts)*dboundary
    
    ### External virtual power \delta \mathcal{P}_{\mathrm{ext}}
    
    # Neumann load (Cauchy traction)
    # TeX:
    # \int\limits_{\Gamma} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}a = 
    # \int\limits_{\Gamma_0} J\boldsymbol{F}^{-\mathrm{T}}\,\hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}A
    def deltaW_ext_neumann_cur(self, func, dboundary, Fale=None):
        J = ufl.det(Fale)
        return J*ufl.dot(ufl.inv(Fale).T*func, self.var_v)*dboundary
    
    # Neumann load in current normal direction (Cauchy traction)
    # TeX: \int\limits_{\Gamma} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a = 
    #      \int\limits_{\Gamma_0} p\,J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\cdot\delta\boldsymbol{v}\,\mathrm{d}A
    def deltaW_ext_neumann_normal_cur(self, func, dboundary, Fale=None):
        J = ufl.det(Fale)
        return func*J*ufl.dot(ufl.inv(Fale).T*self.n0, self.var_v)*dboundary
    
    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann_cur(self, v, par1, par2, dboundary, w=None, Fale=None):
        J = ufl.det(Fale)
        vwn = ufl.dot(v-w, J*ufl.inv(Fale).T*self.n)
        return par1*(vwn**2.)/(vwn**2. + 0.01*par2**2.) * ufl.min_value(vwn,0.) * ufl.dot(v,self.var_v)*dboundary # version from Esmaily Moghadam et al. 2011 if param2 = 0

    # Robin condition (dashpot)
    # TeX:
    # \int\limits_{\Gamma} c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\,\mathrm{d}a = 
    # \int\limits_{\Gamma_0} J c\,\boldsymbol{v}\cdot\delta\boldsymbol{v} \sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A
    def deltaW_ext_robin_dashpot(self, v, c, dboundary, Fale=None):
        J = ufl.det(Fale)
        return -c*(ufl.dot(v, self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0))*dboundary)
    
    # Robin condition (dashpot) in normal direction
    def deltaW_ext_robin_dashpot_normal_cur(self, v, c_n, dboundary, Fale=None):
        raise ValueError("Robin condition in current normal direction not implemented")
        

    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} (\boldsymbol{v}-\boldsymbol{w})\cdot\boldsymbol{n}\,\mathrm{d}a = 
    #      \int\limits_{\Gamma_0} (\boldsymbol{v}-\boldsymbol{w})\cdot J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
    def flux(self, v, dboundary, Fale=None):
        J = ufl.det(Fale)
        return J*ufl.dot(ufl.inv(Fale).T*self.n0, v)*dboundary
