#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from .mathutils import spectral_decomposition_3x3

"""
Variational form base class
"""

class variationalform_base:

    def __init__(self, var_u, var_p=None, du=None, dp=None, n0=None, x_ref=None, formulation=None):

        self.var_u = var_u  # displacement/velocity test functions
        self.var_p = var_p  # pressure test functions
        self.du = du        # displacement/velocity trial functions
        self.dp = dp        # pressure trial functions

        self.var_v = var_u  # for naming convenience, to use var_v in derived fluid class

        self.n0 = n0        # reference normal field
        self.x_ref = x_ref  # reference coordinates

        self.formulation = formulation # fluid formulation (conservative or non-conservative)

        self.I = ufl.Identity(len(self.var_u)) # identity
        self.dim = self.I.ufl_shape[0] # dimension


    # Neumann load on reference configuration (1st Piola-Kirchhoff traction)
    # TeX: \int\limits_{\Gamma_{0}} \hat{\boldsymbol{t}}_{0} \cdot \delta\boldsymbol{u} \,\mathrm{d}A
    def deltaW_ext_neumann_ref(self, func, dboundary):

        return ufl.dot(func, self.var_u)*dboundary

    # Neumann load in reference normal (1st Piola-Kirchhoff traction)
    # TeX: \int\limits_{\Gamma_{0}} p\,\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_neumann_normal_ref(self, func, dboundary):

        return func*ufl.dot(self.n0, self.var_u)*dboundary

    # Neumann follower load on current configuration (Cauchy traction)
    # TeX: \int\limits_{\Gamma_0} J\boldsymbol{F}^{-\mathrm{T}}\,\hat{\boldsymbol{t}} \cdot \delta\boldsymbol{u} \,\mathrm{d}A
    def deltaW_ext_neumann_cur(self, func, dboundary, F=None):
        if F is not None:
            J = ufl.det(F)
            return J*ufl.dot(ufl.inv(F).T*func, self.var_u)*dboundary
        else:
            return self.deltaW_ext_neumann_ref(func, dboundary)

    # Neumann follower load in current normal direction
    # TeX: \int\limits_{\Gamma_{0}} p\,J \boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_{0}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_neumann_normal_cur(self, func, dboundary, F=None):
        if F is not None:
            J = ufl.det(F)
            return func*J*ufl.dot(ufl.inv(F).T*self.n0, self.var_u)*dboundary
        else:
            return self.deltaW_ext_neumann_normal_ref(func, dboundary)


    # body force external virtual work
    # TeX: \int\limits_{\Omega_{0}} \hat{\boldsymbol{b}}\cdot\delta\boldsymbol{u}\,\mathrm{d}V
    def deltaW_ext_bodyforce(self, func, funcdir, ddomain, F=None):
        if F is not None:
            J = ufl.det(F)
            return func*ufl.dot(funcdir, self.var_u)*J*ddomain # for ALE fluid
        else:
            return func*ufl.dot(funcdir, self.var_u)*ddomain


    # Robin condition (spring)
    # TeX: \int\limits_{\Gamma_0} k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_robin_spring(self, u, k, dboundary, u_prestr=None):

        if u_prestr is not None:
            return -k*(ufl.dot(u + u_prestr, self.var_u)*dboundary)
        else:
            return -k*(ufl.dot(u, self.var_u)*dboundary)

    # Robin condition (spring) in reference normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}_{0}\otimes \boldsymbol{n}_{0})\,k\,\boldsymbol{u}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_robin_spring_normal_ref(self, u, k_n, dboundary, u_prestr=None):

        if u_prestr is not None:
            return -k_n*(ufl.dot(ufl.outer(self.n0,self.n0)*(u + u_prestr), self.var_u)*dboundary)
        else:
            return -k_n*(ufl.dot(ufl.outer(self.n0,self.n0)*u, self.var_u)*dboundary)

    # Robin condition (spring) in cross normal direction
    def deltaW_ext_robin_spring_normal_cross(self, u, k_c, dboundary, u_prestr=None):

        if u_prestr is not None:
            return -k_c*(ufl.dot((self.I - ufl.outer(self.n0,self.n0))*(u + u_prestr), self.var_u)*dboundary)
        else:
            return -k_c*(ufl.dot((self.I - ufl.outer(self.n0,self.n0))*u, self.var_u)*dboundary)

    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma_0} c\,\dot{\boldsymbol{u}}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_robin_dashpot(self, v, c, dboundary):

        if not isinstance(v, ufl.constantvalue.Zero):
            return -c*(ufl.dot(v, self.var_u)*dboundary)
        else:
            return ufl.as_ufl(0)

    # Robin condition (dashpot) in reference normal direction
    # TeX: \int\limits_{\Gamma_0} (\boldsymbol{n}_{0}\otimes \boldsymbol{n}_{0})\,c\,\dot{\boldsymbol{u}}\cdot\delta\boldsymbol{u}\,\mathrm{d}A
    def deltaW_ext_robin_dashpot_normal_ref(self, v, c_n, dboundary):

        if not isinstance(v, ufl.constantvalue.Zero):
            return -c_n*(ufl.dot(ufl.outer(self.n0,self.n0)*v, self.var_u)*dboundary)
        else:
            return ufl.as_ufl(0)

    # Robin condition (dashpot) in cross normal direction
    def deltaW_ext_robin_dashpot_normal_cross(self, v, c_c, dboundary):

        if not isinstance(v, ufl.constantvalue.Zero):
            return -c_c*(ufl.dot((self.I - ufl.outer(self.n0,self.n0))*v, self.var_u)*dboundary)
        else:
            return ufl.as_ufl(0)


    # Hyper-visco-elastic membrane model defined on a surface
    # for solid mechanics, contribution to virtual work is:
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{P}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) : \boldsymbol{\nabla}_{\tilde{\boldsymbol{X}}}\delta\boldsymbol{u}\,\mathrm{d}A
    # for fluid mechanics, contribution to virtual power is:
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{P}(\boldsymbol{u}_{\mathrm{f}}(\boldsymbol{v}),\boldsymbol{v}) : \boldsymbol{\nabla}_{\tilde{\boldsymbol{X}}}\delta\boldsymbol{v}\,\mathrm{d}A
    def deltaW_ext_membrane(self, F, Fdot, a, params, dboundary, ivar=None, fibfnc=None, wallfield=None, fcts=None, returnquantity='weakform'):

        C = F.T*F

        n0n0 = ufl.outer(self.n0,self.n0)

        model = params['model']

        try: material = params['material']
        except: material = 'isoexp'

        try: active = params['active_stress']
        except: active = None

        if active is not None:
            tau = ivar['tau_a']
            if params['active_stress']['dir']=='cl':
                c0, l0 = fibfnc[0], fibfnc[1]
                omega, iota, gamma = params['active_stress']['omega'], params['active_stress']['iota'], params['active_stress']['gamma']
            elif params['active_stress']['dir']=='iso':
                pass
            else:
                ValueError("Unknown ative stress dir!")

        # wall thickness - can be constant or a field
        wall_thickness = params['h0']
        if 'val' in wall_thickness.keys():
            h0 = wall_thickness['val']
        elif 'field' in wall_thickness.keys():
            h0 = wallfield
        else:
            raise ValueError("Have to have either val or field in h0 dict!")

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
        Cmoddot = Fdotmod.T*F0 + F0.T*Fdotmod - (IIIplanedot/(IIIplane**2.)) * n0n0

        if model=='membrane':
            Fmod = F0
        elif model=='membrane_fmod':
            raise RuntimeError("Model 'membrane_fmod' seems incompatible and gives erroneous results. To be investigated...")
            # get eigenvalues and eigenvectors of C
            evalC, _, EprojC = spectral_decomposition_3x3(C)
            U = ufl.sqrt(evalC[0])*EprojC[0] + ufl.sqrt(evalC[1])*EprojC[1] + ufl.sqrt(evalC[2])*EprojC[2]
            R = F*ufl.inv(U)
            # get eigenvalues and eigenvectors of modified C
            evalCmod, _, EprojCmod = spectral_decomposition_3x3(Cmod)
            Umod = ufl.sqrt(evalCmod[0])*EprojCmod[0] + ufl.sqrt(evalCmod[1])*EprojCmod[1] + ufl.sqrt(evalCmod[2])*EprojCmod[2]
            Fmod = R*Umod
        else:
            raise NameError("Unknown membrane model type!")

        # first and second invariant
        Ic = ufl.tr(Cmod)
        IIc = 0.5*(ufl.tr(Cmod)**2. - ufl.tr(Cmod*Cmod))
        # declare variables for diff
        Ic_ = ufl.variable(Ic)
        IIc_ = ufl.variable(IIc)
        Cmoddot_ = ufl.variable(Cmoddot)

        a_0 = params['a_0']
        try: eta = params['eta']
        except: eta = 0.
        try: rho0 = params['rho0']
        except: rho0 = 0.

        # exponential isotropic strain energy
        if material == 'isoexp':
            b_0 = params['b_0']
            Psi = a_0/(2.*b_0)*(ufl.exp(b_0*(Ic_-self.dim)) - 1.)
        elif material == 'neohooke':
            Psi = (a_0/2.) * (Ic_ - self.dim)
        else:
            raise ValueError("Unknown membrane elastic material!")

        # viscous pseudo-potential
        Psi_v = (eta/8.) * ufl.tr(Cmoddot_*Cmoddot_)

        dPsi_dIc = ufl.diff(Psi,Ic_)
        dPsi_dIIc = ufl.diff(Psi,IIc_)

        # elastic 2nd PK stress
        S = 2.*(dPsi_dIc + Ic*dPsi_dIIc) * self.I - 2.*dPsi_dIIc * Cmod
        # viscous 2nd PK stress
        S += 2.*ufl.diff(Psi_v,Cmoddot_)

        # pressure contribution of plane stress model: -p C^(-1), with p = 2 (1/(lambda_t1^2 lambda_t2^2) dW/dIc - lambda_t1^2 lambda_t2^2 dW/dIIc) (cf. Holzapfel eq. (6.75) - we don't have an IIc term here)
        p = 2.*(dPsi_dIc/(IIIplane) - IIIplane*dPsi_dIIc)
        # balance viscous normal stresses
        p -= (eta/2.) * (IIIplanedot/(IIIplane**3.))

        S += -p * ufl.inv(Cmod)

        # add active stress
        if active is not None:
            if params['active_stress']['dir']=='cl':
                S += tau * ( omega*ufl.outer(c0,c0) + iota*ufl.outer(l0,l0) + 2.*gamma*ufl.sym(ufl.outer(c0,l0)) )
            if params['active_stress']['dir']=='iso':
                S += tau * I

        # 1st PK stress P = FS
        P = Fmod * S

        # Cauchy stress for postprocessing: sigma = (1/J) P*F^T --> membrane is incompressible, hence J=1
        sigma = P * Fmod.T

        # strain energy and internal power of membrane, for postprocessing
        strainenergy = h0 * Psi
        internalpower = h0 * 0.5*ufl.inner(S,Cmoddot)

        # only in-plane components of test function derivatives should be used!
        var_F = ufl.grad(self.var_u) - ufl.grad(self.var_u)*n0n0

        # boundary inner virtual work/power
        if fcts is None:
            dWb_int = h0*ufl.inner(P,var_F)*dboundary
        else:
            dWb_int = (h0*ufl.inner(P,var_F))(fcts)*dboundary

        # boundary kinetic virtual work/power
        if not isinstance(a, ufl.constantvalue.Zero):
            if fcts is None:
                dWb_kin = rho0*(h0*ufl.dot(a,self.var_u)*dboundary)
            else:
                dWb_kin = rho0*(h0*ufl.dot(a,self.var_u)(fcts)*dboundary)
        else:
            dWb_kin = ufl.as_ufl(0)

        # minus signs, since this sums into external virtual work/power!
        if returnquantity=='weakform':
            return -dWb_int - dWb_kin
        elif returnquantity=='stress_energy_power':
            return sigma, strainenergy, internalpower
        else:
            raise ValueError("Unknown return type.")
