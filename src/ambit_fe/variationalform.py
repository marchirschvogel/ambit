#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from .mathutils import spectral_decomposition_3x3

# variational form base class
class variationalform_base:

    # Hyper-visco-elastic membrane model defined on a surface
    # for solid mechanics, contribution to virtual work is:
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{P}(\boldsymbol{u},\boldsymbol{v}(\boldsymbol{u})) : \boldsymbol{\nabla}_{\tilde{\boldsymbol{X}}}\delta\boldsymbol{u}\,\mathrm{d}A
    # for fluid mechanics, contribution to virtual power is:
    # TeX: h_0\int\limits_{\Gamma_0} \boldsymbol{P}(\boldsymbol{u}_{\mathrm{f}}(\boldsymbol{v}),\boldsymbol{v}) : \boldsymbol{\nabla}_{\tilde{\boldsymbol{X}}}\delta\boldsymbol{v}\,\mathrm{d}A
    def deltaW_ext_membrane(self, F, Fdot, a, varu, params, dboundary, ivar=None, fibfnc=None, stress=False, wallfield=None):

        C = F.T*F

        n0n0 = ufl.outer(self.n0,self.n0)

        I = ufl.Identity(3)

        model = params['model']

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

        # only in-plane components of test function derivatives should be used!
        var_F = ufl.grad(varu) - ufl.grad(varu)*n0n0

        # boundary inner virtual work/power
        dWb_int = h0*ufl.inner(P,var_F)*dboundary

        # boundary kinetic virtual work/power
        if not isinstance(a, ufl.constantvalue.Zero):
            dWb_kin = rho0*(h0*ufl.dot(a,varu)*dboundary)
        else:
            dWb_kin = ufl.as_ufl(0)

        # minus signs, since this sums into external virtual work/power!
        if not stress:
            return -dWb_int - dWb_kin
        else:
            return sigma
