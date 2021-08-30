#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ufl import tr, det, dot, ln, sqrt, exp, diff, cos, pi, conditional, ge, le, gt, inner, outer, cross, Max, Min, And, as_ufl

# returns the 2nd Piola-Kirchhoff stress S for different material laws

class materiallaw:
    
    def __init__(self, C, I):
        self.C = C
        self.I = I

        # Cauchy-Green invariants
        self.Ic   = tr(C)
        self.IIc  = 0.5*(tr(C)**2. - tr(C*C))
        self.IIIc = det(C)
        # isochoric Cauchy-Green invariants
        self.Ic_bar   = self.IIIc**(-1./3.) * self.Ic
        self.IIc_bar  = self.IIIc**(-2./3.) * self.IIc
        
        # Green-Lagrange strain and invariants (for convenience, used e.g. by St.-Venant-Kirchhoff material)
        self.E = 0.5*(C - I)
        self.trE  = tr(self.E)
        self.trE2 = tr(self.E*self.E)
    

    def neohooke_dev(self, params, C):
        
        mu = params['mu']
        
        # classic NeoHookean material (isochoric version)
        psi_dev = (mu/2.) * (self.Ic_bar - 3.)
        
        S = 2.*diff(psi_dev,C)
        
        return S


    def mooneyrivlin_dev(self, params, C):
        
        c1, c2 = params['c1'], params['c2']
        
        # Mooney Rivlin material (isochoric version)
        psi_dev = c1 * (self.Ic_bar - 3.) + c2 * (self.IIc_bar - 3.)
        
        S = 2.*diff(psi_dev,C)
        
        return S


    def exponential_dev(self, params, C):
        
        a_0, b_0 = params['a_0'], params['b_0']
        
        # exponential SEF (isochoric version)
        psi_dev = a_0/(2.*b_0)*(exp(b_0*(self.Ic_bar-3.)) - 1.)
        
        S = 2.*diff(psi_dev,C)
        
        return S
    

    def holzapfelogden_dev(self, params, f0, s0, C):

        # anisotropic invariants - keep in mind that for growth, self.C is the elastic part of C (hence != to function input variable C)
        I4 = dot(dot(self.C,f0), f0)
        I6 = dot(dot(self.C,s0), s0)
        I8 = dot(dot(self.C,s0), f0)

        # to guarantee initial configuration is stress-free (in case of initially non-orthogonal fibers f0 and s0)
        I8 -= dot(f0,s0)

        a_0, b_0 = params['a_0'], params['b_0']
        a_f, b_f = params['a_f'], params['b_f']
        a_s, b_s = params['a_s'], params['b_s']
        a_fs, b_fs = params['a_fs'], params['b_fs']

        try: fiber_comp = params['fiber_comp']
        except: fiber_comp = False

        # conditional parameters: fibers are only active in tension if fiber_comp is False
        if not fiber_comp:
            a_f_c = conditional(ge(I4,1.), a_f, 0.)
            a_s_c = conditional(ge(I6,1.), a_s, 0.)
        else:
            a_f_c = a_f
            a_s_c = a_s
        
        # Holzapfel-Ogden (Holzapfel and Ogden 2009) material w/o split applied to invariants I4, I6, I8 (Nolan et al. 2014, Sansour 2008)
        psi_dev = a_0/(2.*b_0)*(exp(b_0*(self.Ic_bar-3.)) - 1.) + \
            a_f_c/(2.*b_f)*(exp(b_f*(I4-1.)**2.) - 1.) + a_s_c/(2.*b_s)*(exp(b_s*(I6-1.)**2.) - 1.) + \
            a_fs/(2.*b_fs)*(exp(b_fs*I8**2.) - 1.)
        
        S = 2.*diff(psi_dev,C)
        
        return S


    def guccione_dev(self, params, f0, s0, C):

        n0 = cross(f0,s0)

        # anisotropic invariants - keep in mind that for growth, self.E is the elastic part of E
        E_ff = dot(dot(self.E,f0), f0) # fiber GL strain
        E_ss = dot(dot(self.E,s0), s0) # cross-fiber GL strain
        E_nn = dot(dot(self.E,n0), n0) # radial GL strain
        
        E_fs = dot(dot(self.E,f0), s0)
        E_fn = dot(dot(self.E,f0), n0)
        E_sn = dot(dot(self.E,s0), n0)

        c_0 = params['c_0']
        b_f = params['b_f']
        b_t = params['b_t']
        b_fs = params['b_fs']

        Q = b_f*E_ff**2. + b_t*(E_ss**2. + E_nn**2. + 2.*E_sn**2.) + b_fs*(2.*E_fs**2. + 2.*E_fn**2.)

        psi_dev = 0.5*c_0*(exp(Q)-1.)
        
        S = 2.*diff(psi_dev,C)
        
        return S

    
    def stvenantkirchhoff(self, params, C):
    
        Emod, nu = params['Emod'], params['nu']
        
        psi = Emod*nu/( 2.*(1.+nu)*(1.-2.*nu) ) * self.trE**2. + Emod/(2.*(1.+nu)) * self.trE2
        
        S = 2.*diff(psi,C)
        
        return S
    
    
    def sussmanbathe_vol(self, params, C):
        
        kappa = params['kappa']
        
        psi_vol = (kappa/2.) * (sqrt(self.IIIc) - 1.)**2.
        
        S = 2.*diff(psi_vol,C)
        
        return S
    
    
    def ogden_vol(self, params, C):
        
        kappa = params['kappa']
        
        psi_vol = (kappa/4.) * (self.IIIc - 2.*ln(sqrt(self.IIIc)) - 1.)
        
        S = 2.*diff(psi_vol,C)
        
        return S
    
    # simple Green-Lagrange strain rate-dependent material, pseudo potential Psi_v = 0.5 * eta * dEdt : dEdt
    def visco(self, params, dEdt):
        
        eta = params['eta']

        S = 2.*eta*dEdt
        
        return S
    
    
    def active_fiber(self, tau, f0):
        
        S = tau * outer(f0,f0)
        
        return S
    
    
    def active_iso(self, tau):
        
        S = tau * self.I
        
        return S




# inversion of growth tensors can be performed with Sherman-Morrison formula:
# TeX: \left(\boldsymbol{A}-\boldsymbol{u}\otimes \boldsymbol{v}\right)^{-1} = \boldsymbol{A}^{-1} + \frac{\boldsymbol{A}^{-1}\,\boldsymbol{u}\otimes \boldsymbol{v}\,\boldsymbol{A}^{-1}}{1-\boldsymbol{v}\cdot \boldsymbol{A}^{-1}\boldsymbol{u}}

class growth:
    
    def __init__(self, theta, I):
        self.theta = theta
        self.I = I
    

    def isotropic(self):
        
        F_g = self.theta * self.I
        
        return F_g
    
    
    def fiber(self, f0):
        
        F_g = self.I + (self.theta-1.)*outer(f0,f0)
        
        return F_g
    
    
    def crossfiber(self, f0):
        
        F_g = self.theta * self.I + (1.-self.theta)*outer(f0,f0)
        
        return F_g
    
    
    def radial(self, f0, s0):
        
        r0 = cross(f0,s0)
        F_g = self.I + (self.theta-1.)*outer(r0,r0)
        
        return F_g






class growthfunction(growth):
    
    # add possible variations / different growth functions here...
    
    def grfnc1(self, trigger, thres, grparfuncs):
        
        thetamax, thetamin = grparfuncs['thetamax'], grparfuncs['thetamin']
        tau_gr, tau_gr_rev = grparfuncs['tau_gr'], grparfuncs['tau_gr_rev']
        gamma_gr, gamma_gr_rev = grparfuncs['gamma_gr'], grparfuncs['gamma_gr_rev']
        
        k_plus = (1./tau_gr) * ((thetamax-self.theta)/(thetamax-thetamin))**(gamma_gr)
        k_minus = (1./tau_gr_rev) * ((self.theta-thetamin)/(thetamax-thetamin))**(gamma_gr_rev)
        
        k = conditional(ge(trigger,thres), k_plus, k_minus)
            
        return k




# expression for time-dependent active stress activation function
class activestress_activation:
    
    def __init__(self, params, act_curve):
        
        self.params = params
        
        self.sigma0 = self.params['sigma0']
        self.alpha_max = self.params['alpha_max']
        self.alpha_min = self.params['alpha_min']
        
        self.act_curve = act_curve
        
        if 'frankstarling' in self.params.keys(): self.frankstarling = self.params['frankstarling']
        else: self.frankstarling = False


    # activation function for active stress
    def ua(self, t):
        
        # Diss Hirschvogel eq. 2.100
        return self.act_curve(t)*self.alpha_max + (1.-self.act_curve(t))*self.alpha_min
    
    
    # Frank-Staring function
    def g(self, lam):
        
        amp_min = self.params['amp_min']
        amp_max = self.params['amp_max']
        
        lam_threslo = self.params['lam_threslo']
        lam_maxlo = self.params['lam_maxlo']
        
        lam_threshi = self.params['lam_threshi']
        lam_maxhi = self.params['lam_maxhi']

        # Diss Hirschvogel eq. 2.107
        # TeX: g(\lambda_{\mathrm{myo}}) = \begin{cases} a_{\mathrm{min}}, & \lambda_{\mathrm{myo}} \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}}, \\ a_{\mathrm{min}}+\frac{1}{2}\left(a_{\mathrm{max}}-a_{\mathrm{min}}\right)\left(1-\cos \frac{\pi(\lambda_{\mathrm{myo}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}})}{\hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,lo}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}}}\right), &  \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}} \leq \lambda_{\mathrm{myo}}  \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,lo}}, \\ a_{\mathrm{max}}, &  \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,lo}} \leq \lambda_{\mathrm{myo}} \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,hi}}, \\ a_{\mathrm{min}}+\frac{1}{2}\left(a_{\mathrm{max}}-a_{\mathrm{min}}\right)\left(1-\cos \frac{\pi(\lambda_{\mathrm{myo}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}})}{\hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,hi}}}\right), & \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,hi}} \leq \lambda_{\mathrm{myo}} \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}}, \\ a_{\mathrm{min}}, & \lambda_{\mathrm{myo}} \geq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}} \end{cases}
        
        return conditional( le(lam,lam_threslo), amp_min, conditional( And(ge(lam,lam_threslo),le(lam,lam_maxlo)), amp_min + 0.5*(amp_max-amp_min)*(1.-cos(pi*(lam-lam_threslo)/(lam_maxlo-lam_threslo))), conditional( And(ge(lam,lam_maxlo),le(lam,lam_threshi)), amp_max, conditional( And(ge(lam,lam_threshi),le(lam,lam_maxhi)), amp_min + 0.5*(amp_max-amp_min)*(1.-cos(pi*(lam-lam_maxhi)/(lam_maxhi-lam_threshi))), conditional( ge(lam,lam_maxhi), amp_min, as_ufl(0)) ) ) ))


    # Frank Starling amplification factor (Diss Hirschvogel eq. 2.106, 3.29)
    # \dot{a}(\lambda_{\mathrm{myo}}) = \dot{g}(\lambda_{\mathrm{myo}}) \,\mathbb{I}_{|u|_{-}>0}
    def amp(self, t, lam, amp_old):
        
        uabs_minus = Max(-Min(self.ua(t),0),0)

        return conditional(gt(uabs_minus,0.), self.g(lam), amp_old)


    # Backward-Euler integration of active stress
    def tau_act(self, tau_a_old, t, dt, lam=None, amp_old=None):
        
        uabs = abs(self.ua(t))
        uabs_plus = Max(self.ua(t),0)
        
        # Frank Starling amplification factor
        if self.frankstarling:
            amp = self.amp(t, lam, amp_old)
        else:
            amp = 1.
            
        return (tau_a_old + amp*self.sigma0 * uabs_plus*dt) / (1.+uabs*dt)
