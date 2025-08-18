#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
import numpy as np


# returns the 2nd Piola-Kirchhoff stress S for different material laws


class materiallaw:
    def __init__(self, C, Cdot, I):
        self.C = C
        self.I = I
        self.Cdot = Cdot

        self.dim = I.ufl_shape[0]

        # Cauchy-Green invariants
        self.Ic = ufl.tr(C)
        self.IIc = 0.5 * (ufl.tr(C) ** 2.0 - ufl.tr(C * C))
        self.IIIc = ufl.det(C)
        self.J = ufl.sqrt(self.IIIc)
        # isochoric Cauchy-Green invariants
        self.Ic_bar = self.IIIc ** (-1.0 / self.dim) * self.Ic
        self.IIc_bar = self.IIIc ** (-2.0 / self.dim) * self.IIc

        # Green-Lagrange strain and invariants (for convenience, used e.g. by St.-Venant-Kirchhoff material)
        self.E = 0.5 * (C - I)
        self.trE = ufl.tr(self.E)
        self.trE2 = ufl.tr(self.E * self.E)
        # rate of Green-Lagrange strain and invariant
        self.Edot = 0.5 * Cdot

    def neohooke_dev(self, params, C):
        mu = params["mu"]

        # classic NeoHookean material (isochoric version)
        Psi_dev = (mu / 2.0) * (self.Ic_bar - self.dim)

        S = 2.0 * ufl.diff(Psi_dev, C)

        return S, Psi_dev

    def yeoh_dev(self, params, C):
        c1, c2, c3 = params["c1"], params["c2"], params["c3"]

        # Yeoh material (isochoric version) - generalized NeoHookean model
        Psi_dev = (
            c1 * (self.Ic_bar - self.dim) + c2 * (self.Ic_bar - self.dim) ** 2.0 + c3 * (self.Ic_bar - self.dim) ** 3.0
        )

        S = 2.0 * ufl.diff(Psi_dev, C)

        return S, Psi_dev

    def mooneyrivlin_dev(self, params, C):
        c1, c2 = params["c1"], params["c2"]

        # Mooney Rivlin material (isochoric version)
        Psi_dev = c1 * (self.Ic_bar - self.dim) + c2 * (self.IIc_bar - self.dim)

        S = 2.0 * ufl.diff(Psi_dev, C)

        return S, Psi_dev

    def exponential_dev(self, params, C):
        a_0, b_0 = params["a_0"], params["b_0"]

        # exponential SEF (isochoric version)
        Psi_dev = a_0 / (2.0 * b_0) * (ufl.exp(b_0 * (self.Ic_bar - self.dim)) - 1.0)

        S = 2.0 * ufl.diff(Psi_dev, C)

        return S, Psi_dev

    def holzapfelogden_dev(self, params, fib1, fib2, C):
        # to tell the material what kind of fibers we have: fs, fn, or sn
        fibers_type = params.get("fibers_type", "fs")

        if fibers_type == "fs":
            f0, s0 = fib1, fib2
        elif fibers_type == "fn":
            f0, n0 = fib1, fib2
            s0 = ufl.cross(f0, n0)
        elif fibers_type == "sn":
            s0, n0 = fib1, fib2
            f0 = ufl.cross(s0, n0)
        else:
            raise ValueError("Value for fibers_type has to be fs, fn, or sn!")

        # anisotropic invariants - keep in mind that for growth, self.C is the elastic part of C (hence != to function input variable C)
        I4 = ufl.dot(self.C * f0, f0)
        I6 = ufl.dot(self.C * s0, s0)
        I8 = ufl.dot(self.C * s0, f0)

        # to guarantee initial configuration is stress-free (in case of initially non-orthogonal fibers f0 and s0)
        I8 -= ufl.dot(f0, s0)

        a_0, b_0 = params["a_0"], params["b_0"]
        a_f, b_f = params["a_f"], params["b_f"]
        a_s, b_s = params["a_s"], params["b_s"]
        a_fs, b_fs = params["a_fs"], params["b_fs"]

        fiber_comp_switch = params.get("fiber_comp_switch", "hard")

        # conditional parameters: if fiber_comp_switch is 'hard' (default) or 'soft', fibers are only active in tension; for 'no' also in compression
        if fiber_comp_switch == "hard":
            a_f_c = ufl.conditional(ufl.ge(I4, 1.0), a_f, 0.0)
            a_s_c = ufl.conditional(ufl.ge(I6, 1.0), a_s, 0.0)
        elif fiber_comp_switch == "soft":
            k = params["k_fib"]
            a_f_c = a_f * 1.0 / (1.0 + ufl.exp(-k * (I4 - 1.0)))
            a_s_c = a_s * 1.0 / (1.0 + ufl.exp(-k * (I6 - 1.0)))
        elif fiber_comp_switch == "no":
            a_f_c = a_f
            a_s_c = a_s
        else:
            raise ValueError("Unknown fiber_comp_switch option!")

        # Holzapfel-Ogden (Holzapfel and Ogden 2009) material w/o split applied to invariants I4, I6, I8 (Nolan et al. 2014, Sansour 2008)
        Psi_dev = (
            a_0 / (2.0 * b_0) * (ufl.exp(b_0 * (self.Ic_bar - self.dim)) - 1.0)
            + a_f_c / (2.0 * b_f) * (ufl.exp(b_f * (I4 - 1.0) ** 2.0) - 1.0)
            + a_s_c / (2.0 * b_s) * (ufl.exp(b_s * (I6 - 1.0) ** 2.0) - 1.0)
            + a_fs / (2.0 * b_fs) * (ufl.exp(b_fs * I8**2.0) - 1.0)
        )

        S = 2.0 * ufl.diff(Psi_dev, C)

        return S, Psi_dev

    def guccione_dev(self, params, f0, s0, C):
        n0 = ufl.cross(f0, s0)

        # anisotropic invariants - keep in mind that for growth, self.E is the elastic part of E
        E_ff = ufl.dot(self.E * f0, f0)  # fiber GL strain
        E_ss = ufl.dot(self.E * s0, s0)  # cross-fiber GL strain
        E_nn = ufl.dot(self.E * n0, n0)  # radial GL strain

        E_fs = ufl.dot(self.E * f0, s0)
        E_fn = ufl.dot(self.E * f0, n0)
        E_sn = ufl.dot(self.E * s0, n0)

        c_0 = params["c_0"]
        b_f = params["b_f"]
        b_t = params["b_t"]
        b_fs = params["b_fs"]

        Q = (
            b_f * E_ff**2.0
            + b_t * (E_ss**2.0 + E_nn**2.0 + 2.0 * E_sn**2.0)
            + b_fs * (2.0 * E_fs**2.0 + 2.0 * E_fn**2.0)
        )

        Psi_dev = 0.5 * c_0 * (ufl.exp(Q) - 1.0)

        S = 2.0 * ufl.diff(Psi_dev, C)

        return S, Psi_dev

    def neohooke_compressible(self, params, C):
        # shear modulus and Poisson's ratio
        mu, nu = params["mu"], params["nu"]

        beta = nu / (1.0 - 2.0 * nu)

        # compressible NeoHookean material (Holzapfel eq. 6.148)
        Psi = (mu / 2.0) * (self.Ic - self.dim) + mu / (2.0 * beta) * (self.J ** (-2.0 * beta) - 1.0)

        S = 2.0 * ufl.diff(Psi, C)

        return S, Psi

    def stvenantkirchhoff(self, params, C):
        Emod, nu = params["Emod"], params["nu"]

        mu = Emod / (2.0 * (1.0 + nu))
        lam = nu * Emod / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # classical St.-Venant Kirchhoff model (Holzapfel eq. 6.151)
        Psi = (lam / 2.0) * self.trE**2.0 + mu * self.trE2

        S = 2.0 * ufl.diff(Psi, C)

        return S, Psi

    def stvenantkirchhoff_mod(self, params, C):
        Emod, kappa = params["Emod"], params["kappa"]

        mu = 3.0 * kappa * Emod / (9.0 * kappa - Emod)

        # modified St.-Venant Kirchhoff also suitable for large compressive strains (Holzapfel eq. 6.152)
        Psi = (kappa / 2.0) * ufl.ln(self.J) ** 2.0 + mu * self.trE2

        S = 2.0 * ufl.diff(Psi, C)

        return S, Psi

    def sussmanbathe_vol(self, params, C):
        kappa = params["kappa"]

        Psi_vol = (kappa / 2.0) * (self.J - 1.0) ** 2.0

        S = 2.0 * ufl.diff(Psi_vol, C)

        return S, Psi_vol

    def ogden_vol(self, params, C):
        kappa = params["kappa"]
        beta = params.get("beta", -2.0)

        # classical Ogden volumetric material (Holzapfel eq. 6.137)
        Psi_vol = (kappa / (beta**2.0)) * (beta * ufl.ln(self.J) + self.J ** (-beta) - 1.0)

        S = 2.0 * ufl.diff(Psi_vol, C)

        return S, Psi_vol

    def ogden_mod_vol(self, params, C):
        kappa = params["kappa"]
        beta = params.get("beta", -2.0)

        # a modified variant of the classical Ogden model (spotted somewhere... use with care!)
        Psi_vol = (kappa / (beta**2.0)) * (ufl.ln(self.J) ** (-beta) + (self.J - 1.0) ** (-beta))

        S = 2.0 * ufl.diff(Psi_vol, C)

        return S, Psi_vol

    # simple Green-Lagrange strain rate-dependent material
    def visco_green(self, params, Cdot):
        eta = params["eta"]

        # pseudo potential 0.5 * eta * dE/dt : dE/dt
        Psi_pseudo = 0.5 * eta * ufl.tr(self.Edot * self.Edot)

        S = 2.0 * ufl.diff(Psi_pseudo, Cdot)

        return S, None

    def active_fiber(self, tau, f0):
        S = tau * ufl.outer(f0, f0)

        return S, None

    def active_crossfiber(self, tau, f0):
        S = tau * (self.I - ufl.outer(f0, f0))

        return S, None

    def active_iso(self, tau):
        S = tau * self.I

        return S, None


# volumetric growth class: kinematics
class growth:
    def __init__(self, theta, I):
        self.theta = theta
        self.I = I

    def isotropic(self):
        F_g = self.theta * self.I

        return F_g

    def fiber(self, f0):
        F_g = self.I + (self.theta - 1.0) * ufl.outer(f0, f0)

        return F_g

    def crossfiber(self, f0):
        F_g = self.theta * self.I + (1.0 - self.theta) * ufl.outer(f0, f0)

        return F_g

    def radial(self, f0, s0):
        r0 = ufl.cross(f0, s0)
        F_g = self.I + (self.theta - 1.0) * ufl.outer(r0, r0)

        return F_g


# volumetric growth class: growth function
class growthfunction(growth):
    # add possible variations / different growth functions here...

    def grfnc1(self, trigger, thres, params):
        thetamax, thetamin = params["thetamax"], params["thetamin"]
        tau_gr, tau_gr_rev = params["tau_gr"], params["tau_gr_rev"]
        gamma_gr, gamma_gr_rev = params["gamma_gr"], params["gamma_gr_rev"]

        k_plus = (1.0 / tau_gr) * ((thetamax - self.theta) / (thetamax - thetamin)) ** (gamma_gr)
        k_minus = (1.0 / tau_gr_rev) * ((self.theta - thetamin) / (thetamax - thetamin)) ** (gamma_gr_rev)

        k = ufl.conditional(ufl.ge(trigger, thres), k_plus, k_minus)

        return k


# expression for time-dependent active stress activation function
class activestress_activation:
    def __init__(self, params, act_curve, x_ref=None):
        self.params = params

        self.sigma0 = self.params["sigma0"]
        self.alpha_max = self.params["alpha_max"]
        self.alpha_min = self.params["alpha_min"]

        self.act_curve = act_curve
        self.act_curve_old = None

        if "frankstarling" in self.params.keys():
            self.frankstarling = self.params["frankstarling"]
        else:
            self.frankstarling = False

        # reference coordinates - needed e.g. for spatial scaling functions of activation
        self.x_ref = x_ref

        self.activation_weight = self.params.get("activation_weight", None)

        if self.activation_weight is not None:
            self.act_weight_type = self.activation_weight["type"]
            self.act_weight_radius = self.activation_weight["radius"]
            self.act_weight_center = ufl.as_vector(self.activation_weight["center"])
            self.act_weight_max = self.activation_weight["w_max"]
            self.act_weight_min = self.activation_weight["w_min"]

    def distance_to_point(self, p):
        diff_vec = p - self.x_ref
        return ufl.sqrt(ufl.dot(diff_vec, diff_vec))

    def act_weight(self):
        if self.act_weight_type == "radial_decay":
            d = self.distance_to_point(self.act_weight_center)
            # smoothly decaying weight within a defined radius from a maximum to a minimum value
            return ufl.conditional(
                ufl.lt(d, self.act_weight_radius),
                0.5 * (self.act_weight_max - self.act_weight_min) * (1.0 - ufl.cos(np.pi * d / self.act_weight_radius))
                + self.act_weight_min,
                self.act_weight_max,
            )
        else:
            raise ValueError("Unknown act_weight_type!")

    # activation function for active stress
    def ua(self, actcurve):
        # Diss Hirschvogel eq. 2.100
        return actcurve * self.alpha_max + (1.0 - actcurve) * self.alpha_min

    # Frank-Staring function
    def g(self, lam):
        amp_min = self.params["amp_min"]
        amp_max = self.params["amp_max"]

        lam_threslo = self.params["lam_threslo"]
        lam_maxlo = self.params["lam_maxlo"]

        lam_threshi = self.params["lam_threshi"]
        lam_maxhi = self.params["lam_maxhi"]

        # Diss Hirschvogel eq. 2.107
        # TeX: g(\lambda_{\mathrm{myo}}) = \begin{cases} a_{\mathrm{min}}, & \lambda_{\mathrm{myo}} \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}}, \\ a_{\mathrm{min}}+\frac{1}{2}\left(a_{\mathrm{max}}-a_{\mathrm{min}}\right)\left(1-\cos \frac{\pi(\lambda_{\mathrm{myo}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}})}{\hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,lo}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}}}\right), &  \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,lo}} \leq \lambda_{\mathrm{myo}}  \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,lo}}, \\ a_{\mathrm{max}}, &  \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,lo}} \leq \lambda_{\mathrm{myo}} \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,hi}}, \\ a_{\mathrm{min}}+\frac{1}{2}\left(a_{\mathrm{max}}-a_{\mathrm{min}}\right)\left(1-\cos \frac{\pi(\lambda_{\mathrm{myo}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}})}{\hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}}-\hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,hi}}}\right), & \hat{\lambda}_{\mathrm{myo}}^{\mathrm{thres,hi}} \leq \lambda_{\mathrm{myo}} \leq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}}, \\ a_{\mathrm{min}}, & \lambda_{\mathrm{myo}} \geq \hat{\lambda}_{\mathrm{myo}}^{\mathrm{max,hi}} \end{cases}

        return ufl.conditional(
            ufl.le(lam, lam_threslo),
            amp_min,
            ufl.conditional(
                ufl.And(ufl.ge(lam, lam_threslo), ufl.le(lam, lam_maxlo)),
                amp_min
                + 0.5 * (amp_max - amp_min) * (1.0 - ufl.cos(ufl.pi * (lam - lam_threslo) / (lam_maxlo - lam_threslo))),
                ufl.conditional(
                    ufl.And(ufl.ge(lam, lam_maxlo), ufl.le(lam, lam_threshi)),
                    amp_max,
                    ufl.conditional(
                        ufl.And(ufl.ge(lam, lam_threshi), ufl.le(lam, lam_maxhi)),
                        amp_min
                        + 0.5
                        * (amp_max - amp_min)
                        * (1.0 - ufl.cos(ufl.pi * (lam - lam_maxhi) / (lam_maxhi - lam_threshi))),
                        ufl.conditional(ufl.ge(lam, lam_maxhi), amp_min, ufl.as_ufl(0)),
                    ),
                ),
            ),
        )

    # Frank Starling amplification factor (Diss Hirschvogel eq. 2.106, 3.29)
    # \dot{a}(\lambda_{\mathrm{myo}}) = \dot{g}(\lambda_{\mathrm{myo}}) \,\mathbb{I}_{|u|_{-}>0}
    def amp(self, lam, amp_old):
        uabs_minus = ufl.max_value(-ufl.min_value(self.ua(self.act_curve_old), 0), 0)

        return ufl.conditional(ufl.gt(uabs_minus, 0.0), self.g(lam), amp_old)

    # Backward-Euler integration of active stress
    def tau_act(self, tau_a_old, dt, lam=None, amp_old=None):
        uabs = abs(self.ua(self.act_curve))
        uabs_plus = ufl.max_value(self.ua(self.act_curve), 0)

        # amplification/weighting factor
        amp = 1.0

        if self.frankstarling:
            # amplification due to Frank-Starling mechanism
            amp *= self.amp(lam, amp_old)

        if self.activation_weight is not None:
            # weighting factor
            amp *= self.act_weight()

        return (tau_a_old + amp * self.sigma0 * uabs_plus * dt) / (1.0 + uabs * dt)
