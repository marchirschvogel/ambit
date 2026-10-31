#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw_potential:
    def __init__(self, phi, a=0.0, b=1.0):
        self.phi = phi
        self.a = a
        self.b = b

    def mat_ginzburg_landau(self, params):
        D = params["D"]

        # generalized Ginzburg-Landau double-well potential with minima at a and b
        psi = D * (self.a-self.phi)**2.0 * (self.b-self.phi)**2.0

        return ufl.diff(psi,self.phi), psi

    def mat_flory_huggins_regularized(self, params):
        theta, theta0 = params["theta"], params["theta0"]
        delta = params["delta"]
        assert(self.a==-1.0)
        assert(self.b==1.0)

        assert(theta0 - theta >= 0.0)  # requirement for double-well potential
        assert(theta0/theta <= 1.0/(delta*(2.0-delta)))  # convexity condition

        def psi_1(s):
            return (theta0/2.0) * (1.0 - s**2.0)

        def psi_2(s):
            return (theta/2.0) * ((1.0 - s)*ufl.ln((1.0 - s)/2.0) + (1.0 + s)*ufl.ln((1.0 + s)/2.0))

        def dpsi_2(s):
            return (theta / 2.0) * ufl.ln((1.0 + s) / (1.0 - s))
        def d2psi_2(s):
            return theta / (1.0 - s**2.0)

        phi_bar = ufl.max_value(
            -1.0 + delta,
            ufl.min_value(self.phi, 1.0 - delta)
        )

        psi_2_reg = (
            psi_2(phi_bar)
            + dpsi_2(phi_bar) * (self.phi - phi_bar)
            + 0.5 * d2psi_2(phi_bar) * (self.phi - phi_bar)**2.0
        )

        psi = psi_1(self.phi) + psi_2_reg

        return ufl.diff(psi,self.phi), psi


class materiallaw_flux:
    def __init__(self, mu, phi, a=0.0, b=1.0):
        self.mu = mu
        self.phi = phi
        self.a = a
        self.b = b

    def mat_diffusive_flux(self, params, mob, p=None, F=None, alpha=None):
        # fluid pressure proportional term (needed for consistency of mass-averged velocity formulation!)
        if p is not None:
            beta = params.get("beta", 1.0)  # experimental: to tune pressure gradient-driven diffusive flux...
            ap = beta * alpha * p
        else:
            ap = ufl.as_ufl(0)

        if F is not None:
            return -mob*ufl.inv(F).T*ufl.grad(self.mu + ap)
        else:
            return -mob*ufl.grad(self.mu + ap)
