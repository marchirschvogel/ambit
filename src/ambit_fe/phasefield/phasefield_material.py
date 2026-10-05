#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw:
    def __init__(self, phi, a=0.0, b=1.0):
        self.phi = phi
        self.a = a
        self.b = b

    def mat_cahnhilliard(self, params):
        D = params["D"]

        # generalized double-well potential with minima at a and b
        psi = D * (self.a-self.phi)**2.0 * (self.b-self.phi)**2.0

        return ufl.diff(psi,self.phi), psi


class materiallaw_flux:
    def __init__(self, mu, phi, a=0.0, b=1.0):
        self.mu = mu
        self.phi = phi
        self.a = a
        self.b = b

    def mat_cahnhilliard_flux(self, params, mob, p=None, F=None, alpha=None):
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
