#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
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

        return ufl.diff(psi,self.phi)


class materiallaw_flux:
    def __init__(self, mu, phi, a=0.0, b=1.0):
        self.mu = mu
        self.phi = phi
        self.a = a
        self.b = b

    def mat_cahnhilliard_flux(self, params, p=None, F=None):
        mobility = params.get("mobility", "constant")
        M0 = params["M0"]
        if mobility=="constant":
            M = M0
        elif mobility=="degenerate":
            eps = params.get("epsilon", 0.0)
            exp = params.get("exponent", 1.0)
            # degenerate mobility, vanishing in the single-fluid regime (phi=a or phi=b)
            M = M0 * abs((self.a-self.phi)**exp * (self.b-self.phi)**exp + eps)
        else:
            raise ValueError("Unknown mobility type! Choose 'constant' or 'degenerate'.")

        # fluid pressure proportional term (if alpha given)
        alpha = params.get("alpha", None)
        if alpha is not None:
            ap = alpha * p
        else:
            ap = ufl.as_ufl(0)

        if F is not None:
            return -M*ufl.inv(F).T*ufl.grad(self.mu + ap)
        else:
            return -M*ufl.grad(self.mu + ap)
