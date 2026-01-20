#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw:
    def __init__(self, phi):
        self.phi = phi

    def mat_cahnhilliard(self, params):
        D = params["D"]
        psi = D * self.phi**2.0 * (1.0 - self.phi)**2.0

        return ufl.diff(psi,self.phi)


class materiallaw_flux:
    def __init__(self, mu, phi):
        self.mu = mu
        self.phi = phi

    def mat_cahnhilliard_flux(self, params, p=None, F=None):
        mobility = params.get("mobility", "constant")
        M0 = params["M0"]
        if mobility=="constant":
            M = M0
        elif mobility=="degenerate":
            eps = params.get("epsilon", 0.0)
            exp = params.get("exponent", 1.0)
            # degenerate mobility, vanishing in the single-fluid regime (phi=0 or phi=1)
            M = M0 * (self.phi**exp * (1.0 - self.phi)**exp + eps)
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
