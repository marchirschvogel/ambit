#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


# returns the Cauchy stress sigma for different material laws


class materiallaw:
    def __init__(self, shearrate, volstrainrate, use_gen_strainrate, I):
        self.shearrate = shearrate
        self.volstrainrate = volstrainrate

        self.use_gen_strainrate = use_gen_strainrate

        self.dim = I.ufl_shape[0]

    def newtonian(self, params, phi=None):
        if phi is None:
            eta = params["eta"]  # dynamic viscosity
        else:
            eta1, eta2 = params["eta1"], params["eta2"]
            eta = phi * eta1 + (1.0 - phi) * eta2

        # classical Newtonian fluid
        sigma = 2.0 * eta * self.shearrate

        # in case we should add volumetric strain rate (div v not zero!)
        if self.use_gen_strainrate:
            if phi is None:
                zeta = params.get("zeta", 0.0)  # bulk viscosity (default is zero: Stokes' hypothesis)
            else:
                zeta1, zeta2 = params.get("zeta1", 0.0), params.get("zeta2", 0.0)
                zeta = phi * zeta1 + (1.0 - phi) * zeta2

            # volumetric stress contribution
            lambd = zeta - (2.0/self.dim) * eta
            sigma += lambd * self.volstrainrate

        return sigma
