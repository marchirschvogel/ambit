#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from petsc4py import PETSc

from .phasefield_material import materiallaw, materiallaw_flux

"""
Phase field/Cahn-Hilliard constitutive class
"""


class constitutive:
    def __init__(self, materials, phi_range=[0.0, 1.0]):
        self.materials = materials

        # list entries of mats which do not return a flux/driving force
        self.mat_void = ["id", "source"]

        self.phi_range = phi_range

    # diffusive flux
    def diffusive_flux(self, mu_, phi_, p=None, F=None, alpha=None):
        dim = len(ufl.grad(mu_)) # dimension of space

        Jflux = ufl.constantvalue.zero(dim)

        mat_flux = materiallaw_flux(mu_, phi_, a=self.phi_range[0], b=self.phi_range[1])

        for key, value in self.materials.items():
            if key not in self.mat_void:
                if key == "mat_cahnhilliard":
                    Jflux += mat_flux.mat_cahnhilliard_flux(value, p=p, F=F, alpha=alpha)
                else:
                    raise NameError("Unknown Cahn-Hilliard material law '%s'!" % (key))

        return Jflux

    # driving force
    def driv_force(self, phi_, returnquantity="deriv"):
        phi_ = ufl.variable(phi_)

        dpsi_dphi, psi = ufl.as_ufl(0), ufl.as_ufl(0)

        mat = materiallaw(phi_, a=self.phi_range[0], b=self.phi_range[1])

        for key, value in self.materials.items():
            if key not in self.mat_void:
                if key == "mat_cahnhilliard":
                    dpsi_dphi_, psi_ = mat.mat_cahnhilliard(value)
                    dpsi_dphi += dpsi_dphi_
                    psi += psi_
                else:
                    raise NameError("Unknown Cahn-Hilliard material law '%s'!" % (key))

        if returnquantity=="deriv":
            return dpsi_dphi
        elif returnquantity=="doublewell":
            return psi
        else:
            raise ValueError("Unknown returnquantity!")
