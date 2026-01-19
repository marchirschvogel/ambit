#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
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
    def __init__(self, materials):
        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])

    # diffusive flux
    def diffusive_flux(self, mu_, phi_, F=None):
        dim = len(ufl.grad(mu_)) # dimension of space

        Jflux = ufl.constantvalue.zero(dim)

        mat_flux = materiallaw_flux(mu_, phi_)

        m = 0
        for matlaw in self.matmodels:
            # extract associated material parameters
            matparams_m = self.matparams[m]

            if matlaw == "mat_cahnhilliard":
                Jflux += mat_flux.mat_cahnhilliard_flux(matparams_m, F=F)
            elif matlaw == "params_cahnhilliard": # other params not in constitutive relation
                pass
            else:
                raise NameError("Unknown Cahn-Hilliard material law!")

            m += 1

        return Jflux

    # driving force
    def driv_force(self, phi_):
        phi_ = ufl.variable(phi_)

        dpsi_dphi = ufl.as_ufl(0)

        mat = materiallaw(phi_)

        m = 0
        for matlaw in self.matmodels:
            # extract associated material parameters
            matparams_m = self.matparams[m]

            if matlaw == "mat_cahnhilliard":
                dpsi_dphi += mat.mat_cahnhilliard(matparams_m)
            elif matlaw == "params_cahnhilliard": # other params not in constitutive relation
                pass
            else:
                raise NameError("Unknown Cahn-Hilliard material law!")

            m += 1

        return dpsi_dphi
