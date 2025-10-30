#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from petsc4py import PETSc

from .cahnhilliard_material import materiallaw

"""
Cahn-Hilliard constitutive class
"""


class constitutive:
    def __init__(self, materials):
        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])

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
