#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from petsc4py import PETSc

from .scatra_material import materiallaw

"""
Scalar transport constitutive class
"""


class constitutive:
    def __init__(self, materials):
        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])

        # list entries of mats which do not return a flux/driving force
        self.mat_void = ["id", "source"]

    # diffusive flux
    def diffusive_flux(self, c_, cdot_, F=None):
        dim = len(ufl.grad(c_)) # dimension of space

        difflux = ufl.constantvalue.zero(dim)

        mat_flux = materiallaw(c_, cdot_)

        for m, matlaw in enumerate(self.matmodels):
            if matlaw not in self.mat_void:
                # extract associated material parameters
                matparams_m = self.matparams[m]

                if matlaw == "mat_diff":
                    difflux += mat_flux.mat_diff(matparams_m, F=F)
                else:
                    raise NameError("Unknown scalar transport material law!")

        return difflux
