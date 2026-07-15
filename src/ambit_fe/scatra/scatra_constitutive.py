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
        self.materials = materials

        # list entries of mats which do not return a flux/driving force
        self.mat_void = ["id", "source"]

    # diffusive flux
    def diffusive_flux(self, c_, c_coup=None, F=None):
        dim = len(ufl.grad(c_)) # dimension of space

        difflux = ufl.constantvalue.zero(dim)

        mat_flux = materiallaw(c_, c_coup)

        for key, value in self.materials.items():
            if key not in self.mat_void:
                if key == "mat_diff":
                    difflux += mat_flux.mat_diff(value, F=F)
                elif key == "mat_diff_coup":
                    difflux += mat_flux.mat_diff_coup(value, F=F)
                else:
                    raise NameError("Unknown scalar transport material law '%s'!" % (key))

        return difflux
