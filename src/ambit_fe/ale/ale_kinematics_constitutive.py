#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from .ale_material import materiallaw

"""
ALE kinematics and constitutive class
"""


class constitutive:
    def __init__(self, kin, materials):
        self.kin = kin
        self.materials = materials

        # list entries of mats which do not return a stress
        self.mat_void = ["inertia", "id"]

    def stress(self, d_, w_):
        F_ = ufl.variable(self.kin.F(d_))

        dim = len(d_)

        stress = ufl.constantvalue.zero((dim, dim))

        mat = materiallaw(d_, w_, F_, self.kin.elem_metrics)

        for key, value in self.materials.items():
            if key not in self.mat_void:
                if key == "diffusion":
                    stress += mat.diffusion(value)

                elif key == "diffusion_rate":
                    stress += mat.diffusion_rate(value)

                elif key == "diffusion_sym":
                    stress += mat.diffusion_sym(value)

                elif key == "diffusion_rate_sym":
                    stress += mat.diffusion_rate_sym(value)

                elif key == "linelast":
                    stress += mat.linelast(value)

                elif key == "neohooke":
                    stress += mat.neohooke(value)

                elif key == "exponential":
                    stress += mat.exponential(value)

                else:
                    raise NameError("Unknown ALE material law '%s'!" % (key))

        return stress


class kinematics:
    def __init__(self, dim, elem_metrics=None):
        self.dim = dim

        # identity tensor
        self.I = ufl.Identity(self.dim)

        self.elem_metrics = elem_metrics

    # ALE deformation gradient
    def F(self, d_):
        return self.I + ufl.grad(d_)
