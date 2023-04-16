#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ale_material import materiallaw

# ALE kinematics and constitutive class

class constitutive:

    def __init__(self, kin, materials, msh):

        self.kin = kin

        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])


    def stress(self, u_):

        F_ = ufl.variable(self.kin.F(u_))

        dim = len(u_)

        stress = ufl.constantvalue.zero((dim,dim))

        mat = materiallaw(u_,F_)

        m = 0
        for matlaw in self.matmodels:

            # extract associated material parameters
            matparams_m = self.matparams[m]

            if matlaw == 'linelast':

                stress += mat.linelast(matparams_m)

            elif matlaw == 'neohooke':

                stress += mat.neohooke(matparams_m)

            else:

                raise NameError('Unknown ALE material law!')

            m += 1

        return stress


class kinematics:

    def __init__(self, dim):

        self.dim = dim

        # identity tensor
        self.I = ufl.Identity(self.dim)


    # ALE deformation gradient
    def F(self, u_):
        return self.I + ufl.grad(u_)
