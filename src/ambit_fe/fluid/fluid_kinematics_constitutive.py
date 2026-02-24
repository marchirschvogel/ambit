#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from petsc4py import PETSc

from .fluid_material import materiallaw

"""
Fluid kinematics and constitutive class
"""


class constitutive:
    def __init__(self, kin, materials):
        self.kin = kin

        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])

        # list entries of mats which do not return a stress
        self.mat_nostress = ["inertia", "id"]

        # identity tensor
        self.I = ufl.Identity(self.kin.dim)

    # Cauchy stress core routine: most general form
    """ TeX:
    \boldsymbol{\sigma} = -p \boldsymbol{I} + 2\eta\,\frac{1}{2}(\nabla\boldsymbol{v} + \left(\nabla\boldsymbol{v})^{\mathrm{T}}\right) + \left(\zeta-\frac{2}{d}\eta\right)(\nabla\cdot\boldsymbol{v})\boldsymbol{I}
    """
    def sigma(self, v_, p_, F=None, chi=None):
        shearrate_ = self.kin.shearrate(v_, F=F)
        volstrainrate_ = self.kin.volstrainrate(v_, F=F)

        stress = ufl.constantvalue.zero((self.kin.dim, self.kin.dim))

        mat = materiallaw(shearrate_, volstrainrate_, self.kin.use_gen_strainrate, self.I)

        for m, matlaw in enumerate(self.matmodels):
            if matlaw not in self.mat_nostress:
                # extract associated material parameters
                matparams_m = self.matparams[m]

                if matlaw == "newtonian":
                    stress += mat.newtonian(matparams_m, chi=chi)
                else:
                    raise NameError("Unknown fluid material law!")

        # TeX: \sigma_{\mathrm{vol}} = -p\boldsymbol{I}
        stress += -p_ * self.I

        return stress


class kinematics:
    def __init__(self, dim, use_gen_strainrate, uf_pre=None):
        self.dim = dim
        self.use_gen_strainrate = use_gen_strainrate

        # prestress displacement
        self.uf_pre = uf_pre

        # identity tensor
        self.I = ufl.Identity(self.dim)

    # shear rate: symmetric part of velocity gradient
    def shearrate(self, v_, F=None):
        if F is not None:
            return 0.5 * (ufl.grad(v_) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(v_).T)
        else:
            return 0.5 * (ufl.grad(v_) + ufl.grad(v_).T)

    # volumetric strain rare
    def volstrainrate(self, v_, F=None):
        if F is not None:
            return ufl.inner(ufl.grad(v_),ufl.inv(F).T) * self.I
        else:
            return ufl.div(v_) * self.I

    # fluid deformation gradient (relevant on boundary for FrSI): F = I + duf/dx0
    def F(self, uf_):
        if self.uf_pre is not None:
            return self.I + ufl.grad(uf_ + self.uf_pre)  # Schein and Gee 2021, equivalent to Gee et al. 2010
        else:
            return self.I + ufl.grad(uf_)

    # rate of deformation gradient: dF/dt = dv/dx0
    def Fdot(self, v_):
        if not isinstance(v_, ufl.constantvalue.Zero):
            return ufl.grad(v_)
        else:
            return ufl.constantvalue.zero((self.dim, self.dim))

    # prestressing update for FrSI (MULF - Modified Updated Lagrangian Formulation, cf. Gee et al. 2010,
    # displacement formulation according to Schein and Gee 2021)
    def prestress_update(self, dt, v_vec):
        self.uf_pre.x.petsc_vec.axpy(dt, v_vec)
        self.uf_pre.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
