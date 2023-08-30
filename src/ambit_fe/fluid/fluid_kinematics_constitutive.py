#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from petsc4py import PETSc

from .fluid_material import materiallaw


# fluid kinematics and constitutive class

class constitutive:

    def __init__(self, kin, materials):

        self.kin = kin

        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])

        # identity tensor
        self.I = ufl.Identity(self.kin.dim)


    # Cauchy stress core routine
    def sigma(self, v_, p_, Fale=None):

        gamma_ = self.kin.gamma(v_,Fale=Fale)

        stress = ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

        mat = materiallaw(gamma_,self.I)

        m = 0
        for matlaw in self.matmodels:

            # extract associated material parameters
            matparams_m = self.matparams[m]

            if matlaw == 'newtonian':

                stress += mat.newtonian(matparams_m)

            elif matlaw == 'inertia':
                # density is added to kinetic virtual power
                pass

            else:

                raise NameError('Unknown fluid material law!')

            m += 1

        # TeX: \sigma_{\mathrm{vol}} = -p\boldsymbol{I}
        stress += -p_ * self.I

        return stress



class kinematics:

    def __init__(self, dim, uf_pre=None):

        self.dim = dim

        # prestress displacement
        self.uf_pre = uf_pre

        # identity tensor
        self.I = ufl.Identity(self.dim)


    # velocity gradient: gamma = 0.5(dv/dx + (dv/dx)^T)
    def gamma(self, v_, Fale=None):

        if Fale is not None:
            return 0.5*(ufl.grad(v_)*ufl.inv(Fale) + ufl.inv(Fale).T*ufl.grad(v_).T)
        else:
            return 0.5*(ufl.grad(v_) + ufl.grad(v_).T)


    # fluid deformation gradient (relevant on boundary for FrSI): F = I + duf/dx0
    def F(self, uf_):

        if self.uf_pre is not None:
            return self.I + ufl.grad(uf_+self.uf_pre) # Schein and Gee 2021, equivalent to Gee et al. 2010
        else:
            return self.I + ufl.grad(uf_)


    # rate of deformation gradient: dF/dt = dv/dx0
    def Fdot(self, v_):

        if not isinstance(v_, ufl.constantvalue.Zero):
            return ufl.grad(v_)
        else:
            return ufl.constantvalue.zero((self.dim,self.dim))


    # prestressing update for FrSI (MULF - Modified Updated Lagrangian Formulation, cf. Gee et al. 2010,
    # displacement formulation according to Schein and Gee 2021)
    def prestress_update(self, dt, v_vec):

        self.uf_pre.vector.axpy(dt, v_vec)
        self.uf_pre.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
