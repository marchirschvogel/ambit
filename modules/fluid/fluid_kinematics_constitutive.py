#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ufl import tr, det, dot, grad, inv, dev, inner, Identity, variable, ln, sqrt, exp, diff, conditional, ge, outer, cross, as_tensor, indices

from fluid_material import materiallaw

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
        self.I = Identity(3)


    # Cauchy stress core routine
    def sigma(self, v_, p_):
        
        gamma_ = self.kin.gamma(v_)

        stress = 0
            
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

        # TeX: S_{\mathrm{vol}} = -p\boldsymbol{I}
        stress += -p_ * self.I

        return stress





class kinematics:

    # velocity gradient: gamma = 0.5(dv/dx + (dv/dx)^T)
    def gamma(self, v_):
        return 0.5*(grad(v_) + grad(v_).T)
