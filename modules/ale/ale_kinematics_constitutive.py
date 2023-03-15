#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ale_material import materiallaw

# ALE kinematics and constitutive class

class constitutive:
    
    def __init__(self, materials):

        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])
        
        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])


    def stress(self, u_):
        
        dim = len(u_)

        s_grad, s_div, s_ident = ufl.constantvalue.zero((dim,dim)), 0, ufl.constantvalue.zero(dim)
            
        mat = materiallaw(u_)
        
        m = 0
        for matlaw in self.matmodels:
            
            # extract associated material parameters
            matparams_m = self.matparams[m]
        
            if matlaw == 'helmholtz':
                
                sg, sd, si = mat.helmholtz(matparams_m)

                s_grad += sg
                s_div += sd
                s_ident += si
        
            elif matlaw == 'ale_element_dependent_stiffness': # not quite there yet... we need to add the element metrics!
                
                sg, sd, si = mat.ale_element_dependent_stiffness(matparams_m)

                s_grad += sg
                s_div += sd
                s_ident += si

            else:
                
                raise NameError('Unknown ALE material law!')
            
            m += 1

        return s_grad, s_div, s_ident
