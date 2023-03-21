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

        # some mesh metrics
        # cell diameter
        self.hd0 = ufl.CellDiameter(msh)
        # cell circumradius
        self.ro0 = ufl.Circumradius(msh)
        # min and max cell edge lengths
        self.emin0 = ufl.MinCellEdgeLength(msh)
        self.emax0 = ufl.MaxCellEdgeLength(msh)
        # jacobian determinant
        self.detj0 = ufl.JacobianDeterminant(msh)


    def stress(self, w_):
        
        F_ = ufl.variable(self.kin.F(w_))
        
        dim = len(w_)

        s_grad, s_div, s_ident = ufl.constantvalue.zero((dim,dim)), 0, ufl.constantvalue.zero(dim)
            
        mat = materiallaw(w_,F_)
        
        m = 0
        for matlaw in self.matmodels:
            
            # extract associated material parameters
            matparams_m = self.matparams[m]

            if matlaw == 'neohooke':
                
                sg, sd, si = mat.neohooke(matparams_m)

                s_grad += sg
                s_div += sd
                s_ident += si
        
            elif matlaw == 'helmholtz':
                
                sg, sd, si = mat.helmholtz(matparams_m)

                s_grad += sg
                s_div += sd
                s_ident += si

            elif matlaw == 'linelast':
                
                sg, sd, si = mat.linelast(matparams_m)

                s_grad += sg
                s_div += sd
                s_ident += si
        
            elif matlaw == 'element_dependent_stiffness':
                
                #metric = (ufl.min_value(10.,self.emax0/self.emin0))**4. # doesn't seem to work for quadratic cells...
                metric = 1
                
                sg, sd, si = mat.element_dependent_stiffness(matparams_m, metric)

                s_grad += sg
                s_div += sd
                s_ident += si

            else:
                
                raise NameError('Unknown ALE material law!')
            
            m += 1

        return s_grad, s_div, s_ident


class kinematics:
    
    def __init__(self, dim):
        
        self.dim = dim

        # identity tensor
        self.I = ufl.Identity(self.dim)


    # ALE deformation gradient
    def F(self, w_):
        return self.I + ufl.grad(w_)
