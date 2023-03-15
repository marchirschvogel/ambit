#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

class materiallaw:
    
    def __init__(self, u):
        self.u = u
        self.dim = len(self.u)
    

    def helmholtz(self, params):
        
        k = params['k']

        # s_grad (tensor), s_div (scalar), s_ident (vector)
        return (k**2.)*ufl.grad(self.u), 0, self.u
    
    
    def ale_element_dependent_stiffness(self, params):
        
        kappa, alpha = params['kappa'], params['alpha']

        # s_grad (tensor), s_div (scalar), s_ident (vector)
        return kappa*ufl.sym(ufl.grad(self.u)), kappa*alpha*ufl.div(self.u), ufl.constantvalue.zero(self.dim)
