#!/usr/bin/env python3

# Copyright (c) 2019-2022, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dolfinx import fem
import ufl

def project(v, V, dx_, bcs=[], nm=None):

    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)

    a, L = ufl.as_ufl(0), ufl.as_ufl(0)
    zerofnc = fem.Function(V)
    
    for n in range(len(dx_)):
        
        # check if we have passed in a list of functions or a function
        if isinstance(v, list):
            fnc = v[n]
        else:
            fnc = v
        
        if not isinstance(fnc, ufl.constantvalue.Zero):
            a += ufl.inner(w, Pv) * dx_[n]
            L += ufl.inner(w, fnc) * dx_[n]
        else:
            a += ufl.inner(w, Pv) * dx_[n]
            L += ufl.inner(w, zerofnc) * dx_[n]

    # solve linear system for projection
    function = fem.Function(V, name=nm)
    
    lp = fem.LinearProblem(a, L, bcs=bcs, u=function)
    lp.solve()
    
    return function
