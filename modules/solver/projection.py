#!/usr/bin/env python3

from dolfinx import Function
from dolfinx.fem import LinearProblem
from ufl import TrialFunction, TestFunction, inner, as_ufl, constantvalue
from petsc4py import PETSc

def project(v, V, dx_, bcs=[], nm=None):

    w = TestFunction(V)
    Pv = TrialFunction(V)

    a, L = as_ufl(0), as_ufl(0)
    zerofnc = Function(V)
    
    for n in range(len(dx_)):
        
        # check if we have passed in a list of functions or a function
        if isinstance(v, list):
            fnc = v[n]
        else:
            fnc = v
        
        if not isinstance(fnc, constantvalue.Zero):
            a += inner(w, Pv) * dx_[n]
            L += inner(w, fnc) * dx_[n]
        else:
            a += inner(w, Pv) * dx_[n]
            L += inner(w, zerofnc) * dx_[n]

    # solve linear system for projection
    function = Function(V, name=nm)
    
    lp = LinearProblem(a, L, bcs=bcs, u=function)
    lp.solve()
    
    return function
