#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from dolfinx import fem
import ufl
from petsc4py import PETSc


def project(v, V, dx_, bcs=[], nm=None, comm=None):

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
            a += ufl.inner(Pv, w) * dx_[n]
            L += ufl.inner(fnc, w) * dx_[n]
        else:
            a += ufl.inner(Pv, w) * dx_[n]
            L += ufl.inner(zerofnc, w) * dx_[n]

    # solve linear system for projection
    function = fem.Function(V, name=nm)

    a_form, L_form = fem.form(a), fem.form(L)

    # assemble linear system
    A = fem.petsc.assemble_matrix(a_form, bcs)
    A.assemble()

    b = fem.petsc.assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(comm)

    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    ksp.setOperators(A)
    ksp.solve(b, function.vector)

    b.destroy(), A.destroy()
    ksp.destroy()

    function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return function
