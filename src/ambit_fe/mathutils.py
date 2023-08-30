#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc


def spectral_decomposition_3x3(A):
    """Eigenvalues and eigenprojectors of a 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{3} lambda_a * E_a
    with eigenvalues lambda_a and their associated eigenprojectors E_a = n_a \otimes n_a
    ordered by magnitude.
    The eigenprojectors of eigenvalues with multiplicity n are returned as 1/n-fold projector.
    Note: Tensor A must not have complex eigenvalues!
    cf. https://github.com/michalhabera/dolfiny/blob/master/dolfiny/invariants.py
    """
    if ufl.shape(A) != (3,3):
        raise RuntimeError("Tensor A of shape {ufl.shape(A)} != (3,3) is not supported!")

    eps = 1.0e-10

    A = ufl.variable(A)

    # determine eigenvalues lambda1, lambda2, lambda3
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
    q = ufl.tr(A)/3.
    B = A - q * ufl.Identity(3)
    # observe: det(lambda*I - A) = 0  with shift  lambda = q + omega --> det(omega*I - B) = 0 = omega**3 - j * omega - b
    j = ufl.tr(B*B)/2.  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = ufl.tr(B*B*B)/3.  # == I3(B) for trace-free B
    # solve: 0 = omega**3 - j * omega - b  by substitution  omega = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / (p**3)
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / (p**3)  | --> p := sqrt(j * 4/3)
    #        0 = cos(3 * phi) - 4 * b / (p**3)
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = (2./ufl.sqrt(3.)) * ufl.sqrt(j + eps**2.)
    r = 4.*b/(p**3.)
    r = ufl.max_value(ufl.min_value(r, 1.-eps), -1.+eps)
    phi = ufl.acos(r)/3.
    # sorted eigenvalues: lambda1 >= lambda2 >= lambda3
    lambda1 = q + p * ufl.cos(phi)                     # high
    lambda2 = q + p * ufl.cos(phi + (4./3.) * ufl.pi)  # middle
    lambda3 = q + p * ufl.cos(phi + (2./3.) * ufl.pi)  # low

    # determine eigenprojectors E0, E1, E2 - Holzapfel eq. (2.123)
    E1 = ufl.diff(lambda1, A).T
    E2 = ufl.diff(lambda2, A).T
    E3 = ufl.diff(lambda3, A).T

    # eigenvectors
    evc1 = ufl.diag_vector(E1)
    evc2 = ufl.diag_vector(E2)
    evc3 = ufl.diag_vector(E3)
    # normalize
    evc1 /= ufl.sqrt(ufl.dot(evc1,evc1))
    evc2 /= ufl.sqrt(ufl.dot(evc2,evc2))
    evc3 /= ufl.sqrt(ufl.dot(evc3,evc3))

    return [lambda1, lambda2, lambda3], [evc1, evc2, evc3], [E1, E2, E3]


def quad_interpolation(expr, V, msh, quadrature_points, func):
    '''
    See https://github.com/FEniCS/dolfinx/issues/2243
    '''
    e_expr = fem.Expression(expr, quadrature_points)
    map_c = msh.topology.index_map(msh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    e_eval = e_expr.eval(cells=cells, mesh=msh)

    with func.vector.localForm() as func_local:
        func_local.setBlockSize(func.function_space.dofmap.bs)
        func_local.setValuesBlocked(V.dofmap.list, e_eval, addv=PETSc.InsertMode.INSERT)
