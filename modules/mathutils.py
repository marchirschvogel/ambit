#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

def spectral_decomposition_3x3(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
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
    # observe: det(lambda I - A) = 0  with shift  lambda = q + omega --> det(ωI - B) = 0 = omega**3 - j * omega - b
    j = ufl.tr(B*B)/2.  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = ufl.tr(B*B*B)/3.  # == I3(B) for trace-free B
    # solve: 0 = omega**3 - j * omega - b  by substitution  omega = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / p**3
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / p**3  | --> p := sqrt(j * 4/3)
    #        0 = cos(3 * phi) - 4 * b / p**3
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = 2. / ufl.sqrt(3.) * ufl.sqrt(j + eps**2.)  # eps: MMM
    r = 4.*b/(p**3.)
    r = ufl.max_value(ufl.min_value(r, +1. - eps), -1. + eps)  # eps: LMM, MMH
    phi = ufl.acos(r)/3.
    # sorted eigenvalues: lambda1 <= lambda2 <= lambda3
    lambda1 = q + p * ufl.cos(phi + (2./3.) * ufl.pi)  # low
    lambda2 = q + p * ufl.cos(phi + (4./3.) * ufl.pi)  # middle
    lambda3 = q + p * ufl.cos(phi)                     # high

    # determine eigenprojectors E0, E1, E2
    E0 = ufl.diff(lambda1, A).T
    E1 = ufl.diff(lambda2, A).T
    E2 = ufl.diff(lambda3, A).T

    return [lambda1, lambda2, lambda3], [E0, E1, E2]
