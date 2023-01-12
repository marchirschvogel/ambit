#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from petsc4py import PETSc

# check out
# Elman et al. 2008 "A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier–Stokes equations"


def simple2x2(K_00,K_01,K_10,K_11):

    if K_11 is None:
        K_11 = PETSc.Mat().createAIJ(size=((K_01.getLocalSize()[1],K_01.getSize()[1]),(K_01.getLocalSize()[1],K_01.getSize()[1])))
        K_11.setUp()
        K_11.assemble()

    Kuu_diag_vec = K_00.getDiagonal()
    Kuu_diag_vec.reciprocal()

    invdiagK00_K01 = K_01.duplicate(copy=True)
    invdiagK00_K01.diagonalScale(Kuu_diag_vec,None)

    # SIMPLE preconditioner 
    #compute modified Schur complement: S = K_11 - K_10 * diag(K_00)^-1 * K_01 (instead of using expensive K_00^-1)

    S = K_10.matMult(invdiagK00_K01)
    S.axpy(-1., K_11)
    
    # identity matrices
    I_00 = PETSc.Mat().createAIJ(size=((K_00.getLocalSize()[0],K_00.getSize()[0]),(K_00.getLocalSize()[0],K_00.getSize()[0])))
    I_00.setUp()
    
    one = I_00.createVecLeft()
    one.set(1.0)
    
    I_00.setDiagonal(one)
    I_00.assemble()
    
    I_11 = PETSc.Mat().createAIJ(size=(K_11.getSize()[0],K_11.getSize()[0]))
    I_11.setUp()
    
    one = I_11.createVecLeft()
    one.set(1.0)
    I_11.setDiagonal(one)
    I_11.assemble()

    K00_invdiagK00_K01 = K_00.matMult(invdiagK00_K01)
    
    K10_invdiagK00_K01 = K_10.matMult(invdiagK00_K01)
    
    alpha = 1.0

    # SIMPLE preconditioner matrix
    P = PETSc.Mat().createNest([[K_00, K00_invdiagK00_K01], [K_10, K10_invdiagK00_K01 - alpha*S]])
    P.assemble()
    
    return P
