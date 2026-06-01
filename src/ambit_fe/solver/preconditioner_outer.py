#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from petsc4py import PETSc

from .. import utilities
from . import preconditioner

"""
Ambit outer block preconditioner classes
"""


class block_precond_outer:
    def __init__(self, iset, iset_blocked, precond_fields, io, solparams, comm=None):
        self.iset = iset
        self.iset_blocked = iset_blocked

        num_precs = len(precond_fields)

        if num_precs > 1:
            self.is_blockprec = True
        else:
            self.is_blockprec = False

        self.precond_fields = precond_fields

        self.comm = comm

        self.num_precs = len(self.precond_fields)

        self.inner_precs = [[]] * self.num_precs
        self.bi = [[]] * self.num_precs

        # index set blocks seen by outer BGS
        self.iset_block = [[]] * self.num_precs

        for i, pr in enumerate(self.precond_fields):
            self.bi[i] = pr["blocks"]
            if isinstance(pr["prec"], str):
                self.iset_block[i] = self.iset[self.bi[i][0]]
                if len(self.bi[i])>1:
                    for k in range(1,len(self.bi[i])):
                        self.iset_block[i] = self.iset_block[i].expand(self.iset[self.bi[i][k]])
                    self.iset_block[i].sort()
                self.inner_precs[i] = preconditioner.block_precond([self.iset_block[i]], [pr], io, solparams, comm=comm)
            elif isinstance(pr["prec"], dict):
                if list(pr["prec"].keys())[0]=="s2x2":
                    assert(len(self.bi[i])==2)
                    blocklist = [self.iset_blocked[self.bi[i][0]], self.iset_blocked[self.bi[i][1]]]
                    self.inner_precs[i] = preconditioner.schur2x2(blocklist, pr["prec"]["s2x2"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i][0]].expand(self.iset[self.bi[i][1]])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="s2x2full":
                    assert(len(self.bi[i])==2)
                    blocklist = [self.iset_blocked[self.bi[i][0]], self.iset_blocked[self.bi[i][1]]]
                    self.inner_precs[i] = preconditioner.schur2x2full(blocklist, pr["prec"]["s2x2full"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i][0]].expand(self.iset[self.bi[i][1]])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="s3x3":
                    assert(len(self.bi[i])==3)
                    blocklist = [self.iset_blocked[self.bi[i][0]], self.iset_blocked[self.bi[i][1]], self.iset_blocked[self.bi[i][2]]]
                    self.inner_precs[i] = preconditioner.schur3x3(blocklist, pr["prec"]["s3x3"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i][0]].expand(self.iset[self.bi[i][1]])
                    self.iset_block[i] = self.iset_block[i].expand(self.iset[self.bi[i][2]])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="s3x3full":
                    assert(len(self.bi[i])==3)
                    blocklist = [self.iset_blocked[self.bi[i][0]], self.iset_blocked[self.bi[i][1]], self.iset_blocked[self.bi[i][2]]]
                    self.inner_precs[i] = preconditioner.schur3x3full(blocklist, pr["prec"]["s3x3full"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i][0]].expand(self.iset[self.bi[i][1]])
                    self.iset_block[i] = self.iset_block[i].expand(self.iset[self.bi[i][2]])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="bgs2x2":
                    assert(len(self.bi[i])==2)
                    blocklist = [self.iset_blocked[self.bi[i][0]], self.iset_blocked[self.bi[i][1]]]
                    self.inner_precs[i] = preconditioner.bgs2x2(blocklist, pr["prec"]["bgs2x2"], io, solparams, comm=comm)
                    # create index set encompassing inner BGS blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i][0]].expand(self.iset[self.bi[i][1]])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="bgssym2x2":
                    assert(len(self.bi[i])==2)
                    blocklist = [self.iset_blocked[self.bi[i][0]], self.iset_blocked[self.bi[i][1]]]
                    self.inner_precs[i] = preconditioner.bgssym2x2(blocklist, pr["prec"]["bgssym2x2"], io, solparams, comm=comm)
                    # create index set encompassing inner BGS blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i][0]].expand(self.iset[self.bi[i][1]])
                    self.iset_block[i].sort()
                else:
                    raise ValueError("Unknown inner block preconditioner.")
            else:
                raise ValueError("Unknown instance of 'prec'. Has to be str or list!")

    def create(self, pc):
        # get reference to preconditioner matrix object
        _, self.P = pc.getOperators()
        operator_mats = self.init_mat_vec()
        # set field precs for individual single and block preconditioners
        for i in range(self.num_precs):
            self.inner_precs[i].create_fieldprec(operator_mats[i])

    def mat_mat_mult_nested_2x2(self, A, B, result=None):
        # if result is None:
        #     result = [[None for _ in range(2)] for _ in range(2)]
        #     initial = True
        # else:
        #     initial = False
        for i in range(2):
            for j in range(2):
                Yij = None
                for k in range(2):
                    if isinstance(A, list):
                        Aik = A[i][k]
                    else:
                        Aik = A.getNestSubMatrix(i,k)
                    if isinstance(B, list):
                        Bkj = B[k][j]
                    else:
                        Bkj = B.getNestSubMatrix(k,j)

                    if not bool(Aik) or not bool(Bkj):
                        continue

                    AB = Aik.matMult(Bkj)

                    if Yij is None:
                        Yij = AB
                    else:
                        Yij.axpy(1.0, AB, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                        AB.destroy()

                if Yij is not None:
                    Yij.assemble()

                result[i][j] = Yij

        # return result

# outer forward Block Gauss-Seidel that can have inner block precs (like Schur complement precs)
class BGS_outer(block_precond_outer):
    def init_mat_vec(self):
        self.operator_mats = [[]] * self.num_precs
        self.block_operators = [[]] * self.num_precs

        for i in range(self.num_precs):
            if self.inner_precs[i].is_blockprec:
                self.block_operators[i] = self.P.createSubMatrix(self.iset_block[i], self.iset_block[i])
                self.inner_precs[i].P = self.block_operators[i]
                self.operator_mats[i] = self.inner_precs[i].init_mat_vec()
            else:
                self.operator_mats[i] = [self.P.createSubMatrix(self.iset_block[i], self.iset_block[i])]
                # do we need this???
                self.operator_mats[i][0].setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        self.tridiag_mat_outer_bgs = [[None] * self.num_precs for _ in range(self.num_precs)]
        for i in range(self.num_precs):
            for j in range(self.num_precs):
                if i>=j: # forward BGS - create lower tridiagonal additional blocks
                    self.tridiag_mat_outer_bgs[i][j] = self.P.createSubMatrix(self.iset_block[i], self.iset_block[j])

        self.x, self.y, self.z = [[]] * self.num_precs, [[]] * self.num_precs, [[]] * self.num_precs
        for i in range(self.num_precs):
            self.x[i] = self.tridiag_mat_outer_bgs[i][i].createVecLeft()
            self.y[i] = self.tridiag_mat_outer_bgs[i][i].createVecLeft()
            self.z[i] = self.tridiag_mat_outer_bgs[i][i].createVecLeft()

        self.Oy = []
        for i in range(self.num_precs):
            for j in range(self.num_precs):
                if i>j:
                    self.Oy.append(self.tridiag_mat_outer_bgs[i][j].createVecLeft())

        return self.operator_mats

    def setUp(self, pc):
        # single-prec operator mats, or block preconditioner operators
        for i in range(self.num_precs):
            if not self.inner_precs[i].is_blockprec:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.operator_mats[i][0])
                self.operator_mats[i][0].assemble()
            else:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.block_operators[i])
                self.block_operators[i].assemble()

        for i in range(self.num_precs):
            if self.inner_precs[i].is_blockprec:
                self.inner_precs[i].setUp(pc)  # also sets operators - after block_operators have been created!

        for i in range(self.num_precs):
            for j in range(self.num_precs):
                if i>j: # forward BGS - create lower tridiagonal additional blocks
                    self.P.createSubMatrix(self.iset_block[i], self.iset_block[j], submat=self.tridiag_mat_outer_bgs[i][j])
                    self.tridiag_mat_outer_bgs[i][j].assemble()

        # operator values have changed - do we need to re-set them?
        for i in range(self.num_precs):
            if not self.inner_precs[i].is_blockprec:
                self.inner_precs[i].ksp_fields[0].setOperators(self.operator_mats[i][0])

    # computes y = P^{-1} x
    def apply(self, pc, x, y):
        # get subvectors
        for i in range(self.num_precs):
            x.getSubVector(self.iset_block[i], subvec=self.x[i])

        off=0
        for i in range(self.num_precs):
            self.z[i].axpby(1.0, 0.0, self.x[i])
            for k in range(i):
                self.tridiag_mat_outer_bgs[i][k].mult(self.y[k], self.Oy[k+off])
                self.z[i].axpy(-1.0, self.Oy[k+off])

            # solve
            self.inner_precs[i].apply(pc, self.z[i], self.y[i])

            off+=i

        # restore/clean up
        for i in range(self.num_precs):
            x.restoreSubVector(self.iset_block[i], subvec=self.x[i])

        # set into y vector
        for i in range(self.num_precs):
            y.setValues(self.iset_block[i], self.y[i].array)

        y.assemble()


# outer Schur 2x2 that can have inner block precs (like Schur complement precs)
class Schur2x2_outer(block_precond_outer):
    def init_mat_vec(self):
        self.operator_mats = [[]] * 2
        self.block_operators = [[]] * 2

        self.A = self.P.createSubMatrix(self.iset_block[0], self.iset_block[0])
        self.Bt = self.P.createSubMatrix(self.iset_block[0], self.iset_block[1])
        self.B = self.P.createSubMatrix(self.iset_block[1], self.iset_block[0])
        self.C = self.P.createSubMatrix(self.iset_block[1], self.iset_block[1])

        if self.A.getType()=="nest":
            I_00 = PETSc.Mat().createAIJ(self.A.getNestSubMatrix(0,0).getSizes(), bsize=None, nnz=(1, 1), csr=None, comm=self.comm)
            I_00.setUp()
            I_00.assemble()
            I_11 = PETSc.Mat().createAIJ(self.A.getNestSubMatrix(1,1).getSizes(), bsize=None, nnz=(1, 1), csr=None, comm=self.comm)
            I_11.setUp()
            I_11.assemble()
            self.Adinv = PETSc.Mat().createNest(
                [[I_00, None],
                [None, I_11]],
                comm=self.comm,
            )
        else:
            self.Adinv = PETSc.Mat().createAIJ(self.A.getSizes(), bsize=None, nnz=(1, 1), csr=None, comm=self.comm)

        self.Adinv.setUp()
        self.Adinv.assemble()
        # set 1's to get correct allocation pattern
        self.Adinv.shift(1.0)

        self.adinv_vec = self.A.getDiagonal()

        self.Smod = self.C.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        if self.Bt.getType()=="nest":
            self.Adinv_Bt_ = [[None for j in range(2)] for i in range(2)]
            self.mat_mat_mult_nested_2x2(self.Adinv, self.Bt, result=self.Adinv_Bt_)
        else:
            self.Adinv_Bt = self.Adinv.matMult(self.Bt)

        if self.B.getType()=="nest":
            self.B_Adinv_Bt_ = [[None] * 2 for _ in range(2)]
            self.mat_mat_mult_nested_2x2(self.B, self.Adinv_Bt_, result=self.B_Adinv_Bt_)
            self.B_Adinv_Bt = PETSc.Mat().createNest(self.B_Adinv_Bt_, isrows=None, iscols=None, comm=self.comm)
        else:
            self.B_Adinv_Bt = self.B.matMult(self.Adinv_Bt)

        # need to set Smod here to get the data structures right
        self.Smod.axpy(-1.0, self.B_Adinv_Bt)

        self.x = [self.A.createVecLeft(), self.C.createVecLeft()]
        self.y = [self.A.createVecLeft(), self.C.createVecLeft()]
        self.z = [self.A.createVecLeft(), self.C.createVecLeft()]

        self.By = self.B.createVecLeft()
        self.Bty = self.Bt.createVecLeft()

        # do we need these???
        self.A.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.Smod.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        self.block_operators[0] = self.A
        self.block_operators[1] = self.Smod

        for i in range(2):
            if self.inner_precs[i].is_blockprec:
                #self.block_operators[i] = self.P.createSubMatrix(self.iset_block[i], self.iset_block[i]) #### should goooo
                self.inner_precs[i].P = self.block_operators[i]
                self.operator_mats[i] = self.inner_precs[i].init_mat_vec()
            else:
                self.operator_mats[i] = [self.P.createSubMatrix(self.iset_block[i], self.iset_block[i])]
                # do we need this???
                self.operator_mats[i][0].setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        return self.operator_mats


    def setUp(self, pc):
        self.P.createSubMatrix(self.iset_block[0], self.iset_block[0], submat=self.A)
        self.P.createSubMatrix(self.iset_block[0], self.iset_block[1], submat=self.Bt)
        self.P.createSubMatrix(self.iset_block[1], self.iset_block[0], submat=self.B)
        self.P.createSubMatrix(self.iset_block[1], self.iset_block[1], submat=self.C)
        self.A.assemble()
        self.Bt.assemble()
        self.B.assemble()
        self.C.assemble()

        self.A.getDiagonal(result=self.adinv_vec)
        self.adinv_vec.reciprocal()

        # form diag(A)^{-1}
        self.Adinv.setDiagonal(self.adinv_vec, addv=PETSc.InsertMode.INSERT)

        # --- modified Schur complement Smod = C - B diag(A)^{-1} Bt
        # compute self.Smod = self.C - B_Adinv_Bt
        self.C.copy(result=self.Smod)

        if self.Bt.getType()=="nest":
            self.Adinv_Bt_ = [[None for j in range(2)] for i in range(2)]
            self.mat_mat_mult_nested_2x2(self.Adinv, self.Bt, result=self.Adinv_Bt_)
        else:
            self.Adinv.matMult(self.Bt, result=self.Adinv_Bt)  # diag(A)^{-1} Bt

        if self.B.getType()=="nest":
            self.B_Adinv_Bt_ = [[None] * 2 for _ in range(2)]
            self.mat_mat_mult_nested_2x2(self.B, self.Adinv_Bt_, result=self.B_Adinv_Bt_)
            self.B_Adinv_Bt = PETSc.Mat().createNest(self.B_Adinv_Bt_, isrows=None, iscols=None, comm=self.comm)
        else:
            self.B.matMult(self.Adinv_Bt, result=self.B_Adinv_Bt)  # B diag(A)^{-1} Bt

        self.Smod.axpy(-1.0, self.B_Adinv_Bt)

        # single-prec operator mats, or block preconditioner operators
        for i in range(2):
            if not self.inner_precs[i].is_blockprec:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.operator_mats[i][0])
                self.operator_mats[i][0].assemble()
            # else:
            #     self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.block_operators[i])  #### should goooo
            #     self.block_operators[i].assemble()

        self.block_operators[0] = self.A
        self.block_operators[1] = self.Smod

        for i in range(2):
            if self.inner_precs[i].is_blockprec:
                self.inner_precs[i].setUp(pc)

        # operator values have changed - do we need to re-set them?
        for i in range(2):
            if not self.inner_precs[i].is_blockprec:
                self.inner_precs[i].ksp_fields[0].setOperators(self.operator_mats[i][0])

    # computes y = P^{-1} x
    def apply(self, pc, x, y):
        # get subvectors
        for i in range(2):
            x.getSubVector(self.iset_block[i], subvec=self.x[i])

        # solve
        self.inner_precs[0].apply(pc, self.x[0], self.y[0])

        self.B.mult(self.y[0], self.By)
        self.z[1].axpby(1.0, 0.0, self.x[1])
        self.z[1].axpy(-1.0, self.By)

        # solve - Schur
        self.inner_precs[1].apply(pc, self.z[1], self.y[1])

        self.Bt.mult(self.y[1], self.Bty)

        # compute z1 = x1 - self.Bty
        self.z[0].axpby(1.0, 0.0, self.x[0])
        self.z[0].axpy(-1.0, self.Bty)

        # solve
        self.inner_precs[0].apply(pc, self.z[0], self.y[0])

        # restore/clean up
        for i in range(2):
            x.restoreSubVector(self.iset_block[i], subvec=self.x[i])

        # set into y vector
        for i in range(2):
            y.setValues(self.iset_block[i], self.y[i].array)

        y.assemble()
