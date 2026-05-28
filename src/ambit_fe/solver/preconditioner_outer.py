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
            self.bi[i] = pr["block_index_0"]
            if isinstance(pr["prec"], str):
                self.inner_precs[i] = preconditioner.block_precond(self.iset[self.bi[i]:self.bi[i]+1], [pr], io, solparams, comm=comm)
                self.iset_block[i] = self.iset[self.bi[i]]
            elif isinstance(pr["prec"], dict):
                if list(pr["prec"].keys())[0]=="s2x2":
                    self.inner_precs[i] = preconditioner.schur2x2(self.iset_blocked[self.bi[i]:self.bi[i]+2], pr["prec"]["s2x2"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i]].expand(self.iset[self.bi[i]+1])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="s2x2full":
                    self.inner_precs[i] = preconditioner.schur2x2full(self.iset_blocked[self.bi[i]:self.bi[i]+2], pr["prec"]["s2x2full"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i]].expand(self.iset[self.bi[i]+1])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="s3x3":
                    self.inner_precs[i] = preconditioner.schur3x3(self.iset_blocked[self.bi[i]:self.bi[i]+3], pr["prec"]["s3x3"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i]].expand(self.iset[self.bi[i]+1])
                    self.iset_block[i] = self.iset_block[i].expand(self.iset[self.bi[i]+2])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="s3x3full":
                    self.inner_precs[i] = preconditioner.schur3x3full(self.iset_blocked[self.bi[i]:self.bi[i]+3], pr["prec"]["s3x3full"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i]].expand(self.iset[self.bi[i]+1])
                    self.iset_block[i] = self.iset_block[i].expand(self.iset[self.bi[i]+2])
                    self.iset_block[i].sort()
                elif list(pr["prec"].keys())[0]=="bgs2x2":
                    self.inner_precs[i] = preconditioner.bgs2x2(self.iset_blocked[self.bi[i]:self.bi[i]+2], pr["prec"]["bgs2x2"], io, solparams, comm=comm)
                    # create index set encompassing inner BGS blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i]].expand(self.iset[self.bi[i]+1])
                    self.iset_block[i].sort()
                # allow an outer Schur in an (then even more outer ;-)) outer BGS - TODO, not yet operational!
                elif list(pr["prec"].keys())[0]=="Schur2x2_outer":
                    self.inner_precs[i] = Schur2x2_outer(self.iset_blocked[self.bi[i]:self.bi[i]+2], pr["prec"]["Schur2x2_outer"], io, solparams, comm=comm)
                    # create index set encompassing Schur blocks - seen by outer BGS
                    self.iset_block[i] = self.iset[self.bi[i]].expand(self.iset[self.bi[i]+1])
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
        for i in range(self.num_precs):
            if self.inner_precs[i].is_blockprec:
                self.inner_precs[i].setUp(pc)

        # single-prec operator mats
        for i in range(self.num_precs):
            if not self.inner_precs[i].is_blockprec:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.operator_mats[i][0])
            else:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.block_operators[i])

        for i in range(self.num_precs):
            for j in range(self.num_precs):
                if i>j: # forward BGS - create lower tridiagonal additional blocks
                    self.P.createSubMatrix(self.iset_block[i], self.iset_block[j], submat=self.tridiag_mat_outer_bgs[i][j])

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

        self.A_ = self.P.createSubMatrix(self.iset_block[0], self.iset_block[0])
        self.Bt_ = self.P.createSubMatrix(self.iset_block[0], self.iset_block[1])
        self.B_ = self.P.createSubMatrix(self.iset_block[1], self.iset_block[0])
        self.C_ = self.P.createSubMatrix(self.iset_block[1], self.iset_block[1])

        # mats are nested - we need to convert
        self.A = self.A_.convert("aij")
        self.Bt = self.Bt_.convert("aij")
        self.B = self.B_.convert("aij")
        self.C = self.C_.convert("aij")

        # the matrix to later insert the diagonal
        self.Adinv = PETSc.Mat().createAIJ(self.A.getSizes(), bsize=None, nnz=(1, 1), csr=None, comm=self.comm)
        self.Adinv.setUp()
        self.Adinv.assemble()
        # set 1's to get correct allocation pattern
        self.Adinv.shift(1.0)

        self.adinv_vec = self.A.getDiagonal()

        self.Smod = self.C.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        self.Adinv_Bt = self.Adinv.matMult(self.Bt)
        self.B_Adinv_Bt = self.B.matMult(self.Adinv_Bt)

        # need to set Smod here to get the data structures right
        self.Smod.axpy(-1.0, self.B_Adinv_Bt)

        self.x = [self.A.createVecLeft(), self.Smod.createVecLeft()]
        self.y = [self.A.createVecLeft(), self.Smod.createVecLeft()]
        self.z = [self.A.createVecLeft(), self.Smod.createVecLeft()]

        self.By = self.B.createVecLeft()
        self.Bty = self.Bt.createVecLeft()

        # do we need these???
        self.A.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.Smod.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        self.block_operators[0] = self.A
        self.block_operators[1] = self.Smod

        for i in range(2):
            if self.inner_precs[i].is_blockprec:
                self.block_operators[i] = self.P.createSubMatrix(self.iset_block[i], self.iset_block[i])
                self.inner_precs[i].P = self.block_operators[i]
                self.operator_mats[i] = self.inner_precs[i].init_mat_vec()
            else:
                self.operator_mats[i] = [self.P.createSubMatrix(self.iset_block[i], self.iset_block[i])]
                # do we need this???
                self.operator_mats[i][0].setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        return self.operator_mats

    def setUp(self, pc):
        for i in range(2):
            if self.inner_precs[i].is_blockprec:
                self.inner_precs[i].setUp(pc)

        # single-prec operator mats
        for i in range(2):
            if not self.inner_precs[i].is_blockprec:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.operator_mats[i][0])
            else:
                self.P.createSubMatrix(self.iset_block[i], self.iset_block[i], submat=self.block_operators[i])

        # operator values have changed - do we need to re-set them?
        for i in range(self.num_precs):
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

        # compute z2 = x2 - self.By
        self.z[1].axpby(1.0, 0.0, self.x[1])
        self.z[1].axpy(-1.0, self.By)

        # solve
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
