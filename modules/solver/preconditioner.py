#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from petsc4py import PETSc


class block_precond():

    def __init__(self, iset, precond_fields, comm=None):

        self.iset = iset
        self.precond_fields = precond_fields
        self.nfields = len(precond_fields)
        assert(len(self.iset)==self.nfields)
        self.comm = comm


    def create(self, pc):
        self.ksp_fields = []
        # create field ksps
        for n in range(self.nfields):
            self.ksp_fields.append( PETSc.KSP().create(self.comm) )
        # set the options
        for n in range(self.nfields):
            self.ksp_fields[n].setType("preonly")
            if self.precond_fields[n] == 'amg':
                self.ksp_fields[n].getPC().setType("hypre")
                self.ksp_fields[n].getPC().setMGLevels(3)
                self.ksp_fields[n].getPC().setHYPREType("boomeramg")
            elif self.precond_fields[n] == 'direct':
                self.ksp_fields[n].getPC().setType("lu")
            else:
                raise ValueError("Currently, only either 'amg' or 'direct' are supported as field-specific preconditioner.")

        self.check_field_size()

    def view(self, pc, vw):
        pass


    def setFromOptions(self, pc):
        pass


class sblock_2x2(block_precond):

    def check_field_size(self):
        assert(self.nfields==2)


    def setUp(self, pc):
        _, self.P = pc.getOperators()

        self.A  = self.P.createSubMatrix(self.iset[0],self.iset[0])
        self.Bt = self.P.createSubMatrix(self.iset[0],self.iset[1])
        self.B  = self.P.createSubMatrix(self.iset[1],self.iset[0])
        self.C  = self.P.createSubMatrix(self.iset[1],self.iset[1])

        ad_vec, adinv_vec = self.A.getDiagonal(), self.A.getDiagonal()
        adinv_vec.reciprocal()

        self.B.diagonalScale(R=adinv_vec)         # right scaling columns of B, corresponds to B * diag(A)^(-1)
        self.B_Adinv_Bt = self.B.matMult(self.Bt) # B diag(A)^-1 Bt

        # --- modified Schur complement Smod = C - B diag(A)^-1 Bt
        self.Smod = self.C - self.B_Adinv_Bt

        self.B.diagonalScale(R=ad_vec)     # restore B


    # computes y = P^(-1) x
    def apply(self, pc, x, y):

        # get subvectors (references!)
        x1, x2 = PETSc.Vec(), PETSc.Vec()
        x.getSubVector(self.iset[0], subvec=x1)
        x.getSubVector(self.iset[1], subvec=x2)

        y1, y2 = self.A.createVecLeft(), self.Smod.createVecLeft()

        # 1) solve A * y_1 = x_1
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[0].solve(x1, y1)

        By1 = PETSc.Vec().createMPI(size=(self.B.getLocalSize()[0],self.B.getSize()[0]), comm=self.comm)
        self.B.mult(y1, By1)
        z2 = x2 - By1

        # 2) solve Smod * y_2 = z_2
        self.ksp_fields[1].setOperators(self.Smod)
        self.ksp_fields[1].solve(z2, y2)

        Bty2 = PETSc.Vec().createMPI(size=(self.Bt.getLocalSize()[0],self.Bt.getSize()[0]), comm=self.comm)
        self.Bt.mult(y2, Bty2)
        z1 = x1 - Bty2

        # 3) solve A * y_1 = z_1
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[0].solve(z1, y1)

        # restore/clean up
        x.restoreSubVector(self.iset[0], subvec=x1)
        x.restoreSubVector(self.iset[1], subvec=x2)

        # set into y vector
        arr_y1, arr_y2 = y1.getArray(), y2.getArray()

        y.setValues(self.iset[0], arr_y1)
        y.setValues(self.iset[1], arr_y2)

        y1.resetArray()
        y2.resetArray()

        y.assemble()


class sblock_3x3(block_precond):

    def check_field_size(self):
        assert(self.nfields==3)


    def setUp(self, pc):
        _, self.P = pc.getOperators()

        self.A  = self.P.createSubMatrix(self.iset[0],self.iset[0])
        self.Bt = self.P.createSubMatrix(self.iset[0],self.iset[1])
        self.Dt = self.P.createSubMatrix(self.iset[0],self.iset[2])
        self.B  = self.P.createSubMatrix(self.iset[1],self.iset[0])
        self.C  = self.P.createSubMatrix(self.iset[1],self.iset[1])
        self.Et = self.P.createSubMatrix(self.iset[1],self.iset[2])
        self.D  = self.P.createSubMatrix(self.iset[2],self.iset[0])
        self.E  = self.P.createSubMatrix(self.iset[2],self.iset[1])
        self.R  = self.P.createSubMatrix(self.iset[2],self.iset[2])

        ad_vec, adinv_vec = self.A.getDiagonal(), self.A.getDiagonal()
        adinv_vec.reciprocal()

        self.B.diagonalScale(R=adinv_vec)         # right scaling columns of B, corresponds to B * diag(A)^(-1)
        self.B_Adinv_Bt = self.B.matMult(self.Bt) # B diag(A)^-1 Bt

        # --- modified Schur complement Smod = C - B diag(A)^-1 Bt
        self.Smod = self.C - self.B_Adinv_Bt

        # --- Tmod = Et - B diag(A)^-1 Dt

        self.B_Adinv_Dt = self.B.matMult(self.Dt) # B diag(A)^-1 Dt

        self.Tmod = self.Et - self.B_Adinv_Dt

        # --- Wmod = R - D diag(A)^(-1) Dt - E diag(Smod)^(-1) Tmod + D diag(A)^(-1) Bt diag(Smod)^(-1) Tmod

        smodd_vec, smoddinv_vec = self.Smod.getDiagonal(), self.Smod.getDiagonal()
        smoddinv_vec.reciprocal()

        self.Bt.diagonalScale(R=smoddinv_vec)                       # right scaling columns of Bt, corresponds to Bt * diag(Smod)^(-1)
        Bt_Smoddinv_Tmod = self.Bt.matMult(self.Tmod)               # Bt diag(Smod)^-1 Tmod

        self.D.diagonalScale(R=adinv_vec)                           # right scaling columns of D, corresponds to D * diag(A)^(-1)
        D_Adinv_Bt_Smoddinv_Tmod = self.D.matMult(Bt_Smoddinv_Tmod) # D diag(A)^(-1) ( Bt diag(Smod)^-1 Tmod )

        self.E.diagonalScale(R=smoddinv_vec)                        # right scaling columns of E, corresponds to E * diag(Smod)^(-1)
        E_Smoddinv_Tmod = self.E.matMult(self.Tmod)                 # E diag(Smod)^-1 Tmod

        D_Adinv_Dt = self.D.matMult(self.Dt)                        # D diag(A)^(-1) Dt - note that D is still scaled by diag(A)^(-1)

        self.Wmod = self.R - D_Adinv_Dt - E_Smoddinv_Tmod + D_Adinv_Bt_Smoddinv_Tmod

        self.B.diagonalScale(R=ad_vec)     # restore B
        self.Bt.diagonalScale(R=smodd_vec) # restore Bt
        self.D.diagonalScale(R=ad_vec)     # restore D
        self.E.diagonalScale(R=smodd_vec)  # restore E

        self.DBt = self.D.matMult(self.Bt)


    # computes y = P^(-1) x
    def apply(self, pc, x, y):

        # get subvectors (references!)
        x1, x2, x3 = PETSc.Vec(), PETSc.Vec(), PETSc.Vec()
        x.getSubVector(self.iset[0], subvec=x1)
        x.getSubVector(self.iset[1], subvec=x2)
        x.getSubVector(self.iset[2], subvec=x3)

        y1, y2, y3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()

        # 1) solve A * y_1 = x_1
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[0].solve(x1, y1)

        By1 = PETSc.Vec().createMPI(size=(self.B.getLocalSize()[0],self.B.getSize()[0]), comm=self.comm)
        self.B.mult(y1, By1)
        z2 = x2 - By1

        # 2) solve Smod * y_2 = z_2
        self.ksp_fields[1].setOperators(self.Smod)
        self.ksp_fields[1].solve(z2, y2)

        Dy1 = PETSc.Vec().createMPI(size=(self.D.getLocalSize()[0],self.D.getSize()[0]), comm=self.comm)

        self.D.mult(y1, Dy1)
        DBty2 = PETSc.Vec().createMPI(size=(self.DBt.getLocalSize()[0],self.DBt.getSize()[0]), comm=self.comm)
        self.DBt.mult(y2, DBty2)
        Ey2 = PETSc.Vec().createMPI(size=(self.E.getLocalSize()[0],self.E.getSize()[0]), comm=self.comm)
        self.E.mult(y2, Ey2)

        z3 = x3 - (Dy1 - DBty2 + Ey2)

        # 3) solve Wmod * y_3 = z_3
        self.ksp_fields[2].setOperators(self.Wmod)
        self.ksp_fields[2].solve(z3, y3)

        Tmody3 = PETSc.Vec().createMPI(size=(self.Tmod.getLocalSize()[0],self.Tmod.getSize()[0]), comm=self.comm)
        self.Tmod.mult(y3, Tmody3)
        z2 = x2 - By1 - Tmody3

        # 4) solve Smod * y_2 = z_2
        self.ksp_fields[1].setOperators(self.Smod)
        self.ksp_fields[1].solve(z2, y2)

        Bty2 = PETSc.Vec().createMPI(size=(self.Bt.getLocalSize()[0],self.Bt.getSize()[0]), comm=self.comm)
        self.Bt.mult(y2, Bty2)
        Dty3 = PETSc.Vec().createMPI(size=(self.Dt.getLocalSize()[0],self.Dt.getSize()[0]), comm=self.comm)
        self.Dt.mult(y3, Dty3)
        z1 = x1 - Bty2 - Dty3

        # 5) solve A * y_1 = z_1
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[0].solve(z1, y1)

        # restore/clean up
        x.restoreSubVector(self.iset[0], subvec=x1)
        x.restoreSubVector(self.iset[1], subvec=x2)
        x.restoreSubVector(self.iset[2], subvec=x3)

        # set into y vector
        arr_y1, arr_y2, arr_y3 = y1.getArray(), y2.getArray(), y3.getArray()

        y.setValues(self.iset[0], arr_y1)
        y.setValues(self.iset[1], arr_y2)
        y.setValues(self.iset[2], arr_y3)

        y1.resetArray()
        y2.resetArray()
        y3.resetArray()

        y.assemble()


# a 4x4 block that does sblock3x3 and a decoupled solve on the 4th block
class sblock_4x4(sblock_3x3):

    def check_field_size(self):
        assert(self.nfields==4)


    def setUp(self, pc):
        super().setUp(pc)

        self.G = self.P.createSubMatrix(self.iset[3],self.iset[3])


    # computes y = P^(-1) x
    def apply(self, pc, x, y):
        super().apply(pc,x,y)

        x4 = PETSc.Vec()
        x.getSubVector(self.iset[3], subvec=x4)

        y4 = self.G.createVecLeft()

        # solve A * y_4 = x_4
        self.ksp_fields[3].setOperators(self.G)
        self.ksp_fields[3].solve(x4, y4)

        # restore/clean up
        x.restoreSubVector(self.iset[3], subvec=x4)

        # set into y vector
        arr_y4 = y4.getArray()

        y.setValues(self.iset[3], arr_y4)

        y4.resetArray()

        y.assemble()
