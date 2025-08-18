#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

# import h5py
import time
import copy
import math
import os
from dolfinx import fem, io
from petsc4py import PETSc
import numpy as np

from .. import utilities
from .. import ioparams
from ..meshutils import gather_surface_dof_indices


class ModelOrderReduction:
    def __init__(self, pb):
        # underlying physics problem
        self.pb = pb
        self.params = self.pb.mor_params

        ioparams.check_params_rom(self.params)

        self.modes_from_files = self.params.get("modes_from_files", False)

        if not self.modes_from_files:
            self.hdmfilenames = self.params["hdmfilenames"]
            self.num_hdms = len(self.hdmfilenames)
            self.numsnapshots = self.params["numsnapshots"]
            self.snapshotincr = self.params.get("snapshotincr", 1)
            self.snapshotoffset = self.params.get("snapshotoffset", 0)
            self.print_eigenproblem = self.params.get("print_eigenproblem", False)
            self.eigenvalue_cutoff = self.params.get("eigenvalue_cutoff", 0.0)
            self.pod_only = self.params.get("pod_only", False)
        else:
            self.num_hdms, self.numsnapshots = len(self.modes_from_files), 1
            self.pod_only = False

        self.numredbasisvec = self.params.get("numredbasisvec", self.numsnapshots)
        self.orthogonalize_rom_basis = self.params.get("orthogonalize_rom_basis", False)
        self.surface_rom = self.params.get("surface_rom", [])
        self.filetype = self.params.get("filetype", "xdmf_h5")
        self.write_pod_modes = self.params.get("write_pod_modes", False)

        try:
            self.redbasisvec_indices = self.params["redbasisvec_indices"]
        except:
            self.redbasisvec_indices = []
            for i in range(self.numredbasisvec):
                self.redbasisvec_indices.append(i)

        self.regularizations = self.params.get("regularizations", [])
        self.regularizations_integ = self.params.get("regularizations_integ", [])
        self.regularizations_deriv = self.params.get("regularizations_deriv", [])

        if bool(self.regularizations) or bool(self.regularizations_integ) or bool(self.regularizations_deriv):
            self.have_regularization_terms = True
        else:
            self.have_regularization_terms = False

        self.partitions = self.params.get("partitions", [])
        self.exclude_from_snap = self.params.get("exclude_from_snap", [])

        # # mode partitions are either determined by the mode files or partition files
        # if bool(self.modes_from_files):
        #     self.num_partitions = len(self.modes_from_files)
        # else:
        if bool(self.partitions):
            self.num_partitions = len(self.partitions)
        else:
            self.num_partitions = 1

        self.numredbasisvec_partition = self.params.get(
            "numredbasisvec_partition",
            [self.numredbasisvec] * self.num_partitions,
        )

        # some sanity checks
        if not self.modes_from_files:
            if self.num_hdms <= 0:
                raise ValueError("Number of HDMs has to be > 0!")
            if self.numsnapshots <= 0:
                raise ValueError("Number of snapshots has to be > 0!")
            if self.snapshotincr <= 0:
                raise ValueError("Snapshot increment has to be > 0!")
            if len(self.redbasisvec_indices) <= 0 or len(self.redbasisvec_indices) > self.num_hdms * self.numsnapshots:
                raise ValueError(
                    "Number of reduced-basis vectors has to be > 0 and <= number of HDMs times number of snapshots!"
                )

        # function space of variable to be reduced
        self.Vspace = self.pb.V_rom
        # scalar function space
        self.Vspace_sc = self.pb.V_scalar

        # index set for block iterative solvers
        self.im_rom_r = []

        self.locmatsize_u = self.Vspace.dofmap.index_map.size_local * self.Vspace.dofmap.index_map_bs
        self.matsize_u = self.Vspace.dofmap.index_map.size_global * self.Vspace.dofmap.index_map_bs

        # snapshot matrix
        self.S_d = PETSc.Mat().createDense(
            size=(
                (self.locmatsize_u, self.matsize_u),
                (self.num_hdms * self.numsnapshots),
            ),
            bsize=None,
            array=None,
            comm=self.pb.comm,
        )
        self.S_d.setUp()

        # row ownership range of snapshhot matrix (same for ROB operator and non-reduced stiffness matrix)
        self.ss, self.se = self.S_d.getOwnershipRange()

    # offline phase: preparation of reduced order basis
    def prepare_rob(self):
        if bool(self.surface_rom):
            self.fd_set = set(gather_surface_dof_indices(self.pb.io, self.Vspace, self.surface_rom, self.pb.comm))

        # dofs to be excluded from snapshots (e.g. where DBCs are present)
        if bool(self.exclude_from_snap):
            self.excl_set = set(
                gather_surface_dof_indices(
                    self.pb.io,
                    self.Vspace,
                    self.exclude_from_snap,
                    self.pb.comm,
                )
            )

        if not self.modes_from_files:
            self.POD()
        else:
            self.readin_modes()

        self.partition_pod_space()

        # gather Phi
        self.Phi = self.pb.comm.allreduce(self.Phi)

        if self.orthogonalize_rom_basis:
            ts = time.time()
            utilities.print_status("ROM: Orthonormalizing ROM basis...", self.pb.comm, e=" ")

            self.Phi, _ = np.linalg.qr(self.Phi, mode="reduced")

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)

        if self.write_pod_modes and self.pb.pbase.restart_step == 0:
            self.write_modes()

        # exit in case we only want to do POD and write the modes
        if self.pod_only:
            os._exit(0)

        # build reduced basis - either only on designated surface(s) or for the whole model
        if bool(self.surface_rom):
            self.build_reduced_surface_basis()
        else:
            self.build_reduced_basis()

        self.VTV = self.V.transposeMatMult(self.V)
        norm_vtv = self.VTV.norm(PETSc.NormType.NORM_MAX)
        utilities.print_status("ROM: Max-norm of V^{T}*V: %.16f" % (norm_vtv), self.pb.comm)

        # we need to add Cpen * V^T * V to the stiffness - compute here since term is constant
        # V^T * V - normally I, but for badly converged eigenvalues may have non-zero off-diagonal terms...
        if bool(self.regularizations):
            self.xreg = self.V.createVecLeft()
            self.CpenVTV = self.Cpen.matMult(self.VTV)  # Cpen * V^T * V
            self.Vtx, self.regtermx = (
                self.V.createVecRight(),
                self.V.createVecRight(),
            )
            if self.pb.xrpre_ is not None:
                self.Vtxpre = self.V.createVecRight()
        if bool(self.regularizations_integ):
            self.xreginteg = self.V.createVecLeft()
            self.CpenintegVTV = self.Cpeninteg.matMult(self.VTV)  # Cpeninteg * V^T * V
            self.Vtx_integ, self.regtermx_integ = (
                self.V.createVecRight(),
                self.V.createVecRight(),
            )
            if self.pb.xintrpre_ is not None:
                self.Vtxpre = self.V.createVecRight()
        if bool(self.regularizations_deriv):
            self.xregderiv = self.V.createVecLeft()
            self.CpenderivVTV = self.Cpenderiv.matMult(self.VTV)  # Cpeninteg * V^T * V
            self.Vtx_deriv, self.regtermx_deriv = (
                self.V.createVecRight(),
                self.V.createVecRight(),
            )

        self.VTV.destroy()
        self.S_d.destroy()

    # Proper Orthogonal Decomposition
    def POD(self):
        from slepc4py import SLEPc  # only import when we do POD

        ts = time.time()

        utilities.print_status("Performing Proper Orthogonal Decomposition (POD)...", self.pb.comm)

        # gather snapshots (mostly displacements or velocities)
        for h in range(self.num_hdms):
            for i in range(self.numsnapshots):
                step = self.snapshotoffset + (i + 1) * self.snapshotincr

                utilities.print_status("Snapshot %i ..." % (step), self.pb.comm)

                field = fem.Function(self.Vspace)

                self.pb.io.readfunction(
                    field,
                    self.hdmfilenames[h].replace("*", str(step)),
                    filetype=self.filetype,
                )

                self.S_d[self.ss : self.se, self.numsnapshots * h + i] = field.x.petsc_vec[self.ss : self.se]

        # for a surface-restricted ROM, we need to eliminate any snapshots related to non-surface dofs
        if bool(self.surface_rom):
            self.eliminate_mat_all_rows_but_from_id(self.S_d, self.fd_set)

        # eliminate any other unwanted snapshots (e.g. at Dirichlet dofs)
        if bool(self.exclude_from_snap):
            self.eliminate_mat_rows_from_id(self.S_d, self.excl_set)

        self.S_d.assemble()

        # covariance matrix
        C_d = self.S_d.transposeMatMult(self.S_d)  # S^T * S

        # setup eigenvalue problem
        eigsolver = SLEPc.EPS()
        eigsolver.create()
        eigsolver.setOperators(C_d)
        eigsolver.setProblemType(SLEPc.EPS.ProblemType.HEP)  # Hermitian problem
        eigsolver.setType(SLEPc.EPS.Type.LAPACK)
        eigsolver.setFromOptions()

        # solve eigenvalue problem
        eigsolver.solve()

        nconv = eigsolver.getConverged()

        if self.print_eigenproblem:
            utilities.print_status("Number of converged eigenpairs: %d" % (nconv), self.pb.comm)

        evecs, evals = [], []
        self.numredbasisvec_true = 0

        if nconv > 0:
            # create the results vectors
            vr, _ = C_d.getVecs()
            vi, _ = C_d.getVecs()

            if self.print_eigenproblem:
                utilities.print_status(
                    "   k            k/k0         ||Ax-kx||/||kx||",
                    self.pb.comm,
                )
                utilities.print_status(
                    "   ------------ ------------ ----------------",
                    self.pb.comm,
                )

            for i in range(len(self.redbasisvec_indices)):
                k = eigsolver.getEigenpair(self.redbasisvec_indices[i], vr, vi)
                error = eigsolver.computeError(self.redbasisvec_indices[i])
                if self.print_eigenproblem:
                    if i == 0:
                        k0 = k.real
                    if k.imag != 0.0:
                        utilities.print_status(
                            "{:<3s}{:<4.4e}{:<1s}{:<4.4e}{:<1s}{:<3s}{:<4.4e}".format(
                                " ", k.real, "+", k.imag, "j", " ", error
                            ),
                            self.pb.comm,
                        )
                    else:
                        utilities.print_status(
                            "{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<3s}{:<4.4e}".format(
                                " ", k.real, " ", k.real / k0, " ", error
                            ),
                            self.pb.comm,
                        )

                # store
                evecs.append(copy.deepcopy(vr))  # need copy here, otherwise reference changes
                evals.append(k.real)

                if k.real > self.eigenvalue_cutoff:
                    self.numredbasisvec_true += 1

        if len(self.redbasisvec_indices) != self.numredbasisvec_true:
            utilities.print_status(
                "Eigenvalues below cutoff tolerance: Number of reduced basis vectors for ROB changed from %i to %i."
                % (len(self.redbasisvec_indices), self.numredbasisvec_true),
                self.pb.comm,
            )

        # override if some larger than self.numredbasisvec_true are requested
        for i in range(len(self.numredbasisvec_partition)):
            if self.numredbasisvec_partition[i] > self.numredbasisvec_true:
                self.numredbasisvec_partition[i] = self.numredbasisvec_true

        # pop out undesired ones
        for i in range(len(self.redbasisvec_indices) - self.numredbasisvec_true):
            evecs.pop(-1)
            evals.pop(-1)

        # eigenvectors, scaled with 1 / sqrt(eigenval)
        # calculate first numredbasisvec_true POD modes
        self.Phi_all = np.zeros((self.matsize_u, self.numredbasisvec_true))
        for i in range(self.numredbasisvec_true):
            self.Phi_all[self.ss : self.se, i] = self.S_d * evecs[i] / math.sqrt(evals[i])

        te = time.time() - ts

        utilities.print_status("POD done... Time: %.4f s" % (te), self.pb.comm)

    def partition_pod_space(self):
        self.Phi = np.zeros((self.matsize_u, sum(self.numredbasisvec_partition)))

        # first set the entries for the partitions (same for all prior to weighting)
        off = 0
        for h in range(self.num_partitions):
            for i in range(self.numredbasisvec_partition[h]):
                self.Phi[self.ss : self.se, off + i] = self.Phi_all[self.ss : self.se, i]
            off += self.numredbasisvec_partition[h]

        # read partitions and apply to reduced-order basis
        if bool(self.partitions):
            self.readin_partitions()
            off = 0
            for h in range(self.num_partitions):
                for i in range(self.numredbasisvec_partition[h]):
                    self.Phi[self.ss : self.se, off + i] *= self.part_rvar[h].x.petsc_vec[self.ss : self.se]
                off += self.numredbasisvec_partition[h]

    def write_modes(self):
        # write out POD modes
        off = 0
        for h in range(self.num_partitions):
            for i in range(self.numredbasisvec_partition[h]):
                outfile = io.XDMFFile(
                    self.pb.comm,
                    self.pb.io.output_path
                    + "/results_"
                    + self.pb.pbase.simname
                    + "_PODmode_P"
                    + str(h + 1)
                    + "_"
                    + str(i + 1)
                    + ".xdmf",
                    "w",
                )
                outfile.write_mesh(self.pb.io.mesh)
                podfunc = fem.Function(
                    self.Vspace,
                    name="POD_Mode_P" + str(h + 1) + "_" + str(i + 1),
                )
                podfunc.x.petsc_vec[self.ss : self.se] = self.Phi[self.ss : self.se, off + i]
                outfile.write_function(podfunc)
            off += self.numredbasisvec_partition[h]

    # read modes from files
    def readin_modes(self):
        self.numredbasisvec_true = self.numredbasisvec

        self.Phi_all = np.zeros((self.matsize_u, self.numredbasisvec_true))

        # own read function: requires plain txt format of type valx valy valz x z y
        for h in range(self.num_hdms):
            for i in range(self.numredbasisvec_true):
                utilities.print_status("Mode %i ..." % (i + 1), self.pb.comm)

                field = fem.Function(self.Vspace)
                self.pb.io.readfunction(
                    field,
                    self.modes_from_files[h].replace("*", str(i + 1)),
                    filetype=self.filetype,
                )
                self.Phi_all[self.ss : self.se, i] = field.x.petsc_vec[self.ss : self.se]

    # read partitions from files
    def readin_partitions(self):
        self.part, self.part_rvar = [], []

        # own read function: requires plain txt format of type val x z y
        for h in range(self.num_partitions):
            utilities.print_status("Partition %i ..." % (h + 1), self.pb.comm)

            self.part.append(fem.Function(self.Vspace_sc))
            self.pb.io.readfunction(self.part[-1], self.partitions[h], filetype=self.filetype)

            self.part_rvar.append(fem.Function(self.Vspace))

            # map to a vector with same block size as the reduced variable
            bs = self.part_rvar[-1].x.petsc_vec.getBlockSize()
            ps, pe = self.part[-1].x.petsc_vec.getOwnershipRange()
            for i in range(ps, pe):
                for j in range(bs):
                    self.part_rvar[-1].x.petsc_vec[bs * i + j] = self.part[-1].x.petsc_vec[i]

            self.part_rvar[-1].x.petsc_vec.assemble()

    def build_reduced_basis(self):
        ts = time.time()
        utilities.print_status("ROM: Building reduced basis operator...", self.pb.comm, e=" ")

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(
            size=(
                (self.locmatsize_u, self.matsize_u),
                (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
            ),
            bsize=None,
            nnz=(sum(self.numredbasisvec_partition) + 1),
            csr=None,
            comm=self.pb.comm,
        )
        self.V.setUp()

        vrs, vre = self.V.getOwnershipRange()

        # set Phi columns
        self.V[vrs:vre, :] = self.Phi[vrs:vre, :]

        self.V.assemble()

        # set regularizations terms on reduced variable
        if bool(self.regularizations):
            assert len(self.regularizations) == sum(self.numredbasisvec_partition)

            self.Cpen = PETSc.Mat().createAIJ(
                size=(
                    (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
                    (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
                ),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.pb.comm,
            )
            self.Cpen.setUp()

            for i in range(len(self.regularizations)):
                self.Cpen[i, i] = self.regularizations[i]

            self.Cpen.assemble()

        # set regularizations terms on integration of reduced variable
        if bool(self.regularizations_integ):
            assert len(self.regularizations_integ) == sum(self.numredbasisvec_partition)

            self.Cpeninteg = PETSc.Mat().createAIJ(
                size=(
                    (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
                    (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
                ),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.pb.comm,
            )
            self.Cpeninteg.setUp()

            for i in range(len(self.regularizations_integ)):
                self.Cpeninteg[i, i] = self.regularizations_integ[i]

            self.Cpeninteg.assemble()

        # set regularizations terms on derivative of reduced variable
        if bool(self.regularizations_deriv):
            assert len(self.regularizations_deriv) == sum(self.numredbasisvec_partition)

            self.Cpenderiv = PETSc.Mat().createAIJ(
                size=(
                    (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
                    (PETSc.DECIDE, sum(self.numredbasisvec_partition)),
                ),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.pb.comm,
            )
            self.Cpenderiv.setUp()

            for i in range(len(self.regularizations_deriv)):
                self.Cpenderiv[i, i] = self.regularizations_deriv[i]

            self.Cpenderiv.assemble()

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    def build_reduced_surface_basis(self):
        ts = time.time()
        utilities.print_status(
            "ROM: Building reduced basis operator on boundary id(s) " + str(self.surface_rom) + "...",
            self.pb.comm,
            e=" ",
        )

        # number of non-reduced "bulk" dofs
        ndof_bulk = self.matsize_u - len(self.fd_set)

        # all global indices (known to all processes)
        iall = PETSc.IS().createStride(self.matsize_u, first=0, step=1, comm=self.pb.comm)
        # set for faster checking
        iall_set = set(iall.array)

        # row loop to get entries (1's) for "non-reduced" dofs
        nr, a = 0, 0
        row_1, col_1, col_fd = [], [], []

        for row in range(self.matsize_u):
            if row in self.fd_set:
                # increase counter for number of reduced dofs
                nr += 1
                # column shift if we've exceeded the number of reduced basis vectors
                if nr <= sum(self.numredbasisvec_partition):
                    col_fd.append(row)
                if nr > sum(self.numredbasisvec_partition):
                    a += 1
            else:
                # column id of non-reduced dof (left-shifted by a)
                col_id = row - a

                # store
                row_1.append(row)
                col_1.append(col_id)

        # make set for faster checking
        col_fd_set = set(col_fd)

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(
            size=(
                (self.locmatsize_u, self.matsize_u),
                (PETSc.DECIDE, sum(self.numredbasisvec_partition) + ndof_bulk),
            ),
            bsize=None,
            nnz=(sum(self.numredbasisvec_partition) + 1),
            csr=None,
            comm=self.pb.comm,
        )
        self.V.setUp()

        vrs, vre = self.V.getOwnershipRange()
        vcs, vce = self.V.getOwnershipRangeColumn()

        # Phi should not have any non-zero rows that do not belong to a surface dof which is reduced
        for i in range(vrs, vre):
            if i not in self.fd_set:
                assert np.isclose(np.sum(self.Phi[i, :]), 0.0)

        # now set entries
        for k in range(len(row_1)):
            self.V[row_1[k], col_1[k]] = 1.0

        # column loop to insert columns of Phi
        n = 0
        for col in range(sum(self.numredbasisvec_partition) + ndof_bulk):
            # set Phi column
            if col in col_fd_set:
                # prepare index set list for block iterative solver
                if col in range(vcs, vce):
                    self.im_rom_r.append(col)
                # NOTE: We actually do not want to set the columns at once like this, since PETSc may treat close-zero entries as non-zeros
                # self.V[vrs:vre,col] = self.Phi[vrs:vre,n]
                # instead, set like this:
                for k in range(vrs, vre):
                    if not np.isclose(self.Phi[k, n], 0.0):
                        self.V[k, col] = self.Phi[k, n]
                n += 1

        self.V.assemble()

        # set regularizations terms on reduced variable
        if bool(self.regularizations):
            assert len(self.regularizations) == sum(self.numredbasisvec_partition)

            self.Cpen = PETSc.Mat().createAIJ(
                size=(
                    (
                        PETSc.DECIDE,
                        sum(self.numredbasisvec_partition) + ndof_bulk,
                    ),
                    (
                        PETSc.DECIDE,
                        sum(self.numredbasisvec_partition) + ndof_bulk,
                    ),
                ),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.pb.comm,
            )
            self.Cpen.setUp()

            n = 0
            for col in range(sum(self.numredbasisvec_partition) + ndof_bulk):
                if col in col_fd_set:
                    self.Cpen[col, col] = self.regularizations[n]
                    n += 1

            self.Cpen.assemble()

        # set regularizations terms on integration of reduced variable
        if bool(self.regularizations_integ):
            assert len(self.regularizations_integ) == sum(self.numredbasisvec_partition)

            self.Cpeninteg = PETSc.Mat().createAIJ(
                size=(
                    (
                        PETSc.DECIDE,
                        sum(self.numredbasisvec_partition) + ndof_bulk,
                    ),
                    (
                        PETSc.DECIDE,
                        sum(self.numredbasisvec_partition) + ndof_bulk,
                    ),
                ),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.pb.comm,
            )
            self.Cpeninteg.setUp()

            n = 0
            for col in range(sum(self.numredbasisvec_partition) + ndof_bulk):
                if col in col_fd_set:
                    self.Cpeninteg[col, col] = self.regularizations_integ[n]
                    n += 1

            self.Cpeninteg.assemble()

        # set regularizations terms on derivative of reduced variable
        if bool(self.regularizations_deriv):
            assert len(self.regularizations_deriv) == sum(self.numredbasisvec_partition)

            self.Cpenderiv = PETSc.Mat().createAIJ(
                size=(
                    (
                        PETSc.DECIDE,
                        sum(self.numredbasisvec_partition) + ndof_bulk,
                    ),
                    (
                        PETSc.DECIDE,
                        sum(self.numredbasisvec_partition) + ndof_bulk,
                    ),
                ),
                bsize=None,
                nnz=(1, 1),
                csr=None,
                comm=self.pb.comm,
            )
            self.Cpenderiv.setUp()

            n = 0
            for col in range(sum(self.numredbasisvec_partition) + ndof_bulk):
                if col in col_fd_set:
                    self.Cpenderiv[col, col] = self.regularizations_deriv[n]
                    n += 1

            self.Cpenderiv.assemble()

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    # eliminate all rows in matrix but from a set of surface IDs
    def eliminate_mat_all_rows_but_from_id(self, mat, dofs):
        ncol = mat.getSize()[1]
        rs, re = mat.getOwnershipRange()
        for i in range(rs, re):
            if i not in dofs:
                mat[i, :] = np.zeros(ncol)

    # eliminate rows in matrix from a set of surface IDs
    def eliminate_mat_rows_from_id(self, mat, dofs):
        ncol = mat.getSize()[1]
        rs, re = mat.getOwnershipRange()
        for i in range(rs, re):
            if i in dofs:
                mat[i, :] = np.zeros(ncol)

    def set_reduced_data_structures_residual(self, r_list, r_list_rom):
        ts = time.time()
        utilities.print_status("ROM: Project residual, V^{T} * r[0]...", self.pb.comm, e=" ")

        # projection of main block: residual
        r_list_rom[0] = self.V.createVecRight()
        self.V.multTranspose(r_list[0], r_list_rom[0])  # V^T * r_u

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    def set_reduced_data_structures_matrix(self, K_list, K_list_rom, K_list_tmp):
        ts = time.time()
        utilities.print_status("ROM: Project Jacobian, V^{T} * K * V...", self.pb.comm, e=" ")

        # projection of main block: system matrix
        K_list_tmp[0][0] = K_list[0][0].matMult(self.V)  # K_00 * V
        K_list_rom[0][0] = self.V.transposeMatMult(K_list_tmp[0][0])  # V^T * K_00 * V

        nfields = len(K_list)

        # now the offdiagonal blocks
        if nfields > 1:
            for n in range(1, nfields):
                if K_list[0][n] is not None:
                    K_list_rom[0][n] = self.V.transposeMatMult(K_list[0][n])  # V^T * K_{0,n+1}
                if K_list[n][0] is not None:
                    K_list_rom[n][0] = K_list[n][0].matMult(self.V)  # K_{n+1,0} * V

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.pb.comm)

    # online functions
    def reduce_residual(self, r_list, r_list_rom):
        ts = time.time()

        nfields = len(r_list)

        # projection of main block: residual
        self.V.multTranspose(r_list[0], r_list_rom[0])  # V^T * r_u

        # deal with regularizations that may be added to reduced residual to penalize certain modes
        if self.have_regularization_terms:
            self.add_residual_regularization(r_list_rom)

        if nfields > 1:  # only implemented for the first var in list so far!
            for n in range(1, nfields):
                r_list_rom[n] = r_list[n]

        te = time.time() - ts
        if self.pb.io.print_enhanced_info:
            utilities.print_status(
                "       === ROM: Computed V^{T} * r[0], t = %.4f s" % (te),
                self.pb.comm,
            )

    def reduce_stiffness(self, K_list, K_list_rom, K_list_tmp):
        ts = time.time()

        nfields = len(K_list)

        # projection of main block: stiffness
        K_list[0][0].matMult(self.V, result=K_list_tmp[0][0])  # K_00 * V
        self.V.transposeMatMult(K_list_tmp[0][0], result=K_list_rom[0][0])  # V^T * K_00 * V

        # deal with regularizations that may be added to reduced residual to penalize certain modes
        if self.have_regularization_terms:
            self.add_jacobian_regularization(K_list_rom)

        # now the offdiagonal blocks
        if nfields > 1:
            for n in range(1, nfields):
                if K_list[0][n] is not None:
                    self.V.transposeMatMult(K_list[0][n], result=K_list_rom[0][n])  # V^T * K_{0,n+1}
                if K_list[n][0] is not None:
                    K_list[n][0].matMult(self.V, result=K_list_rom[n][0])  # K_{n+1,0} * V
                # no reduction for all other matrices not referring to first field index
                for m in range(1, nfields):
                    K_list_rom[n][m] = K_list[n][m]

        te = time.time() - ts
        if self.pb.io.print_enhanced_info:
            utilities.print_status(
                "       === ROM: Computed V^{T} * K * V, te = %.4f s" % (te),
                self.pb.comm,
            )

    def reconstruct_solution_increment(self, del_x_rom, del_x):
        ts = time.time()

        nfields = len(del_x)

        self.V.mult(del_x_rom[0], del_x[0])  # V * dx_red

        if nfields > 1:  # only implemented for the first var in list so far!
            for n in range(1, nfields):
                del_x[n] = del_x_rom[n]

        te = time.time() - ts

        if self.pb.io.print_enhanced_info:
            utilities.print_status(
                "       === ROM: Computed V * dx_rom[0], te = %.4f s" % (te),
                self.pb.comm,
            )

    def add_residual_regularization(self, r_list_rom):
        _, timefac = self.pb.ti.timefactors()
        dt = self.pb.pbase.dt
        if self.pb.pre:
            timefac, dt = 1.0, self.pb.prestress_dt

        if bool(self.regularizations):
            self.xreg.axpby(timefac, 0.0, self.pb.xr_.x.petsc_vec)
            self.xreg.axpy(1.0 - timefac, self.pb.xr_old_.x.petsc_vec)
            if self.pb.xrpre_ is not None:
                self.xreg.axpy(1.0, self.pb.xrpre_.x.petsc_vec)
            # project
            self.V.multTranspose(self.xreg, self.Vtx)  # V^T * x
            self.Cpen.mult(self.Vtx, self.regtermx)  # Cpen * V^T * x
            r_list_rom[0].axpy(1.0, self.regtermx)  # add penalty term to reduced residual

        if bool(self.regularizations_integ):
            # get integration of variable
            self.pb.ti.update_varint(
                self.pb.xr_.x.petsc_vec,
                self.pb.xr_old_.x.petsc_vec,
                self.pb.xintr_old_.x.petsc_vec,
                dt,
                varintout=self.xreginteg,
                uflform=False,
            )
            self.xreginteg.axpby(1.0 - timefac, timefac, self.pb.xintr_old_.x.petsc_vec)
            if self.pb.xintrpre_ is not None:
                self.xreginteg.axpy(1.0, self.pb.xintrpre_.x.petsc_vec)
            # project
            self.V.multTranspose(self.xreginteg, self.Vtx_integ)  # V^T * x_integ
            self.Cpeninteg.mult(self.Vtx_integ, self.regtermx_integ)  # Cpeninteg * V^T * x_integ
            r_list_rom[0].axpy(1.0, self.regtermx_integ)  # add penalty term to reduced residual

        if bool(self.regularizations_deriv):
            # get derivative of variable
            self.pb.ti.update_dvar(
                self.pb.xr_.x.petsc_vec,
                self.pb.xr_old_.x.petsc_vec,
                self.pb.xdtr_old_.x.petsc_vec,
                dt,
                dvarout=self.xregderiv,
                uflform=False,
            )
            self.xregderiv.axpby(1.0 - timefac, timefac, self.pb.xdtr_old_.x.petsc_vec)
            # project
            self.V.multTranspose(self.xregderiv, self.Vtx_deriv)  # V^T * x_deriv
            self.Cpenderiv.mult(self.Vtx_deriv, self.regtermx_deriv)  # Cpenderiv * V^T * x_deriv
            r_list_rom[0].axpy(1.0, self.regtermx_deriv)  # add penalty term to reduced residual

    def add_jacobian_regularization(self, K_list_rom):
        _, timefac = self.pb.ti.timefactors()
        if self.pb.pre:
            timefac = 1.0

        if bool(self.regularizations):
            K_list_rom[0][0].axpy(timefac, self.CpenVTV)  # K_00 + Cpen * V^T * V - add penalty to stiffness

        if bool(self.regularizations_integ):
            fac_timint = self.pb.ti.get_factor_deriv_varint(self.pb.pbase.dt)
            if self.pb.pre:
                fac_timint = self.pb.prestress_dt
            K_list_rom[0][0].axpy(
                timefac * fac_timint, self.CpenintegVTV
            )  # K_00 + Cpeninteg * V^T * V - add penalty to stiffness

        if bool(self.regularizations_deriv):
            fac_timint = self.pb.ti.get_factor_deriv_dvar(self.pb.pbase.dt)
            if self.pb.pre:
                fac_timint = 1.0 / self.pb.prestress_dt
            K_list_rom[0][0].axpy(
                timefac * fac_timint, self.CpenderivVTV
            )  # K_00 + Cpenderiv * V^T * V - add penalty to stiffness

    def destroy(self):
        self.V.destroy()
        if bool(self.regularizations):
            self.CpenVTV.destroy()
            self.Vtx.destroy()
            self.regtermx.destroy()
            if self.pb.xrpre_ is not None:
                self.Vtxpre.destroy()
        if bool(self.regularizations_integ):
            if self.pb.xintrpre_ is not None:
                self.Vtxpre.destroy()
            self.CpenintegVTV.destroy()
            self.Vtx_integ.destroy()
            self.regtermx_integ.destroy()
        if bool(self.regularizations_deriv):
            self.CpenderivVTV.destroy()
            self.Vtx_deriv.destroy()
            self.regtermx_deriv.destroy()
