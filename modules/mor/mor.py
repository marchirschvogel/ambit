#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#import h5py
import time, sys, copy, math
from dolfinx import fem, io
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
from meshutils import gather_surface_dof_indices


class ModelOrderReduction():

    def __init__(self, params, Vspace, io, comm):

        try: self.modes_from_files = params['modes_from_files']
        except: self.modes_from_files = False

        if not self.modes_from_files:

            self.hdmfilenames = params['hdmfilenames']
            self.numhdms = len(self.hdmfilenames)
            self.numsnapshots = params['numsnapshots']

            try: self.snapshotincr = params['snapshotincr']
            except: self.snapshotincr = 1

            try: self.snapshotoffset = params['snapshotoffset']
            except: self.snapshotoffset = 0

            try: self.print_eigenproblem = params['print_eigenproblem']
            except: self.print_eigenproblem = False

            try: self.eigenvalue_cutoff = params['eigenvalue_cutoff']
            except: self.eigenvalue_cutoff = 0.0
        else:
            self.numhdms, self.numsnapshots = 1, 1

        try: self.numredbasisvec = params['numredbasisvec']
        except: self.numredbasisvec = self.numsnapshots

        try: self.surface_rom = params['surface_rom']
        except: self.surface_rom = []

        try:
            self.filesource = params['filesource']
        except:
            self.filesource = 'petscvector'

        try: self.write_pod_modes = params['write_pod_modes']
        except: self.write_pod_modes = False

        try: self.redbasisvec_indices = params['redbasisvec_indices']
        except:
            self.redbasisvec_indices = []
            for i in range(self.numredbasisvec): self.redbasisvec_indices.append(i)

        try: self.redbasisvec_penalties = params['redbasisvec_penalties']
        except: self.redbasisvec_penalties = []

        try: self.partitions = params['partitions']
        except: self.partitions = []

        try: self.exclude_from_snap = params['exclude_from_snap']
        except: self.exclude_from_snap = []

        # mode partitions are either determined by the mode files or partition files
        if bool(self.modes_from_files):
            self.num_partitions = len(self.modes_from_files)
        else:
            if bool(self.partitions):
                self.num_partitions = len(self.partitions)
            else:
                self.num_partitions = 1

        # some sanity checks
        if not self.modes_from_files:
            if self.numhdms <= 0:
                raise ValueError('Number of HDMs has to be > 0!')
            if self.numsnapshots <= 0:
                raise ValueError('Number of snapshots has to be > 0!')
            if self.snapshotincr <= 0:
                raise ValueError('Snapshot increment has to be > 0!')
            if len(self.redbasisvec_indices) <= 0 or len(self.redbasisvec_indices) > self.numhdms*self.numsnapshots:
                raise ValueError('Number of reduced-basis vectors has to be > 0 and <= number of HDMs times number of snapshots!')

        # to access mesh data
        self.io = io
        # function space of variable to be reduced
        self.Vspace = Vspace[0]
        # scalar function space
        self.Vspace_sc = Vspace[1]

        # index set for block iterative solvers
        self.im_rom_r = []

        self.comm = comm

        self.locmatsize_u = self.Vspace.dofmap.index_map.size_local * self.Vspace.dofmap.index_map_bs
        self.matsize_u = self.Vspace.dofmap.index_map.size_global * self.Vspace.dofmap.index_map_bs

        # snapshot matrix
        self.S_d = PETSc.Mat().createDense(size=((self.locmatsize_u,self.matsize_u),(self.numhdms*self.numsnapshots)), bsize=None, array=None, comm=self.comm)
        self.S_d.setUp()

        # row ownership range of snapshhot matrix (same for ROB operator and non-reduced stiffness matrix)
        self.ss, self.se = self.S_d.getOwnershipRange()


    # offline phase: preparation of reduced order basis
    def prepare_rob(self):

        if bool(self.surface_rom):
            self.fd_set = set(gather_surface_dof_indices(self.io, self.Vspace, self.surface_rom, self.comm))

        # dofs to be excluded from snapshots (e.g. where DBCs are present)
        if bool(self.exclude_from_snap):
            self.excl_set = set(gather_surface_dof_indices(self.io, self.Vspace, self.exclude_from_snap, self.comm))

        if not self.modes_from_files:
            self.POD()
        else:
            self.readin_modes()

        if self.write_pod_modes:
            self.write_modes()

        # build reduced basis - either only on designated surface(s) or for the whole model
        if bool(self.surface_rom):
            self.build_reduced_surface_basis()
        else:
            self.build_reduced_basis()

        if bool(self.redbasisvec_penalties):
            # we need to add Cpen * V^T * V to the stiffness - compute here since term is constant
            # V^T * V - normally I, but for badly converged eigenvalues may have non-zero off-diagonal terms...
            self.VTV = self.V.transposeMatMult(self.V)
            self.CpenVTV = self.Cpen.matMult(self.VTV) # Cpen * V^T * V


    # Proper Orthogonal Decomposition
    def POD(self):

        ts = time.time()

        if self.comm.rank==0:
            print("Performing Proper Orthogonal Decomposition (POD) ...")
            sys.stdout.flush()

        # gather snapshots (mostly displacements or velocities)
        for h in range(self.numhdms):

            for i in range(self.numsnapshots):

                if self.comm.rank==0:
                    print("Reading snapshot %i ..." % (i+1))
                    sys.stdout.flush()

                step = self.snapshotoffset + (i+1)*self.snapshotincr

                field = fem.Function(self.Vspace)

                if self.filesource == 'petscvector':
                    # WARNING: Like this, we can only load data with the same amount of processes as it has been written!
                    viewer = PETSc.Viewer().createMPIIO(self.hdmfilenames[h].replace('*',str(step)), 'r', self.comm)
                    field.vector.load(viewer)

                    field.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                elif self.filesource == 'rawtxt':

                    # own read function: requires plain txt format of type valx valy valz x z y
                    self.io.readfunction(field, self.Vspace, self.hdmfilenames[h].replace('*',str(step)))

                else:
                    raise NameError("Unknown filesource!")

                self.S_d[self.ss:self.se, self.numsnapshots*h+i] = field.vector[self.ss:self.se]

        # for a surface-restricted ROM, we need to eliminate any snapshots related to non-surface dofs
        if bool(self.surface_rom):
            self.eliminate_mat_all_rows_but_from_id(self.S_d, self.fd_set)

        # eliminate any other unwanted snapshots (e.g. at Dirichlet dofs)
        if bool(self.exclude_from_snap):
            self.eliminate_mat_rows_from_id(self.S_d, self.excl_set)

        self.S_d.assemble()

        # covariance matrix
        C_d = self.S_d.transposeMatMult(self.S_d) # S^T * S

        # setup eigenvalue problem
        eigsolver = SLEPc.EPS()
        eigsolver.create()
        eigsolver.setOperators(C_d)
        eigsolver.setProblemType(SLEPc.EPS.ProblemType.HEP) # Hermitian problem
        eigsolver.setType(SLEPc.EPS.Type.LAPACK)
        eigsolver.setFromOptions()

        # solve eigenvalue problem
        eigsolver.solve()

        nconv = eigsolver.getConverged()

        if self.print_eigenproblem:
            if self.comm.rank==0:
                print("Number of converged eigenpairs: %d" % nconv)
                sys.stdout.flush()

        evecs, evals = [], []
        self.numredbasisvec_true = 0

        if nconv > 0:
            # create the results vectors
            vr, _ = C_d.getVecs()
            vi, _ = C_d.getVecs()

            if self.print_eigenproblem:
                if self.comm.rank==0:
                    print("   k                        ||Ax-kx||/||kx||")
                    print("   ----------------------   ----------------")
                    sys.stdout.flush()

            for i in range(len(self.redbasisvec_indices)):
                k = eigsolver.getEigenpair(self.redbasisvec_indices[i], vr, vi)
                error = eigsolver.computeError(self.redbasisvec_indices[i])
                if self.print_eigenproblem:
                    if k.imag != 0.0:
                        if self.comm.rank==0:
                            print('{:<3s}{:<4.4e}{:<1s}{:<4.4e}{:<1s}{:<3s}{:<4.4e}'.format(' ',k.real,'+',k.imag,'j',' ',error))
                            sys.stdout.flush()
                    else:
                        if self.comm.rank==0:
                            print('{:<3s}{:<4.4e}{:<15s}{:<4.4e}'.format(' ',k.real,' ',error))
                            sys.stdout.flush()

                # store
                evecs.append(copy.deepcopy(vr)) # need copy here, otherwise reference changes
                evals.append(k.real)

                if k.real > self.eigenvalue_cutoff: self.numredbasisvec_true += 1

        if len(self.redbasisvec_indices) != self.numredbasisvec_true:
            if self.comm.rank==0:
                print("Eigenvalues below cutoff tolerance: Number of reduced basis vectors for ROB changed from %i to %i." % (len(self.redbasisvec_indices),self.numredbasisvec_true))
                sys.stdout.flush()

        # pop out undesired ones
        for i in range(len(self.redbasisvec_indices)-self.numredbasisvec_true):
            evecs.pop(-1)
            evals.pop(-1)

        # eigenvectors, scaled with 1 / sqrt(eigenval)
        # calculate first numredbasisvec_true POD modes
        self.Phi = np.zeros((self.matsize_u, self.numredbasisvec_true*self.num_partitions))
        # first set the entries for the partitions (same for all prior to weighting)
        for h in range(self.num_partitions):
            for i in range(self.numredbasisvec_true):
                self.Phi[self.ss:self.se, self.numredbasisvec_true*h+i] = self.S_d * evecs[i] / math.sqrt(evals[i])

        # read partitions and apply to reduced-order basis
        if bool(self.partitions):
            self.readin_partitions()
            for h in range(self.num_partitions):
                for i in range(self.numredbasisvec_true):
                    self.Phi[self.ss:self.se, self.numredbasisvec_true*h+i] *= self.part_rvar[h].vector[self.ss:self.se]

        te = time.time() - ts

        if self.comm.rank==0:
            print("POD done... Time: %.4f s" % (te))
            sys.stdout.flush()


    def write_modes(self):
        # write out POD modes
        for h in range(self.num_partitions):
            for i in range(self.numredbasisvec_true):
                outfile = io.XDMFFile(self.comm, self.io.output_path+'/results_'+self.io.sname+'_PODmode_P'+str(h+1)+'_'+str(i+1)+'.xdmf', 'w')
                outfile.write_mesh(self.io.mesh)
                podfunc = fem.Function(self.Vspace, name="POD_Mode_P"+str(h+1)+"_"+str(i+1))
                podfunc.vector[self.ss:self.se] = self.Phi[self.ss:self.se, self.numredbasisvec_true*h+i]
                outfile.write_function(podfunc)


    # read modes from files
    def readin_modes(self):

        self.numredbasisvec_true = self.numredbasisvec

        self.Phi = np.zeros((self.matsize_u, self.numredbasisvec_true*self.num_partitions))

        # own read function: requires plain txt format of type valx valy valz x z y
        for h in range(self.num_partitions):

            if self.num_partitions > 1:
                if self.comm.rank==0:
                    print("Modes for partition %i:" % (h+1))
                    sys.stdout.flush()

            for i in range(self.numredbasisvec_true):

                if self.comm.rank==0:
                    print("Reading mode %i ..." % (i+1))
                    sys.stdout.flush()

                field = fem.Function(self.Vspace)
                self.io.readfunction(field, self.Vspace, self.modes_from_files[h].replace('*',str(i+1)))
                self.Phi[self.ss:self.se, self.numredbasisvec_true*h+i] = field.vector[self.ss:self.se]


    # read partitions from files
    def readin_partitions(self):

        self.part, self.part_rvar = [], []

        # own read function: requires plain txt format of type val x z y
        for h in range(self.num_partitions):

            if self.comm.rank==0:
                print("Reading partition %i ..." % (h+1))
                sys.stdout.flush()

            self.part.append( fem.Function(self.Vspace_sc) )
            self.io.readfunction(self.part[-1], self.Vspace_sc, self.partitions[h])

            self.part_rvar.append( fem.Function(self.Vspace) )

            # map to a vector with same block size as the reduced variable
            bs = self.part_rvar[-1].vector.getBlockSize()
            ps,pe = self.part[-1].vector.getOwnershipRange()
            for i in range(ps,pe):
                for j in range(bs):
                    self.part_rvar[-1].vector[bs*i+j] = self.part[-1].vector[i]

            self.part_rvar[-1].vector.assemble()


    def build_reduced_basis(self):

        ts = time.time()

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(size=((self.locmatsize_u,self.matsize_u),(self.numredbasisvec_true*self.num_partitions)), bsize=None, nnz=(self.numredbasisvec_true*self.num_partitions,self.locmatsize_u), csr=None, comm=self.comm)
        self.V.setUp()

        vrs, vre = self.V.getOwnershipRange()

        # set Phi columns
        self.V[vrs:vre,:] = self.Phi[vrs:vre,:]

        self.V.assemble()

        # set penalties
        if bool(self.redbasisvec_penalties):
            self.Cpen = PETSc.Mat().createAIJ(size=((self.numredbasisvec_true*self.num_partitions),(self.numredbasisvec_true*self.num_partitions)), bsize=None, nnz=(self.numredbasisvec_true*self.num_partitions), csr=None, comm=self.comm)
            self.Cpen.setUp()

            for i in range(len(self.redbasisvec_penalties)):
                self.Cpen[i,i] = self.redbasisvec_penalties[i]

            self.Cpen.assemble()

        te = time.time() - ts

        if self.comm.rank==0:
            print("Built reduced basis operator for ROM. Time: %.4f s" % (te))
            sys.stdout.flush()


    def build_reduced_surface_basis(self):

        ts = time.time()

        # number of non-reduced "bulk" dofs
        ndof_bulk = self.matsize_u - len(self.fd_set)

        # all global indices (known to all processes)
        iall = PETSc.IS().createStride(self.matsize_u, first=0, step=1, comm=self.comm)
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
                if nr <= self.numredbasisvec_true*self.num_partitions:
                    col_fd.append(row)
                if nr > self.numredbasisvec_true*self.num_partitions:
                    a += 1
            else:
                # column id of non-reduced dof (left-shifted by a)
                col_id = row-a

                # store
                row_1.append(row)
                col_1.append(col_id)

        # make set for faster checking
        col_fd_set = set(col_fd)

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(size=((self.locmatsize_u,self.matsize_u),(self.numredbasisvec_true*self.num_partitions+ndof_bulk)), bsize=None, nnz=(self.numredbasisvec_true*self.num_partitions+1,self.locmatsize_u), csr=None, comm=self.comm)
        self.V.setUp()

        vrs, vre = self.V.getOwnershipRange()
        vcs, vce = self.V.getOwnershipRangeColumn()

        # Phi should not have any non-zero rows that do not belong to a surface dof which is reduced
        for i in range(vrs, vre):
            if i not in self.fd_set:
                assert(np.isclose(np.sum(self.Phi[i,:]), 0.0))

        # now set entries
        for k in range(len(row_1)):
            self.V[row_1[k],col_1[k]] = 1.0

        # column loop to insert columns of Phi
        n=0
        for col in range(self.numredbasisvec_true*self.num_partitions+ndof_bulk):
            # set Phi column
            if col in col_fd_set:
                # prepare index set list for block iterative solver
                if col in range(vcs, vce): self.im_rom_r.append(col)
                # NOTE: We actually do not want to set the columns at once like this, since PETSc may treat close-zero entries as non-zeros
                # self.V[vrs:vre,col] = self.Phi[vrs:vre,n]
                # instead, set like this:
                for k in range(vrs,vre): # TODO: Find out why setting this way yields to initial NaNs in FrSI testcase, only when run with 2 or 3 cores!
                    if not np.isclose(self.Phi[k,n],0.0): self.V[k,col] = self.Phi[k,n]
                n += 1

        self.V.assemble()

        # set penalties
        if bool(self.redbasisvec_penalties):

            self.Cpen = PETSc.Mat().createAIJ(size=((self.numredbasisvec_true*self.num_partitions+ndof_bulk),(self.numredbasisvec_true*self.num_partitions+ndof_bulk)), bsize=None, nnz=(self.numredbasisvec_true*self.num_partitions), csr=None, comm=self.comm)
            self.Cpen.setUp()

            n=0
            for col in range(self.numredbasisvec_true*self.num_partitions+ndof_bulk):
                if col in col_fd_set:
                    self.Cpen[col,col] = self.redbasisvec_penalties[n]
                    n += 1

            self.Cpen.assemble()

        te = time.time() - ts

        if self.comm.rank==0:
            print("Built reduced basis operator for ROM on boundary id(s) "+str(self.surface_rom)+". Time: %.4f s" % (te))
            sys.stdout.flush()


    # eliminate all rows in matrix but from a set of surface IDs
    def eliminate_mat_all_rows_but_from_id(self, mat, dofs):

        ncol = mat.getSize()[1]
        rs,re = mat.getOwnershipRange()
        for i in range(rs,re):
            if i not in dofs:
                mat[i,:] = np.zeros(ncol)


    # eliminate rows in matrix from a set of surface IDs
    def eliminate_mat_rows_from_id(self, mat, dofs):

        ncol = mat.getSize()[1]
        rs,re = mat.getOwnershipRange()
        for i in range(rs,re):
            if i in dofs:
                mat[i,:] = np.zeros(ncol)
