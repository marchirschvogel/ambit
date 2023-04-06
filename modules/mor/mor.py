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

    def __init__(self, params, comm):
        
        self.numhdms = params['numhdms']
        self.numsnapshots = params['numsnapshots']
        
        try: self.snapshotincr = params['snapshotincr']
        except: self.snapshotincr = 1
        
        try: self.snapshotoffset = params['snapshotoffset']
        except: self.snapshotoffset = 0
        
        try: self.numredbasisvec = params['numredbasisvec']
        except: self.numredbasisvec = self.numsnapshots
        
        try: self.print_eigenproblem = params['print_eigenproblem']
        except: self.print_eigenproblem = False
        
        try: self.eigenvalue_cutoff = params['eigenvalue_cutoff']
        except: self.eigenvalue_cutoff = 0.0
        
        try: self.surface_rom = params['surface_rom']
        except: self.surface_rom = []
        
        try: self.snapshotsource = params['snapshotsource']
        except:
            self.snapshotsource = []
            for h in range(self.numhdms): self.snapshotsource.append('petscvector')
        
        try: self.snapshotreadin_tol = params['snapshotreadin_tol']
        except: self.snapshotreadin_tol = 1.0e-5
        
        try: self.write_pod_modes = params['write_pod_modes']
        except: self.write_pod_modes = False
        
        try: self.redbasisvec_indices = params['redbasisvec_indices']
        except:
            self.redbasisvec_indices = []
            for i in range(self.numredbasisvec): self.redbasisvec_indices.append(i)
        
        try: self.redbasisvec_penalties = params['redbasisvec_penalties']
        except: self.redbasisvec_penalties = []
        
        try: self.modes_from_files = params['modes_from_files']
        except: self.modes_from_files = False
        
        # some sanity checks
        if self.numhdms <= 0:
            raise ValueError('Number of HDMs has to be > 0!')
        if self.numsnapshots <= 0:
            raise ValueError('Number of snapshots has to be > 0!')
        if self.snapshotincr <= 0:
            raise ValueError('Snapshot increment has to be > 0!')
        if len(self.redbasisvec_indices) <= 0 or len(self.redbasisvec_indices) > self.numhdms*self.numsnapshots:
            raise ValueError('Number of reduced-basis vectors has to be > 0 and <= number of HDMs times number of snapshots!')
        
        self.hdmfilenames = params['hdmfilenames']

        self.comm = comm
        
    
    # Proper Orthogonal Decomposition
    def POD(self, pb, Vspace):
        
        locmatsize_u = Vspace.dofmap.index_map.size_local * Vspace.dofmap.index_map_bs
        matsize_u = Vspace.dofmap.index_map.size_global * Vspace.dofmap.index_map_bs
        
        # snapshot matrix
        S_d = PETSc.Mat().createDense(size=((locmatsize_u,matsize_u),(self.numhdms*self.numsnapshots)), bsize=None, array=None, comm=self.comm)
        S_d.setUp()
            
        ss, se = S_d.getOwnershipRange()
        
        if bool(self.surface_rom):
            self.fd_set = set(gather_surface_dof_indices(pb, Vspace, self.surface_rom,self.comm))
        
        ts = time.time()
        
        if not self.modes_from_files:

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
                
                    field = fem.Function(Vspace)

                    if self.snapshotsource[h] == 'petscvector':
                        # WARNING: Like this, we can only load data with the same amount of processes as it has been written!
                        viewer = PETSc.Viewer().createMPIIO(self.hdmfilenames[h].replace('*',str(step)), 'r', self.comm)
                        field.vector.load(viewer)
                        
                        field.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    
                    elif self.snapshotsource[h] == 'rawtxt':
                        
                        # own read function: requires plain txt format of type valx valy valz x z y
                        pb.io.readfunction(field, Vspace, self.hdmfilenames[h].replace('*',str(step)), tol=self.snapshotreadin_tol)
                        
                    else:
                        raise NameError("Unknown snapshotsource!")

                    S_d[ss:se, self.numhdms*h+i] = field.vector[ss:se]

            # for a surface-restricted ROM, we need to eliminate any snapshots related to non-surface dofs
            if bool(self.surface_rom):
                # eliminate corresponding rows in S_d
                for i in range(ss, se):
                    if i not in self.fd_set:
                        S_d[i,:] = np.zeros(self.numsnapshots)

            S_d.assemble()

            # covariance matrix
            C_d = S_d.transposeMatMult(S_d) # S^T * S
            
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
            numredbasisvec_true = 0

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
                    
                    if k.real > self.eigenvalue_cutoff: numredbasisvec_true += 1

            if len(self.redbasisvec_indices) != numredbasisvec_true:
                if self.comm.rank==0:
                    print("Eigenvalues below cutoff tolerance: Number of reduced basis vectors for ROB changed from %i to %i." % (len(self.redbasisvec_indices),numredbasisvec_true))
                    sys.stdout.flush()
            
            # pop out undesired ones
            for i in range(len(self.redbasisvec_indices)-numredbasisvec_true):
                evecs.pop(-1)
                evals.pop(-1)

            # eigenvectors, scaled with 1 / sqrt(eigenval)
            # calculate first numredbasisvec_true POD modes
            self.Phi = np.zeros((matsize_u, numredbasisvec_true))
            for i in range(numredbasisvec_true):
                self.Phi[ss:se,i] = S_d * evecs[i] / math.sqrt(evals[i])       
        
        else:
            
            numredbasisvec_true = self.numredbasisvec
            
            self.Phi = np.zeros((matsize_u, numredbasisvec_true))
            # read modes from files
            # own read function: requires plain txt format of type valx valy valz x z y
            for i in range(numredbasisvec_true):
                
                if self.comm.rank==0:
                    print("Reading mode %i ..." % (i+1))
                    sys.stdout.flush()
                
                field = fem.Function(Vspace)
                pb.io.readfunction(field, Vspace, self.modes_from_files.replace('*',str(i+1)), tol=self.snapshotreadin_tol)
                self.Phi[ss:se,i] = field.vector[ss:se]

        # write out POD modes
        if self.write_pod_modes:
            for i in range(numredbasisvec_true):
                outfile = io.XDMFFile(self.comm, pb.io.output_path+'/results_'+pb.simname+'_PODmode_'+str(i+1)+'.xdmf', 'w')
                outfile.write_mesh(pb.io.mesh)
                podfunc = fem.Function(Vspace)
                podfunc.vector[ss:se] = self.Phi[ss:se,i]
                outfile.write_function(podfunc)
        
        # build reduced basis - either only on designated surfaces or for the whole model
        if bool(self.surface_rom):
            self.build_reduced_surface_basis(Vspace,numredbasisvec_true,ts)
        else:
            self.build_reduced_basis(Vspace,numredbasisvec_true,ts)

        if bool(self.redbasisvec_penalties):
            # we need to add Cpen * V^T * V to the stiffness - compute here since term is constant
            # V^T * V - normally I, but for badly converged eigenvalues may have non-zero off-diagonal terms...
            self.VTV = self.V.transposeMatMult(self.V) 
            self.CpenVTV = self.Cpen.matMult(self.VTV) # Cpen * V^T * V


    def build_reduced_basis(self, Vspace, rb, ts):

        locmatsize_u = Vspace.dofmap.index_map.size_local * Vspace.dofmap.index_map_bs
        matsize_u = Vspace.dofmap.index_map.size_global * Vspace.dofmap.index_map_bs

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(rb)), bsize=None, nnz=(rb,locmatsize_u), csr=None, comm=self.comm)
        self.V.setUp()
        
        vrs, vre = self.V.getOwnershipRange()
        
        # set Phi columns
        self.V[vrs:vre,:] = self.Phi[vrs:vre,:]
  
        self.V.assemble()
       
        # set penalties
        if bool(self.redbasisvec_penalties):
            self.Cpen = PETSc.Mat().createAIJ(size=((rb),(rb)), bsize=None, nnz=(rb), csr=None, comm=self.comm)
            self.Cpen.setUp()
            
            for i in range(len(self.redbasisvec_penalties)):
                self.Cpen[i,i] = self.redbasisvec_penalties[i]
                
            self.Cpen.assemble()

        te = time.time() - ts
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for ROM. Time: %.4f s" % (te))
            sys.stdout.flush()


    def build_reduced_surface_basis(self, Vspace, rb, ts):

        locmatsize_u = Vspace.dofmap.index_map.size_local * Vspace.dofmap.index_map_bs
        matsize_u = Vspace.dofmap.index_map.size_global * Vspace.dofmap.index_map_bs

        # number of non-reduced "bulk" dofs
        ndof_bulk = matsize_u - len(self.fd_set)

        # row loop to get entries (1's) for "non-reduced" dofs
        nr, a = 0, 0
        row_1, col_1, col_fd = [], [], []

        for row in range(matsize_u):
            
            if row in self.fd_set:
                # increase counter for number of reduced dofs
                nr += 1
                # column shift if we've exceeded the number of reduced basis vectors
                if nr <= rb: col_fd.append(row)
                if nr > rb: a += 1
            else:
                # column id of non-reduced dof (left-shifted by a)
                col_id = row-a
                # store
                row_1.append(row)
                col_1.append(col_id)
        
        # make set for faster checking
        col_fd_set = set(col_fd)

        # create aij matrix - important to specify an approximation for nnz (number of non-zeros per row) for efficient value setting
        self.V = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(rb+ndof_bulk)), bsize=None, nnz=(2*rb,locmatsize_u), csr=None, comm=self.comm)
        self.V.setUp()
        
        vrs, vre = self.V.getOwnershipRange()

        # Phi should not have any non-zero rows that do not belong to a surface dof which is reduced
        for i in range(vrs, vre):
            if i not in self.fd_set:
                assert(np.isclose(np.sum(self.Phi[i,:]), 0.0))

        # now set entries
        for k in range(len(row_1)):
            self.V[row_1[k],col_1[k]] = 1.0

        # column loop to insert columns of Phi
        n=0
        for col in range(rb+ndof_bulk):
            # set Phi column
            if col in col_fd_set:
                self.V[vrs:vre,col] = self.Phi[vrs:vre,n]
                n += 1

        self.V.assemble()
        
        # set penalties
        if bool(self.redbasisvec_penalties):
            
            self.Cpen = PETSc.Mat().createAIJ(size=((rb+ndof_bulk),(rb+ndof_bulk)), bsize=None, nnz=(rb), csr=None, comm=self.comm)
            self.Cpen.setUp()
                
            n=0
            for col in range(rb+ndof_bulk):
                if col in col_fd_set:
                    self.Cpen[col,col] = self.redbasisvec_penalties[n]
                    n += 1
                    
            self.Cpen.assemble()

        te = time.time() - ts
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for surface ROM on boundary id(s) "+str(self.surface_rom)+". Time: %.4f s" % (te))
            sys.stdout.flush()
        
        

