#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#import h5py
import time, sys, copy, math
from dolfinx import Function
from dolfinx.fem import assemble_matrix, assemble_vector, locate_dofs_topological
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

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
        
        # some sanity checks
        if self.numhdms <= 0:
            raise ValueError('Number of HDMs has to be > 0!')
        if self.numsnapshots <= 0:
            raise ValueError('Number of snapshots has to be > 0!')
        if self.snapshotincr <= 0:
            raise ValueError('Snapshot increment has to be > 0!')
        if self.numredbasisvec <= 0 or self.numredbasisvec > self.numhdms*self.numsnapshots:
            raise ValueError('Number of reduced-basis vectors has to be > 0 and <= number of HDMs times number of snapshots!')
        
        self.hdmfilenames = params['hdmfilenames']

        self.comm = comm
        
    
    # Proper Orthogonal Decomposition
    def POD(self, pb):
        
        if self.comm.rank==0:
            print("Performing Proper Orthogonal Decomposition (POD) ...")
            sys.stdout.flush()
        
        locmatsize_u = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs
        matsize_u = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs

        # snapshot matrix
        S_d = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(self.numhdms*self.numsnapshots)), bsize=None, nnz=None, csr=None, comm=self.comm)
        S_d.setUp()

        # gather snapshots (mostly displacements or velocities)
        S_cols=[]
        for h in range(self.numhdms):
            
            for i in range(self.numsnapshots):
                
                step = self.snapshotoffset + (i+1)*self.snapshotincr
            
                field = Function(pb.V_u)
                
                # TODO: Temporary - we need parallel and multi-core I vs. O (with PETSc viewer, we need the same amount of cores for I as for O!!)
                viewer = PETSc.Viewer().createMPIIO(self.hdmfilenames.replace('*',str(step)), 'r', self.comm)
                field.vector.load(viewer)
                
                field.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                
                S_cols.append(field.vector)

        # build snapshot matrix S_d
        Sdrow_s, Sdrow_e = S_d.getOwnershipRange()

        for i in range(len(S_cols)):

            for row in range(Sdrow_s, Sdrow_e):
                S_d.setValue(row,i, S_cols[i][row])
        
        S_d.assemble()
        
        # covariance matrix
        C_d = S_d.transposeMatMult(S_d) # S^T * S
        #D_d = S_d.matTransposeMult(S_d) # S * S^T
        
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
                    
            for i in range(self.numredbasisvec):
                k = eigsolver.getEigenpair(i, vr, vi)
                error = eigsolver.computeError(i)
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
        
        if self.numredbasisvec != numredbasisvec_true:
            if self.comm.rank==0:
                print("Eigenvalues below cutoff tolerance: Number of reduced basis vectors for ROB changed from %i to %i." % (self.numredbasisvec,numredbasisvec_true))
                sys.stdout.flush()
        
        # pop out undesired ones
        for i in range(self.numredbasisvec-numredbasisvec_true):
            evecs.pop(-1)
            evals.pop(-1)

        # eigenvectors, scaled with 1 / sqrt(eigenval)
        # calculate first numredbasisvec_true POD modes
        self.Phi = np.zeros((locmatsize_u, numredbasisvec_true))
        for i in range(numredbasisvec_true):
            self.Phi[:,i] = S_d * evecs[i] / math.sqrt(evals[i])       

        # build reduced basis - either only on designated surfaces or for the whole model
        if bool(self.surface_rom):
            self.build_reduced_surface_basis(pb,numredbasisvec_true)
        else:
            self.build_reduced_basis(pb,numredbasisvec_true)
        

    def build_reduced_basis(self, pb, rb):

        locmatsize_u = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs
        matsize_u = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs

        # create dense matrix directly from Phi array
        self.Vd = PETSc.Mat().createDense(size=((locmatsize_u,matsize_u),(rb)), bsize=None, array=self.Phi, comm=self.comm)
        self.Vd.setUp()
        self.Vd.assemble()
        
        # convert to aij matrix for solver
        self.V = PETSc.Mat()
        self.Vd.convert("aij", out=self.V)
        
        self.V.assemble()
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for ROM.")
            sys.stdout.flush()


    def build_reduced_surface_basis(self, pb, rb):

        locmatsize_u = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs
        matsize_u = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs

        # get boundary dofs which should be reduced
        fd=[]
        for i in range(len(self.surface_rom)):
            
            fdof_indices = locate_dofs_topological(pb.V_u, pb.io.mesh.topology.dim-1, pb.io.mt_b1.indices[pb.io.mt_b1.values == self.surface_rom[i]])
            
            # gather indices
            fdof_indices_gathered = self.comm.allgather(fdof_indices)
            
            # flatten indices from all the processes
            fdof_indices_flat = [item for sublist in fdof_indices_gathered for item in sublist]

            # remove duplicates
            fdof_indices_unique = list(dict.fromkeys(fdof_indices_flat))

            fd.append(fdof_indices_unique)
        
        # flatten list
        fd_flat = [item for sublist in fd for item in sublist]
        
        # remove duplicates
        fd_unique = list(dict.fromkeys(fd_flat))

        # number of surface dofs that get reduced
        ndof_surf = len(fd_unique)
        
        fd_unique_set = set(fd_unique)

        # number of non-reduced "bulk" dofs
        ndof_bulk = matsize_u - ndof_surf
        
        # first, eliminate all rows in Phi that do not belong to a surface dof which is reduced
        for i in range(len(self.Phi)):
            if i not in fd_unique_set: self.Phi[i,:] = 0.
        
        v = np.zeros((locmatsize_u, rb+ndof_bulk))

        nr, a = 0, 0
        for row in range(len(v)):
            
            if row in fd_unique_set:
                
                col_id = row-a

                nr += 1
                if nr <= rb: v[row,col_id] = self.Phi[row,nr-1]
            
            else:
                # determine column id of non-reduced dof
                col_id = row-a
                
                v[row,col_id] = 1.0
                
                # if we reached rb, no further surface columns are needed
                if nr > rb: a += 1
                
        
        #uu=[]
        #for row in range(len(v)):
            #if sum(v[row,:]) == 0: uu.append(row)
        
        #np.set_printoptions(threshold=np.inf)
        
        # create dense matrix directly from v array
        self.Vd = PETSc.Mat().createDense(size=((locmatsize_u,matsize_u),(rb+ndof_bulk)), bsize=None, array=v, comm=self.comm)
        self.Vd.setUp()
        self.Vd.assemble()
        
        # convert to aij matrix for solver
        self.V = PETSc.Mat()
        self.Vd.convert("aij", out=self.V)
        
        self.V.assemble()
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for surface ROM on boundary id(s) "+str(self.surface_rom)+".")
            sys.stdout.flush()
