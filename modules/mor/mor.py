#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#import h5py
import time, sys, copy, math
from dolfinx import Function
from dolfinx.fem import assemble_matrix, assemble_vector
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np

#from dolfinx import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, Function, DirichletBC
#from dolfinx.fem import assemble_scalar
#from ufl import TrialFunction, TestFunction, FiniteElement, VectorElement, TensorElement, derivative, diff, det, inner, inv, dx, ds, as_vector, as_ufl, Identity, constantvalue
#from petsc4py import PETSc

#import ioroutines


#from projection import project


#from base import problem_base


class MorBase():

    def __init__(self, params, comm):
        
        self.numhdms = params['numhdms']
        self.numsnapshots = params['numsnapshots']
        self.snapshotincr = params['snapshotincr']
        self.numredbasisvec = params['numredbasisvec']
        
        try: self.print_eigenproblem = params['print_eigenproblem']
        except: self.print_eigenproblem = False
        
        try: self.eigenvalue_cutoff = params['eigenvalue_cutoff']
        except: self.eigenvalue_cutoff = 0.0
        
        # some sanity checks
        if self.numhdms <= 0:
            raise ValueError('Number of HDMs has to be > 0!')
        if self.numsnapshots <= 0:
            raise ValueError('Number of snapshots has to be > 0!')
        if self.snapshotincr <= 0:
            raise ValueError('Snapshot increment has to be > 0!')
        if self.numredbasisvec <= 0 or self.numredbasisvec > self.numhdms*self.numsnapshots:
            raise ValueError('Number of reduced-basis vectors has to be > 0 and <= number of HDMs times number of snapshots!')
        
        self.hdmpath = params['hdmpath']

        self.comm = comm
        
    
    # Proper Orthogonal Decomposition
    def POD(self, pb):
        
        if self.comm.rank==0:
            print("Performing Proper Orthogonal Decomposition (POD) ...")
            sys.stdout.flush()
        
        locmatsize_u, locmatsize_p = pb.V_u.dofmap.index_map.size_local * pb.V_u.dofmap.index_map_bs, pb.V_p.dofmap.index_map.size_local * pb.V_p.dofmap.index_map_bs
        matsize_u, matsize_p = pb.V_u.dofmap.index_map.size_global * pb.V_u.dofmap.index_map_bs, pb.V_p.dofmap.index_map.size_global * pb.V_p.dofmap.index_map_bs

        # snapshot matrix
        S_d = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(self.numhdms*self.numsnapshots)), bsize=None, nnz=None, csr=None, comm=self.comm)
        S_d.setUp()

        # gather snapshots
        S_cols=[]
        for h in range(self.numhdms):
            
            for i in range(self.numsnapshots):
            
                disp = Function(pb.V_u)

                # temporary - we need parallel and multi-core I vs. O (with PETSc viewer, we need the same amount of cores for I as for O!!)
                viewer = PETSc.Viewer().createMPIIO(self.hdmpath+'/checkpoint_romsnaps_u_'+str(i+1)+'_1proc.dat', 'r', self.comm)
                disp.vector.load(viewer)
                
                disp.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                
                S_cols.append(disp.vector)


        # build snapshot matrix S_d
        Sdrow_s, Sdrow_e = S_d.getOwnershipRange()

        for i in range(len(S_cols)):

            for row in range(Sdrow_s, Sdrow_e):
                S_d.setValue(row,i, S_cols[i][row])
        
        S_d.assemble()
        
        # covariance matrix
        C_d = S_d.transposeMatMult(S_d)
        
        eigsolver = SLEPc.EPS()
        eigsolver.create()
        
        eigsolver.setOperators(C_d)
        #eigsolver.setProblemType(SLEPc.EPS.ProblemType.HEP) # TODO: What are the options?
        eigsolver.setFromOptions()
        
        eigsolver.solve()
        
        nconv = eigsolver.getConverged()
        
        if self.print_eigenproblem:
            if self.comm.rank==0:
                print("Number of converged eigenpairs %d" % nconv)
                sys.stdout.flush()

        evecs, evals = [], []
        numredbasisvec_true = 0

        if nconv > 0:
            # create the results vectors
            vr, wr = C_d.getVecs()
            vi, wi = C_d.getVecs()
            
            if self.print_eigenproblem:
                if self.comm.rank==0:
                    print("   k                        ||Ax-kx||/||kx||")
                    print("   ----------------------   ----------------")
                    sys.stdout.flush()
                    
            for i in range(nconv):
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
                evecs.append(vr)
                evals.append(k.real)
                
                if k.real > self.eigenvalue_cutoff: numredbasisvec_true += 1
        
        #if self.numredbasisvec != numredbasisvec_true:
            #if self.comm.rank==0:
                #print("Warning: Near-zero or negative eigenvalues occurred! Number of reduced basis vectors for ROB changed from %i to %i." % (self.numredbasisvec,numredbasisvec_true))
                #sys.stdout.flush()
        
        ## pop out undesired ones
        #for i in range(self.numsnapshots-numredbasisvec_true):
            #evecs.pop(-1)
            #evals.pop(-1)

        # eigenvectors, scaled with 1 / sqrt(eigenval)
        # calculate first numredbasisvec_true POD modes
        y = []
        for i in range(numredbasisvec_true):
            y.append(S_d * evecs[i] / math.sqrt(evals[i]))

        # build reduced basis V
        self.V = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(numredbasisvec_true)), bsize=None, nnz=None, csr=None, comm=self.comm)
        self.V.setUp()
        
        Vrow_s, Vrow_e = self.V.getOwnershipRange()

        for i in range(len(y)):

            for row in range(Vrow_s, Vrow_e):
                self.V.setValue(row,i, y[i][row])

        self.V.assemble()
        
        if self.comm.rank==0:
            print("POD done... Created reduced-order basis for ROM.")
            sys.stdout.flush()


