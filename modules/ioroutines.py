#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
from petsc4py import PETSc
from dolfinx import Function, VectorFunctionSpace
from dolfinx.io import XDMFFile

from projection import project
from ufl import FacetNormal, CellDiameter, dot, sqrt, tr


class IO:
    
    def __init__(self, io_params, comm):

        self.write_results_every = io_params['write_results_every']
        self.output_path = io_params['output_path']
        self.results_to_write = io_params['results_to_write']
        self.simname = io_params['simname']
        
        self.mesh_domain = io_params['mesh_domain']
        self.mesh_boundary = io_params['mesh_boundary']
        
        if 'fiber_data' in io_params.keys(): self.fiber_data = io_params['fiber_data']
        else: self.fiber_data = {}
        
        if 'have_bd1_bcs' in io_params.keys(): self.have_b1_bcs = io_params['have_bd1_bcs']
        else:                                  self.have_b1_bcs = True # most likely!
        if 'have_bd2_bcs' in io_params.keys(): self.have_b2_bcs = io_params['have_bd2_bcs']
        else:                                  self.have_b2_bcs = False
        if 'have_bd3_bcs' in io_params.keys(): self.have_b3_bcs = io_params['have_bd3_bcs']
        else:                                  self.have_b3_bcs = False
        
        self.comm = comm


    def readin_mesh(self):
        
        encoding = XDMFFile.Encoding.ASCII

        # read in xdmf mesh - domain
        with XDMFFile(self.comm, self.mesh_domain, 'r', encoding=encoding) as infile:
            self.mesh = infile.read_mesh(name="Grid")
            self.mt_d = infile.read_meshtags(self.mesh, name="Grid")
        
        # read in xdmf mesh - boundary
        
        # here, we define b1 BCs as BCs associated to a topology one dimension less than the problem,
        # b2 BCs two dimensions less, and b3 BCs three dimensions less
        # for a 3D problem - b1: surface BCs, b2: edge BCs, b3: point BCs
        # for a 2D problem - b1: edge BCs, b2: point BCs
        # 1D problems not supported (currently...)
        
        if self.mesh.topology.dim == 3:
            
            if self.have_b1_bcs:
        
                self.mesh.topology.create_connectivity(2, self.mesh.topology.dim)
                with XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name="Grid_surf")
                    
            if self.have_b2_bcs:
                
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name="Grid_edge")
                    
            if self.have_b3_bcs:
                
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b3 = infile.read_meshtags(self.mesh, name="Grid_point")

        # edge BCs
        elif self.mesh.topology.dim == 2:
            
            if self.have_b1_bcs:

                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name="Grid_edge")
                    
            if self.have_b2_bcs:
                
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name="Grid_point")
                
        else:
            raise AttributeError("Your mesh seems to be 1D! Not supported!")


        # useful fields:
        
        # facet normal
        self.n0 = FacetNormal(self.mesh)
        # cell diameter
        self.h0 = CellDiameter(self.mesh)



class IO_solid(IO):

    # read in fibers defined at nodes (nodal fiber and coordiante files have to be present)
    def readin_fibers(self, fibarray, V_fib, dx_, tol=1.0e-8):

        # V_fib_input is function space the fiber vector is defined on (only CG1 or DG0 supported, add further depending on your input...)
        if list(self.fiber_data.keys())[0] == 'nodal':
            V_fib_input = VectorFunctionSpace(self.mesh, ("CG", 1))
        elif list(self.fiber_data.keys())[0] == 'elemental':
            V_fib_input = VectorFunctionSpace(self.mesh, ("DG", 0))
        else:
            raise AttributeError("Specify 'nodal' or 'elemental' for the fiber data input!")

        fib_func = []
        fib_func_input = []

        si = 0
        for s in fibarray:
            
            fib_func_input.append(Function(V_fib_input, name='Fiber'+str(si+1)+'_input'))
            
            # load fiber and node coordinates data - numnodes x 3 arrays
            fib_data = np.loadtxt(list(self.fiber_data.values())[0][si],usecols=(0,1,2))
            coords = np.loadtxt(list(self.fiber_data.values())[0][si],usecols=(3,4,5))
            
            # new node coordinates (nodes might be re-ordered in parallel)
            # in case of elemental fibers, these are the element centroids
            co = V_fib_input.tabulate_dof_coordinates()

            # index map
            im = V_fib_input.dofmap.index_map.global_indices()

            tolerance = int(-np.log10(tol))

            # since in parallel, the ordering of the node ids might change, so we have to find the
            # mapping between original and new id via the coordinates
            ci = 0
            for i in im:
                
                ind = np.where((np.round(coords,tolerance) == np.round(co[ci],tolerance)).all(axis=1))[0]
                
                # only write if we've found the index - so, we don't need to specify fiber data for subdomains that don't need them
                if len(ind):
                    
                    # normalize fiber vectors (in the case there are some that aren't...)
                    norm_sq = 0.
                    for j in range(3):
                        norm_sq += fib_data[ind[0],j]**2.
                    norm = np.sqrt(norm_sq)
                    
                    for j in range(3):
                        fib_func_input[si].vector[3*i+j] = fib_data[ind[0],j] / norm
                
                ci+=1
            
            # update ghosts
            fib_func_input[si].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            
            # project to output fiber function space
            ff = project(fib_func_input[si], V_fib, dx_, bcs=[], nm='fib_'+s+'')
            
            # assure that projected field still has unit length (not always necessarily the case)
            fib_func.append(ff / sqrt(dot(ff,ff)))

            ## write input fiber field for checking...
            #outfile = XDMFFile(self.comm, self.output_path+'/fiber'+str(si+1)+'_input.xdmf', 'w')
            #outfile.write_mesh(self.mesh)
            #outfile.write_function(fib_func_input[si])

            si+=1

        return fib_func




    def write_output(self, pb=None, writemesh=False, N=0, t=0):
        
        if writemesh and self.write_results_every > 0:
            
            self.resultsfiles = {}
            for res in self.results_to_write:
                outfile = XDMFFile(self.comm, self.output_path+'/results_'+self.simname+'_'+res+'.xdmf', 'w')
                outfile.write_mesh(self.mesh)
                self.resultsfiles[res] = outfile
                
            return
        
        
        else:
        
            # write results every write_results_every steps
            if (N+1) % self.write_results_every == 0:
                
                # save solution to XDMF format
                for res in self.results_to_write:
                    
                    if res=='displacement':
                        self.resultsfiles[res].write_function(pb.u, t)
                    elif res=='velocity': # passed in v is not a function but form, so we have to project
                        v_proj = project(pb.vel, pb.V_u, pb.dx_, nm="Velocity")
                        self.resultsfiles[res].write_function(v_proj, t)
                    elif res=='acceleration': # passed in a is not a function but form, so we have to project
                        a_proj = project(pb.acc, pb.V_u, pb.dx_, nm="Acceleration")
                        self.resultsfiles[res].write_function(a_proj, t)
                    elif res=='pressure':
                        self.resultsfiles[res].write_function(pb.p, t)
                    elif res=='cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.u,pb.p,ivar=pb.internalvars))
                        cauchystress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="CauchyStress")
                        self.resultsfiles[res].write_function(cauchystress, t)
                    elif res=='trmandelstress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(tr(pb.ma[n].M(pb.u,pb.p,ivar=pb.internalvars)))
                        trmandelstress = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="trMandelStress")
                        self.resultsfiles[res].write_function(trmandelstress, t)
                    elif res=='trmandelstress_e':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(tr(pb.ma[n].M_e(pb.u,pb.p,pb.ki.C(pb.u),ivar=pb.internalvars)))
                        trmandelstress_e = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="trMandelStress_e")
                        self.resultsfiles[res].write_function(trmandelstress_e, t)
                    elif res=='vonmises_cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma_vonmises(pb.u,pb.p,ivar=pb.internalvars))
                        vonmises_cauchystress = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="vonMises_CauchyStress")
                        self.resultsfiles[res].write_function(vonmises_cauchystress, t)
                    elif res=='pk1stress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].P(pb.u,pb.p,ivar=pb.internalvars))
                        pk1stress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="PK1Stress")
                        self.resultsfiles[res].write_function(pk1stress, t)
                    elif res=='pk2stress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].S(pb.u,pb.p,ivar=pb.internalvars))
                        pk2stress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="PK2Stress")
                        self.resultsfiles[res].write_function(pk2stress, t)
                    elif res=='jacobian':
                        jacobian = project(pb.ki.J(pb.u), pb.Vd_scalar, pb.dx_, nm="Jacobian")
                        self.resultsfiles[res].write_function(jacobian, t)
                    elif res=='glstrain':
                        glstrain = project(pb.ki.E(pb.u), pb.Vd_tensor, pb.dx_, nm="GreenLagrangeStrain")
                        self.resultsfiles[res].write_function(glstrain, t)
                    elif res=='eastrain':
                        eastrain = project(pb.ki.e(pb.u), pb.Vd_tensor, pb.dx_, nm="EulerAlmansiStrain")
                        self.resultsfiles[res].write_function(eastrain, t)
                    elif res=='fiberstretch':
                        fiberstretch = project(pb.ki.fibstretch(pb.u,pb.fib_func[0]), pb.Vd_scalar, pb.dx_, nm="FiberStretch")
                        self.resultsfiles[res].write_function(fiberstretch, t)
                    elif res=='fiberstretch_e':
                        stretchfuncs=[]
                        for n in range(pb.num_domains):
                            stretchfuncs.append(pb.ma[n].fibstretch_e(pb.ki.C(pb.u),pb.theta,pb.fib_func[0]))
                        fiberstretch_e = project(stretchfuncs, pb.Vd_scalar, pb.dx_, nm="FiberStretch_e")
                        self.resultsfiles[res].write_function(fiberstretch_e, t)
                    elif res=='theta':
                        self.resultsfiles[res].write_function(pb.theta, t)
                    elif res=='phi_remod':
                        phifuncs=[]
                        for n in range(pb.num_domains):
                            phifuncs.append(pb.ma[n].phi_remod(pb.theta))
                        phiremod = project(phifuncs, pb.Vd_scalar, pb.dx_, nm="phiRemodel")
                        self.resultsfiles[res].write_function(phiremod, t)
                    elif res=='tau_a':
                        self.resultsfiles[res].write_function(pb.tau_a, t)
                    elif res=='fiber1':
                        fiber1 = project(pb.fib_func[0], pb.Vd_vector, pb.dx_, nm="Fiber1")
                        #print(fiber1.vector[:])
                        self.resultsfiles[res].write_function(fiber1, t)
                    elif res=='fiber2':
                        fiber2 = project(pb.fib_func[1], pb.Vd_vector, pb.dx_, nm="Fiber2")
                        self.resultsfiles[res].write_function(fiber2, t)
                    else:
                        raise NameError("Unknown output to write for solid mechanics!")


class IO_fluid(IO):
    
    
    def write_output(self, pb=None, writemesh=False, N=0, t=0):
        
        if writemesh and self.write_results_every > 0:
            
            self.resultsfiles = {}
            for res in self.results_to_write:
                outfile = XDMFFile(self.comm, self.output_path+'/results_'+self.simname+'_'+res+'.xdmf', 'w')
                outfile.write_mesh(self.mesh)
                self.resultsfiles[res] = outfile
            
            return
        
        
        else:
        
            # write results every write_results_every steps
            if (N+1) % self.write_results_every == 0:
                
                # save solution to XDMF format
                for res in self.results_to_write:
                    
                    if res=='velocity':
                        self.resultsfiles[res].write_function(pb.v, t)
                    elif res=='acceleration': # passed in a is not a function but form, so we have to project
                        a_proj = project(pb.acc, pb.V_v, pb.dx_, nm="Acceleration")
                        self.resultsfiles[res].write_function(a_proj, t)
                    elif res=='pressure':
                        self.resultsfiles[res].write_function(pb.p, t)
                    elif res=='cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.v,pb.p))
                        cauchystress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="CauchyStress")
                        self.resultsfiles[res].write_function(cauchystress, t)
                    elif res=='reynolds':
                        reynolds = project(re, pb.Vd_scalar, pb.dx_, nm="Reynolds")
                        self.resultsfiles[res].write_function(reynolds, t)
                    else:
                        raise NameError("Unknown output to write for fluid mechanics!")
