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
from ufl import FacetNormal, CellDiameter, dot, sqrt, tr

from projection import project
from mpiroutines import allgather_vec


class IO:
    
    def __init__(self, io_params, comm):

        self.write_results_every = io_params['write_results_every']
        self.output_path = io_params['output_path']
        self.results_to_write = io_params['results_to_write']
        self.simname = io_params['simname']
        
        self.mesh_domain = io_params['mesh_domain']
        self.mesh_boundary = io_params['mesh_boundary']
        
        try: self.fiber_data = io_params['fiber_data']
        except: self.fiber_data = {}
        
        try: self.restart_step = io_params['restart_step']
        except: self.restart_step = 0
        
        try: self.write_restart_every = io_params['write_restart_every']
        except: self.write_restart_every = -1
        
        try: self.have_b1_bcs = io_params['have_bd1_bcs']
        except: self.have_b1_bcs = True # most likely!
        try: self.have_b2_bcs = io_params['have_bd2_bcs']
        except: self.have_b2_bcs = False
        try: self.have_b3_bcs = io_params['have_bd3_bcs']
        except: self.have_b3_bcs = False
        
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
    def readin_fibers(self, fibarray, V_fib, dx_):

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
            
            self.readfunction(fib_func_input[si], V_fib_input, list(self.fiber_data.values())[0][si])

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



    def readfunction(self, f, V, datafile):
        
        # block size of vector
        bs = f.vector.getBlockSize()
        
        # load data and coordinates
        data = np.loadtxt(datafile,usecols=(np.arange(0,bs)),ndmin=2)
        coords = np.loadtxt(datafile,usecols=(-3,-2,-1)) # last three always are the coordinates
        
        # new node coordinates (dofs might be re-ordered in parallel)
        # in case of DG fields, these are the Gauss point coordinates
        co = V.tabulate_dof_coordinates()

        # index map
        im = V.dofmap.index_map.global_indices()

        tol = 1.0e-8
        tolerance = int(-np.log10(tol))

        # since in parallel, the ordering of the dof ids might change, so we have to find the
        # mapping between original and new id via the coordinates
        ci = 0
        for i in im:
            
            ind = np.where((np.round(coords,tolerance) == np.round(co[ci],tolerance)).all(axis=1))[0]
            
            # only write if we've found the index
            if len(ind):
                
                for j in range(bs):
                    f.vector[bs*i+j] = data[ind[0],j]
            
            ci+=1

        f.vector.assemble()
        
        # update ghosts
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        

    # TODO and FIXME: Currently only works in serial!!!
    def writefunction(self, f, V, nm, N):

        # block size of vector
        bs = f.vector.getBlockSize()

        co = V.tabulate_dof_coordinates()
        
        # index map
        im = V.dofmap.index_map.global_indices()

        f_sq = allgather_vec(f.vector, self.comm)

        coords_tmp, coords_all = np.zeros((int(f.vector.getSize()/bs),3)), np.zeros((int(f.vector.getSize()/bs),3))

        for i in range(len(co)):
            coords_tmp[im[i],:] = co[i,:]

        coords_arr = self.comm.allgather(coords_tmp)

        for i in range(len(coords_arr)):
            coords_all += coords_arr[i]
        #coords_all = coords_arr[0]

        #print(coords_arr)

        if self.comm.rank == 0:
        
            filename = self.output_path+'/'+self.simname+'_checkpoint_'+nm+'_'+str(N)+'.txt' # conditions at beginning of cycle
            fl = open(filename, 'wt')
            
            for i in range(int(len(f_sq)/bs)):
                for j in range(bs):
                    fl.write('%.16E ' % (f_sq[bs*i+j]))
            
                for k in range(3):
                    fl.write('%.16E ' % (coords_all[i][k]))

                fl.write('\n')
                
            fl.close()

        


    def write_output(self, pb=None, writemesh=False, N=1, t=0):
        
        if writemesh:
            
            if self.write_results_every > 0:
            
                self.resultsfiles = {}
                for res in self.results_to_write:
                    outfile = XDMFFile(self.comm, self.output_path+'/results_'+self.simname+'_'+res+'.xdmf', 'w')
                    outfile.write_mesh(self.mesh)
                    self.resultsfiles[res] = outfile
                
            return
        
        
        else:

            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:
                
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


            if self.write_restart_every > 0 and N % self.write_restart_every == 0:

                self.writecheckpoint(pb, N)


    def readcheckpoint(self, pb):

        self.readfunction(pb.u, pb.V_u, self.output_path+'/'+self.simname+'_checkpoint_u_'+str(self.restart_step)+'.txt')
        if pb.incompressible_2field:
            self.readfunction(pb.p, pb.V_p, self.output_path+'/'+self.simname+'_checkpoint_p_'+str(self.restart_step)+'.txt')
        if pb.have_growth:
            self.readfunction(pb.theta, pb.Vd_scalar, self.output_path+'/'+self.simname+'_checkpoint_theta_'+str(self.restart_step)+'.txt')
            self.readfunction(pb.theta_old, pb.Vd_scalar, self.output_path+'/'+self.simname+'_checkpoint_theta_old_'+str(self.restart_step)+'.txt')
        if pb.have_active_stress:
            self.readfunction(pb.tau_a, pb.Vd_scalar, self.output_path+'/'+self.simname+'_checkpoint_tau_a_'+str(self.restart_step)+'.txt')
            self.readfunction(pb.tau_a_old, pb.Vd_scalar, self.output_path+'/'+self.simname+'_checkpoint_tau_a_old_'+str(self.restart_step)+'.txt')
        if pb.F_hist is not None:
            self.readfunction(pb.F_hist, pb.Vd_tensor, self.output_path+'/'+self.simname+'_checkpoint_F_hist_'+str(self.restart_step)+'.txt')
            self.readfunction(pb.u_pre, pb.V_u, self.output_path+'/'+self.simname+'_checkpoint_u_pre_'+str(self.restart_step)+'.txt')

        if pb.timint != 'static':
            self.readfunction(pb.u_old, pb.V_u, self.output_path+'/'+self.simname+'_checkpoint_u_old_'+str(self.restart_step)+'.txt')
            self.readfunction(pb.v_old, pb.V_u, self.output_path+'/'+self.simname+'_checkpoint_v_old_'+str(self.restart_step)+'.txt')
            self.readfunction(pb.a_old, pb.V_u, self.output_path+'/'+self.simname+'_checkpoint_a_old_'+str(self.restart_step)+'.txt')
            if pb.incompressible_2field:
                self.readfunction(pb.p_old, pb.V_p, self.output_path+'/'+self.simname+'_checkpoint_p_old_'+str(self.restart_step)+'.txt')


    # TODO: Currently only works in serial!!!
    def writecheckpoint(self, pb, N):

        self.writefunction(pb.u, pb.V_u, 'u', N)
        if pb.incompressible_2field:
            self.writefunction(pb.p, pb.V_p, 'p', N)
        if pb.have_growth:
            self.writefunction(pb.theta, pb.Vd_scalar, 'theta', N)
            self.writefunction(pb.theta_old, pb.Vd_scalar, 'theta_old', N)
        if pb.have_active_stress:
            self.writefunction(pb.tau_a, pb.Vd_scalar, 'tau_a', N)
            self.writefunction(pb.tau_a_old, pb.Vd_scalar, 'tau_a_old', N)
        if pb.F_hist is not None:
            self.writefunction(pb.F_hist, pb.Vd_tensor, 'F_hist', N)
            self.writefunction(pb.u_pre, pb.V_u, 'u_pre', N)

        if pb.timint != 'static':
            self.writefunction(pb.u_old, pb.V_u, 'u_old', N)
            self.writefunction(pb.v_old, pb.V_u, 'v_old', N)
            self.writefunction(pb.a_old, pb.V_u, 'a_old', N)
            if pb.incompressible_2field:
                self.writefunction(pb.p_old, pb.V_p, 'p_old', N)


class IO_fluid(IO):
    
    
    def write_output(self, pb=None, writemesh=False, N=1, t=0):
        
        if writemesh:
            
            if self.write_results_every > 0:
            
                self.resultsfiles = {}
                for res in self.results_to_write:
                    outfile = XDMFFile(self.comm, self.output_path+'/results_'+self.simname+'_'+res+'.xdmf', 'w')
                    outfile.write_mesh(self.mesh)
                    self.resultsfiles[res] = outfile
            
            return
        
        
        else:
            
            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:
                
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

            if self.write_restart_every > 0 and N % self.write_restart_every == 0:

                self.writecheckpoint(pb, N)



    def readcheckpoint(self, pb):

        self.readfunction(pb.v, pb.V_v, self.output_path+'/'+self.simname+'_checkpoint_v_'+str(self.restart_step)+'.txt')
        self.readfunction(pb.p, pb.V_p, self.output_path+'/'+self.simname+'_checkpoint_p_'+str(self.restart_step)+'.txt')
        self.readfunction(pb.v_old, pb.V_v, self.output_path+'/'+self.simname+'_checkpoint_v_old_'+str(self.restart_step)+'.txt')
        self.readfunction(pb.a_old, pb.V_v, self.output_path+'/'+self.simname+'_checkpoint_a_old_'+str(self.restart_step)+'.txt')
        self.readfunction(pb.p_old, pb.V_p, self.output_path+'/'+self.simname+'_checkpoint_p_old_'+str(self.restart_step)+'.txt')


    # TODO: Currently only works in serial!!!
    def writecheckpoint(self, pb, N):

        self.writefunction(pb.v, pb.V_v, 'v', N)
        self.writefunction(pb.p, pb.V_p, 'p', N)
