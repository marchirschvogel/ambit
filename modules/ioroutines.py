#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, io
import ufl

from projection import project
from mpiroutines import allgather_vec
import expression
from mathutils import spectral_decomposition_3x3


class IO:

    def __init__(self, io_params, comm):

        self.io_params = io_params

        self.write_results_every = io_params['write_results_every']
        self.output_path = io_params['output_path']

        self.mesh_domain = io_params['mesh_domain']
        self.mesh_boundary = io_params['mesh_boundary']

        self.sname = io_params['simname']

        try: self.fiber_data = io_params['fiber_data']
        except: self.fiber_data = []

        try: self.write_restart_every = io_params['write_restart_every']
        except: self.write_restart_every = -1

        try: self.meshfile_type = io_params['meshfile_type']
        except: self.meshfile_type = 'ASCII'

        try: self.gridname_domain = io_params['gridname_domain']
        except: self.gridname_domain = 'Grid'

        try: self.gridname_boundary = io_params['gridname_boundary']
        except: self.gridname_boundary = 'Grid'

        # readin tolerance for own read function that uses coordinate values to properly read in a field
        try: self.readin_tol = io_params['readin_tol']
        except: self.readin_tol = 1.0e-8

        # TODO: Currently, for coupled problems, all append to this dict, so output names should not conflict... hence, make this problem-specific!
        self.resultsfiles = {}

        self.comm = comm


    def readin_mesh(self):

        if self.meshfile_type=='ASCII':
            encoding = io.XDMFFile.Encoding.ASCII
        elif self.meshfile_type=='HDF5':
            encoding = io.XDMFFile.Encoding.HDF5
        else:
            raise NameError('Choose either ASCII or HDF5 as meshfile_type, or add a different encoding!')

        # read in xdmf mesh - domain
        with io.XDMFFile(self.comm, self.mesh_domain, 'r', encoding=encoding) as infile:
            self.mesh = infile.read_mesh(name=self.gridname_domain)
            try: self.mt_d = infile.read_meshtags(self.mesh, name=self.gridname_domain)
            except: self.mt_d = None

        # read in xdmf mesh - boundary

        # here, we define b1 BCs as BCs associated to a topology one dimension less than the problem (most common),
        # b2 BCs two dimensions less, and b3 BCs three dimensions less
        # for a 3D problem - b1: surface BCs, b2: edge BCs, b3: point BCs
        # for a 2D problem - b1: edge BCs, b2: point BCs
        # 1D problems not supported (currently...)

        if self.mesh.topology.dim == 3:

            try:
                self.mesh.topology.create_connectivity(2, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)
            except:
                pass

            try:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name=self.gridname_boundary+'_b2')
            except:
                pass

            try:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b3 = infile.read_meshtags(self.mesh, name=self.gridname_boundary+'_b3')
            except:
                pass

        elif self.mesh.topology.dim == 2:

            try:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)
            except:
                pass

            try:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name=self.gridname_boundary+'_b2')
            except:
                pass

        else:
            raise AttributeError("Your mesh seems to be 1D! Not supported!")

        # some mesh data that we might wanna use in some problems...
        # facet normal
        self.n0 = ufl.FacetNormal(self.mesh)
        # cell diameter
        self.hd0 = ufl.CellDiameter(self.mesh)
        # cell circumradius
        self.ro0 = ufl.Circumradius(self.mesh)
        # min and max cell edge lengths
        self.emin0 = ufl.MinCellEdgeLength(self.mesh)
        self.emax0 = ufl.MaxCellEdgeLength(self.mesh)
        # jacobian determinant
        self.detj0 = ufl.JacobianDeterminant(self.mesh)


    def write_output_pre(self, pb, func, t, name):

        outfile = io.XDMFFile(self.comm, self.output_path+'/results_'+pb.simname+'_'+name+'.xdmf', 'w')
        outfile.write_mesh(self.mesh)
        outfile.write_function(func, t)


    def write_restart(self, pb, N):

        if self.write_restart_every > 0 and N % self.write_restart_every == 0:

            self.writecheckpoint(pb, N)


    def set_func_const(self, func, array):

        load = expression.template_vector()
        load.val_x, load.val_y, load.val_z = array[0], array[1], array[2]
        func.interpolate(load.evaluate)


    def readfunction(self, f, V, datafile, normalize=False, tol=1.0e-8):

        # block size of vector
        bs = f.vector.getBlockSize()

        # load data and coordinates
        data = np.loadtxt(datafile,usecols=(np.arange(0,bs)),ndmin=2)
        coords = np.loadtxt(datafile,usecols=(-3,-2,-1)) # last three always are the coordinates

        # new node coordinates (dofs might be re-ordered in parallel)
        # in case of DG fields, these are the Gauss point coordinates
        co = V.tabulate_dof_coordinates()

        # index map
        #im = V.dofmap.index_map.global_indices() # function seems to have gone!
        im = np.asarray(V.dofmap.index_map.local_to_global(np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32)), dtype=PETSc.IntType)

        tolerance = int(-np.log10(tol))

        # since in parallel, the ordering of the dof ids might change, so we have to find the
        # mapping between original and new id via the coordinates
        ci = 0
        for i in im:

            ind = np.where((np.round(coords,tolerance) == np.round(co[ci],tolerance)).all(axis=1))[0]

            # only write if we've found the index
            if len(ind):

                if normalize:
                    norm_sq = 0.
                    for j in range(bs):
                        norm_sq += data[ind[0],j]**2.
                    norm = np.sqrt(norm_sq)
                else:
                    norm = 1.

                for j in range(bs):
                    f.vector[bs*i+j] = data[ind[0],j] / norm

            ci+=1

        f.vector.assemble()

        # update ghosts
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # read in fibers defined at nodes (nodal fiber-coordiante files have to be present)
    def readin_fibers(self, fibarray, V_fib, dx_, order_disp):

        fib_func_input, fib_func = [], []

        try: self.order_fib_input = self.io_params['order_fib_input']
        except: self.order_fib_input = order_disp

        # define input fiber function space
        V_fib_input = fem.VectorFunctionSpace(self.mesh, ("CG", self.order_fib_input))

        si = 0
        for s in fibarray:

            if isinstance(self.fiber_data[si], str):
                dat = np.loadtxt(self.fiber_data[si])
                if len(dat)!=V_fib_input.dofmap.index_map.size_global:
                    raise RuntimeError("Your order of fiber input data does not match the (assumed) order of the fiber function space, %i. Specify 'order_fib_input' in your IO section." % (self.order_fib_input))

            fib_func_input.append(fem.Function(V_fib_input, name='Fiber'+str(si+1)))

            if isinstance(self.fiber_data[si], str):
                self.readfunction(fib_func_input[si], V_fib_input, self.fiber_data[si], normalize=True, tol=self.readin_tol)
            else: # assume a constant-in-space list or array
                self.set_func_const(fib_func_input[si], self.fiber_data[si])

            # project to fiber function space
            if self.order_fib_input != order_disp:
                fib_func.append( project(fib_func_input[si], V_fib, dx_, nm='Fiber'+str(si+1)) )
            else:
                fib_func.append( fib_func_input[si] )

            # assert that field is actually always normalized!
            fib_func[si] /= ufl.sqrt(ufl.dot(fib_func[si],fib_func[si]))

            si+=1

        return fib_func



class IO_solid(IO):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        if writemesh:

            if self.write_results_every > 0:

                for res in pb.results_to_write:
                    outfile = io.XDMFFile(self.comm, self.output_path+'/results_'+pb.simname+'_'+res+'.xdmf', 'w')
                    outfile.write_mesh(self.mesh)
                    self.resultsfiles[res] = outfile

            return

        else:

            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:

                # save solution to XDMF format
                for res in pb.results_to_write:

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
                            stressfuncs.append(pb.ma[n].sigma(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        cauchystress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="CauchyStress")
                        self.resultsfiles[res].write_function(cauchystress, t)
                    elif res=='cauchystress_nodal':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        cauchystress_nodal = project(stressfuncs, pb.V_tensor, pb.dx_, nm="CauchyStress_nodal")
                        self.resultsfiles[res].write_function(cauchystress_nodal, t)
                    elif res=='cauchystress_principal':
                        stressfuncs_eval = []
                        for n in range(pb.num_domains):
                            evals, _, _ = spectral_decomposition_3x3(pb.ma[n].sigma(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                            stressfuncs_eval.append(ufl.as_vector(evals)) # written as vector
                        cauchystress_principal = project(stressfuncs_eval, pb.Vd_vector, pb.dx_, nm="CauchyStress_princ")
                        self.resultsfiles[res].write_function(cauchystress_principal, t)
                    elif res=='cauchystress_membrane':
                        stressfuncs=[]
                        for n in range(len(pb.bstress)):
                            stressfuncs.append(pb.bstress[n])
                        cauchystress_membrane = project(stressfuncs, pb.Vd_tensor, pb.dbmem, nm="CauchyStress_membrane")
                        self.resultsfiles[res].write_function(cauchystress_membrane, t)
                    elif res=='cauchystress_membrane_principal':
                        stressfuncs=[]
                        for n in range(len(pb.bstress)):
                            evals, _, _ = spectral_decomposition_3x3(pb.bstress[n])
                            stressfuncs.append(ufl.as_vector(evals)) # written as vector
                        self.cauchystress_membrane_principal = project(stressfuncs, pb.Vd_vector, pb.dbmem, nm="CauchyStress_membrane_princ")
                        self.resultsfiles[res].write_function(self.cauchystress_membrane_principal, t)
                    elif res=='trmandelstress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(tr(pb.ma[n].M(pb.u,pb.p,pb.vel,ivar=pb.internalvars)))
                        trmandelstress = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="trMandelStress")
                        self.resultsfiles[res].write_function(trmandelstress, t)
                    elif res=='trmandelstress_e':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            if pb.mat_growth[n]: stressfuncs.append(tr(pb.ma[n].M_e(pb.u,pb.p,pb.vel,pb.ki.C(pb.u),ivar=pb.internalvars)))
                            else: stressfuncs.append(ufl.as_ufl(0))
                        trmandelstress_e = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="trMandelStress_e")
                        self.resultsfiles[res].write_function(trmandelstress_e, t)
                    elif res=='vonmises_cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma_vonmises(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        vonmises_cauchystress = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="vonMises_CauchyStress")
                        self.resultsfiles[res].write_function(vonmises_cauchystress, t)
                    elif res=='pk1stress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].P(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        pk1stress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="PK1Stress")
                        self.resultsfiles[res].write_function(pk1stress, t)
                    elif res=='pk2stress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].S(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        pk2stress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="PK2Stress")
                        self.resultsfiles[res].write_function(pk2stress, t)
                    elif res=='jacobian':
                        jacobian = project(pb.ki.J(pb.u), pb.Vd_scalar, pb.dx_, nm="Jacobian")
                        self.resultsfiles[res].write_function(jacobian, t)
                    elif res=='glstrain':
                        glstrain = project(pb.ki.E(pb.u), pb.Vd_tensor, pb.dx_, nm="GreenLagrangeStrain")
                        self.resultsfiles[res].write_function(glstrain, t)
                    elif res=='glstrain_principal':
                        evals, _, _ = spectral_decomposition_3x3(pb.ki.E(pb.u))
                        evals_gl = ufl.as_vector(evals) # written as vector
                        glstrain_principal = project(evals_gl, pb.Vd_vector, pb.dx_, nm="GreenLagrangeStrain_princ")
                        self.resultsfiles[res].write_function(glstrain_principal, t)
                    elif res=='eastrain':
                        eastrain = project(pb.ki.e(pb.u), pb.Vd_tensor, pb.dx_, nm="EulerAlmansiStrain")
                        self.resultsfiles[res].write_function(eastrain, t)
                    elif res=='eastrain_principal':
                        evals, _, _ = spectral_decomposition_3x3(pb.ki.e(pb.u))
                        evals_ea = ufl.as_vector(evals) # written as vector
                        eastrain_principal = project(evals_gl, pb.Vd_vector, pb.dx_, nm="EulerAlmansiStrain_princ")
                        self.resultsfiles[res].write_function(eastrain_principal, t)
                    elif res=='fiberstretch':
                        fiberstretch = project(pb.ki.fibstretch(pb.u,pb.fib_func[0]), pb.Vd_scalar, pb.dx_, nm="FiberStretch")
                        self.resultsfiles[res].write_function(fiberstretch, t)
                    elif res=='fiberstretch_e':
                        stretchfuncs=[]
                        for n in range(pb.num_domains):
                            if pb.mat_growth[n]: stretchfuncs.append(pb.ma[n].fibstretch_e(pb.ki.C(pb.u),pb.theta,pb.fib_func[0]))
                            else: stretchfuncs.append(ufl.as_ufl(0))
                        fiberstretch_e = project(stretchfuncs, pb.Vd_scalar, pb.dx_, nm="FiberStretch_e")
                        self.resultsfiles[res].write_function(fiberstretch_e, t)
                    elif res=='theta':
                        self.resultsfiles[res].write_function(pb.theta, t)
                    elif res=='phi_remod':
                        phifuncs=[]
                        for n in range(pb.num_domains):
                            if pb.mat_remodel[n]: phifuncs.append(pb.ma[n].phi_remod(pb.theta))
                            else: phifuncs.append(ufl.as_ufl(0))
                        phiremod = project(phifuncs, pb.Vd_scalar, pb.dx_, nm="phiRemodel")
                        self.resultsfiles[res].write_function(phiremod, t)
                    elif res=='tau_a':
                        self.resultsfiles[res].write_function(pb.tau_a, t)
                    elif res=='fibers':
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    else:
                        raise NameError("Unknown output to write for solid mechanics!")


    def readcheckpoint(self, pb, N_rest):

        vecs_to_read = {pb.u : 'u'}
        if pb.incompressible_2field:
            vecs_to_read[pb.p] = 'p'
        if pb.have_growth:
            vecs_to_read[pb.theta] = 'theta'
            vecs_to_read[pb.theta_old] = 'theta'
        if pb.have_active_stress:
            vecs_to_read[pb.tau_a] = 'tau_a'
            vecs_to_read[pb.tau_a_old] = 'tau_a'
            if pb.have_frank_starling:
                vecs_to_read[pb.amp_old] = 'amp_old'
        if pb.u_pre is not None:
            vecs_to_read[pb.u_pre] = 'u_pre'

        if pb.timint != 'static':
            vecs_to_read[pb.u_old] = 'u'
            vecs_to_read[pb.v_old] = 'v_old'
            vecs_to_read[pb.a_old] = 'a_old'
            if pb.incompressible_2field:
                vecs_to_read[pb.p_old] = 'p'

        if pb.problem_type == 'solid_flow0d_multiscale_gandr':
            vecs_to_read[pb.u_set] = 'u_set'
            vecs_to_read[pb.growth_thres] = 'growth_thres'
            if pb.incompressible_2field:
                vecs_to_read[pb.p_set] = 'p_set'
            if pb.have_active_stress:
                vecs_to_read[pb.tau_a_set] = 'tau_a_set'
                if pb.have_frank_starling:
                    vecs_to_read[pb.amp_old_set] = 'amp_old_set'

        for key in vecs_to_read:

            # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
            # and for safety reasons, include the number of cores in the dat file name
            viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'_'+str(self.comm.size)+'proc.dat', 'r', self.comm)
            key.vector.load(viewer)

            key.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def writecheckpoint(self, pb, N):

        vecs_to_write = {pb.u : 'u'}
        if pb.incompressible_2field:
            vecs_to_write[pb.p] = 'p'
        if pb.have_growth:
            vecs_to_write[pb.theta] = 'theta'
        if pb.have_active_stress:
            vecs_to_write[pb.tau_a] = 'tau_a'
            if pb.have_frank_starling:
                vecs_to_write[pb.amp_old] = 'amp_old'
        if pb.u_pre is not None:
            vecs_to_write[pb.u_pre] = 'u_pre'

        if pb.timint != 'static':
            vecs_to_write[pb.v_old] = 'v_old'
            vecs_to_write[pb.a_old] = 'a_old'

        if pb.problem_type == 'solid_flow0d_multiscale_gandr':
            vecs_to_write[pb.u_set] = 'u_set'
            vecs_to_write[pb.growth_thres] = 'growth_thres'
            if pb.incompressible_2field:
                vecs_to_write[pb.p_set] = 'p_set'
            if pb.have_active_stress:
                vecs_to_write[pb.tau_a_set] = 'tau_a_set'
                if pb.have_active_stress:
                    vecs_to_write[pb.amp_old_set] = 'amp_old_set'

        for key in vecs_to_write:

            # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
            # and for safety reasons, include the number of cores in the dat file name
            viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'_'+str(self.comm.size)+'proc.dat', 'w', self.comm)
            key.vector.view(viewer)



class IO_fluid(IO):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        if writemesh:

            if self.write_results_every > 0:

                for res in pb.results_to_write:
                    outfile = io.XDMFFile(self.comm, self.output_path+'/results_'+pb.simname+'_'+res+'.xdmf', 'w')
                    outfile.write_mesh(self.mesh)
                    self.resultsfiles[res] = outfile

            return

        else:

            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:

                # save solution to XDMF format
                for res in pb.results_to_write:

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
                        reynolds = project(pb.Re, pb.Vd_scalar, pb.dx_, nm="Reynolds")
                        self.resultsfiles[res].write_function(reynolds, t)
                    elif res=='fluiddisplacement': # passed in uf is not a function but form, so we have to project
                        uf_proj = project(pb.ufluid, pb.V_v, pb.dx_, nm="FluidDisplacement")
                        self.resultsfiles[res].write_function(uf_proj, t)
                    elif res=='fibers':
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    elif res=='cauchystress_membrane':
                        stressfuncs=[]
                        for n in range(len(pb.bstress)):
                            stressfuncs.append(pb.bstress[n])
                        cauchystress_membrane = project(stressfuncs, pb.Vd_tensor, pb.dbmem, nm="CauchyStress_membrane")
                        self.resultsfiles[res].write_function(cauchystress_membrane, t)
                    else:
                        raise NameError("Unknown output to write for fluid mechanics!")


    def readcheckpoint(self, pb):

        vecs_to_read = {'v' : pb.v}
        vecs_to_read = {'p' : pb.p}
        vecs_to_read = {'v_old' : pb.v_old}
        vecs_to_read = {'a_old' : pb.a_old}
        vecs_to_read = {'p_old' : pb.p_old}

        for key in vecs_to_read:

            # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
            # and for safety reasons, include the number of cores in the dat file name
            viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+key+'_'+str(self.restart_step)+'_'+str(self.comm.size)+'proc.dat', 'r', self.comm)
            vecs_to_read[key].vector.load(viewer)

            vecs_to_read[key].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def writecheckpoint(self, pb, N):

        vecs_to_write = {'v' : pb.v}
        vecs_to_write = {'p' : pb.p}
        vecs_to_write = {'v_old' : pb.v_old}
        vecs_to_write = {'a_old' : pb.a_old}
        vecs_to_write = {'p_old' : pb.p_old}

        for key in vecs_to_write:

            # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
            # and for safety reasons, include the number of cores in the dat file name
            viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+key+'_'+str(N)+'_'+str(self.comm.size)+'proc.dat', 'w', self.comm)



class IO_ale(IO):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        if writemesh:

            if self.write_results_every > 0:

                for res in pb.results_to_write:
                    outfile = io.XDMFFile(self.comm, self.output_path+'/results_'+pb.simname+'_'+res+'.xdmf', 'w')
                    outfile.write_mesh(self.mesh)
                    self.resultsfiles[res] = outfile

            return

        else:

            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:

                # save solution to XDMF format
                for res in pb.results_to_write:

                    if res=='aledisplacement':
                        self.resultsfiles[res].write_function(pb.d, t)
                    elif res=='alevelocity':
                        w_proj = project(pb.wel, pb.V_d, pb.dx_, nm="AleVelocity")
                        self.resultsfiles[res].write_function(w_proj, t)
                    else:
                        raise NameError("Unknown output to write for ALE mechanics!")


    def readcheckpoint(self, pb):

        vecs_to_read = {'u' : pb.d}

        for key in vecs_to_read:

            # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
            # and for safety reasons, include the number of cores in the dat file name
            viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+key+'_'+str(self.restart_step)+'_'+str(self.comm.size)+'proc.dat', 'r', self.comm)
            vecs_to_read[key].vector.load(viewer)

            vecs_to_read[key].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def writecheckpoint(self, pb, N):

        vecs_to_write = {'u' : pb.d}

        for key in vecs_to_write:

            # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
            # and for safety reasons, include the number of cores in the dat file name
            viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+key+'_'+str(N)+'_'+str(self.comm.size)+'proc.dat', 'w', self.comm)


class IO_fluid_ale(IO_fluid,IO_ale):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        IO_fluid.write_output(self, pb.pbf, writemesh=writemesh, N=N, t=t)
        IO_ale.write_output(self, pb.pba, writemesh=writemesh, N=N, t=t)

    def readcheckpoint(self, pb):

        IO_fluid.readcheckpoint(self, pb)
        IO_ale.readcheckpoint(self, pb)

    def writecheckpoint(self, pb, N):

        IO_fluid.writecheckpoint(self, pb, N)
        IO_ale.writecheckpoint(self, pb, N)
