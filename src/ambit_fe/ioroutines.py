#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, io, mesh
import ufl
# import adios4dolfinx

#from solver import projection
from .solver.projection import project
from .mpiroutines import allgather_vec
from . import meshutils, expression, ioparams
from . import mathutils
from .mathutils import spectral_decomposition_3x3


class IO:

    def __init__(self, io_params, entity_maps, comm):

        ioparams.check_params_io(io_params)

        self.io_params = io_params

        self.write_results_every = io_params['write_results_every']
        self.output_path = io_params['output_path']

        try: self.output_path_pre = io_params['output_path_pre']
        except: self.output_path_pre = self.output_path

        self.mesh_domain = io_params['mesh_domain']
        self.mesh_boundary = io_params['mesh_boundary']

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

        try: self.duplicate_mesh_domains = io_params['duplicate_mesh_domains']
        except: self.duplicate_mesh_domains = []

        try: self.restart_io_type = io_params['restart_io_type']
        except: self.restart_io_type = 'petscvector'

        try: self.indicate_results_by = io_params['indicate_results_by']
        except: self.indicate_results_by = 'time'

        try: self.print_enhanced_info = io_params['print_enhanced_info']
        except: self.print_enhanced_info = False

        # TODO: Currently, for coupled problems, all append to this dict, so output names should not conflict... hence, make this problem-specific!
        self.resultsfiles = {}

        # TODO: Should go away once mixed branch has been merged into nightly dolfinx
        try: self.USE_MIXED_DOLFINX_BRANCH = io_params['USE_MIXED_DOLFINX_BRANCH']
        except: self.USE_MIXED_DOLFINX_BRANCH = False

        # entity map dict - for coupled multiphysics/multimesh problems
        self.entity_maps = entity_maps

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

        # mesh degree and scalar, vector, tensor output function spaces
        self.mesh_degree = self.mesh._ufl_domain._ufl_coordinate_element.degree()
        self.V_out_scalar = fem.FunctionSpace(self.mesh, ("CG", self.mesh_degree))
        self.V_out_vector = fem.VectorFunctionSpace(self.mesh, ("CG", self.mesh_degree))
        self.V_out_tensor = fem.TensorFunctionSpace(self.mesh, ("CG", self.mesh_degree))

        # master mesh object (need if fields are actually subdomains, e.g. in FSI)
        self.mesh_master = self.mesh
        self.mt_d_master = self.mt_d

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
                self.mt_b1_master = self.mt_b1
            except:
                pass

            try:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name=self.gridname_boundary+'_b2')
                self.mt_b2_master = self.mt_b2
            except:
                pass

            try:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b3 = infile.read_meshtags(self.mesh, name=self.gridname_boundary+'_b3')
                self.mt_b3_master = self.mt_b3
            except:
                pass

        elif self.mesh.topology.dim == 2:

            try:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)
                self.mt_b1_master = self.mt_b1
            except:
                pass

            try:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, 'r', encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name=self.gridname_boundary+'_b2')
                self.mt_b2_master = self.mt_b2
            except:
                pass

        else:
            raise AttributeError("Your mesh seems to be 1D! Not supported!")


    # some mesh data that we might wanna use in some problems...
    def set_mesh_fields(self, msh):

        # facet normal field
        self.n0 = ufl.FacetNormal(msh)
        # cell diameter
        self.hd0 = ufl.CellDiameter(msh)
        # cell circumradius
        self.ro0 = ufl.Circumradius(msh)
        # min and max cell edge lengths
        self.emin0 = ufl.MinCellEdgeLength(msh)
        self.emax0 = ufl.MaxCellEdgeLength(msh)
        # jacobian determinant
        self.detj0 = ufl.JacobianDeterminant(msh)


    def write_output_pre(self, pb, func, t, name):

        outfile = io.XDMFFile(self.comm, self.output_path_pre+'/results_'+pb.simname+'_'+name+'.xdmf', 'w')
        outfile.write_mesh(self.mesh)
        func_out = fem.Function(self.V_out_vector, name=func.name)
        func_out.interpolate(func)
        outfile.write_function(func_out, t)


    def write_restart(self, pb, N):

        if self.write_restart_every > 0 and N % self.write_restart_every == 0:

            self.writecheckpoint(pb, N)


    def set_func_const_vec(self, func, array):

        load = expression.template_vector()
        load.val_x, load.val_y, load.val_z = array[0], array[1], array[2]
        func.interpolate(load.evaluate)


    # own read function
    def readfunction(self, f, datafile, normalize=False):

        # block size of vector
        bs = f.vector.getBlockSize()

        # load data and input node indices
        data = np.loadtxt(datafile,usecols=(np.arange(1,bs+1)),ndmin=2)
        ind_file = np.loadtxt(datafile,usecols=(0),dtype=int)

        # index map and input indices
        im = np.asarray(f.function_space.dofmap.index_map.local_to_global(np.arange(f.function_space.dofmap.index_map.size_local + f.function_space.dofmap.index_map.num_ghosts, dtype=np.int32)), dtype=PETSc.IntType)
        igi = self.mesh.geometry.input_global_indices

        # since in parallel, the ordering of the dof ids might change, so we have to find the
        # mapping between original and new id via the coordinates
        ci = 0
        for i in im:

            ind = np.where(ind_file == igi[ci])[0]

            # only read if we've found the index
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
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # own write function - working for nodal fields that are defined on the input mesh
    # also works "bottom-down" for lower-order functions defined on a higher-order input mesh
    # (e.g. linear pressure field defined on a quadratic input mesh), but not the other way
    # around (since we would not have all input node ids for a higher-order mesh defined using
    # a lower order input mesh)
    def writefunction(self, f, filenm):

        # non-ghosted index map and input global node indices
        im_no_ghosts = f.function_space.dofmap.index_map.local_to_global(np.arange(f.function_space.dofmap.index_map.size_local, dtype=np.int32)).tolist()
        igi = self.mesh.geometry.input_global_indices

        # gather indices
        im_no_ghosts_gathered = self.comm.allgather(im_no_ghosts)
        igi_gathered = self.comm.allgather(igi)

        # number of present partitions (number of cores we're running on)
        npart = len(im_no_ghosts_gathered)

        # get the (non-ghosted) input node indices of all partitions
        igi_flat = []
        for n in range(npart):
            for i in range(len(im_no_ghosts_gathered[n])):
                igi_flat.append(igi_gathered[n][i])

        # gather PETSc vector
        vec_sq = allgather_vec(f.vector, self.comm)

        sz = f.vector.getSize()
        bs = f.vector.getBlockSize()

        # write to file
        if self.comm.rank==0:
            f = open(filenm, 'wt')
            for i in range(int(sz/bs)):
                f.write(str(igi_flat[i]) + ' ' + ' '.join(map(str, vec_sq[bs*i:bs*(i+1)])) + '\n')
            f.close()

        self.comm.Barrier()


    # read in fibers defined at nodes (nodal fiber-coordiante files have to be present)
    def readin_fibers(self, fibarray, V_fib, dx_, order_disp):

        ts = time.time()
        if self.comm.rank==0:
            print("Reading in fibers ...")
            sys.stdout.flush()

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
                self.readfunction(fib_func_input[si], self.fiber_data[si], normalize=True)
            else: # assume a constant-in-space list or array
                self.set_func_const_vec(fib_func_input[si], self.fiber_data[si])

            # project to fiber function space
            if self.order_fib_input != order_disp:
                fib_func.append( project(fib_func_input[si], V_fib, dx_, nm='Fiber'+str(si+1), comm=self.comm) )
            else:
                fib_func.append( fib_func_input[si] )

            # assert that field is actually always normalized!
            fib_func[si] /= ufl.sqrt(ufl.dot(fib_func[si],fib_func[si]))

            si+=1

        te = time.time() - ts
        if self.comm.rank==0:
            print("Finished fiber read-in. Time: %.4f s" % (te))
            sys.stdout.flush()

        return fib_func



class IO_solid(IO):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        results_pre = ['fibers','counters']

        if self.indicate_results_by=='time':
            indicator = t
        elif self.indicate_results_by=='step':
            indicator = N-1
        else:
            raise ValueError("Unknown indicate_results_by option. Choose 'time' or 'step'.")

        if writemesh:

            if self.write_results_every > 0:

                for res in pb.results_to_write:
                    if res not in results_pre:
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
                        u_out = fem.Function(self.V_out_vector, name=pb.u.name)
                        u_out.interpolate(pb.u)
                        self.resultsfiles[res].write_function(u_out, indicator)
                    elif res=='velocity': # passed in v is not a function but form, so we have to project
                        self.v_proj = project(pb.vel, pb.V_u, pb.dx_, nm="Velocity", comm=self.comm) # class variable for testing
                        v_out = fem.Function(self.V_out_vector, name=self.v_proj.name)
                        v_out.interpolate(self.v_proj)
                        self.resultsfiles[res].write_function(v_out, indicator)
                    elif res=='acceleration': # passed in a is not a function but form, so we have to project
                        self.a_proj = project(pb.acc, pb.V_u, pb.dx_, nm="Acceleration", comm=self.comm) # class variable for testing
                        a_out = fem.Function(self.V_out_vector, name=self.a_proj.name)
                        a_out.interpolate(self.a_proj)
                        self.resultsfiles[res].write_function(a_out, indicator)
                    elif res=='pressure':
                        p_out = fem.Function(self.V_out_scalar, name=pb.p.name)
                        p_out.interpolate(pb.p)
                        self.resultsfiles[res].write_function(p_out, indicator)
                    elif res=='cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        cauchystress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="CauchyStress", comm=self.comm)
                        cauchystress_out = fem.Function(self.V_out_tensor, name=cauchystress.name)
                        cauchystress_out.interpolate(cauchystress)
                        self.resultsfiles[res].write_function(cauchystress_out, indicator)
                    elif res=='cauchystress_nodal':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        cauchystress_nodal = project(stressfuncs, pb.V_tensor, pb.dx_, nm="CauchyStress_nodal", comm=self.comm)
                        cauchystress_nodal_out = fem.Function(self.V_out_tensor, name=cauchystress_nodal.name)
                        cauchystress_nodal_out.interpolate(cauchystress_nodal)
                        self.resultsfiles[res].write_function(cauchystress_nodal_out, indicator)
                    elif res=='cauchystress_principal':
                        stressfuncs_eval = []
                        for n in range(pb.num_domains):
                            evals, _, _ = spectral_decomposition_3x3(pb.ma[n].sigma(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                            stressfuncs_eval.append(ufl.as_vector(evals)) # written as vector
                        cauchystress_principal = project(stressfuncs_eval, pb.Vd_vector, pb.dx_, nm="CauchyStress_princ", comm=self.comm)
                        cauchystress_principal_out = fem.Function(self.V_out_vector, name=cauchystress_principal.name)
                        cauchystress_principal_out.interpolate(cauchystress_principal)
                        self.resultsfiles[res].write_function(cauchystress_principal_out, indicator)
                    elif res=='cauchystress_membrane':
                        stressfuncs=[]
                        for n in range(len(pb.bstress)):
                            stressfuncs.append(pb.bstress[n])
                        cauchystress_membrane = project(stressfuncs, pb.Vd_tensor, pb.dbmem, nm="CauchyStress_membrane", comm=self.comm)
                        cauchystress_membrane_out = fem.Function(self.V_out_tensor, name=cauchystress_membrane.name)
                        cauchystress_membrane_out.interpolate(cauchystress_membrane)
                        self.resultsfiles[res].write_function(cauchystress_membrane_out, indicator)
                    elif res=='cauchystress_membrane_principal':
                        stressfuncs=[]
                        for n in range(len(pb.bstress)):
                            evals, _, _ = spectral_decomposition_3x3(pb.bstress[n])
                            stressfuncs.append(ufl.as_vector(evals)) # written as vector
                        self.cauchystress_membrane_principal = project(stressfuncs, pb.Vd_vector, pb.dbmem, nm="CauchyStress_membrane_princ", comm=self.comm)
                        cauchystress_membrane_principal_out = fem.Function(self.V_out_vector, name=self.cauchystress_membrane_principal.name)
                        cauchystress_membrane_principal_out.interpolate(self.cauchystress_membrane_principal)
                        self.resultsfiles[res].write_function(cauchystress_membrane_principal_out, indicator)
                    elif res=='trmandelstress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(tr(pb.ma[n].M(pb.u,pb.p,pb.vel,ivar=pb.internalvars)))
                        trmandelstress = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="trMandelStress", comm=self.comm)
                        trmandelstress_out = fem.Function(self.V_out_scalar, name=trmandelstress.name)
                        trmandelstress_out.interpolate(trmandelstress)
                        self.resultsfiles[res].write_function(trmandelstress_out, indicator)
                    elif res=='trmandelstress_e':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            if pb.mat_growth[n]: stressfuncs.append(tr(pb.ma[n].M_e(pb.u,pb.p,pb.vel,pb.ki.C(pb.u),ivar=pb.internalvars)))
                            else: stressfuncs.append(ufl.as_ufl(0))
                        trmandelstress_e = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="trMandelStress_e", comm=self.comm)
                        trmandelstress_e_out = fem.Function(self.V_out_scalar, name=trmandelstress_e.name)
                        trmandelstress_e_out.interpolate(trmandelstress_e)
                        self.resultsfiles[res].write_function(trmandelstress_e_out, indicator)
                    elif res=='vonmises_cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma_vonmises(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        vonmises_cauchystress = project(stressfuncs, pb.Vd_scalar, pb.dx_, nm="vonMises_CauchyStress", comm=self.comm)
                        vonmises_cauchystress_out = fem.Function(self.V_out_scalar, name=vonmises_cauchystress.name)
                        vonmises_cauchystress_out.interpolate(vonmises_cauchystress)
                        self.resultsfiles[res].write_function(vonmises_cauchystress_out, indicator)
                    elif res=='pk1stress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].P(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        pk1stress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="PK1Stress", comm=self.comm)
                        pk1stress_out = fem.Function(self.V_out_tensor, name=pk1stress.name)
                        pk1stress_out.interpolate(pk1stress)
                        self.resultsfiles[res].write_function(pk1stress_out, indicator)
                    elif res=='pk2stress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].S(pb.u,pb.p,pb.vel,ivar=pb.internalvars))
                        pk2stress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="PK2Stress", comm=self.comm)
                        pk2stress_out = fem.Function(self.V_out_tensor, name=pk2stress.name)
                        pk2stress_out.interpolate(pk2stress)
                        self.resultsfiles[res].write_function(pk2stress_out, indicator)
                    elif res=='jacobian':
                        jacobian = project(pb.ki.J(pb.u), pb.Vd_scalar, pb.dx_, nm="Jacobian", comm=self.comm)
                        jacobian_out = fem.Function(self.V_out_scalar, name=jacobian.name)
                        jacobian_out.interpolate(jacobian)
                        self.resultsfiles[res].write_function(jacobian_out, indicator)
                    elif res=='glstrain':
                        glstrain = project(pb.ki.E(pb.u), pb.Vd_tensor, pb.dx_, nm="GreenLagrangeStrain", comm=self.comm)
                        glstrain_out = fem.Function(self.V_out_tensor, name=glstrain.name)
                        glstrain_out.interpolate(glstrain)
                        self.resultsfiles[res].write_function(glstrain_out, indicator)
                    elif res=='glstrain_principal':
                        evals, _, _ = spectral_decomposition_3x3(pb.ki.E(pb.u))
                        evals_gl = ufl.as_vector(evals) # written as vector
                        glstrain_principal = project(evals_gl, pb.Vd_vector, pb.dx_, nm="GreenLagrangeStrain_princ", comm=self.comm)
                        glstrain_principal_out = fem.Function(self.V_out_vector, name=glstrain_principal.name)
                        glstrain_principal_out.interpolate(glstrain_principal)
                        self.resultsfiles[res].write_function(glstrain_principal_out, indicator)
                    elif res=='eastrain':
                        eastrain = project(pb.ki.e(pb.u), pb.Vd_tensor, pb.dx_, nm="EulerAlmansiStrain", comm=self.comm)
                        eastrain_out = fem.Function(self.V_out_tensor, name=eastrain.name)
                        eastrain_out.interpolate(eastrain)
                        self.resultsfiles[res].write_function(eastrain_out, indicator)
                    elif res=='eastrain_principal':
                        evals, _, _ = spectral_decomposition_3x3(pb.ki.e(pb.u))
                        evals_ea = ufl.as_vector(evals) # written as vector
                        eastrain_principal = project(evals_gl, pb.Vd_vector, pb.dx_, nm="EulerAlmansiStrain_princ", comm=self.comm)
                        eastrain_principal_out = fem.Function(self.V_out_vector, name=eastrain_principal.name)
                        eastrain_principal_out.interpolate(eastrain_principal)
                        self.resultsfiles[res].write_function(eastrain_principal_out, indicator)
                    elif res=='fiberstretch':
                        fiberstretch = project(pb.ki.fibstretch(pb.u,pb.fib_func[0]), pb.Vd_scalar, pb.dx_, nm="FiberStretch", comm=self.comm)
                        fiberstretch_out = fem.Function(self.V_out_scalar, name=fiberstretch.name)
                        fiberstretch_out.interpolate(fiberstretch)
                        self.resultsfiles[res].write_function(fiberstretch_out, indicator)
                    elif res=='fiberstretch_e':
                        stretchfuncs=[]
                        for n in range(pb.num_domains):
                            if pb.mat_growth[n]: stretchfuncs.append(pb.ma[n].fibstretch_e(pb.ki.C(pb.u),pb.theta,pb.fib_func[0]))
                            else: stretchfuncs.append(ufl.as_ufl(0))
                        fiberstretch_e = project(stretchfuncs, pb.Vd_scalar, pb.dx_, nm="FiberStretch_e", comm=self.comm)
                        fiberstretch_e_out = fem.Function(self.V_out_scalar, name=fiberstretch_e.name)
                        fiberstretch_e_out.interpolate(fiberstretch_e)
                        self.resultsfiles[res].write_function(fiberstretch_e_out, indicator)
                    elif res=='theta':
                        theta_out = fem.Function(self.V_out_scalar, name=pb.theta.name)
                        theta_out.interpolate(pb.theta)
                        self.resultsfiles[res].write_function(theta_out, indicator)
                    elif res=='phi_remod':
                        phifuncs=[]
                        for n in range(pb.num_domains):
                            if pb.mat_remodel[n]: phifuncs.append(pb.ma[n].phi_remod(pb.theta))
                            else: phifuncs.append(ufl.as_ufl(0))
                        phiremod = project(phifuncs, pb.Vd_scalar, pb.dx_, nm="phiRemodel", comm=self.comm)
                        phiremod_out = fem.Function(self.V_out_scalar, name=phiremod.name)
                        phiremod_out.interpolate(phiremod)
                        self.resultsfiles[res].write_function(phiremod_out, indicator)
                    elif res=='tau_a':
                        tau_out = fem.Function(self.V_out_scalar, name=pb.tau_a.name)
                        tau_out.interpolate(pb.tau_a)
                        self.resultsfiles[res].write_function(tau_out, indicator)
                    elif res=='fibers':
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    elif res=='counters':
                        # iteration counters, written by base class
                        pass
                    else:
                        raise NameError("Unknown output to write for solid mechanics!")


    def readcheckpoint(self, pb, N_rest):

        vecs_to_read = {}
        vecs_to_read[pb.u] = 'u'
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

            if self.restart_io_type=='petscvector':
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'_'+str(self.comm.size)+'proc.dat', 'r', self.comm)
                key.vector.load(viewer)
                key.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                viewer.destroy()
            elif self.restart_io_type=='rawtxt': # only working for nodal fields!
                self.readfunction(key, self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'.txt')
            else:
                raise ValueError("Unknown restart_io_type!")


    def writecheckpoint(self, pb, N):

        vecs_to_write = {}
        vecs_to_write[pb.u] = 'u'
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

            if self.restart_io_type=='petscvector':
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'_'+str(self.comm.size)+'proc.dat', 'w', self.comm)
                key.vector.view(viewer)
                viewer.destroy()
            elif self.restart_io_type=='rawtxt': # only working for nodal fields!
                self.writefunction(key, self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'.txt')
            else:
                raise ValueError("Unknown restart_io_type!")


class IO_fluid(IO):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        results_pre = ['fibers','counters']

        if self.indicate_results_by=='time':
            indicator = t
        elif self.indicate_results_by=='step':
            indicator = N-1
        else:
            raise ValueError("Unknown indicate_results_by optin. Choose 'time' or 'step'.")

        if writemesh:

            if self.write_results_every > 0:

                for res in pb.results_to_write:
                    if res not in results_pre:
                        if res=='pressure' and bool(self.duplicate_mesh_domains):
                            for j in self.duplicate_mesh_domains:
                                outfile = io.XDMFFile(self.comm, self.output_path+'/results_'+pb.simname+'_'+res+str(j)+'.xdmf', 'w')
                                outfile.write_mesh(self.submshes_emap[j][0])
                                self.resultsfiles[res+str(j)] = outfile
                        else:
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
                        v_out = fem.Function(self.V_out_vector, name=pb.v.name)
                        v_out.interpolate(pb.v)
                        self.resultsfiles[res].write_function(v_out, indicator)
                    elif res=='acceleration': # passed in a is not a function but form, so we have to project
                        a_proj = project(pb.acc, pb.V_v, pb.dx_, nm="Acceleration", comm=self.comm)
                        a_out = fem.Function(self.V_out_vector, name=a_proj.name)
                        a_out.interpolate(a_proj)
                        self.resultsfiles[res].write_function(a_out, indicator)
                    elif res=='pressure':
                        if bool(self.duplicate_mesh_domains):
                            m=0
                            for j in self.duplicate_mesh_domains:
                                V_out_scalar_sub = fem.FunctionSpace(self.submshes_emap[j][0], ("CG", self.mesh_degree))
                                p_out = fem.Function(V_out_scalar_sub, name=pb.p_[m].name)
                                p_out.interpolate(pb.p_[m])
                                self.resultsfiles[res+str(j)].write_function(p_out, indicator)
                                m+=1
                        else:
                            p_out = fem.Function(self.V_out_scalar, name=pb.p_[0].name)
                            p_out.interpolate(pb.p_[0])
                            self.resultsfiles[res].write_function(p_out, indicator)
                    elif res=='cauchystress':
                        stressfuncs=[]
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.v,pb.p))
                        cauchystress = project(stressfuncs, pb.Vd_tensor, pb.dx_, nm="CauchyStress", comm=self.comm)
                        cauchystress_out = fem.Function(self.V_out_tensor, name=cauchystress.name)
                        cauchystress_out.interpolate(cauchystress)
                        self.resultsfiles[res].write_function(cauchystress_out, indicator)
                    elif res=='fluiddisplacement': # passed in uf is not a function but form, so we have to project
                        uf_proj = project(pb.ufluid, pb.V_v, pb.dx_, nm="FluidDisplacement", comm=self.comm)
                        uf_out = fem.Function(self.V_out_vector, name=uf_proj.name)
                        uf_out.interpolate(uf_proj)
                        self.resultsfiles[res].write_function(uf_out, indicator)
                    elif res=='fibers':
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    elif res=='cauchystress_membrane':
                        stressfuncs=[]
                        for n in range(len(pb.bstress)):
                            stressfuncs.append(pb.bstress[n])
                        cauchystress_membrane = project(stressfuncs, pb.Vd_tensor, pb.dbmem, nm="CauchyStress_membrane", comm=self.comm)
                        cauchystress_membrane_out = fem.Function(self.V_out_tensor, name=cauchystress_membrane.name)
                        cauchystress_membrane_out.interpolate(cauchystress_membrane)
                        self.resultsfiles[res].write_function(cauchystress_membrane_out, indicator)
                    elif res=='counters':
                        # iteration counters, written by base class
                        pass
                    else:
                        raise NameError("Unknown output to write for fluid mechanics!")


    def readcheckpoint(self, pb, N_rest):

        vecs_to_read = {}
        vecs_to_read[pb.v] = 'v'
        vecs_to_read[pb.v_old] = 'v_old'
        vecs_to_read[pb.a_old] = 'a_old'
        vecs_to_read[pb.uf_old] = 'uf_old' # needed for ALE fluid / FSI / FrSI
        if pb.have_active_stress: # for active membrane model (FrSI)
            vecs_to_read[pb.tau_a] = 'tau_a'
            vecs_to_read[pb.tau_a_old] = 'tau_a'

        # pressure may be discontinuous across domains
        if bool(self.duplicate_mesh_domains):
            for mp in self.duplicate_mesh_domains:
                vecs_to_read[pb.p__[mp]] = 'p'+str(mp)
                vecs_to_read[pb.p_old__[mp]] = 'p_old'+str(mp)
        else:
            vecs_to_read[pb.p] = 'p'
            vecs_to_read[pb.p_old] = 'p_old'

        for key in vecs_to_read:

            if self.restart_io_type=='petscvector':
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'_'+str(self.comm.size)+'proc.dat', 'r', self.comm)
                key.vector.load(viewer)
                key.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                viewer.destroy()
            elif self.restart_io_type=='rawtxt': # only working for nodal fields!
                self.readfunction(key, self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'.txt')
            else:
                raise ValueError("Unknown restart_io_type!")


    def writecheckpoint(self, pb, N):

        vecs_to_write = {}
        vecs_to_write[pb.v] = 'v'
        vecs_to_write[pb.v_old] = 'v_old'
        vecs_to_write[pb.a_old] = 'a_old'
        vecs_to_write[pb.uf_old] = 'uf_old' # needed for ALE fluid / FSI / FrSI
        if pb.have_active_stress:
            vecs_to_write[pb.tau_a] = 'tau_a'

        # pressure may be discontinuous across domains
        if bool(self.duplicate_mesh_domains):
            for mp in self.duplicate_mesh_domains:
                vecs_to_write[pb.p__[mp]] = 'p'+str(mp)
                vecs_to_write[pb.p_old__[mp]] = 'p_old'+str(mp)
        else:
            vecs_to_write[pb.p] = 'p'
            vecs_to_write[pb.p_old] = 'p_old'

        for key in vecs_to_write:

            if self.restart_io_type=='petscvector':
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'_'+str(self.comm.size)+'proc.dat', 'w', self.comm)
                key.vector.view(viewer)
                viewer.destroy()
            elif self.restart_io_type=='rawtxt': # only working for nodal fields!
                self.writefunction(key, self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'.txt')
            else:
                raise ValueError("Unknown restart_io_type!")


class IO_ale(IO):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        results_pre = ['counters']

        if self.indicate_results_by=='time':
            indicator = t
        elif self.indicate_results_by=='step':
            indicator = N-1
        else:
            raise ValueError("Unknown indicate_results_by optin. Choose 'time' or 'step'.")

        if writemesh:

            if self.write_results_every > 0:

                for res in pb.results_to_write:
                    if res not in results_pre:
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
                        d_out = fem.Function(self.V_out_vector, name=pb.d.name)
                        d_out.interpolate(pb.d)
                        self.resultsfiles[res].write_function(d_out, indicator)
                    elif res=='alevelocity':
                        w_proj = project(pb.wel, pb.V_d, pb.dx_, nm="AleVelocity", comm=self.comm)
                        w_out = fem.Function(self.V_out_vector, name=w_proj.name)
                        w_out.interpolate(w_proj)
                        self.resultsfiles[res].write_function(w_out, indicator)
                    elif res=='counters':
                        # iteration counters, written by base class
                        pass
                    else:
                        raise NameError("Unknown output to write for ALE mechanics!")


    def readcheckpoint(self, pb, N_rest):

        vecs_to_read = {}
        vecs_to_read[pb.d] = 'd'
        vecs_to_read[pb.d_old] = 'd_old'
        vecs_to_read[pb.w_old] = 'w_old'

        for key in vecs_to_read:

            if self.restart_io_type=='petscvector':
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'_'+str(self.comm.size)+'proc.dat', 'r', self.comm)
                key.vector.load(viewer)
                key.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                viewer.destroy()
            elif self.restart_io_type=='rawtxt': # only working for nodal fields!
                self.readfunction(key, self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_read[key]+'_'+str(N_rest)+'.txt')
            else:
                raise ValueError("Unknown restart_io_type!")


    def writecheckpoint(self, pb, N):

        vecs_to_write = {}
        vecs_to_write[pb.d] = 'd'
        vecs_to_write[pb.d_old] = 'd_old'
        vecs_to_write[pb.w_old] = 'w_old'

        for key in vecs_to_write:

            if self.restart_io_type=='petscvector':
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'_'+str(self.comm.size)+'proc.dat', 'w', self.comm)
                key.vector.view(viewer)
                viewer.destroy()
            elif self.restart_io_type=='rawtxt': # only working for nodal fields!
                self.writefunction(key, self.output_path+'/checkpoint_'+pb.simname+'_'+vecs_to_write[key]+'_'+str(N)+'.txt')
            else:
                raise ValueError("Unknown restart_io_type!")


class IO_fluid_ale(IO_fluid,IO_ale):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        IO_fluid.write_output(self, pb.pbf, writemesh=writemesh, N=N, t=t)
        IO_ale.write_output(self, pb.pba, writemesh=writemesh, N=N, t=t)

    def readcheckpoint(self, pb, N_rest):

        IO_fluid.readcheckpoint(self, pb.pbf, N_rest)
        IO_ale.readcheckpoint(self, pb.pba, N_rest)

    def write_restart(self, pb, N):

        if self.write_restart_every > 0 and N % self.write_restart_every == 0:

            IO_fluid.writecheckpoint(self, pb.pbf, N)
            IO_ale.writecheckpoint(self, pb.pba, N)


class IO_fsi(IO_solid,IO_fluid_ale):

    def write_output(self, pb, writemesh=False, N=1, t=0):

        IO_solid.write_output(self, pb.pbs, writemesh=writemesh, N=N, t=t)
        IO_fluid_ale.write_output(self, pb.pbfa, writemesh=writemesh, N=N, t=t)

    def readcheckpoint(self, pb, N_rest):

        IO_solid.readcheckpoint(self, pb.pbs, N_rest)
        IO_fluid_ale.readcheckpoint(self, pb.pbfa, N_rest)

    def write_restart(self, pb, N):

        if self.write_restart_every > 0 and N % self.write_restart_every == 0:

            IO_solid.writecheckpoint(self, pb.pbs, N)
            IO_fluid_ale.writecheckpoint(self, pb.pbfa, N)

    def create_submeshes(self):

        self.dom_solid, self.dom_fluid, self.surf_interf = self.io_params['domain_ids_solid'], self.io_params['domain_ids_fluid'], self.io_params['surface_ids_interface']

        self.msh_emap_solid = mesh.create_submesh(self.mesh, self.mesh.topology.dim, self.mt_d.indices[self.mt_d.values == self.dom_solid])[0:2]
        self.msh_emap_fluid = mesh.create_submesh(self.mesh, self.mesh.topology.dim, self.mt_d.indices[self.mt_d.values == self.dom_fluid])[0:2]

        # self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
        # self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)

        # TODO: Assert that meshtags start actually from 1 when transferred!
        self.mt_d_solid = meshutils.meshtags_parent_to_child(self.mt_d, self.msh_emap_solid[0], self.msh_emap_solid[1], self.mesh, 'domain')
        self.mt_d_fluid = meshutils.meshtags_parent_to_child(self.mt_d, self.msh_emap_fluid[0], self.msh_emap_fluid[1], self.mesh, 'domain')

        self.mt_b1_solid = meshutils.meshtags_parent_to_child(self.mt_b1, self.msh_emap_solid[0], self.msh_emap_solid[1], self.mesh, 'boundary')
        self.mt_b1_fluid = meshutils.meshtags_parent_to_child(self.mt_b1, self.msh_emap_fluid[0], self.msh_emap_fluid[1], self.mesh, 'boundary')

        self.msh_emap_lm = mesh.create_submesh(self.mesh, self.mesh.topology.dim-1, self.mt_b1.indices[self.mt_b1.values == self.surf_interf])[0:2]
        # self.msh_emap_lm = mesh.create_submesh(self.msh_emap_solid[0], self.msh_emap_solid[0].topology.dim-1, self.mt_b1_solid.indices[self.mt_b1_solid.values == self.surf_interf])[0:2]
        # self.msh_emap_lm = mesh.create_submesh(self.msh_emap_fluid[0], self.msh_emap_fluid[0].topology.dim-1, self.mt_b1_fluid.indices[self.mt_b1_fluid.values == self.surf_interf])[0:2]

        # # needed??
        # self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim-1)
        # self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)

        cell_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        facet_imap = self.mesh.topology.index_map(self.mesh.topology.dim-1)

        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        num_cells = cell_imap.size_local + cell_imap.num_ghosts

        inv_emap_solid = np.full(num_cells, -1)
        inv_emap_solid[self.msh_emap_solid[1]] = np.arange(len(self.msh_emap_solid[1]))
        self.entity_maps[self.msh_emap_solid[0]] = inv_emap_solid

        inv_emap_fluid = np.full(num_cells, -1)
        inv_emap_fluid[self.msh_emap_fluid[1]] = np.arange(len(self.msh_emap_fluid[1]))
        self.entity_maps[self.msh_emap_fluid[0]] = inv_emap_fluid

        inv_emap_lm = np.full(num_facets, -1)
        inv_emap_lm[self.msh_emap_lm[1]] = np.arange(len(self.msh_emap_lm[1]))
        self.entity_maps[self.msh_emap_lm[0]] = inv_emap_lm
        #
        # self.em = {}
        # self.em[self.msh_emap_lm[0]] = inv_emap_lm

        # with io.XDMFFile(self.comm, "sub_tag_solid.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(self.msh_emap_solid[0])
        #     self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
        #     xdmf.write_meshtags(self.mt_b1_solid)
        #
        # with io.XDMFFile(self.comm, "sub_tag_fluid.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(self.msh_emap_fluid[0])
        #     self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
        #     xdmf.write_meshtags(self.mt_b1_fluid)

        # with io.XDMFFile(self.comm, "mesh_lm.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(self.msh_emap_lm[0])
