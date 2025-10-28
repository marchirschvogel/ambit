#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, io, mesh
import ufl
# import adios4dolfinx

from .solver.projection import project
from .mpiroutines import allgather_vec
from . import meshutils, expression, ioparams, utilities
from .mathutils import spectral_decomposition_3x3


class IO:
    def __init__(self, io_params, fem_params, entity_maps, comm):
        ioparams.check_params_io(io_params)

        self.io_params = io_params

        self.write_results_every = io_params["write_results_every"]
        self.output_path = io_params["output_path"]
        self.output_path_pre = io_params.get("output_path_pre", self.output_path)

        self.mesh_domain = io_params["mesh_domain"]
        self.mesh_boundary = io_params.get("mesh_boundary", None)
        self.mesh_edge = io_params.get("mesh_edge", None)
        self.mesh_point = io_params.get("mesh_point", None)

        self.quad_degree = fem_params["quad_degree"]

        self.fiber_data = io_params.get("fiber_data", [])

        self.meshfile_format = io_params.get("meshfile_format", "XDMF")
        self.meshfile_type = io_params.get("meshfile_type", "ASCII")
        self.mesh_dim = io_params.get("mesh_dim", 3)  # actually only needed/used for gmsh read-in
        self.gridname_domain = io_params.get("gridname_domain", "Grid")
        self.gridname_boundary = io_params.get("gridname_boundary", "Grid")
        self.duplicate_mesh_domains = io_params.get("duplicate_mesh_domains", [])

        self.write_restart_every = io_params.get("write_restart_every", -1)
        self.restart_io_type = io_params.get("restart_io_type", "petscvector")
        self.indicate_results_by = io_params.get("indicate_results_by", "time")

        self.print_enhanced_info = io_params.get("print_enhanced_info", False)

        self.write_submeshes = io_params.get("write_submeshes", False)

        # TODO: Currently, for coupled problems, all append to this dict, so output names should not conflict... hence, make this problem-specific!
        self.resultsfiles = {}

        # entity map dict - for coupled multiphysics/multimesh problems
        self.entity_maps = entity_maps

        self.comm = comm

    def readin_mesh(self):
        if self.meshfile_type == "ASCII":
            encoding = io.XDMFFile.Encoding.ASCII
        elif self.meshfile_type == "HDF5":
            encoding = io.XDMFFile.Encoding.HDF5
        else:
            raise NameError("Choose either ASCII or HDF5 as meshfile_type, or add a different encoding!")

        self.mt_d, self.mt_b1, self.mt_b2, self.mt_b3 = None, None, None, None

        if self.meshfile_format == "XDMF":
            # read in xdmf mesh - domain
            with io.XDMFFile(self.comm, self.mesh_domain, "r", encoding=encoding) as infile:
                self.mesh = infile.read_mesh(name=self.gridname_domain)
                try:
                    self.mt_d = infile.read_meshtags(self.mesh, name=self.gridname_domain)
                except:
                    self.mt_d = None

        elif self.meshfile_format == "gmsh":
            # seems that we cannot infer the dimension from the mesh file but have to provide it to the read function...
            self.mesh, self.mt_d, self.mt_b1 = io.gmshio.read_from_msh(self.mesh_domain, self.comm, gdim=self.mesh_dim)[
                0:3
            ]
            assert self.mesh.geometry.dim == self.mesh_dim  # would be weird if this wasn't true...

        else:
            raise NameError("Choose either XDMF or gmsh as meshfile_format!")

        # seems that we need this (at least when locating dofs for DBCs on volume portions)
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

        # read in xdmf mesh - boundary

        # here, we define b1 BCs as BCs associated to a topology one dimension less than the problem (most common),
        # b2 BCs two dimensions less, and b3 BCs three dimensions less
        # for a 3D problem - b1: surface BCs, b2: edge BCs, b3: point BCs
        # for a 2D problem - b1: edge BCs, b2: point BCs
        # 1D problems not supported (currently...)

        if self.mesh.topology.dim == 3:
            if self.mesh_boundary is not None:
                self.mesh.topology.create_connectivity(2, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, "r", encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

            if self.mesh_edge is not None:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_edge, "r", encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

            if self.mesh_point is not None:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_point, "r", encoding=encoding) as infile:
                    self.mt_b3 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

        elif self.mesh.topology.dim == 2:
            if self.mesh_boundary is not None:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, "r", encoding=encoding) as infile:
                    self.mt_b1 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

            if self.mesh_point is not None:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_point, "r", encoding=encoding) as infile:
                    self.mt_b2 = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

        else:
            raise AttributeError("Your mesh seems to be 1D! Not supported!")

    # create domain and boundary integration measures
    def create_integration_measures(self, msh, mt_data):
        # create domain and boundary integration measures
        if mt_data[0] is not None:
            dx = ufl.Measure(
                "dx",
                domain=msh,
                subdomain_data=mt_data[0],
                metadata={"quadrature_degree": self.quad_degree},
            )
        else:
            dx_ = ufl.Measure(
                "dx",
                domain=msh,
                metadata={"quadrature_degree": self.quad_degree},
            )
            dx = lambda a: dx_  # so that we can call dx(1) even without domain meshtags

        if mt_data[1] is not None:
            ds = ufl.Measure(
                "ds",
                domain=msh,
                subdomain_data=mt_data[1],
                metadata={"quadrature_degree": self.quad_degree},
            )
        else:
            ds = None
        if mt_data[2] is not None:
            de = ufl.Measure(
                "ds",
                domain=msh,
                subdomain_data=mt_data[2],
                metadata={"quadrature_degree": self.quad_degree},
            )
        else:
            de = None
        if mt_data[1] is not None:
            dS = ufl.Measure(
                "dS",
                domain=msh,
                subdomain_data=mt_data[1],
                metadata={"quadrature_degree": self.quad_degree},
            )
        else:
            dS = None
        # self.de = self.io.de
        bmeasures = [ds, de, dS]

        return dx, bmeasures

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
        outfile = io.XDMFFile(
            self.comm,
            self.output_path_pre + "/results_" + pb.pbase.simname + "_" + pb.problem_physics + "_" + name + ".xdmf",
            "w",
        )
        outfile.write_mesh(self.mesh)
        func_out = fem.Function(pb.V_out_vector, name=func.name)
        func_out.interpolate(func)
        outfile.write_function(func_out, t)
        outfile.close()

    def write_restart(self, pb, N, force=False):
        if (self.write_restart_every > 0 and N % self.write_restart_every == 0) or force:
            self.writecheckpoint(pb, N)

    def set_func_const_vec(self, func, array):
        load = expression.template_vector()
        load.val_x, load.val_y, load.val_z = array[0], array[1], array[2]
        func.interpolate(load.evaluate)

    # own read function requiting reading from .h5 format and redistributing the field correctly to the PETSc vector
    def readfunction(
        self,
        f,
        datafile,
        tol=1e-6,
        filetype='xdmf_h5'
    ):

        ts = time.time()
        utilities.print_status("Reading file %s..." % (datafile), self.comm, e=" ")

        # get local size and block size of PETSc vector
        sz_loc, bs = f.x.petsc_vec.getLocalSize(), f.x.petsc_vec.getBlockSize()

        # default: when reading data from an .xdmf/.h5 file, we assume that we always have the nodal
        # coordinates that serve as basis for establishing the mapping
        if filetype=='xdmf_h5':

            import h5py
            import xml.etree.ElementTree as ET
            from scipy.spatial import cKDTree

            xtree = ET.parse(datafile)
            xroot = xtree.getroot()

            # typically, at third entry 0 we have the nodes, at 1 the cells, and at 2 the (first) field
            name_geo = "".join(xroot[0][0][0][0].text.rsplit("/", 1)[-1].split())
            name_fld = "".join(xroot[0][0][2][0].text.rsplit("/", 1)[-1].split())

            datafile_h5 = datafile.replace(".xdmf", ".h5")

            hf = h5py.File(datafile_h5, "r")

            nodes0_ = hf[name_geo]
            dim = nodes0_.shape[1]
            nodes0 = np.zeros((len(nodes0_), dim))
            nodes0_.read_direct(nodes0)

            data_ = hf[name_fld]
            if data_.ndim > 1:
                dim_ = data_.shape[1]
                data = np.zeros((len(data_), dim_))
            else:
                data = np.zeros(len(data_))
            data_.read_direct(data)
            if data_.ndim == 1:
                data.resize((len(data_), 1))

            # the reordered and local nodes on a rank
            nodes_reordered = f.function_space.tabulate_dof_coordinates()
            # create index mapping according to distance tree
            tree = cKDTree(nodes0)
            _, mapping_indices = tree.query(nodes_reordered, distance_upper_bound=tol)

        # ATTENTION: This only works when your field to read is ordered according to the original mesh!
        elif filetype=='plaintext' or filetype=='cheart':

            if filetype=='plaintext':
                data = np.loadtxt(datafile,usecols=(np.arange(0,bs)),ndmin=2)
            if filetype=='cheart': # CHeart .D files: skip first row
                data = np.loadtxt(datafile,usecols=(np.arange(0,bs)),ndmin=2,skiprows=1)

            # for discontinuous cell-wise functions, we need the original cell, otherwise the node index
            if f.function_space._ufl_element.is_cellwise_constant():
                mapping_indices = f.function_space.mesh.topology.original_cell_index
            else:
                mapping_indices = f.function_space.mesh.geometry.input_global_indices

        # order data
        data_mapped = data[mapping_indices[: int(sz_loc / bs)]]

        # flatten mapped data - TODO: Check again in case of 2D meshes!
        # data_mapped_flat = data_mapped[:,:f.function_space.mesh.topology.dim].flatten()
        data_mapped_flat = data_mapped[:].flatten()

        # now set values
        f.x.petsc_vec.array[:] = data_mapped_flat[:]

        f.x.petsc_vec.assemble()
        f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)

    # own write function - working for nodal fields that are defined on the input mesh
    # also works "bottom-down" for lower-order functions defined on a higher-order input mesh
    # (e.g. linear pressure field defined on a quadratic input mesh), but not the other way
    # around (since we would not have all input node ids for a higher-order mesh defined using
    # a lower order input mesh)
    def writefunction(self, f, filenm, filetype="plaintext"):
        # non-ghosted index map and input global node indices
        im_no_ghosts = f.function_space.dofmap.index_map.local_to_global(
            np.arange(f.function_space.dofmap.index_map.size_local, dtype=np.int32)
        ).tolist()

        # for discontinuous cell-wise constant functions, we need the original cell, otherwise the node index
        if f.function_space._ufl_element.is_cellwise_constant():
            igi = f.function_space.mesh.topology.original_cell_index
        else:
            igi = f.function_space.mesh.geometry.input_global_indices

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
        vec_sq = allgather_vec(f.x.petsc_vec, self.comm)

        sz = f.x.petsc_vec.getSize()
        bs = f.x.petsc_vec.getBlockSize()
        nb = int(sz / bs)

        # first collect vector as numpy array in sorted manner
        vec_out = np.zeros(sz)
        for i, ind in enumerate(igi_flat):
            vec_out[bs * ind : bs * (ind + 1)] = vec_sq[bs * i : bs * (i + 1)]

        # write to file
        if filetype == "plaintext":
            if self.comm.rank == 0:
                f = open(filenm + ".txt", "wt")
                for i in range(nb):
                    f.write(" ".join(map(str, vec_out[bs * i : bs * (i + 1)])) + "\n")
                f.close()
        elif filetype == "cheart":  # CHeart .D file
            if self.comm.rank == 0:
                f = open(filenm + ".D", "wt")
                f.write(str(nb) + " " + str(bs) + "\n")
                for i in range(nb):
                    f.write(" ".join(map(str, vec_out[bs * i : bs * (i + 1)])) + "\n")
                f.close()
        else:
            raise ValueError("Unknown filetype!")

        self.comm.Barrier()

    # read in fibers defined at nodes (nodal fiber-coordiante files have to be present)
    def readin_fibers(self, fibarray, V_fib, dx_, domids, order_disp):

        fib_func_input, fib_func = [], []

        self.order_fib_input = self.io_params.get("order_fib_input", order_disp)

        # define input fiber function space
        V_fib_input = fem.functionspace(
            self.mesh,
            ("Lagrange", self.order_fib_input, (self.mesh.geometry.dim,)),
        )

        si = 0
        for s in fibarray:
            # if isinstance(self.fiber_data[si], str):
            #     dat = np.loadtxt(self.fiber_data[si])
            #     if len(dat) != V_fib_input.dofmap.index_map.size_global:
            #         raise RuntimeError(
            #             "Your order of fiber input data does not match the (assumed) order of the fiber function space, %i. Specify 'order_fib_input' in your IO section."
            #             % (self.order_fib_input)
            #         )

            fib_func_input.append(fem.Function(V_fib_input, name="Fiber" + str(si + 1)))

            if isinstance(self.fiber_data[si], str):
                self.readfunction(fib_func_input[si], self.fiber_data[si])
            else:  # assume a constant-in-space list or array
                self.set_func_const_vec(fib_func_input[si], self.fiber_data[si])

            # project to fiber function space
            if self.order_fib_input != order_disp:
                fib_func.append(
                    project(
                        fib_func_input[si],
                        V_fib,
                        dx_,
                        domids=domids,
                        nm="Fiber" + str(si + 1),
                        comm=self.comm,
                        entity_maps=self.entity_maps,
                    )
                )
            else:
                fib_func.append(fib_func_input[si])

            # assert that field is actually always normalized!
            fib_func[si] /= ufl.sqrt(ufl.dot(fib_func[si], fib_func[si]))

            si += 1

        return fib_func

    def close_output_files(self, pb):
        if self.write_results_every > 0:
            for res in pb.results_to_write:
                if res not in self.results_pre:
                    self.resultsfiles[res].close()

    def export_matrix(self, mat, fname):
        viewer = PETSc.Viewer().create(self.comm)
        viewer.setType(PETSc.Viewer.Type.ASCII)
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        viewer.setFileMode("w")
        viewer.setFileName(self.output_path + "/" + fname)
        mat.view(viewer=viewer)
        viewer.destroy()


class IO_solid(IO):
    def write_output(self, pb, writemesh=False, N=1, t=0):
        self.results_pre = ["fibers", "counters"]

        if self.indicate_results_by == "time":
            indicator = t
        elif self.indicate_results_by == "step":
            indicator = N
        elif self.indicate_results_by == "step0":
            if self.write_results_every > 0:
                indicator = int(N / self.write_results_every) - 1
            else:
                indicator = 0
        else:
            raise ValueError("Unknown indicate_results_by option. Choose 'time' or 'step'.")

        if writemesh:
            if self.write_results_every > 0:
                for res in pb.results_to_write:
                    if res not in self.results_pre:
                        outfile = io.XDMFFile(
                            self.comm,
                            self.output_path
                            + "/results_"
                            + pb.pbase.simname
                            + "_"
                            + pb.problem_physics
                            + "_"
                            + res
                            + ".xdmf",
                            "w",
                        )
                        outfile.write_mesh(self.mesh)
                        self.resultsfiles[res] = outfile

            return

        else:
            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:
                # save solution to XDMF format
                for res in pb.results_to_write:
                    if res == "displacement":
                        u_out = fem.Function(pb.V_out_vector, name=pb.u.name)
                        u_out.interpolate(pb.u)
                        self.resultsfiles[res].write_function(u_out, indicator)
                    elif res == "velocity":  # passed in v is not a function but form, so we have to project
                        self.v_proj = project(
                            pb.vel,
                            pb.V_u,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="Velocity",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )  # class variable for testing
                        v_out = fem.Function(pb.V_out_vector, name=self.v_proj.name)
                        v_out.interpolate(self.v_proj)
                        self.resultsfiles[res].write_function(v_out, indicator)
                    elif res == "acceleration":  # passed in a is not a function but form, so we have to project
                        self.a_proj = project(
                            pb.acc,
                            pb.V_u,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="Acceleration",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )  # class variable for testing
                        a_out = fem.Function(pb.V_out_vector, name=self.a_proj.name)
                        a_out.interpolate(self.a_proj)
                        self.resultsfiles[res].write_function(a_out, indicator)
                    elif res == "pressure":
                        p_out = fem.Function(pb.V_out_scalar, name=pb.p.name)
                        p_out.interpolate(pb.p)
                        self.resultsfiles[res].write_function(p_out, indicator)
                    elif res == "cauchystress":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.u, pb.p, pb.vel, ivar=pb.internalvars))
                        cauchystress = project(
                            stressfuncs,
                            pb.Vd_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="CauchyStress",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_out = fem.Function(pb.V_out_tensor, name=cauchystress.name)
                        cauchystress_out.interpolate(cauchystress)
                        self.resultsfiles[res].write_function(cauchystress_out, indicator)
                    elif res == "cauchystress_nodal":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.u, pb.p, pb.vel, ivar=pb.internalvars))
                        cauchystress_nodal = project(
                            stressfuncs,
                            pb.V_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="CauchyStress_nodal",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_nodal_out = fem.Function(pb.V_out_tensor, name=cauchystress_nodal.name)
                        cauchystress_nodal_out.interpolate(cauchystress_nodal)
                        self.resultsfiles[res].write_function(cauchystress_nodal_out, indicator)
                    elif res == "cauchystress_principal":
                        stressfuncs_eval = []
                        for n in range(pb.num_domains):
                            evals, _, _ = spectral_decomposition_3x3(
                                pb.ma[n].sigma(pb.u, pb.p, pb.vel, ivar=pb.internalvars)
                            )
                            stressfuncs_eval.append(ufl.as_vector(evals))  # written as vector
                        cauchystress_principal = project(
                            stressfuncs_eval,
                            pb.Vd_vector,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="CauchyStress_princ",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_principal_out = fem.Function(pb.V_out_vector, name=cauchystress_principal.name)
                        cauchystress_principal_out.interpolate(cauchystress_principal)
                        self.resultsfiles[res].write_function(cauchystress_principal_out, indicator)
                    elif res == "cauchystress_membrane":
                        stressfuncs = []
                        for n in range(len(pb.bstress)):
                            stressfuncs.append(pb.bstress[n])
                        cauchystress_membrane = project(
                            stressfuncs,
                            pb.Vd_tensor,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="CauchyStress_membrane",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_membrane_out = fem.Function(pb.V_out_tensor, name=cauchystress_membrane.name)
                        cauchystress_membrane_out.interpolate(cauchystress_membrane)
                        self.resultsfiles[res].write_function(cauchystress_membrane_out, indicator)
                    elif res == "cauchystress_membrane_principal":
                        stressfuncs = []
                        for n in range(len(pb.bstress)):
                            evals, _, _ = spectral_decomposition_3x3(pb.bstress[n])
                            stressfuncs.append(ufl.as_vector(evals))  # written as vector
                        self.cauchystress_membrane_principal = project(
                            stressfuncs,
                            pb.Vd_vector,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="CauchyStress_membrane_princ",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_membrane_principal_out = fem.Function(
                            pb.V_out_vector,
                            name=self.cauchystress_membrane_principal.name,
                        )
                        cauchystress_membrane_principal_out.interpolate(self.cauchystress_membrane_principal)
                        self.resultsfiles[res].write_function(cauchystress_membrane_principal_out, indicator)
                    elif res == "strainenergy_membrane":
                        sefuncs = []
                        for n in range(len(pb.bstrainenergy)):
                            sefuncs.append(pb.bstrainenergy[n])
                        strainenergy_membrane = project(
                            sefuncs,
                            pb.Vd_scalar,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="StrainEnergy_membrane",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        strainenergy_membrane_out = fem.Function(pb.V_out_scalar, name=strainenergy_membrane.name)
                        strainenergy_membrane_out.interpolate(strainenergy_membrane)
                        self.resultsfiles[res].write_function(strainenergy_membrane_out, indicator)
                    elif res == "internalpower_membrane":
                        pwfuncs = []
                        for n in range(len(pb.bintpower)):
                            pwfuncs.append(pb.bintpower[n])
                        internalpower_membrane = project(
                            pwfuncs,
                            pb.Vd_scalar,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="InternalPower_membrane",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        internalpower_membrane_out = fem.Function(pb.V_out_scalar, name=internalpower_membrane.name)
                        internalpower_membrane_out.interpolate(internalpower_membrane)
                        self.resultsfiles[res].write_function(internalpower_membrane_out, indicator)
                    elif res == "trmandelstress":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(
                                tr(
                                    pb.ma[n].M(
                                        pb.u,
                                        pb.p,
                                        pb.vel,
                                        ivar=pb.internalvars,
                                    )
                                )
                            )
                        trmandelstress = project(
                            stressfuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="trMandelStress",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        trmandelstress_out = fem.Function(pb.V_out_scalar, name=trmandelstress.name)
                        trmandelstress_out.interpolate(trmandelstress)
                        self.resultsfiles[res].write_function(trmandelstress_out, indicator)
                    elif res == "trmandelstress_e":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            if pb.mat_growth[n]:
                                stressfuncs.append(
                                    tr(
                                        pb.ma[n].M_e(
                                            pb.u,
                                            pb.p,
                                            pb.vel,
                                            pb.ki.C(pb.u),
                                            ivar=pb.internalvars,
                                        )
                                    )
                                )
                            else:
                                stressfuncs.append(ufl.as_ufl(0))
                        trmandelstress_e = project(
                            stressfuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="trMandelStress_e",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        trmandelstress_e_out = fem.Function(pb.V_out_scalar, name=trmandelstress_e.name)
                        trmandelstress_e_out.interpolate(trmandelstress_e)
                        self.resultsfiles[res].write_function(trmandelstress_e_out, indicator)
                    elif res == "vonmises_cauchystress":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma_vonmises(pb.u, pb.p, pb.vel, ivar=pb.internalvars))
                        vonmises_cauchystress = project(
                            stressfuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="vonMises_CauchyStress",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        vonmises_cauchystress_out = fem.Function(pb.V_out_scalar, name=vonmises_cauchystress.name)
                        vonmises_cauchystress_out.interpolate(vonmises_cauchystress)
                        self.resultsfiles[res].write_function(vonmises_cauchystress_out, indicator)
                    elif res == "pk1stress":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].P(pb.u, pb.p, pb.vel, ivar=pb.internalvars))
                        pk1stress = project(
                            stressfuncs,
                            pb.Vd_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="PK1Stress",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        pk1stress_out = fem.Function(pb.V_out_tensor, name=pk1stress.name)
                        pk1stress_out.interpolate(pk1stress)
                        self.resultsfiles[res].write_function(pk1stress_out, indicator)
                    elif res == "pk2stress":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].S(pb.u, pb.p, pb.vel, ivar=pb.internalvars))
                        pk2stress = project(
                            stressfuncs,
                            pb.Vd_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="PK2Stress",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        pk2stress_out = fem.Function(pb.V_out_tensor, name=pk2stress.name)
                        pk2stress_out.interpolate(pk2stress)
                        self.resultsfiles[res].write_function(pk2stress_out, indicator)
                    elif res == "jacobian":
                        jacobian = project(
                            pb.ki.J(pb.u),
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="Jacobian",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        jacobian_out = fem.Function(pb.V_out_scalar, name=jacobian.name)
                        jacobian_out.interpolate(jacobian)
                        self.resultsfiles[res].write_function(jacobian_out, indicator)
                    elif res == "glstrain":
                        glstrain = project(
                            pb.ki.E(pb.u),
                            pb.Vd_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="GreenLagrangeStrain",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        glstrain_out = fem.Function(pb.V_out_tensor, name=glstrain.name)
                        glstrain_out.interpolate(glstrain)
                        self.resultsfiles[res].write_function(glstrain_out, indicator)
                    elif res == "glstrain_principal":
                        evals, _, _ = spectral_decomposition_3x3(pb.ki.E(pb.u))
                        evals_gl = ufl.as_vector(evals)  # written as vector
                        glstrain_principal = project(
                            evals_gl,
                            pb.Vd_vector,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="GreenLagrangeStrain_princ",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        glstrain_principal_out = fem.Function(pb.V_out_vector, name=glstrain_principal.name)
                        glstrain_principal_out.interpolate(glstrain_principal)
                        self.resultsfiles[res].write_function(glstrain_principal_out, indicator)
                    elif res == "eastrain":
                        eastrain = project(
                            pb.ki.e(pb.u),
                            pb.Vd_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="EulerAlmansiStrain",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        eastrain_out = fem.Function(pb.V_out_tensor, name=eastrain.name)
                        eastrain_out.interpolate(eastrain)
                        self.resultsfiles[res].write_function(eastrain_out, indicator)
                    elif res == "eastrain_principal":
                        evals, _, _ = spectral_decomposition_3x3(pb.ki.e(pb.u))
                        evals_ea = ufl.as_vector(evals)  # written as vector
                        eastrain_principal = project(
                            evals_gl,
                            pb.Vd_vector,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="EulerAlmansiStrain_princ",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        eastrain_principal_out = fem.Function(pb.V_out_vector, name=eastrain_principal.name)
                        eastrain_principal_out.interpolate(eastrain_principal)
                        self.resultsfiles[res].write_function(eastrain_principal_out, indicator)
                    elif res == "strainenergy":
                        sefuncs = []
                        for n in range(pb.num_domains):
                            sefuncs.append(
                                pb.ma[n].S(
                                    pb.u,
                                    pb.p,
                                    pb.vel,
                                    ivar=pb.internalvars,
                                    returnquantity="strainenergy",
                                )
                            )
                        se = project(
                            sefuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="StrainEnergy",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        se_out = fem.Function(pb.V_out_scalar, name=se.name)
                        se_out.interpolate(se)
                        self.resultsfiles[res].write_function(se_out, indicator)
                    elif res == "internalpower":
                        pwfuncs = []
                        for n in range(pb.num_domains):
                            pwfuncs.append(
                                ufl.inner(
                                    pb.ma[n].S(
                                        pb.u,
                                        pb.p,
                                        pb.vel,
                                        ivar=pb.internalvars,
                                    ),
                                    pb.ki.Edot(pb.u, pb.vel),
                                )
                            )
                        pw = project(
                            pwfuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="InternalPower",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        pw_out = fem.Function(pb.V_out_scalar, name=pw.name)
                        pw_out.interpolate(pw)
                        self.resultsfiles[res].write_function(pw_out, indicator)
                    elif res == "fiberstretch":
                        fiberstretch = project(
                            pb.ki.fibstretch(pb.u, pb.fib_func[0]),
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="FiberStretch",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        fiberstretch_out = fem.Function(pb.V_out_scalar, name=fiberstretch.name)
                        fiberstretch_out.interpolate(fiberstretch)
                        self.resultsfiles[res].write_function(fiberstretch_out, indicator)
                    elif res == "fiberstretch_e":
                        stretchfuncs = []
                        for n in range(pb.num_domains):
                            if pb.mat_growth[n]:
                                stretchfuncs.append(pb.ma[n].fibstretch_e(pb.ki.C(pb.u), pb.theta, pb.fib_func[0]))
                            else:
                                stretchfuncs.append(ufl.as_ufl(0))
                        fiberstretch_e = project(
                            stretchfuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="FiberStretch_e",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        fiberstretch_e_out = fem.Function(pb.V_out_scalar, name=fiberstretch_e.name)
                        fiberstretch_e_out.interpolate(fiberstretch_e)
                        self.resultsfiles[res].write_function(fiberstretch_e_out, indicator)
                    elif res == "theta":
                        theta_out = fem.Function(pb.V_out_scalar, name=pb.theta.name)
                        theta_out.interpolate(pb.theta)
                        self.resultsfiles[res].write_function(theta_out, indicator)
                    elif res == "phi_remod":
                        phifuncs = []
                        for n in range(pb.num_domains):
                            if pb.mat_remodel[n]:
                                phifuncs.append(pb.ma[n].phi_remod(pb.theta))
                            else:
                                phifuncs.append(ufl.as_ufl(0))
                        phiremod = project(
                            phifuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="phiRemodel",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        phiremod_out = fem.Function(pb.V_out_scalar, name=phiremod.name)
                        phiremod_out.interpolate(phiremod)
                        self.resultsfiles[res].write_function(phiremod_out, indicator)
                    elif res == "tau_a":
                        tau_out = fem.Function(pb.V_out_scalar, name=pb.tau_a.name)
                        tau_out.interpolate(pb.tau_a)
                        self.resultsfiles[res].write_function(tau_out, indicator)
                    elif res == "fibers":
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    elif res == "counters":
                        # iteration counters, written by base class
                        pass
                    else:
                        raise NameError("Unknown output to write for solid mechanics!")

    def readcheckpoint(self, pb, N_rest):
        vecs_to_read = {}
        vecs_to_read[pb.u] = "u"
        if pb.incompressible_2field:
            vecs_to_read[pb.p] = "p"
        if pb.have_growth:
            vecs_to_read[pb.theta] = "theta"
            vecs_to_read[pb.theta_old] = "theta"
        if any(pb.mat_active_stress):
            vecs_to_read[pb.tau_a] = "tau_a"
            vecs_to_read[pb.tau_a_old] = "tau_a"
            if pb.have_frank_starling:
                vecs_to_read[pb.amp_old] = "amp_old"
        if pb.u_pre is not None:
            vecs_to_read[pb.u_pre] = "u_pre"

        if pb.timint != "static":
            vecs_to_read[pb.u_old] = "u"
            vecs_to_read[pb.v_old] = "v_old"
            vecs_to_read[pb.a_old] = "a_old"
            if pb.incompressible_2field:
                vecs_to_read[pb.p_old] = "p"

        if pb.pbase.problem_type == "solid_flow0d_multiscale_gandr":
            vecs_to_read[pb.u_set] = "u_set"
            vecs_to_read[pb.growth_thres] = "growth_thres"
            if pb.incompressible_2field:
                vecs_to_read[pb.p_set] = "p_set"
            if any(pb.mat_active_stress):
                vecs_to_read[pb.tau_a_set] = "tau_a_set"
                if pb.have_frank_starling:
                    vecs_to_read[pb.amp_old_set] = "amp_old_set"

        for key in vecs_to_read:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "r",
                    self.comm,
                )
                key.x.petsc_vec.load(viewer)
                key.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.readfunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def writecheckpoint(self, pb, N):
        vecs_to_write = {}
        vecs_to_write[pb.u] = "u"
        if pb.incompressible_2field:
            vecs_to_write[pb.p] = "p"
        if pb.have_growth:
            vecs_to_write[pb.theta] = "theta"
        if any(pb.mat_active_stress):
            vecs_to_write[pb.tau_a] = "tau_a"
            if pb.have_frank_starling:
                vecs_to_write[pb.amp_old] = "amp_old"
        if pb.u_pre is not None:
            vecs_to_write[pb.u_pre] = "u_pre"

        if pb.timint != "static":
            vecs_to_write[pb.v_old] = "v_old"
            vecs_to_write[pb.a_old] = "a_old"

        if pb.pbase.problem_type == "solid_flow0d_multiscale_gandr":
            vecs_to_write[pb.u_set] = "u_set"
            vecs_to_write[pb.growth_thres] = "growth_thres"
            if pb.incompressible_2field:
                vecs_to_write[pb.p_set] = "p_set"
            if any(pb.mat_active_stress):
                vecs_to_write[pb.tau_a_set] = "tau_a_set"
                if pb.have_frank_starling:
                    vecs_to_write[pb.amp_old_set] = "amp_old_set"

        for key in vecs_to_write:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "w",
                    self.comm,
                )
                key.x.petsc_vec.view(viewer)
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.writefunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N),
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")


class IO_fluid(IO):
    def write_output(self, pb, writemesh=False, N=1, t=0):
        self.results_pre = ["fibers", "counters"]

        if self.indicate_results_by == "time":
            indicator = t
        elif self.indicate_results_by == "step":
            indicator = N
        elif self.indicate_results_by == "step0":
            if self.write_results_every > 0:
                indicator = int(N / self.write_results_every) - 1
            else:
                indicator = 0
        else:
            raise ValueError("Unknown indicate_results_by optin. Choose 'time' or 'step'.")

        if writemesh:
            if self.write_results_every > 0:
                for res in pb.results_to_write:
                    if res not in self.results_pre:
                        if res == "pressure" and bool(self.duplicate_mesh_domains):
                            for m, mp in enumerate(self.duplicate_mesh_domains):
                                outfile = io.XDMFFile(
                                    self.comm,
                                    self.output_path
                                    + "/results_"
                                    + pb.pbase.simname
                                    + "_"
                                    + pb.problem_physics
                                    + "_"
                                    + res
                                    + str(m + 1)
                                    + ".xdmf",
                                    "w",
                                )
                                outfile.write_mesh(self.submshes_emap[m + 1][0])
                                self.resultsfiles[res + str(m + 1)] = outfile
                        else:
                            outfile = io.XDMFFile(
                                self.comm,
                                self.output_path
                                + "/results_"
                                + pb.pbase.simname
                                + "_"
                                + pb.problem_physics
                                + "_"
                                + res
                                + ".xdmf",
                                "w",
                            )
                            outfile.write_mesh(self.mesh)
                            self.resultsfiles[res] = outfile

            return

        else:
            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:
                # save solution to XDMF format
                for res in pb.results_to_write:
                    if res == "velocity":
                        v_out = fem.Function(pb.V_out_vector, name=pb.v.name)
                        v_out.interpolate(pb.v)
                        self.resultsfiles[res].write_function(v_out, indicator)
                    elif res == "acceleration":  # passed in a is not a function but form, so we have to project
                        a_proj = project(
                            pb.acc,
                            pb.V_v,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="Acceleration",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        a_out = fem.Function(pb.V_out_vector, name=a_proj.name)
                        a_out.interpolate(a_proj)
                        self.resultsfiles[res].write_function(a_out, indicator)
                    elif res == "pressure":
                        if bool(self.duplicate_mesh_domains):
                            for m, mp in enumerate(self.duplicate_mesh_domains):
                                V_out_scalar_sub = fem.functionspace(
                                    self.submshes_emap[m + 1][0],
                                    ("Lagrange", pb.mesh_degree),
                                )
                                p_out = fem.Function(V_out_scalar_sub, name=pb.p_[m].name)
                                p_out.interpolate(pb.p_[m])
                                self.resultsfiles[res + str(m + 1)].write_function(p_out, indicator)
                        else:
                            p_out = fem.Function(pb.V_out_scalar, name=pb.p_[0].name)
                            p_out.interpolate(pb.p_[0])
                            self.resultsfiles[res].write_function(p_out, indicator)
                    elif res == "cauchystress":
                        stressfuncs = []
                        for n in range(pb.num_domains):
                            stressfuncs.append(pb.ma[n].sigma(pb.v, pb.p, F=pb.alevar["Fale"]))
                        cauchystress = project(
                            stressfuncs,
                            pb.Vd_tensor,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="CauchyStress",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_out = fem.Function(pb.V_out_tensor, name=cauchystress.name)
                        cauchystress_out.interpolate(cauchystress)
                        self.resultsfiles[res].write_function(cauchystress_out, indicator)
                    elif res == "fluiddisplacement":  # passed in uf is not a function but form, so we have to project
                        uf_proj = project(
                            pb.ufluid,
                            pb.V_v,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="FluidDisplacement",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        uf_out = fem.Function(pb.V_out_vector, name=uf_proj.name)
                        uf_out.interpolate(uf_proj)
                        self.resultsfiles[res].write_function(uf_out, indicator)
                    elif res == "fibers":
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    elif res == "cauchystress_membrane":
                        stressfuncs = []
                        for n in range(len(pb.bstress)):
                            stressfuncs.append(pb.bstress[n])
                        cauchystress_membrane = project(
                            stressfuncs,
                            pb.Vd_tensor,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="CauchyStress_membrane",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        cauchystress_membrane_out = fem.Function(pb.V_out_tensor, name=cauchystress_membrane.name)
                        cauchystress_membrane_out.interpolate(cauchystress_membrane)
                        self.resultsfiles[res].write_function(cauchystress_membrane_out, indicator)
                    elif res == "strainenergy_membrane":
                        sefuncs = []
                        for n in range(len(pb.bstrainenergy)):
                            sefuncs.append(pb.bstrainenergy[n])
                        strainenergy_membrane = project(
                            sefuncs,
                            pb.Vd_scalar,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="StrainEnergy_membrane",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        strainenergy_membrane_out = fem.Function(pb.V_out_scalar, name=strainenergy_membrane.name)
                        strainenergy_membrane_out.interpolate(strainenergy_membrane)
                        self.resultsfiles[res].write_function(strainenergy_membrane_out, indicator)
                    elif res == "internalpower_membrane":
                        pwfuncs = []
                        for n in range(len(pb.bintpower)):
                            pwfuncs.append(pb.bintpower[n])
                        internalpower_membrane = project(
                            pwfuncs,
                            pb.Vd_scalar,
                            pb.bmeasures[0],
                            domids=pb.idmem,
                            nm="InternalPower_membrane",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        internalpower_membrane_out = fem.Function(pb.V_out_scalar, name=internalpower_membrane.name)
                        internalpower_membrane_out.interpolate(internalpower_membrane)
                        self.resultsfiles[res].write_function(internalpower_membrane_out, indicator)
                    elif res == "internalpower":
                        pwfuncs = []
                        for n in range(pb.num_domains):
                            pwfuncs.append(
                                ufl.inner(
                                    pb.ma[n].sigma(pb.v, pb.p, F=pb.alevar["Fale"]),
                                    pb.ki.gamma(pb.v, F=pb.alevar["Fale"]),
                                )
                            )
                        pw = project(
                            pwfuncs,
                            pb.Vd_scalar,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="InternalPower",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        pw_out = fem.Function(pb.V_out_scalar, name=pw.name)
                        pw_out.interpolate(pw)
                        self.resultsfiles[res].write_function(pw_out, indicator)
                    elif res == "counters":
                        # iteration counters, written by base class
                        pass
                    else:
                        raise NameError("Unknown output to write for fluid mechanics!")

    def readcheckpoint(self, pb, N_rest):
        vecs_to_read = {}
        vecs_to_read[pb.v] = "v"
        vecs_to_read[pb.v_old] = "v_old"
        vecs_to_read[pb.a_old] = "a_old"
        vecs_to_read[pb.uf_old] = "uf_old"  # needed for ALE fluid / FSI / FrSI
        if any(pb.mem_active_stress):  # for active membrane model (FrSI)
            vecs_to_read[pb.tau_a] = "tau_a"
            vecs_to_read[pb.tau_a_old] = "tau_a"

        # pressure may be discontinuous across domains
        if bool(self.duplicate_mesh_domains):
            for m, mp in enumerate(self.duplicate_mesh_domains):
                vecs_to_read[pb.p__[m + 1]] = "p" + str(m + 1)
                vecs_to_read[pb.p_old__[m + 1]] = "p_old" + str(m + 1)
        else:
            vecs_to_read[pb.p] = "p"
            vecs_to_read[pb.p_old] = "p_old"

        for key in vecs_to_read:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "r",
                    self.comm,
                )
                key.x.petsc_vec.load(viewer)
                key.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.readfunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def writecheckpoint(self, pb, N):
        vecs_to_write = {}
        vecs_to_write[pb.v] = "v"
        vecs_to_write[pb.v_old] = "v_old"
        vecs_to_write[pb.a_old] = "a_old"
        vecs_to_write[pb.uf_old] = "uf_old"  # needed for ALE fluid / FSI / FrSI
        if any(pb.mem_active_stress):
            vecs_to_write[pb.tau_a] = "tau_a"

        # pressure may be discontinuous across domains
        if bool(self.duplicate_mesh_domains):
            for m, mp in enumerate(self.duplicate_mesh_domains):
                vecs_to_write[pb.p__[m + 1]] = "p" + str(m + 1)
                vecs_to_write[pb.p_old__[m + 1]] = "p_old" + str(m + 1)
        else:
            vecs_to_write[pb.p] = "p"
            vecs_to_write[pb.p_old] = "p_old"

        for key in vecs_to_write:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "w",
                    self.comm,
                )
                key.x.petsc_vec.view(viewer)
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.writefunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N),
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def close_output_files(self, pb):
        if self.write_results_every > 0:
            for res in pb.results_to_write:
                if res not in self.results_pre:
                    if res == "pressure" and bool(self.duplicate_mesh_domains):
                        for m, mp in enumerate(self.duplicate_mesh_domains):
                            self.resultsfiles[res + str(m + 1)].close()
                    else:
                        self.resultsfiles[res].close()


class IO_ale(IO):
    def write_output(self, pb, writemesh=False, N=1, t=0):
        self.results_pre = ["fibers", "counters"]

        if self.indicate_results_by == "time":
            indicator = t
        elif self.indicate_results_by == "step":
            indicator = N
        elif self.indicate_results_by == "step0":
            if self.write_results_every > 0:
                indicator = int(N / self.write_results_every) - 1
            else:
                indicator = 0
        else:
            raise ValueError("Unknown indicate_results_by optin. Choose 'time' or 'step'.")

        if writemesh:
            if self.write_results_every > 0:
                for res in pb.results_to_write:
                    if res not in self.results_pre:
                        outfile = io.XDMFFile(
                            self.comm,
                            self.output_path
                            + "/results_"
                            + pb.pbase.simname
                            + "_"
                            + pb.problem_physics
                            + "_"
                            + res
                            + ".xdmf",
                            "w",
                        )
                        outfile.write_mesh(self.mesh)
                        self.resultsfiles[res] = outfile

            return

        else:
            # write results every write_results_every steps
            if self.write_results_every > 0 and N % self.write_results_every == 0:
                # save solution to XDMF format
                for res in pb.results_to_write:
                    if res == "aledisplacement":
                        d_out = fem.Function(pb.V_out_vector, name=pb.d.name)
                        d_out.interpolate(pb.d)
                        self.resultsfiles[res].write_function(d_out, indicator)
                    elif res == "alevelocity":
                        w_proj = project(
                            pb.wel,
                            pb.V_d,
                            pb.dx,
                            domids=pb.domain_ids,
                            nm="AleVelocity",
                            comm=self.comm,
                            entity_maps=self.entity_maps,
                        )
                        w_out = fem.Function(pb.V_out_vector, name=w_proj.name)
                        w_out.interpolate(w_proj)
                        self.resultsfiles[res].write_function(w_out, indicator)
                    elif res == "counters":
                        # iteration counters, written by base class
                        pass
                    else:
                        raise NameError("Unknown output to write for ALE mechanics!")

    def readcheckpoint(self, pb, N_rest):
        vecs_to_read = {}
        vecs_to_read[pb.d] = "d"
        vecs_to_read[pb.d_old] = "d_old"
        vecs_to_read[pb.w_old] = "w_old"

        for key in vecs_to_read:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "r",
                    self.comm,
                )
                key.x.petsc_vec.load(viewer)
                key.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.readfunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def writecheckpoint(self, pb, N):
        vecs_to_write = {}
        vecs_to_write[pb.d] = "d"
        vecs_to_write[pb.d_old] = "d_old"
        vecs_to_write[pb.w_old] = "w_old"

        for key in vecs_to_write:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "w",
                    self.comm,
                )
                key.x.petsc_vec.view(viewer)
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.writefunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N),
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")


class IO_fluid_ale(IO_fluid, IO_ale):
    def write_output(self, pb, writemesh=False, N=1, t=0):
        IO_fluid.write_output(self, pb.pbf, writemesh=writemesh, N=N, t=t)
        IO_ale.write_output(self, pb.pba, writemesh=writemesh, N=N, t=t)

    def readcheckpoint(self, pb, N_rest):
        IO_fluid.readcheckpoint(self, pb.pbf, N_rest)
        IO_ale.readcheckpoint(self, pb.pba, N_rest)

    def write_restart(self, pb, N, force=False):
        if (self.write_restart_every > 0 and N % self.write_restart_every == 0) or force:
            IO_fluid.writecheckpoint(self, pb.pbf, N)
            IO_ale.writecheckpoint(self, pb.pba, N)


class IO_fsi(IO_solid, IO_fluid, IO_ale):
    def write_output(self, pb, writemesh=False, N=1, t=0):
        IO_solid.write_output(self, pb.pbs, writemesh=writemesh, N=N, t=t)
        IO_fluid.write_output(self, pb.pbf, writemesh=writemesh, N=N, t=t)
        IO_ale.write_output(self, pb.pba, writemesh=writemesh, N=N, t=t)

    def readcheckpoint(self, pb, N_rest):
        IO_solid.readcheckpoint(self, pb.pbs, N_rest)
        IO_fluid.readcheckpoint(self, pb.pbf, N_rest)
        IO_ale.readcheckpoint(self, pb.pba, N_rest)

        vecs_to_read = {}
        if pb.fsi_system == "neumann_neumann":
            vecs_to_read[pb.lm] = "LM"
            vecs_to_read[pb.lm_old] = "LM_old"

        for key in vecs_to_read:
            if self.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.comm.size)
                    + "proc.dat",
                    "r",
                    self.comm,
                )
                key.x.petsc_vec.load(viewer)
                key.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                self.readfunction(
                    key,
                    self.output_path
                    + "/checkpoint_"
                    + pb.pbase.simname
                    + "_"
                    + pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def write_restart(self, pb, N, force=False):
        if (self.write_restart_every > 0 and N % self.write_restart_every == 0) or force:
            IO_solid.writecheckpoint(self, pb.pbs, N)
            IO_fluid.writecheckpoint(self, pb.pbf, N)
            IO_ale.writecheckpoint(self, pb.pba, N)

            vecs_to_write = {}
            if pb.fsi_system == "neumann_neumann":
                vecs_to_write[pb.lm] = "LM"
                vecs_to_write[pb.lm_old] = "LM_old"

            for key in vecs_to_write:
                if self.restart_io_type == "petscvector":
                    # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                    # and for safety reasons, include the number of cores in the dat file name
                    viewer = PETSc.Viewer().createMPIIO(
                        self.output_path
                        + "/checkpoint_"
                        + pb.pbase.simname
                        + "_"
                        + pb.problem_physics
                        + "_"
                        + vecs_to_write[key]
                        + "_"
                        + str(N)
                        + "_"
                        + str(self.comm.size)
                        + "proc.dat",
                        "w",
                        self.comm,
                    )
                    key.x.petsc_vec.view(viewer)
                    viewer.destroy()
                elif self.restart_io_type == "plaintext":  # only working for nodal fields!
                    self.writefunction(
                        key,
                        self.output_path
                        + "/checkpoint_"
                        + pb.pbase.simname
                        + "_"
                        + pb.problem_physics
                        + "_"
                        + vecs_to_write[key]
                        + "_"
                        + str(N),
                        filetype='plaintext',
                    )
                else:
                    raise ValueError("Unknown restart_io_type!")

    def create_submeshes(self):
        self.dom_solid, self.dom_fluid, self.surf_interf = (
            self.io_params["domain_ids_solid"],
            self.io_params["domain_ids_fluid"],
            self.io_params["surface_ids_interface"],
        )

        self.msh_emap_solid = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim,
            self.mt_d.indices[np.isin(self.mt_d.values, self.dom_solid)],
        )[0:4] # returns: submesh, cell entity_map, vert entity map, original geo verts
        self.msh_emap_fluid = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim,
            self.mt_d.indices[np.isin(self.mt_d.values, self.dom_fluid)],
        )[0:4] # returns: submesh, cell entity_map, vert entity map, original geo verts

        self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim - 2, self.mesh.topology.dim)

        self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim - 2, self.mesh.topology.dim)

        # TODO: Assert that meshtags start actually from 1 when transferred!
        self.mt_d_solid = meshutils.meshtags_parent_to_child(
            self.mt_d,
            self.msh_emap_solid[0],
            self.msh_emap_solid[1],
            self.mesh,
            "domain",
        )
        self.mt_d_fluid = meshutils.meshtags_parent_to_child(
            self.mt_d,
            self.msh_emap_fluid[0],
            self.msh_emap_fluid[1],
            self.mesh,
            "domain",
        )

        self.mt_b1_solid = meshutils.meshtags_parent_to_child(
            self.mt_b1,
            self.msh_emap_solid[0],
            self.msh_emap_solid[1],
            self.mesh,
            "boundary",
        )
        self.mt_b1_fluid = meshutils.meshtags_parent_to_child(
            self.mt_b1,
            self.msh_emap_fluid[0],
            self.msh_emap_fluid[1],
            self.mesh,
            "boundary",
        )

        if self.mt_b2 is not None:
            self.mt_b2_solid = meshutils.meshtags_parent_to_child(
                self.mt_b2,
                self.msh_emap_solid[0],
                self.msh_emap_solid[1],
                self.mesh,
                "boundary_2",
            )
            self.mt_b2_fluid = meshutils.meshtags_parent_to_child(
                self.mt_b2,
                self.msh_emap_fluid[0],
                self.msh_emap_fluid[1],
                self.mesh,
                "boundary_2",
            )
        else:
            self.mt_b2_solid, self.mt_b2_fluid = None, None

        self.msh_emap_lm = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim - 1,
            self.mt_b1.indices[np.isin(self.mt_b1.values, self.surf_interf)],
        )[0:2]

        self.msh_emap_lm[0].topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim - 1)
        self.msh_emap_lm[0].topology.create_connectivity(self.mesh.topology.dim - 2, self.mesh.topology.dim - 1)

        cell_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        facet_imap = self.mesh.topology.index_map(self.mesh.topology.dim - 1)

        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        num_cells = cell_imap.size_local + cell_imap.num_ghosts

        # append to entity map list
        self.entity_maps.append(self.msh_emap_solid[1])
        self.entity_maps.append(self.msh_emap_fluid[1])
        self.entity_maps.append(self.msh_emap_lm[1])

        if self.write_submeshes:
            tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_solid.xdmf", "w")
            tmp.write_mesh(self.msh_emap_solid[0])
            tmp.write_meshtags(self.mt_d_solid, self.msh_emap_solid[0].geometry)
            tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_fluid.xdmf", "w")
            tmp.write_mesh(self.msh_emap_fluid[0])
            tmp.write_meshtags(self.mt_d_fluid, self.msh_emap_fluid[0].geometry)
            tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_interface.xdmf", "w")
            tmp.write_mesh(self.msh_emap_lm[0])

    # create domain and boundary integration measures
    def create_integration_measures(self, msh):
        # domain integration measure
        if self.mt_d is not None:
            self.dx = ufl.Measure(
                "dx",
                domain=msh,
                subdomain_data=self.mt_d,
                metadata={"quadrature_degree": self.quad_degree},
            )
        else:
            self.dx_ = ufl.Measure(
                "dx",
                domain=msh,
                metadata={"quadrature_degree": self.quad_degree},
            )
            self.dx = lambda a: self.dx_  # so that we can call dx(1) even without domain meshtags

        # now the boundary ones
        interface_facets = self.mt_b1.indices[np.isin(self.mt_b1.values, self.surf_interf)]
        solid_cells = self.mt_d.indices[np.isin(self.mt_d.values, self.dom_solid)]

        integration_entities = []

        # we need one global "master" ds measure, so need to append all other facets from mesh tags
        # using the format (cell, local_facet_index)

        # first, get all mesh tags
        meshtags = list(set(self.mt_b1.values))
        # loop over mesh tags
        for mt in meshtags:
            other_integration_entities = []
            other_indices = self.mt_b1.indices[self.mt_b1.values == mt]
            meshutils.get_integration_entities(
                msh,
                other_indices,
                self.mesh.topology.dim - 1,
                other_integration_entities,
            )
            # append
            integration_entities.append((mt, other_integration_entities))

        # now get the interface and use ids larger than the previous ones (we need a sorted list here!)
        self.interface_id_s = 1001
        self.interface_id_f = 1002

        meshutils.get_integration_entities_internal(
            msh,
            interface_facets,
            solid_cells,
            self.mesh.topology.dim - 1,
            integration_entities,
            [self.interface_id_s, self.interface_id_f],
        )

        self.ds = ufl.Measure(
            "ds",
            domain=msh,
            subdomain_data=integration_entities,
            metadata={"quadrature_degree": self.quad_degree},
        )
        self.de = None
        self.dS = ufl.Measure(
            "dS",
            domain=msh,
            subdomain_data=integration_entities,
            metadata={"quadrature_degree": self.quad_degree},
        )

        self.bmeasures = [self.ds, self.de, self.dS]
