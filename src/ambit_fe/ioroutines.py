#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
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
    def __init__(self, io_params, constitutive_params=[{}], entity_maps=None, comm=None):
        ioparams.check_params_io(io_params)

        self.io_params = io_params

        self.num_domains = []
        self.num_meshes = len(constitutive_params)  # number of different meshes involved (normally one; in FSI, we have 2: one for solid, the other for fluid+ALE, interface not counted here)
        for i in range(self.num_meshes):
            self.num_domains.append(len(constitutive_params[i]))

        # collect given domain ids
        self.domain_ids = [[] * self.num_meshes for _ in range(self.num_meshes)]
        for i in range(self.num_meshes):
            for n in range(self.num_domains[i]):
                self.domain_ids[i].append( constitutive_params[i]["MAT"+str(n+1)].get("id", n+1) )

        self.write_results_every = io_params["write_results_every"]
        self.output_path = io_params["output_path"]
        self.output_path_pre = io_params.get("output_path_pre", self.output_path)

        self.mesh_domain = io_params.get("mesh_domain", {"type":"unit_square", "celltype":"triangle", "meshsize":[10,10,10]}) # unit_square, unit_cube, rectangle
        self.mesh_boundary = io_params.get("mesh_boundary", None) # in 3D: surfaces, in 2D: edges
        self.mesh_subboundary = io_params.get("mesh_subboundary", None) # in 3D: edges, in 2D: points
        self.mesh_subsubboundary = io_params.get("mesh_subsubboundary", None) # in 3D: points, in 2D: -

        self.fiber_data = io_params.get("fiber_data", [])

        self.mesh_format = io_params.get("mesh_format", "XDMF")
        self.mesh_encoding = io_params.get("mesh_encoding", "ASCII")
        self.mesh_dim = io_params.get("mesh_dim", 3)  # actually only needed/used for gmsh read-in
        self.gridname_domain = io_params.get("gridname_domain", "Grid")
        self.gridname_boundary = io_params.get("gridname_boundary", "Grid")
        self.duplicate_mesh_domains = io_params.get("duplicate_mesh_domains", [])

        self.refine_mesh = io_params.get("refine_mesh", {})

        self.write_restart_every = io_params.get("write_restart_every", -1)
        self.restart_io_type = io_params.get("restart_io_type", "petscvector")
        self.indicate_results_by = io_params.get("indicate_results_by", "time")

        self.print_enhanced_info = io_params.get("print_enhanced_info", False)
        self.report_conservation_properties = io_params.get("report_conservation_properties", False)

        self.write_counters = io_params.get("write_counters", False)
        self.write_submeshes = io_params.get("write_submeshes", False)

        self.output_midpoint = io_params.get("output_midpoint", False)

        # entity map dict - for coupled multiphysics/multimesh problems
        self.entity_maps = entity_maps

        self.comm = comm

    def readin_mesh(self):
        if self.mesh_encoding == "ASCII":
            encoding = io.XDMFFile.Encoding.ASCII
        elif self.mesh_encoding == "HDF5":
            encoding = io.XDMFFile.Encoding.HDF5
        else:
            raise NameError("Choose either ASCII or HDF5 as mesh_encoding, or add a different encoding!")

        self.mt_d, self.mt_b, self.mt_sb, self.mt_ssb = None, None, None, None

        if type(self.mesh_domain) is not dict:
            if self.mesh_format == "XDMF":
                # read in xdmf mesh - domain
                with io.XDMFFile(self.comm, self.mesh_domain, "r", encoding=encoding) as infile:
                    self.mesh = infile.read_mesh(name=self.gridname_domain)
                    try:
                        self.mt_d = infile.read_meshtags(self.mesh, name=self.gridname_domain)
                    except:
                        self.mt_d = None

            elif self.mesh_format == "gmsh":
                # seems that we cannot infer the dimension from the mesh file but have to provide it to the read function...
                self.mesh, self.mt_d, self.mt_b = io.gmsh.read_from_msh(self.mesh_domain, self.comm, gdim=self.mesh_dim)[
                    0:3
                ]
                assert self.mesh.geometry.dim == self.mesh_dim  # would be weird if this wasn't true...

            else:
                raise NameError("Choose either XDMF or gmsh as mesh_format!")
        else:
            celltp = self.mesh_domain.get("celltype", "triangle")
            # 'hexahedron', 'prism', 'pyramid', 'quadrilateral', 'tetrahedron', 'triangle', 'interval', 'point'
            if celltp=="hexahedron":
                ctp = mesh.CellType.hexahedron
            elif celltp=="quadrilateral":
                ctp = mesh.CellType.quadrilateral
            elif celltp=="tetrahedron":
                ctp = mesh.CellType.tetrahedron
            elif celltp=="triangle":
                ctp = mesh.CellType.triangle
            elif celltp=="pyramid":
                ctp = mesh.CellType.pyramid
            elif celltp=="prism":
                ctp = mesh.CellType.prism
            else:
                raise ValueError("Unknown celltype!")
            msze = self.mesh_domain.get("meshsize", [10, 10, 10])
            coords_a = self.mesh_domain.get("coords_a", [0.0, 0.0, 0.0])
            coords_b = self.mesh_domain.get("coords_b", [1.0, 1.0, 1.0])
            # dolfinx internal mesh generation
            if self.mesh_domain["type"]=="unit_square":
                self.mesh = mesh.create_unit_square(self.comm, msze[0], msze[1], ctp)
            elif self.mesh_domain["type"]=="unit_cube":
                self.mesh = mesh.create_unit_cube(self.comm, msze[0], msze[1], msze[2], ctp)
            elif self.mesh_domain["type"]=="rectangle":
                self.mesh = mesh.create_rectangle(self.comm, [np.array(coords_a),np.array(coords_b)], msze, ctp)
            elif self.mesh_domain["type"]=="box":
                self.mesh = mesh.create_box(self.comm, [np.array(coords_a),np.array(coords_b)], msze, ctp)
            else:
                raise ValueError("Unknown type for mesh.")

        # seems that we need this (at least when locating dofs for DBCs on volume portions)
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

        # if requested, do an initial refinement around a region, or uniformly
        if bool(self.refine_mesh):
            refine_region = self.refine_mesh.get("region", None)
            refine_steps = self.refine_mesh.get("steps", 1)
            for i in range(refine_steps):
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                if refine_region is not None: # TODO: Always need edges (not faces?) even in 3D?
                    ee = mesh.locate_entities(self.mesh, 1, self.refine_mesh["region"].evaluate)
                else:
                    ee = None
                self.mesh, _, _ = mesh.refine(self.mesh, ee)

        # read in xdmf mesh - boundary
        # here, mt_b refers to BCs as BCs associated to a topology one dimension less than the problem (most common),
        # mt_sb BCs two dimensions less, and mt_ssb BCs three dimensions less
        # for a 3D problem - b: surface BCs, sb: edge BCs, ssb: point BCs
        # for a 2D problem - b: edge BCs, sb: point BCs
        # 1D problems not supported (currently...)

        if self.mesh.topology.dim == 3:
            if self.mesh_boundary is not None:
                self.mesh.topology.create_connectivity(2, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, "r", encoding=encoding) as infile:
                    self.mt_b = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

            if self.mesh_subboundary is not None:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_subboundary, "r", encoding=encoding) as infile:
                    self.mt_sb = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

            if self.mesh_subsubboundary is not None:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_subsubboundary, "r", encoding=encoding) as infile:
                    self.mt_ssb = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

        elif self.mesh.topology.dim == 2:
            if self.mesh_boundary is not None:
                self.mesh.topology.create_connectivity(1, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_boundary, "r", encoding=encoding) as infile:
                    self.mt_b = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

            if self.mesh_subboundary is not None:
                self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
                with io.XDMFFile(self.comm, self.mesh_subboundary, "r", encoding=encoding) as infile:
                    self.mt_sb = infile.read_meshtags(self.mesh, name=self.gridname_boundary)

        else:
            raise AttributeError("Your mesh seems to be 1D! Not supported!")

    # create domain and boundary integration measures
    def create_integration_measures(self, msh, mt_data, qdeg, bcdict=None):
        if all(isinstance(x, int) for x in self.domain_ids[0]):
            pass
        else: # can only be a locator function otherwise
            id_loc = 0
            cells_domain = []
            for i, lc in enumerate(self.domain_ids[0]):
                id_loc += 1
                cells_domain.append(mesh.locate_entities(msh, msh.topology.dim, lc.evaluate))
                # need to overwrite with an id
                self.domain_ids[0][i] = id_loc
            self.cells_domain = np.concatenate(cells_domain).ravel()

            domain_indices, domain_markers = [], []
            for i in range(len(cells_domain)):
                domain_indices.append(cells_domain[i])
                domain_markers.append(np.full_like(cells_domain[i], self.domain_ids[0][i]))

            domain_indices = np.hstack(domain_indices).astype(np.int32)
            domain_markers = np.hstack(domain_markers).astype(np.int32)
            sorter_domain_indices = np.argsort(domain_indices)

        if mt_data[0] is not None:
            dx = ufl.Measure(
                "dx",
                domain=msh,
                subdomain_data=mt_data[0],
                metadata={"quadrature_degree": qdeg},
            )
        else:
            # if we don't have meshtags, create them out of locator functions
            domain_tags = mesh.meshtags(msh, msh.topology.dim, domain_indices[sorter_domain_indices], domain_markers[sorter_domain_indices])
            dx = ufl.Measure("dx", domain=msh, subdomain_data=domain_tags, metadata={"quadrature_degree": qdeg})

        if mt_data[1] is not None:
            ds = ufl.Measure(
                "ds",
                domain=msh,
                subdomain_data=mt_data[1],
                metadata={"quadrature_degree": qdeg},
            )
        else:
            ds = None

        if mt_data[1] is not None:
            dS = ufl.Measure(
                "dS",
                domain=msh,
                subdomain_data=mt_data[1],
                metadata={"quadrature_degree": qdeg},
            )
        else:
            dS = None

        bmeasures = [ds, dS]

        # if user-defined locators (instead of ids) are given, create a separate measure
        facets_bc_indices, facets_bc_markers = [], []
        non_facet_bcs = ["dirichlet", "dirichlet_pres"] # treated differently (no facet integration measure needed)
        if bcdict is not None:
            id_=0
            for B in bcdict:
                for k in B.keys():
                    if k not in non_facet_bcs:
                        for i in range(len(B[k])):
                            if all(isinstance(x, int) for x in B[k][i]["id"]):
                                pass
                            else:
                                codim = B[k][i].get("codimension", msh.topology.dim-1)
                                id_+=1
                                facets_bc_ = []
                                for lc in B[k][i]["id"]:
                                    facets_bc_.append(mesh.locate_entities(msh, codim, lc.evaluate))
                                B[k][i]["id_loc"] = [id_] # set new integer identifier
                                B[k][i]["is_locator"] = True # TODO: Find nicer solution for this in boundaryconditions.py
                                facets_bc = np.concatenate(facets_bc_).ravel()
                                # append to list of all locator BCs
                                facets_bc_indices.append(facets_bc)
                                facets_bc_markers.append(np.full_like(facets_bc, id_))

            if id_>0:
                # stack all facets/markers
                facets_bc_indices = np.hstack(facets_bc_indices).astype(np.int32)
                facets_bc_markers = np.hstack(facets_bc_markers).astype(np.int32)
                sorter_facets_bc = np.argsort(facets_bc_indices)

                facet_tag = mesh.meshtags(msh, codim, facets_bc_indices[sorter_facets_bc], facets_bc_markers[sorter_facets_bc])
                ds_loc = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag, metadata={"quadrature_degree": qdeg})

                bmeasures.append(ds_loc)

        if bool(self.duplicate_mesh_domains):
            self.submshes_emap, self.sub_mt_d, self.sub_mt_b = {}, {}, {}
            self.create_fluid_duplicate_pressure_mesh(self.mesh, [self.mt_d,self.mt_b], self.submshes_emap, self.sub_mt_d, self.sub_mt_b)
            self.submshes_emap_aux, self.sub_mt_d_aux, self.sub_mt_b_aux = self.submshes_emap, self.sub_mt_d, self.sub_mt_b

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

    def set_func_const_vec(self, func, array):
        load = expression.template_vector()
        load.val_x, load.val_y, load.val_z = array[0], array[1], array[2]
        func.interpolate(load.evaluate)

    # own read function requiting reading from .h5 format and redistributing the field correctly to the PETSc vector
    def readfunction(
        self,
        f,
        datafile,
        fieldname=None,
        tol=1e-6,
        filetype="xdmf_h5"
    ):

        ts = time.time()
        utilities.print_status("Reading file %s..." % (datafile), self.comm, e=" ")

        # get local size and block size of PETSc vector
        sz_loc, bs = f.x.petsc_vec.getLocalSize(), f.x.petsc_vec.getBlockSize()

        # default: when reading data from an .xdmf/.h5 file, we assume that we always have the nodal
        # coordinates that serve as basis for establishing the mapping
        if filetype=="xdmf_h5":
            import h5py
            import xml.etree.ElementTree as ET
            from scipy.spatial import cKDTree

            xtree = ET.parse(datafile)
            xroot = xtree.getroot()

            def clean_hdf_text(text):
                """
                Convert e.g.
                    fibers.h5:/Mesh/mesh/fiber
                into
                    ("fibers.h5", "Mesh/mesh/fiber")
                """
                text = "".join(text.split())  # remove whitespace/newlines
                h5file, h5path = text.split(":/", 1)
                return h5file, h5path

            # Geometry
            geo_dataitem = xroot.find(".//Geometry/DataItem")
            geo_h5file, name_geo = clean_hdf_text(geo_dataitem.text)

            # Topology / cells, if needed
            topo_dataitem = xroot.find(".//Topology/DataItem")
            topo_h5file, name_cells = clean_hdf_text(topo_dataitem.text)

            # All fields/attributes
            fields = {}

            for attr in xroot.findall(".//Attribute"):
                attr_name = attr.get("Name")          # "fiber", "sheet", ...
                attr_type = attr.get("AttributeType") # "Vector" or "Scalar"
                center = attr.get("Center")           # "Node", "Cell", ...

                dataitem = attr.find("DataItem")
                h5file, h5path = clean_hdf_text(dataitem.text)

                fields[attr_name] = {
                    "h5file": h5file,
                    "h5path": h5path,
                    "attribute_type": attr_type,
                    "center": center,
                }

            if fieldname is None:  # read first field if not specified
                name_fld = next(iter(fields.values()))["h5path"]
            else:
                name_fld = fields[fieldname]["h5path"]

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
            nodes_reordered = f.function_space.tabulate_dof_coordinates()[:,:f.function_space.mesh.geometry.dim]

            # create index mapping according to distance tree
            tree = cKDTree(nodes0)
            _, mapping_indices = tree.query(nodes_reordered, distance_upper_bound=tol)

        # ATTENTION: This only works when your field to read is ordered according to the original mesh!
        elif filetype=="plaintext" or filetype=="cheart":

            if filetype=="plaintext":
                data = np.loadtxt(datafile,usecols=(np.arange(0,bs)),ndmin=2)
            if filetype=="cheart": # CHeart .D files: skip first row
                data = np.loadtxt(datafile,usecols=(np.arange(0,bs)),ndmin=2,skiprows=1)

            # for discontinuous cell-wise functions, we need the original cell, otherwise the node index
            if f.function_space._ufl_element.is_cellwise_constant():
                mapping_indices = f.function_space.mesh.topology.original_cell_index
            else:
                mapping_indices = f.function_space.mesh.geometry.input_global_indices

        # order data
        data_mapped = data[mapping_indices[: int(sz_loc / bs)]]

        # flatten mapped data
        data_mapped_flat = data_mapped[:,:f.function_space.mesh.geometry.dim].flatten()

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
            V_fib.mesh,
            ("Lagrange", self.order_fib_input, (V_fib.mesh.geometry.dim,)),
        )

        # up to now, these are given... maybe have user specify...
        fibernames = ["fiber", "sheet"]

        for i, s in enumerate(fibarray):
            # if isinstance(self.fiber_data[i], str):
            #     dat = np.loadtxt(self.fiber_data[i])
            #     if len(dat) != V_fib_input.dofmap.index_map.size_global:
            #         raise RuntimeError(
            #             "Your order of fiber input data does not match the (assumed) order of the fiber function space, %i. Specify 'order_fib_input' in your IO section."
            #             % (self.order_fib_input)
            #         )

            fib_func_input.append(fem.Function(V_fib_input, name="Fiber" + str(i + 1)))

            if isinstance(self.fiber_data[i], str):
                self.readfunction(fib_func_input[i], self.fiber_data[i], fieldname=fibernames[i])
            else:  # assume a constant-in-space list or array
                self.set_func_const_vec(fib_func_input[i], self.fiber_data[i])

            # project to fiber function space
            if self.order_fib_input != order_disp:
                fib_func.append(
                    project(
                        fib_func_input[i],
                        V_fib,
                        dx_,
                        domids=domids,
                        nm="Fiber" + str(i + 1),
                        comm=self.comm,
                        entity_maps=self.entity_maps,
                    )
                )
            else:
                fib_func.append(fib_func_input[i])

            # assert that field is actually always normalized!
            fib_func[i] /= ufl.sqrt(ufl.dot(fib_func[i], fib_func[i]))

        return fib_func

    def export_matrix(self, mat, fname):
        viewer = PETSc.Viewer().create(self.comm)
        viewer.setType(PETSc.Viewer.Type.ASCII)
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        viewer.setFileMode("w")
        viewer.setFileName(self.output_path + "/" + fname)
        mat.view(viewer=viewer)
        viewer.destroy()

    # for duplicate (fluid) pressure nodes at an internal boundary, we can split the domain into two subdomains for the pressure function space
    def create_fluid_duplicate_pressure_mesh(self, msh, mtags, submshes_emap, sub_mt_d, sub_mt_b, aux=False):
        for m, mp in enumerate(self.duplicate_mesh_domains):
            cells_part_ = []
            if all(isinstance(x, int) for x in mp):
                for id_ in mp:
                    cells_part_.append(mtags[0].indices[mtags[0].values == id_])
            else: # can only be a locator function otherwise
                for i, lc in enumerate(mp):
                    cells_part_.append(mesh.locate_entities(msh, msh.topology.dim, lc.evaluate))
            cells_part = np.concatenate(cells_part_).ravel()

            submshes_emap[m + 1] = mesh.create_submesh(
                msh,
                msh.topology.dim,
                cells_part,
            )[0:2]

        for m, mp in enumerate(self.duplicate_mesh_domains):  # needed ?!
            submshes_emap[m + 1][0].topology.create_connectivity(submshes_emap[m + 1][0].topology.dim, submshes_emap[m + 1][0].topology.dim)
            submshes_emap[m + 1][0].topology.create_connectivity(submshes_emap[m + 1][0].topology.dim - 1, submshes_emap[m + 1][0].topology.dim)
            submshes_emap[m + 1][0].topology.create_connectivity(submshes_emap[m + 1][0].topology.dim - 2, submshes_emap[m + 1][0].topology.dim)

        for m, mp in enumerate(self.duplicate_mesh_domains):
            if not aux: self.entity_maps.append(submshes_emap[m + 1][1])
            # transfer meshtags to submesh
            sub_mt_d[m + 1] = meshutils.meshtags_parent_to_child(
                mtags[0],
                submshes_emap[m + 1][0],
                submshes_emap[m + 1][1],
                msh,
                "domain",
            )
            sub_mt_b[m + 1] = meshutils.meshtags_parent_to_child(
                mtags[1],
                submshes_emap[m + 1][0],
                submshes_emap[m + 1][1],
                msh,
                "boundary",
            )

        if self.write_submeshes and not aux:
            for m, mp in enumerate(self.duplicate_mesh_domains):
                tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_fluid"+str(m+1)+".xdmf", "w")
                tmp.write_mesh(submshes_emap[m + 1][0])
                if sub_mt_d[m + 1] is not None:
                    tmp.write_meshtags(sub_mt_d[m + 1], submshes_emap[m + 1][0].geometry)

class IO_field:
    def __init__(self, pb):
        self.pb = pb

    def write_output_pre(self, func, V_out, t, name):
        outfile = io.XDMFFile(
            self.pb.comm,
            self.pb.io.output_path_pre + "/results_" + self.pb.pbase.simname + "_" + self.pb.problem_physics + "_" + name + ".xdmf",
            "w",
        )
        outfile.write_mesh(self.pb.mesh)
        func_out = fem.Function(V_out, name=func.name)
        func_out.interpolate(func)
        outfile.write_function(func_out, t)
        outfile.close()

    def write_restart(self, N, force=False):
        if (self.pb.io.write_restart_every > 0 and N % self.pb.io.write_restart_every == 0) or force:
            self.writecheckpoint(N)

    def close_output_files(self):
        if self.pb.io.write_results_every > 0:
            for res in self.pb.results_to_write:
                if res not in self.results_pre:
                    self.pb.resultsfiles[res].close()


class IO_fsi(IO):
    def create_submeshes(self):
        self.msh_emap_solid = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim,
            self.cells_solid,
        )[0:4] # returns: submesh, cell entity_map, vert entity map, original geo verts
        self.msh_emap_fluid = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim,
            self.cells_fluid,
        )[0:4] # returns: submesh, cell entity_map, vert entity map, original geo verts

        self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        self.msh_emap_solid[0].topology.create_connectivity(self.mesh.topology.dim - 2, self.mesh.topology.dim)

        self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        self.msh_emap_fluid[0].topology.create_connectivity(self.mesh.topology.dim - 2, self.mesh.topology.dim)

        # TODO: Assert that meshtags start actually from 1 when transferred!
        if self.mt_d is not None:
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
        else:
            self.mt_d_solid, self.mt_d_fluid = None, None

        if self.mt_b is not None:
            self.mt_b_solid = meshutils.meshtags_parent_to_child(
                self.mt_b,
                self.msh_emap_solid[0],
                self.msh_emap_solid[1],
                self.mesh,
                "boundary",
            )
            self.mt_b_fluid = meshutils.meshtags_parent_to_child(
                self.mt_b,
                self.msh_emap_fluid[0],
                self.msh_emap_fluid[1],
                self.mesh,
                "boundary",
            )
        else:
            self.mt_b_solid, self.mt_b_fluid = None, None

        if self.mt_sb is not None:
            self.mt_sb_solid = meshutils.meshtags_parent_to_child(
                self.mt_sb,
                self.msh_emap_solid[0],
                self.msh_emap_solid[1],
                self.mesh,
                "boundary_2",
            )
            self.mt_sb_fluid = meshutils.meshtags_parent_to_child(
                self.mt_sb,
                self.msh_emap_fluid[0],
                self.msh_emap_fluid[1],
                self.mesh,
                "boundary_2",
            )
        else:
            self.mt_sb_solid, self.mt_sb_fluid = None, None

        self.msh_emap_lm = mesh.create_submesh(
            self.mesh,
            self.mesh.topology.dim - 1,
            self.facets_interface,
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

        # in FSI, we need addtional auxiliary entity maps from the fluid submesh to the pressure submeshes if we want to integrate on these
        if bool(self.duplicate_mesh_domains):
            self.submshes_emap_aux, self.sub_mt_d_aux, self.sub_mt_b_aux = {}, {}, {}
            self.create_fluid_duplicate_pressure_mesh(self.msh_emap_fluid[0], [self.mt_d_fluid,self.mt_b_fluid], self.submshes_emap_aux, self.sub_mt_d_aux, self.sub_mt_b_aux, aux=True)

        if self.write_submeshes:
            tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_solid.xdmf", "w")
            tmp.write_mesh(self.msh_emap_solid[0])
            if self.mt_d_solid is not None:
                tmp.write_meshtags(self.mt_d_solid, self.msh_emap_solid[0].geometry)
            tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_fluid.xdmf", "w")
            tmp.write_mesh(self.msh_emap_fluid[0])
            if self.mt_d_fluid is not None:
                tmp.write_meshtags(self.mt_d_fluid, self.msh_emap_fluid[0].geometry)
            tmp = io.XDMFFile(self.comm, self.output_path_pre+"/mesh_interface.xdmf", "w")
            tmp.write_mesh(self.msh_emap_lm[0])


    # create domain and boundary integration measures
    def create_integration_measures(self, msh, sids, fids, iids, qdeg, bcdict=None):
        self.dom_solid, self.dom_fluid, self.surf_interf = (
            sids,
            fids,
            iids,
        )

        id_loc = 0

        cells_solid = []
        if all(isinstance(x, int) for x in self.dom_solid):
            for id_ in self.dom_solid:
                cells_solid.append(self.mt_d.indices[self.mt_d.values == id_])
        else: # can only be a locator function otherwise
            for i, lc in enumerate(self.dom_solid):
                id_loc += 1
                cells_solid.append(mesh.locate_entities(msh, msh.topology.dim, lc.evaluate))
                # need to overwrite with an id
                self.dom_solid[i] = id_loc

        self.cells_solid = np.concatenate(cells_solid).ravel()

        cells_fluid = []
        if all(isinstance(x, int) for x in self.dom_fluid):
            for id_ in self.dom_fluid:
                cells_fluid.append(self.mt_d.indices[self.mt_d.values == id_])
        else: # can only be a locator function otherwise
            for i, lc in enumerate(self.dom_fluid):
                id_loc += 1
                cells_fluid.append(mesh.locate_entities(msh, msh.topology.dim, lc.evaluate))
                # need to overwrite with an id
                self.dom_fluid[i] = id_loc

        self.cells_fluid = np.concatenate(cells_fluid).ravel()

        if all(isinstance(x, int) for x in self.surf_interf):
            self.facets_interface = self.mt_b.indices[np.isin(self.mt_b.values, self.surf_interf)]
        else: # can only be a locator function otherwise
            facets_interf = []
            for lc in self.surf_interf:
                locator_interf = lc.evaluate
                facets_interf.append(mesh.locate_entities(msh, msh.topology.dim-1, locator_interf))
            self.facets_interface = np.concatenate(facets_interf).ravel()

        # create global dx measure
        # build new meshtags (we might have locators that specify the cells)
        domain_indices, domain_markers = [], []
        for i in range(len(cells_solid)):
            domain_indices.append(cells_solid[i])
            domain_markers.append(np.full_like(cells_solid[i], self.dom_solid[i]))
        for i in range(len(cells_fluid)):
            domain_indices.append(cells_fluid[i])
            domain_markers.append(np.full_like(cells_fluid[i], self.dom_fluid[i]))

        domain_indices = np.hstack(domain_indices).astype(np.int32)
        domain_markers = np.hstack(domain_markers).astype(np.int32)
        sorter_domain_indices = np.argsort(domain_indices)

        domain_tags = mesh.meshtags(msh, self.mesh.topology.dim, domain_indices[sorter_domain_indices], domain_markers[sorter_domain_indices])
        self.dx = ufl.Measure("dx", domain=msh, subdomain_data=domain_tags, metadata={"quadrature_degree": qdeg})

        # now take care of the global ds measure...
        integration_entities = []
        # first, get all mesh tags
        if self.mt_b is not None:
            meshtags = list(set(self.mt_b.values))
            meshtags = self.comm.allreduce(meshtags)
            # loop over mesh tags
            for mt in meshtags:
                other_integration_entities = []
                other_indices = self.mt_b.indices[self.mt_b.values == mt]
                meshutils.get_integration_entities(msh, other_indices, self.mesh.topology.dim - 1, other_integration_entities)
                # append
                integration_entities.append((mt, other_integration_entities))
        else:
            meshtags = [0]

        # we need one global "master" ds measure, so need to append all additional facets from locators
        non_facet_bcs = ["dirichlet", "dirichlet_pres"] # treated differently (no facet integration measure needed)
        if bcdict is not None:
            id_loc = max(meshtags)
            for B in bcdict:
                for k in B.keys():
                    if k not in non_facet_bcs:
                        for i in range(len(B[k])):
                            codim = B[k][i].get("codimension", msh.topology.dim-1)
                            if all(isinstance(x, int) for x in B[k][i]["id"]):
                                pass
                            else:
                                B[k][i]["id_loc"] = []
                                for lc in B[k][i]["id"]:
                                    id_loc += 1
                                    other_integration_entities = []
                                    other_indices = mesh.locate_entities(msh, codim, lc.evaluate)
                                    meshutils.get_integration_entities(msh, other_indices, codim, other_integration_entities)
                                    integration_entities.append((id_loc, other_integration_entities))
                                    # append for later access in BC routines
                                    B[k][i]["id_loc"].append(id_loc)

        # now get the interface and use ids larger than the previous ones (we need a sorted list here!)
        self.interface_id_s = 1001
        self.interface_id_f = 1002

        meshutils.get_integration_entities_internal(
            msh,
            self.facets_interface,
            self.cells_solid,
            self.cells_fluid,
            msh.topology.dim - 1,
            integration_entities,
            [self.interface_id_s, self.interface_id_f],
        )

        self.ds = ufl.Measure(
            "ds",
            domain=msh,
            subdomain_data=integration_entities,
            metadata={"quadrature_degree": qdeg},
        )

        # NOTE: "internal" dS measure cannot be properly used with integration_entities,
        # hence we use the mt_b data directly - TODO: If we have locators, we'd need to create mt_b first... :-/
        self.dS = ufl.Measure(
            "dS",
            domain=msh,
            subdomain_data=self.mt_b,
            metadata={"quadrature_degree": qdeg},
        )

        self.bmeasures = [self.ds, self.dS]

        if bool(self.duplicate_mesh_domains):
            self.submshes_emap, self.sub_mt_d, self.sub_mt_b = {}, {}, {}
            self.create_fluid_duplicate_pressure_mesh(msh, [self.mt_d,self.mt_b], self.submshes_emap, self.sub_mt_d, self.sub_mt_b)


class IO_0D():  # 0D "dummy" IO class, carries only one parameter so far...
    def __init__(self, io_params):
        self.write_restart_every = io_params.get("write_restart_every", -1)
        self.write_counters = io_params.get("write_counters", False)


class IO_field_fsi(IO_field):
    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        if self.pb.fsi_system == "neumann_neumann":
            vecs_to_read[self.pb.lm] = "LM"
            vecs_to_read[self.pb.lm_old] = "LM_old"

        for key in vecs_to_read:
            if self.pb.io.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.pb.comm.size)
                    + "proc.dat",
                    "r",
                    self.pb.comm,
                )
                key.x.petsc_vec.load(viewer)
                key.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.pb.io.restart_io_type == "plaintext":  # only working for nodal fields!
                self.readfunction(
                    key,
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def write_restart(self, N, force=False):
        if (self.pb.io.write_restart_every > 0 and N % self.pb.io.write_restart_every == 0) or force:
            vecs_to_write = {}
            if self.pb.fsi_system == "neumann_neumann":
                vecs_to_write[self.pb.lm] = "LM"
                vecs_to_write[self.pb.lm_old] = "LM_old"

            for key in vecs_to_write:
                if self.pb.io.restart_io_type == "petscvector":
                    # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                    # and for safety reasons, include the number of cores in the dat file name
                    viewer = PETSc.Viewer().createMPIIO(
                        self.pb.io.output_path
                        + "/checkpoint_"
                        + self.pb.pbase.simname
                        + "_"
                        + self.pb.problem_physics
                        + "_"
                        + vecs_to_write[key]
                        + "_"
                        + str(N)
                        + "_"
                        + str(self.pb.comm.size)
                        + "proc.dat",
                        "w",
                        self.pb.comm,
                    )
                    key.x.petsc_vec.view(viewer)
                    viewer.destroy()
                elif self.pb.io.restart_io_type == "plaintext":  # only working for nodal fields!
                    self.writefunction(
                        key,
                        self.pb.io.output_path
                        + "/checkpoint_"
                        + self.pb.pbase.simname
                        + "_"
                        + self.pb.problem_physics
                        + "_"
                        + vecs_to_write[key]
                        + "_"
                        + str(N),
                        filetype='plaintext',
                    )
                else:
                    raise ValueError("Unknown restart_io_type!")
