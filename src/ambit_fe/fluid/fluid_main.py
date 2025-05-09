#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, os, inspect
import numpy as np
from dolfinx import fem, mesh
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc

from . import fluid_kinematics_constitutive
from . import fluid_variationalform
from .. import timeintegration
from .. import utilities
from .. import boundaryconditions
from .. import meshutils, expression, ioparams
from ..solver import solver_nonlin
from ..solver.projection import project
from ..solid.solid_material import activestress_activation

from ..base import problem_base, solver_base

"""
Fluid mechanics, governed by incompressible Navier-Stokes equations:

\begin{align}
\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \left(\boldsymbol{\nabla}\boldsymbol{v}\right) \boldsymbol{v}\right) = \boldsymbol{\nabla} \cdot \boldsymbol{\sigma} + \hat{\boldsymbol{b}} \quad \text{in} \; \Omega \times [0, T] \\
\boldsymbol{\nabla} \cdot \boldsymbol{v} = 0 \quad \text{in} \; \Omega \times [0, T]
\end{align}
"""

class FluidmechanicsProblem(problem_base):

    def __init__(self, pbase, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, iof, mor_params={}, comm=None, alevar={}):

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_fem_fluid(fem_params)
        ioparams.check_params_time_fluid(time_params)

        self.problem_physics = 'fluid'

        self.results_to_write = io_params['results_to_write']

        self.io = iof

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        # for FSI, we want to specify the subdomains
        self.domain_ids = self.io.io_params.get('domain_ids_fluid', np.arange(1,self.num_domains+1))

        # TODO: Find nicer solution here...
        if self.pbase.problem_type=='fsi' or self.pbase.problem_type=='fsi_flow0d':
            self.dx, self.bmeasures = self.io.dx, self.io.bmeasures
        else:
            self.dx, self.bmeasures = self.io.create_integration_measures(self.io.mesh, [self.io.mt_d,self.io.mt_b1,self.io.mt_b2])

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_vel = fem_params['order_vel']
        self.order_pres = fem_params['order_pres']
        self.quad_degree = fem_params['quad_degree']

        # collect domain data
        self.rho = []
        for n, M in enumerate(self.domain_ids):
            # data for inertial forces: density
            self.rho.append(self.constitutive_models['MAT'+str(n+1)]['inertia']['rho'])

        self.fluid_formulation = fem_params.get('fluid_formulation', 'nonconservative')
        self.fluid_governing_type = time_params.get('fluid_governing_type', 'navierstokes_transient')
        self.stabilization = fem_params.get('stabilization', None)

        self.prestress_initial = fem_params.get('prestress_initial', False)
        self.prestress_initial_only = fem_params.get('prestress_initial_only', False)
        self.prestress_maxtime = self.pbase.ctrl_params.get('prestress_maxtime', 1.0)
        self.prestress_numstep = self.pbase.ctrl_params.get('prestress_numstep', 1)
        self.prestress_dt = self.pbase.ctrl_params.get('prestress_dt', self.prestress_maxtime/self.prestress_numstep)
        if 'prestress_dt' in self.pbase.ctrl_params.keys(): self.prestress_numstep = int(self.prestress_maxtime/self.prestress_dt)
        self.prestress_ptc = fem_params.get('prestress_ptc', False)
        self.prestress_kinetic = fem_params.get('prestress_kinetic', 'none')
        self.prestress_from_file = fem_params.get('prestress_from_file', False)
        self.initial_fluid_pressure = fem_params.get('initial_fluid_pressure', [])

        if self.prestress_from_file: self.prestress_initial, self.prestress_initial_only  = False, False

        self.localsolve = False # no idea what might have to be solved locally...

        self.sub_solve = False
        self.print_subiter = False

        self.have_flux_monitor, self.have_dp_monitor, self.have_robin_valve, self.have_robin_valve_implicit = False, False, False, False
        self.have_condensed_variables = False

        self.dim = self.io.mesh.geometry.dim

        # dicts for evaluations of surface integrals (fluxes, pressures), to be queried by other models
        self.qv_, self.pu_, self.pd_, self.dp_ = {}, {}, {}, {}
        self.qv_old_, self.pu_old_, self.pd_old_, self.dp_old_ = {}, {}, {}, {}

        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            dg_type = "DG"
            if (self.order_vel > 1 or self.order_pres > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            dg_type = "DQ"
            if (self.order_vel > 1 or self.order_pres > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")

        self.Vex = self.io.mesh.ufl_domain().ufl_coordinate_element()

        if self.stabilization is None and self.order_vel == self.order_pres:
            raise ValueError("Equal order velocity and pressure interpolation is not recommended for non-stabilized Navier-Stokes!")

        # model order reduction
        self.mor_params = mor_params
        if bool(self.mor_params): self.pbase.have_rom = True
        else: self.pbase.have_rom = False
        # will be set by solver base class
        self.rom = None

        # ALE fluid problem variables
        self.alevar = alevar

        # function space for v
        self.V_v = fem.functionspace(self.io.mesh, ("Lagrange", self.order_vel, (self.io.mesh.geometry.dim,)))

        # now collect those for p (pressure space may be duplicate!)
        self.V_p__ = {}

        self.dx_p, self.bmeasures_p = [], []

        if bool(self.io.duplicate_mesh_domains):

            assert(self.num_domains>1)

            self.num_dupl = self.num_domains

            self.io.submshes_emap, inv_emap, self.io.sub_mt_d, self.io.sub_mt_b1 = {}, {}, {}, {}

            for m, mp in enumerate(self.io.duplicate_mesh_domains):
                self.io.submshes_emap[m+1] = mesh.create_submesh(self.io.mesh, self.io.mesh.topology.dim, self.io.mt_d.indices[np.isin(self.io.mt_d.values, mp)])[0:2]

            # self.io.submshes_emap[m+1][0].topology.create_connectivity(3, 3)
            # self.io.submshes_emap[m+1][0].topology.create_connectivity(2, 2)
            # self.io.submshes_emap[m+1][0].topology.create_connectivity(3, 2)
            # self.io.submshes_emap[m+1][0].topology.create_connectivity(2, 3)

            cell_imap = self.io.mesh.topology.index_map(self.io.mesh.topology.dim)
            num_cells = cell_imap.size_local + cell_imap.num_ghosts

            for m, mp in enumerate(self.io.duplicate_mesh_domains):
                self.V_p__[m+1] = fem.functionspace(self.io.submshes_emap[m+1][0], ("Lagrange", self.order_pres))

                inv_emap[m+1] = np.full(num_cells, -1)

            for m, mp in enumerate(self.io.duplicate_mesh_domains):
                inv_emap[m+1][self.io.submshes_emap[m+1][1]] = np.arange(len(self.io.submshes_emap[m+1][1]))
                self.io.entity_maps[self.io.submshes_emap[m+1][0]] = inv_emap[m+1]
                # transfer meshtags to submesh
                self.io.sub_mt_d[m+1] = meshutils.meshtags_parent_to_child(self.io.mt_d, self.io.submshes_emap[m+1][0], self.io.submshes_emap[m+1][1], self.io.mesh, 'domain')
                self.io.sub_mt_b1[m+1] = meshutils.meshtags_parent_to_child(self.io.mt_b1, self.io.submshes_emap[m+1][0], self.io.submshes_emap[m+1][1], self.io.mesh, 'boundary')

                dxp, bmeasuresp = self.io.create_integration_measures(self.io.submshes_emap[m+1][0], [self.io.sub_mt_d[m+1],self.io.sub_mt_b1[m+1],None])
                # self.dx_p.append(dxp)
                # self.bmeasures_p.append(bmeasuresp)
                self.dx_p.append(self.dx)
                self.bmeasures_p.append(self.bmeasures)

        else:
            self.num_dupl = 1
            self.V_p_ = [ fem.functionspace(self.io.mesh, ("Lagrange", self.order_pres)) ]
            self.dx_p.append(self.dx)
            self.bmeasures_p.append(self.bmeasures)

        # continuous tensor and scalar function spaces of order order_vel
        self.V_tensor = fem.functionspace(self.io.mesh, ("Lagrange", self.order_vel, (self.io.mesh.geometry.dim,self.io.mesh.geometry.dim)))
        self.V_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.order_vel))

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.functionspace(self.io.mesh, (dg_type, self.order_vel-1, (self.io.mesh.geometry.dim,self.io.mesh.geometry.dim)))
        self.Vd_vector = fem.functionspace(self.io.mesh, (dg_type, self.order_vel-1, (self.io.mesh.geometry.dim,)))
        self.Vd_scalar = fem.functionspace(self.io.mesh, (dg_type, self.order_vel-1))

        # for output writing - function spaces on the degree of the mesh
        self.mesh_degree = self.io.mesh._ufl_domain._ufl_coordinate_element._degree
        self.V_out_tensor = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree, (self.io.mesh.geometry.dim,self.io.mesh.geometry.dim)))
        self.V_out_vector = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree, (self.io.mesh.geometry.dim,)))
        self.V_out_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree))

        # coordinate element function space - based on input mesh
        self.Vcoord = fem.functionspace(self.io.mesh, self.Vex)

        # functions
        self.dv     = ufl.TrialFunction(self.V_v)            # Incremental velocity
        self.var_v  = ufl.TestFunction(self.V_v)             # Test function
        self.v      = fem.Function(self.V_v, name="Velocity")

        # pressure can have duplicate nodes (one variable per domain)
        self.p__, self.p_old__, self.var_p__, self.dp__ = {}, {}, {}, {}
        if self.num_dupl > 1:
            for m, mp in enumerate(self.io.duplicate_mesh_domains):
                self.p__[m+1] = fem.Function(self.V_p__[m+1], name="Pressure"+str(m+1))
                self.dp__[m+1] = ufl.TrialFunction(self.V_p__[m+1])            # Incremental pressure
                self.var_p__[m+1] = ufl.TestFunction(self.V_p__[m+1])          # Test function
                # values of previous time step
                self.p_old__[m+1] = fem.Function(self.V_p__[m+1])
            # make lists
            self.V_p_ = list(self.V_p__.values())
            self.p_, self.dp_, self.var_p_, self.p_old_ = list(self.p__.values()), list(self.dp__.values()), list(self.var_p__.values()), list(self.p_old__.values())
        else:
            self.p_ = [ fem.Function(self.V_p_[0], name="Pressure") ]
            self.dp_ = [ ufl.TrialFunction(self.V_p_[0]) ]            # Incremental pressure
            self.var_p_ = [ ufl.TestFunction(self.V_p_[0]) ]          # Test function
            # values of previous time step
            self.p_old_ = [ fem.Function(self.V_p_[0]) ]

        # values of previous time step
        self.v_old  = fem.Function(self.V_v)
        self.a_old  = fem.Function(self.V_v)
        # auxiliary acceleration vector
        self.a      = fem.Function(self.V_v, name="Acceleration")
        # a fluid displacement
        self.uf     = fem.Function(self.V_v, name="FluidDisplacement")
        self.uf_old = fem.Function(self.V_v)
        # active stress for reduced solid
        self.tau_a  = fem.Function(self.Vd_scalar, name="tau_a")
        self.tau_a_old = fem.Function(self.Vd_scalar)
        # prestress displacement for FrSI
        if (self.prestress_initial or self.prestress_initial_only) or bool(self.prestress_from_file):
            self.uf_pre = fem.Function(self.V_v, name="Displacement_prestress")
        else:
            self.uf_pre = None
        if (self.prestress_initial or self.prestress_initial_only) and self.pbase.restart_step == 0:
            self.pre = True
        else:
            self.pre = False

        # for ROM, provide pointers to main variable, its derivative, and possibly its time integrated value
        if self.pbase.have_rom:
            self.xr_, self.xr_old_, self.xrpre_ = self.v, self.v_old, None
            self.xdtr_old_, self.xintr_old_, self.xintrpre_ = self.a_old, self.uf_old, self.uf_pre

        # collect references to pressure vectors
        self.pvecs_, self.pvecs_old_ = [], []
        if self.num_dupl > 1:
            for m in range(self.num_dupl):
                self.pvecs_.append(self.p_[m].x.petsc_vec)
                self.pvecs_old_.append(self.p_old_[m].x.petsc_vec)
                # full pressure vectors - dummy function that holds a class variable "vector"
                self.p = expression.function_dummy(self.pvecs_,self.comm)
                self.p_old = expression.function_dummy(self.pvecs_old_,self.comm)
        else:
            self.V_p, self.p, self.p_old = self.V_p_[0], self.p_[0], self.p_old_[0] # pointer to first p's...

        # if we want to initialize the pressure (domain wise) with a scalar value
        if self.pbase.restart_step==0:
            if bool(self.initial_fluid_pressure):
                for m in range(self.num_dupl):
                    val = self.initial_fluid_pressure[m]
                    self.p_[m].x.petsc_vec.set(val)
                    self.p_[m].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    self.p_old_[m].x.petsc_vec.set(val)
                    self.p_old_[m].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # own read function: requires plain txt format of type "node-id val-x val-y val-z" (or one value in case of a scalar)
        if bool(self.prestress_from_file):
            self.io.readfunction(self.uf_pre, self.prestress_from_file[0])
            # if available, we might want to read in the pressure field, too
            if len(self.prestress_from_file)>1:
                if bool(self.duplicate_mesh_domains):
                    for m, mp in enumerate(self.io.duplicate_mesh_domains):
                        self.io.readfunction(self.p_[m], self.prestress_from_file[1].replace('*',str(m+1)))
                        self.io.readfunction(self.p_old_[m], self.prestress_from_file[1].replace('*',str(m+1)))
                else:
                    self.io.readfunction(self.p_[0], self.prestress_from_file[1])
                    self.io.readfunction(self.p_old_[0], self.prestress_from_file[1])

        # dictionaries of internal variables
        self.internalvars, self.internalvars_old, self.internalvars_mid = {}, {}, {}

        # reference coordinates
        self.x_ref = ufl.SpatialCoordinate(self.io.mesh)

        self.numdof = self.v.x.petsc_vec.getSize() + self.p.x.petsc_vec.getSize()

        # initialize fluid time-integration class
        self.ti = timeintegration.timeintegration_fluid(time_params, self.pbase.dt, self.pbase.numstep, time_curves=time_curves, t_init=self.pbase.t_init, dim=self.dim, comm=self.comm)

        # get time factors
        self.timefac_m, self.timefac = self.ti.timefactors()
        if self.ti.eval_nonlin_terms=='midpoint': self.midp = True
        else: self.midp = False

        # initialize kinematics_constitutive class
        self.ki = fluid_kinematics_constitutive.kinematics(self.dim, uf_pre=self.uf_pre)

        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(fluid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)]))

        # initialize fluid variational form class
        if not bool(self.alevar):
            # standard Eulerian fluid
            self.alevar = {'Fale' : None, 'Fale_old' : None, 'Fale_mid' : None, 'w' : None, 'w_old' : None, 'w_mid' : None}
            self.vf = fluid_variationalform.variationalform(self.var_v, var_p=self.var_p_, n0=self.io.n0, formulation=self.fluid_formulation)

        else:
            # mid-point representation of ALE velocity
            self.alevar['w_mid']    = self.timefac * self.alevar['w']    + (1.-self.timefac) * self.alevar['w_old']
            # mid-point representation of ALE deformation gradient - linear in ALE displacement, hence we can combine it like this
            self.alevar['Fale_mid'] = self.timefac * self.alevar['Fale'] + (1.-self.timefac) * self.alevar['Fale_old']

            # fully consistent ALE formulation of Navier-Stokes
            self.vf = fluid_variationalform.variationalform_ale(self.var_v, var_p=self.var_p_, n0=self.io.n0, formulation=self.fluid_formulation)

        # read in fiber data - for reduced solid (FrSI)
        if bool(self.io.fiber_data) and (self.pbase.problem_type=='fluid_ale' or self.pbase.problem_type=='fluid_ale_flow0d' or self.pbase.problem_type=='fluid_ale_constraint'): # only for FrSI problem

            self.fibarray = ['circ']
            if len(self.io.fiber_data)>1: self.fibarray.append('long')

            self.fib_func = self.io.readin_fibers(self.fibarray, self.V_v, self.dx, self.domain_ids, self.order_vel)

        else:
            self.fib_func = None

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_fluid(self.io, fem_params=fem_params, vf=self.vf, ti=self.ti, ki=self.ki, ff=self.fib_func, V_field=self.V_v, Vdisc_scalar=self.Vd_scalar)

        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.bc_dict['dirichlet'])

        if 'dirichlet_vol' in self.bc_dict.keys():
            self.bc.dirichlet_vol(self.bc_dict['dirichlet_vol'])

        self.set_variational_forms()

        self.pbrom = self # self-pointer needed for ROM solver access
        self.pbrom_host = self
        self.V_rom = self.V_v

        # number of fields involved
        self.nfields = 2
        if self.have_robin_valve_implicit: self.nfields+=1

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)],  [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.num_dupl > 1: is_ghosted = [1, 2]
        else:                 is_ghosted = [1, 1]
        varlist = [self.v.x.petsc_vec, self.p.x.petsc_vec]

        if self.have_robin_valve_implicit:
            varlist.append(self.z)
            is_ghosted.append(0)

        return varlist, is_ghosted


    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):

        # set form for acceleration
        self.acc = self.ti.set_acc(self.v, self.v_old, self.a_old)
        # set form for fluid displacement (needed for FrSI)
        self.ufluid = self.ti.set_uf(self.v, self.v_old, self.uf_old)

        # set mid-point representations
        self.acc_mid     = self.timefac_m * self.acc    + (1.-self.timefac_m) * self.a_old
        self.vel_mid     = self.timefac   * self.v      + (1.-self.timefac)   * self.v_old
        self.ufluid_mid  = self.timefac   * self.ufluid + (1.-self.timefac)   * self.uf_old

        self.pf_mid__ = {}
        if self.num_dupl > 1:
            for m, mp in enumerate(self.io.duplicate_mesh_domains):
                self.pf_mid__[m+1] = self.timefac * self.p__[m+1] + (1.-self.timefac) * self.p_old__[m+1]
            # make list
            self.pf_mid_ = list(self.pf_mid__.values())
        else:
            self.pf_mid_ = [ self.timefac * self.p_[0] + (1.-self.timefac) * self.p_old_[0] ]

        # kinetic, internal, and pressure virtual power
        self.deltaW_kin, self.deltaW_kin_old, self.deltaW_kin_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_int, self.deltaW_int_old, self.deltaW_int_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_p,   self.deltaW_p_old,   self.deltaW_p_mid   = [], [], []
        # element level Reynolds number components
        self.Re_c,      self.Re_c_old,        self.Re_c_mid       = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        self.Re_ktilde, self.Re_ktilde_old,   self.Re_ktilde_mid  = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

        for n, M in enumerate(self.domain_ids):

            if self.num_dupl==1: j=0
            else: j=n

            # kinetic virtual power
            if self.fluid_governing_type=='navierstokes_transient':
                self.deltaW_kin     += self.vf.deltaW_kin_navierstokes_transient(self.acc, self.v, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_navierstokes_transient(self.a_old, self.v_old, self.rho[n], self.dx(M), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
                self.deltaW_kin_mid += self.vf.deltaW_kin_navierstokes_transient(self.acc_mid, self.vel_mid, self.rho[n], self.dx(M), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
            elif self.fluid_governing_type=='navierstokes_steady':
                self.deltaW_kin     += self.vf.deltaW_kin_navierstokes_steady(self.v, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_navierstokes_steady(self.v_old, self.rho[n], self.dx(M), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
                self.deltaW_kin_mid += self.vf.deltaW_kin_navierstokes_steady(self.vel_mid, self.rho[n], self.dx(M), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
            elif self.fluid_governing_type=='stokes_transient':
                self.deltaW_kin     += self.vf.deltaW_kin_stokes_transient(self.acc, self.v, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_stokes_transient(self.a_old, self.v_old, self.rho[n], self.dx(M), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
                self.deltaW_kin_mid += self.vf.deltaW_kin_stokes_transient(self.acc_mid, self.vel_mid, self.rho[n], self.dx(M), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
            elif self.fluid_governing_type=='stokes_steady':
                pass # no kinetic term to add for steady Stokes flow
            else:
                raise ValueError("Unknown fluid_governing_type!")

            # internal virtual power
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), self.dx(M), F=self.alevar['Fale'])
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].sigma(self.v_old, self.p_old_[j], F=self.alevar['Fale_old']), self.dx(M), F=self.alevar['Fale_old'])
            self.deltaW_int_mid += self.vf.deltaW_int(self.ma[n].sigma(self.vel_mid, self.pf_mid_[j], F=self.alevar['Fale_mid']), self.dx(M), F=self.alevar['Fale_mid'])

            # pressure virtual power
            self.deltaW_p.append( self.vf.deltaW_int_pres(self.v, self.var_p_[j], self.dx_p[j](M), F=self.alevar['Fale']) )
            self.deltaW_p_old.append( self.vf.deltaW_int_pres(self.v_old, self.var_p_[j], self.dx_p[j](M), F=self.alevar['Fale_old']) )
            self.deltaW_p_mid.append( self.vf.deltaW_int_pres(self.vel_mid, self.var_p_[j], self.dx_p[j](M), F=self.alevar['Fale_mid']) )

            # element level Reynolds number components - not used so far! Need to assemble a cell-based vector in order to evaluate these...
            self.Re_c      += self.vf.re_c(self.rho[n], self.v, self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
            self.Re_c_old  += self.vf.re_c(self.rho[n], self.v_old, self.dx(M), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
            self.Re_c_mid  += self.vf.re_c(self.rho[n], self.vel_mid, self.dx(M), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])

            self.Re_ktilde     += self.vf.re_ktilde(self.rho[n], self.v, self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
            self.Re_ktilde_old += self.vf.re_ktilde(self.rho[n], self.v_old, self.dx(M), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
            self.Re_ktilde_mid += self.vf.re_ktilde(self.rho[n], self.vel_mid, self.dx(M), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])

        # external virtual power (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_body, w_robin, w_stabneumann, w_stabneumann_mod, w_robin_valve, w_membrane = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        w_neumann_old, w_body_old, w_robin_old, w_stabneumann_old, w_stabneumann_mod_old, w_robin_valve_old, w_membrane_old = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        w_neumann_mid, w_body_mid, w_robin_mid, w_stabneumann_mid, w_stabneumann_mod_mid, w_robin_valve_mid, w_membrane_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann     = self.bc.neumann_bcs(self.bc_dict['neumann'], self.bmeasures, F=self.alevar['Fale'], funcs_to_update=self.ti.funcs_to_update, funcs_to_update_vec=self.ti.funcs_to_update_vec, funcsexpr_to_update=self.ti.funcsexpr_to_update, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec)
            w_neumann_old = self.bc.neumann_bcs(self.bc_dict['neumann'], self.bmeasures, F=self.alevar['Fale_old'], funcs_to_update=self.ti.funcs_to_update_old, funcs_to_update_vec=self.ti.funcs_to_update_vec_old, funcsexpr_to_update=self.ti.funcsexpr_to_update_old, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec_old)
            w_neumann_mid = self.bc.neumann_bcs(self.bc_dict['neumann'], self.bmeasures, F=self.alevar['Fale_mid'], funcs_to_update=self.ti.funcs_to_update_mid, funcs_to_update_vec=self.ti.funcs_to_update_vec_mid, funcsexpr_to_update=self.ti.funcsexpr_to_update_mid, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec_mid)
        if 'bodyforce' in self.bc_dict.keys():
            w_body      = self.bc.bodyforce(self.bc_dict['bodyforce'], self.dx, F=self.alevar['Fale'], funcs_to_update=self.ti.funcs_to_update, funcsexpr_to_update=self.ti.funcsexpr_to_update)
            w_body_old  = self.bc.bodyforce(self.bc_dict['bodyforce'], self.dx, F=self.alevar['Fale_old'], funcs_to_update=self.ti.funcs_to_update_old, funcsexpr_to_update=self.ti.funcsexpr_to_update_old)
            w_body_mid  = self.bc.bodyforce(self.bc_dict['bodyforce'], self.dx, F=self.alevar['Fale_mid'], funcs_to_update=self.ti.funcs_to_update_mid, funcsexpr_to_update=self.ti.funcsexpr_to_update_mid)
        if 'robin' in self.bc_dict.keys():
            w_robin     = self.bc.robin_bcs(self.bc_dict['robin'], self.ufluid, self.v, self.bmeasures, F=self.alevar['Fale'], u_pre=self.uf_pre)
            w_robin_old = self.bc.robin_bcs(self.bc_dict['robin'], self.uf_old, self.v_old, self.bmeasures, F=self.alevar['Fale_old'], u_pre=self.uf_pre)
            w_robin_mid = self.bc.robin_bcs(self.bc_dict['robin'], self.ufluid_mid, self.vel_mid, self.bmeasures, F=self.alevar['Fale_mid'], u_pre=self.uf_pre)
        if 'stabilized_neumann' in self.bc_dict.keys():
            w_stabneumann     = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.v, self.bmeasures, wel=self.alevar['w'], F=self.alevar['Fale'])
            w_stabneumann_old = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.v_old, self.bmeasures, wel=self.alevar['w_old'], F=self.alevar['Fale_old'])
            w_stabneumann_mid = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.vel_mid, self.bmeasures, wel=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
        if 'stabilized_neumann_mod' in self.bc_dict.keys():
            w_stabneumann_mod     = self.bc.stabilized_neumann_mod_bcs(self.bc_dict['stabilized_neumann_mod'], self.v, self.bmeasures, wel=self.alevar['w'], F=self.alevar['Fale'])
            w_stabneumann_mod_old = self.bc.stabilized_neumann_mod_bcs(self.bc_dict['stabilized_neumann_mod'], self.v_old, self.bmeasures, wel=self.alevar['w_old'], F=self.alevar['Fale_old'])
            w_stabneumann_mod_mid = self.bc.stabilized_neumann_mod_bcs(self.bc_dict['stabilized_neumann_mod'], self.vel_mid, self.bmeasures, wel=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
        if 'robin_valve' in self.bc_dict.keys():
            assert(self.num_dupl>1) # only makes sense if we have duplicate pressure domains
            self.have_robin_valve = True
            self.beta_valve, self.beta_valve_old = [], []
            w_robin_valve     = self.bc.robin_valve_bcs(self.bc_dict['robin_valve'], self.v, self.beta_valve, [self.bmeasures[2]], wel=self.alevar['w'], F=self.alevar['Fale'])
            w_robin_valve_old = self.bc.robin_valve_bcs(self.bc_dict['robin_valve'], self.v_old, self.beta_valve_old, [self.bmeasures[2]], wel=self.alevar['w_old'], F=self.alevar['Fale_old'])
            w_robin_valve_mid = self.bc.robin_valve_bcs(self.bc_dict['robin_valve'], self.vel_mid, self.beta_valve, [self.bmeasures[2]], wel=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
            # in case we manna make the min and max values spatially dependent (usage of an expression), prepare some functions
            self.beta_valve_min_expr, self.beta_valve_max_expr = [], []
            for m in range(len(self.bc_dict['robin_valve'])):
                beta_min, beta_max = self.bc_dict['robin_valve'][m]['beta_min'], self.bc_dict['robin_valve'][m]['beta_max']
                # check if we have passed an expression (class) or a constant value
                if inspect.isclass(beta_min):
                    self.beta_valve_min_expr.append(beta_min())
                else:
                    self.beta_valve_min_expr.append(expression.template())
                    self.beta_valve_min_expr[-1].val = beta_min
                # check if we have passed an expression (class) or a constant value
                if inspect.isclass(beta_max):
                    self.beta_valve_max_expr.append(beta_max())
                else:
                    self.beta_valve_max_expr.append(expression.template())
                    self.beta_valve_max_expr[-1].val = beta_max
        if 'robin_valve_implicit' in self.bc_dict.keys():
            assert(self.num_dupl>1) # only makes sense if we have duplicate pressure domains
            self.have_robin_valve_implicit = True
            raise RuntimeError("Implicit valve law not yet fully implemented!")
            self.state_valve, self.state_valve_old = [], []
            w_robin_valve     = self.bc.robin_valve_bcs(self.bc_dict['robin_valve_implicit'], self.v, self.state_valve, [self.bmeasures[2]], wel=self.alevar['w'], F=self.alevar['Fale'])
            w_robin_valve_old = self.bc.robin_valve_bcs(self.bc_dict['robin_valve_implicit'], self.v_old, self.state_valve_old, [self.bmeasures[2]], wel=self.alevar['w_old'], F=self.alevar['Fale_old'])
            w_robin_valve_mid = self.bc.robin_valve_bcs(self.bc_dict['robin_valve_implicit'], self.vel_mid, self.state_valve, [self.bmeasures[2]], wel=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
            self.dbeta_dz_valve = []
            self.dw_robin_valve_dz = []
            for i in range(len(self.beta_valve)):
                self.bc.robin_valve_bcs(self.bc_dict['robin_valve_implicit'], self.v, self.dbeta_dz_valve, [self.bmeasures[2]], wel=self.alevar['w'], F=self.alevar['Fale'], dw=self.dw_robin_valve_dz)
            # reduced valve variables (integrated pressures)
            self.num_valve_coupling_surf = len(self.bc_dict['robin_valve_implicit'])
            self.z, self.z_old = PETSc.Vec().createMPI(size=self.num_valve_coupling_surf), PETSc.Vec().createMPI(size=self.num_valve_coupling_surf)
            self.surface_vlv_ids = []
            for i in range(self.num_valve_coupling_surf):
                self.surface_vlv_ids.append(self.bc_dict['robin_valve_implicit'][i]['id'])
        if 'flux_monitor' in self.bc_dict.keys():
            self.have_flux_monitor = True
            self.q_, self.q_old_, self.q_mid_ = [], [], []
            self.bc.flux_monitor_bcs(self.bc_dict['flux_monitor'], self.v, self.q_, wel=self.alevar['w'], F=self.alevar['Fale'])
            self.bc.flux_monitor_bcs(self.bc_dict['flux_monitor'], self.v_old, self.q_old_, wel=self.alevar['w_old'], F=self.alevar['Fale_old'])
            self.bc.flux_monitor_bcs(self.bc_dict['flux_monitor'], self.vel_mid, self.q_mid_, wel=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
        if 'dp_monitor' in self.bc_dict.keys():
            assert(self.num_dupl>1) # only makes sense if we have duplicate pressure domains
            self.have_dp_monitor = True
            self.a_u_, self.a_d_, self.pint_u_, self.pint_d_ = [], [], [], []
            self.a_u_old_, self.a_d_old_, self.pint_u_old_, self.pint_d_old_ = [], [], [], []
            self.a_u_mid_, self.a_d_mid_, self.pint_u_mid_, self.pint_d_mid_ = [], [], [], []
            self.bc.dp_monitor_bcs(self.bc_dict['dp_monitor'], self.a_u_, self.a_d_, self.pint_u_, self.pint_d_, self.p__, F=self.alevar['Fale'])
            self.bc.dp_monitor_bcs(self.bc_dict['dp_monitor'], self.a_u_old_, self.a_d_old_, self.pint_u_old_, self.pint_d_old_, self.p_old__, F=self.alevar['Fale_old'])
            self.bc.dp_monitor_bcs(self.bc_dict['dp_monitor'], self.a_u_mid_, self.a_d_mid_, self.pint_u_mid_, self.pint_d_mid_, self.pf_mid__, F=self.alevar['Fale_mid'])

        self.mem_active_stress = [False]

        # reduced-solid for FrSI problem
        if 'membrane' in self.bc_dict.keys():

            self.mem_active_stress, self.mem_active_stress_type = [False]*len(self.bc_dict['membrane']), ['ode']*len(self.bc_dict['membrane'])

            self.internalvars['tau_a'], self.internalvars_old['tau_a'], self.internalvars_mid['tau_a'] = self.tau_a, self.tau_a_old, self.timefac*self.tau_a + (1.-self.timefac)*self.tau_a_old

            self.actstress, self.act_curve, self.wallfields, self.actweights = [], [], [], []
            for nm in range(len(self.bc_dict['membrane'])):

                if 'active_stress' in self.bc_dict['membrane'][nm]['params'].keys():
                    self.mem_active_stress[nm] = True

                    try: self.mem_active_stress_type[nm] = self.bc_dict['membrane'][nm]['params']['active_stress']['type']
                    except: pass # default is 'ode'

                    if self.mem_active_stress_type[nm] == 'ode':
                        self.act_curve.append( fem.Function(self.Vd_scalar) )
                        self.ti.funcs_to_update.append({self.act_curve[-1] : self.ti.timecurves(self.bc_dict['membrane'][nm]['params']['active_stress']['activation_curve'])})
                        self.ti.funcs_to_update_old.append({None : -1}) # not needed, since tau_a_old <- tau_a at end of time step
                        # the active stress ODE class
                        self.actstress.append(activestress_activation(self.bc_dict['membrane'][nm]['params']['active_stress'], self.act_curve[-1], x_ref=self.x_ref))
                    elif self.mem_active_stress_type[nm] == 'prescribed': # here we use act_curve as the prescibed stress directly
                        self.act_curve.append( fem.Function(self.Vd_scalar) )
                        self.ti.funcs_to_update.append({self.act_curve[-1] : self.ti.timecurves(self.bc_dict['membrane'][nm]['params']['active_stress']['prescribed_curve'])})
                        self.ti.funcs_to_update_old.append({None : -1}) # not needed, since tau_a_old <- tau_a at end of time step
                    else:
                        raise NameError("Unknown active stress type for membrane!")

                    if 'weight' in self.bc_dict['membrane'][nm]['params']['active_stress'].keys():
                        # active stress weighting for reduced solid
                        wact_func = fem.Function(self.V_scalar)
                        self.io.readfunction(wact_func, self.bc_dict['membrane'][nm]['params']['active_stress']['weight'])
                        self.actweights.append(wact_func)
                    else:
                        self.actweights.append(None)
                else:
                    self.actweights.append(None)

                if 'field' in self.bc_dict['membrane'][nm]['params']['h0'].keys():
                    # wall thickness field for reduced solid
                    h0_func = fem.Function(self.V_scalar)
                    self.io.readfunction(h0_func, self.bc_dict['membrane'][nm]['params']['h0']['field'])
                    self.wallfields.append(h0_func)
                else:
                    self.wallfields.append(None)

            w_membrane, self.idmem, self.bstress, self.bstrainenergy, self.bintpower = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.ufluid, self.v, self.acc, self.bmeasures, ivar=self.internalvars, wallfields=self.wallfields, actweights=self.actweights)
            w_membrane_old, _, _, _, _                                               = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.uf_old, self.v_old, self.a_old, self.bmeasures, ivar=self.internalvars_old, wallfields=self.wallfields, actweights=self.actweights)
            w_membrane_mid, _, _, _, _                                               = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.ufluid_mid, self.vel_mid, self.acc_mid, self.bmeasures, ivar=self.internalvars_mid, wallfields=self.wallfields, actweights=self.actweights)

        w_neumann_prestr, self.deltaW_prestr_kin, self.deltaW_prestr_int, self.deltaW_p_prestr = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), []
        if self.prestress_initial or self.prestress_initial_only:
            self.acc_prestr = (self.v - self.v_old)/self.prestress_dt # in case acceleration is used (for kinetic prestress option)
            for n, M in enumerate(self.domain_ids):
                if self.num_dupl==1: j=0
                else: j=n
                self.deltaW_prestr_int += self.vf.deltaW_int(self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), self.dx(M), F=self.alevar['Fale'])
                self.deltaW_p_prestr.append( self.vf.deltaW_int_pres(self.v, self.var_p_[j], self.dx(M), F=self.alevar['Fale']) )
                # it seems that we need some slight inertia for this to work smoothly, so let's use transient Stokes here (instead of steady Navier-Stokes or steady Stokes...)
                if self.prestress_kinetic=='navierstokes_transient':
                    self.deltaW_prestr_kin += self.vf.deltaW_kin_navierstokes_transient(self.acc_prestr, self.v, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
                elif self.prestress_kinetic=='navierstokes_steady':
                    self.deltaW_prestr_kin += self.vf.deltaW_kin_navierstokes_steady(self.v, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
                elif self.prestress_kinetic=='stokes_transient':
                    self.deltaW_prestr_kin += self.vf.deltaW_kin_stokes_transient(self.acc_prestr, self.v, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'])
                elif self.prestress_kinetic=='none':
                    pass
                else:
                    raise ValueError("Unknown prestress_kinetic option. Choose either 'navierstokes_transient', 'navierstokes_steady', 'stokes_transient', or 'none'.")
            if 'neumann_prestress' in self.bc_dict.keys():
                w_neumann_prestr = self.bc.neumann_prestress_bcs(self.bc_dict['neumann_prestress'], self.bmeasures, funcs_to_update=self.ti.funcs_to_update_pre, funcs_to_update_vec=self.ti.funcs_to_update_vec_pre, funcsexpr_to_update=self.ti.funcsexpr_to_update_pre, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec_pre)
            if 'membrane' in self.bc_dict.keys():
                self.ufluid_prestr = self.v * self.prestress_dt # only incremental displacement needed, since MULF update actually yields a zero displacement after the step
                w_membrane_prestr, _, _, _, _ = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.ufluid_prestr, self.v, self.acc_prestr, self.bmeasures, ivar=self.internalvars, wallfields=self.wallfields)
            self.deltaW_prestr_ext = w_neumann_prestr + w_robin + w_stabneumann + w_stabneumann_mod + w_membrane_prestr + w_robin_valve
        else:
            assert('neumann_prestress' not in self.bc_dict.keys())

        self.deltaW_ext     = w_neumann + w_body + w_robin + w_stabneumann + w_stabneumann_mod + w_membrane + w_robin_valve
        self.deltaW_ext_old = w_neumann_old + w_body_old + w_robin_old + w_stabneumann_old + w_stabneumann_mod_old + w_membrane_old + w_robin_valve_old
        self.deltaW_ext_mid = w_neumann_mid + w_body_mid + w_robin_mid + w_stabneumann_mid + w_stabneumann_mod_mid + w_membrane_mid + w_robin_valve_mid

        # stabilization
        if self.stabilization is not None:

            # should only be used for equal-order approximations
            assert(self.order_vel==self.order_pres)

            vscale = self.stabilization['vscale']
            vscale_vel_dep = self.stabilization.get('vscale_vel_dep', False)
            vscale_amp = self.stabilization.get('vscale_amp', 1.0)

            # TODO: Is this a good choice in general if we wanna make it state dependent?
            if vscale_vel_dep:
                vscale_max = vscale_amp * ufl.max_value(ufl.sqrt(ufl.dot(self.v_old,self.v_old)), vscale)
            else:
                vscale_max = vscale_amp * vscale

            h = self.io.hd0 # cell diameter (could also use max edge length self.io.emax0, but seems to yield similar/same results)

            symm = self.stabilization.get('symmetric', False)

            # reduced stabiliztion scheme optimized for first-order: missing transient NS term as well as divergence stress term of strong residual
            red_scheme = self.stabilization.get('reduced_scheme', False)

            dscales = self.stabilization.get('dscales', [1., 1., 1.])

            if red_scheme:
                assert(self.order_vel==1)
                assert(self.order_pres==1)

            # full scheme
            if self.stabilization['scheme']=='supg_pspg':

                for n, M in enumerate(self.domain_ids):

                    if self.num_dupl==1: j=0
                    else: j=n

                    dscales = self.stabilization['dscales']

                    tau_supg = dscales[0] * h / vscale_max
                    tau_lsic = dscales[1] * h * vscale_max
                    tau_pspg = dscales[2] * h / vscale_max

                    # strong momentum residuals
                    if self.fluid_governing_type=='navierstokes_transient':
                        if not red_scheme:
                            residual_v_strong     = self.vf.res_v_strong_navierstokes_transient(self.acc, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), w=self.alevar['w'], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.res_v_strong_navierstokes_transient(self.a_old, self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], F=self.alevar['Fale_old']), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.res_v_strong_navierstokes_transient(self.acc_mid, self.vel_mid, self.rho[n], self.ma[n].sigma(self.vel_mid, self.pf_mid_[j], F=self.alevar['Fale_mid']), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
                        else: # no viscous stress term and no dv/dt term
                            residual_v_strong     = self.vf.f_inert_strong_navierstokes_steady(self.v, self.rho[n], w=self.alevar['w'], F=self.alevar['Fale']) + self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.f_inert_strong_navierstokes_steady(self.v_old, self.rho[n], w=self.alevar['w_old'], F=self.alevar['Fale_old']) + self.vf.f_gradp_strong(self.p_old_[j], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.f_inert_strong_navierstokes_steady(self.vel_mid, self.rho[n], w=self.alevar['w_mid'], F=self.alevar['Fale_mid']) + self.vf.f_gradp_strong(self.pf_mid_[j], F=self.alevar['Fale_mid'])
                    elif self.fluid_governing_type=='navierstokes_steady':
                        if not red_scheme:
                            residual_v_strong     = self.vf.res_v_strong_navierstokes_steady(self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), w=self.alevar['w'], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.res_v_strong_navierstokes_steady(self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], F=self.alevar['Fale_old']), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.res_v_strong_navierstokes_steady(self.vel_mid, self.rho[n], self.ma[n].sigma(self.vel_mid, self.pf_mid_[j], F=self.alevar['Fale_mid']), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
                        else: # no viscous stress term
                            residual_v_strong     = self.vf.f_inert_strong_navierstokes_steady(self.v, self.rho[n], w=self.alevar['w'], F=self.alevar['Fale']) + self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.f_inert_strong_navierstokes_steady(self.v_old, self.rho[n], w=self.alevar['w_old'], F=self.alevar['Fale_old']) + self.vf.f_gradp_strong(self.p_old_[j], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.f_inert_strong_navierstokes_steady(self.vel_mid, self.rho[n], w=self.alevar['w_mid'], F=self.alevar['Fale_mid']) + self.vf.f_gradp_strong(self.pf_mid_[j], F=self.alevar['Fale_mid'])
                    elif self.fluid_governing_type=='stokes_transient':
                        if not red_scheme:
                            residual_v_strong     = self.vf.res_v_strong_stokes_transient(self.acc, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), w=self.alevar['w'], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.res_v_strong_stokes_transient(self.a_old, self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], F=self.alevar['Fale_old']), w=self.alevar['w_old'], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.res_v_strong_stokes_transient(self.acc_mid, self.vel_mid, self.rho[n], self.ma[n].sigma(self.vel_mid, self.pf_mid_[j], F=self.alevar['Fale_mid']), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'])
                        else: # no viscous stress term
                            residual_v_strong     = self.vf.f_inert_strong_stokes_transient(self.acc, self.v, self.rho[n], w=self.alevar['w'], F=self.alevar['Fale']) + self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.f_inert_strong_stokes_transient(self.a_old, self.v_old, self.rho[n], w=self.alevar['w_old'], F=self.alevar['Fale_old']) + self.vf.f_gradp_strong(self.p_old_[j], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.f_inert_strong_stokes_transient(self.acc_mid, self.vel_mid, self.rho[n], w=self.alevar['w_mid'], F=self.alevar['Fale_mid']) + self.vf.f_gradp_strong(self.pf_mid_[j], F=self.alevar['Fale_mid'])
                    elif self.fluid_governing_type=='stokes_steady':
                        if not red_scheme:
                            residual_v_strong     = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], F=self.alevar['Fale_old']), F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.vel_mid, self.pf_mid_[j], F=self.alevar['Fale_mid']), F=self.alevar['Fale_mid'])
                        else: # no viscous stress term
                            residual_v_strong     = self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                            residual_v_strong_old = self.vf.f_gradp_strong(self.p_old_[j], F=self.alevar['Fale_old'])
                            residual_v_strong_mid = self.vf.f_gradp_strong(self.pf_mid_[j], F=self.alevar['Fale_mid'])
                    else:
                        raise ValueError("Unknown fluid_governing_type!")

                    # SUPG (streamline-upwind Petrov-Galerkin) for Navier-Stokes
                    if self.fluid_governing_type=='navierstokes_transient' or self.fluid_governing_type=='navierstokes_steady':
                        self.deltaW_int     += self.vf.stab_supg(self.v, residual_v_strong, tau_supg, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'], symmetric=symm)
                        self.deltaW_int_old += self.vf.stab_supg(self.v_old, residual_v_strong_old, tau_supg, self.rho[n], self.dx(M), w=self.alevar['w_old'], F=self.alevar['Fale_old'], symmetric=symm)
                        self.deltaW_int_mid += self.vf.stab_supg(self.vel_mid, residual_v_strong_mid, tau_supg, self.rho[n], self.dx(M), w=self.alevar['w_mid'], F=self.alevar['Fale_mid'], symmetric=symm)
                    # LSIC (least-squares on incompressibility constraint) for Navier-Stokes and Stokes
                    self.deltaW_int     += self.vf.stab_lsic(self.v, tau_lsic, self.rho[n], self.dx(M), F=self.alevar['Fale'])
                    self.deltaW_int_old += self.vf.stab_lsic(self.v_old, tau_lsic, self.rho[n], self.dx(M), F=self.alevar['Fale_old'])
                    self.deltaW_int_mid += self.vf.stab_lsic(self.vel_mid, tau_lsic, self.rho[n], self.dx(M), F=self.alevar['Fale_mid'])
                    # PSPG (pressure-stabilizing Petrov-Galerkin) for Navier-Stokes and Stokes
                    self.deltaW_p[n]     += self.vf.stab_pspg(self.var_p_[j], residual_v_strong, tau_pspg, self.rho[n], self.dx_p[j](M), F=self.alevar['Fale'])
                    self.deltaW_p_old[n] += self.vf.stab_pspg(self.var_p_[j], residual_v_strong_old, tau_pspg, self.rho[n], self.dx_p[j](M), F=self.alevar['Fale_old'])
                    self.deltaW_p_mid[n] += self.vf.stab_pspg(self.var_p_[j], residual_v_strong_mid, tau_pspg, self.rho[n], self.dx_p[j](M), F=self.alevar['Fale_mid'])

                    # now take care of stabilization for the prestress problem (only FrSI)
                    if self.prestress_initial or self.prestress_initial_only:
                        # LSIC term
                        self.deltaW_prestr_int += self.vf.stab_lsic(self.v, tau_lsic, self.rho[n], self.dx(M), F=self.alevar['Fale'])
                        # get the respective strong residual depending on the prestress kinetic type...
                        if self.prestress_kinetic=='navierstokes_transient':
                            if not red_scheme:
                                residual_v_strong_prestr = self.vf.res_v_strong_navierstokes_transient(self.acc_prestr, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), w=self.alevar['w'], F=self.alevar['Fale'])
                            else: # no viscous stress term and no dv/dt term
                                residual_v_strong_prestr = self.vf.f_inert_strong_navierstokes_steady(self.v, self.rho[n], w=self.alevar['w'], F=self.alevar['Fale']) + self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                        elif self.prestress_kinetic=='navierstokes_steady':
                            if not red_scheme:
                                residual_v_strong_prestr = self.vf.res_v_strong_navierstokes_steady(self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), w=self.alevar['w'], F=self.alevar['Fale'])
                            else: # no viscous stress term
                                residual_v_strong_prestr = self.vf.f_inert_strong_navierstokes_steady(self.v, self.rho[n], w=self.alevar['w'], F=self.alevar['Fale']) + self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                        elif self.prestress_kinetic=='stokes_transient':
                            if not red_scheme:
                                residual_v_strong_prestr = self.vf.res_v_strong_stokes_transient(self.acc_prestr, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), w=self.alevar['w'], F=self.alevar['Fale'])
                            else: # no viscous stress term
                                residual_v_strong_prestr = self.vf.f_inert_strong_stokes_transient(self.acc_prestr, self.v, self.rho[n], w=self.alevar['w'], F=self.alevar['Fale']) + self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                        elif self.prestress_kinetic=='none':
                            if not red_scheme:
                                residual_v_strong_prestr = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), F=self.alevar['Fale'])
                            else: # no viscous stress term
                                residual_v_strong_prestr = self.vf.f_gradp_strong(self.p_[j], F=self.alevar['Fale'])
                        else:
                            raise ValueError("Unknown prestress_kinetic option!")
                        # PSPG term
                        self.deltaW_p_prestr[n] += self.vf.stab_pspg(self.var_p_[j], residual_v_strong_prestr, tau_pspg, self.rho[n], self.dx(M), F=self.alevar['Fale'])
                        # SUPG term only for kinetic prestress...
                        if self.prestress_kinetic=='navierstokes_transient' or self.prestress_kinetic=='navierstokes_steady':
                            self.deltaW_prestr_int += self.vf.stab_supg(self.v, residual_v_strong_prestr, tau_supg, self.rho[n], self.dx(M), w=self.alevar['w'], F=self.alevar['Fale'], symmetric=symm)

            else:
                raise ValueError("Unknown stabilization scheme!")


        ### full weakforms

        # kinetic plus internal minus external virtual power
        # evaluate nonlinear terms trapezoidal-like: a * f(u_{n+1}) + (1-a) * f(u_{n})
        if self.ti.eval_nonlin_terms=='trapezoidal':

            self.weakform_v = self.timefac_m * self.deltaW_kin + (1.-self.timefac_m) * self.deltaW_kin_old + \
                              self.timefac   * self.deltaW_int + (1.-self.timefac)   * self.deltaW_int_old - \
                              self.timefac   * self.deltaW_ext - (1.-self.timefac)   * self.deltaW_ext_old
        # evaluate nonlinear terms midpoint-like: f(a*u_{n+1} + (1-a)*u_{n})
        elif self.ti.eval_nonlin_terms=='midpoint':

            self.weakform_v = self.deltaW_kin_mid + self.deltaW_int_mid - self.deltaW_ext_mid

        else:
            raise ValueError("Unknown eval_nonlin_terms option. Choose 'trapezoidal' or 'midpoint'.")

        self.weakform_p, self.weakform_lin_vp, self.weakform_lin_pv, self.weakform_lin_pp = [], [], [], []

        for n in range(self.num_domains):
            self.weakform_p.append( self.deltaW_p[n] )

        self.weakform_lin_vv = ufl.derivative(self.weakform_v, self.v, self.dv)
        for j in range(self.num_dupl):
            self.weakform_lin_vp.append( ufl.derivative(self.weakform_v, self.p_[j], self.dp_[j]) )
        for n in range(self.num_domains):
            self.weakform_lin_pv.append( ufl.derivative(self.weakform_p[n], self.v, self.dv) )
        if self.stabilization is not None:
            for n in range(self.num_domains):
                if self.num_dupl==1: j=0
                else: j=n
                self.weakform_lin_pp.append( ufl.derivative(self.weakform_p[n], self.p_[j], self.dp_[j]) )

        if any(self.mem_active_stress):
            # active stress for reduced solid (FrSI)
            self.tau_a_, na = [], 0
            for nm in range(len(self.bc_dict['membrane'])):

                if self.mem_active_stress[nm]:
                    if self.mem_active_stress_type[nm] == 'ode':
                        self.tau_a_.append(self.actstress[na].tau_act(self.tau_a_old, self.pbase.dt))
                        na+=1
                    if self.mem_active_stress_type[nm] == 'prescribed':
                        self.tau_a_.append(self.act_curve[nm]) # act_curve now stores the prescribed active stress
                else:
                    self.tau_a_.append(ufl.as_ufl(0))

        if self.prestress_initial or self.prestress_initial_only:
            # prestressing weak forms
            self.weakform_prestress_p, self.weakform_lin_prestress_vp, self.weakform_lin_prestress_pv, self.weakform_lin_prestress_pp = [], [], [], []
            self.weakform_prestress_v = self.deltaW_prestr_kin + self.deltaW_prestr_int - self.deltaW_prestr_ext
            self.weakform_lin_prestress_vv = ufl.derivative(self.weakform_prestress_v, self.v, self.dv)
            for n in range(self.num_domains):
                self.weakform_prestress_p.append( self.deltaW_p_prestr[n] )
            for j in range(self.num_dupl):
                self.weakform_lin_prestress_vp.append( ufl.derivative(self.weakform_prestress_v, self.p_[j], self.dp_[j]) )
            for n in range(self.num_domains):
                self.weakform_lin_prestress_pv.append( ufl.derivative(self.weakform_prestress_p[n], self.v, self.dv) )
            if self.stabilization is not None:
                for n in range(self.num_domains):
                    if self.num_dupl==1: j=0
                    else: j=n
                    self.weakform_lin_prestress_pp.append( ufl.derivative(self.weakform_prestress_p[n], self.p_[j], self.dp_[j]) )


    # active stress projection - for reduced solid model
    def evaluate_active_stress(self):

        # project and interpolate to quadrature function space
        tau_a_proj = project(self.tau_a_, self.Vd_scalar, self.dx, domids=self.domain_ids, comm=self.comm, entity_maps=self.io.entity_maps) # TODO: Should be self.ds here, but yields error; why?
        self.tau_a.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.tau_a.interpolate(tau_a_proj)


    # computes the fluid's total internal power
    def compute_power(self, N, t):

        ip_all = ufl.as_ufl(0)
        for n, M in enumerate(self.domain_ids):
            if self.num_dupl==1: j=0
            else: j=n
            ip_all += ufl.inner(self.ma[n].sigma(self.v, self.p_[j], F=self.alevar['Fale']), self.ki.gamma(self.v, F=self.alevar['Fale'])) * self.dx(M)

        ip = fem.assemble_scalar(fem.form(ip_all))
        ip = self.comm.allgather(ip)
        internal_power = sum(ip)

        if self.comm.rank == 0:
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.pbase.dt): mode='wt'
                else: mode='a'
                fp = open(self.io.output_path+'/results_'+self.pbase.simname+'_internalpower.txt', mode)
                fp.write('%.16E %.16E\n' % (t,internal_power))
                fp.close()


    # computes the total strain energy and internal power of a membrane (reduced) solid model
    def compute_strain_energy_power_membrane(self, N, t):

        se_mem_all, ip_mem_all = ufl.as_ufl(0), ufl.as_ufl(0)
        for nm in range(len(self.bc_dict['membrane'])):

            internal = self.bc_dict['membrane'][nm].get('internal', False)

            if internal:
                fcts = self.bc_dict['membrane'][nm].get('facet_side', '+')
                se_mem_all += (self.bstrainenergy[nm])(fcts) * self.bmeasures[2](self.idmem[nm])
                ip_mem_all += (self.bintpower[nm])(fcts) * self.bmeasures[2](self.idmem[nm])
            else:
                se_mem_all += self.bstrainenergy[nm] * self.bmeasures[0](self.idmem[nm])
                ip_mem_all += self.bintpower[nm] * self.bmeasures[0](self.idmem[nm])

        se_mem = fem.assemble_scalar(fem.form(se_mem_all))
        se_mem = self.comm.allgather(se_mem)
        strain_energy_mem = sum(se_mem)

        ip_mem = fem.assemble_scalar(fem.form(ip_mem_all))
        ip_mem = self.comm.allgather(ip_mem)
        internal_power_mem = sum(ip_mem)

        if self.comm.rank == 0:
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.pbase.dt): mode='wt'
                else: mode='a'
                if 'strainenergy_membrane' in self.results_to_write:
                    fe = open(self.io.output_path+'/results_'+self.pbase.simname+'_strainenergy_membrane.txt', mode)
                    fe.write('%.16E %.16E\n' % (t,strain_energy_mem))
                    fe.close()
                if 'internalpower_membrane' in self.results_to_write:
                    fp = open(self.io.output_path+'/results_'+self.pbase.simname+'_internalpower_membrane.txt', mode)
                    fp.write('%.16E %.16E\n' % (t,internal_power_mem))
                    fp.close()


    # rate equations
    def evaluate_rate_equations(self, t_abs):

        # take care of active stress
        if any(self.mem_active_stress):
            self.evaluate_active_stress()


    def set_problem_residual_jacobian_forms(self, pre=False):

        ts = time.time()
        utilities.print_status("FEM form compilation for fluid...", self.comm, e=" ")

        if not bool(self.io.duplicate_mesh_domains):
            if not pre:
                self.weakform_p = sum(self.weakform_p)
                self.weakform_lin_vp = sum(self.weakform_lin_vp)
                self.weakform_lin_pv = sum(self.weakform_lin_pv)
                if self.stabilization is not None:
                    self.weakform_lin_pp = sum(self.weakform_lin_pp)
            else:
                self.weakform_prestress_p = sum(self.weakform_prestress_p)
                self.weakform_lin_prestress_vp = sum(self.weakform_lin_prestress_vp)
                self.weakform_lin_prestress_pv = sum(self.weakform_lin_prestress_pv)
                if self.stabilization is not None:
                    self.weakform_lin_prestress_pp = sum(self.weakform_lin_prestress_pp)

        if not pre:
            self.res_v = fem.form(self.weakform_v, entity_maps=self.io.entity_maps)
            self.res_p = fem.form(self.weakform_p, entity_maps=self.io.entity_maps)
            self.jac_vv = fem.form(self.weakform_lin_vv, entity_maps=self.io.entity_maps)
            self.jac_vp = fem.form(self.weakform_lin_vp, entity_maps=self.io.entity_maps)
            self.jac_pv = fem.form(self.weakform_lin_pv, entity_maps=self.io.entity_maps)
            if self.num_dupl > 1:
                self.dummat = [[None]*self.num_dupl for _ in range(self.num_dupl)] # needed for block vector assembly...
                # make lists for offdiagonal block mat assembly
                self.jac_vp_ = [self.jac_vp]
                self.jac_pv_ = []
                for j in range(self.num_dupl):
                    self.jac_pv_.append([self.jac_pv[j]])
            if self.stabilization is not None:
                self.jac_pp = fem.form(self.weakform_lin_pp, entity_maps=self.io.entity_maps)
                if self.num_dupl > 1:
                    self.jac_pp_ = [[None]*self.num_dupl for _ in range(self.num_dupl)]
                    for j in range(self.num_dupl):
                        self.jac_pp_[j][j] = self.jac_pp[j]
        else:
            self.res_v  = fem.form(self.weakform_prestress_v, entity_maps=self.io.entity_maps)
            self.res_p  = fem.form(self.weakform_prestress_p, entity_maps=self.io.entity_maps)
            self.jac_vv = fem.form(self.weakform_lin_prestress_vv, entity_maps=self.io.entity_maps)
            self.jac_vp = fem.form(self.weakform_lin_prestress_vp, entity_maps=self.io.entity_maps)
            self.jac_pv = fem.form(self.weakform_lin_prestress_pv, entity_maps=self.io.entity_maps)
            if self.num_dupl > 1:
                self.dummat = [[None]*self.num_dupl for _ in range(self.num_dupl)] # needed for block vector assembly...
                # make lists for offdiagonal block mat assembly
                self.jac_vp_ = [self.jac_vp]
                self.jac_pv_ = []
                for j in range(self.num_dupl):
                    self.jac_pv_.append([self.jac_pv[j]])
            if self.stabilization is not None:
                self.jac_pp = fem.form(self.weakform_lin_prestress_pp, entity_maps=self.io.entity_maps)
                if self.num_dupl > 1:
                    self.jac_pp_ = [[None]*self.num_dupl for _ in range(self.num_dupl)]
                    for j in range(self.num_dupl):
                        self.jac_pp_[j][j] = self.jac_pp[j]

        if self.have_robin_valve_implicit:
            self.dw_robin_valve_dz_form, self.drz_dp = [], []
            for i in range(self.num_valve_coupling_surf):
                self.dw_robin_valve_dz_form.append(fem.form(self.dw_robin_valve_dz[i], entity_maps=self.io.entity_maps))

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.comm)


    def set_problem_vector_matrix_structures(self, rom=None):

        self.r_v = fem.petsc.create_vector(self.res_v)
        if self.num_dupl > 1:
            self.r_p = fem.petsc.create_vector_block(self.res_p)
        else:
            self.r_p = fem.petsc.create_vector(self.res_p)

        self.K_vv = fem.petsc.create_matrix(self.jac_vv)
        if self.num_dupl > 1:
            self.K_vp = fem.petsc.create_matrix_block(self.jac_vp_)
            self.K_pv = fem.petsc.create_matrix_block(self.jac_pv_)
        else:
            self.K_vp = fem.petsc.create_matrix(self.jac_vp)
            self.K_pv = fem.petsc.create_matrix(self.jac_pv)

        if self.stabilization is not None:
            if self.num_dupl > 1:
                self.K_pp = fem.petsc.create_matrix_block(self.jac_pp_)
            else:
                self.K_pp = fem.petsc.create_matrix(self.jac_pp)
        else:
            self.K_pp = None

        if self.have_robin_valve_implicit:

            self.r_z = PETSc.Vec().createMPI(size=self.num_valve_coupling_surf)
            self.K_zz = PETSc.Mat().createAIJ(size=(self.num_valve_coupling_surf,self.num_valve_coupling_surf), bsize=None, nnz=None, csr=None, comm=self.comm)
            self.K_zz.setUp()

            self.row_ids = list(range(self.num_valve_coupling_surf))
            self.col_ids = list(range(self.num_valve_coupling_surf))

            # setup offdiagonal matrices
            locmatsize = self.V_v.dofmap.index_map.size_local * self.V_v.dofmap.index_map_bs
            matsize = self.V_v.dofmap.index_map.size_global * self.V_v.dofmap.index_map_bs
            #locmatsize_p = self.V_p.dofmap.index_map.size_local * self.V_p.dofmap.index_map_bs
            #matsize_p = self.V_p.dofmap.index_map.size_global * self.V_p.dofmap.index_map_bs

            self.k_vz_vec = []
            for i in range(len(self.col_ids)):
                self.k_vz_vec.append(fem.petsc.create_vector(self.dw_robin_valve_dz_form[i]))

            # self.k_zp_vec = []
            # for i in range(len(self.row_ids)):
            #     self.k_sp_vec.append(fem.petsc.create_vector(self.dcq_form[i]))

            self.dofs_coupling_v = [[]]*self.num_valve_coupling_surf

            self.k_vz_subvec, self.k_zp_subvec, sze_vz, sze_zp = [], [], [], []

            for n in range(self.num_valve_coupling_surf):

                # nds_p_local = fem.locate_dofs_topological(self.V_p, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[np.isin(self.io.mt_b1.values, self.surface_vlv_ids[n])])
                # nds_p = np.array( self.V_v.dofmap.index_map.local_to_global(np.asarray(nds_p_local, dtype=np.int32)), dtype=np.int32 )
                # self.dofs_coupling_p[n] = PETSc.IS().createBlock(self.V_p.dofmap.index_map_bs, nds_p, comm=self.comm)
                #
                # self.k_zp_subvec.append( self.k_zp_vec[n].getSubVector(self.dofs_coupling_p[n]) )
                #
                # sze_zp.append(self.k_zp_subvec[-1].getSize())

                nds_v_local = fem.locate_dofs_topological(self.V_v, self.io.mesh.topology.dim-1, self.io.mt_b1.indices[np.isin(self.io.mt_b1.values, self.surface_vlv_ids[n])])
                nds_v = np.array( self.V_v.dofmap.index_map.local_to_global(np.asarray(nds_v_local, dtype=np.int32)), dtype=np.int32 )
                self.dofs_coupling_v[n] = PETSc.IS().createBlock(self.V_v.dofmap.index_map_bs, nds_v, comm=self.comm)

                self.k_vz_subvec.append( self.k_vz_vec[n].getSubVector(self.dofs_coupling_v[n]) )

                sze_vz.append(self.k_vz_subvec[-1].getSize())

            # derivative of fluid residual w.r.t. valve state variables
            self.K_vz = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(PETSc.DECIDE,self.num_valve_coupling_surf)), bsize=None, nnz=self.num_valve_coupling_surf, csr=None, comm=self.comm)
            self.K_vz.setUp()

            # # derivative of valve state residual w.r.t. fluid pressures
            # self.K_zp = PETSc.Mat().createAIJ(size=((PETSc.DECIDE,self.num_valve_coupling_surf),(locmatsize,matsize)), bsize=None, nnz=max(sze_sv), csr=None, comm=self.comm)
            # self.K_zp.setUp()
            # self.K_zp.setOption(PETSc.Mat.Option.ROW_ORIENTED, False)


    def assemble_residual(self, t, subsolver=None):

        # NOTE: we do not linearize integrated pressure-dependent valves w.r.t. p,
        # hence evaluation within the nonlinear solver loop may cause convergence problems
        # (linearization would mean that every velocity at the valve surface depends
        # on every pressure, which yields a fully dense matrix block!)
        # if self.have_robin_valve:
        #     self.evaluate_robin_valve(t)

        # assemble velocity rhs vector
        with self.r_v.localForm() as r_local: r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_v, self.res_v)
        fem.apply_lifting(self.r_v, [self.jac_vv], [self.bc.dbcs], x0=[self.v.x.petsc_vec])
        self.r_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.r_v, self.bc.dbcs, x0=self.v.x.petsc_vec, scale=-1.0)

        # assemble pressure rhs vector
        with self.r_p.localForm() as r_local: r_local.set(0.0)
        if self.num_dupl > 1:
            fem.petsc.assemble_vector_block(self.r_p, self.res_p, self.dummat, bcs=[]) # ghosts are updated inside assemble_vector_block
        else:
            fem.petsc.assemble_vector(self.r_p, self.res_p)
            self.r_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.r_list[0] = self.r_v
        self.r_list[1] = self.r_p

        if self.have_robin_valve_implicit:
            self.evaluate_dp_monitor(self.pu_, self.pd_, self.pint_u_, self.pint_d_, self.dp_, self.a_u_, self.a_d_, prnt=False)
            self.evaluate_robin_valve_implicit(self.dp_)
            ls, le = self.z.getOwnershipRange()
            for i in range(ls,le):
                self.r_z[i] = self.z[i] - self.dp_[i]
            self.r_z.assemble()
            self.r_list[2] = self.r_z


    def assemble_stiffness(self, t, subsolver=None):

        # assemble system matrix
        self.K_vv.zeroEntries()
        fem.petsc.assemble_matrix(self.K_vv, self.jac_vv, self.bc.dbcs)
        self.K_vv.assemble()

        # assemble system matrices
        self.K_vp.zeroEntries()
        if self.num_dupl > 1:
            fem.petsc.assemble_matrix_block(self.K_vp, self.jac_vp_, self.bc.dbcs)
        else:
            fem.petsc.assemble_matrix(self.K_vp, self.jac_vp, self.bc.dbcs)
        self.K_vp.assemble()

        self.K_pv.zeroEntries()
        if self.num_dupl > 1:
            fem.petsc.assemble_matrix_block(self.K_pv, self.jac_pv_, []) # currently, we do not consider pressure DBCs
        else:
            fem.petsc.assemble_matrix(self.K_pv, self.jac_pv, []) # currently, we do not consider pressure DBCs
        self.K_pv.assemble()

        if self.stabilization is not None:
            self.K_pp.zeroEntries()
            if self.num_dupl > 1:
                fem.petsc.assemble_matrix_block(self.K_pp, self.jac_pp_, []) # currently, we do not consider pressure DBCs
            else:
                fem.petsc.assemble_matrix(self.K_pp, self.jac_pp, []) # currently, we do not consider pressure DBCs
            self.K_pp.assemble()
        else:
            self.K_pp = None

        self.K_list[0][0] = self.K_vv
        self.K_list[0][1] = self.K_vp
        self.K_list[1][0] = self.K_pv
        self.K_list[1][1] = self.K_pp

        if self.have_robin_valve_implicit:

            # offdiagonal v-z columns
            for i in range(len(self.col_ids)):
                with self.k_vz_vec[i].localForm() as r_local: r_local.set(0.0)
                fem.petsc.assemble_vector(self.k_vz_vec[i], self.dw_robin_valve_dz_form[i]) # already multiplied by time-integration factor
                self.k_vz_vec[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                # set zeros at DBC entries
                fem.set_bc(self.k_vz_vec[i], self.bc.dbcs, x0=self.v.x.petsc_vec)#, scale=0.0)

            # set columns
            for i in range(len(self.col_ids)):
                # NOTE: only set the surface-subset of the k_vz vector entries to avoid placing unnecessary zeros!
                self.k_vz_vec[i].getSubVector(self.dofs_coupling_v[i], subvec=self.k_vz_subvec[i])
                self.K_vz.setValues(self.dofs_coupling_v[i], self.col_ids[i], self.k_vz_subvec[i].array, addv=PETSc.InsertMode.INSERT)
                self.k_vz_vec[i].restoreSubVector(self.dofs_coupling_v[i], subvec=self.k_vz_subvec[i])

            self.K_vz.assemble()

            ls, le = self.K_zz.getOwnershipRange()
            for i in range(ls, le):
                self.K_zz.setValue(i, i, 1.0, addv=PETSc.InsertMode.INSERT)

            self.K_zz.assemble()

            self.K_list[0][2] = self.K_vz
            self.K_list[2][2] = self.K_zz
            # K_zp needed...
            # self.K_list[2][1] =



    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.v.x.petsc_vec.getOwnershipRange()[0]
            vvec_ls = self.v.x.petsc_vec.getLocalSize()

        offset_v = vvec_or0 + self.p.x.petsc_vec.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.p.x.petsc_vec.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            ilist = [iset_v, iset_p, iset_r]
        else:
            ilist = [iset_v, iset_p]

        return ilist


    # valve law on "immersed" surface (an internal boundary)
    def evaluate_robin_valve(self, t, dp_):

        # dp_ = pd_ - pu_ (downstream minus upstream)
        for m in range(len(self.bc_dict['robin_valve'])):

            # try to set time if present in expression
            try: self.beta_valve_max_expr[m].t = t
            except: pass

            try: self.beta_valve_min_expr[m].t = t
            except: pass

            if self.bc_dict['robin_valve'][m]['type'] == 'dp':
                dp_id = self.bc_dict['robin_valve'][m]['dp_monitor_id']
                if dp_[dp_id] > 0.:
                    self.beta_valve[m].interpolate(self.beta_valve_max_expr[m].evaluate)
                else:
                    self.beta_valve[m].interpolate(self.beta_valve_min_expr[m].evaluate)

            elif self.bc_dict['robin_valve'][m]['type'] == 'dp_smooth': # min, max values have to be spatially constant (no expression!) in this case...
                beta = expression.template()
                dp_id = self.bc_dict['robin_valve'][m]['dp_monitor_id']
                epsilon = self.bc_dict['robin_valve'][m]['epsilon']
                beta.val = 0.5*(self.beta_valve_max_expr[m].val - self.beta_valve_min_expr[m].val)*(ufl.tanh(dp_[dp_id]/epsilon) + 1.) + self.beta_valve_min_expr[m].val
                self.beta_valve[m].interpolate(beta.evaluate)

            elif self.bc_dict['robin_valve'][m]['type'] == 'temporal':
                to, tc = self.bc_dict['robin_valve'][m]['to'], self.bc_dict['robin_valve'][m]['tc']
                if to > tc:
                    if t < to and t >= tc:
                        self.beta_valve[m].interpolate(self.beta_valve_max_expr[m].evaluate)
                    if t >= to or t < tc:
                        self.beta_valve[m].interpolate(self.beta_valve_min_expr[m].evaluate)
                else:
                    if t < to or t >= tc:
                        self.beta_valve[m].interpolate(self.beta_valve_max_expr[m].evaluate)
                    if t >= to and t < tc:
                        self.beta_valve[m].interpolate(self.beta_valve_min_expr[m].evaluate)

            else:
                raise ValueError("Unknown Robin valve type!")

            self.beta_valve[m].x.scatter_forward()


    # valve law on "immersed" surface (an internal boundary) - implicit version
    def evaluate_robin_valve_implicit(self, dp_):

        # dp_ = pd_ - pu_ (downstream minus upstream)
        for m in range(self.num_valve_coupling_surf):

            if self.bc_dict['robin_valve_implicit'][m]['type'] == 'dp':
                dp_id = self.bc_dict['robin_valve_implicit'][m]['dp_monitor_id']
                if dp_[dp_id] > 0.:
                    self.beta_valve[m].interpolate(self.beta_valve_max_expr[m].evaluate)
                else:
                    self.beta_valve[m].interpolate(self.beta_valve_min_expr[m].evaluate)

            elif self.bc_dict['robin_valve_implicit'][m]['type'] == 'dp_smooth':
                beta = expression.template()
                dp_id = self.bc_dict['robin_valve_implicit'][m]['dp_monitor_id']
                epsilon = self.bc_dict['robin_valve_implicit'][m]['epsilon']
                beta.val = 0.5*(self.beta_valve_max_expr[m].val - self.beta_valve_min_expr[m].val)*(ufl.tanh(dp_[dp_id]/epsilon) + 1.) + self.beta_valve_min_expr[m].val
                self.beta_valve[m].interpolate(beta.evaluate)

            else:
                raise ValueError("Unknown implicit Robin valve type!")

            self.beta_valve[m].interpolate(beta.evaluate)


    def evaluate_dp_monitor(self, pu_, pd_, pint_u_, pint_d_, dp_, a_u_, a_d_, prnt=True):

        for m in range(len(self.bc_dict['dp_monitor'])):

            # area of up- and downstream surfaces
            au = fem.assemble_scalar(a_u_[m])
            au = self.comm.allgather(au)
            au_ = sum(au)

            ad = fem.assemble_scalar(a_d_[m])
            ad = self.comm.allgather(ad)
            ad_ = sum(ad)

            # assert that the two parts of the valve are actually of same size
            # acutally not needed - domain partitioning should assert this...
            #assert(np.isclose(au_, ad_))

            # surface-averaged pressures on up- and downstream sides
            pu = (1./au_)*fem.assemble_scalar(pint_u_[m])
            pu = self.comm.allgather(pu)
            pu_[m] = sum(pu)

            pd = (1./ad_)*fem.assemble_scalar(pint_d_[m])
            pd = self.comm.allgather(pd)
            pd_[m] = sum(pd)

            dp_[m] = sum(pd) - sum(pu)

            if prnt: utilities.print_status("dp ID "+str(self.bc_dict['dp_monitor'][m]['id'])+": pu = %.4e, pd = %.4e" % (pu_[m],pd_[m]), self.comm)


    def evaluate_flux_monitor(self, qv_, q_, prnt=True):

        for m in range(len(self.bc_dict['flux_monitor'])):

            q = fem.assemble_scalar(q_[m])
            q = self.comm.allgather(q)
            qv_[m] = sum(q)

            if prnt: utilities.print_status("Flux ID "+str(self.bc_dict['flux_monitor'][m]['id'])+": q = %.4e" % (qv_[m]), self.comm)


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # read restart information
        if self.pbase.restart_step > 0:
            self.io.readcheckpoint(self, N)


    def evaluate_initial(self):

        if self.have_flux_monitor:
            self.evaluate_flux_monitor(self.qv_old_, self.q_old_)
            for k in self.qv_old_: self.qv_[k] = self.qv_old_[k]
        if self.have_dp_monitor:
            self.evaluate_dp_monitor(self.pu_old_, self.pd_old_, self.pint_u_old_, self.pint_d_old_, self.dp_old_, self.a_u_old_, self.a_d_old_)
            for k in self.pu_old_: self.pu_[k] = self.pu_old_[k]
            for k in self.pd_old_: self.pd_[k] = self.pd_old_[k]
        if self.have_robin_valve:
            self.evaluate_robin_valve(self.pbase.t_init, self.dp_old_)


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def write_output_pre(self):

        if 'fibers' in self.results_to_write and self.io.write_results_every > 0:
            for i in range(len(self.fibarray)):
                fib_proj = project(self.fib_func[i], self.V_v, self.dx, domids=self.domain_ids, nm='Fiber'+str(i+1), comm=self.comm, entity_maps=self.io.entity_maps)
                self.io.write_output_pre(self, fib_proj, 0.0, 'fib_'+self.fibarray[i])


    def evaluate_pre_solve(self, t, N, dt):

        # set time-dependent functions
        self.ti.set_time_funcs(t, dt, midp=self.midp)

        # evaluate rate equations
        self.evaluate_rate_equations(t)

        # DBC from files
        if self.bc.have_dirichlet_file:
            for m in self.ti.funcs_data:
                file = list(m.values())[0].replace('*',str(N))
                func = list(m.keys())[0]
                self.io.readfunction(func, file)
                sc = m['scale']
                if sc != 1.0: func.x.petsc_vec.scale(sc)


    def evaluate_post_solve(self, t, N):

        if self.have_flux_monitor:
            self.evaluate_flux_monitor(self.qv_, self.q_)
        if self.have_dp_monitor:
            self.evaluate_dp_monitor(self.pu_, self.pd_, self.pint_u_, self.pint_d_, self.dp_, self.a_u_, self.a_d_)
        if self.have_robin_valve:
            self.evaluate_robin_valve(t, self.dp_)
        if 'internalpower' in self.results_to_write:
            self.compute_power(N, t)
        if 'membrane' in self.bc_dict.keys() and ('strainenergy_membrane' in self.results_to_write or 'internalpower_membrane' in self.results_to_write):
            self.compute_strain_energy_power_membrane(N, t)


    def set_output_state(self, t):
        pass


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t)


    def update(self):

        # update - velocity, acceleration, pressure, all internal variables, all time functions
        self.ti.update_timestep(self.v, self.v_old, self.a, self.a_old, self.p, self.p_old, self.internalvars, self.internalvars_old, uf=self.uf, uf_old=self.uf_old)
        # update monitor dicts
        if self.have_flux_monitor:
            for k in self.qv_: self.qv_old_[k] = self.qv_[k]
        if self.have_dp_monitor:
            for k in self.pu_: self.pu_old_[k] = self.pu_[k]
            for k in self.pd_: self.pd_old_[k] = self.pd_[k]


    def print_to_screen(self):
        pass


    def induce_state_change(self):
        pass


    def write_restart(self, sname, N, force=False):

        self.io.write_restart(self, N, force=force)


    def check_abort(self, t):
        pass


    def destroy(self):

        self.io.close_output_files(self)



class FluidmechanicsSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.fluid_governing_type == 'navierstokes_transient' or self.pb.fluid_governing_type == 'stokes_transient') and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial acceleration...", self.pb.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.a_old, self.pb.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.io.entity_maps)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.comm)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)


    def solve_initial_prestress(self):

        utilities.print_prestress('start', self.pb.comm)

        if self.pb.prestress_ptc: self.solnln.PTC = True

        for N in range(1,self.pb.prestress_numstep+1):

            wts = time.time()

            tprestr = N * self.pb.prestress_dt

            self.pb.ti.set_time_funcs_pre(tprestr)

            self.solnln.newton(tprestr)

            # update uf_pre
            self.pb.ki.prestress_update(self.pb.prestress_dt, self.pb.v.x.petsc_vec)
            utilities.print_prestress('updt', self.pb.comm)

            # update fluid velocity: v_old <- v
            if self.pb.prestress_kinetic!='none':
                self.pb.v_old.x.petsc_vec.axpby(1.0, 0.0, self.pb.v.x.petsc_vec)
                self.pb.v_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            wt = time.time() - wts

            # print time step info to screen
            self.pb.ti.print_prestress_step(N, tprestr, self.pb.prestress_numstep, self.solnln.lsp, ni=self.solnln.ni, li=self.solnln.li, wt=wt)

        utilities.print_prestress('end', self.pb.comm)

        # write prestress displacement (given that we want to write the fluid displacement)
        if 'fluiddisplacement' in self.pb.results_to_write and self.pb.io.write_results_every > 0:
            self.pb.io.write_output_pre(self.pb, self.pb.uf_pre, 0, 'fluiddisplacement_pre')

        if self.pb.prestress_initial_only:
            # it may be convenient to write the prestress displacement field to a file for later read-in
            self.pb.io.writefunction(self.pb.uf_pre, self.pb.io.output_path_pre+'/results_'+self.pb.pbase.simname+'_fluiddisplacement_pre')
            if bool(self.pb.io.duplicate_mesh_domains):
                for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                     # TODO: Might not work for duplicate mesh, since we do not have the input node indices (do we...?)
                    self.pb.io.writefunction(self.pb.p_[m], self.pb.io.output_path_pre+'/results_'+self.pb.pbase.simname+'_pressure'+str(m+1)+'_pre')
            else:
                self.pb.io.writefunction(self.pb.p_[0], self.pb.io.output_path_pre+'/results_'+self.pb.pbase.simname+'_pressure_pre')
            utilities.print_status("Prestress only done. To resume, set file path(s) in 'prestress_from_file' and read in uf_pre.", self.pb.comm)
            os._exit(0)

        # reset state
        self.pb.v.x.petsc_vec.set(0.0)
        self.pb.v.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.v_old.x.petsc_vec.set(0.0)
        self.pb.v_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # reset PTC flag to what it was
        if self.pb.prestress_ptc:
            self.solnln.PTC = self.solver_params.get('ptc', False)

        # now build main (non-prestress) forms
        self.pb.set_problem_residual_jacobian_forms()


# prestress solver, to be called from other (coupled) problems
class FluidmechanicsSolverPrestr(FluidmechanicsSolver):

    def __init__(self, problem, solver_params):

        self.pb = problem
        self.solver_params = solver_params

        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):
        raise RuntimeError("You should not be here!")
