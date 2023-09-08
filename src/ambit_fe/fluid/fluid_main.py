#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, os
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

# fluid mechanics, governed by incompressible Navier-Stokes equations:

#\begin{align}
#\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \left(\boldsymbol{\nabla}\boldsymbol{v}\right) \boldsymbol{v}\right) = \boldsymbol{\nabla} \cdot \boldsymbol{\sigma} + \hat{\boldsymbol{b}} \quad \text{in} \; \Omega \times [0, T] \\
#\boldsymbol{\nabla} \cdot \boldsymbol{v} = 0 \quad \text{in} \; \Omega \times [0, T]
#\end{align}

class FluidmechanicsProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, iof, mor_params={}, comm=None, alevar={}):
        super().__init__(io_params, time_params, comm)

        ioparams.check_params_fem_fluid(fem_params)
        ioparams.check_params_time_fluid(time_params)

        self.problem_physics = 'fluid'

        self.results_to_write = io_params['results_to_write']

        self.io = iof

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        # for FSI, we want to specify the subdomains
        try: domain_ids = self.io.io_params['domain_ids_fluid']
        except: domain_ids = np.arange(1,self.num_domains+1)

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_vel = fem_params['order_vel']
        self.order_pres = fem_params['order_pres']
        self.quad_degree = fem_params['quad_degree']

        # collect domain data
        self.dx_, self.rho = [], []
        for i, n in enumerate(domain_ids):
            # integration domains
            if self.io.mt_d_master is not None: self.dx_.append(ufl.dx(domain=self.io.mesh_master, subdomain_data=self.io.mt_d_master, subdomain_id=n, metadata={'quadrature_degree': self.quad_degree}))
            else:                               self.dx_.append(ufl.dx(domain=self.io.mesh_master, metadata={'quadrature_degree': self.quad_degree}))
            # data for inertial forces: density
            self.rho.append(self.constitutive_models['MAT'+str(i+1)]['inertia']['rho'])

        # whether to enforce continuity of mass at midpoint or not
        try: self.pressure_at_midpoint = fem_params['pressure_at_midpoint']
        except: self.pressure_at_midpoint = False

        try: self.fluid_formulation = fem_params['fluid_formulation']
        except: self.fluid_formulation = 'nonconservative'

        try: self.fluid_governing_type = time_params['fluid_governing_type']
        except: self.fluid_governing_type = 'navierstokes_transient'

        try: self.stabilization = fem_params['stabilization']
        except: self.stabilization = None

        try: self.prestress_initial = fem_params['prestress_initial']
        except: self.prestress_initial = False
        try: self.prestress_initial_only = fem_params['prestress_initial_only']
        except: self.prestress_initial_only = False
        try: self.prestress_numstep = fem_params['prestress_numstep']
        except: self.prestress_numstep = 1
        try: self.prestress_maxtime = fem_params['prestress_maxtime']
        except: self.prestress_maxtime = 1.0
        try: self.prestress_ptc = fem_params['prestress_ptc']
        except: self.prestress_ptc = False
        try: self.prestress_from_file = fem_params['prestress_from_file']
        except: self.prestress_from_file = False
        try: self.initial_fluid_pressure = fem_params['initial_fluid_pressure']
        except: self.initial_fluid_pressure = []

        if self.prestress_from_file: self.prestress_initial, self.prestress_initial_only  = False, False

        self.localsolve = False # no idea what might have to be solved locally...

        self.sub_solve = False

        self.have_flux_monitor, self.have_dp_monitor, self.have_robin_valve = False, False, False

        self.dim = self.io.mesh.geometry.dim

        # dicts for evaluations of surface integrals (fluxes, pressures), to be queried by other models
        self.qv_, self.pu_, self.pd_ = {}, {}, {}
        self.qv_old_, self.pu_old_, self.pd_old_ = {}, {}, {}

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

        # check if we want to use model order reduction and if yes, initialize MOR class
        try: self.have_rom = io_params['use_model_order_red']
        except: self.have_rom = False

        # ALE fluid problem variables
        self.alevar = alevar

        # create finite element objects for v and p
        P_v = ufl.VectorElement("CG", self.io.mesh.ufl_cell(), self.order_vel)
        P_p = ufl.FiniteElement("CG", self.io.mesh.ufl_cell(), self.order_pres)
        # function space for v
        self.V_v = fem.FunctionSpace(self.io.mesh, P_v)

        self.V_p__ = {}

        if bool(self.io.duplicate_mesh_domains):

            assert(self.io.USE_MIXED_DOLFINX_BRANCH)
            assert(self.num_domains>1)

            self.num_dupl = self.num_domains

            self.io.submshes_emap, inv_emap, self.io.sub_mt_b1 = {}, {}, {}

            for mp in self.io.duplicate_mesh_domains:
                self.io.submshes_emap[mp] = mesh.create_submesh(self.io.mesh, self.io.mesh.topology.dim, self.io.mt_d.indices[self.io.mt_d.values == mp])[0:2]

            cell_imap = self.io.mesh.topology.index_map(self.io.mesh.topology.dim)
            num_cells = cell_imap.size_local + cell_imap.num_ghosts

            for mp in self.io.duplicate_mesh_domains:
                self.V_p__[mp] = fem.FunctionSpace(self.io.submshes_emap[mp][0], P_p)
                inv_emap[mp] = np.full(num_cells, -1)

            for mp in self.io.duplicate_mesh_domains:
                inv_emap[mp][self.io.submshes_emap[mp][1]] = np.arange(len(self.io.submshes_emap[mp][1]))
                self.io.entity_maps[self.io.submshes_emap[mp][0]] = inv_emap[mp]
                # transfer boundary meshtags to submesh
                self.io.sub_mt_b1[mp] = meshutils.meshtags_parent_to_child(self.io.mt_b1, self.io.submshes_emap[mp][0], self.io.submshes_emap[mp][1], self.io.mesh, 'boundary')

        else:
            self.num_dupl = 1
            self.V_p_ = [ fem.FunctionSpace(self.io.mesh, P_p) ]

        # continuous tensor and scalar function spaces of order order_vel
        self.V_tensor = fem.TensorFunctionSpace(self.io.mesh, ("CG", self.order_vel))
        self.V_scalar = fem.FunctionSpace(self.io.mesh, ("CG", self.order_vel))

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.TensorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_vector = fem.VectorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_scalar = fem.FunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))

        # functions
        self.dv     = ufl.TrialFunction(self.V_v)            # Incremental velocity
        self.var_v  = ufl.TestFunction(self.V_v)             # Test function
        self.v      = fem.Function(self.V_v, name="Velocity")

        # pressure can have duplicate nodes (one variable per domain)
        self.p__, self.p_old__, self.var_p__, self.dp__ = {}, {}, {}, {}
        if self.num_dupl > 1:
            for mp in self.io.duplicate_mesh_domains:
                self.p__[mp] = fem.Function(self.V_p__[mp], name="Pressure"+str(mp))
                self.dp__[mp] = ufl.TrialFunction(self.V_p__[mp])            # Incremental pressure
                self.var_p__[mp] = ufl.TestFunction(self.V_p__[mp])          # Test function
                # values of previous time step
                self.p_old__[mp] = fem.Function(self.V_p__[mp])
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
        self.a     = fem.Function(self.V_v, name="Acceleration")
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

        # collect references to pressure vectors
        self.pvecs_, self.pvecs_old_ = [], []
        if self.num_dupl > 1:
            for mp in range(self.num_dupl):
                self.pvecs_.append(self.p_[mp].vector)
                self.pvecs_old_.append(self.p_old_[mp].vector)
                # full pressure vectors - dummy function that holds a class variable "vector"
                self.p = expression.function_dummy(self.pvecs_,self.comm)
                self.p_old = expression.function_dummy(self.pvecs_old_,self.comm)
        else:
            self.V_p, self.p, self.p_old = self.V_p_[0], self.p_[0], self.p_old_[0] # pointer to first p's...

        # if we want to initialize the pressure (domain wise) with a scalar value
        if self.restart_step==0:
            if bool(self.initial_fluid_pressure):
                for mp in range(self.num_dupl):
                    val = self.initial_fluid_pressure[mp]
                    self.p_[mp].vector.set(val)
                    self.p_[mp].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    self.p_old_[mp].vector.set(val)
                    self.p_old_[mp].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # own read function: requires plain txt format of type "node-id val-x val-y val-z" (or one value in case of a scalar)
        if bool(self.prestress_from_file):
            self.io.readfunction(self.uf_pre, self.prestress_from_file[0])
            # if available, we might want to read in the pressure field, too
            if len(self.prestress_from_file)>1:
                if bool(self.duplicate_mesh_domains):
                    m=0
                    for j in self.duplicate_mesh_domains:
                        self.io.readfunction(self.p_[m], self.prestress_from_file[1].replace('*',str(m+1)))
                        self.io.readfunction(self.p_old_[m], self.prestress_from_file[1].replace('*',str(m+1)))
                        m+=1
                else:
                    self.io.readfunction(self.p_[0], self.prestress_from_file[1])
                    self.io.readfunction(self.p_old_[0], self.prestress_from_file[1])

        # dictionaries of internal variables
        self.internalvars, self.internalvars_old = {}, {}

        self.numdof = self.v.vector.getSize() + self.p.vector.getSize()

        self.mor_params = mor_params

        # initialize fluid time-integration class
        self.ti = timeintegration.timeintegration_fluid(time_params, fem_params, time_curves=time_curves, t_init=self.t_init, comm=self.comm)

        # get time factors
        self.timefac_m, self.timefac = self.ti.timefactors()

        # initialize kinematics_constitutive class
        self.ki = fluid_kinematics_constitutive.kinematics(self.dim, uf_pre=self.uf_pre)

        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(fluid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)]))

        # initialize fluid variational form class
        if not bool(self.alevar):
            # standard Eulerian fluid
            self.alevar = {'Fale' : None, 'Fale_old' : None, 'w' : None, 'w_old' : None, 'fluid_on_deformed' : 'no'}
            self.vf = fluid_variationalform.variationalform(self.var_v, self.var_p_, self.io.n0, formulation=self.fluid_formulation)

        else:
            # mid-point representation of ALE velocity
            self.alevar['w_mid']    = self.timefac * self.alevar['w']    + (1.-self.timefac) * self.alevar['w_old']
            # mid-point representation of ALE deformation gradient - linear in ALE displacement, hence we can combine it like this
            self.alevar['Fale_mid'] = self.timefac * self.alevar['Fale'] + (1.-self.timefac) * self.alevar['Fale_old']

            if self.alevar['fluid_on_deformed'] == 'consistent':
                # fully consistent ALE formulation of Navier-Stokes
                self.vf = fluid_variationalform.variationalform_ale(self.var_v, self.var_p_, self.io.n0, formulation=self.fluid_formulation)

            elif self.alevar['fluid_on_deformed'] == 'from_last_step':
                # ALE formulation of Navier-Stokes using metrics (Fale, w) from the last converged step... more efficient but not fully consistent
                self.alevar['Fale'], self.alevar['w'] = self.alevar['Fale_old'], self.alevar['w_old']
                self.vf = fluid_variationalform.variationalform_ale(self.var_v, self.var_p_, self.io.n0, formulation=self.fluid_formulation)

            elif self.alevar['fluid_on_deformed'] == 'mesh_move':
                # Navier-Stokes formulated w.r.t. the current, moved frame... more efficient than 'consistent' approach but not fully consistent
                # WARNING: This is unsuitable for FrSI, as we need gradients w.r.t. the reference frame on the reduced boundary!
                self.alevar = {'Fale' : None, 'Fale_old' : None, 'w' : None, 'w_old' : None, 'fluid_on_deformed' : 'mesh_move'}
                self.vf = fluid_variationalform.variationalform(self.var_v, self.var_p_, self.io.n0, formulation=self.fluid_formulation)

            else:
                raise ValueError("Unknown fluid_on_deformed option!")

        # read in fiber data - for reduced solid (FrSI)
        if bool(self.io.fiber_data):

            fibarray = ['circ']
            if len(self.io.fiber_data)>1: fibarray.append('long')

            self.fib_func = self.io.readin_fibers(fibarray, self.V_v, self.dx_, self.order_vel)

            if 'fibers' in self.results_to_write and self.io.write_results_every > 0:
                for i in range(len(fibarray)):
                    fib_proj = project(self.fib_func[i], self.V_v, self.dx_, nm='Fiber'+str(i+1), comm=self.comm)
                    self.io.write_output_pre(self, fib_proj, 0.0, 'fib_'+fibarray[i])

        else:
            self.fib_func = None

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_fluid(fem_params, self.io, self.vf, self.ti, ki=self.ki, ff=self.fib_func)

        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.bc_dict['dirichlet'], self.V_v)

        if 'dirichlet_vol' in self.bc_dict.keys():
            self.bc.dirichlet_vol(self.bc_dict['dirichlet_vol'], self.V_v)

        self.set_variational_forms()

        self.pbrom = self # self-pointer needed for ROM solver access
        self.V_rom = self.V_v
        self.print_enhanced_info = self.io.print_enhanced_info

        # number of fields involved
        self.nfields = 2

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)],  [[None]*self.nfields for _ in range(self.nfields)]


    def get_problem_var_list(self):

        if self.num_dupl > 1: is_ghosted = [1, 2]
        else:                 is_ghosted = [1, 1]
        return [self.v.vector, self.p.vector], is_ghosted


    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):

        # set form for acceleration
        self.acc = self.ti.set_acc(self.v, self.v_old, self.a_old)
        # set form for fluid displacement (needed for FrSI)
        self.ufluid = self.ti.set_uf(self.v, self.v_old, self.uf_old)

        # set mid-point representations (if needed...)
        self.acc_mid     = self.timefac_m * self.acc    + (1.-self.timefac_m) * self.a_old
        self.vel_mid     = self.timefac   * self.v      + (1.-self.timefac)   * self.v_old
        self.ufluid_mid  = self.timefac   * self.ufluid + (1.-self.timefac)   * self.uf_old

        # kinetic, internal, and pressure virtual power
        self.deltaW_kin, self.deltaW_kin_old = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_int, self.deltaW_int_old = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_p,   self.deltaW_p_old   = [], []

        for n in range(self.num_domains):

            if self.num_dupl==1: j=0
            else: j=n

            # kinetic virtual power
            if self.fluid_governing_type=='navierstokes_transient':
                self.deltaW_kin     += self.vf.deltaW_kin_navierstokes_transient(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_navierstokes_transient(self.a_old, self.v_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
            elif self.fluid_governing_type=='navierstokes_steady':
                self.deltaW_kin     += self.vf.deltaW_kin_navierstokes_steady(self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_navierstokes_steady(self.v_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
            elif self.fluid_governing_type=='stokes_transient':
                self.deltaW_kin     += self.vf.deltaW_kin_stokes_transient(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_stokes_transient(self.a_old, self.v_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
            elif self.fluid_governing_type=='stokes_steady':
                pass # no kinetic term to add for steady Stokes flow
            else:
                raise ValueError("Unknown fluid_governing_type!")

            # internal virtual power
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].sigma(self.v, self.p_[j], Fale=self.alevar['Fale']), self.dx_[n], Fale=self.alevar['Fale'])
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].sigma(self.v_old, self.p_old_[j], Fale=self.alevar['Fale_old']), self.dx_[n], Fale=self.alevar['Fale_old'])

            # pressure virtual power
            self.deltaW_p.append( self.vf.deltaW_int_pres(self.v, self.var_p_[j], self.dx_[n], Fale=self.alevar['Fale']) )
            self.deltaW_p_old.append( self.vf.deltaW_int_pres(self.v_old, self.var_p_[j], self.dx_[n], Fale=self.alevar['Fale_old']) )

        # external virtual power (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_neumann_old, w_body, w_body_old, w_robin, w_robin_old, w_stabneumann, w_stabneumann_old, w_robin_valve, w_robin_valve_old, w_membrane, w_membrane_old = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann     = self.bc.neumann_bcs(self.bc_dict['neumann'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale'], funcs_to_update=self.ti.funcs_to_update, funcs_to_update_vec=self.ti.funcs_to_update_vec)
            w_neumann_old = self.bc.neumann_bcs(self.bc_dict['neumann'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale_old'], funcs_to_update=self.ti.funcs_to_update_old, funcs_to_update_vec=self.ti.funcs_to_update_vec_old)
        if 'bodyforce' in self.bc_dict.keys():
            w_body      = self.bc.bodyforce(self.bc_dict['bodyforce'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale'], funcs_to_update=self.ti.funcs_to_update)
            w_body_old  = self.bc.bodyforce(self.bc_dict['bodyforce'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale_old'], funcs_to_update=self.ti.funcs_to_update_old)
        if 'robin' in self.bc_dict.keys():
            w_robin     = self.bc.robin_bcs(self.bc_dict['robin'], self.v, Fale=self.alevar['Fale'])
            w_robin_old = self.bc.robin_bcs(self.bc_dict['robin'], self.v_old, Fale=self.alevar['Fale_old'])
        if 'stabilized_neumann' in self.bc_dict.keys():
            w_stabneumann     = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.v, wel=self.alevar['w'], Fale=self.alevar['Fale'])
            w_stabneumann_old = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.v_old, wel=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
        if 'robin_valve' in self.bc_dict.keys():
            assert(self.num_dupl>1) # only makes sense if we have duplicate pressure domains
            self.have_robin_valve = True
            self.beta_valve, self.beta_valve_old = [], []
            w_robin_valve     = self.bc.robin_valve_bcs(self.bc_dict['robin_valve'], self.v, self.Vd_scalar, self.beta_valve, wel=self.alevar['w'], Fale=self.alevar['Fale'])
            w_robin_valve_old = self.bc.robin_valve_bcs(self.bc_dict['robin_valve'], self.v_old, self.Vd_scalar, self.beta_valve_old, wel=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
        if 'flux_monitor' in self.bc_dict.keys():
            self.have_flux_monitor = True
            self.q_, self.q_old_ = [], []
            self.bc.flux_monitor_bcs(self.bc_dict['flux_monitor'], self.v, self.q_, wel=self.alevar['w'], Fale=self.alevar['Fale'])
            self.bc.flux_monitor_bcs(self.bc_dict['flux_monitor'], self.v_old, self.q_old_, wel=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
        if 'dp_monitor' in self.bc_dict.keys():
            assert(self.num_dupl>1) # only makes sense if we have duplicate pressure domains
            self.have_dp_monitor = True
            self.a_u_, self.a_d_, self.pint_u_, self.pint_d_ = [], [], [], []
            self.a_u_old_, self.a_d_old_, self.pint_u_old_, self.pint_d_old_ = [], [], [], []
            self.bc.dp_monitor_bcs(self.bc_dict['dp_monitor'], self.a_u_, self.a_d_, self.pint_u_, self.pint_d_, self.p__, wel=self.alevar['w'], Fale=self.alevar['Fale'])
            self.bc.dp_monitor_bcs(self.bc_dict['dp_monitor'], self.a_u_old_, self.a_d_old_, self.pint_u_old_, self.pint_d_old_, self.p_old__, wel=self.alevar['w_old'], Fale=self.alevar['Fale_old'])

        # reduced-solid for FrSI problem
        self.have_active_stress, self.active_stress_trig = False, 'ode'
        if 'membrane' in self.bc_dict.keys():
            assert(self.alevar['fluid_on_deformed']!='mesh_move')

            self.mem_active_stress = [False]*len(self.bc_dict['membrane'])

            self.internalvars['tau_a'], self.internalvars_old['tau_a'] = self.tau_a, self.tau_a_old

            self.actstress, self.wallfields = [], []
            for nm in range(len(self.bc_dict['membrane'])):

                if 'active_stress' in self.bc_dict['membrane'][nm]['params'].keys():
                    self.mem_active_stress[nm], self.have_active_stress = True, True

                    act_curve = self.ti.timecurves(self.bc_dict['membrane'][nm]['params']['active_stress']['activation_curve'])
                    self.actstress.append(activestress_activation(self.bc_dict['membrane'][nm]['params']['active_stress'], act_curve))

                if 'field' in self.bc_dict['membrane'][nm]['params']['h0'].keys():
                    # wall thickness field for reduced solid
                    h0_func = fem.Function(self.V_scalar)
                    self.io.readfunction(h0_func, self.bc_dict['membrane'][nm]['params']['h0']['field'])
                    self.wallfields.append(h0_func)

            w_membrane, self.dbmem, self.bstress = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.ufluid, self.v, self.acc, self.var_v, ivar=self.internalvars, wallfields=self.wallfields)
            w_membrane_old, _, _                 = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.uf_old, self.v_old, self.a_old, self.var_v, ivar=self.internalvars_old, wallfields=self.wallfields)

        w_neumann_prestr, self.deltaW_prestr_kin = ufl.as_ufl(0), ufl.as_ufl(0)
        if self.prestress_initial or self.prestress_initial_only:
            self.funcs_to_update_pre, self.funcs_to_update_vec_pre = [], []
            # Stokes kinetic virtual power
            for n in range(self.num_domains):
                # it seems that we need some slight inertia for this to work smoothly, so let's use transient Stokes here (instead of steady Navier-Stokes or steady Stokes...)
                self.deltaW_prestr_kin += self.vf.deltaW_kin_stokes_transient(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
            if 'neumann_prestress' in self.bc_dict.keys():
                w_neumann_prestr = self.bc.neumann_prestress_bcs(self.bc_dict['neumann_prestress'], self.V_v, self.Vd_scalar, funcs_to_update=self.funcs_to_update_pre, funcs_to_update_vec=self.funcs_to_update_vec_pre)
            if 'membrane' in self.bc_dict.keys():
                self.ufluid_prestr = self.v * self.dt # only incremental displacement needed, since MULF update actually yields a zero displacement after the step
                w_membrane_prestr, _, _ = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.ufluid_prestr, self.v, self.acc, self.var_v, ivar=self.internalvars, wallfields=self.wallfields)
            self.deltaW_prestr_ext = w_neumann_prestr + w_robin + w_stabneumann + w_membrane_prestr
        else:
            assert('neumann_prestress' not in self.bc_dict.keys())

        self.deltaW_ext     = w_neumann + w_body + w_robin + w_stabneumann + w_membrane + w_robin_valve
        self.deltaW_ext_old = w_neumann_old + w_body_old + w_robin_old + w_stabneumann_old + w_membrane_old + w_robin_valve_old

        # stabilization
        if self.stabilization is not None:

            # should only be used for equal-order approximations
            assert(self.order_vel==self.order_pres)

            vscale = self.stabilization['vscale']

            h = self.io.hd0 # cell diameter (could also use max edge length self.io.emax0, but seems to yield similar/same results)

            # full scheme
            if self.stabilization['scheme']=='supg_pspg':

                for n in range(self.num_domains):

                    if self.num_dupl==1: j=0
                    else: j=n

                    tau_supg = h / vscale
                    tau_pspg = h / vscale
                    tau_lsic = h * vscale

                    # strong momentum residuals
                    if self.fluid_governing_type=='navierstokes_transient':
                        residual_v_strong     = self.vf.res_v_strong_navierstokes_transient(self.acc, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], Fale=self.alevar['Fale']), w=self.alevar['w'], Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_navierstokes_transient(self.a_old, self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], Fale=self.alevar['Fale_old']), w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    elif self.fluid_governing_type=='navierstokes_steady':
                        residual_v_strong     = self.vf.res_v_strong_navierstokes_steady(self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], Fale=self.alevar['Fale']), w=self.alevar['w'], Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_navierstokes_steady(self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], Fale=self.alevar['Fale_old']), w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    elif self.fluid_governing_type=='stokes_transient':
                        residual_v_strong     = self.vf.res_v_strong_stokes_transient(self.acc, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p_[j], Fale=self.alevar['Fale']), w=self.alevar['w'], Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_stokes_transient(self.a_old, self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], Fale=self.alevar['Fale_old']), w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    elif self.fluid_governing_type=='stokes_steady':
                        residual_v_strong     = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v, self.p_[j], Fale=self.alevar['Fale']), Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v_old, self.p_old_[j], Fale=self.alevar['Fale_old']), Fale=self.alevar['Fale_old'])
                    else:
                        raise ValueError("Unknown fluid_governing_type!")

                    # SUPG (streamline-upwind Petrov-Galerkin) for Navier-Stokes
                    if self.fluid_governing_type=='navierstokes_transient' or self.fluid_governing_type=='navierstokes_steady':
                        self.deltaW_int     += self.vf.stab_supg(self.acc, self.v, self.p_[j], residual_v_strong, tau_supg, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                        self.deltaW_int_old += self.vf.stab_supg(self.a_old, self.v_old, self.p_old_[j], residual_v_strong_old, tau_supg, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    # PSPG (pressure-stabilizing Petrov-Galerkin) for Navier-Stokes and Stokes
                    self.deltaW_p[n]     += self.vf.stab_pspg(self.acc, self.v, self.p_[j], self.var_p_[j], residual_v_strong, tau_pspg, self.rho[n], self.dx_[n], Fale=self.alevar['Fale'])
                    self.deltaW_p_old[n] += self.vf.stab_pspg(self.a_old, self.v_old, self.p_old_[j], self.var_p_[j], residual_v_strong_old, tau_pspg, self.rho[n], self.dx_[n], Fale=self.alevar['Fale_old'])
                    # LSIC (least-squares on incompressibility constraint) for Navier-Stokes and Stokes
                    self.deltaW_int     += self.vf.stab_lsic(self.v, tau_lsic, self.rho[n], self.dx_[n], Fale=self.alevar['Fale'])
                    self.deltaW_int_old += self.vf.stab_lsic(self.v_old, tau_lsic, self.rho[n], self.dx_[n], Fale=self.alevar['Fale_old'])

            # reduced scheme: missing transient NS term as well as divergence stress term of strong residual
            if self.stabilization['scheme']=='supg_pspg2':

                try: symm = self.stabilization['symmetric']
                except: symm = False

                # optimized for first-order
                assert(self.order_vel==1)
                assert(self.order_pres==1)

                for n in range(self.num_domains):

                    if self.num_dupl==1: j=0
                    else: j=n

                    dscales = self.stabilization['dscales']

                    # yields same result as above for navierstokes_steady
                    delta1 = dscales[0] * self.rho[n] * h / vscale
                    delta2 = dscales[1] * self.rho[n] * h * vscale
                    delta3 = dscales[2] * h / vscale

                    self.deltaW_int     += self.vf.stab_v(delta1, delta2, delta3, self.v, self.p_[j], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'], symmetric=symm)
                    self.deltaW_int_old += self.vf.stab_v(delta1, delta2, delta3, self.v_old, self.p_old_[j], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'], symmetric=symm)

                    self.deltaW_p[n]     += self.vf.stab_p(delta1, delta3, self.v, self.p_[j], self.var_p_[j], self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                    self.deltaW_p_old[n] += self.vf.stab_p(delta1, delta3, self.v_old, self.p_old_[j], self.var_p_[j], self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])

        ### full weakforms

        # kinetic plus internal minus external virtual power
        self.weakform_v = self.timefac_m * self.deltaW_kin + (1.-self.timefac_m) * self.deltaW_kin_old + \
                          self.timefac   * self.deltaW_int + (1.-self.timefac)   * self.deltaW_int_old - \
                          self.timefac   * self.deltaW_ext - (1.-self.timefac)   * self.deltaW_ext_old

        self.weakform_p, self.weakform_lin_vp, self.weakform_lin_pv, self.weakform_lin_pp = [], [], [], []

        for n in range(self.num_domains):
            if self.pressure_at_midpoint:
                self.weakform_p.append( self.timefac * self.deltaW_p[n] + (1.-self.timefac) * self.deltaW_p_old[n] )
            else:
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

        if self.prestress_initial or self.prestress_initial_only:
            # prestressing weak forms
            self.weakform_prestress_p, self.weakform_lin_prestress_vp, self.weakform_lin_prestress_pv, self.weakform_lin_prestress_pp = [], [], [], []
            self.weakform_prestress_v = self.deltaW_prestr_kin + self.deltaW_int - self.deltaW_prestr_ext
            self.weakform_lin_prestress_vv = ufl.derivative(self.weakform_prestress_v, self.v, self.dv)
            for n in range(self.num_domains):
                self.weakform_prestress_p.append( self.deltaW_p[n] )
            for j in range(self.num_dupl):
                self.weakform_lin_prestress_vp.append( ufl.derivative(self.weakform_prestress_v, self.p_[j], self.dp_[j]) )
            for n in range(self.num_domains):
                self.weakform_lin_prestress_pv.append( ufl.derivative(self.weakform_prestress_p[n], self.v, self.dv) )
            if self.stabilization is not None:
                for n in range(self.num_domains):
                    if self.num_dupl==1: j=0
                    else: j=n
                    self.weakform_lin_prestress_pp.append( ufl.derivative(self.weakform_prestress_p, self.p_[j], self.dp_[j]) )


    # active stress ODE evaluation - for reduced solid model
    def evaluate_active_stress_ode(self, t):

        tau_a_, na = [], 0
        for nm in range(len(self.bc_dict['membrane'])):

            if self.mem_active_stress[nm]:
                tau_a_.append(self.actstress[na].tau_act(self.tau_a_old, t, self.dt))
                na+=1
            else:
                tau_a_.append(ufl.as_ufl(0))

        # project and interpolate to quadrature function space
        tau_a_proj = project(tau_a_, self.Vd_scalar, self.dx_, comm=self.comm) # TODO: Should be self.dbmem here, but yields error; why?
        self.tau_a.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.tau_a.interpolate(tau_a_proj)


    # rate equations
    def evaluate_rate_equations(self, t_abs, t_off=0):

        # take care of active stress
        if self.have_active_stress and self.active_stress_trig == 'ode':
            self.evaluate_active_stress_ode(t_abs-t_off)


    def set_problem_residual_jacobian_forms(self):

        tes = time.time()
        if self.comm.rank == 0:
            print('FEM form compilation for fluid...')
            sys.stdout.flush()

        if not bool(self.io.duplicate_mesh_domains):
            if (not self.prestress_initial and not self.prestress_initial_only) or self.restart_step > 0:
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

        if (not self.prestress_initial and not self.prestress_initial_only) or self.restart_step > 0:
            if self.io.USE_MIXED_DOLFINX_BRANCH:
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
                self.res_v = fem.form(self.weakform_v)
                self.res_p = fem.form(self.weakform_p)
                self.jac_vv = fem.form(self.weakform_lin_vv)
                self.jac_vp = fem.form(self.weakform_lin_vp)
                self.jac_pv = fem.form(self.weakform_lin_pv)
                if self.stabilization is not None:
                    self.jac_pp = fem.form(self.weakform_lin_pp)
        else:
            if self.io.USE_MIXED_DOLFINX_BRANCH:
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
            else:
                self.res_v  = fem.form(self.weakform_prestress_v)
                self.res_p  = fem.form(self.weakform_prestress_p)
                self.jac_vv = fem.form(self.weakform_lin_prestress_vv)
                self.jac_vp = fem.form(self.weakform_lin_prestress_vp)
                self.jac_pv = fem.form(self.weakform_lin_prestress_pv)
                if self.stabilization is not None:
                    self.jac_pp = fem.form(self.weakform_lin_prestress_pp)

        tee = time.time() - tes
        if self.comm.rank == 0:
            print('FEM form compilation for fluid finished, te = %.2f s' % (tee))
            sys.stdout.flush()


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
        fem.apply_lifting(self.r_v, [self.jac_vv], [self.bc.dbcs], x0=[self.v.vector], scale=-1.0)
        self.r_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.r_v, self.bc.dbcs, x0=self.v.vector, scale=-1.0)

        # assemble pressure rhs vector
        with self.r_p.localForm() as r_local: r_local.set(0.0)
        if self.num_dupl > 1:
            fem.petsc.assemble_vector_block(self.r_p, self.res_p, self.dummat, bcs=[]) # ghosts are updated inside assemble_vector_block
        else:
            fem.petsc.assemble_vector(self.r_p, self.res_p)
            self.r_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.r_list[0] = self.r_v
        self.r_list[1] = self.r_p

        if bool(self.residual_scale):
            self.scale_residual_list(self.r_list, self.residual_scale)


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

        if bool(self.residual_scale):
            self.scale_jacobian_list(self.K_list, self.residual_scale)


    def get_index_sets(self, isoptions={}):

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            vvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            vvec_ls = self.rom.V.getLocalSize()[1]
        else:
            vvec_or0 = self.v.vector.getOwnershipRange()[0]
            vvec_ls = self.v.vector.getLocalSize()

        offset_v = vvec_or0 + self.p.vector.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec_ls, first=offset_v, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            iset_r = PETSc.IS().createGeneral(self.rom.im_rom_r, comm=self.comm)
            iset_v = iset_v.difference(iset_r) # subtract

        offset_p = offset_v + vvec_ls
        iset_p = PETSc.IS().createStride(self.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        if isoptions['rom_to_new']:
            ilist = [iset_v, iset_p, iset_r]
        else:
            ilist = [iset_v, iset_p]

        return ilist


    # valve law on "immersed" surface (an internal boundary)
    def evaluate_robin_valve(self, t, pu_, pd_):

        for m in range(len(self.bc_dict['robin_valve'])):

            beta_min, beta_max = self.bc_dict['robin_valve'][m]['beta_min'], self.bc_dict['robin_valve'][m]['beta_max']

            beta = expression.template()

            if self.bc_dict['robin_valve'][m]['type'] == 'dp':
                dp_id = self.bc_dict['robin_valve'][m]['dp_monitor_id']
                if pu_[dp_id] < pd_[dp_id]:
                    beta.val = beta_max
                else:
                    beta.val = beta_min

            elif self.bc_dict['robin_valve'][m]['type'] == 'dp_smooth':
                dp_id = self.bc_dict['robin_valve'][m]['dp_monitor_id']
                epsilon = self.bc_dict['robin_valve'][m]['epsilon']
                beta.val = 0.5*(beta_max - beta_min)*(ufl.tanh((pd_[dp_id] - pu_[dp_id])/epsilon) + 1.) + beta_min

            elif self.bc_dict['robin_valve'][m]['type'] == 'temporal':
                to, tc = self.bc_dict['robin_valve'][m]['to'], self.bc_dict['robin_valve'][m]['tc']
                if to > tc:
                    if t < to and t >= tc:
                        beta.val = beta_max
                    if t >= to or t < tc:
                        beta.val = beta_min
                else:
                    if t < to or t >= tc:
                        beta.val = beta_max
                    if t >= to and t < tc:
                        beta.val = beta_min

            else:
                raise ValueError("Unknown Robin valve type!")

            self.beta_valve[m].interpolate(beta.evaluate)


    def evaluate_dp_monitor(self, pu_, pd_, pint_u_, pint_d_, a_u_, a_d_, prnt=True):

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

            if prnt:
                if self.comm.rank == 0:
                    print("dp ID "+str(self.bc_dict['dp_monitor'][m]['id'])+": pu = %.4e, pd = %.4e" % (pu_[m],pd_[m]))
                    sys.stdout.flush()


    def evaluate_flux_monitor(self, qv_, q_, prnt=True):

        for m in range(len(self.bc_dict['flux_monitor'])):

            q = fem.assemble_scalar(q_[m])
            q = self.comm.allgather(q)
            qv_[m] = sum(q)

            if prnt:
                if self.comm.rank == 0:
                    print("Flux ID "+str(self.bc_dict['flux_monitor'][m]['id'])+": q = %.4e" % (qv_[m]))
                    sys.stdout.flush()


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # read restart information
        if self.restart_step > 0:
            self.io.readcheckpoint(self, N)
            self.simname += '_r'+str(N)


    def evaluate_initial(self):

        if self.have_flux_monitor:
            self.evaluate_flux_monitor(self.qv_old_, self.q_old_)
            for k in self.qv_old_: self.qv_[k] = self.qv_old_[k]
        if self.have_dp_monitor:
            self.evaluate_dp_monitor(self.pu_old_, self.pd_old_, self.pint_u_old_, self.pint_d_old_, self.a_u_old_, self.a_d_old_)
            for k in self.pu_old_: self.pu_[k] = self.pu_old_[k]
            for k in self.pd_old_: self.pd_[k] = self.pd_old_[k]
        if self.have_robin_valve:
            self.evaluate_robin_valve(self.t_init, self.pu_old_, self.pd_old_)


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):
        return 0.


    def evaluate_pre_solve(self, t, N):

        # set time-dependent functions
        self.ti.set_time_funcs(t, self.ti.funcs_to_update, self.ti.funcs_to_update_vec)

        # evaluate rate equations
        self.evaluate_rate_equations(t)

        # DBC from files
        if self.bc.have_dirichlet_file:
            for m in self.ti.funcs_data:
                file = list(m.values())[0].replace('*',str(N))
                func = list(m.keys())[0]
                self.io.readfunction(func, file)


    def evaluate_post_solve(self, t, N):

        if self.have_flux_monitor:
            self.evaluate_flux_monitor(self.qv_, self.q_)
        if self.have_dp_monitor:
            self.evaluate_dp_monitor(self.pu_, self.pd_, self.pint_u_, self.pint_d_, self.a_u_, self.a_d_)
        if self.have_robin_valve:
            self.evaluate_robin_valve(t, self.pu_, self.pd_)


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


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)


    def check_abort(self, t):
        pass


    def destroy(self):
        pass



class FluidmechanicsSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.fluid_governing_type == 'navierstokes_transient' or self.pb.fluid_governing_type == 'stokes_transient') and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.a_old, self.pb.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            if self.pb.io.USE_MIXED_DOLFINX_BRANCH:
                res_a, jac_aa  = fem.form(weakform_a, entity_maps=self.pb.io.entity_maps), fem.form(weakform_lin_aa, entity_maps=self.pb.io.entity_maps)
            else:
                res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)


    def solve_initial_prestress(self):

        utilities.print_prestress('start', self.pb.comm)

        if self.pb.prestress_ptc: self.solnln.PTC = True

        dt_prestr = self.pb.prestress_maxtime/self.pb.prestress_numstep

        for N in range(1,self.pb.prestress_numstep+1):

            wts = time.time()

            tprestr = N * dt_prestr

            self.pb.ti.set_time_funcs(tprestr, self.pb.funcs_to_update_pre, self.pb.funcs_to_update_vec_pre)

            self.solnln.newton(tprestr)

            # update uf_pre
            self.pb.ki.prestress_update(dt_prestr, self.pb.v.vector)
            utilities.print_prestress('updt', self.pb.comm)

            # update fluid velocity: v_old <- v - we need some slight inertia for this to work smoothly...
            self.pb.v_old.vector.axpby(1.0, 0.0, self.pb.v.vector)
            self.pb.v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            wt = time.time() - wts

            # print time step info to screen
            self.pb.ti.print_prestress_step(N, tprestr, self.pb.prestress_numstep, self.solnln.lsp, ni=self.solnln.ni, li=self.solnln.li, wt=wt)

        utilities.print_prestress('end', self.pb.comm)

        # write prestress displacement (given that we want to write the fluid displacement)
        if 'fluiddisplacement' in self.pb.results_to_write and self.pb.io.write_results_every > 0:
            self.pb.io.write_output_pre(self.pb, self.pb.uf_pre, 0, 'fluiddisplacement_pre')

        if self.pb.prestress_initial_only:
            # it may be convenient to write the prestress displacement field to a file for later read-in
            self.pb.io.writefunction(self.pb.uf_pre, self.pb.io.output_path_pre+'/results_'+self.pb.simname+'_fluiddisplacement_pre.txt')
            if bool(self.pb.io.duplicate_mesh_domains):
                m=0
                for j in self.pb.io.duplicate_mesh_domains:
                     # TODO: Might not work for duplicate mesh, since we do not have the input node indices (do we...?)
                    self.pb.io.writefunction(self.pb.p_[m], self.pb.io.output_path_pre+'/results_'+self.pb.simname+'_pressure'+str(m+1)+'_pre.txt')
                    m+=1
            else:
                self.pb.io.writefunction(self.pb.p_[0], self.pb.io.output_path_pre+'/results_'+self.pb.simname+'_pressure_pre.txt')
            if self.pb.comm.rank == 0:
                print("Prestress only done. To resume, set file path(s) in 'prestress_from_file' and read in uf_pre.")
                sys.stdout.flush()
            os._exit(0)

        # reset state
        self.pb.v.vector.set(0.0)
        self.pb.v.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.v_old.vector.set(0.0)
        self.pb.v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # reset PTC flag to what it was
        if self.pb.prestress_ptc:
            try: self.solnln.PTC = self.solver_params['ptc']
            except: self.solnln.PTC = False

        # set flag to false again
        self.pb.prestress_initial = False
        self.pb.set_problem_residual_jacobian_forms()


# prestress solver, to be called from other (coupled) problems
class FluidmechanicsSolverPrestr(FluidmechanicsSolver):

    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):
        raise RuntimeError("You should not be here!")
