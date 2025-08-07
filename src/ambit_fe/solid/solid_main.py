#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy, os
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
import basix
from petsc4py import PETSc

from . import solid_kinematics_constitutive
from . import solid_variationalform
from .. import timeintegration
from .. import utilities, mathutils
from .. import boundaryconditions
from .. import ioparams, expression
from ..solver import solver_nonlin
from ..solver.projection import project
from . import solid_material
from .solid_material import activestress_activation

from ..base import problem_base, solver_base


"""
Solid mechanics governing equation

\rho_{0} \ddot{\boldsymbol{u}} = \boldsymbol{\nabla}_{0} \cdot (\boldsymbol{F}\boldsymbol{S}) + \hat{\boldsymbol{b}}_{0} \quad \text{in} \; \Omega_{0} \times [0, T]

can be solved together with constraint J = 1 (2-field variational principle with u and p as degrees of freedom)
J-1 = 0 \quad \text{in} \; \Omega_{0} \times [0, T]
"""

class SolidmechanicsProblem(problem_base):

    def __init__(self, pbase, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params={}):

        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm

        ioparams.check_params_fem_solid(fem_params)
        ioparams.check_params_time_solid(time_params)

        self.problem_physics = 'solid'

        self.timint = time_params.get('timint', 'static')

        self.results_to_write = io_params['results_to_write']

        self.io = io

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        # for FSI, we want to specify the subdomains
        self.domain_ids = self.io.io_params.get('domain_ids_solid', np.arange(1,self.num_domains+1))

        # TODO: Find nicer solution here...
        if self.pbase.problem_type=='fsi' or self.pbase.problem_type=='fsi_flow0d':
            self.dx, self.bmeasures = self.io.dx, self.io.bmeasures
        else:
            self.dx, self.bmeasures = self.io.create_integration_measures(self.io.mesh, [self.io.mt_d,self.io.mt_b1,self.io.mt_b2])

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_disp = fem_params['order_disp']
        self.order_pres = fem_params.get('order_pres', 1)
        self.quad_degree = fem_params['quad_degree']

        self.incompressibility = fem_params.get('incompressibility', 'no')

        if self.incompressibility=='no':
            self.incompressible_2field = False
        elif self.incompressibility=='full':
            self.incompressible_2field = True
        elif self.incompressibility=='nearly':
            self.incompressible_2field = True
            self.bulkmod = fem_params['bulkmod']
        else:
            raise ValueError("Unknown setting for 'incompressibility'. Choose 'no', 'full', or 'nearly'.")

        self.fem_params = fem_params

        # collect domain data
        self.rho0 = []
        for n, M in enumerate(self.domain_ids):
            # data for inertial forces: density
            if self.timint != 'static':
                self.rho0.append(self.constitutive_models['MAT'+str(n+1)]['inertia']['rho0'])

        self.prestress_initial = fem_params.get('prestress_initial', False)
        self.prestress_initial_only = fem_params.get('prestress_initial_only', False)
        self.prestress_maxtime = self.pbase.ctrl_params.get('prestress_maxtime', 1.0)
        self.prestress_numstep = self.pbase.ctrl_params.get('prestress_numstep', 1)
        self.prestress_dt = self.pbase.ctrl_params.get('prestress_dt', self.prestress_maxtime/self.prestress_numstep)
        if 'prestress_dt' in self.pbase.ctrl_params.keys(): self.prestress_numstep = int(self.prestress_maxtime/self.prestress_dt)
        self.prestress_ptc = fem_params.get('prestress_ptc', False)
        self.prestress_from_file = fem_params.get('prestress_from_file', False)

        if bool(self.prestress_from_file): self.prestress_initial, self.prestress_initial_only = False, False

        if self.prestress_initial or self.prestress_initial_only:
            self.constitutive_models_prestr = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.have_condensed_variables = False

        self.dim = self.io.mesh.geometry.dim

        self.sub_solve = False
        self.print_subiter = False

        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            self.dg_type = "DG"
            if (self.order_disp > 1 or self.order_pres > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            self.dg_type = "DQ"
            if (self.order_disp > 1 or self.order_pres > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
            if self.quad_degree < 2:
                raise ValueError("Use at least a quadrature degree >= 2 for a hexahedral mesh!")
        else:
            raise NameError("Unknown cell/element type!")

        self.basix_celltype = utilities.get_basix_cell_type(self.io.mesh.ufl_cell())

        self.Vex = self.io.mesh.ufl_domain().ufl_coordinate_element()

        # model order reduction
        self.mor_params = mor_params
        if bool(self.mor_params): self.pbase.have_rom = True
        else: self.pbase.have_rom = False
        # will be set by solver base class
        self.rom = None

        # function spaces for u and p
        self.V_u = fem.functionspace(self.io.mesh, ("Lagrange", self.order_disp, (self.io.mesh.geometry.dim,)))
        self.V_p = fem.functionspace(self.io.mesh, ("Lagrange", self.order_pres))

        # continuous tensor and scalar function spaces of order order_disp
        self.V_tensor = fem.functionspace(self.io.mesh, ("Lagrange", self.order_disp, (self.io.mesh.geometry.dim,self.io.mesh.geometry.dim)))
        self.V_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.order_disp))

        # discontinuous function spaces
        self.Vd_tensor = fem.functionspace(self.io.mesh, (self.dg_type, self.order_disp-1, (self.io.mesh.geometry.dim,self.io.mesh.geometry.dim)))
        self.Vd_vector = fem.functionspace(self.io.mesh, (self.dg_type, self.order_disp-1, (self.io.mesh.geometry.dim,)))
        self.Vd_scalar = fem.functionspace(self.io.mesh, (self.dg_type, self.order_disp-1))

        # for output writing - function spaces on the degree of the mesh
        self.mesh_degree = self.io.mesh._ufl_domain._ufl_coordinate_element._degree
        self.V_out_tensor = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree, (self.io.mesh.geometry.dim,self.io.mesh.geometry.dim)))
        self.V_out_vector = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree, (self.io.mesh.geometry.dim,)))
        self.V_out_scalar = fem.functionspace(self.io.mesh, ("Lagrange", self.mesh_degree))

        # coordinate element function space - based on input mesh
        self.Vcoord = fem.functionspace(self.io.mesh, self.Vex)

        # # Quadrature tensor, vector, and scalar elements
        # Q_tensor = ufl.TensorElement("Quadrature", self.io.mesh.ufl_cell(), degree=self.quad_degree, quad_scheme="default")
        # Q_vector = ufl.VectorElement("Quadrature", self.io.mesh.ufl_cell(), degree=self.quad_degree, quad_scheme="default")
        # Q_scalar = ufl.FiniteElement("Quadrature", self.io.mesh.ufl_cell(), degree=self.quad_degree, quad_scheme="default")
        #
        # # quadrature function spaces
        # self.Vq_tensor = fem.FunctionSpace(self.io.mesh, Q_tensor)
        # self.Vq_vector = fem.FunctionSpace(self.io.mesh, Q_vector)
        # self.Vq_scalar = fem.FunctionSpace(self.io.mesh, Q_scalar)
        #
        # self.quadrature_points, wts = basix.make_quadrature(basix_celltype, self.quad_degree)

        # functions
        self.du    = ufl.TrialFunction(self.V_u)            # Incremental displacement
        self.var_u = ufl.TestFunction(self.V_u)             # Test function
        self.dp    = ufl.TrialFunction(self.V_p)            # Incremental pressure
        self.var_p = ufl.TestFunction(self.V_p)             # Test function
        self.u     = fem.Function(self.V_u, name="Displacement")
        self.p     = fem.Function(self.V_p, name="Pressure")

        # auxiliary velocity and acceleration vectors
        self.v     = fem.Function(self.V_u, name="Velocity")
        self.a     = fem.Function(self.V_u, name="Acceleration")
        # values of previous time step
        self.u_old = fem.Function(self.V_u)
        self.v_old = fem.Function(self.V_u)
        self.a_old = fem.Function(self.V_u)
        self.p_old = fem.Function(self.V_p)
        # a setpoint displacement for multiscale analysis
        self.u_set = fem.Function(self.V_u)
        self.p_set = fem.Function(self.V_p)
        self.tau_a_set = fem.Function(self.Vd_scalar)
        # growth stretch
        self.theta = fem.Function(self.Vd_scalar, name="theta")
        self.theta_old = fem.Function(self.Vd_scalar)
        self.growth_thres = fem.Function(self.Vd_scalar)
        # plastic deformation gradient
        self.F_plast = fem.Function(self.Vd_tensor)
        self.F_plast_old = fem.Function(self.Vd_tensor)
        # initialize to one (theta = 1 means no growth)
        self.theta.x.petsc_vec.set(1.0), self.theta_old.x.petsc_vec.set(1.0)
        self.theta.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD), self.theta_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # active stress
        # self.tau_a = fem.Function(self.Vq_scalar, name="tau_a")
        self.tau_a = fem.Function(self.Vd_scalar, name="tau_a")
        self.tau_a_old = fem.Function(self.Vd_scalar)
        self.amp_old, self.amp_old_set = fem.Function(self.Vd_scalar), fem.Function(self.Vd_scalar)
        self.amp_old.x.petsc_vec.set(1.0), self.amp_old_set.x.petsc_vec.set(1.0)
        self.amp_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD), self.amp_old_set.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # prestress displacement
        if (self.prestress_initial or self.prestress_initial_only) or bool(self.prestress_from_file):
            self.u_pre = fem.Function(self.V_u, name="Displacement_prestress")
        else:
            self.u_pre = None
        if (self.prestress_initial or self.prestress_initial_only) and self.pbase.restart_step == 0:
            self.pre = True
        else:
            self.pre = False

        # for ROM, provide pointers to main variable and its derivative
        if self.pbase.have_rom:
            self.xr_, self.xr_old_, self.xrpre_ = self.u, self.u_old, self.u_pre
            self.xdtr_old_, self.xintrpre_ = self.v_old, None

        # own read function: requires plain txt format of type "node-id val-x val-y val-z" (or one value in case of a scalar)
        if bool(self.prestress_from_file):
            self.io.readfunction(self.u_pre, self.prestress_from_file[0])
            # if available, we might want to read in the pressure field, too
            if self.incompressible_2field:
                if len(self.prestress_from_file)>1:
                    self.io.readfunction(self.p, self.prestress_from_file[1])
                    self.io.readfunction(self.p_old, self.prestress_from_file[1])

        self.volume_laplace = io_params.get('volume_laplace', [])

        # dictionaries of internal variables
        self.internalvars, self.internalvars_old, self.internalvars_mid = {}, {}, {}

        # reference coordinates
        self.x_ref = ufl.SpatialCoordinate(self.io.mesh)

        if self.incompressible_2field:
            self.numdof = self.u.x.petsc_vec.getSize() + self.p.x.petsc_vec.getSize()
        else:
            self.numdof = self.u.x.petsc_vec.getSize()

        self.mor_params = mor_params

        # initialize solid time-integration class
        self.ti = timeintegration.timeintegration_solid(time_params, self.pbase.dt, self.pbase.numstep, incompr=self.incompressible_2field, time_curves=time_curves, t_init=self.pbase.t_init, dim=self.dim, comm=self.pbase.comm)

        # get time factors
        self.timefac_m, self.timefac = self.ti.timefactors()
        if self.ti.eval_nonlin_terms=='midpoint': self.midp = True
        else: self.midp = False

        # check for materials that need extra treatment (anisotropic, active stress, growth, ...)
        self.have_frank_starling, self.have_growth, self.have_plasticity = False, False, False
        self.mat_active_stress, self.mat_growth, self.mat_remodel, self.mat_growth_dir, self.mat_growth_trig, self.mat_growth_thres, self.mat_plastic = [False]*self.num_domains, [False]*self.num_domains, [False]*self.num_domains, [None]*self.num_domains, [None]*self.num_domains, []*self.num_domains, [False]*self.num_domains
        self.mat_active_stress_type = ['ode']*self.num_domains

        self.localsolve, growth_dir = False, None
        self.actstress, self.act_curve, self.act_curve_old, self.activemodel = [], [], [], [None]*self.num_domains
        for n in range(self.num_domains):

            if 'holzapfelogden_dev' in self.constitutive_models['MAT'+str(n+1)].keys() or 'guccione_dev' in self.constitutive_models['MAT'+str(n+1)].keys():
                assert(len(self.io.fiber_data)>1)

            if 'active_fiber' in self.constitutive_models['MAT'+str(n+1)].keys() or 'active_crossfiber' in self.constitutive_models['MAT'+str(n+1)].keys() or 'active_iso' in self.constitutive_models['MAT'+str(n+1)].keys():
                if 'active_fiber' in self.constitutive_models['MAT'+str(n+1)].keys(): self.activemodel[n] = 'active_fiber'
                if 'active_crossfiber' in self.constitutive_models['MAT'+str(n+1)].keys(): self.activemodel[n] = 'active_crossfiber'
                if 'active_iso' in self.constitutive_models['MAT'+str(n+1)].keys(): self.activemodel[n] = 'active_iso'
                if self.activemodel[n] == 'active_fiber' or self.activemodel[n] == 'active_crossfiber': assert(bool(self.io.fiber_data))
                self.mat_active_stress[n] = True
                # get type of active stress
                try: self.mat_active_stress_type[n] = self.constitutive_models['MAT'+str(n+1)][self.activemodel[n]]['type']
                except: pass # default is 'ode'
                if self.mat_active_stress_type[n] == 'ode':
                    self.act_curve.append( fem.Function(self.Vd_scalar) )
                    self.ti.funcs_to_update.append({self.act_curve[-1] : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)][self.activemodel[n]]['activation_curve'])})
                    self.actstress.append(activestress_activation(self.constitutive_models['MAT'+str(n+1)][self.activemodel[n]], self.act_curve[-1], x_ref=self.x_ref))
                    if self.actstress[-1].frankstarling:
                        self.have_frank_starling = True
                        self.act_curve_old.append( fem.Function(self.Vd_scalar) )
                        # we need to initialize the old activation curve here to get the correct stretch state evaluation
                        load = expression.template()
                        load.val = self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)][self.activemodel[n]]['activation_curve'])(self.pbase.t_init)
                        self.act_curve_old[-1].interpolate(load.evaluate)
                        self.act_curve_old[-1].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                        self.ti.funcs_to_update_old.append({self.act_curve_old[-1] : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)][self.activemodel[n]]['activation_curve'])})
                        self.actstress[-1].act_curve_old = self.act_curve_old[-1] # needed for Frank-Starling law
                    else:
                        self.ti.funcs_to_update_old.append({None : -1}) # not needed, since tau_a_old <- tau_a at end of time step
                if self.mat_active_stress_type[n] == 'prescribed':
                    self.act_curve.append( fem.Function(self.Vd_scalar) )
                    self.ti.funcs_to_update.append({self.tau_a : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)][self.activemodel[n]]['prescribed_curve'])})
                    self.ti.funcs_to_update_old.append({None : -1}) # not needed, since tau_a_old <- tau_a at end of time step
                if self.mat_active_stress_type[n] == 'prescribed_from_file':
                    self.actpid = n+1 # file acts for all active stress models in all domains!
                self.internalvars['tau_a'], self.internalvars_old['tau_a'], self.internalvars_mid['tau_a'] = self.tau_a, self.tau_a_old, self.timefac*self.tau_a + (1.-self.timefac)*self.tau_a_old

            if 'growth' in self.constitutive_models['MAT'+str(n+1)].keys():
                self.mat_growth[n], self.have_growth = True, True
                self.mat_growth_dir[n] = self.constitutive_models['MAT'+str(n+1)]['growth']['growth_dir']
                self.mat_growth_trig[n] = self.constitutive_models['MAT'+str(n+1)]['growth']['growth_trig']
                # need to have fiber fields for the following growth options
                if self.mat_growth_dir[n] == 'fiber' or self.mat_growth_trig[n] == 'fibstretch':
                    assert(bool(self.io.fiber_data))
                if self.mat_growth_dir[n] == 'radial':
                    assert(len(self.io.fiber_data)>1)
                # in this case, we have a theta that is (nonlinearly) dependent on the deformation, theta = theta(C(u)),
                # therefore we need a local Newton iteration to solve for equilibrium theta (return mapping) prior to entering
                # the global Newton scheme - so flag localsolve to true
                if self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                    self.localsolve = True
                    self.mat_growth_thres.append(self.constitutive_models['MAT'+str(n+1)]['growth']['growth_thres'])
                else:
                    self.mat_growth_thres.append(ufl.as_ufl(0))
                # for the case that we have a prescribed growth stretch over time, append curve to functions that need time updates
                # if one mat has a prescribed growth model, all have to be!
                if self.mat_growth_trig[n] == 'prescribed':
                    self.ti.funcs_to_update.append({self.theta : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)]['growth']['prescribed_curve'])})
                    self.ti.funcs_to_update_old.append({None : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)]['growth']['prescribed_curve'])})
                if 'remodeling_mat' in self.constitutive_models['MAT'+str(n+1)]['growth'].keys():
                    self.mat_remodel[n] = True
                self.internalvars['theta'], self.internalvars_old['theta'], self.internalvars_mid['theta'] = self.theta, self.theta_old, self.timefac*self.theta + (1.-self.timefac)*self.theta_old
            else:
                self.mat_growth_thres.append(ufl.as_ufl(0))

            if 'plastic' in self.constitutive_models['MAT'+str(n+1)].keys():
                self.mat_plastic[n], self.have_plasticity = True, True
                self.localsolve = True
                self.internalvars['e_plast'], self.internalvars_old['e_plast'], self.internalvars_mid['e_plast'] = self.F_plast, self.F_plast_old, self.timefac*self.F_plast + (1.-self.timefac)*self.F_plast_old

        # full linearization of our remodeling law can lead to excessive compiler times for FFCx... :-/
        # let's try if we might can go without one of the critial terms (derivative of remodeling fraction w.r.t. C)
        self.lin_remod_full = fem_params.get('lin_remodeling_full', True)

        # growth threshold (as function, since in multiscale approach, it can vary element-wise)
        if self.have_growth and self.localsolve:
            growth_thres_proj = project(self.mat_growth_thres, self.Vd_scalar, self.dx, domids=self.domain_ids, comm=self.pbase.comm, entity_maps=self.io.entity_maps)
            self.growth_thres.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.growth_thres.interpolate(growth_thres_proj)

        # read in fiber data
        if bool(self.io.fiber_data):

            self.fibarray = []
            for nf in range(len(self.io.fiber_data)):
                self.fibarray.append('f'+str(nf+1))

            self.fib_func = self.io.readin_fibers(self.fibarray, self.V_u, self.dx, self.domain_ids, self.order_disp)

        else:
            self.fib_func = None

        # for multiscale G&R analysis
        self.tol_stop_large = 0

        # initialize kinematics class
        self.ki = solid_kinematics_constitutive.kinematics(self.dim, fib_funcs=self.fib_func, u_pre=self.u_pre)

        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(solid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)], self.incompressible_2field, mat_growth=self.mat_growth[n], mat_remodel=self.mat_remodel[n], mat_plastic=self.mat_plastic[n]))

        # for prestress, we don't have any inelastic or rate-dependent stuff
        if self.prestress_initial or self.prestress_initial_only:
            self.ma_prestr = []
            mat_remove = ['visco_green','growth','plastic']
            for n in range(self.num_domains):
                for mr in mat_remove:
                    try:    self.constitutive_models_prestr['MAT'+str(n+1)].pop(mr)
                    except: pass
                self.ma_prestr.append(solid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models_prestr['MAT'+str(n+1)], self.incompressible_2field, mat_growth=False, mat_remodel=False, mat_plastic=False))

        # initialize solid variational form class
        self.vf = solid_variationalform.variationalform(self.var_u, var_p=self.var_p, du=self.du, dp=self.dp, n0=self.io.n0, x_ref=self.x_ref)

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond(self.io, fem_params=self.fem_params, vf=self.vf, ti=self.ti, ki=self.ki, V_field=self.V_u, Vdisc_scalar=self.Vd_scalar)
        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.bc_dict['dirichlet'])

        if 'dirichlet_vol' in self.bc_dict.keys():
            self.bc.dirichlet_vol(self.bc_dict['dirichlet_vol'])

        self.set_variational_forms()

        self.pbrom = self # self-pointer needed for ROM solver access
        self.pbrom_host = self
        self.V_rom = self.V_u


    def get_problem_var_list(self):

        if self.incompressible_2field:
            is_ghosted = [1, 1]
            return [self.u.x.petsc_vec, self.p.x.petsc_vec], is_ghosted
        else:
            is_ghosted = [1]
            return [self.u.x.petsc_vec], is_ghosted


    # the main function that defines the solid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):

        # set forms for acceleration and velocity
        self.acc, self.vel = self.ti.set_acc_vel(self.u, self.u_old, self.v_old, self.a_old)

        # set mid-point representations (if needed...)
        self.acc_mid = self.timefac_m * self.acc + (1.-self.timefac_m) * self.a_old
        self.vel_mid = self.timefac   * self.vel + (1.-self.timefac)   * self.v_old
        self.us_mid  = self.timefac   * self.u   + (1.-self.timefac)   * self.u_old
        self.ps_mid  = self.timefac   * self.p   + (1.-self.timefac)   * self.p_old

        # kinetic, internal, and pressure virtual work
        self.deltaW_kin, self.deltaW_kin_old, self.deltaW_kin_mid  = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_int, self.deltaW_int_old, self.deltaW_int_mid  = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_p,   self.deltaW_p_old,   self.deltaW_p_mid    = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

        for n, M in enumerate(self.domain_ids):

            if self.timint != 'static':
                # kinetic virtual work
                self.deltaW_kin     += self.vf.deltaW_kin(self.acc, self.rho0[n], self.dx(M))
                self.deltaW_kin_old += self.vf.deltaW_kin(self.a_old, self.rho0[n], self.dx(M))
                self.deltaW_kin_mid += self.vf.deltaW_kin(self.acc_mid, self.rho0[n], self.dx(M))

            # internal virtual work
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.ki.F(self.u), self.dx(M))
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].S(self.u_old, self.p_old, self.v_old, ivar=self.internalvars_old), self.ki.F(self.u_old), self.dx(M))
            self.deltaW_int_mid += self.vf.deltaW_int(self.ma[n].S(self.us_mid, self.ps_mid, self.vel_mid, ivar=self.internalvars_mid), self.ki.F(self.us_mid), self.dx(M))

            # pressure virtual work (for incompressible formulation)
            # this has to be treated like the evaluation of a volumetric material, hence with the elastic part of J
            if self.mat_growth[n]: J, J_old, J_mid = self.ma[n].J_e(self.u, self.theta), self.ma[n].J_e(self.u_old, self.theta_old), self.ma[n].J_e(self.us_mid, self.timefac*self.theta+(1.-self.timefac)*self.theta_old)
            else:                  J, J_old, J_mid = self.ki.J(self.u), self.ki.J(self.u_old), self.ki.J(self.us_mid)
            if self.incompressibility=='full':
                self.deltaW_p       += self.vf.deltaW_int_pres(J, self.dx(M))
                self.deltaW_p_old   += self.vf.deltaW_int_pres(J_old, self.dx(M))
                self.deltaW_p_mid   += self.vf.deltaW_int_pres(J_mid, self.dx(M))
            if self.incompressibility=='nearly':
                self.deltaW_p       += self.vf.deltaW_int_pres_nearly(J, self.p, self.bulkmod, self.dx(M))
                self.deltaW_p_old   += self.vf.deltaW_int_pres_nearly(J_old, self.p_old, self.bulkmod, self.dx(M))
                self.deltaW_p_mid   += self.vf.deltaW_int_pres_nearly(J_mid, self.ps_mid, self.bulkmod, self.dx(M))

        # external virtual work (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_body, w_robin, w_membrane = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        w_neumann_old, w_body_old, w_robin_old, w_membrane_old = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        w_neumann_mid, w_body_mid, w_robin_mid, w_membrane_mid = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann     = self.bc.neumann_bcs(self.bc_dict['neumann'], self.bmeasures, F=self.ki.F(self.u,ext=True), funcs_to_update=self.ti.funcs_to_update, funcs_to_update_vec=self.ti.funcs_to_update_vec, funcsexpr_to_update=self.ti.funcsexpr_to_update, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec)
            w_neumann_old = self.bc.neumann_bcs(self.bc_dict['neumann'], self.bmeasures, F=self.ki.F(self.u_old,ext=True), funcs_to_update=self.ti.funcs_to_update_old, funcs_to_update_vec=self.ti.funcs_to_update_vec_old, funcsexpr_to_update=self.ti.funcsexpr_to_update_old, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec_old)
            w_neumann_mid = self.bc.neumann_bcs(self.bc_dict['neumann'], self.bmeasures, F=self.ki.F(self.us_mid,ext=True), funcs_to_update=self.ti.funcs_to_update_mid, funcs_to_update_vec=self.ti.funcs_to_update_vec_mid, funcsexpr_to_update=self.ti.funcsexpr_to_update_mid, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec_mid)
        if 'bodyforce' in self.bc_dict.keys():
            w_body      = self.bc.bodyforce(self.bc_dict['bodyforce'], self.dx, funcs_to_update=self.ti.funcs_to_update, funcsexpr_to_update=self.ti.funcsexpr_to_update)
            w_body_old  = self.bc.bodyforce(self.bc_dict['bodyforce'], self.dx, funcs_to_update=self.ti.funcs_to_update_old, funcsexpr_to_update=self.ti.funcsexpr_to_update_old)
            w_body_mid  = self.bc.bodyforce(self.bc_dict['bodyforce'], self.dx, funcs_to_update=self.ti.funcs_to_update_mid, funcsexpr_to_update=self.ti.funcsexpr_to_update_mid)
        if 'robin' in self.bc_dict.keys():
            w_robin     = self.bc.robin_bcs(self.bc_dict['robin'], self.u, self.vel, self.bmeasures, u_pre=self.u_pre)
            w_robin_old = self.bc.robin_bcs(self.bc_dict['robin'], self.u_old, self.v_old, self.bmeasures, u_pre=self.u_pre)
            w_robin_mid = self.bc.robin_bcs(self.bc_dict['robin'], self.us_mid, self.vel_mid, self.bmeasures, u_pre=self.u_pre)
        if 'membrane' in self.bc_dict.keys():
            w_membrane, self.idmem, self.bstress, self.bstrainenergy, self.bintpower = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.u, self.vel, self.acc, self.bmeasures)
            w_membrane_old, _, _, _, _                                               = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.u_old, self.v_old, self.a_old, self.bmeasures)
            w_membrane_mid, _, _, _, _                                               = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.us_mid, self.vel_mid, self.acc_mid, self.bmeasures)

        # for (quasi-static) prestressing, we need to eliminate dashpots in our external virtual work
        # plus no rate-dependent or inelastic constitutive models
        w_neumann_prestr, w_robin_prestr, self.deltaW_prestr_int = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if self.prestress_initial or self.prestress_initial_only:
            # internal virtual work
            for n, M in enumerate(self.domain_ids):
                self.deltaW_prestr_int += self.vf.deltaW_int(self.ma_prestr[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.ki.F(self.u), self.dx(M))
            # boundary conditions
            bc_dict_prestr = copy.deepcopy(self.bc_dict)
            # get rid of dashpots
            if 'robin' in bc_dict_prestr.keys():
                for r in bc_dict_prestr['robin']:
                    if r['type'] == 'dashpot': r['visc'] = 0.
            bc_prestr = boundaryconditions.boundary_cond(self.io, fem_params=self.fem_params, vf=self.vf, ti=self.ti, ki=self.ki, V_field=self.V_u, Vdisc_scalar=self.Vd_scalar)
            if 'neumann_prestress' in bc_dict_prestr.keys():
                w_neumann_prestr = bc_prestr.neumann_prestress_bcs(bc_dict_prestr['neumann_prestress'], self.bmeasures, funcs_to_update=self.ti.funcs_to_update_pre, funcs_to_update_vec=self.ti.funcs_to_update_vec_pre, funcsexpr_to_update=self.ti.funcsexpr_to_update_pre, funcsexpr_to_update_vec=self.ti.funcsexpr_to_update_vec_pre)
            if 'robin' in bc_dict_prestr.keys():
                w_robin_prestr = bc_prestr.robin_bcs(bc_dict_prestr['robin'], self.u, self.vel, self.bmeasures, u_pre=self.u_pre)
            self.deltaW_prestr_ext = w_neumann_prestr + w_robin_prestr
        else:
            assert('neumann_prestress' not in self.bc_dict.keys())

        self.deltaW_ext     = w_neumann + w_body + w_robin + w_membrane
        self.deltaW_ext_old = w_neumann_old + w_body_old + w_robin_old + w_membrane_old
        self.deltaW_ext_mid = w_neumann_mid + w_body_mid + w_robin_mid + w_membrane_mid

        ### full weakforms

        # quasi-static weak form: internal minus external virtual work
        if self.timint == 'static':

            self.weakform_u = self.deltaW_int - self.deltaW_ext

            if self.incompressible_2field:
                self.weakform_p = self.deltaW_p

        # full dynamic weak form: kinetic plus internal minus external virtual work
        else:

            # evaluate nonlinear terms trapezoidal-like: a * f(u_{n+1}) + (1-a) * f(u_{n})
            if self.ti.eval_nonlin_terms=='trapezoidal':

                self.weakform_u = self.timefac_m * self.deltaW_kin + (1.-self.timefac_m) * self.deltaW_kin_old + \
                                  self.timefac   * self.deltaW_int + (1.-self.timefac)   * self.deltaW_int_old - \
                                  self.timefac   * self.deltaW_ext - (1.-self.timefac)   * self.deltaW_ext_old

            # evaluate nonlinear terms midpoint-like: f(a*u_{n+1} + (1-a)*u_{n})
            elif self.ti.eval_nonlin_terms=='midpoint':

                self.weakform_u = self.deltaW_kin_mid + self.deltaW_int_mid - self.deltaW_ext_mid

            else:
                raise ValueError("Unknown eval_nonlin_terms option. Choose 'trapezoidal' or 'midpoint'.")

            if self.incompressible_2field:
                self.weakform_p = self.deltaW_p

        ### local weak forms at Gauss points for inelastic materials
        self.localdata = {}
        self.localdata['var'], self.localdata['res'], self.localdata['inc'], self.localdata['fnc'] = [], [], [], []

        if self.have_growth:

            self.r_growth, self.del_theta = [], []

            for n in range(self.num_domains):

                if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                    # growth residual and increment
                    a, b = self.ma[n].res_dtheta_growth(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres, 'res_del')
                    self.r_growth.append(a), self.del_theta.append(b)
                else:
                    self.r_growth.append(ufl.as_ufl(0)), self.del_theta.append(ufl.as_ufl(0))

            self.localdata['var'].append([self.theta])
            self.localdata['res'].append([self.r_growth])
            self.localdata['inc'].append([self.del_theta])
            self.localdata['fnc'].append([self.Vd_scalar])

        if self.have_plasticity:

            for n in range(self.num_domains):

                if self.mat_plastic[n]: raise ValueError("Finite strain plasticity not yet implemented!")

        ### Jacobians

        # kinetic virtual work linearization (deltaW_kin already has contributions from all domains)
        # since this is actually linear in the acceleration (and hence the displacement), 'trapezoidal' and 'midpoint' yield the same
        if self.ti.eval_nonlin_terms=='trapezoidal':
            self.weakform_lin_uu = self.timefac_m * ufl.derivative(self.deltaW_kin, self.u, self.du)
        if self.ti.eval_nonlin_terms=='midpoint':
            self.weakform_lin_uu = ufl.derivative(self.deltaW_kin_mid, self.u, self.du)

        # internal virtual work linearization treated differently: since we want to be able to account for nonlinear materials at Gauss
        # point level with deformation-dependent internal variables (i.e. growth or plasticity), we make use of a more explicit formulation
        # of the linearization which involves the fourth-order material tangent operator Ctang ("derivative" cannot take care of the
        # dependence of the internal variables on the deformation if this dependence is nonlinear and cannot be expressed analytically)
        for n, M in enumerate(self.domain_ids):

            # elastic and viscous material tangent operator
            if self.ti.eval_nonlin_terms=='trapezoidal': Cmat, Cmat_v = self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars, returnquantity='tangent')
            if self.ti.eval_nonlin_terms=='midpoint': Cmat, Cmat_v = self.ma[n].S(self.us_mid, self.ps_mid, self.vel_mid, ivar=self.internalvars_mid, returnquantity='tangent')

            if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                # growth tangent operator
                Cgrowth = self.ma[n].Cgrowth(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres)
                if self.mat_remodel[n] and self.lin_remod_full:
                    # remodeling tangent operator
                    Cremod = self.ma[n].Cremod(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres)
                    Ctang = Cmat + Cgrowth + Cremod
                else:
                    Ctang = Cmat + Cgrowth
            else:
                Ctang = Cmat

            if self.ti.eval_nonlin_terms=='trapezoidal':
                self.weakform_lin_uu += self.timefac * self.vf.Lin_deltaW_int_du(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.ki.F(self.u), self.ki.Fdot(self.vel), self.u, Ctang, Cmat_v, self.dx(M))
            if self.ti.eval_nonlin_terms=='midpoint':
                self.weakform_lin_uu += self.vf.Lin_deltaW_int_du(self.ma[n].S(self.us_mid, self.ps_mid, self.vel_mid, ivar=self.internalvars_mid), self.ki.F(self.us_mid), self.ki.Fdot(self.vel_mid), self.u, Ctang, Cmat_v, self.dx(M))

        # external virtual work contribution to stiffness (from nonlinear follower loads or Robin boundary tractions)
        # since external tractions might be nonlinear w.r.t. displacement, there's a difference between 'trapezoidal' and 'midpoint'
        if self.ti.eval_nonlin_terms=='trapezoidal':
            self.weakform_lin_uu += -self.timefac * ufl.derivative(self.deltaW_ext, self.u, self.du)
        if self.ti.eval_nonlin_terms=='midpoint':
            self.weakform_lin_uu += -ufl.derivative(self.deltaW_ext_mid, self.u, self.du)

        # pressure contributions
        if self.incompressible_2field:

            self.weakform_lin_up, self.weakform_lin_pu, self.weakform_lin_pp = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

            for n, M in enumerate(self.domain_ids):
                # this has to be treated like the evaluation of a volumetric material, hence with the elastic part of J
                if self.mat_growth[n]:
                    J    = self.ma[n].J_e(self.u, self.theta)
                    Jmat = self.ma[n].dJedC(self.u, self.theta)
                else:
                    J    = self.ki.J(self.u)
                    Jmat = self.ki.dJdC(self.u)

                if self.ti.eval_nonlin_terms=='trapezoidal': Cmat_p = ufl.diff(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.p)
                if self.ti.eval_nonlin_terms=='midpoint': Cmat_p = ufl.diff(self.ma[n].S(self.us_mid, self.ps_mid, self.vel_mid, ivar=self.internalvars_mid), self.p)

                if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                    # elastic and viscous material tangent operator
                    Cmat, Cmat_v = self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars, returnquantity='tangent')
                    # growth tangent operators - keep in mind that we have theta = theta(C(u),p) in general!
                    # for stress-mediated growth, we get a contribution to the pressure material tangent operator
                    Cgrowth_p = self.ma[n].Cgrowth_p(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres)
                    if self.mat_remodel[n] and self.lin_remod_full:
                        # remodeling tangent operator
                        Cremod_p = self.ma[n].Cremod_p(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres)
                        Ctang_p = Cmat_p + Cgrowth_p + Cremod_p
                    else:
                        Ctang_p = Cmat_p + Cgrowth_p
                    # for all types of deformation-dependent growth, we need to add the growth contributions to the Jacobian tangent operator
                    Jgrowth = ufl.diff(J,self.theta) * self.ma[n].dtheta_dC(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres)
                    Jtang = Jmat + Jgrowth
                    # ok... for stress-mediated growth, we actually get a non-zero right-bottom (11) block in our saddle-point system matrix,
                    # since Je = Je(C,theta(C,p)) ---> dJe/dp = dJe/dtheta * dtheta/dp
                    # TeX: D_{\Delta p}\!\int\limits_{\Omega_0} (J^{\mathrm{e}}-1)\delta p\,\mathrm{d}V = \int\limits_{\Omega_0} \frac{\partial J^{\mathrm{e}}}{\partial p}\Delta p \,\delta p\,\mathrm{d}V,
                    # with \frac{\partial J^{\mathrm{e}}}{\partial p} = \frac{\partial J^{\mathrm{e}}}{\partial \vartheta}\frac{\partial \vartheta}{\partial p}
                    dthetadp = self.ma[n].dtheta_dp(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.pbase.dt, self.growth_thres)
                    if not isinstance(dthetadp, ufl.constantvalue.Zero):
                        self.weakform_lin_pp += ufl.diff(J,self.theta) * dthetadp * self.dp * self.var_p * self.dx(M)
                else:
                    Ctang_p = Cmat_p
                    Jtang = Jmat

                if self.ti.eval_nonlin_terms=='trapezoidal': self.weakform_lin_up += self.timefac * self.vf.Lin_deltaW_int_dp(self.ki.F(self.u), Ctang_p, self.dx(M))
                if self.ti.eval_nonlin_terms=='midpoint': self.weakform_lin_up += self.vf.Lin_deltaW_int_dp(self.ki.F(self.us_mid), Ctang_p, self.dx(M))

                self.weakform_lin_pu += self.vf.Lin_deltaW_int_pres_du(self.ki.F(self.u), Jtang, self.u, self.dx(M))
                if self.incompressibility=='nearly':
                    self.weakform_lin_pp += self.vf.Lin_deltaW_int_pres_nearly_dp(self.bulkmod, self.dx(M))

        # set forms for active stress
        if any(self.mat_active_stress):
            # take care of Frank-Starling law (fiber stretch-dependent contractility)
            if self.have_frank_starling:

                self.amp_old_, na = [], 0
                for n in range(self.num_domains):

                    if self.mat_active_stress[n] and self.actstress[na].frankstarling:
                        # old stretch state (needed for Frank-Starling law) - a stretch that corresponds to the active model is used
                        if self.activemodel[n]=='active_fiber':
                            if self.mat_growth[n]: lam_fib_old = self.ma[n].fibstretch_e(self.ki.C(self.u_old), self.theta_old, self.fib_func[0])
                            else:                  lam_fib_old = self.ki.fibstretch(self.u_old, self.fib_func[0])
                        elif self.activemodel[n]=='active_crossfiber':
                            if self.mat_growth[n]: lam_fib_old = self.ma[n].crossfibstretch_e(self.ki.C(self.u_old), self.theta_old, self.fib_func[0])
                            else:                  lam_fib_old = self.ki.crossfibstretch(self.u_old, self.fib_func[0])
                        elif self.activemodel[n]=='active_iso':
                            if self.mat_growth[n]: lam_fib_old = self.ma[n].isostretch_e(self.ki.C(self.u_old), self.theta_old)
                            else:                  lam_fib_old = self.ki.isostretch(self.u_old)
                        else:
                            raise ValueError("Unknown active model!")

                        self.amp_old_.append(self.actstress[na].amp(lam_fib_old, self.amp_old))
                        na+=1
                    else:
                        self.amp_old_.append(ufl.as_ufl(0))

            self.tau_a_, na = [], 0
            for n in range(self.num_domains):

                if self.mat_active_stress[n]:
                    if self.mat_active_stress_type[n] == 'ode':
                        # stretch state (needed for Frank-Starling law) - a stretch that corresponds to the active model is used
                        if self.actstress[na].frankstarling:
                            if self.activemodel[n]=='active_fiber':
                                if self.mat_growth[n]: lam_fib = self.ma[n].fibstretch_e(self.ki.C(self.u), self.theta, self.fib_func[0])
                                else:                  lam_fib = self.ki.fibstretch(self.u, self.fib_func[0])
                            elif self.activemodel[n]=='active_crossfiber':
                                if self.mat_growth[n]: lam_fib = self.ma[n].crossfibstretch_e(self.ki.C(self.u), self.theta, self.fib_func[0])
                                else:                  lam_fib = self.ki.crossfibstretch(self.u, self.fib_func[0])
                            elif self.activemodel[n]=='active_iso':
                                if self.mat_growth[n]: lam_fib = self.ma[n].isostretch_e(self.ki.C(self.u), self.theta)
                                else:                  lam_fib = self.ki.isostretch(self.u)
                            else:
                                raise ValueError("Unknown active model!")
                        else:
                            lam_fib = ufl.as_ufl(1)

                        self.tau_a_.append(self.actstress[na].tau_act(self.tau_a_old, self.pbase.dt, lam=lam_fib, amp_old=self.amp_old))
                        na+=1
                    if self.mat_active_stress_type[n] == 'prescribed':
                        self.tau_a_.append(self.act_curve[n]) # act_curve now stores the prescribed active stress
                    if self.mat_active_stress_type[n] == 'prescribed_from_file':
                        pass
                else:
                    self.tau_a_.append(ufl.as_ufl(0))

        if self.prestress_initial or self.prestress_initial_only:
            # quasi-static weak forms (don't dare to use fancy growth laws or other inelastic stuff during prestressing...)
            self.weakform_prestress_u = self.deltaW_prestr_int - self.deltaW_prestr_ext
            self.weakform_lin_prestress_uu = ufl.derivative(self.weakform_prestress_u, self.u, self.du)
            if self.incompressible_2field:
                self.weakform_prestress_p = self.deltaW_p
                self.weakform_lin_prestress_up = ufl.derivative(self.weakform_prestress_u, self.p, self.dp)
                self.weakform_lin_prestress_pu = ufl.derivative(self.weakform_prestress_p, self.u, self.du)
                self.weakform_lin_prestress_pp = ufl.derivative(self.weakform_prestress_p, self.p, self.dp)

        # number of fields involved
        if self.incompressible_2field: self.nfields=2
        else: self.nfields=1

        # residual and matrix lists
        self.r_list, self.r_list_rom = [None]*self.nfields, [None]*self.nfields
        self.K_list, self.K_list_rom = [[None]*self.nfields for _ in range(self.nfields)],  [[None]*self.nfields for _ in range(self.nfields)]


    # active stress projection
    def evaluate_active_stress(self):

        if self.have_frank_starling:
            amp_old_proj = project(self.amp_old_, self.Vd_scalar, self.dx, domids=self.domain_ids, comm=self.pbase.comm, entity_maps=self.io.entity_maps)
            self.amp_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.amp_old.interpolate(amp_old_proj)

        # project and interpolate to quadrature function space
        tau_a_proj = project(self.tau_a_, self.Vd_scalar, self.dx, domids=self.domain_ids, comm=self.pbase.comm, entity_maps=self.io.entity_maps)
        self.tau_a.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.tau_a.interpolate(tau_a_proj)

        #mathutils.quad_interpolation(tau_a_[0], self.Vq_scalar, self.io.mesh, self.quadrature_points, self.tau_a)
        #sys.exit()


    # computes and prints the growth rate of the whole solid
    def compute_solid_growth_rate(self, N, t):

        dtheta_all = ufl.as_ufl(0)
        for n, M in enumerate(self.domain_ids):
            dtheta_all += (self.theta - self.theta_old) / (self.pbase.dt) * self.dx(M)

        gr = fem.assemble_scalar(fem.form(dtheta_all))
        gr = self.comm.allgather(gr)
        self.growth_rate = sum(gr)

        utilities.print_status('Solid growth rate: %.4e' % (self.growth_rate), self.pbase.comm)

        if self.comm.rank == 0:
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.pbase.dt): mode='wt'
                else: mode='a'
                fl = self.io.output_path+'/results_'+self.pbase.simname+'_growthrate.txt'
                f = open(fl, mode)
                f.write('%.16E %.16E\n' % (t,self.growth_rate))
                f.close()


    # computes the solid's total strain energy and internal power
    def compute_strain_energy_power(self, N, t):

        se_all, ip_all = ufl.as_ufl(0), ufl.as_ufl(0)
        for n, M in enumerate(self.domain_ids):
            se_all += self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars, returnquantity='strainenergy') * self.dx(M)
            ip_all += ufl.inner(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars),self.ki.Edot(self.u, self.vel)) * self.dx(M)

        se = fem.assemble_scalar(fem.form(se_all))
        se = self.pbase.comm.allgather(se)
        strain_energy = sum(se)

        ip = fem.assemble_scalar(fem.form(ip_all))
        ip = self.pbase.comm.allgather(ip)
        internal_power = sum(ip)

        if self.pbase.comm.rank == 0:
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.pbase.dt): mode='wt'
                else: mode='a'
                if 'strainenergy' in self.results_to_write:
                    fe = open(self.io.output_path+'/results_'+self.pbase.simname+'_strainenergy.txt', mode)
                    fe.write('%.16E %.16E\n' % (t,strain_energy))
                    fe.close()
                if 'internalpower' in self.results_to_write:
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
        se_mem = self.pbase.comm.allgather(se_mem)
        strain_energy_mem = sum(se_mem)

        ip_mem = fem.assemble_scalar(fem.form(ip_mem_all))
        ip_mem = self.pbase.comm.allgather(ip_mem)
        internal_power_mem = sum(ip_mem)

        if self.pbase.comm.rank == 0:
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
        if any(self.mat_active_stress) and 'prescribed_from_file' not in self.mat_active_stress_type:
            self.evaluate_active_stress()


    # compute volumes of a surface from a Laplace problem
    def solve_volume_laplace(self, N, t):

        # Define variational problem
        uf = ufl.TrialFunction(self.V_u)
        vf = ufl.TestFunction(self.V_u)

        f = fem.Function(self.V_u) # zero source term

        a, L = ufl.as_ufl(0), ufl.as_ufl(0)
        for n, M in enumerate(self.domain_ids):
            a += ufl.inner(ufl.grad(uf), ufl.grad(vf))*self.dx(M)
            L += ufl.dot(f,vf)*self.dx(M)

        uf = fem.Function(self.V_u, name="uf")

        dbcs_laplace=[]
        dbcs_laplace.append( fem.dirichletbc(self.u, fem.locate_dofs_topological(self.V_u, 2, self.io.mt_b1.indices[np.isin(self.io.mt_b1.values, self.volume_laplace)])) )

        # solve linear Laplace problem
        lp = fem.petsc.LinearProblem(a, L, bcs=dbcs_laplace, u=uf)
        lp.solve()

        vol_all = ufl.as_ufl(0)
        for n, M in enumerate(self.domain_ids):
            vol_all += ufl.det(ufl.Identity(len(uf)) + ufl.grad(uf)) * self.dx(M)

        vol = fem.assemble_scalar(fem.form(vol_all))
        vol = self.pbase.comm.allgather(vol)
        volume = sum(vol)

        if self.pbase.comm.rank == 0:
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.pbase.dt): mode='wt'
                else: mode='a'
                fl = self.io.output_path+'/results_'+self.pbase.simname+'_volume_laplace.txt'
                f = open(fl, mode)
                f.write('%.16E %.16E\n' % (t,volume))
                f.close()


    def set_problem_residual_jacobian_forms(self, pre=False):

        ts = time.time()
        utilities.print_status("FEM form compilation for solid...", self.pbase.comm, e=" ")

        if not pre:
            self.res_u  = fem.form(self.weakform_u, entity_maps=self.io.entity_maps)
            self.jac_uu = fem.form(self.weakform_lin_uu, entity_maps=self.io.entity_maps)
            if self.incompressible_2field:
                self.res_p  = fem.form(self.weakform_p, entity_maps=self.io.entity_maps)
                self.jac_up = fem.form(self.weakform_lin_up, entity_maps=self.io.entity_maps)
                self.jac_pu = fem.form(self.weakform_lin_pu, entity_maps=self.io.entity_maps)
                if not isinstance(self.weakform_lin_pp, ufl.constantvalue.Zero):
                    self.jac_pp = fem.form(self.weakform_lin_pp, entity_maps=self.io.entity_maps)
                else:
                    self.jac_pp = None
        else:
            self.res_u  = fem.form(self.weakform_prestress_u, entity_maps=self.io.entity_maps)
            self.jac_uu = fem.form(self.weakform_lin_prestress_uu, entity_maps=self.io.entity_maps)
            if self.incompressible_2field:
                self.res_p  = fem.form(self.weakform_prestress_p, entity_maps=self.io.entity_maps)
                self.jac_up = fem.form(self.weakform_lin_prestress_up, entity_maps=self.io.entity_maps)
                self.jac_pu = fem.form(self.weakform_lin_prestress_pu, entity_maps=self.io.entity_maps)
                self.jac_pp = fem.form(self.weakform_lin_prestress_pp, entity_maps=self.io.entity_maps)

        te = time.time() - ts
        utilities.print_status("t = %.4f s" % (te), self.pbase.comm)


    def set_problem_vector_matrix_structures(self):

        self.r_u = fem.petsc.create_vector(self.res_u)
        self.K_uu = fem.petsc.create_matrix(self.jac_uu)

        if self.incompressible_2field:
            self.r_p = fem.petsc.create_vector(self.res_p)

            self.K_up = fem.petsc.create_matrix(self.jac_up)
            self.K_pu = fem.petsc.create_matrix(self.jac_pu)

            if self.jac_pp is not None:
                self.K_pp = fem.petsc.create_matrix(self.jac_pp)
            else:
                self.K_pp = None


    def assemble_residual(self, t, subsolver=None):

        # assemble rhs vector
        with self.r_u.localForm() as r_local: r_local.set(0.0)
        fem.petsc.assemble_vector(self.r_u, self.res_u)
        fem.apply_lifting(self.r_u, [self.jac_uu], [self.bc.dbcs], x0=[self.u.x.petsc_vec], alpha=-1.0)
        self.r_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.r_u, self.bc.dbcs, x0=self.u.x.petsc_vec, alpha=-1.0)

        if self.incompressible_2field:

            # assemble pressure rhs vector
            with self.r_p.localForm() as r_local: r_local.set(0.0)
            fem.petsc.assemble_vector(self.r_p, self.res_p)
            self.r_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            self.r_list[0] = self.r_u
            self.r_list[1] = self.r_p

        else:

            self.r_list[0] = self.r_u


    def assemble_stiffness(self, t, subsolver=None):

        # assemble system matrix
        self.K_uu.zeroEntries()
        fem.petsc.assemble_matrix(self.K_uu, self.jac_uu, self.bc.dbcs)
        self.K_uu.assemble()

        if self.incompressible_2field:

            # assemble system matrices
            self.K_up.zeroEntries()
            fem.petsc.assemble_matrix(self.K_up, self.jac_up, self.bc.dbcs)
            self.K_up.assemble()

            self.K_pu.zeroEntries()
            fem.petsc.assemble_matrix(self.K_pu, self.jac_pu, []) # currently, we do not consider pressure DBCs
            self.K_pu.assemble()

            # for stress-mediated volumetric growth, K_pp is not zero!
            if self.jac_pp is not None:
                self.K_pp.zeroEntries()
                fem.petsc.assemble_matrix(self.K_pp, self.jac_pp, [])
                self.K_pp.assemble()
            else:
                self.K_pp = None

            self.K_list[0][0] = self.K_uu
            self.K_list[0][1] = self.K_up
            self.K_list[1][0] = self.K_pu
            self.K_list[1][1] = self.K_pp

        else:

            self.K_list[0][0] = self.K_uu


    def get_index_sets(self, isoptions={}):

        assert(self.incompressible_2field) # index sets only needed for 2-field problem

        if self.rom is not None: # currently, ROM can only be on (subset of) first variable
            uvec_or0 = self.rom.V.getOwnershipRangeColumn()[0]
            uvec_ls = self.rom.V.getLocalSize()[1]
        else:
            uvec_or0 = self.u.x.petsc_vec.getOwnershipRange()[0]
            uvec_ls = self.u.x.petsc_vec.getLocalSize()

        offset_u = uvec_or0 + self.p.x.petsc_vec.getOwnershipRange()[0]
        iset_u = PETSc.IS().createStride(uvec_ls, first=offset_u, step=1, comm=self.pbase.comm)

        offset_p = offset_u + uvec_ls
        iset_p = PETSc.IS().createStride(self.p.x.petsc_vec.getLocalSize(), first=offset_p, step=1, comm=self.pbase.comm)

        return [iset_u, iset_p]


    ### now the base routines for this problem

    def read_restart(self, sname, N):

        # read restart information
        if N > 0:
            self.io.readcheckpoint(self, N)


    def evaluate_initial(self):
        pass


    def write_output_ini(self):

        self.io.write_output(self, writemesh=True)


    def write_output_pre(self):

        if 'fibers' in self.results_to_write and self.io.write_results_every > 0:
            for i in range(len(self.fibarray)):
                fib_proj = project(self.fib_func[i], self.V_u, self.dx, domids=self.domain_ids, nm='Fiber'+str(i+1), comm=self.pbase.comm, entity_maps=self.io.entity_maps)
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

        if 'prescribed_from_file' in (self.mat_active_stress_type):
            self.io.readfunction(self.tau_a, self.constitutive_models['MAT'+str(self.actpid)][self.activemodel[self.actpid-1]]['prescribed_file'].replace('*',str(N)))


    def evaluate_post_solve(self, t, N):

        # solve volume laplace (for cardiac benchmark)
        if bool(self.volume_laplace): self.solve_volume_laplace(N, t)

        # compute the growth rate (has to be called before update_timestep)
        if self.have_growth:
            self.compute_solid_growth_rate(N, t)
        if 'strainenergy' in self.results_to_write or 'internalpower' in self.results_to_write:
            self.compute_strain_energy_power(N, t)
        if 'membrane' in self.bc_dict.keys() and ('strainenergy_membrane' in self.results_to_write or 'internalpower_membrane' in self.results_to_write):
            self.compute_strain_energy_power_membrane(N, t)


    def set_output_state(self, t):
        pass


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t)


    def update(self):

        # update - displacement, velocity, acceleration, pressure, all internal variables, all time functions
        self.ti.update_timestep(self.u, self.u_old, self.v, self.v_old, self.a, self.a_old, self.p, self.p_old, self.internalvars, self.internalvars_old)


    def print_to_screen(self):
        pass


    def induce_state_change(self):
        pass


    def write_restart(self, sname, N, force=False):

        self.io.write_restart(self, N, force=force)


    def check_abort(self, t):

        if self.pbase.problem_type == 'solid_flow0d_multiscale_gandr' and abs(self.growth_rate) <= self.tol_stop_large:
            return True


    def destroy(self):

        self.io.close_output_files(self)



class SolidmechanicsSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms(pre=self.pb.pre)
        self.pb.set_problem_vector_matrix_structures()

        self.evaluate_assemble_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the full solid problem
        if self.pb.pre:
            self.solve_initial_prestress()

        # consider consistent initial acceleration
        if self.pb.timint != 'static' and self.pb.pbase.restart_step == 0:

            ts = time.time()
            utilities.print_status("Setting forms and solving for consistent initial acceleration...", self.pb.pbase.comm, e=" ")

            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.a_old, self.pb.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            res_a, jac_aa  = fem.form(weakform_a), fem.form(weakform_lin_aa)
            self.solnln.solve_consistent_ini_acc(res_a, jac_aa, self.pb.a_old)

            te = time.time() - ts
            utilities.print_status("t = %.4f s" % (te), self.pb.pbase.comm)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t, localdata=self.pb.localdata)


    def print_timestep_info(self, N, t, ni, li, wt):

        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, ni=ni, li=li, wt=wt)


    def solve_initial_prestress(self):

        utilities.print_prestress('start', self.pb.pbase.comm)

        if self.pb.prestress_ptc: self.solnln.PTC = True

        for N in range(1,self.pb.prestress_numstep+1):

            wts = time.time()

            tprestr = N * self.pb.prestress_dt

            self.pb.ti.set_time_funcs_pre(tprestr)

            self.solnln.newton(tprestr)

            # MULF update
            self.pb.ki.prestress_update(self.pb.u)
            utilities.print_prestress('updt', self.pb.pbase.comm)

            wt = time.time() - wts

            # print time step info to screen
            self.pb.ti.print_prestress_step(N, tprestr, self.pb.prestress_numstep, self.solnln.lsp, ni=self.solnln.ni, li=self.solnln.li, wt=wt)

        utilities.print_prestress('end', self.pb.pbase.comm)

        # write prestress displacement (given that we want to write the displacement)
        if 'displacement' in self.pb.results_to_write and self.pb.io.write_results_every > 0:
            self.pb.io.write_output_pre(self.pb, self.pb.u_pre, 0, 'displacement_pre')

        if self.pb.prestress_initial_only:
            # it may be convenient to write the prestress displacement field to a file for later read-in
            self.pb.io.writefunction(self.pb.u_pre, self.pb.io.output_path_pre+'/results_'+self.pb.pbase.simname+'_displacement_pre')
            if self.pb.incompressible_2field:
                self.pb.io.writefunction(self.pb.p, self.pb.io.output_path_pre+'/results_'+self.pb.pbase.simname+'_pressure_pre')
            utilities.print_status("Prestress only done. To resume, set file path(s) in 'prestress_from_file' and read in u_pre.", self.pb.pbase.comm)
            os._exit(0)

        # reset PTC flag to what it was
        if self.pb.prestress_ptc:
            self.solnln.PTC = self.solver_params.get('ptc', False)

        # now build main (non-prestress) forms
        self.pb.set_problem_residual_jacobian_forms()


# prestress solver, to be called from other (coupled) problems
class SolidmechanicsSolverPrestr(SolidmechanicsSolver):

    def __init__(self, problem, solver_params):

        self.pb = problem
        self.solver_params = solver_params

        self.initialize_nonlinear_solver()


    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear([self.pb], self.solver_params)


    def solve_initial_state(self):
        raise RuntimeError("You should not be here!")
