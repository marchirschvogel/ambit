#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import ioroutines
import solid_kinematics_constitutive
import solid_variationalform
import timeintegration
import utilities
import solver_nonlin
import boundaryconditions
from projection import project
from solid_material import activestress_activation

from base import problem_base, solver_base


# solid mechanics governing equation

#\rho_{0} \ddot{\boldsymbol{u}} = \boldsymbol{\nabla}_{0} \cdot (\boldsymbol{F}\boldsymbol{S}) + \hat{\boldsymbol{b}}_{0} \quad \text{in} \; \Omega_{0} \times [0, T]

# can be solved together with constraint J = 1 (2-field variational principle with u and p as degrees of freedom)
#J-1 = 0 \quad \text{in} \; \Omega_{0} \times [0, T]


class SolidmechanicsProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params={}, comm=None):
        problem_base.__init__(self, io_params, time_params, comm)
        
        self.problem_physics = 'solid'

        self.simname = io_params['simname']
        self.results_to_write = io_params['results_to_write']

        self.io = io

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        
        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_disp = fem_params['order_disp']
        try: self.order_pres = fem_params['order_pres']
        except: self.order_pres = 1
        self.quad_degree = fem_params['quad_degree']
        self.incompressible_2field = fem_params['incompressible_2field']
        
        # whether to enforce continuity of mass at midpoint or not - only relevant for incompressible_2field option
        try: self.pressure_at_midpoint = fem_params['pressure_at_midpoint']
        except: self.pressure_at_midpoint = False
        
        self.fem_params = fem_params

        # collect domain data
        self.dx_, self.rho0, self.eta_m, self.eta_k = [], [], [], []

        for n in range(self.num_domains):
            # integration domains
            if self.io.mt_d is not None: self.dx_.append(ufl.dx(subdomain_data=self.io.mt_d, subdomain_id=n+1, metadata={'quadrature_degree': self.quad_degree}))
            else:                        self.dx_.append(ufl.dx(metadata={'quadrature_degree': self.quad_degree}))
            # data for inertial forces: density
            if self.timint != 'static':
                self.rho0.append(self.constitutive_models['MAT'+str(n+1)]['inertia']['rho0'])

        try: self.prestress_initial = fem_params['prestress_initial']
        except: self.prestress_initial = False

        if self.prestress_initial:
            self.constitutive_models_prestr = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.dim = self.io.mesh.geometry.dim
        
        self.sub_solve = False

        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            dg_type = "DG"
            if (self.order_disp > 1 or self.order_pres > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            dg_type = "DQ"
            if (self.order_disp > 1 or self.order_pres > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
            if self.quad_degree < 2:
                raise ValueError("Use at least a quadrature degree >= 2 for a hexahedral mesh!")
        else:
            raise NameError("Unknown cell/element type!")

        self.Vex = self.io.mesh.ufl_domain().ufl_coordinate_element()
        
        # check if we want to use model order reduction and if yes, initialize MOR class
        try: self.have_rom = io_params['use_model_order_red']
        except: self.have_rom = False

        if self.have_rom:
            import mor
            self.rom = mor.ModelOrderReduction(mor_params, self.comm)
        
        # create finite element objects for u and p
        P_u = ufl.VectorElement("CG", self.io.mesh.ufl_cell(), self.order_disp)
        P_p = ufl.FiniteElement("CG", self.io.mesh.ufl_cell(), self.order_pres)
        # function spaces for u and p
        self.V_u = fem.FunctionSpace(self.io.mesh, P_u)
        self.V_p = fem.FunctionSpace(self.io.mesh, P_p)
        # tensor finite element and function space
        P_tensor = ufl.TensorElement("CG", self.io.mesh.ufl_cell(), self.order_disp)
        self.V_tensor = fem.FunctionSpace(self.io.mesh, P_tensor)

        # Quadrature tensor, vector, and scalar elements
        Q_tensor = ufl.TensorElement("Quadrature", self.io.mesh.ufl_cell(), degree=1, quad_scheme="default")
        Q_vector = ufl.VectorElement("Quadrature", self.io.mesh.ufl_cell(), degree=1, quad_scheme="default")
        Q_scalar = ufl.FiniteElement("Quadrature", self.io.mesh.ufl_cell(), degree=1, quad_scheme="default")

        # not yet working - we cannot interpolate into Quadrature elements with the current dolfinx version currently!
        #self.Vd_tensor = fem.FunctionSpace(self.io.mesh, Q_tensor)
        #self.Vd_vector = fem.FunctionSpace(self.io.mesh, Q_vector)
        #self.Vd_scalar = fem.FunctionSpace(self.io.mesh, Q_scalar)

        # Quadrature function spaces (currently not properly functioning for higher-order meshes!!!)
        self.Vd_tensor = fem.TensorFunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))
        self.Vd_vector = fem.VectorFunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))
        self.Vd_scalar = fem.FunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))

        # functions
        self.du    = ufl.TrialFunction(self.V_u)            # Incremental displacement
        self.var_u = ufl.TestFunction(self.V_u)             # Test function
        self.dp    = ufl.TrialFunction(self.V_p)            # Incremental pressure
        self.var_p = ufl.TestFunction(self.V_p)             # Test function
        self.u     = fem.Function(self.V_u, name="Displacement")
        self.p     = fem.Function(self.V_p, name="Pressure")
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
        self.theta.vector.set(1.0), self.theta_old.vector.set(1.0)
        self.theta.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD), self.theta_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # active stress
        self.tau_a = fem.Function(self.Vd_scalar, name="tau_a")
        self.tau_a_old = fem.Function(self.Vd_scalar)
        self.amp_old, self.amp_old_set = fem.Function(self.Vd_scalar), fem.Function(self.Vd_scalar)
        self.amp_old.vector.set(1.0), self.amp_old_set.vector.set(1.0)
        self.amp_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD), self.amp_old_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # prestress displacement
        if self.prestress_initial:
            self.u_pre = fem.Function(self.V_u, name="Displacement_prestress")
        else:
            self.u_pre = None

        try: self.volume_laplace = io_params['volume_laplace']
        except: self.volume_laplace = []

        # dictionaries of internal variables
        self.internalvars, self.internalvars_old = {}, {}
        
        # reference coordinates
        self.x_ref = fem.Function(self.V_u)
        self.x_ref.interpolate(self.x_ref_expr)
        
        if self.incompressible_2field:
            self.numdof = self.u.vector.getSize() + self.p.vector.getSize()
        else:
            self.numdof = self.u.vector.getSize()

        # initialize solid time-integration class
        self.ti = timeintegration.timeintegration_solid(time_params, fem_params, time_curves, self.t_init, self.comm)

        # check for materials that need extra treatment (anisotropic, active stress, growth, ...)
        have_fiber1, have_fiber2 = False, False
        self.have_active_stress, self.active_stress_trig, self.have_frank_starling, self.have_growth, self.have_plasticity = False, 'ode', False, False, False
        self.mat_active_stress, self.mat_growth, self.mat_remodel, self.mat_growth_dir, self.mat_growth_trig, self.mat_growth_thres, self.mat_plastic = [False]*self.num_domains, [False]*self.num_domains, [False]*self.num_domains, [None]*self.num_domains, [None]*self.num_domains, []*self.num_domains, [False]*self.num_domains

        self.localsolve, growth_dir = False, None
        self.actstress = []
        for n in range(self.num_domains):
            
            if 'holzapfelogden_dev' in self.constitutive_models['MAT'+str(n+1)].keys() or 'guccione_dev' in self.constitutive_models['MAT'+str(n+1)].keys():
                have_fiber1, have_fiber2 = True, True
            
            if 'active_fiber' in self.constitutive_models['MAT'+str(n+1)].keys():
                have_fiber1 = True
                self.mat_active_stress[n], self.have_active_stress = True, True
                # if one mat has a prescribed active stress, all have to be!
                if 'prescribed_curve' in self.constitutive_models['MAT'+str(n+1)]['active_fiber']:
                    self.active_stress_trig = 'prescribed'
                if 'prescribed_multiscale' in self.constitutive_models['MAT'+str(n+1)]['active_fiber']:
                    self.active_stress_trig = 'prescribed_multiscale'
                if self.active_stress_trig == 'ode':
                    act_curve = self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)]['active_fiber']['activation_curve'])
                    self.actstress.append(activestress_activation(self.constitutive_models['MAT'+str(n+1)]['active_fiber'], act_curve))
                    if self.actstress[-1].frankstarling: self.have_frank_starling = True
                if self.active_stress_trig == 'prescribed':
                    self.ti.funcs_to_update.append({self.tau_a : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)]['active_fiber']['prescribed_curve'])})
                self.internalvars['tau_a'], self.internalvars_old['tau_a'] = self.tau_a, self.tau_a_old

            if 'active_iso' in self.constitutive_models['MAT'+str(n+1)].keys():
                self.mat_active_stress[n], self.have_active_stress = True, True
                # if one mat has a prescribed active stress, all have to be!
                if 'prescribed_curve' in self.constitutive_models['MAT'+str(n+1)]['active_iso']:
                    self.active_stress_trig = 'prescribed'
                if 'prescribed_multiscale' in self.constitutive_models['MAT'+str(n+1)]['active_iso']:
                    self.active_stress_trig = 'prescribed_multiscale'
                if self.active_stress_trig == 'ode':
                    act_curve = self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)]['active_iso']['activation_curve'])
                    self.actstress.append(activestress_activation(self.constitutive_models['MAT'+str(n+1)]['active_iso'], act_curve))
                if self.active_stress_trig == 'prescribed':
                    self.ti.funcs_to_update.append({self.tau_a : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)]['active_iso']['prescribed_curve'])})
                self.internalvars['tau_a'], self.internalvars_old['tau_a'] = self.tau_a, self.tau_a_old

            if 'growth' in self.constitutive_models['MAT'+str(n+1)].keys():
                self.mat_growth[n], self.have_growth = True, True
                self.mat_growth_dir[n] = self.constitutive_models['MAT'+str(n+1)]['growth']['growth_dir']
                self.mat_growth_trig[n] = self.constitutive_models['MAT'+str(n+1)]['growth']['growth_trig']
                # need to have fiber fields for the following growth options
                if self.mat_growth_dir[n] == 'fiber' or self.mat_growth_trig[n] == 'fibstretch':
                    have_fiber1 = True
                if self.mat_growth_dir[n] == 'radial':
                    have_fiber1, have_fiber2 = True, True
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
                if 'remodeling_mat' in self.constitutive_models['MAT'+str(n+1)]['growth'].keys():
                    self.mat_remodel[n] = True
                self.internalvars['theta'], self.internalvars_old['theta'] = self.theta, self.theta_old
            else:
                self.mat_growth_thres.append(ufl.as_ufl(0))

            if 'plastic' in self.constitutive_models['MAT'+str(n+1)].keys():
                self.mat_plastic[n], self.have_plasticity = True, True
                self.localsolve = True
                self.internalvars['e_plast'], self.internalvars_old['e_plast'] = self.F_plast, self.F_plast_old
                
        # full linearization of our remodeling law can lead to excessive compiler times for FFCx... :-/
        # let's try if we might can go without one of the critial terms (derivative of remodeling fraction w.r.t. C)
        try: self.lin_remod_full = fem_params['lin_remodeling_full']
        except: self.lin_remod_full = True

        # growth threshold (as function, since in multiscale approach, it can vary element-wise)
        if self.have_growth and self.localsolve:
            growth_thres_proj = project(self.mat_growth_thres, self.Vd_scalar, self.dx_)
            self.growth_thres.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.growth_thres.interpolate(growth_thres_proj)

        # read in fiber data
        if have_fiber1:

            fibarray = ['fiber']
            if have_fiber2: fibarray.append('sheet')

            # fiber function space - vector defined on quadrature points
            V_fib = self.Vd_vector # self.V_u
            self.fib_func = self.io.readin_fibers(fibarray, V_fib, self.dx_)

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
        if self.prestress_initial:
            self.ma_prestr = []
            mat_remove = ['visco_green','growth','plastic']
            for n in range(self.num_domains):
                for mr in mat_remove:
                    try:    self.constitutive_models_prestr['MAT'+str(n+1)].pop(mr)
                    except: pass
                self.ma_prestr.append(solid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models_prestr['MAT'+str(n+1)], self.incompressible_2field, mat_growth=False, mat_remodel=False, mat_plastic=False))

        # initialize solid variational form class
        self.vf = solid_variationalform.variationalform(self.var_u, self.du, self.var_p, self.dp, self.io.n0, self.x_ref)
        
        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_solid(bc_dict, self.fem_params, self.io, self.vf, self.ti, ki=self.ki)
        self.bc_dict = bc_dict
        
        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.V_u)

        self.set_variational_forms()


    def get_problem_var_list(self):
        
        if self.incompressible_2field:
            is_ghosted = [True]*2
            return [self.u.vector, self.p.vector], is_ghosted
        else:
            is_ghosted = [True]
            return [self.u.vector], is_ghosted


    # the main function that defines the solid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):

        # set forms for acceleration and velocity
        self.acc, self.vel = self.ti.set_acc_vel(self.u, self.u_old, self.v_old, self.a_old)

        # kinetic, internal, and pressure virtual work
        self.deltaW_kin,  self.deltaW_kin_old  = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_int,  self.deltaW_int_old  = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_p,    self.deltaW_p_old    = ufl.as_ufl(0), ufl.as_ufl(0)
        
        for n in range(self.num_domains):

            if self.timint != 'static':
                # kinetic virtual work
                self.deltaW_kin     += self.vf.deltaW_kin(self.acc, self.rho0[n], self.dx_[n])
                self.deltaW_kin_old += self.vf.deltaW_kin(self.a_old, self.rho0[n], self.dx_[n])

            # internal virtual work
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.ki.F(self.u), self.dx_[n])
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].S(self.u_old, self.p_old, self.v_old, ivar=self.internalvars_old), self.ki.F(self.u_old), self.dx_[n])
        
            # pressure virtual work (for incompressible formulation)
            # this has to be treated like the evaluation of a volumetric material, hence with the elastic part of J
            if self.mat_growth[n]: J, J_old = self.ma[n].J_e(self.u, self.theta), self.ma[n].J_e(self.u_old, self.theta_old)
            else:                  J, J_old = self.ki.J(self.u), self.ki.J(self.u_old)
            self.deltaW_p       += self.vf.deltaW_int_pres(J, self.dx_[n])
            self.deltaW_p_old   += self.vf.deltaW_int_pres(J_old, self.dx_[n])
        
        # external virtual work (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_neumann_old, w_robin, w_robin_old, w_membrane, w_membrane_old = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann     = self.bc.neumann_bcs(self.V_u, self.Vd_scalar, self.u, funcs_to_update=self.ti.funcs_to_update, funcs_to_update_vec=self.ti.funcs_to_update_vec)
            w_neumann_old = self.bc.neumann_bcs(self.V_u, self.Vd_scalar, self.u_old, funcs_to_update=self.ti.funcs_to_update_old, funcs_to_update_vec=self.ti.funcs_to_update_vec_old)
        if 'robin' in self.bc_dict.keys():
            w_robin     = self.bc.robin_bcs(self.u, self.vel, self.u_pre)
            w_robin_old = self.bc.robin_bcs(self.u_old, self.v_old, self.u_pre)
        if 'membrane' in self.bc_dict.keys():
            w_membrane     = self.bc.membranesurf_bcs(self.u, self.vel, self.acc)
            w_membrane_old = self.bc.membranesurf_bcs(self.u_old, self.v_old, self.a_old)
        if 'dirichlet_weak' in self.bc_dict.keys():
            raise RuntimeError("Cannot use weak Dirichlet BCs for nonlinear solid mechanics currently!")

        # for (quasi-static) prestressing, we need to eliminate dashpots and replace true with reference Neumann loads in our external virtual work
        # plus no rate-dependent or inelastic constitutive models
        w_neumann_prestr, w_robin_prestr, self.deltaW_prestr_int = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if self.prestress_initial:
            # internal virtual work
            for n in range(self.num_domains):
                self.deltaW_prestr_int += self.vf.deltaW_int(self.ma_prestr[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.ki.F(self.u), self.dx_[n])
            # boundary conditions
            bc_dict_prestr = copy.deepcopy(self.bc_dict)
            # get rid of dashpots
            if 'robin' in bc_dict_prestr.keys():
                for r in bc_dict_prestr['robin']:
                    if r['type'] == 'dashpot': r['visc'] = 0.
            # replace true Neumann loads by reference ones
            if 'neumann' in bc_dict_prestr.keys():
                for n in bc_dict_prestr['neumann']:
                    if n['dir'] == 'xyz_cur': n['dir'] = 'xyz_ref'
                    if n['dir'] == 'normal_cur': n['dir'] = 'normal_ref'
            bc_prestr = boundaryconditions.boundary_cond_solid(bc_dict_prestr, self.fem_params, self.io, self.vf, self.ti, ki=self.ki)
            if 'neumann' in bc_dict_prestr.keys():
                w_neumann_prestr = bc_prestr.neumann_bcs(self.V_u, self.Vd_scalar, self.u, self.ti.funcs_to_update, self.ti.funcs_to_update_vec)
            if 'robin' in bc_dict_prestr.keys():
                w_robin_prestr = bc_prestr.robin_bcs(self.u, self.vel, self.u_pre)
            self.deltaW_prestr_ext = w_neumann_prestr + w_robin_prestr

        # TODO: Body forces!
        self.deltaW_ext     = w_neumann + w_robin + w_membrane
        self.deltaW_ext_old = w_neumann_old + w_robin_old + w_membrane_old

        self.timefac_m, self.timefac = self.ti.timefactors()

        ### full weakforms 

        # quasi-static weak form: internal minus external virtual work
        if self.timint == 'static':
            
            self.weakform_u = self.deltaW_int - self.deltaW_ext
            
            if self.incompressible_2field:
                self.weakform_p = self.deltaW_p

        # full dynamic weak form: kinetic plus internal minus external virtual work
        else:
            
            self.weakform_u = self.timefac_m * self.deltaW_kin  + (1.-self.timefac_m) * self.deltaW_kin_old + \
                              self.timefac   * self.deltaW_int  + (1.-self.timefac)   * self.deltaW_int_old - \
                              self.timefac   * self.deltaW_ext  - (1.-self.timefac)   * self.deltaW_ext_old
            
            if self.incompressible_2field:
                if self.pressure_at_midpoint:
                    self.weakform_p = self.timefac * self.deltaW_p + (1.-self.timefac) * self.deltaW_p_old
                else:
                    self.weakform_p = self.deltaW_p

        ### local weak forms at Gauss points for inelastic materials
        self.localdata = {}
        self.localdata['var'], self.localdata['res'], self.localdata['inc'], self.localdata['fnc'] = [], [], [], []
        
        if self.have_growth:
            
            self.r_growth, self.del_theta = [], []
        
            for n in range(self.num_domains):
                
                if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                    # growth residual and increment
                    a, b = self.ma[n].res_dtheta_growth(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres, 'res_del')
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
        self.weakform_lin_uu = self.timefac_m * ufl.derivative(self.deltaW_kin, self.u, self.du)
        
        # internal virtual work linearization treated differently: since we want to be able to account for nonlinear materials at Gauss
        # point level with deformation-dependent internal variables (i.e. growth or plasticity), we make use of a more explicit formulation
        # of the linearization which involves the fourth-order material tangent operator Ctang ("derivative" cannot take care of the
        # dependence of the internal variables on the deformation if this dependence is nonlinear and cannot be expressed analytically)
        for n in range(self.num_domains):
            
            # elastic and viscous material tangent operator
            Cmat, Cmat_v = self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars, tang=True)

            if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                # growth tangent operator
                Cgrowth = self.ma[n].Cgrowth(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres)
                if self.mat_remodel[n] and self.lin_remod_full:
                    # remodeling tangent operator
                    Cremod = self.ma[n].Cremod(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres)
                    Ctang = Cmat + Cgrowth + Cremod
                else:
                    Ctang = Cmat + Cgrowth
            else:
                Ctang = Cmat
            
            self.weakform_lin_uu += self.timefac * self.vf.Lin_deltaW_int_du(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.ki.F(self.u), self.ki.Fdot(self.vel), self.u, Ctang, Cmat_v, self.dx_[n])
        
        # external virtual work contribution to stiffness (from nonlinear follower loads or Robin boundary tractions)
        self.weakform_lin_uu += -self.timefac * ufl.derivative(self.deltaW_ext, self.u, self.du)

        # pressure contributions
        if self.incompressible_2field:
            
            self.weakform_lin_up, self.weakform_lin_pu, self.a_p11, self.p11 = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
            
            for n in range(self.num_domains):
                # this has to be treated like the evaluation of a volumetric material, hence with the elastic part of J
                if self.mat_growth[n]:
                    J    = self.ma[n].J_e(self.u, self.theta)
                    Jmat = self.ma[n].dJedC(self.u, self.theta)
                else:
                    J    = self.ki.J(self.u)
                    Jmat = self.ki.dJdC(self.u)
                
                Cmat_p = ufl.diff(self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars), self.p)
                
                if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                    # elastic and viscous material tangent operator
                    Cmat, Cmat_v = self.ma[n].S(self.u, self.p, self.vel, ivar=self.internalvars, tang=True)
                    # growth tangent operators - keep in mind that we have theta = theta(C(u),p) in general!
                    # for stress-mediated growth, we get a contribution to the pressure material tangent operator
                    Cgrowth_p = self.ma[n].Cgrowth_p(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres)
                    if self.mat_remodel[n] and self.lin_remod_full:
                        # remodeling tangent operator
                        Cremod_p = self.ma[n].Cremod_p(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres)
                        Ctang_p = Cmat_p + Cgrowth_p + Cremod_p
                    else:
                        Ctang_p = Cmat_p + Cgrowth_p
                    # for all types of deformation-dependent growth, we need to add the growth contributions to the Jacobian tangent operator
                    Jgrowth = ufl.diff(J,self.theta) * self.ma[n].dtheta_dC(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres)
                    Jtang = Jmat + Jgrowth
                    # ok... for stress-mediated growth, we actually get a non-zero right-bottom (11) block in our saddle-point system matrix,
                    # since Je = Je(C,theta(C,p)) ---> dJe/dp = dJe/dtheta * dtheta/dp
                    # TeX: D_{\Delta p}\!\int\limits_{\Omega_0} (J^{\mathrm{e}}-1)\delta p\,\mathrm{d}V = \int\limits_{\Omega_0} \frac{\partial J^{\mathrm{e}}}{\partial p}\Delta p \,\delta p\,\mathrm{d}V,
                    # with \frac{\partial J^{\mathrm{e}}}{\partial p} = \frac{\partial J^{\mathrm{e}}}{\partial \vartheta}\frac{\partial \vartheta}{\partial p}
                    dthetadp = self.ma[n].dtheta_dp(self.u, self.p, self.vel, self.internalvars, self.theta_old, self.dt, self.growth_thres)
                    if not isinstance(dthetadp, ufl.constantvalue.Zero):
                        self.p11 += ufl.diff(J,self.theta) * dthetadp * self.dp * self.var_p * self.dx_[n]
                else:
                    Ctang_p = Cmat_p
                    Jtang = Jmat
                
                self.weakform_lin_up += self.timefac * self.vf.Lin_deltaW_int_dp(self.ki.F(self.u), Ctang_p, self.dx_[n])
                if self.pressure_at_midpoint:
                    self.weakform_lin_pu += self.timefac * self.vf.Lin_deltaW_int_pres_du(self.ki.F(self.u), Jtang, self.u, self.dx_[n])
                else:
                    self.weakform_lin_pu += self.vf.Lin_deltaW_int_pres_du(self.ki.F(self.u), Jtang, self.u, self.dx_[n])
                
                # for saddle-point block-diagonal preconditioner
                self.a_p11 += ufl.inner(self.dp, self.var_p) * self.dx_[n]

        if self.prestress_initial:
            # quasi-static weak forms (don't dare to use fancy growth laws or other inelastic stuff during prestressing...)
            self.weakform_prestress_u = self.deltaW_prestr_int - self.deltaW_prestr_ext
            self.weakform_lin_prestress_uu = ufl.derivative(self.weakform_prestress_u, self.u, self.du)
            if self.incompressible_2field:
                self.weakform_prestress_p = self.deltaW_p
                self.weakform_lin_prestress_up = ufl.derivative(self.weakform_prestress_u, self.p, self.dp)
                self.weakform_lin_prestress_pu = ufl.derivative(self.weakform_prestress_p, self.u, self.du)

        
    # reference coordinates
    def x_ref_expr(self, x):
        if self.dim==3: return np.stack((x[0],x[1],x[2]))
        if self.dim==2: return np.stack((x[0],x[1]))


    # active stress ODE evaluation
    def evaluate_active_stress_ode(self, t):
    
        # take care of Frank-Starling law (fiber stretch-dependent contractility)
        if self.have_frank_starling:
            
            amp_old_, na = [], 0
            for n in range(self.num_domains):

                if self.mat_active_stress[n] and self.actstress[na].frankstarling:

                    # old fiber stretch (needed for Frank-Starling law)
                    if self.mat_growth[n]: lam_fib_old = self.ma[n].fibstretch_e(self.ki.C(self.u_old), self.theta_old, self.fib_func[0])
                    else:                  lam_fib_old = self.ki.fibstretch(self.u_old, self.fib_func[0])
                    
                    amp_old_.append(self.actstress[na].amp(t-self.dt, lam_fib_old, self.amp_old))

                else:
                    
                    amp_old_.append(ufl.as_ufl(0))

            amp_old_proj = project(amp_old_, self.Vd_scalar, self.dx_)
            self.amp_old.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.amp_old.interpolate(amp_old_proj)
        
        tau_a_, na = [], 0
        for n in range(self.num_domains):

            if self.mat_active_stress[n]:

                # fiber stretch (needed for Frank-Starling law)
                if self.actstress[na].frankstarling:
                    if self.mat_growth[n]: lam_fib = self.ma[n].fibstretch_e(self.ki.C(self.u), self.theta, self.fib_func[0])
                    else:                  lam_fib = self.ki.fibstretch(self.u, self.fib_func[0])
                else:
                    lam_fib = ufl.as_ufl(1)
                
                tau_a_.append(self.actstress[na].tau_act(self.tau_a_old, t, self.dt, lam_fib, self.amp_old))
                
                na+=1
                
            else:
                
                tau_a_.append(ufl.as_ufl(0))
                
        # project and interpolate to quadrature function space
        tau_a_proj = project(tau_a_, self.Vd_scalar, self.dx_)
        self.tau_a.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.tau_a.interpolate(tau_a_proj)


    # computes and prints the growth rate of the whole solid
    def compute_solid_growth_rate(self, N, t):
        
        dtheta_all = ufl.as_ufl(0)
        for n in range(self.num_domains):
            dtheta_all += (self.theta - self.theta_old) / (self.dt) * self.dx_[n]

        gr = fem.assemble_scalar(fem.form(dtheta_all))
        gr = self.comm.allgather(gr)
        self.growth_rate = sum(gr)

        if self.comm.rank == 0:
            print('Solid growth rate: %.4e' % (self.growth_rate))
            sys.stdout.flush()
            
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.dt): mode='wt'
                else: mode='a'
                fl = self.io.output_path+'/results_'+self.simname+'_growthrate.txt'
                f = open(fl, mode)
                f.write('%.16E %.16E\n' % (t,self.growth_rate))
                f.close()


    # rate equations
    def evaluate_rate_equations(self, t_abs, t_off=0):

        # take care of active stress
        if self.have_active_stress and self.active_stress_trig == 'ode':
            self.evaluate_active_stress_ode(t_abs-t_off)


    # compute volumes of a surface from a Laplace problem
    def solve_volume_laplace(self, N, t):

        # Define variational problem
        uf = TrialFunction(self.V_u)
        vf = TestFunction(self.V_u)
        
        f = Function(self.V_u) # zero source term
        
        a, L = ufl.as_ufl(0), ufl.as_ufl(0)
        for n in range(self.num_domains):
            a += ufl.inner(ufl.grad(uf), ufl.grad(vf))*self.dx_[n]
            L += ufl.dot(f,vf)*self.dx_[n]

        uf = Function(self.V_u, name="uf")
        
        dbcs_laplace=[]
        dbcs_laplace.append( DirichletBC(self.u, locate_dofs_topological(self.V_u, 2, self.io.mt_b1.indices[self.io.mt_b1.values == self.volume_laplace[0]])) )

        # solve linear Laplace problem
        lp = LinearProblem(a, L, bcs=dbcs_laplace, u=uf)
        lp.solve()

        vol_all = ufl.as_ufl(0)
        for n in range(self.num_domains):
            vol_all += ufl.det(ufl.Identity(len(uf)) + ufl.grad(uf)) * self.dx_[n]

        vol = fem.assemble_scalar(fem.form(vol_all))
        vol = self.comm.allgather(vol)
        volume = sum(vol)
        
        if self.comm.rank == 0:
            if self.io.write_results_every > 0 and N % self.io.write_results_every == 0:
                if np.isclose(t,self.dt): mode='wt'
                else: mode='a'
                fl = self.io.output_path+'/results_'+self.simname+'_volume_laplace.txt'
                f = open(fl, mode)
                f.write('%.16E %.16E\n' % (t,volume))
                f.close()


    def set_problem_residual_jacobian_forms(self):

        self.res_u = fem.form(self.weakform_u)
        self.jac_uu = fem.form(self.weakform_lin_uu)
        
        if self.incompressible_2field:
            self.res_p = fem.form(self.weakform_p)
            self.jac_up = fem.form(self.weakform_lin_up)
            self.jac_pu = fem.form(self.weakform_lin_pu)


    def set_forms_solver(self):

        if not self.prestress_initial:
            self.res_u_sol  = fem.form(self.weakform_u)
            self.jac_uu_sol = fem.form(self.jac_uu)
            if self.incompressible_2field:
                self.res_p_sol  = fem.form(self.weakform_p)
                self.jac_up_sol = fem.form(self.weakform_lin_up)
                self.jac_pu_sol = fem.form(self.weakform_lin_pu)
        else:
            self.res_u_sol  = fem.form(self.weakform_prestress_u)
            self.jac_uu_sol = fem.form(self.weakform_lin_prestress_uu)
            if self.incompressible_2field:
                self.res_p_sol  = fem.form(self.weakform_prestress_p)
                self.jac_up_sol = fem.form(self.weakform_lin_prestress_up)
                self.jac_pu_sol = fem.form(self.weakform_lin_prestress_pu)


    def assemble_residual_stiffness(self, t, subsolver=None):

        # assemble rhs vector
        r_u = fem.petsc.assemble_vector(self.res_u_sol)
        fem.apply_lifting(r_u, [self.jac_uu_sol], [self.bc.dbcs], x0=[self.u.vector], scale=-1.0)
        r_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(r_u, self.bc.dbcs, x0=self.u.vector, scale=-1.0)

        # assemble system matrix
        K_uu = fem.petsc.assemble_matrix(self.jac_uu_sol, self.bc.dbcs)
        K_uu.assemble()
        
        if self.incompressible_2field:
        
            # assemble pressure rhs vector
            r_p = fem.petsc.assemble_vector(self.res_p_sol)
            r_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # assemble system matrices
            K_up = fem.petsc.assemble_matrix(self.jac_up_sol, self.bc.dbcs)
            K_up.assemble()
            K_pu = fem.petsc.assemble_matrix(self.jac_pu_sol, []) # currently, we do not consider pressure DBCs
            K_pu.assemble()
            
            # for stress-mediated volumetric growth, K_pp is not zero!
            if not isinstance(self.p11, ufl.constantvalue.Zero):
                K_pp = fem.petsc.assemble_matrix(fem.form(self.p11), [])
                K_pp.assemble()
            else:
                K_pp = None
            
            return [r_u, r_p], [[K_uu, K_up], [K_pu, K_pp]]
        
        else:
            
            return [r_u], [[K_uu]]


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):

        # perform Proper Orthogonal Decomposition
        if self.have_rom:
            self.rom.POD(self, self.V_u)

                
    def read_restart(self, sname, N):

        # read restart information
        if self.restart_step > 0:
            self.io.readcheckpoint(self, N)
            self.simname += '_r'+str(N)

    
    def evaluate_initial(self):
        pass


    def write_output_ini(self):
        
        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):
        return 0.


    def evaluate_pre_solve(self, t):

        # set time-dependent functions
        self.ti.set_time_funcs(self.ti.funcs_to_update, self.ti.funcs_to_update_vec, t)
            
        # evaluate rate equations
        self.evaluate_rate_equations(t)
            
            
    def evaluate_post_solve(self, t, N):
        
        # solve volume laplace (for cardiac benchmark)
        if bool(self.volume_laplace): self.solve_volume_laplace(N, t)
        
        # compute the growth rate (has to be called before update_timestep)
        if self.have_growth:
            self.compute_solid_growth_rate(N, t)


    def set_output_state(self):
        pass

            
    def write_output(self, N, t, mesh=False): 

        self.io.write_output(self, N=N, t=t)

            
    def update(self):
        
        # update - displacement, velocity, acceleration, pressure, all internal variables, all time functions
        self.ti.update_timestep(self.u, self.u_old, self.v_old, self.a_old, self.p, self.p_old, self.internalvars, self.internalvars_old, self.ti.funcs_to_update, self.ti.funcs_to_update_old, self.ti.funcs_to_update_vec, self.ti.funcs_to_update_vec_old)


    def print_to_screen(self):
        pass
    
    
    def induce_state_change(self):
        pass


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)
        
        
    def check_abort(self, t):
            
        if self.problem_type == 'solid_flow0d_multiscale_gandr' and abs(self.growth_rate) <= self.tol_stop_large:
            return True



class SolidmechanicsSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)


    def solve_initial_state(self):

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the full solid problem
        if self.pb.prestress_initial and self.pb.restart_step == 0:
            self.solve_initial_prestress()
        else:
            # set flag definitely to False if we're restarting
            self.pb.prestress_initial = False
            self.pb.set_forms_solver()

        # consider consistent initial acceleration
        if self.pb.timint != 'static' and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old
            
            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.a_old, self.pb.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, weakform_lin_aa, self.pb.a_old)


    def solve_nonlinear_problem(self, t):
        
        self.solnln.newton(t, localdata=self.pb.localdata)


    def print_timestep_info(self, N, t, wt):
    
        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)


    def solve_initial_prestress(self):
        
        utilities.print_prestress('start', self.pb.comm)

        # solve in 1 load step using PTC!
        self.solnln.PTC = True

        self.solnln.newton(0.0)

        # MULF update
        self.pb.ki.prestress_update(self.pb.u)

        # set flag to false again
        self.pb.prestress_initial = False
        self.pb.set_forms_solver()

        # reset PTC flag to what it was
        try: self.solnln.PTC = self.solver_params['ptc']
        except: self.solnln.PTC = False

        utilities.print_prestress('end', self.pb.comm)

        # write prestress displacement (given that we want to write the displacement)
        if 'displacement' in self.pb.results_to_write:
            self.pb.io.write_output_pre(self.pb, self.pb.u_pre, 'displacement_pre')
