#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys, copy
import numpy as np
from dolfinx import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, Function, DirichletBC
from dolfinx.fem import assemble_scalar
from ufl import TrialFunction, TestFunction, FiniteElement, VectorElement, TensorElement, derivative, diff, det, inner, inv, dx, ds, as_vector, as_ufl, Identity, constantvalue
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

from base import problem_base


# solid mechanics governing equation

#\rho_{0} \ddot{\boldsymbol{u}} = \boldsymbol{\nabla}_{0} \cdot (\boldsymbol{F}\boldsymbol{S}) + \hat{\boldsymbol{b}}_{0} \quad \text{in} \; \Omega_{0} \times [0, T]

# can be solved together with constraint J = 1 (2-field variational principle with u and p as degrees of freedom)
#J-1 = 0 \quad \text{in} \; \Omega_{0} \times [0, T]


class SolidmechanicsProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params={}, comm=None):
        problem_base.__init__(self, io_params, time_params, comm)

        self.problem_physics = 'solid'

        self.simname = io_params['simname']

        self.io = io
        
        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)

        self.order_disp = fem_params['order_disp']
        try: self.order_pres = fem_params['order_pres']
        except: self.order_pres = 1
        self.quad_degree = fem_params['quad_degree']
        self.incompressible_2field = fem_params['incompressible_2field']
        
        self.fem_params = fem_params
        self.constitutive_models = constitutive_models

        # collect domain data
        self.dx_, self.rho0, self.rayleigh, self.eta_m, self.eta_k = [], [], [False]*self.num_domains, [], []
        for n in range(self.num_domains):
            # integration domains
            self.dx_.append(dx(subdomain_data=self.io.mt_d, subdomain_id=n+1, metadata={'quadrature_degree': self.quad_degree}))
            # data for inertial and viscous forces: density and damping
            if self.timint != 'static':
                self.rho0.append(constitutive_models['MAT'+str(n+1)+'']['inertia']['rho0'])
                if 'rayleigh_damping' in constitutive_models['MAT'+str(n+1)+''].keys():
                    self.rayleigh[n] = True
                    self.eta_m.append(constitutive_models['MAT'+str(n+1)+'']['rayleigh_damping']['eta_m'])
                    self.eta_k.append(constitutive_models['MAT'+str(n+1)+'']['rayleigh_damping']['eta_k'])

        try: self.prestress_initial = fem_params['prestress_initial']
        except: self.prestress_initial = False

        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            dg_type = "DG"
            if (self.order_disp > 1 or self.order_pres > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            dg_type = "DQ"
            if (self.order_disp > 1 or self.order_pres > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")
        
        # check if we want to use model order reduction and if yes, initialize MOR class
        try: self.have_rom = io_params['use_model_order_red']
        except: self.have_rom = False

        if self.have_rom:
            import mor
            self.rom = mor.MorBase(mor_params, comm)
        
        # create finite element objects for u and p
        P_u = VectorElement("CG", self.io.mesh.ufl_cell(), self.order_disp)
        P_p = FiniteElement("CG", self.io.mesh.ufl_cell(), self.order_pres)
        # function spaces for u and p
        self.V_u = FunctionSpace(self.io.mesh, P_u)
        self.V_p = FunctionSpace(self.io.mesh, P_p)

        # Quadrature tensor, vector, and scalar elements
        Q_tensor = TensorElement("Quadrature", self.io.mesh.ufl_cell(), degree=1, quad_scheme="default")
        Q_vector = VectorElement("Quadrature", self.io.mesh.ufl_cell(), degree=1, quad_scheme="default")
        Q_scalar = FiniteElement("Quadrature", self.io.mesh.ufl_cell(), degree=1, quad_scheme="default")

        # not yet working - we cannot interpolate into Quadrature elements with the current dolfinx version currently!
        #self.Vd_tensor = FunctionSpace(self.io.mesh, Q_tensor)
        #self.Vd_vector = FunctionSpace(self.io.mesh, Q_vector)
        #self.Vd_scalar = FunctionSpace(self.io.mesh, Q_scalar)

        # Quadrature function spaces (currently not properly functioning for higher-order meshes!!!)
        self.Vd_tensor = TensorFunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))
        self.Vd_vector = VectorFunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))
        self.Vd_scalar = FunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))

        # functions
        self.du    = TrialFunction(self.V_u)            # Incremental displacement
        self.var_u = TestFunction(self.V_u)             # Test function
        self.dp    = TrialFunction(self.V_p)            # Incremental pressure
        self.var_p = TestFunction(self.V_p)             # Test function
        self.u     = Function(self.V_u, name="Displacement")
        self.p     = Function(self.V_p, name="Pressure")
        # values of previous time step
        self.u_old = Function(self.V_u)
        self.v_old = Function(self.V_u)
        self.a_old = Function(self.V_u)
        self.p_old = Function(self.V_p)
        # a setpoint displacement for multiscale analysis
        self.u_set = Function(self.V_u)
        self.p_set = Function(self.V_p)
        self.tau_a_set = Function(self.Vd_scalar)
        # initial (zero) functions for initial stiffness evaluation (e.g. for Rayleigh damping)
        self.u_ini, self.p_ini, self.theta_ini, self.tau_a_ini = Function(self.V_u), Function(self.V_p), Function(self.Vd_scalar), Function(self.Vd_scalar)
        self.theta_ini.vector.set(1.0)
        self.theta_ini.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # growth stretch
        self.theta = Function(self.Vd_scalar, name="theta")
        self.theta_old = Function(self.Vd_scalar)
        # initialize to one (theta = 1 means no growth)
        self.theta.vector.set(1.0), self.theta_old.vector.set(1.0)
        self.theta.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD), self.theta_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # active stress
        self.tau_a = Function(self.Vd_scalar, name="tau_a")
        self.tau_a_old = Function(self.Vd_scalar)
        self.amp_old, self.amp_old_set = Function(self.Vd_scalar), Function(self.Vd_scalar)
        self.amp_old.vector.set(1.0), self.amp_old_set.vector.set(1.0)
        self.amp_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD), self.amp_old_set.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # prestressing history defgrad and spring prestress
        if self.prestress_initial:
            self.F_hist = Function(self.Vd_tensor, name="Defgrad_hist")
            self.u_pre = Function(self.V_u)
        else:
            self.F_hist = None
            self.u_pre = None
        
        self.internalvars     = {"theta" : self.theta, "tau_a" : self.tau_a}
        self.internalvars_old = {"theta" : self.theta_old, "tau_a" : self.tau_a_old}
        
        
        # reference coordinates
        self.x_ref = Function(self.V_u)
        self.x_ref.interpolate(self.x_ref_expr)
        
        if self.incompressible_2field:
            self.ndof = self.u.vector.getSize() + self.p.vector.getSize()
        else:
            self.ndof = self.u.vector.getSize()

        # initialize solid time-integration class
        self.ti = timeintegration.timeintegration_solid(time_params, fem_params, time_curves, self.t_init, self.comm)

        # check for materials that need extra treatment (anisotropic, active stress, growth, ...)
        have_fiber1, have_fiber2 = False, False
        self.have_active_stress, self.active_stress_trig, self.have_frank_starling, self.have_growth = False, 'ode', False, False
        self.mat_active_stress, self.mat_growth, self.mat_remodel, self.mat_growth_dir, self.mat_growth_trig = [False]*self.num_domains, [False]*self.num_domains, [False]*self.num_domains, [None]*self.num_domains, [None]*self.num_domains

        # build up map of (numeric) G&R parameters and corresponding form and function dicts - like this, FFCx does not need (expensive) re-compilation when parameter values are changed
        self.growth_param_map = {'trigger_reduction' : 1, 'growth_thres' : 0, 'growth_thres_0' : 0, 'thres_tol' : 0, 'thetamax' : 1, 'thetamin' : 1, 'tau_gr' : 0, 'gamma_gr' : 0, 'tau_gr_rev' : 0, 'gamma_gr_rev' : 0}
        self.mat_growth_param_forms = {}
        self.growth_param_funcs = {}
        for name in self.growth_param_map.keys():
            self.mat_growth_param_forms[name] = []*self.num_domains
            self.growth_param_funcs[name] = Function(self.Vd_scalar)

        self.localsolve, growth_dir = False, None
        self.actstress = []
        for n in range(self.num_domains):
            
            if 'holzapfelogden_dev' in self.constitutive_models['MAT'+str(n+1)+''].keys() or 'guccione_dev' in self.constitutive_models['MAT'+str(n+1)+''].keys():
                have_fiber1, have_fiber2 = True, True
            
            if 'active_fiber' in self.constitutive_models['MAT'+str(n+1)+''].keys():
                have_fiber1 = True
                self.mat_active_stress[n], self.have_active_stress = True, True
                # if one mat has a prescribed active stress, all have to be!
                if 'prescribed_curve' in self.constitutive_models['MAT'+str(n+1)+'']['active_fiber']:
                    self.active_stress_trig = 'prescribed'
                if 'prescribed_multiscale' in self.constitutive_models['MAT'+str(n+1)+'']['active_fiber']:
                    self.active_stress_trig = 'prescribed_multiscale'
                if self.active_stress_trig == 'ode':
                    act_curve = self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)+'']['active_fiber']['activation_curve'])
                    self.actstress.append(activestress_activation(self.constitutive_models['MAT'+str(n+1)+'']['active_fiber'], act_curve))
                    if self.actstress[-1].frankstarling: self.have_frank_starling = True
                if self.active_stress_trig == 'prescribed':
                    self.ti.funcs_to_update.append({self.tau_a : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)+'']['active_fiber']['prescribed_curve'])})

            if 'active_iso' in self.constitutive_models['MAT'+str(n+1)+''].keys():
                self.mat_active_stress[n], self.have_active_stress = True, True
                # if one mat has a prescribed active stress, all have to be!
                if 'prescribed_curve' in self.constitutive_models['MAT'+str(n+1)+'']['active_iso']:
                    self.active_stress_trig = 'prescribed'
                if 'prescribed_multiscale' in self.constitutive_models['MAT'+str(n+1)+'']['active_iso']:
                    self.active_stress_trig = 'prescribed_multiscale'
                if self.active_stress_trig == 'ode':
                    act_curve = self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)+'']['active_iso']['activation_curve'])
                    self.actstress.append(activestress_activation(self.constitutive_models['MAT'+str(n+1)+'']['active_iso'], act_curve))
                if self.active_stress_trig == 'prescribed':
                    self.ti.funcs_to_update.append({self.tau_a : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)+'']['active_iso']['prescribed_curve'])})

            if 'growth' in self.constitutive_models['MAT'+str(n+1)+''].keys():
                self.mat_growth[n], self.have_growth = True, True
                self.mat_growth_dir[n] = self.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_dir']
                self.mat_growth_trig[n] = self.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_trig']
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
                    self.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_thres_0'] = self.constitutive_models['MAT'+str(n+1)+'']['growth']['growth_thres']
                    for name in self.mat_growth_param_forms.keys():
                        try: self.mat_growth_param_forms[name].append(self.constitutive_models['MAT'+str(n+1)+'']['growth'][name])
                        except: self.mat_growth_param_forms[name].append(as_ufl(self.growth_param_map[name]))
                    
                else:
                    for name in self.mat_growth_param_forms.keys(): self.mat_growth_param_forms[name].append(as_ufl(self.growth_param_map[name]))
                # for the case that we have a prescribed growth stretch over time, append curve to functions that need time updates
                # if one mat has a prescribed growth model, all have to be!
                if self.mat_growth_trig[n] == 'prescribed':
                    self.ti.funcs_to_update.append({self.theta : self.ti.timecurves(self.constitutive_models['MAT'+str(n+1)+'']['growth']['prescribed_curve'])})
                if 'remodeling_mat' in self.constitutive_models['MAT'+str(n+1)+'']['growth'].keys():
                    self.mat_remodel[n] = True
            else:
                for name in self.mat_growth_param_forms.keys(): self.mat_growth_param_forms[name].append(as_ufl(0))
                
        # full linearization of our remodeling law can lead to excessive compiler times for FFCx... :-/
        # let's try if we might can go without one of the critial terms (derivative of remodeling fraction w.r.t. C)
        try: self.lin_remod_full = fem_params['lin_remodeling_full']
        except: self.lin_remod_full = True

        # growth threshold (as function, since in multiscale approach, it can vary element-wise)
        if self.have_growth and self.localsolve:
            for name in self.growth_param_funcs.keys():
                param_proj = project(self.mat_growth_param_forms[name], self.Vd_scalar, self.dx_)
                self.growth_param_funcs[name].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                self.growth_param_funcs[name].interpolate(param_proj)

        # read in fiber data
        if have_fiber1:

            fibarray = ['fiber']
            if have_fiber2: fibarray.append('sheet')

            # fiber function space - vector defined on quadrature points
            V_fib = self.Vd_vector
            self.fib_func = self.io.readin_fibers(fibarray, V_fib, self.dx_)

        else:
            self.fib_func = None

        
        # for multiscale G&R analysis
        self.tol_stop_large = 0

        # initialize kinematics class
        self.ki = solid_kinematics_constitutive.kinematics(fib_funcs=self.fib_func, F_hist=self.F_hist)

        # initialize material/constitutive class
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(solid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)+''], self.incompressible_2field, mat_growth=self.mat_growth[n], mat_remodel=self.mat_remodel[n]))

        # initialize solid variational form class
        self.vf = solid_variationalform.variationalform(self.var_u, self.du, self.var_p, self.dp, self.io.n0, self.x_ref)
        
        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_solid(bc_dict, self.fem_params, self.io, self.ki, self.vf, self.ti)

        if self.prestress_initial:
            # initialize prestressing history deformation gradient
            Id_proj = project(Identity(len(self.u)), self.Vd_tensor, self.dx_)
            self.F_hist.interpolate(Id_proj)
  
        self.bc_dict = bc_dict
        
        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.V_u)

        self.set_variational_forms_and_jacobians()



    # the main function that defines the solid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms_and_jacobians(self):

        # set forms for acceleration and velocity
        self.acc, self.vel = self.ti.set_acc_vel(self.u, self.u_old, self.v_old, self.a_old)

        # kinetic, internal, and pressure virtual work
        self.deltaW_kin,  self.deltaW_kin_old  = as_ufl(0), as_ufl(0)
        self.deltaW_int,  self.deltaW_int_old  = as_ufl(0), as_ufl(0)
        self.deltaW_damp, self.deltaW_damp_old = as_ufl(0), as_ufl(0)
        self.deltaW_p,    self.deltaW_p_old    = as_ufl(0), as_ufl(0)
        
        for n in range(self.num_domains):

            if self.timint != 'static':
                # kinetic virtual work
                self.deltaW_kin     += self.vf.deltaW_kin(self.acc, self.rho0[n], self.dx_[n])
                self.deltaW_kin_old += self.vf.deltaW_kin(self.a_old, self.rho0[n], self.dx_[n])

                # Rayleigh damping virtual work
                if self.rayleigh[n]:
                    self.deltaW_damp     += self.vf.deltaW_damp(self.eta_m[n], self.eta_k[n], self.rho0[n], self.ma[n].S(self.u_ini, self.p_ini, ivar={"theta" : self.theta_ini, "tau_a" : self.tau_a_ini}, tang=True), self.vel, self.dx_[n])
                    self.deltaW_damp_old += self.vf.deltaW_damp(self.eta_m[n], self.eta_k[n], self.rho0[n], self.ma[n].S(self.u_ini, self.p_ini, ivar={"theta" : self.theta_ini, "tau_a" : self.tau_a_ini}, tang=True), self.v_old, self.dx_[n])

            # internal virtual work
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].S(self.u, self.p, ivar=self.internalvars), self.ki.F(self.u), self.dx_[n])
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].S(self.u_old, self.p_old, ivar=self.internalvars_old), self.ki.F(self.u_old), self.dx_[n])
        
            # pressure virtual work (for incompressible formulation)
            # this has to be treated like the evaluation of a volumetric material, hence with the elastic part of J
            if self.mat_growth[n]: J, J_old = self.ma[n].J_e(self.u, self.theta), self.ma[n].J_e(self.u_old, self.theta_old)
            else:                  J, J_old = self.ki.J(self.u), self.ki.J(self.u_old)
            self.deltaW_p       += self.vf.deltaW_int_pres(J, self.dx_[n])
            self.deltaW_p_old   += self.vf.deltaW_int_pres(J_old, self.dx_[n])
        
        
        # external virtual work (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_neumann_old, w_robin, w_robin_old = as_ufl(0), as_ufl(0), as_ufl(0), as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann, w_neumann_old = self.bc.neumann_bcs(self.V_u, self.Vd_scalar, self.u, self.u_old)
        if 'robin' in self.bc_dict.keys():
            w_robin, w_robin_old = self.bc.robin_bcs(self.u, self.vel, self.u_old, self.v_old, self.u_pre)

        # for (quasi-static) prestressing, we need to eliminate dashpots and replace true with reference Neumann loads in our external virtual work
        w_neumann_prestr, w_robin_prestr = as_ufl(0), as_ufl(0)
        if self.prestress_initial:
            bc_dict_prestr = copy.deepcopy(self.bc_dict)
            # get rid of dashpots
            if 'robin' in bc_dict_prestr.keys():
                for r in bc_dict_prestr['robin']:
                    if r['type'] == 'dashpot': r['visc'] = 0.
            # replace true Neumann loads by reference ones
            if 'neumann' in bc_dict_prestr.keys():
                for n in bc_dict_prestr['neumann']:
                    if n['type'] == 'true': n['type'] = 'pk1'
            bc_prestr = boundaryconditions.boundary_cond_solid(bc_dict_prestr, self.fem_params, self.io, self.ki, self.vf, self.ti)
            if 'neumann' in bc_dict_prestr.keys():
                w_neumann_prestr, _ = bc_prestr.neumann_bcs(self.V_u, self.Vd_scalar, self.u, self.u_old)
            if 'robin' in bc_dict_prestr.keys():
                w_robin_prestr, _ = bc_prestr.robin_bcs(self.u, self.vel, self.u_old, self.v_old, self.u_pre)
            self.deltaW_prestr_ext = w_neumann_prestr + w_robin_prestr

        # TODO: Body forces!
        self.deltaW_ext     = w_neumann + w_robin
        self.deltaW_ext_old = w_neumann_old + w_robin_old

        self.timefac_m, self.timefac = self.ti.timefactors()


        ### full weakforms 

        # quasi-static weak form: internal minus external virtual work
        if self.timint == 'static':
            
            self.weakform_u = self.deltaW_int - self.deltaW_ext
            
            if self.incompressible_2field:
                self.weakform_p = self.deltaW_p

        # full dynamic weak form: kinetic plus internal (plus damping) minus external virtual work
        else:
            
            self.weakform_u = self.timefac_m * self.deltaW_kin  + (1.-self.timefac_m) * self.deltaW_kin_old + \
                              self.timefac   * self.deltaW_damp + (1.-self.timefac)   * self.deltaW_damp_old + \
                              self.timefac   * self.deltaW_int  + (1.-self.timefac)   * self.deltaW_int_old - \
                              self.timefac   * self.deltaW_ext  - (1.-self.timefac)   * self.deltaW_ext_old
            
            if self.incompressible_2field:
                self.weakform_p = self.timefac * self.deltaW_p + (1.-self.timefac) * self.deltaW_p_old 


        ### local weak forms at Gauss points for inelastic materials
        self.r_growth, self.del_theta = [], []
        for n in range(self.num_domains):
            
            if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                # growth residual and increment
                a, b = self.ma[n].res_dtheta_growth(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs, 'res_del')
                self.r_growth.append(a), self.del_theta.append(b)
            else:
                self.r_growth.append(as_ufl(0)), self.del_theta.append(as_ufl(0))


        ### Jacobians
        
        # kinetic virtual work linearization (deltaW_kin already has contributions from all domains)
        self.jac_uu = self.timefac_m * derivative(self.deltaW_kin, self.u, self.du)
        
        # internal virtual work linearization treated differently: since we want to be able to account for nonlinear materials at Gauss
        # point level with deformation-dependent internal variables (i.e. growth or plasticity), we make use of a more explicit formulation
        # of the linearization which involves the fourth-order material tangent operator Ctang ("derivative" cannot take care of the
        # dependence of the internal variables on the deformation if this dependence is nonlinear and cannot be expressed analytically)
        for n in range(self.num_domains):
            
            # material tangent operator
            Cmat = self.ma[n].S(self.u, self.p, ivar=self.internalvars, tang=True)

            if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                # growth tangent operator
                Cgrowth = self.ma[n].Cgrowth(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs)
                if self.mat_remodel[n] and self.lin_remod_full:
                    # remodeling tangent operator
                    Cremod = self.ma[n].Cremod(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs)
                    Ctang = Cmat + Cgrowth + Cremod
                else:
                    Ctang = Cmat + Cgrowth
            else:
                Ctang = Cmat
            
            self.jac_uu += self.timefac * self.vf.Lin_deltaW_int_du(self.ma[n].S(self.u, self.p, ivar=self.internalvars), self.ki.F(self.u), self.u, Ctang, self.dx_[n])
        
        # Rayleigh damping virtual work contribution to stiffness
        self.jac_uu += self.timefac * derivative(self.deltaW_damp, self.u, self.du)
        
        # external virtual work contribution to stiffness (from nonlinear follower loads or Robin boundary tractions)
        self.jac_uu += -self.timefac * derivative(self.deltaW_ext, self.u, self.du)

        # pressure contributions
        if self.incompressible_2field:
            
            self.jac_up, self.jac_pu, self.a_p11, self.p11 = as_ufl(0), as_ufl(0), as_ufl(0), as_ufl(0)
            
            for n in range(self.num_domains):
                # this has to be treated like the evaluation of a volumetric material, hence with the elastic part of J
                if self.mat_growth[n]:
                    J    = self.ma[n].J_e(self.u, self.theta)
                    Jmat = self.ma[n].dJedC(self.u, self.theta)
                else:
                    J    = self.ki.J(self.u)
                    Jmat = self.ki.dJdC(self.u)
                
                Cmat_p = diff(self.ma[n].S(self.u, self.p, ivar=self.internalvars), self.p)
                
                if self.mat_growth[n] and self.mat_growth_trig[n] != 'prescribed' and self.mat_growth_trig[n] != 'prescribed_multiscale':
                    Cmat = self.ma[n].S(self.u, self.p, ivar=self.internalvars, tang=True)
                    # growth tangent operators - keep in mind that we have theta = theta(C(u),p) in general!
                    # for stress-mediated growth, we get a contribution to the pressure material tangent operator
                    Cgrowth_p = self.ma[n].Cgrowth_p(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs)
                    if self.mat_remodel[n] and self.lin_remod_full:
                        # remodeling tangent operator
                        Cremod_p = self.ma[n].Cremod_p(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs)
                        Ctang_p = Cmat_p + Cgrowth_p + Cremod_p
                    else:
                        Ctang_p = Cmat_p + Cgrowth_p
                    # for all types of deformation-dependent growth, we need to add the growth contributions to the Jacobian tangent operator
                    Jgrowth = diff(J,self.theta) * self.ma[n].dtheta_dC(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs)
                    Jtang = Jmat + Jgrowth
                    # ok... for stress-mediated growth, we actually get a non-zero right-bottom (11) block in our saddle-point system matrix,
                    # since Je = Je(C,theta(C,p)) ---> dJe/dp = dJe/dtheta * dtheta/dp
                    # TeX: D_{\Delta p}\!\int\limits_{\Omega_0} (J^{\mathrm{e}}-1)\delta p\,\mathrm{d}V = \int\limits_{\Omega_0} \frac{\partial J^{\mathrm{e}}}{\partial p}\Delta p \,\delta p\,\mathrm{d}V,
                    # with \frac{\partial J^{\mathrm{e}}}{\partial p} = \frac{\partial J^{\mathrm{e}}}{\partial \vartheta}\frac{\partial \vartheta}{\partial p}
                    dthetadp = self.ma[n].dtheta_dp(self.u, self.p, self.internalvars, self.theta_old, self.dt, self.growth_param_funcs)
                    if not isinstance(dthetadp, constantvalue.Zero):
                        self.p11 += diff(J,self.theta) * dthetadp * self.dp * self.var_p * self.dx_[n]
                else:
                    Ctang_p = Cmat_p
                    Jtang = Jmat
                
                self.jac_up += self.timefac * self.vf.Lin_deltaW_int_dp(self.ki.F(self.u), Ctang_p, self.dx_[n])
                self.jac_pu += self.timefac * self.vf.Lin_deltaW_int_pres_du(self.ki.F(self.u), Jtang, self.u, self.dx_[n])
                
                # for saddle-point block-diagonal preconditioner
                self.a_p11 += inner(self.dp, self.var_p) * self.dx_[n]

        if self.prestress_initial:
            # quasi-static weak forms (don't dare to use fancy growth laws or other inelastic stuff during prestressing...)
            self.weakform_prestress_u = self.deltaW_int - self.deltaW_prestr_ext
            self.jac_prestress_uu = derivative(self.weakform_prestress_u, self.u, self.du)
            if self.incompressible_2field:
                self.weakform_prestress_p = self.deltaW_p
                self.jac_prestress_up = derivative(self.weakform_prestress_u, self.p, self.dp)
                self.jac_prestress_pu = derivative(self.weakform_prestress_p, self.u, self.du)


        
    # reference coordinates
    def x_ref_expr(self, x):
        return np.stack((x[0],x[1],x[2]))


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
                    
                    amp_old_.append(as_ufl(0))

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
                    lam_fib = as_ufl(1)
                
                tau_a_.append(self.actstress[na].tau_act(self.tau_a_old, t, self.dt, lam_fib, self.amp_old))
                
                na+=1
                
            else:
                
                tau_a_.append(as_ufl(0))
                
        # project and interpolate to quadrature function space
        tau_a_proj = project(tau_a_, self.Vd_scalar, self.dx_)
        self.tau_a.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.tau_a.interpolate(tau_a_proj)


    # computes and prints the growth rate of the whole solid
    def compute_solid_growth_rate(self, N, t):
        
        dtheta_all = as_ufl(0)
        for n in range(self.num_domains):
            
            dtheta_all += (self.theta - self.theta_old) / (self.dt) * self.dx_[n]

        gr = assemble_scalar(dtheta_all)
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


    
class SolidmechanicsSolver():

    def __init__(self, problem, solver_params):
    
        self.pb = problem
        
        self.solver_params = solver_params

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, self.pb.V_u, self.pb.V_p, self.solver_params)


    def solve_problem(self):
        
        start = time.time()

        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.comm, self.pb.ndof)

        # perform Proper Orthogonal Decomposition
        if self.pb.have_rom:
            self.pb.rom.POD(self.pb)

        # read restart information
        if self.pb.restart_step > 0:
            self.pb.io.readcheckpoint(self.pb, self.pb.restart_step)
            self.pb.simname += '_r'+str(self.pb.restart_step)

        # in case we want to prestress with MULF (Gee et al. 2010) prior to solving the full solid problem
        if self.pb.prestress_initial and self.pb.restart_step == 0:

            self.solve_initial_prestress()

        else:
            # set flag definitely to False if we're restarting
            self.pb.prestress_initial = False

        # consider consistent initial acceleration
        if self.pb.timint != 'static' and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old
            
            jac_a = derivative(weakform_a, self.pb.a_old, self.pb.du) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.a_old)

        # write mesh output
        self.pb.io.write_output(self.pb, writemesh=True)
        
        
        # solid main time loop
        for N in range(self.pb.restart_step+1, self.pb.numstep_stop+1):

            wts = time.time()
            
            # current time
            t = N * self.pb.dt

            # set time-dependent functions
            self.pb.ti.set_time_funcs(self.pb.ti.funcs_to_update, self.pb.ti.funcs_to_update_vec, t)
            
            # take care of active stress
            if self.pb.have_active_stress and self.pb.active_stress_trig == 'ode':
                self.pb.evaluate_active_stress_ode(t)

            # solve
            self.solnln.newton(self.pb.u, self.pb.p, locvar=self.pb.theta, locresform=self.pb.r_growth, locincrform=self.pb.del_theta)

            # compute the growth rate (has to be called before update_timestep)
            if self.pb.have_growth:
                self.pb.compute_solid_growth_rate(N, t)

            # write output
            self.pb.io.write_output(self.pb, N=N, t=t)
            
            # update - displacement, velocity, acceleration, pressure, all internal variables, all time functions
            self.pb.ti.update_timestep(self.pb.u, self.pb.u_old, self.pb.v_old, self.pb.a_old, self.pb.p, self.pb.p_old, self.pb.internalvars, self.pb.internalvars_old, self.pb.ti.funcs_to_update, self.pb.ti.funcs_to_update_old, self.pb.ti.funcs_to_update_vec, self.pb.ti.funcs_to_update_vec_old)

            # solve time for time step
            wte = time.time()
            wt = wte - wts

            # print time step info to screen
            self.pb.ti.print_timestep(N, t, wt=wt)

            # write restart info - old and new quantities are the same at this stage
            self.pb.io.write_restart(self.pb, N)

            if self.pb.problem_type == 'solid_flow0d_multiscale_gandr' and abs(self.pb.growth_rate) <= self.pb.tol_stop_large:
                break
            
            
        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()


    def solve_initial_prestress(self):
        
        utilities.print_prestress('start', self.pb.comm)

        # solve in 1 load step using PTC!
        self.solnln.PTC = True

        self.solnln.newton(self.pb.u, self.pb.p)

        # MULF update
        self.pb.ki.prestress_update(self.pb.u, self.pb.Vd_tensor, self.pb.dx_, self.pb.u_pre)
        
        # set flag to false again
        self.pb.prestress_initial = False

        # reset PTC flag to what it was
        try: self.solnln.PTC = self.solver_params['ptc']
        except: self.solnln.PTC = False

        utilities.print_prestress('end', self.pb.comm)
