#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import ioroutines
import fluid_kinematics_constitutive
import fluid_variationalform
import timeintegration
import utilities
import solver_nonlin
import boundaryconditions
from projection import project

from base import problem_base, solver_base

# fluid mechanics, governed by incompressible Navier-Stokes equations:

#\begin{align}
#\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \left(\boldsymbol{\nabla} \otimes \boldsymbol{v}\right)^{\mathrm{T}} \boldsymbol{v}\right) = \boldsymbol{\nabla} \cdot \boldsymbol{\sigma} + \hat{\boldsymbol{b}} \quad \text{in} \; \Omega \times [0, T] \\
#\boldsymbol{\nabla} \cdot \boldsymbol{v} = 0 \quad \text{in} \; \Omega \times [0, T]
#\end{align}

class FluidmechanicsProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params={}, comm=None, aleproblem=None):
        problem_base.__init__(self, io_params, time_params, comm)
        
        self.problem_physics = 'fluid'
        
        self.simname = io_params['simname']
        self.results_to_write = io_params['results_to_write']
        
        self.io = io

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        
        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_vel = fem_params['order_vel']
        self.order_pres = fem_params['order_pres']
        self.quad_degree = fem_params['quad_degree']
        
        # collect domain data
        self.dx_, self.rho = [], []
        for n in range(self.num_domains):
            # integration domains
            self.dx_.append(ufl.dx(subdomain_data=self.io.mt_d, subdomain_id=n+1, metadata={'quadrature_degree': self.quad_degree}))
            # data for inertial forces: density
            self.rho.append(self.constitutive_models['MAT'+str(n+1)]['inertia']['rho'])
        
        self.incompressible_2field = True # always true!
        
        # whether to enforce continuity of mass at midpoint or not
        try: self.pressure_at_midpoint = fem_params['pressure_at_midpoint']
        except: self.pressure_at_midpoint = False
        
        self.localsolve = False # no idea what might have to be solved locally...
        self.prestress_initial = False # guess prestressing in fluid is somehow senseless...
        self.p11 = ufl.as_ufl(0) # can't think of a fluid case with non-zero 11-block in system matrix...

        self.sub_solve = False

        self.dim = self.io.mesh.geometry.dim
    
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

        # make sure that we use the correct velocity order in case of a higher-order mesh
        if self.Vex.degree() > 1:
            if self.Vex.degree() != self.order_vel:
                raise ValueError("Order of velocity field not compatible with degree of finite element!")

        if self.order_vel == self.order_pres:
            raise ValueError("Equal order velocity and pressure interpolation is not recommended for non-stabilized Navier-Stokes!")

        # check if we want to use model order reduction and if yes, initialize MOR class
        try: self.have_rom = io_params['use_model_order_red']
        except: self.have_rom = False

        if self.have_rom:
            import mor
            self.rom = mor.ModelOrderReduction(mor_params, self.comm)

        # ALE fluid problem
        self.aleproblem = aleproblem

        # create finite element objects for v and p
        P_v = ufl.VectorElement("CG", self.io.mesh.ufl_cell(), self.order_vel)
        P_p = ufl.FiniteElement("CG", self.io.mesh.ufl_cell(), self.order_pres)
        # function spaces for v and p
        self.V_v = fem.FunctionSpace(self.io.mesh, P_v)
        self.V_p = fem.FunctionSpace(self.io.mesh, P_p)
        # tensor finite element and function space
        P_tensor = ufl.TensorElement("CG", self.io.mesh.ufl_cell(), self.order_vel)
        self.V_tensor = fem.FunctionSpace(self.io.mesh, P_tensor)

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.TensorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_vector = fem.VectorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_scalar = fem.FunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))

        # functions
        self.dv     = ufl.TrialFunction(self.V_v)            # Incremental velocity
        self.var_v  = ufl.TestFunction(self.V_v)             # Test function
        self.dp     = ufl.TrialFunction(self.V_p)            # Incremental pressure
        self.var_p  = ufl.TestFunction(self.V_p)             # Test function
        self.v      = fem.Function(self.V_v, name="Velocity")
        self.p      = fem.Function(self.V_p, name="Pressure")
        # values of previous time step
        self.v_old  = fem.Function(self.V_v)
        self.a_old  = fem.Function(self.V_v)
        self.p_old  = fem.Function(self.V_p)
        # a fluid displacement
        self.uf_old = fem.Function(self.V_v)

        self.numdof = self.v.vector.getSize() + self.p.vector.getSize()

        # initialize fluid time-integration class
        self.ti = timeintegration.timeintegration_fluid(time_params, fem_params, time_curves, self.t_init, self.comm)

        # initialize kinematics_constitutive class
        self.ki = fluid_kinematics_constitutive.kinematics(self.dim)
        
        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(fluid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)]))
        
        # initialize fluid variational form class
        if self.aleproblem is None:
            self.u, self.u_old, self.wel, self.w_old, self.Fale, self.Fale_old = None, None, None, None, None, None
            self.vf = fluid_variationalform.variationalform(self.var_v, self.dv, self.var_p, self.dp, self.io.n0)
        else:
            self.u, self.u_old = self.aleproblem.u, self.aleproblem.u_old
            self.wel, self.w_old = self.aleproblem.wel, self.aleproblem.w_old
            self.Fale, self.Fale_old = self.aleproblem.ki.F(self.u), self.aleproblem.ki.F(self.u_old)
            self.vf = fluid_variationalform.variationalform_ale(self.var_v, self.dv, self.var_p, self.dp, self.io.n0)

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_fluid(bc_dict, fem_params, self.io, self.vf, self.ti, ki=self.ki)
        
        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.V_v)

        self.set_variational_forms_and_jacobians()
            
            
    def get_problem_var_list(self):
        
        is_ghosted = [True]*2
        return [self.v.vector, self.p.vector], is_ghosted


    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms_and_jacobians(self):

        # set form for acceleration
        self.acc = self.ti.set_acc(self.v, self.v_old, self.a_old)

        # set form for fluid displacement (needed for FrSI)
        self.ufluid = self.ti.set_uf(self.v, self.v_old, self.uf_old)

        # kinetic, internal, and pressure virtual power
        self.deltaW_kin, self.deltaW_kin_old = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_int, self.deltaW_int_old = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_p,   self.deltaW_p_old   = ufl.as_ufl(0), ufl.as_ufl(0)
        
        for n in range(self.num_domains):

            if self.timint != 'static':
                # kinetic virtual power
                self.deltaW_kin     += self.vf.deltaW_kin(self.acc, self.v, self.rho[n], self.dx_[n], w=self.wel, Fale=self.Fale)
                self.deltaW_kin_old += self.vf.deltaW_kin(self.a_old, self.v_old, self.rho[n], self.dx_[n], w=self.w_old, Fale=self.Fale_old)
            
            # internal virtual power
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].sigma(self.v, self.p), self.dx_[n], Fale=self.Fale)
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].sigma(self.v_old, self.p_old), self.dx_[n], Fale=self.Fale_old)
            
            # pressure virtual power
            self.deltaW_p       += self.vf.deltaW_int_pres(self.v, self.dx_[n], Fale=self.Fale)
            self.deltaW_p_old   += self.vf.deltaW_int_pres(self.v_old, self.dx_[n], Fale=self.Fale_old)
        
        # external virtual power (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_neumann_old, w_robin, w_robin_old, w_membrane, w_membrane_old = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann     = self.bc.neumann_bcs(self.V_v, self.Vd_scalar, Fale=self.Fale, funcs_to_update=self.ti.funcs_to_update, funcs_to_update_vec=self.ti.funcs_to_update_vec)
            w_neumann_old = self.bc.neumann_bcs(self.V_v, self.Vd_scalar, Fale=self.Fale_old, funcs_to_update=self.ti.funcs_to_update_old, funcs_to_update_vec=self.ti.funcs_to_update_vec_old)
        if 'robin' in self.bc_dict.keys():
            w_robin     = self.bc.robin_bcs(self.v)
            w_robin_old = self.bc.robin_bcs(self.v_old)
        # reduced-solid for FrSI problem
        if 'membrane' in self.bc_dict.keys():
            w_membrane     = self.bc.membranesurf_bcs(self.ufluid, self.v, self.acc)
            w_membrane_old = self.bc.membranesurf_bcs(self.uf_old, self.v_old, self.a_old)
        if 'dirichlet_weak' in self.bc_dict.keys():
            raise RuntimeError("Cannot use weak Dirichlet BCs for fluid mechanics currently!")

        # TODO: Body forces!
        self.deltaW_ext     = w_neumann + w_robin + w_membrane
        self.deltaW_ext_old = w_neumann_old + w_robin_old + w_membrane_old
        
        self.timefac_m, self.timefac = self.ti.timefactors()

        ### full weakforms 
        
        # kinetic plus internal minus external virtual power
        self.weakform_v = self.timefac_m * self.deltaW_kin + (1.-self.timefac_m) * self.deltaW_kin_old + \
                          self.timefac   * self.deltaW_int + (1.-self.timefac)   * self.deltaW_int_old - \
                          self.timefac   * self.deltaW_ext - (1.-self.timefac)   * self.deltaW_ext_old
        
        if self.pressure_at_midpoint:
            self.weakform_p = self.timefac   * self.deltaW_p   + (1.-self.timefac)   * self.deltaW_p_old
        else:
            self.weakform_p = self.deltaW_p
        
        # Reynolds number: ratio of inertial to viscous forces
        self.Re = ufl.as_ufl(0)
        for n in range(self.num_domains):
            self.Re += ufl.sqrt(ufl.dot(self.vf.f_inert(self.acc,self.v,self.rho[n]), self.vf.f_inert(self.acc,self.v,self.rho[n]))) / ufl.sqrt(ufl.dot(self.vf.f_viscous(self.ma[n].sigma(self.v, self.p)), self.vf.f_viscous(self.ma[n].sigma(self.v, self.p))))
        
        ### Jacobians
        
        self.jac_vv = ufl.derivative(self.weakform_v, self.v, self.dv)
        self.jac_vp = ufl.derivative(self.weakform_v, self.p, self.dp)
        self.jac_pv = ufl.derivative(self.weakform_p, self.v, self.dv)

        # for saddle-point block-diagonal preconditioner - TODO: Doesn't work very well...
        self.a_p11 = ufl.as_ufl(0)
        
        for n in range(self.num_domains):
            self.a_p11 += ufl.inner(self.dp, self.var_p) * self.dx_[n]

            

    def set_forms_solver(self):
        pass


    def assemble_residual_stiffness(self, t, subsolver=None):

        # assemble velocity rhs vector
        r_v = fem.petsc.assemble_vector(fem.form(self.weakform_v))
        fem.apply_lifting(r_v, [fem.form(self.jac_vv)], [self.bc.dbcs], x0=[self.v.vector], scale=-1.0)
        r_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(r_v, self.bc.dbcs, x0=self.v.vector, scale=-1.0)

        # assemble system matrix
        K_vv = fem.petsc.assemble_matrix(fem.form(self.jac_vv), self.bc.dbcs)
        K_vv.assemble()
        
        # assemble pressure rhs vector
        r_p = fem.petsc.assemble_vector(fem.form(self.weakform_p))
        r_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # assemble system matrices
        K_vp = fem.petsc.assemble_matrix(fem.form(self.jac_vp), self.bc.dbcs)
        K_vp.assemble()
        K_pv = fem.petsc.assemble_matrix(fem.form(self.jac_pv), []) # currently, we do not consider pressure DBCs
        K_pv.assemble()
        K_pp = None

        return [r_v, r_p], [[K_vv, K_vp], [K_pv, K_pp]]


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):

        # perform Proper Orthogonal Decomposition
        if self.have_rom:
            self.rom.POD(self, self.V_v)

                
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
            
            
    def evaluate_post_solve(self, t, N):
        pass


    def set_output_state(self):
        pass

            
    def write_output(self, N, t, mesh=False): 

        self.io.write_output(self, N=N, t=t)

            
    def update(self):
        
        # update - velocity, acceleration, pressure, all internal variables, all time functions
        self.ti.update_timestep(self.v, self.v_old, self.a_old, self.p, self.p_old, self.ti.funcs_to_update, self.ti.funcs_to_update_old, self.ti.funcs_to_update_vec, self.ti.funcs_to_update_vec_old, uf_old=self.uf_old)


    def print_to_screen(self):
        pass
    
    
    def induce_state_change(self):
        pass


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)
        
        
    def check_abort(self, t):
        pass



class FluidmechanicsSolver(solver_base):

    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if self.pb.timint != 'static' and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old
            
            jac_a = ufl.derivative(weakform_a, self.pb.a_old, self.pb.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, wt):
    
        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)
