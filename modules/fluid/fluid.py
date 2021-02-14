#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np
from dolfinx import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, Function, DirichletBC
from ufl import TrialFunction, TestFunction, FiniteElement, VectorElement, TensorElement, derivative, diff, inner, dx, ds, as_vector, as_ufl, dot, grad, sqrt, conditional, ge, Min

import ioroutines
import fluid_kinematics_constitutive
import fluid_variationalform
import timeintegration
import utilities
import solver_nonlin
import boundaryconditions

from base import problem_base

# NOTE: Fluid code is still experimental and NOT tested!!!

# fluid mechanics, governed by incompressible Navier-Stokes equations:

#\begin{align}
#\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \left(\boldsymbol{\nabla} \otimes \boldsymbol{v}\right)^{\mathrm{T}} \boldsymbol{v}\right) = \boldsymbol{\nabla} \cdot \boldsymbol{\sigma} + \hat{\boldsymbol{b}} \quad \text{in} \; \Omega \times [0, T] \\
#\boldsymbol{\nabla} \cdot \boldsymbol{v} = 0 \quad \text{in} \; \Omega \times [0, T]
#\end{align}

class FluidmechanicsProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, comm=None):
        problem_base.__init__(self, io_params, time_params, comm)
        
        self.problem_physics = 'fluid'
        
        self.io = ioroutines.IO_fluid(io_params, self.comm)
        self.io.readin_mesh()

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)

        self.order_vel = fem_params['order_vel']
        self.order_pres = fem_params['order_pres']
        self.quad_degree = fem_params['quad_degree']
        
        # collect domain data
        self.dx_, self.rho = [], []
        for n in range(self.num_domains):
            # integration domains
            self.dx_.append(dx(subdomain_data=self.io.mt_d, subdomain_id=n+1, metadata={'quadrature_degree': self.quad_degree}))
            # data for inertial forces: density
            self.rho.append(constitutive_models['MAT'+str(n+1)+'']['inertia']['rho'])
        
        self.incompressible_2field = True # always true!
        self.localsolve = False # no idea what might have to be solved locally...

        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            dg_type = "DG"
            if (self.order_vel > 1 or self.order_pres > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            dg_type = "DQ"
            if (self.order_vel > 1 or self.order_pres > 1) and self.quad_degree < 4:
                raise ValueError("Use at least a quadrature degree of 4 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")

        # create finite element objects for v and p
        self.P_v = VectorElement("CG", self.io.mesh.ufl_cell(), self.order_vel)
        self.P_p = FiniteElement("CG", self.io.mesh.ufl_cell(), self.order_pres)
        # function spaces for v and p
        self.V_v = FunctionSpace(self.io.mesh, self.P_v)
        self.V_p = FunctionSpace(self.io.mesh, self.P_p)

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = TensorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_vector = VectorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_scalar = FunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))

        # functions
        self.dv    = TrialFunction(self.V_v)            # Incremental velocity
        self.var_v = TestFunction(self.V_v)             # Test function
        self.dp    = TrialFunction(self.V_p)            # Incremental pressure
        self.var_p = TestFunction(self.V_p)             # Test function
        self.v     = Function(self.V_v, name="Velocity")
        self.a     = Function(self.V_v, name="Acceleration")
        self.p     = Function(self.V_p, name="Pressure")
        # values of previous time step
        self.v_old = Function(self.V_v)
        self.a_old = Function(self.V_v)
        self.p_old = Function(self.V_p)

        self.ndof = self.v.vector.getSize() + self.p.vector.getSize()

        # initialize fluid time-integration class
        self.ti = timeintegration.timeintegration_fluid(time_params, fem_params, time_curves, self.t_init, self.comm)

        # initialize kinematics_constitutive class
        self.ki = fluid_kinematics_constitutive.kinematics()
        
        # initialize material/constitutive class
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(fluid_kinematics_constitutive.constitutive(self.ki, constitutive_models['MAT'+str(n+1)+'']))
        
        # initialize fluid variational form class
        self.vf = fluid_variationalform.variationalform(self.var_v, self.dv, self.var_p, self.dp, self.io.n0)

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_fluid(bc_dict, fem_params, self.io, self.ki, self.vf, self.ti)
        
        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.V_v)

        self.set_variational_forms_and_jacobians()




    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms_and_jacobians(self):

        # set form for acceleration
        self.acc = self.ti.set_acc(self.v, self.v_old, self.a_old)

        # kinetic, internal, and pressure virtual power
        self.deltaP_kin, self.deltaP_kin_old = as_ufl(0), as_ufl(0)
        self.deltaP_int, self.deltaP_int_old = as_ufl(0), as_ufl(0)
        self.deltaP_p,   self.deltaP_p_old   = as_ufl(0), as_ufl(0)
        
        for n in range(self.num_domains):
        
            if self.timint != 'static':
                # kinetic virtual power
                self.deltaP_kin     += self.vf.deltaP_kin(self.acc, self.v, self.rho[n], self.dx_[n])
                self.deltaP_kin_old += self.vf.deltaP_kin(self.a_old, self.v_old, self.rho[n], self.dx_[n])
            
            # internal virtual power
            self.deltaP_int     += self.vf.deltaP_int(self.ma[n].sigma(self.v, self.p), self.dx_[n])
            self.deltaP_int_old += self.vf.deltaP_int(self.ma[n].sigma(self.v_old, self.p_old), self.dx_[n])
            
            # pressure virtual power
            self.deltaP_p       += self.vf.deltaP_int_pres(self.v, self.dx_[n])
            self.deltaP_p_old   += self.vf.deltaP_int_pres(self.v_old, self.dx_[n])
            
        
        # external virtual power (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_neumann_old, w_robin, w_robin_old = as_ufl(0), as_ufl(0), as_ufl(0), as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann, w_neumann_old = self.bc.neumann_bcs(self.V_v, self.Vd_scalar)
        if 'robin' in self.bc_dict.keys():
            w_robin, w_robin_old = self.bc.robin_bcs(self.v, self.v_old)

        # TODO: Body forces!
        self.deltaP_ext     = w_neumann + w_robin
        self.deltaP_ext_old = w_neumann_old + w_robin_old
        

        self.timefac_m, self.timefac = self.ti.timefactors()


        ### full weakforms 
        
        # kinetic plus internal minus external virtual power
        self.weakform_u = self.timefac_m * self.deltaP_kin + (1.-self.timefac_m) * self.deltaP_kin_old + \
                          self.timefac   * self.deltaP_int + (1.-self.timefac)   * self.deltaP_int_old - \
                          self.timefac   * self.deltaP_ext - (1.-self.timefac)   * self.deltaP_ext_old
        
        self.weakform_p = self.timefac   * self.deltaP_p   + (1.-self.timefac)   * self.deltaP_p_old
        
        # Reynolds number: ratio of inertial to viscous forces
        #self.Re = sqrt(dot(self.vf.f_inert(self.acc,self.v,self.rho), self.vf.f_inert(self.acc,self.v,self.rho))) / sqrt(dot(self.vf.f_viscous(self.ma[0].sigma(self.v, self.p)), self.vf.f_viscous(self.ma[0].sigma(self.v, self.p))))
        
        if self.order_vel == self.order_pres:
            
            raise ValueError("Equal order velocity and pressure interpolation requires stabilization! Not yet implemented! Use order_vel > order_pres.") 
            
            #dx1_stab = dx(subdomain_data=self.io.mt_d, subdomain_id=1, metadata={'quadrature_degree': 2*3})

            ## stabilization stuff - TODO: FIXME and finish!!!
            #res_v_strong = self.vf.residual_v_strong(self.acc, self.v, self.rho, self.ma[0].sigma(self.v, self.p))
            #res_v_strong_old = self.vf.residual_v_strong(self.a_old, self.v_old, self.rho, self.ma[0].sigma(self.v_old, self.p_old))
            
            #res_p_strong = self.vf.residual_p_strong(self.v)
            #res_p_strong_old = self.vf.residual_p_strong(self.v_old)
            
            #vnorm, vnorm_old = sqrt(dot(self.v, self.v)), sqrt(dot(self.v_old, self.v_old))

            #nu = 0.004/self.rho
            #Cinv = 16.*self.Re
            #tau_SUPG = Min(self.io.h0**2./(Cinv*nu), self.io.h0/(2.*vnorm))


            ##tau = ( (2.*self.dt)**2. + (2.0*vnorm_/self.io.h0)**2 + (4.0*nu/self.io.h0**2.)**2. )**(-0.5)

            ##delta = conditional(ge(vnorm,1.0e-8), self.io.h0/(2.*vnorm), 0.)
            ##delta_old = conditional(ge(vnorm_old,1.0e-8), self.io.h0/(2.*vnorm_old), 0.)
            
            #stab_v     = tau_SUPG * dot(dot(self.v, grad(self.var_v)),res_v_strong)*dx1_stab

            ##stab_p     = tau_PSPG * dot(dot(self.v, grad(self.var_v)),res_p_strong)*dx1_stab

            #self.weakform_u += self.timefac * stab_v #+ (1.-self.timefac) * stab_old
            
            ##self.weakform_p += tau_SUPG*inner(grad(self.var_p), res_strong)*self.dx1
        
        ### Jacobians
        
        self.jac_uu = derivative(self.weakform_u, self.v, self.dv)
        self.jac_up = derivative(self.weakform_u, self.p, self.dp)
        self.jac_pu = derivative(self.weakform_p, self.v, self.dv)

        # for saddle-point block-diagonal preconditioner - TODO: Doesn't work very well...
        self.a_p11 = as_ufl(0)
        
        for n in range(self.num_domains):
            self.a_p11 += inner(self.dp, self.var_p) * self.dx_[n]






class FluidmechanicsSolver():

    def __init__(self, problem, solver_params):
    
        self.pb = problem
        
        self.solver_params = solver_params

        self.solve_type = self.solver_params['solve_type']


    def solve_problem(self):
        
        start = time.time()

        # print header
        utilities.print_problem(self.pb.problem_physics, self.pb.comm, self.pb.ndof)

        # read restart information
        if self.pb.restart_step > 0:
            self.pb.io.readcheckpoint(self.pb, self.pb.restart_step)
            self.pb.io.simname += '_r'+str(self.pb.restart_step)

        # initialize nonlinear solver class
        solnln = solver_nonlin.solver_nonlinear(self.pb, self.pb.V_v, self.pb.V_p, self.solver_params)

        if self.pb.timint != 'static' and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaP_kin_old + self.pb.deltaP_int_old - self.pb.deltaP_ext_old
            
            jac_a = derivative(weakform_a, self.pb.a_old, self.pb.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            solnln.solve_consistent_ini_acc(weakform_a, jac_a, self.pb.a_old)

        # write mesh output
        self.pb.io.write_output(writemesh=True)


        # fluid main time loop
        for N in range(self.pb.restart_step+1, self.pb.numstep+1):
            
            wts = time.time()
            
            # current time
            t = N * self.pb.dt
            
            # set time-dependent functions
            self.pb.ti.set_time_funcs(self.pb.ti.funcs_to_update, self.pb.ti.funcs_to_update_vec, t)
            
            # solve
            solnln.newton(self.pb.v, self.pb.p)

            # update
            self.pb.ti.update_timestep(self.pb.v, self.pb.v_old, self.pb.a_old, self.pb.p, self.pb.p_old, self.pb.ti.funcs_to_update, self.pb.ti.funcs_to_update_old, self.pb.ti.funcs_to_update_vec, self.pb.ti.funcs_to_update_vec_old)

            # solve time for time step
            wte = time.time()
            wt = wte - wts
            
            # write output and restart info
            self.pb.io.write_output(pb=self.pb, N=N, t=t)

            # print time step info to screen
            self.pb.ti.print_timestep(N, t, wt=wt)
            
            # maximum number of steps to perform
            try:
                if N == self.pb.numstep_stop:
                    break
            except:
                pass


        if self.pb.comm.rank == 0: # only proc 0 should print this
            print('Time for computation: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
            sys.stdout.flush()
