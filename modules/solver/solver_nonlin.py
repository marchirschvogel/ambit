#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
import ufl

from projection import project
from mpiroutines import allgather_vec

from solver_utils import sol_utils
import preconditioner

### useful infos for PETSc mats, vecs, solvers...
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Vec-class.html
# https://www.mcs.anl.gov/petsc/documentation/faq.html
# https://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC-class.html

# standard nonlinear solver for FEM problems
class solver_nonlinear:
    
    def __init__(self, pb, solver_params={}):

        self.pb = pb
        
        # problem variables list
        self.x, self.is_ghosted = self.pb.get_problem_var_list()
        
        self.nfields = len(self.x)
        
        self.ptype = self.pb.problem_physics
        
        self.set_solver_params(solver_params)
        
        self.tolerances = {}
        for n in range(self.nfields):
            if len(self.tolres)>1: # if we have a list here, we need one tolerance per variable!
                self.tolerances['res'+str(n+1)] = self.tolres[n]
                self.tolerances['inc'+str(n+1)] = self.tolinc[n]
            else:
                self.tolerances['res'+str(n+1)] = self.tolres[0]
                self.tolerances['inc'+str(n+1)] = self.tolinc[0]

        self.solutils = sol_utils(self.pb, self.ptype, solver_params)
        self.sepstring = self.solutils.timestep_separator(self.tolerances)
        
        self.initialize_petsc_solver()
        
        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlinear_ode(self.pb.pb0, solver_params['subsolver_params'])
        else:
            self.subsol = None
        

    def set_solver_params(self, solver_params):
        
        try: self.maxiter = solver_params['maxiter']
        except: self.maxiter = 25

        try: self.divcont = solver_params['divergence_continue']
        except: self.divcont = None
        
        try: self.PTC = solver_params['ptc']
        except: self.PTC = False
        
        try: self.k_PTC_initial = solver_params['k_ptc_initial']
        except: self.k_PTC_initial = 0.1

        try: self.PTC_randadapt_range = solver_params['ptc_randadapt_range']
        except: self.PTC_randadapt_range = [0.85, 1.35]
        
        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'superlu_dist'
        
        try: self.adapt_linsolv_tol = solver_params['adapt_linsolv_tol']
        except: self.adapt_linsolv_tol = False
        
        try: self.adapt_factor = solver_params['adapt_factor']
        except: self.adapt_factor = 0.1
        
        try: self.tollin = solver_params['tol_lin']
        except: self.tollin = 1.0e-8

        try: self.maxliniter = solver_params['max_liniter']
        except: self.maxliniter = 1200
        
        try: self.print_local_iter = solver_params['print_local_iter']
        except: self.print_local_iter = False
        
        try: self.tol_res_local = solver_params['tol_res_local']
        except: self.tol_res_local = 1.0e-10
        
        try: self.tol_inc_local = solver_params['tol_inc_local']
        except: self.tol_inc_local = 1.0e-10

        self.solvetype = solver_params['solve_type']
        
        # check if we have a list of tolerances (for coupled problems) or just one value
        self.tolres, self.tolinc = [], []
        if isinstance(solver_params['tol_res'], list):
            for n in range(len(solver_params['tol_res'])):
                self.tolres.append(solver_params['tol_res'][n])
                self.tolinc.append(solver_params['tol_inc'][n])
        else:
            self.tolres.append(solver_params['tol_res'])
            self.tolinc.append(solver_params['tol_inc'])          


    def initialize_petsc_solver(self):
        
        # set forms to use (differ in case of initial prestress)
        self.pb.set_forms_solver()

        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)

        if self.solvetype=='direct':
            
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType(self.direct_solver)
            
        elif self.solvetype=='iterative':
            
            # 2x2 block iterative method
            if self.nfields > 1:

                self.ksp.setType("gmres")
                self.ksp.getPC().setType("fieldsplit")
                # TODO: What is the difference btw. ADDITIVE, MULTIPLICATIVE, SCHUR, SPECIAL?
                self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
                #self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                #self.ksp.getPC().setFieldSplitSchurFactType(0)

                # build "dummy" nested matrix in order to get the nested ISs (index sets)
                locmatsize_u, locmatsize_p = self.pb.pbf.V_v.dofmap.index_map.size_local * self.pb.pbf.V_v.dofmap.index_map_bs, self.pb.pbf.V_p.dofmap.index_map.size_local * self.pb.pbf.V_p.dofmap.index_map_bs
                matsize_u, matsize_p = self.pb.pbf.V_v.dofmap.index_map.size_global * self.pb.pbf.V_v.dofmap.index_map_bs, self.pb.pbf.V_p.dofmap.index_map.size_global * self.pb.pbf.V_p.dofmap.index_map_bs
                K_uu = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_u,matsize_u)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_uu.setUp()
                K_up = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_p,matsize_p)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_up.setUp()                
                K_pu = PETSc.Mat().createAIJ(size=((locmatsize_p,matsize_p),(locmatsize_u,matsize_u)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_pu.setUp()  
                
                K_nest = PETSc.Mat().createNest([[K_uu, K_up], [K_pu, None]], isrows=None, iscols=None, comm=self.pb.comm)
                
                nested_IS = K_nest.getNestISs()
                self.ksp.getPC().setFieldSplitIS(
                    ("u", nested_IS[0][0]),
                    ("p", nested_IS[0][1]))

                # set the preconditioners for each block
                ksp_u, ksp_p = self.ksp.getPC().getFieldSplitSubKSP()
                
                # AMG for displacement/velocity block
                ksp_u.setType("preonly")
                ksp_u.getPC().setType("hypre")
                ksp_u.getPC().setMGLevels(3)
                ksp_u.getPC().setHYPREType("boomeramg")
                
                # AMG for pressure block
                ksp_p.setType("preonly")
                ksp_p.getPC().setType("hypre")
                ksp_p.getPC().setMGLevels(3)
                ksp_p.getPC().setHYPREType("boomeramg")

            else:
                
                # AMG
                self.ksp.getPC().setType("hypre")
                self.ksp.getPC().setMGLevels(3)
                self.ksp.getPC().setHYPREType("boomeramg")
            
            # set tolerances and print routine
            self.ksp.setTolerances(rtol=self.tollin, atol=None, divtol=None, max_it=self.maxliniter)
            self.ksp.setMonitor(lambda ksp, its, rnorm: self.solutils.print_linear_iter(its,rnorm))

        else:
            
            raise NameError("Unknown solvetype!")
        

    # solve for consistent initial acceleration a_old
    def solve_consistent_ini_acc(self, weakform_old, jac_a, a_old):

        # create solver
        ksp = PETSc.KSP().create(self.pb.comm)
        
        if self.solvetype=='direct':
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType(self.direct_solver)
        elif self.solvetype=='iterative':
            ksp.getPC().setType("hypre")
            ksp.getPC().setMGLevels(3)
            ksp.getPC().setHYPREType("boomeramg")
        else:
            raise NameError("Unknown solvetype!")
            
        # solve for consistent initial acceleration a_old
        M_a = fem.petsc.assemble_matrix(fem.form(jac_a), [])
        M_a.assemble()
        
        r_a = fem.petsc.assemble_vector(fem.form(weakform_old))
        r_a.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        ksp.setOperators(M_a)
        ksp.solve(-r_a, a_old.vector)
        
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        ksp.destroy()
    

    def solve_local(self, localdata):
        
        for l in range(len(localdata['var'])):
            self.newton_local(localdata['var'][l],localdata['res'][l],localdata['inc'][l],localdata['fnc'][l])
        

    def newton(self, t, localdata={}):

        # offset array for multi-field systems
        self.offsetarr = [0]     
        off=0
        for n in range(self.nfields):
            if n==0:
                if self.pb.have_rom: # currently, ROM is only implemented for the first variable in the system!
                    off += self.pb.rom.V.getLocalSize()[1]
                else:
                    off += self.x[0].getLocalSize()
            else:
                off += self.x[n].getLocalSize()

            self.offsetarr.append(off)

        del_x, x_start = [[]]*self.nfields, [[]]*self.nfields
      
        for n in range(self.nfields):
            # solution increments for Newton
            del_x[n] = self.x[n].duplicate()
            del_x[n].set(0.0)
            # start vector (needed for reset of Newton in case of divergence)
            x_start[n] = self.x[n].duplicate()
            self.x[n].assemble()
            x_start[n].axpby(1.0, 0.0, self.x[n])

        # Newton iteration index
        it = 0
        # for PTC
        k_PTC = self.k_PTC_initial
        counter_adapt, max_adapt = 0, 50
        maxresval = 1.0e16

        self.solutils.print_nonlinear_iter(header=True)

        while it < self.maxiter and counter_adapt < max_adapt:

            tes = time.time()

            if self.pb.localsolve:
                self.solve_local(localdata)

            r_list, K_list = self.pb.assemble_residual_stiffness(t, subsolver=self.subsol)

            if self.PTC:
                # computes K_uu + k_PTC * I
                K_list[0][0].shift(k_PTC)

            # model order reduction stuff - currently only on first mat in system...
            if self.pb.have_rom and not self.pb.get_presolve_state():
                
                # projection of main block: system matrix, residual, and increment
                tmp = K_list[0][0].matMult(self.pb.rom.V) # K_uu * V
                K_list[0][0] = self.pb.rom.V.transposeMatMult(tmp) # V^T * K_uu * V
                r_u_, del_u_ = self.pb.rom.V.createVecRight(), self.pb.rom.V.createVecRight()
                self.pb.rom.V.multTranspose(r_list[0], r_u_) # V^T * r_u
                # deal with penalties that may be added to reduced residual to penalize certain modes
                if bool(self.pb.rom.redbasisvec_penalties):
                    u_ = K_list[0][0].createVecRight()
                    self.pb.rom.V.multTranspose(self.x[0], u_) # V^T * u
                    penterm_ = self.pb.rom.V.createVecRight()
                    self.pb.rom.Cpen.mult(u_, penterm_) # Cpen * V^T * u
                    r_u_.axpy(1.0, penterm_) # add penalty term to reduced residual
                    K_list[0][0].aypx(1.0, self.pb.rom.CpenVTV) # K_uu + Cpen * V^T * V
                r_list[0], del_x[0] = r_u_, del_u_
                # now the offdiagonal blocks
                if self.nfields > 1:
                    for n in range(self.nfields-1):
                        if K_list[0][n+1] is not None:
                            K_list[0][n+1] = self.pb.rom.V.transposeMatMult(K_list[0][n+1]) # V^T * K_{0,n+1}
                        if K_list[n+1][0] is not None:
                            K_list[n+1][0] = K_list[n+1][0].matMult(self.pb.rom.V) # K_{n+1,0} * V

            te = time.time() - tes

            # we use a block matrix (either with merge-into-one or for a nested iterative solver) if we have more than one field
            if self.nfields > 1:
                
                tes = time.time()

                # nested u-p vector
                r_full_nest = PETSc.Vec().createNest(r_list)
                
                # nested uu-up,pu-zero matrix
                K_full_nest = PETSc.Mat().createNest(K_list, isrows=None, iscols=None, comm=self.pb.comm)
                K_full_nest.assemble()

                te += time.time() - tes
                
                # for monolithic direct solver
                if self.solvetype=='direct':
                    
                    tes = time.time()
                    
                    K_full = PETSc.Mat()
                    K_full_nest.convert("aij", out=K_full)
            
                    K_full.assemble()
                    
                    r_full = PETSc.Vec().createWithArray(r_full_nest.getArray())
                    r_full.assemble()

                    del_full = K_full.createVecLeft()
                    self.ksp.setOperators(K_full)
                    te += time.time() - tes
                    
                    tss = time.time()
                    self.ksp.solve(-r_full, del_full)
                    ts = time.time() - tss
                
                # for nested iterative solver
                elif self.solvetype=='iterative': # TODO: Extend to n-x-n!

                    tes = time.time()

                    P_pp = fem.petsc.assemble_matrix(fem.form(self.pb.a_p11), [])
                    P = PETSc.Mat().createNest([[K_list[0][0], None], [None, P_pp]])
                    P.assemble()

                    del_full = PETSc.Vec().createNest([del_u, del_p])
                    self.ksp.setOperators(K_full_nest, P)
                    
                    te += time.time() - tes
                    
                    tss = time.time()
                    self.ksp.solve(-r_full_nest, del_full)
                    ts = time.time() - tss
                    
                    self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())
                    
                    if self.adapt_linsolv_tol:
                        self.solutils.adapt_linear_solver(r_list[0].norm())

                else:
                    
                    raise NameError("Unknown solvetype!")

                for n in range(self.nfields):
                    del_x[n].array[:] = del_full.array_r[self.offsetarr[n]:self.offsetarr[n+1]]
                
            else:
                    
                # solve linear system
                self.ksp.setOperators(K_list[0][0])
                
                tss = time.time()
                self.ksp.solve(-r_list[0], del_x[0])
                ts = time.time() - tss
            
                if self.solvetype=='iterative':
                    
                    self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())
                        
                    if self.adapt_linsolv_tol:
                        self.solutils.adapt_linear_solver(r_list[0].norm())
            
            # get residual and increment norms
            resnorms, incnorms = {}, {}
            for n in range(self.nfields):
                resnorms['res'+str(n+1)] = r_list[n].norm()
                incnorms['inc'+str(n+1)] = del_x[n].norm()

            # reconstruct full-length increment vector - currently only for first var!
            if self.pb.have_rom and not self.pb.get_presolve_state():
                del_x[0] = self.pb.rom.V.createVecLeft()
                self.pb.rom.V.mult(del_u_, del_x[0]) # V * dx_red

            # update variables
            for n in range(self.nfields):
                self.x[n].axpy(1.0, del_x[n])
                if self.is_ghosted[n]: self.x[n].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.solutils.print_nonlinear_iter(it,resnorms,incnorms,self.PTC,k_PTC,ts=ts,te=te)
            
            it += 1
            
            # for PTC - only applied to first main block so far...
            if self.PTC and it > 1 and res_norm_main_last > 0.: k_PTC *= resnorms['res1']/res_norm_main_last
            res_norm_main_last = resnorms['res1']
            
            # adaptive PTC (for 3D block K_uu only!)
            if self.divcont=='PTC':
                
                self.maxiter = 250
                err = self.solutils.catch_solver_errors(resnorms['res1'], incnorm=incnorms['inc1'], maxval=maxresval)
                
                if err:
                    self.PTC = True
                    # reset Newton step
                    it, k_PTC = 0, self.k_PTC_initial
                    k_PTC *= np.random.uniform(self.PTC_randadapt_range[0], self.PTC_randadapt_range[1])
                    for n in range(self.nfields):
                        self.reset_step(self.x[n], x_start[n], self.is_ghosted[n])
                    counter_adapt += 1
            
            # check if converged
            converged = self.solutils.check_converged(resnorms,incnorms,self.tolerances)
            if converged:
                if self.divcont=='PTC':
                    self.PTC = False
                    counter_adapt = 0
                break
        
        else:

            raise RuntimeError("Newton did not converge after %i iterations!" % (it))


    def reset_step(self, vec, vec_start, ghosted):
        
        vec.axpby(1.0, 0.0, vec_start)
        
        if ghosted:
            vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # local Newton where increment can be expressed as form at integration point level
    def newton_local(self, var, residual_forms, increment_forms, functionspaces, maxiter_local=20):

        it_local = 0
        
        num_loc_res = len(residual_forms)
        
        residuals, increments = [], []
        
        for i in range(num_loc_res):
            residuals.append(fem.Function(functionspaces[i]))
            increments.append(fem.Function(functionspaces[i]))

        res_norms, inc_norms = np.ones(num_loc_res), np.ones(num_loc_res)

        # return mapping scheme for nonlinear constitutive laws
        while it_local < maxiter_local:

            for i in range(num_loc_res):
                
                # interpolate symbolic increment form into increment vector
                increment_proj = project(increment_forms[i], functionspaces[i], self.pb.dx_)
                increments[i].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                increments[i].interpolate(increment_proj)
                
            for i in range(num_loc_res):
                # update var vector
                var[i].vector.axpy(1.0, increments[i].vector)
                var[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            for i in range(num_loc_res):
                # interpolate symbolic residual form into residual vector
                residual_proj = project(residual_forms[i], functionspaces[i], self.pb.dx_)
                residuals[i].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                residuals[i].interpolate(residual_proj)
                # get residual and increment inf norms
                res_norms[i] = residuals[i].vector.norm(norm_type=3)
                inc_norms[i] = increments[i].vector.norm(norm_type=3)
            
            if self.print_local_iter:
                if self.pb.comm.rank == 0:
                    print("      (it_local = %i, res: %.4e, inc: %.4e)" % (it_local,np.sum(res_norms),np.sum(inc_norms)))
                    sys.stdout.flush()
            
            # increase iteration index
            it_local += 1
            
            # check if converged
            if np.sum(res_norms) <= self.tol_res_local and np.sum(inc_norms) <= self.tol_inc_local:

                break
            
        else:

            raise RuntimeError("Local Newton did not converge after %i iterations!" % (it_local))



# solver for pure ODE (0D) problems (e.g. a system of first order ODEs integrated with One-Step-Theta method)
class solver_nonlinear_ode(solver_nonlinear):

    def __init__(self, pb, solver_params={}):

        self.pb = pb
        
        self.ptype = self.pb.problem_physics

        try: self.maxiter = solver_params['maxiter']
        except: self.maxiter = 25
        
        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'superlu_dist'        

        self.tolres = solver_params['tol_res']
        self.tolinc = solver_params['tol_inc']

        self.tolerances = {'res_0d' : self.tolres, 'inc_0d' : self.tolinc}
        
        self.PTC = False # don't think we'll ever need PTC for the 0D ODE problem...

        self.solutils = sol_utils(self.pb, self.ptype, solver_params)
        
        self.sepstring = self.solutils.timestep_separator(self.tolerances)
        
        self.initialize_petsc_solver()
        
        
    def initialize_petsc_solver(self):
        
        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType(self.direct_solver)


    def newton(self, t, print_iter=True, sub=False):

        # Newton iteration index
        it = 0
        
        if print_iter: self.solutils.print_nonlinear_iter(header=True,sub=sub)
        
        while it < self.maxiter:
            
            tes = time.time()

            self.pb.odemodel.evaluate(self.pb.s, t, self.pb.df, self.pb.f, self.pb.dK, self.pb.K, self.pb.c, self.pb.y, self.pb.aux)
            
            # ODE rhs vector and stiffness matrix
            r, K = self.pb.assemble_residual_stiffness(t)

            # if we have prescribed variable values over time
            if bool(self.pb.prescribed_variables):
                for a in self.pb.prescribed_variables:
                    varindex = self.pb.odemodel.varmap[a]
                    curvenumber = self.pb.prescribed_variables[a]
                    val = self.pb.ti.timecurves(curvenumber)(t)
                    self.pb.odemodel.set_prescribed_variables(self.pb.s, r, K, val, varindex)
            
            ds = K.createVecLeft()
            
            # solve linear system
            self.ksp.setOperators(K)
            
            te = time.time() - tes
            
            tss = time.time()
            self.ksp.solve(-r, ds)
            ts = time.time() - tss
            
            # update solution
            self.pb.s.axpy(1.0, ds)
            
            # get norms
            res_norm = r.norm()
            inc_norm = ds.norm()
            
            if print_iter: self.solutils.print_nonlinear_iter(it,{'res_0d' : res_norm},{'inc_0d' : inc_norm},ts=ts,te=te,sub=sub)
            
            it += 1

            # check if converged
            converged = self.solutils.check_converged({'res_0d' : res_norm},{'inc_0d' : inc_norm},self.tolerances,ptype='flow0d')
            if converged:
                if print_iter and sub:
                    if self.pb.comm.rank == 0:
                        print('      **************************************************************')
                        print(' ')
                        sys.stdout.flush()
                break

        else:

            raise RuntimeError("Newton for ODE system did not converge after %i iterations!" % (it))
