#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time

import numpy as np
from petsc4py import PETSc
from dolfinx import fem

from .projection import project
from .solver_utils import sol_utils
from . import preconditioner
from .. import ioparams

### useful infos for PETSc mats, vecs, solvers...
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Vec-class.html
# https://www.mcs.anl.gov/petsc/documentation/faq.html
# https://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP-class.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC-class.html

# standard nonlinear solver for FEM problems
class solver_nonlinear:

    def __init__(self, pb, solver_params, subsolver=None, cp=None):

        ioparams.check_params_solver(solver_params)

        self.comm = pb[0].comm

        self.pb = pb
        self.nprob = len(pb)

        self.solver_params = solver_params

        self.printenh = self.pb[0].print_enhanced_info

        self.x, self.is_ghosted = [[]]*self.nprob, [[]]*self.nprob
        self.nfields, self.ptype = [], []

        # problem variables list
        for npr in range(self.nprob):
            self.x[npr], self.is_ghosted[npr] = self.pb[npr].get_problem_var_list()
            self.nfields.append(self.pb[npr].nfields)
            self.ptype.append(self.pb[npr].problem_physics)

        self.set_solver_params(self.solver_params)

        # list of dicts for tolerances, residual, and increment norms
        self.tolerances, self.resnorms, self.incnorms = [], [], []
        # caution: [{}]*self.nprob actually would produce a list of dicts with same reference!
        # so do it like this...
        for npr in range(self.nprob):
            self.tolerances.append({})
            self.resnorms.append({})
            self.incnorms.append({})

        # set tolerances required by the user - may be a scalar, list, or list of lists
        for npr in range(self.nprob):
            for n in range(self.nfields[npr]):
                if isinstance(self.tolres, list):
                    if isinstance(self.tolres[npr], list):
                        self.tolerances[npr]['res'+str(n+1)] = self.tolres[npr][n]
                        self.tolerances[npr]['inc'+str(n+1)] = self.tolinc[npr][n]
                    else:
                        self.tolerances[npr]['res'+str(n+1)] = self.tolres[n]
                        self.tolerances[npr]['inc'+str(n+1)] = self.tolinc[n]
                else:
                    self.tolerances[npr]['res'+str(n+1)] = self.tolres
                    self.tolerances[npr]['inc'+str(n+1)] = self.tolinc

        self.solutils = sol_utils(self)
        self.lsp = self.solutils.timestep_separator_len()

        if self.nprob>1:
            self.indp2 = self.lsp+2 # indent length for partitioned problem print
            self.lsp += 54

        self.indlen_ = [1]*self.nprob
        # currently, we only have setup the output for two partitioned problems (will there ever be cases with more...?)
        if self.nprob>1: self.indlen_[1] = self.indp2

        self.r_list_sol = [[]]*self.nprob
        self.K_list_sol = [[]]*self.nprob

        for npr in range(self.nprob):
            if self.pb[npr].rom:
                self.r_list_sol[npr] = self.pb[npr].r_list_rom
                self.K_list_sol[npr] = self.pb[npr].K_list_rom
            else:
                self.r_list_sol[npr] = self.pb[npr].r_list
                self.K_list_sol[npr] = self.pb[npr].K_list

        # nested and merged (monolithic) residual and matrix objects
        self.r_full_nest = [None]*self.nprob
        self.K_full_nest = [None]*self.nprob
        self.P_full_nest = [None]*self.nprob

        self.r_full_merged = [None]*self.nprob
        self.K_full_merged = [None]*self.nprob
        self.P_full_merged = [None]*self.nprob

        self.subsol = subsolver

        # offset array for multi-field systems
        self.offsetarr = [[]]*self.nprob
        for npr in range(self.nprob):

            self.offsetarr[npr] = [0]
            off=0
            for n in range(self.nfields[npr]):
                if n==0:
                    if self.pb[npr].rom: # currently, ROM is only implemented for the first variable in the system!
                        off += self.pb[npr].rom.V.getLocalSize()[1]
                    else:
                        off += self.x[npr][0].getLocalSize()
                else:
                    off += self.x[npr][n].getLocalSize()

                self.offsetarr[npr].append(off)

        self.del_x, self.del_x_rom, self.x_start = [], [], []
        self.del_x_sol = [[]]*self.nprob

        for npr in range(self.nprob):

            self.del_x.append([[]]*self.nfields[npr])
            self.x_start.append([[]]*self.nfields[npr])
            self.del_x_rom.append([[]]*self.nfields[npr])

        for npr in range(self.nprob):

            for n in range(self.nfields[npr]):
                # solution increments for Newton
                self.del_x[npr][n] = self.x[npr][n].duplicate()
                self.del_x[npr][n].set(0.0)
                if self.pb[npr].rom and npr==0:
                    if n==0:
                        self.del_x_rom[npr][n] = self.pb[npr].rom.V.createVecRight()
                        self.del_x_rom[npr][n].set(0.0)
                    else:
                        self.del_x_rom[npr][n] = self.del_x[npr][n]
                # start vector (needed for reset of Newton in case of divergence)
                self.x_start[npr][n] = self.x[npr][n].duplicate()
                self.x[npr][n].assemble()
                if self.pb[npr].sub_solve: # can only be a 0D model so far...
                    self.s_start = self.pb[npr].pb0.s.duplicate()
                    self.pb[npr].pb0.s.assemble()

            if self.pb[npr].rom:
                self.del_x_sol[npr] = self.del_x_rom[npr]
            else:
                self.del_x_sol[npr] = self.del_x[npr]

        self.initialize_petsc_solver()

        self.li_s = [] # linear iterations over all solves
        self.ni_all = 0 # all nonlinear iterations over all solves (to determine prec updates for instance)

        self.cp = cp


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

        try: self.maxresval = solver_params['catch_max_res_value']
        except: self.maxresval = 1e16

        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'mumps'

        try: self.iterative_solver = solver_params['iterative_solver']
        except: self.iterative_solver = 'gmres'

        try: precond_fields = solver_params['precond_fields']
        except: precond_fields = [[]]

        self.precond_fields = [[]]*self.nprob
        for npr in range(self.nprob):
            if isinstance(precond_fields[npr], list):
                self.precond_fields[npr] = precond_fields[npr]
            else:
                self.precond_fields[npr] = precond_fields

        try: self.fieldsplit_type = solver_params['fieldsplit_type']
        except: self.fieldsplit_type = 'jacobi'

        try: block_precond = solver_params['block_precond']
        except: block_precond = 'fieldsplit'

        self.block_precond = [[]]*self.nprob
        for npr in range(self.nprob):
            if isinstance(block_precond, list):
                self.block_precond[npr] = block_precond[npr]
            else:
                self.block_precond[npr] = block_precond

        try: self.tol_lin_rel = solver_params['tol_lin_rel']
        except: self.tol_lin_rel = 1.0e-5

        try: self.tol_lin_abs = solver_params['tol_lin_abs']
        except: self.tol_lin_abs = 1.0e-50

        try: self.res_lin_monitor = solver_params['res_lin_monitor']
        except: self.res_lin_monitor = 'rel'

        try: self.maxliniter = solver_params['max_liniter']
        except: self.maxliniter = 1200

        try: self.lin_norm_type = solver_params['lin_norm_type']
        except: self.lin_norm_type = 'unpreconditioned'

        # cf. https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP.NormType-class.html
        if self.lin_norm_type=='preconditioned':
            self.linnormtype = PETSc.KSP.NormType.NORM_PRECONDITIONED
        elif self.lin_norm_type=='unpreconditioned':
            self.linnormtype = PETSc.KSP.NormType.NORM_UNPRECONDITIONED
        else:
            raise ValueError("Unknown lin_norm_type option!")

        try: self.print_liniter_every = solver_params['print_liniter_every']
        except: self.print_liniter_every = 1

        try: self.iset_options = solver_params['indexset_options']
        except: self.iset_options = {}
        is_option_keys = ['lms_to_p','lms_to_v','rom_to_new','ale_to_v']
        # revert to defaults if not set by the user
        for k in is_option_keys:
            if k not in self.iset_options.keys(): self.iset_options[k] = False

        if any(list(self.iset_options.values())):
            self.merge_prec_mat = True
        else:
            self.merge_prec_mat = False

        self.iset = [[]]*self.nprob

        try: self.print_local_iter = solver_params['print_local_iter']
        except: self.print_local_iter = False

        try: self.rebuild_prec_every_it = solver_params['rebuild_prec_every_it']
        except: self.rebuild_prec_every_it = 1

        try: self.tol_res_local = solver_params['tol_res_local']
        except: self.tol_res_local = 1.0e-10

        try: self.tol_inc_local = solver_params['tol_inc_local']
        except: self.tol_inc_local = 1.0e-10

        self.solvetype = [[]]*self.nprob
        for npr in range(self.nprob):
            if isinstance(solver_params['solve_type'], list):
                self.solvetype[npr] = solver_params['solve_type'][npr]
            else:
                self.solvetype[npr] = solver_params['solve_type']

        self.tolres = solver_params['tol_res']
        self.tolinc = solver_params['tol_inc']


    def initialize_petsc_solver(self):

        self.ksp = [[]]*self.nprob

        for npr in range(self.nprob):

            # create solver
            self.ksp[npr] = PETSc.KSP().create(self.comm)

            # perform initial matrix and residual reduction to set the correct arrays
            if self.pb[npr].rom:
                self.pb[npr].rom.reduce_stiffness(self.pb[npr].K_list, self.pb[npr].K_list_rom, self.pb[npr].K_list_tmp)
                self.pb[npr].rom.reduce_residual(self.pb[npr].r_list, self.pb[npr].r_list_rom, x=self.x[npr][0])

            # create nested matrix and residual structures
            self.K_full_nest[npr] = PETSc.Mat().createNest(self.K_list_sol[npr], isrows=None, iscols=None, comm=self.comm)
            self.r_full_nest[npr] = PETSc.Vec().createNest(self.r_list_sol[npr])

            if self.solvetype[npr]=='direct':

                self.ksp[npr].setType("preonly")
                self.ksp[npr].getPC().setType("lu")
                self.ksp[npr].getPC().setFactorSolverType(self.direct_solver)

                # prepare merged matrix structure
                if self.solvetype[npr]=='direct' and self.nfields[npr] > 1:

                    Kfullnesttmp = self.K_full_nest[npr].duplicate(copy=False)
                    self.K_full_merged[npr] = Kfullnesttmp.convert("aij")
                    # solution increment
                    self.del_full = self.K_full_merged[npr].createVecLeft()

                    self.r_arr = np.zeros(self.r_full_nest[npr].getLocalSize())
                    self.r_full_merged[npr] = PETSc.Vec().createWithArray(self.r_arr)

            elif self.solvetype[npr]=='iterative':

                self.ksp[npr].setInitialGuessNonzero(False)
                self.ksp[npr].setNormType(self.linnormtype)

                self.P_full_nest[npr] = self.K_full_nest[npr]

                # solution increment
                if self.nfields[npr] > 1:
                    if not self.block_precond[npr] == 'fieldsplit':
                        self.del_full = self.K_full_nest[npr].createVecLeft()
                        self.r_arr = np.zeros(self.r_full_nest[npr].getLocalSize())
                        self.r_full_merged[npr] = PETSc.Vec().createWithArray(self.r_arr)
                    else:
                        self.del_full = PETSc.Vec().createNest(self.del_x_sol[npr])

                # prepare merged preconditioner matrix structure
                if self.merge_prec_mat:
                    Pfullnesttmp = self.P_full_nest[npr].duplicate(copy=False)
                    self.P_full_merged[npr] = Pfullnesttmp.convert("aij")
                    self.P = self.P_full_merged[npr]
                else:
                    self.P = self.P_full_nest[npr]

                self.ksp[npr].setOperators(self.K_full_nest[npr], self.P)

                # block iterative method
                if self.nfields[npr] > 1:

                    self.iset[npr] = self.pb[npr].get_index_sets(isoptions=self.iset_options)

                    self.ksp[npr].setType(self.iterative_solver) # cf. https://petsc.org/release/petsc4py/petsc4py.PETSc.KSP.Type-class.html

                    if self.block_precond[npr] == 'fieldsplit':

                        # see e.g. https://petsc.org/main/manual/ksp/#sec-block-matrices
                        self.ksp[npr].getPC().setType("fieldsplit")
                        # cf. https://petsc.org/main/manualpages/PC/PCCompositeType

                        if self.fieldsplit_type=='jacobi':
                            splittype = PETSc.PC.CompositeType.ADDITIVE # block Jacobi
                        elif self.fieldsplit_type=='gauss_seidel':
                            splittype = PETSc.PC.CompositeType.MULTIPLICATIVE # block Gauss-Seidel
                        elif self.fieldsplit_type=='gauss_seidel_sym':
                            splittype = PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE # symmetric block Gauss-Seidel
                        elif self.fieldsplit_type=='schur':
                            assert(self.nfields[npr]==2)
                            splittype = PETSc.PC.CompositeType.SCHUR # block Schur - for 2x2 block systems only
                        else:
                            raise ValueError("Unknown fieldsplit_type option.")

                        self.ksp[npr].getPC().setFieldSplitType(splittype)

                        nsets = len(self.iset[npr])

                        # normally, nsets = self.nfields, but for a surface-projected ROM (FrSI) problem, we have one more index set than fields
                        if nsets==2:   self.ksp[npr].getPC().setFieldSplitIS(("f1", self.iset[npr][0]),("f2", self.iset[npr][1]))
                        elif nsets==3: self.ksp[npr].getPC().setFieldSplitIS(("f1", self.iset[npr][0]),("f2", self.iset[npr][1]),("f3", self.iset[npr][2]))
                        elif nsets==4: self.ksp[npr].getPC().setFieldSplitIS(("f1", self.iset[npr][0]),("f2", self.iset[npr][1]),("f3", self.iset[npr][2]),("f4", self.iset[npr][3]))
                        elif nsets==5: self.ksp[npr].getPC().setFieldSplitIS(("f1", self.iset[npr][0]),("f2", self.iset[npr][1]),("f3", self.iset[npr][2]),("f4", self.iset[npr][3]),("f5", self.iset[npr][4]))
                        else: raise RuntimeError("Currently, no more than 5 fields/index sets are supported.")

                        # get the preconditioners for each block
                        self.ksp[npr].getPC().setUp()
                        ksp_fields = self.ksp[npr].getPC().getFieldSplitSubKSP()

                        assert(nsets==len(self.precond_fields[npr])) # sanity check

                        # set field-specific preconditioners
                        for n in range(nsets):

                            pc_field = ksp_fields[n].getPC()
                            if self.precond_fields[npr][n]['prec'] == 'amg':
                                try: solvetype = self.precond_fields[npr][n]['solve']
                                except: solvetype = "preonly"
                                ksp_fields[n].setType(solvetype)
                                try: amgtype = self.precond_fields[npr][n]['amgtype']
                                except: amgtype = "hypre"
                                pc_field.setType(amgtype)
                                if amgtype=="hypre":
                                    pc_field.setHYPREType("boomeramg")
                            elif self.precond_fields[npr][n]['prec'] == 'direct':
                                ksp_fields[n].setType("preonly")
                                pc_field.setType("lu")
                                pc_field.setFactorSolverType("mumps")
                            else:
                                raise ValueError("Currently, only either 'amg' or 'direct' are supported as field-specific preconditioner.")

                    elif self.block_precond[npr] == 'schur2x2':

                        self.ksp[npr].getPC().setType(PETSc.PC.Type.PYTHON)
                        bj = preconditioner.schur_2x2(self.iset[npr], self.precond_fields[npr], self.printenh, self.solver_params, self.comm)
                        self.ksp[npr].getPC().setPythonContext(bj)

                    elif self.block_precond[npr] == 'simple2x2':

                        self.ksp[npr].getPC().setType(PETSc.PC.Type.PYTHON)
                        bj = preconditioner.simple_2x2(self.iset[npr], self.precond_fields[npr], self.printenh, self.solver_params, self.comm)
                        self.ksp[npr].getPC().setPythonContext(bj)

                    elif self.block_precond[npr] == 'schur3x3':

                        self.ksp[npr].getPC().setType(PETSc.PC.Type.PYTHON)
                        bj = preconditioner.schur_3x3(self.iset[npr], self.precond_fields[npr], self.printenh, self.solver_params, self.comm)
                        self.ksp[npr].getPC().setPythonContext(bj)

                    elif self.block_precond[npr] == 'schur4x4':

                        self.ksp[npr].getPC().setType(PETSc.PC.Type.PYTHON)
                        bj = preconditioner.schur_4x4(self.iset[npr], self.precond_fields[npr], self.printenh, self.solver_params, self.comm)
                        self.ksp[npr].getPC().setPythonContext(bj)

                    elif self.block_precond[npr] == 'bgs2x2': # can also be called via PETSc's fieldsplit

                        self.ksp[npr].getPC().setType(PETSc.PC.Type.PYTHON)
                        bj = preconditioner.bgs_2x2(self.iset[npr], self.precond_fields[npr], self.printenh, self.solver_params, self.comm)
                        self.ksp[npr].getPC().setPythonContext(bj)

                    elif self.block_precond[npr] == 'jacobi2x2': # can also be called via PETSc's fieldsplit

                        self.ksp[npr].getPC().setType(PETSc.PC.Type.PYTHON)
                        bj = preconditioner.jacobi_2x2(self.iset[npr], self.precond_fields[npr], self.printenh, self.solver_params, self.comm)
                        self.ksp[npr].getPC().setPythonContext(bj)

                    else:
                        raise ValueError("Unknown block_precond option!")

                else:

                    if self.precond_fields[npr][0]['prec'] == 'amg':
                        self.ksp[npr].getPC().setType("hypre")
                        self.ksp[npr].getPC().setMGLevels(3)
                        self.ksp[npr].getPC().setHYPREType("boomeramg")
                    else:
                        raise ValueError("Currently, only 'amg' is supported as single-field preconditioner.")

                # set tolerances and print routine
                self.ksp[npr].setTolerances(rtol=self.tol_lin_rel, atol=self.tol_lin_abs, divtol=None, max_it=self.maxliniter)
                self.ksp[npr].setMonitor(lambda ksp, its, rnorm: self.solutils.print_linear_iter(its, rnorm))

                # set some additional PETSc options
                petsc_options = PETSc.Options()
                petsc_options.setValue('-ksp_gmres_modifiedgramschmidt', True)
                self.ksp[npr].setFromOptions()

            else:

                raise NameError("Unknown solvetype!")


    # solve for consistent initial acceleration a_old
    def solve_consistent_ini_acc(self, res_a, jac_aa, a_old):

        tss = time.time()
        if self.comm.rank == 0:
            print('Solve for consistent initial acceleration...')
            sys.stdout.flush()

        # create solver
        ksp = PETSc.KSP().create(self.comm)

        if self.solvetype[0]=='direct':
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType(self.direct_solver)
        elif self.solvetype[0]=='iterative':
            ksp.setType(self.iterative_solver)
            ksp.getPC().setType("hypre")
            ksp.getPC().setMGLevels(3)
            ksp.getPC().setHYPREType("boomeramg")
        else:
            raise NameError("Unknown solvetype!")

        # solve for consistent initial acceleration a_old
        M_a = fem.petsc.assemble_matrix(jac_aa, [])
        M_a.assemble()

        r_a = fem.petsc.assemble_vector(res_a)
        r_a.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        ksp.setOperators(M_a)
        ksp.solve(-r_a, a_old.vector)

        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        r_a.destroy(), M_a.destroy()
        ksp.destroy()

        tse = time.time() - tss
        if self.comm.rank == 0:
            print('Solve for consistent initial acceleration finished, ts = %.4f s' % (tse))
            sys.stdout.flush()


    def solve_local(self, localdata):

        for l in range(len(localdata['var'])):
            self.newton_local(localdata['var'][l],localdata['res'][l],localdata['inc'][l],localdata['fnc'][l])


    def newton(self, t, localdata={}):

        # set start vectors
        for npr in range(self.nprob):
            for n in range(self.nfields[npr]):
                self.x_start[npr][n].axpby(1.0, 0.0, self.x[npr][n])
                if self.pb[npr].sub_solve: # can only be a 0D model so far...
                    self.s_start.axpby(1.0, 0.0, self.pb[npr].pb0.s)

        # Newton iteration index
        it = 0
        # for PTC
        k_PTC = self.k_PTC_initial
        counter_adapt, max_adapt = 0, 10
        self.ni, self.li = 0, 0 # nonlinear and linear iteration counters

        for npr in range(self.nprob):

            if npr==0: self.indlen=self.indlen_[0]
            else: self.indlen=self.indlen_[1]

            self.solutils.print_nonlinear_iter(header=True, ptype=self.ptype[npr])

            tes = time.time()

            # initial redidual actions due to predictor
            self.residual_problem_actions(t, npr, localdata)

            te = time.time() - tes

            self.solutils.print_nonlinear_iter(it, resnorms=self.resnorms[npr], te=te, ptype=self.ptype[npr])

        it += 1

        while it < self.maxiter and counter_adapt < max_adapt:

            converged, err = [], []

            # problem loop (in case of partitioned solves)
            for npr in range(self.nprob):

                tes = time.time()

                if npr==0: self.indlen=self.indlen_[0]
                else: self.indlen=self.indlen_[1]

                # assemble Jacobian
                self.jacobian_problem_actions(t, npr, k_PTC)

                te = time.time() - tes

                # we use a block matrix (either with merge-into-one or for a nested iterative solver) if we have more than one field
                if self.nfields[npr] > 1:

                    tes = time.time()

                    # nested residual and matrix references have been updated - we should call assemble again (?)
                    self.r_full_nest[npr].assemble()
                    self.K_full_nest[npr].assemble()

                    te += time.time() - tes

                    # for monolithic direct solver
                    if self.solvetype[npr]=='direct':

                        tes = time.time()

                        self.K_full_nest[npr].convert("aij", out=self.K_full_merged[npr])
                        tme = time.time() - tes
                        if self.printenh:
                            if self.comm.rank == 0:
                                print(' '*self.indlen_[npr] + '      === MAT merge, te = %.4f s' % (tme))
                                sys.stdout.flush()
                        tmp=self.r_full_nest[npr].getArray(readonly=True)

                        self.r_arr[:] = self.r_full_nest[npr].getArray(readonly=True)
                        self.r_full_merged[npr].placeArray(self.r_arr)

                        self.ksp[npr].setOperators(self.K_full_merged[npr])
                        te += time.time() - tes

                        tss = time.time()
                        self.ksp[npr].solve(-self.r_full_merged[npr], self.del_full)
                        ts = time.time() - tss

                        self.r_full_merged[npr].resetArray()

                    # for nested iterative solver
                    elif self.solvetype[npr]=='iterative':

                        tes = time.time()

                        # re-build preconditioner if requested (default is every iteration)
                        if self.ni_all % self.rebuild_prec_every_it == 0:

                            self.ksp[npr].getPC().setReusePreconditioner(False)

                            # use same matrix as preconditioner
                            self.P_full_nest[npr] = self.K_full_nest[npr]

                            # if index sets do not align with the nested matrix structure
                            # anymore, we need a merged matrix to extract the submats
                            if self.merge_prec_mat:
                                tms = time.time()
                                self.P_full_nest[npr].convert("aij", out=self.P_full_merged[npr])
                                self.P = self.P_full_merged[npr]
                                tme = time.time() - tms
                                if self.printenh:
                                    if self.comm.rank == 0:
                                        print(' '*self.indlen_[npr] + '      === PREC MAT merge, te = %.4f s' % (tme))
                                        sys.stdout.flush()
                            else:
                                self.P = self.P_full_nest[npr]

                        else:

                            self.ksp[npr].getPC().setReusePreconditioner(True)

                        # set operators for linear system solve: Jacobian and preconditioner (we use the same here)
                        self.ksp[npr].setOperators(self.K_full_nest[npr], self.P)

                        # need to merge for non-fieldsplit-type preconditioners
                        if not self.block_precond[npr] == 'fieldsplit':
                            self.r_arr[:] = self.r_full_nest[npr].getArray(readonly=True)
                            self.r_full_merged[npr].placeArray(self.r_arr)
                            r = self.r_full_merged[npr]
                        else:
                            r = self.r_full_nest[npr]

                        te += time.time() - tes

                        tss = time.time()
                        # solve the linear system
                        self.ksp[npr].solve(-r, self.del_full)

                        ts = time.time() - tss

                        self.solutils.print_linear_iter_last(self.ksp[npr].getIterationNumber(), self.ksp[npr].getResidualNorm(), self.ksp[npr].getConvergedReason())

                        if not self.block_precond[npr] == 'fieldsplit':
                            self.r_full_merged[npr].resetArray()

                    else:

                        raise NameError("Unknown solvetype!")

                    for n in range(self.nfields[npr]):
                        self.del_x_sol[npr][n].array[:] = self.del_full.array_r[self.offsetarr[npr][n]:self.offsetarr[npr][n+1]]

                else:

                    # set operators for linear system solve
                    self.ksp[npr].setOperators(self.K_list_sol[npr][0][0])

                    tss = time.time()
                    # solve the linear system
                    self.ksp[npr].solve(-self.r_list_sol[npr][0], self.del_x_sol[npr][0])
                    ts = time.time() - tss

                    if self.solvetype[npr]=='iterative':
                        self.solutils.print_linear_iter_last(self.ksp[npr].getIterationNumber(), self.ksp[npr].getResidualNorm(), self.ksp[npr].getConvergedReason())

                tes = time.time()

                # get increment norm
                for n in range(self.nfields[npr]):
                    self.incnorms[npr]['inc'+str(n+1)] = self.del_x_sol[npr][n].norm()

                # reconstruct full-length increment vector
                if self.pb[npr].rom:
                    self.pb[npr].rom.reconstruct_solution_increment(self.del_x_sol[npr], self.del_x[npr])

                # norm from last step for potential PTC adaption - prior to res update
                res_norm_main_last = self.resnorms[npr]['res1']

                # update variables
                for n in range(self.nfields[npr]):
                    self.x[npr][n].axpy(1.0, self.del_x[npr][n])
                    if self.is_ghosted[npr][n]==1:
                        self.x[npr][n].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    if self.is_ghosted[npr][n]==2:
                        subvecs = self.x[npr][n].getNestSubVecs()
                        for j in range(len(subvecs)): subvecs[j].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                # compute new residual actions after updated solution
                self.residual_problem_actions(t, npr, localdata)

                # for partitioned solves, we now have to update all dependent other residuals, too
                if self.nprob > 1:
                    for mpr in range(self.nprob):
                        if mpr!=npr:
                            self.residual_problem_actions(t, mpr, localdata)

                te += time.time() - tes

                self.solutils.print_nonlinear_iter(it, resnorms=self.resnorms[npr], incnorms=self.incnorms[npr], ts=ts, te=te, ptype=self.ptype[npr])

                # get converged state of each problem
                converged.append(self.solutils.check_converged(self.resnorms[npr], self.incnorms[npr], self.tolerances[npr], ptype=self.ptype[npr]))

                # for PTC - scale k_PTC with ratio of current to previous residual norm
                if self.PTC:
                    k_PTC *= self.resnorms[npr]['res1']/res_norm_main_last

                # adaptive PTC (for 3D block K_00 only!)
                if self.divcont=='PTC':

                    self.maxiter = 100 # should be enough...

                    # collect errors
                    err.append(self.solutils.catch_solver_errors(self.resnorms[npr]['res1'], incnorm=self.incnorms[npr]['inc1'], maxval=self.maxresval))

            # iteration update after all problems have been solved
            it += 1
            self.ni_all += 1

            # now check if errors occurred
            if any(err):

                self.PTC = True
                # reset Newton step
                it, k_PTC = 1, self.k_PTC_initial

                # try a new (random) PTC parameter if even the solve with k_PTC_initial fails
                if counter_adapt>0:
                    k_PTC *= np.random.uniform(self.PTC_randadapt_range[0], self.PTC_randadapt_range[1])

                if self.comm.rank == 0:
                    print("PTC factor: %.4f" % (k_PTC))
                    sys.stdout.flush()

                counter_adapt += 1

                for npr in range(self.nprob):

                    # reset solver
                    for n in range(self.nfields[npr]):
                        self.reset_step(self.x[npr][n], self.x_start[npr][n], self.is_ghosted[npr][n])
                        if self.pb[npr].sub_solve: # can only be a 0D model so far...
                            self.reset_step(self.pb[npr].pb0.s, self.s_start, 0)

                    # re-set residual actions
                    self.residual_problem_actions(t, npr, localdata)

            # check if all problems have converged
            if all(converged):

                # reset to normal Newton if PTC was used in a divcont action
                if self.divcont=='PTC':
                    self.PTC = False
                    counter_adapt = 0
                self.ni = it-1
                break

        else:

            raise RuntimeError("Newton did not converge after %i iterations!" % (it))


    def residual_problem_actions(self, t, npr, localdata):

        # any local solve that is needed
        if self.pb[npr].localsolve:
            self.solve_local(localdata)

        # if we have two problems in a partitioned solve which exchange DBCs, we need to take care
        # of this separately by calling a coupled problem function
        if self.cp is not None:
            self.cp.evaluate_residual_dbc_coupling()

        tes = time.time()

        # compute residual
        self.pb[npr].assemble_residual(t, subsolver=self.subsol)

        tee = time.time() - tes
        if self.printenh:
            if self.comm.rank == 0:
                print(' '*self.indlen_[npr] + '      === Residual assemble, te = %.4f s' % (tee))
                sys.stdout.flush()

        # apply model order reduction of residual
        if self.pb[npr].rom:
            self.pb[npr].rom.reduce_residual(self.pb[npr].r_list, self.pb[npr].r_list_rom, x=self.x[npr][0])

        # get residual norms
        for n in range(self.nfields[npr]):
            self.r_list_sol[npr][n].assemble()
            self.resnorms[npr]['res'+str(n+1)] = self.r_list_sol[npr][n].norm()


    def jacobian_problem_actions(self, t, npr, k_PTC):

        tes = time.time()

        # compute Jacobian
        self.pb[npr].assemble_stiffness(t, subsolver=self.subsol)

        tee = time.time() - tes
        if self.printenh:
            if self.comm.rank == 0:
                print(' '*self.indlen_[npr] + '      === Jacobian assemble, te = %.4f s' % (tee))
                sys.stdout.flush()

        # apply model order reduction of stiffness
        if self.pb[npr].rom:
            self.pb[npr].rom.reduce_stiffness(self.pb[npr].K_list, self.pb[npr].K_list_rom, self.pb[npr].K_list_tmp)

        if self.PTC:
            # computes K_00 + k_PTC * I
            self.K_list_sol[npr][0][0].shift(k_PTC)


    def reset_step(self, vec, vec_start, ghosted):

        vec.axpby(1.0, 0.0, vec_start)

        if ghosted==1:
            vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if ghosted==2:
            subvecs = vec.getNestSubVecs()
            for j in range(len(subvecs)): subvecs[j].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


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
                increment_proj = project(increment_forms[i], functionspaces[i], self.pb[0].dx_, comm=self.comm)
                increments[i].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                increments[i].interpolate(increment_proj)

            for i in range(num_loc_res):
                # update var vector
                var[i].vector.axpy(1.0, increments[i].vector)
                var[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            for i in range(num_loc_res):
                # interpolate symbolic residual form into residual vector
                residual_proj = project(residual_forms[i], functionspaces[i], self.pb[0].dx_, comm=self.comm)
                residuals[i].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                residuals[i].interpolate(residual_proj)
                # get residual and increment inf norms
                res_norms[i] = residuals[i].vector.norm(norm_type=3)
                inc_norms[i] = increments[i].vector.norm(norm_type=3)

            if self.print_local_iter:
                if self.comm.rank == 0:
                    print("      (it_local = %i, res: %.4e, inc: %.4e)" % (it_local,np.sum(res_norms),np.sum(inc_norms)))
                    sys.stdout.flush()

            # increase iteration index
            it_local += 1

            # check if converged
            if np.sum(res_norms) <= self.tol_res_local and np.sum(inc_norms) <= self.tol_inc_local:

                break

        else:

            raise RuntimeError("Local Newton did not converge after %i iterations!" % (it_local))


    def destroy(self):

        for npr in range(self.nprob):

            for n in range(self.nfields[npr]):
                self.del_x_sol[npr][n].destroy()
                self.x_start[npr][n].destroy()
                self.del_x[npr][n].destroy()
                if self.pb[npr].rom and npr==0:
                    self.del_x_rom[npr][n].destroy()
            if self.pb[npr].sub_solve:
                self.s_start.destroy()

            if self.nfields[npr] > 1:
                self.del_full.destroy()

            if self.r_full_nest[npr] is not None: self.r_full_nest[npr].destroy()
            if self.K_full_nest[npr] is not None: self.K_full_nest[npr].destroy()
            if self.P_full_nest[npr] is not None: self.P_full_nest[npr].destroy()

            if self.r_full_merged[npr] is not None: self.r_full_merged[npr].destroy()
            if self.K_full_merged[npr] is not None: self.K_full_merged[npr].destroy()
            if self.P_full_merged[npr] is not None: self.P_full_merged[npr].destroy()

            self.ksp[npr].destroy()


# solver for pure ODE (0D) problems (e.g. a system of first order ODEs integrated with One-Step-Theta method)
class solver_nonlinear_ode(solver_nonlinear):

    def __init__(self, pb, solver_params):

        ioparams.check_params_solver(solver_params)

        self.comm = pb[0].comm

        self.pb = pb[0] # only one problem considered here
        self.nprob = 1

        self.ptype = self.pb.problem_physics

        try: self.maxiter = solver_params['maxiter']
        except: self.maxiter = 25

        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'mumps'

        self.tolres = solver_params['tol_res']
        self.tolinc = solver_params['tol_inc']

        self.tolerances = [[]]
        self.tolerances[0] = {'res1' : self.tolres, 'inc1' : self.tolinc}

        # dicts for residual and increment norms
        self.resnorms, self.incnorms = {}, {}

        self.PTC = False # don't think we'll ever need PTC for the 0D ODE problem...
        self.solvetype = 'direct' # only a direct solver is available for ODE problems

        self.solutils = sol_utils(self)

        self.lsp = self.solutils.timestep_separator_len()

        self.indlen = 1

        self.initialize_petsc_solver()


    def initialize_petsc_solver(self):

        self.ksp = [[]]

        # create solver
        self.ksp[0] = PETSc.KSP().create(self.comm)
        self.ksp[0].setType("preonly")
        pc = self.ksp[0].getPC()
        pc.setType("lu")
        pc.setFactorSolverType(self.direct_solver)

        # solution increment
        self.del_s = self.pb.K.createVecLeft()


    def newton(self, t, print_iter=True, sub=False):

        # Newton iteration index
        it = 0

        if print_iter: self.solutils.print_nonlinear_iter(header=True, sub=sub, ptype=self.ptype)

        self.ni, self.li = 0, 0 # nonlinear and linear iteration counters (latter probably never relevant for ODE problems...)

        tes = time.time()

        # compute initial residual
        self.pb.assemble_residual(t)

        # get initial residual norm
        self.resnorms['res1'] = self.pb.r_list[0].norm()

        te = time.time() - tes

        if print_iter: self.solutils.print_nonlinear_iter(it, resnorms=self.resnorms, te=te, sub=sub, ptype=self.ptype)

        it += 1

        while it < self.maxiter:

            tes = time.time()

            # compute Jacobian
            self.pb.assemble_stiffness(t)

            # solve linear system
            self.ksp[0].setOperators(self.pb.K_list[0][0])

            te = time.time() - tes

            tss = time.time()
            self.ksp[0].solve(-self.pb.r_list[0], self.del_s)
            ts = time.time() - tss

            tes = time.time()

            # update solution
            self.pb.s.axpy(1.0, self.del_s)

            # compute new residual after updated solution
            self.pb.assemble_residual(t)

            # get norms
            self.resnorms['res1'], self.incnorms['inc1'] = self.pb.r_list[0].norm(), self.del_s.norm()

            te += time.time() - tes

            if print_iter: self.solutils.print_nonlinear_iter(it, resnorms=self.resnorms, incnorms=self.incnorms, ts=ts, te=te, sub=sub, ptype=self.ptype)

            it += 1

            # check if converged
            converged = self.solutils.check_converged(self.resnorms, self.incnorms, self.tolerances[0], ptype='flow0d')
            if converged:
                if print_iter and sub:
                    if self.comm.rank == 0:
                        print('       ****************************************************\n')
                        sys.stdout.flush()
                self.ni = it-1
                break

        else:

            raise RuntimeError("Newton for ODE system did not converge after %i iterations!" % (it))


    def destroy(self):

        self.del_s.destroy()
        self.ksp[0].destroy()
