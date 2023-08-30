#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np


class sol_utils():

    def __init__(self, solver):

        self.solver = solver


    def catch_solver_errors(self, resnorm, incnorm=0, maxval=1e16):

        err = 0

        if np.isnan(resnorm):

            if self.solver.comm.rank == 0:
                print("NaN encountered. Reset Newton and perform PTC adaption.")
                sys.stdout.flush()

            err = 1

        if resnorm >= maxval:

            if self.solver.comm.rank == 0:
                print("Large residual > max val %.1E encountered. Reset Newton and perform PTC adaption." % (maxval))
                sys.stdout.flush()

            err = 1

        if np.isinf(incnorm):

            if self.solver.comm.rank == 0:
                print("Inf encountered. Reset Newton and perform PTC adaption.")
                sys.stdout.flush()

            err = 1

        return err


    def print_nonlinear_iter(self, it=0, resnorms=None, incnorms=None, header=False, ts=0, te=0, sub=False, ptype=None):

        if ptype=='solid':
            if self.solver.pb[0].incompressible_2field:
                eq1, eq2 = 'solid momentum', 'solid incompressibility'
                v1, v2 = 'u', 'p'
                numres = 2
            else:
                eq1 = 'solid momentum'
                v1 = 'u'
                numres = 1
        elif ptype=='fluid':
            eq1, eq2 = 'fluid momentum', 'fluid continuity'
            v1, v2 = 'v', 'p'
            numres = 2
        elif ptype=='ale':
            eq1 = 'ALE momentum'
            v1 = 'd'
            numres = 1
        elif ptype=='flow0d':
            eq1 = 'flow-0d'
            v1 = 's'
            numres = 1
        elif ptype=='signet':
            eq1 = 'signet-0d'
            v1 = 's'
            numres = 1
        elif ptype=='solid_flow0d':
            if self.solver.pb[0].incompressible_2field:
                if self.solver.pb[0].coupling_type == 'monolithic_direct':
                    eq1, eq2, eq3 = 'solid momentum', 'solid incompressibility', 'flow-0d'
                    v1, v2, v3 = 'u', 'p', 's'
                if self.solver.pb[0].coupling_type == 'monolithic_lagrange':
                    eq1, eq2, eq3 = 'solid momentum', 'solid incompressibility', 'lm constraint'
                    v1, v2, v3 = 'u', 'p', 'LM' # using greek symbol print (Λ) is not supported everywhere...
                numres = 3
            else:
                if self.solver.pb[0].coupling_type == 'monolithic_direct':
                    eq1, eq2 = 'solid momentum', 'flow-0d'
                    v1, v2 = 'u', 's'
                if self.solver.pb[0].coupling_type == 'monolithic_lagrange':
                    eq1, eq2 = 'solid momentum', 'lm constraint'
                    v1, v2 = 'u', 'LM' # using greek symbol print (Λ) is not supported everywhere...
                numres = 2
        elif ptype=='solid_constraint':
            if self.solver.pb[0].incompressible_2field:
                eq1, eq2, eq3 = 'solid momentum', 'solid incompressibility', 'lm constraint'
                v1, v2, v3 = 'u', 'p', 'LM' # using greek symbol print (Λ) is not supported everywhere...
                numres = 3
            else:
                eq1, eq2 = 'solid momentum', 'lm constraint'
                v1, v2 = 'u', 'lm'
                numres = 2
        elif ptype=='fluid_flow0d':
            eq1, eq2, eq3 = 'fluid momentum', 'fluid continuity', 'lm constraint'
            v1, v2, v3 = 'v', 'p', 'LM' # using greek symbol print (Λ) is not supported everywhere...
            numres = 3
        elif ptype=='fluid_ale':
            eq1, eq2, eq3 = 'fluid momentum', 'fluid continuity', 'ALE momentum'
            v1, v2, v3 = 'v', 'p', 'd'
            numres = 3
        elif ptype=='fluid_ale_flow0d':
            eq1, eq2, eq3, eq4 = 'fluid momentum', 'fluid continuity', 'lm constraint', 'ALE momentum'
            v1, v2, v3, v4 = 'v', 'p', 'LM', 'd' # using greek symbol print (Λ) is not supported everywhere...
            numres = 4
        elif ptype=='fsi':
            if self.solver.pb[0].incompressible_2field:
                eq1, eq2, eq3, eq4, eq5, eq6 = 'solid momentum', 'solid incompressibility', 'fluid momentum', 'fluid continuity', 'LM constraint', 'ALE momentum'
                v1, v2, v3, v4, v5, v6 = 'u', 'p', 'v', 'p', 'lm', 'd' # using greek symbol print (λ) is not supported everywhere...
                numres = 6
            else:
                eq1, eq2, eq3, eq4, eq5 = 'solid momentum', 'fluid momentum', 'fluid continuity', 'LM constraint', 'ALE momentum'
                v1, v2, v3, v4, v5 = 'u', 'v', 'p', 'lm', 'd' # using greek symbol print (λ) is not supported everywhere...
                numres = 5
        elif ptype=='fsi_flow0d':
            if self.solver.pb[0].incompressible_2field:
                eq1, eq2, eq3, eq4, eq5, eq6, eq7 = 'solid momentum', 'solid incompressibility', 'fluid momentum', 'fluid continuity', 'LM constraint', 'lm constraint', 'ALE momentum'
                v1, v2, v3, v4, v5, v6, v7 = 'u', 'p', 'v', 'p', 'lm', 'LM', 'd' # using greek symbol print (λ, Λ) is not supported everywhere...
                numres = 7
            else:
                eq1, eq2, eq3, eq4, eq5, eq6 = 'solid momentum', 'fluid momentum', 'fluid continuity', 'LM constraint', 'lm constraint', 'ALE momentum'
                v1, v2, v3, v4, v5, v6 = 'u', 'v', 'p', 'lm', 'LM', 'd' # using greek symbol print (λ, Λ) is not supported everywhere...
                numres = 6
        else:
            raise NameError("Unknown problem type!")

        # using greek symbol print (Δ) is not supported everywhere... so use d instead
        if header:
            if self.solver.comm.rank == 0:
                if numres==1:
                    if not sub:
                        print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ','timings'))
                        print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ','te','ts'))
                    else:
                        print(' ')
                        print('       ****************** 0D model solve ******************')
                        print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<6s}{:<25s}{:<3s}{:<7s}').format(' ',' ','it |',eq1,'| ','timings'))
                        print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<6s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ',' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ','te','ts'))
                elif numres==2:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ',eq2,'| ','timings'))
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ' ,'||r_'+v2+'||_2','||d'+v2+'||_2','| ','te','ts'))
                elif numres==3:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ',eq2,'| ',eq3,'| ','timings'))
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ' ,'||r_'+v2+'||_2','||d'+v2+'||_2','| ' ,'||r_'+v3+'||_2','||d'+v3+'||_2','| ','te','ts'))
                elif numres==4:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ',eq2,'| ',eq3,'| ',eq4,'| ','timings'))
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ' ,'||r_'+v2+'||_2','||d'+v2+'||_2','| ' ,'||r_'+v3+'||_2','||d'+v3+'||_2','| ' ,'||r_'+v4+'||_2','||d'+v4+'||_2','| ','te','ts'))
                elif numres==5:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ',eq2,'| ',eq3,'| ',eq4,'| ',eq5,'| ','timings'))
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ' ,'||r_'+v2+'||_2','||d'+v2+'||_2','| ' ,'||r_'+v3+'||_2','||d'+v3+'||_2','| ' ,'||r_'+v4+'||_2','||d'+v4+'||_2','| ','||r_'+v5+'||_2','||d'+v5+'||_2','| ','te','ts'))
                elif numres==6:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ',eq2,'| ',eq3,'| ',eq4,'| ',eq5,'| ',eq6,'| ','timings'))
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ' ,'||r_'+v2+'||_2','||d'+v2+'||_2','| ' ,'||r_'+v3+'||_2','||d'+v3+'||_2','| ' ,'||r_'+v4+'||_2','||d'+v4+'||_2','| ','||r_'+v5+'||_2','||d'+v5+'||_2','| ','||r_'+v6+'||_2','||d'+v6+'||_2','| ','te','ts'))
                elif numres==7:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<25s}{:<3s}{:<7s}').format(' ','it |',eq1,'| ',eq2,'| ',eq3,'| ',eq4,'| ',eq5,'| ',eq6,'| ',eq7,'| ','timings'))
                    print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}').format(' ','#  |','||r_'+v1+'||_2','||d'+v1+'||_2','| ' ,'||r_'+v2+'||_2','||d'+v2+'||_2','| ' ,'||r_'+v3+'||_2','||d'+v3+'||_2','| ' ,'||r_'+v4+'||_2','||d'+v4+'||_2','| ','||r_'+v5+'||_2','||d'+v5+'||_2','| ','||r_'+v6+'||_2','||d'+v6+'||_2','| ','||r_'+v7+'||_2','||d'+v7+'||_2','| ','te','ts'))
                else:
                    raise RuntimeError("Error. You should not be here!")
                sys.stdout.flush()

            return


        if self.solver.comm.rank == 0:

            if it==0:

                if numres==1:
                    if not sub:
                        print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',te,' ',' '))
                    else:
                        print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',' ',it,'| ',resnorms['res1'],' ',' ','  |  ',te,' ',' '))
                elif numres==2:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',resnorms['res2'],' ',' ','  |  ',te,' ',' '))
                elif numres==3:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',resnorms['res2'],' ',' ','  |  ',resnorms['res3'],' ',' ','  |  ',te,' ',' '))
                elif numres==4:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',resnorms['res2'],' ',' ','  |  ',resnorms['res3'],' ',' ','  |  ',resnorms['res4'],' ',' ','  |  ',te,' ',' '))
                elif numres==5:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',resnorms['res2'],' ',' ','  |  ',resnorms['res3'],' ',' ','  |  ',resnorms['res4'],' ',' ','  |  ',resnorms['res5'],' ',' ','  |  ',te,' ',' '))
                elif numres==6:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',resnorms['res2'],' ',' ','  |  ',resnorms['res3'],' ',' ','  |  ',resnorms['res4'],' ',' ','  |  ',resnorms['res5'],' ',' ','  |  ',resnorms['res6'],' ',' ','  |  ',te,' ',' '))
                elif numres==7:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}').format(' ',it,'| ',resnorms['res1'],' ',' ','  |  ',resnorms['res2'],' ',' ','  |  ',resnorms['res3'],' ',' ','  |  ',resnorms['res4'],' ',' ','  |  ',resnorms['res5'],' ',' ','  |  ',resnorms['res6'],' ',' ','  |  ',resnorms['res7'],' ',' ','  |  ',te,' ',' '))
                else:
                    raise RuntimeError("Number of residual norms inconsistent.")
                sys.stdout.flush()

            else:

                if numres==1:
                    if not sub:
                        print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',te,' ',ts))
                    else:
                        print(('{:<'+str(self.solver.indlen)+'s}{:<6s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',te,' ',ts))
                elif numres==2:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',resnorms['res2'],' ',incnorms['inc2'],'  |  ',te,' ',ts))
                elif numres==3:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',resnorms['res2'],' ',incnorms['inc2'],'  |  ',resnorms['res3'],' ',incnorms['inc3'],'  |  ',te,' ',ts))
                elif numres==4:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',resnorms['res2'],' ',incnorms['inc2'],'  |  ',resnorms['res3'],' ',incnorms['inc3'],'  |  ',resnorms['res4'],' ',incnorms['inc4'],'  |  ',te,' ',ts))
                elif numres==5:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',resnorms['res2'],' ',incnorms['inc2'],'  |  ',resnorms['res3'],' ',incnorms['inc3'],'  |  ',resnorms['res4'],' ',incnorms['inc4'],'  |  ',resnorms['res5'],' ',incnorms['inc5'],'  |  ',te,' ',ts))
                elif numres==6:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',resnorms['res2'],' ',incnorms['inc2'],'  |  ',resnorms['res3'],' ',incnorms['inc3'],'  |  ',resnorms['res4'],' ',incnorms['inc4'],'  |  ',resnorms['res5'],' ',incnorms['inc5'],'  |  ',resnorms['res6'],' ',incnorms['inc6'],'  |  ',te,' ',ts))
                elif numres==7:
                    print(('{:<'+str(self.solver.indlen)+'s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}').format(' ',it,'| ',resnorms['res1'],' ',incnorms['inc1'],'  |  ',resnorms['res2'],' ',incnorms['inc2'],'  |  ',resnorms['res3'],' ',incnorms['inc3'],'  |  ',resnorms['res4'],' ',incnorms['inc4'],'  |  ',resnorms['res5'],' ',incnorms['inc5'],'  |  ',resnorms['res6'],' ',incnorms['inc6'],'  |  ',resnorms['res7'],' ',incnorms['inc7'],'  |  ',te,' ',ts))
                else:
                    raise RuntimeError("Number of residual norms inconsistent.")
                sys.stdout.flush()


    def print_linear_iter(self, it, rnorm):

        if it == 0:
            self.rnorm_start = rnorm
            if self.solver.comm.rank == 0:
                print(' ')
                print(('{:<'+str(self.solver.indlen)+'s}{:<8s}{:<47s}').format(' ',' ','***************** linear solve ****************'))
                sys.stdout.flush()

        if it % self.solver.print_liniter_every == 0:

            if self.solver.res_lin_monitor=='rel': resnorm = rnorm/self.rnorm_start
            elif self.solver.res_lin_monitor=='abs': resnorm = rnorm
            else: raise ValueError("Unknown res_lin_monitor value. Choose 'rel' or 'abs'.")

            if self.solver.comm.rank == 0:
                print(('{:<'+str(self.solver.indlen)+'s}{:<17s}{:<4d}{:<21s}{:<4e}').format(' ','        lin. it.: ',it,'     '+self.solver.res_lin_monitor+'. res. norm:',resnorm))
                sys.stdout.flush()


    def print_linear_iter_last(self, it, rnorm, conv_reason):

        if self.solver.res_lin_monitor=='rel': resnorm = rnorm/self.rnorm_start
        elif self.solver.res_lin_monitor=='abs': resnorm = rnorm
        else: raise ValueError("Unknown res_lin_monitor value. Choose 'rel' or 'abs'.")

        if self.solver.comm.rank == 0:
            if it % self.solver.print_liniter_every != 0: # otherwise already printed
                print(('{:<'+str(self.solver.indlen)+'s}{:<17s}{:<4d}{:<21s}{:<4e}').format(' ','        lin. it.: ',it,'     '+self.solver.res_lin_monitor+'. res. norm:',resnorm))
            # cf. https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP.ConvergedReason-class.html for converge codes
            print(('{:<'+str(self.solver.indlen)+'s}{:<8s}{:<13s}{:<18s}{:<2d}{:<14s}').format(' ',' ','************ ',' PETSc conv code: ',conv_reason,' *************'))
            if not self.solver.pb[0].print_subiter: print(' ') # no extra line if we have an "intermediate" print from another model... TODO: Find a nicer solution here...
            sys.stdout.flush()

        # update counters
        self.solver.li += it
        self.solver.li_s.append(it)


    def check_converged(self, resnorms, incnorms, tolerances, ptype=None):

        if ptype is None:
            ptype = self.solver.ptype

        converged = False

        if ptype=='solid' and not self.solver.pb[0].incompressible_2field:
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1']:
                converged = True

        elif ptype=='solid' and self.solver.pb[0].incompressible_2field:
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2']:
                converged = True

        elif ptype=='fluid':
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2']:
                converged = True

        elif ptype=='ale':
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1']:
                converged = True

        elif ptype=='flow0d':
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1']:
                converged = True

        elif (ptype=='solid_flow0d' or ptype=='solid_constraint') and not self.solver.pb[0].incompressible_2field:
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2']:
                converged = True

        elif (ptype=='solid_flow0d' or ptype=='solid_constraint') and self.solver.pb[0].incompressible_2field:
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2'] and resnorms['res3'] <= tolerances['res3'] and incnorms['inc3'] <= tolerances['inc3']:
                converged = True

        elif ptype=='fluid_flow0d':
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2'] and resnorms['res3'] <= tolerances['res3'] and incnorms['inc3'] <= tolerances['inc3']:
                converged = True

        elif ptype=='fluid_ale':
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2'] and resnorms['res3'] <= tolerances['res3'] and incnorms['inc3'] <= tolerances['inc3']:
                converged = True

        elif ptype=='fluid_ale_flow0d':
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2'] and resnorms['res3'] <= tolerances['res3'] and incnorms['inc3'] <= tolerances['inc3'] and resnorms['res4'] <= tolerances['res4'] and incnorms['inc4'] <= tolerances['inc4']:
                converged = True

        elif ptype=='fsi' and not self.solver.pb[0].incompressible_2field:
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2'] and resnorms['res3'] <= tolerances['res3'] and incnorms['inc3'] <= tolerances['inc3'] and resnorms['res4'] <= tolerances['res4'] and incnorms['inc4'] <= tolerances['inc4'] and resnorms['res5'] <= tolerances['res5'] and incnorms['inc5'] <= tolerances['inc5']:
                converged = True

        elif ptype=='fsi' and self.solver.pb[0].incompressible_2field:
            if resnorms['res1'] <= tolerances['res1'] and incnorms['inc1'] <= tolerances['inc1'] and resnorms['res2'] <= tolerances['res2'] and incnorms['inc2'] <= tolerances['inc2'] and resnorms['res3'] <= tolerances['res3'] and incnorms['inc3'] <= tolerances['inc3'] and resnorms['res4'] <= tolerances['res4'] and incnorms['inc4'] <= tolerances['inc4'] and resnorms['res5'] <= tolerances['res5'] and incnorms['inc5'] <= tolerances['inc5'] and resnorms['res6'] <= tolerances['res6'] and incnorms['inc6'] <= tolerances['inc6']:
                converged = True

        else:
            raise NameError("Unknown problem type!")

        return converged


    def timestep_separator_len(self):

        if len(self.solver.tolerances[0])==2:
            seplen = 53
        elif len(self.solver.tolerances[0])==4:
            seplen = 81
        elif len(self.solver.tolerances[0])==6:
            seplen = 109
        elif len(self.solver.tolerances[0])==8:
            seplen = 137
        elif len(self.solver.tolerances[0])==10:
            seplen = 165
        elif len(self.solver.tolerances[0])==12:
            seplen = 195
        else:
            raise ValueError("Unknown size of tolerances!")

        return seplen
