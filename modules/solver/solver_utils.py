#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np


class sol_utils():

    def __init__(self, pb, ptype, solver_params):
        
        self.pb = pb
        self.ptype = ptype

        try: self.print_liniter_every = solver_params['print_liniter_every']
        except: self.print_liniter_every = 50
        

    def catch_solver_errors(self, resnorm, incnorm=0, maxval=1.0e16):
        
        err = 0
        
        if np.isnan(resnorm):
                    
            if self.pb.comm.rank == 0:
                print("NaN encountered. Reset Newton and perform PTC adaption.")
                sys.stdout.flush()
                
            err = 1
            
        if resnorm >= maxval:
                    
            if self.pb.comm.rank == 0:
                print("Large residual > max val %.1E encountered. Reset Newton and perform PTC adaption." % (maxval))
                sys.stdout.flush()
                
            err = 1
            
        if np.isinf(incnorm):
                    
            if self.pb.comm.rank == 0:
                print("Inf encountered. Reset Newton and perform PTC adaption.")
                sys.stdout.flush()
                
            err = 1
        
        return err


    def print_nonlinear_iter(self,it=0,resnorms=0,incnorms=0,PTC=False,k_PTC=0,header=False,ts=0,te=0):
        
        if PTC:
            nkptc='k_ptc = '+str(format(k_PTC, '.4e'))+''
        else:
            nkptc=''

        if header:
            if self.pb.comm.rank == 0:
                if self.ptype=='solid' and not self.pb.incompressible_2field:
                    print('{:<6s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','solid res 2-norm','solid inc 2-norm','ts','te'))
                    sys.stdout.flush()
                elif self.ptype=='solid' and self.pb.incompressible_2field:
                    print('{:<6s}{:<21s}{:<21s}{:<21s}{:<21s}{:<10s}{:<5s}'.format('iter','solid res_u 2-norm','solid inc_u 2-norm','solid res_p 2-norm','solid inc_p 2-norm','ts','te'))
                    sys.stdout.flush()
                elif self.ptype=='fluid':
                    print('{:<6s}{:<21s}{:<21s}{:<21s}{:<21s}{:<10s}{:<5s}'.format('iter','fluid res_v 2-norm','fluid inc_v 2-norm','fluid res_p 2-norm','fluid inc_p 2-norm','ts','te'))
                    sys.stdout.flush()
                elif self.ptype=='flow0d':
                    print('{:<6s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','flow0d res 2-norm','flow0d inc 2-norm','ts','te'))
                    sys.stdout.flush()
                elif (self.ptype=='solid_flow0d' or self.ptype=='solid_constraint') and not self.pb.incompressible_2field:
                    if self.pbc.coupling_type == 'monolithic_direct':
                        print('{:<6s}{:<19s}{:<19s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','solid res 2-norm','solid inc 2-norm','flow0d res 2-norm','flow0d inc 2-norm','ts','te'))
                    if self.pbc.coupling_type == 'monolithic_lagrange':
                        print('{:<6s}{:<19s}{:<19s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','solid res 2-norm','solid inc 2-norm','lmcoup res 2-norm','lmcoup inc 2-norm','ts','te'))
                    sys.stdout.flush()
                elif (self.ptype=='solid_flow0d' or self.ptype=='solid_constraint') and self.pb.incompressible_2field:
                    if self.pbc.coupling_type == 'monolithic_direct':
                        print('{:<6s}{:<21s}{:<21s}{:<21s}{:<21s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','solid res_u 2-norm','solid inc_u 2-norm','solid res_p 2-norm','solid inc_p 2-norm','flow0d res 2-norm','flow0d inc 2-norm','ts','te'))
                    if self.pbc.coupling_type == 'monolithic_lagrange':
                        print('{:<6s}{:<21s}{:<21s}{:<21s}{:<21s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','solid res_u 2-norm','solid inc_u 2-norm','solid res_p 2-norm','solid inc_p 2-norm','lmcoup res 2-norm','lmcoup inc 2-norm','ts','te'))
                    sys.stdout.flush()
                elif self.ptype=='fluid_flow0d':
                    if self.pbc.coupling_type == 'monolithic_direct':
                        print('{:<6s}{:<21s}{:<21s}{:<21s}{:<21s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','fluid res_v 2-norm','fluid inc_v 2-norm','fluid res_p 2-norm','fluid inc_p 2-norm','flow0d res 2-norm','flow0d inc 2-norm','ts','te'))
                    if self.pbc.coupling_type == 'monolithic_lagrange':
                        print('{:<6s}{:<21s}{:<21s}{:<21s}{:<21s}{:<19s}{:<19s}{:<10s}{:<5s}'.format('iter','fluid res_v 2-norm','fluid inc_v 2-norm','fluid res_p 2-norm','fluid inc_p 2-norm','lmcoup res 2-norm','lmcoup inc 2-norm','ts','te'))
                    sys.stdout.flush()
                else:
                    raise NameError("Unknown problem type!")
            return
        
        if self.pb.comm.rank == 0:

            if self.ptype=='solid' and not self.pb.incompressible_2field: 
                print('{:<3d}{:<3s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_u'],' ',incnorms['inc_u'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            elif self.ptype=='solid' and self.pb.incompressible_2field:
                print('{:<3d}{:<3s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_u'],' ',incnorms['inc_u'],' ',resnorms['res_p'],' ',incnorms['inc_p'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            elif self.ptype=='fluid':
                print('{:<3d}{:<3s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_u'],' ',incnorms['inc_u'],' ',resnorms['res_p'],' ',incnorms['inc_p'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            elif self.ptype=='flow0d':
                print('{:<3d}{:<3s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_0d'],' ',incnorms['inc_0d'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            elif (self.ptype=='solid_flow0d' or self.ptype=='solid_constraint') and not self.pb.incompressible_2field:
                print('{:<3d}{:<3s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_u'],' ',incnorms['inc_u'],' ',resnorms['res_0d'],' ',incnorms['inc_0d'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            elif (self.ptype=='solid_flow0d' or self.ptype=='solid_constraint') and self.pb.incompressible_2field:
                print('{:<3d}{:<3s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_u'],' ',incnorms['inc_u'],' ',resnorms['res_p'],' ',incnorms['inc_p'],' ',resnorms['res_0d'],' ',incnorms['inc_0d'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            elif self.ptype=='fluid_flow0d':
                print('{:<3d}{:<3s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<11s}{:<4.4e}{:<9s}{:<4.4e}{:<9s}{:<4.2e}{:<2s}{:<4.2e}{:<9s}{:<18s}'.format(it,' ',resnorms['res_u'],' ',incnorms['inc_u'],' ',resnorms['res_p'],' ',incnorms['inc_p'],' ',resnorms['res_0d'],' ',incnorms['inc_0d'],' ',ts,' ',te,' ',nkptc))
                sys.stdout.flush()
            else:
                raise NameError("Unknown problem type!")


    def print_linear_iter(self,it,rnorm):
        
        if it == 0:
            self.rnorm_start = rnorm
            if self.pb.comm.rank == 0:
                print("\n            ***************** linear solve ****************")
                sys.stdout.flush()

        if it % self.print_liniter_every == 0:
            
            if self.pb.comm.rank == 0:
                print('{:<21s}{:<4d}{:<21s}{:<4e}'.format('            lin. it.: ',it,'     rel. res. norm:',rnorm/self.rnorm_start))
                sys.stdout.flush()


    def print_linear_iter_last(self,it,rnorm):
        
        if self.pb.comm.rank == 0:
            if it % self.print_liniter_every != 0: # otherwise already printed
                print('{:<21s}{:<4d}{:<21s}{:<4e}'.format('            lin. it.: ',it,'     rel. res. norm:',rnorm/self.rnorm_start))
            print("            ***********************************************\n")
            sys.stdout.flush()


    def check_converged(self, resnorms, incnorms, tolerances, ptype=None):

        if ptype is None:
            ptype = self.ptype

        converged = False

        if ptype=='solid' and not self.pb.incompressible_2field:
            if resnorms['res_u'] <= tolerances['res_u'] and incnorms['inc_u'] <= tolerances['inc_u']:
                converged = True
                
        elif ptype=='solid' and self.pb.incompressible_2field:
            if resnorms['res_u'] <= tolerances['res_u'] and incnorms['inc_u'] <= tolerances['inc_u'] and resnorms['res_p'] <= tolerances['res_p'] and incnorms['inc_p'] <= tolerances['inc_p']:
                converged = True
                
        elif ptype=='fluid':
            if resnorms['res_u'] <= tolerances['res_u'] and incnorms['inc_u'] <= tolerances['inc_u'] and resnorms['res_p'] <= tolerances['res_p'] and incnorms['inc_p'] <= tolerances['inc_p']:
                converged = True
        
        elif ptype=='flow0d':
            if resnorms['res_0d'] <= tolerances['res_0d'] and incnorms['inc_0d'] <= tolerances['inc_0d']:
                converged = True
                
        elif (ptype=='solid_flow0d' or self.ptype=='solid_constraint') and not self.pb.incompressible_2field:
            if resnorms['res_u'] <= tolerances['res_u'] and incnorms['inc_u'] <= tolerances['inc_u'] and resnorms['res_0d'] <= tolerances['res_0d'] and incnorms['inc_0d'] <= tolerances['inc_0d']:
                converged = True
                
        elif (ptype=='solid_flow0d' or self.ptype=='solid_constraint') and self.pb.incompressible_2field:
            if resnorms['res_u'] <= tolerances['res_u'] and incnorms['inc_u'] <= tolerances['inc_u'] and resnorms['res_p'] <= tolerances['res_p'] and incnorms['inc_p'] <= tolerances['inc_p'] and resnorms['res_0d'] <= tolerances['res_0d'] and incnorms['inc_0d'] <= tolerances['inc_0d']:
                converged = True
                
        elif ptype=='fluid_flow0d':
            if resnorms['res_u'] <= tolerances['res_u'] and incnorms['inc_u'] <= tolerances['inc_u'] and resnorms['res_p'] <= tolerances['res_p'] and incnorms['inc_p'] <= tolerances['inc_p'] and resnorms['res_0d'] <= tolerances['res_0d'] and incnorms['inc_0d'] <= tolerances['inc_0d']:
                converged = True
            
        else:
            raise NameError("Unknown problem type!")
        
        return converged
        

    def adapt_linear_solver(self, rabsnorm):

        rnorm = self.ksp.getResidualNorm()

        if rnorm*self.tollin < self.tolres: # currentnlnres*tol < desirednlnres

            # formula: "desirednlnres * factor / currentnlnres"
            tollin_new = self.adapt_factor * self.tolres/rnorm

            if tollin_new > 1.0:
                if self.pb.comm.rank == 0:
                    print("Warning: Adapted relative tolerance > 1. --> Constrained to 0.999, but consider changing parameter 'adapt_factor'!")
                    sys.stdout.flush()
                tollin_new = 0.999
            
            if tollin_new < self.tollin:
                tollin_new = self.tollin

            if self.pb.comm.rank == 0 and tollin_new > self.tollin:
                print("            Adapted linear tolerance to %.1e\n" % tollin_new)
                sys.stdout.flush()

            # adapt tolerance
            self.ksp.setTolerances(rtol=tollin_new)


    def timestep_separator(self, tolerances):

        if len(tolerances)==2:
            return "------------------------------------------------------------------------------"
        
        elif len(tolerances)==4:
            return "------------------------------------------------------------------------------------------------------------"

        elif len(tolerances)==6:
            return "------------------------------------------------------------------------------------------------------------"
        
        else:
            raise ValueError("Unknown size of tolerances!")
