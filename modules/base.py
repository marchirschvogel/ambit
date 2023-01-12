#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

class problem_base():
    
    def __init__(self, io_params, time_params, comm):
        
        self.comm = comm
        
        self.problem_type = io_params['problem_type']

        self.timint = time_params['timint']
        
        if 'maxtime' in time_params.keys(): self.maxtime = time_params['maxtime']
        if 'numstep' in time_params.keys(): self.numstep = time_params['numstep']
        if 'maxtime' in time_params.keys(): self.dt = self.maxtime/self.numstep
        
        try: self.restart_step = io_params['restart_step']
        except: self.restart_step = 0

        try: self.numstep_stop = time_params['numstep_stop']
        except: self.numstep_stop = self.numstep

        self.t_init = self.restart_step * self.dt
