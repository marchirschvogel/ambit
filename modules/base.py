#!/usr/bin/env python3

class problem_base():
    
    def __init__(self, io_params, time_params, comm):
        
        self.comm = comm
        
        self.problem_type = io_params['problem_type']

        self.timint = time_params['timint']
        
        if 'maxtime' in time_params.keys(): self.maxtime = time_params['maxtime']
        if 'numstep' in time_params.keys(): self.numstep = time_params['numstep']
        if 'numstep_stop' in time_params.keys(): self.numstep_stop = time_params['numstep_stop']

        if 'maxtime' in time_params.keys(): self.dt = self.maxtime/self.numstep
