#!/usr/bin/env python3

import sys, os, subprocess, time
import math
from ufl import tr, det, dot, ln, sqrt, exp, diff, conditional, ge, outer, cross

# returns the Cauchy stress sigma for different material laws

class materiallaw:
    
    def __init__(self, gamma, I):
        self.gamma = gamma
        self.I = I
    

    def newtonian(self, params):
        
        eta = params['eta'] # dynamic viscosity

        # classical Newtonian fluid
        sigma = 2.*eta*self.gamma
        
        return sigma

