#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys, os, subprocess, time
import numpy as np

category = 'all'

try: # from command line
    category = sys.argv[1]
except:
    pass

# TODO: Add fluid and fluid_flow0d testcases!
if category=='all':
    solid, flow0d, solid_flow0d, solid_constraint = True, True, True, True
elif category=='solid':
    solid, flow0d, solid_flow0d, solid_constraint = True, False, False, False
elif category=='flow0d':
    solid, flow0d, solid_flow0d, solid_constraint = False, True, False, False
elif category=='solid_flow0d':
    solid, flow0d, solid_flow0d, solid_constraint = False, False, True, False
elif category=='solid_constraint':
    solid, flow0d, solid_flow0d, solid_constraint = False, False, False, True
else:
    raise NameError("Unknown test category!")

errs = []

start = time.time()

# make directory for temporary results output
subprocess.call(['mkdir', '-p', 'tmp'])


if solid:
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_mat_uniax_hex_2field.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_mat_uniax_hex_2field.py']) )
    
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_genalpha.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_robin_genalpha.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_static_prestress.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_robin_static_prestress.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_volstressmandel.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_growth_volstressmandel.py']) )
    
    # TODO: Re-include once we can have higher-order Quadrature function spaces!
    #errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_volstressmandel_incomp.py']) )
    #errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_growth_volstressmandel_incomp.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_prescribed_iso_lv.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_growth_prescribed_iso_lv.py']) )
    
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growthremodeling_fiberstretch.py']) ) # only 1 element, cannot run on multiple cores
    
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_2Dheart_frankstarling.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_2Dheart_frankstarling.py']) )

if flow0d:
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dvol_4elwindkesselLsZ.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dvol_4elwindkesselLsZ.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dvol_4elwindkesselLpZ.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dvol_4elwindkesselLpZ.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dheart_syspul.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspul.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspul.py', str(450)]) ) # tests restart from step 450

    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspulcor.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspulcap.py']) )

    # very slow... we have to make this one faster! But should pass...
    #errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dheart_syspulcaprespir_periodic.py']) )
    #errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspulcaprespir_periodic.py']) )

if solid_flow0d:
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect_4elwindkesselLsZ_chamber.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolithicdirect_4elwindkesselLsZ_chamber.py']) )
    
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect2field_4elwindkesselLpZ_chamber.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolithicdirect2field_4elwindkesselLpZ_chamber.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithiclagrange2field_2elwindkessel_chamber.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolithiclagrange2field_2elwindkessel_chamber.py']) )

    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect_syspul_2Dheart_prestress.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_flow0d_monolithicdirect_syspul_2Dheart_prestress.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_flow0d_monolithicdirect_syspul_2Dheart_prestress.py', str(3)]) ) # tests restart from step 3
    
    errs.append( subprocess.call(['mpiexec', '-n', '4', 'python3', 'solid_flow0d_monolithicdirect2field_flux_syspulcap_3Dheart_iterative.py']) )

    # TODO: ROM works only in serial so far!
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect_syspulcor_2Dheart_ROM.py']) )

if solid_constraint:
    errs.append( subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_constraint_volume_chamber.py']) )
    errs.append( subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_constraint_volume_chamber.py']) )
    

err = 0
for e in range(len(errs)):
    err += errs[e]

if err == 0:
    print("\n##################################")
    print("All tests passed successfully! :-)")
    print("##################################")
else:
    print("\n##################################")
    print("%i tests failed!!!" % (err))
    print("##################################")

print('Total runtime for tests: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
