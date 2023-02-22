#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
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

# TODO: Add fluid_flow0d testcases!
if category=='all':
    solid, fluid, flow0d, solid_flow0d, solid_constraint = True, True, True, True, True
elif category=='solid':
    solid, fluid, flow0d, solid_flow0d, solid_constraint = True, False, False, False, False
elif category=='fluid':
    solid, fluid, flow0d, solid_flow0d, solid_constraint = False, True, False, False, False
elif category=='flow0d':
    solid, fluid, flow0d, solid_flow0d, solid_constraint = False, False, True, False, False
elif category=='solid_flow0d':
    solid, fluid, flow0d, solid_flow0d, solid_constraint = False, False, False, True, False
elif category=='solid_constraint':
    solid, fluid, flow0d, solid_flow0d, solid_constraint = False, False, False, False, True
else:
    raise NameError("Unknown test category!")

errs = {}

start = time.time()

# make directory for temporary results output
subprocess.call(['mkdir', '-p', 'tmp'])


if solid:
    errs['solid_mat_uniax_hex_2field 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_mat_uniax_hex_2field.py'])
    errs['solid_mat_uniax_hex_2field 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_mat_uniax_hex_2field.py'])
    
    errs['solid_robin_genalpha 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_genalpha.py'])
    errs['solid_robin_genalpha 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_robin_genalpha.py'])
    
    errs['solid_robin_visco 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_visco.py'])

    errs['solid_robin_static_prestress 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_static_prestress.py'])
    errs['solid_robin_static_prestress 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_robin_static_prestress.py'])

    errs['solid_growth_volstressmandel 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_volstressmandel.py'])
    errs['solid_growth_volstressmandel 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_growth_volstressmandel.py'])
    
    errs['solid_growth_volstressmandel_incomp 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_volstressmandel_incomp.py'])
    errs['solid_growth_volstressmandel_incomp 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_growth_volstressmandel_incomp.py'])

    errs['solid_growth_prescribed_iso_lv 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_prescribed_iso_lv.py'])
    errs['solid_growth_prescribed_iso_lv 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_growth_prescribed_iso_lv.py'])
    
    errs['solid_growthremodeling_fiberstretch 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growthremodeling_fiberstretch.py']) # only 1 element, cannot run on multiple cores
    
    errs['solid_2Dheart_frankstarling 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_2Dheart_frankstarling.py'])
    errs['solid_2Dheart_frankstarling 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_2Dheart_frankstarling.py'])

if fluid:
    errs['fluid_taylorhood_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'fluid_taylorhood_cylinder.py'])
    errs['fluid_taylorhood_cylinder 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'fluid_taylorhood_cylinder.py'])

if flow0d:
    errs['flow0d_0Dvol_4elwindkesselLsZ 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dvol_4elwindkesselLsZ.py'])
    errs['flow0d_0Dvol_4elwindkesselLsZ 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dvol_4elwindkesselLsZ.py'])

    errs['flow0d_0Dvol_4elwindkesselLpZ 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dvol_4elwindkesselLpZ.py'])
    errs['flow0d_0Dvol_4elwindkesselLpZ 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dvol_4elwindkesselLpZ.py'])

    errs['flow0d_0Dheart_syspul 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dheart_syspul.py'])
    errs['flow0d_0Dheart_syspul 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspul.py'])
    errs['flow0d_0Dheart_syspul 2 restart'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspul.py', str(450)]) # tests restart from step 450

    errs['flow0d_0Dheart_syspulcor 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspulcor.py'])
    errs['flow0d_0Dheart_syspulcap 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspulcap.py'])

    errs['flow0d_0Dheart_syspulcaprespir_periodic 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'flow0d_0Dheart_syspulcaprespir_periodic.py'])
    errs['flow0d_0Dheart_syspulcaprespir_periodic 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'flow0d_0Dheart_syspulcaprespir_periodic.py'])

if solid_flow0d:
    errs['solid_flow0d_monolithicdirect_4elwindkesselLsZ_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect_4elwindkesselLsZ_chamber.py'])
    errs['solid_flow0d_monolithicdirect_4elwindkesselLsZ_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolithicdirect_4elwindkesselLsZ_chamber.py'])
    
    errs['solid_flow0d_monolithicdirect2field_4elwindkesselLpZ_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect2field_4elwindkesselLpZ_chamber.py'])
    errs['solid_flow0d_monolithicdirect2field_4elwindkesselLpZ_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolithicdirect2field_4elwindkesselLpZ_chamber.py'])

    errs['solid_flow0d_monolithiclagrange2field_2elwindkessel_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithiclagrange2field_2elwindkessel_chamber.py'])
    errs['solid_flow0d_monolithiclagrange2field_2elwindkessel_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolithiclagrange2field_2elwindkessel_chamber.py'])

    errs['solid_flow0d_monolithicdirect_syspul_2Dheart_prestress 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect_syspul_2Dheart_prestress.py'])
    errs['solid_flow0d_monolithicdirect_syspul_2Dheart_prestress 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_flow0d_monolithicdirect_syspul_2Dheart_prestress.py'])
    errs['solid_flow0d_monolithicdirect_syspul_2Dheart_prestress 3 restart'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_flow0d_monolithicdirect_syspul_2Dheart_prestress.py', str(3)]) # tests restart from step 3
    
    errs['solid_flow0d_monolithicdirect2field_flux_syspulcap_3Dheart_iterative 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'solid_flow0d_monolithicdirect2field_flux_syspulcap_3Dheart_iterative.py'])

    errs['solid_flow0d_monolithicdirect_syspulcor_2Dheart_ROM 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolithicdirect_syspulcor_2Dheart_ROM.py'])

if solid_constraint:
    errs['solid_constraint_volume_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_constraint_volume_chamber.py'])
    errs['solid_constraint_volume_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_constraint_volume_chamber.py'])
     

err = 0
for e in range(len(errs)):
    if list(errs.values())[e] != 0:
        err += 1

print("\nSummary:")
print("========")
for e in range(len(errs)):
    if list(errs.values())[e] == 0:
        print('{:<75s}{:<18s}'.format(list(errs.keys())[e],'status: passed :-)'))
    else:
        print('{:<75s}{:<18s}'.format(list(errs.keys())[e],'status: FAILED !!!!!!'))

if err == 0:
    print("\n##################################")
    print("All tests passed successfully! :-)")
    print("##################################\n")
else:
    print("\n##################################")
    print("%i tests failed!!!" % (err))
    print("##################################\n")

print('Total runtime for tests: %.4f s (= %.2f min)' % ( time.time()-start, (time.time()-start)/60. ))
