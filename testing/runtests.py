#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, subprocess, time
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--category', dest='c', action='store', type=str, default='all') # all, solid, fluid, flow0d, solid_flow0d, fluid_flow0d, solid_constraint, frsi
parser.add_argument('-b', '--branch', dest='b', action='store', type=str, default='nightly') # nightly, mixed

args = parser.parse_args()

category = args.c
branch = args.b

errs = {}

start = time.time()

# make directory for temporary results output
subprocess.call(['mkdir', '-p', 'tmp'])


if category=='solid' or category=='all':
    errs['solid_mat_uniax_hex_2field 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_mat_uniax_hex_2field.py'])
    errs['solid_mat_uniax_hex_2field 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_mat_uniax_hex_2field.py'])

    errs['solid_robin_genalpha 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_genalpha.py'])
    errs['solid_robin_genalpha 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_robin_genalpha.py'])
    errs['solid_robin_genalpha 3 restart'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_robin_genalpha.py', str(8)])

    errs['solid_robin_visco 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_visco.py'])

    errs['solid_bodyforce_gravity 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_bodyforce_gravity.py'])
    errs['solid_bodyforce_gravity 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_bodyforce_gravity.py'])

    errs['solid_robin_static_prestress 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_robin_static_prestress.py'])
    errs['solid_robin_static_prestress 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_robin_static_prestress.py'])
    
    errs['solid_divcont_ptc 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_divcont_ptc.py'])

    errs['solid_growth_volstressmandel 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_volstressmandel.py'])
    errs['solid_growth_volstressmandel 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_growth_volstressmandel.py'])

    errs['solid_growth_volstressmandel_incomp 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_volstressmandel_incomp.py'])
    errs['solid_growth_volstressmandel_incomp 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_growth_volstressmandel_incomp.py'])

    errs['solid_growth_prescribed_iso_lv 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growth_prescribed_iso_lv.py'])
    errs['solid_growth_prescribed_iso_lv 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_growth_prescribed_iso_lv.py'])

    errs['solid_growthremodeling_fiberstretch 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_growthremodeling_fiberstretch.py']) # only 1 element, cannot run on multiple cores

    errs['solid_2Dheart_frankstarling 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_2Dheart_frankstarling.py'])
    errs['solid_2Dheart_frankstarling 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_2Dheart_frankstarling.py'])

    errs['solid_membrane 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_membrane.py'])
    errs['solid_membrane 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_membrane.py'])

if category=='fluid' or category=='all':
    errs['fluid_taylorhood_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'fluid_taylorhood_cylinder.py'])
    errs['fluid_taylorhood_cylinder 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'fluid_taylorhood_cylinder.py'])

    errs['fluid_p1p1_stab_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'fluid_p1p1_stab_cylinder.py'])
    errs['fluid_p1p1_stab_cylinder 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'fluid_p1p1_stab_cylinder.py'])

    if branch=='mixed':
        errs['fluid_p1p1_stab_cylinder_valve 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'fluid_p1p1_stab_cylinder_valve.py'])
        errs['fluid_p1p1_stab_cylinder_valve 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'fluid_p1p1_stab_cylinder_valve.py'])

if category=='fluid_flow0d' or category=='all':
    errs['fluid_flow0d_monolagr_taylorhood_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'fluid_flow0d_monolagr_taylorhood_cylinder.py'])
    errs['fluid_flow0d_monolagr_taylorhood_cylinder 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'fluid_flow0d_monolagr_taylorhood_cylinder.py'])

if category=='fluid_ale_flow0d' or category=='all':
    if branch=='mixed':
        errs['fluid_ale_flow0d_lalv_syspul_prescribed 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'fluid_ale_flow0d_lalv_syspul_prescribed.py'])

if category=='flow0d' or category=='all':
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

if category=='solid_flow0d' or category=='all':
    errs['solid_flow0d_monodir_4elwindkesselLsZ_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monodir_4elwindkesselLsZ_chamber.py'])
    errs['solid_flow0d_monodir_4elwindkesselLsZ_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monodir_4elwindkesselLsZ_chamber.py'])

    errs['solid_flow0d_monodir_4elwindkesselLsZ_chamber_iterative 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monodir_4elwindkesselLsZ_chamber_iterative.py'])

    errs['solid_flow0d_monodir2field_4elwindkesselLpZ_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monodir2field_4elwindkesselLpZ_chamber.py'])
    errs['solid_flow0d_monodir2field_4elwindkesselLpZ_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monodir2field_4elwindkesselLpZ_chamber.py'])

    errs['solid_flow0d_monolagr2field_2elwindkessel_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monolagr2field_2elwindkessel_chamber.py'])
    errs['solid_flow0d_monolagr2field_2elwindkessel_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monolagr2field_2elwindkessel_chamber.py'])

    errs['solid_flow0d_monodir_syspul_2Dheart_prestress 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monodir_syspul_2Dheart_prestress.py'])
    errs['solid_flow0d_monodir_syspul_2Dheart_prestress 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_flow0d_monodir_syspul_2Dheart_prestress.py'])
    errs['solid_flow0d_monodir_syspul_2Dheart_prestress 3 restart'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'solid_flow0d_monodir_syspul_2Dheart_prestress.py', str(3)]) # tests restart from step 3

    errs['solid_flow0d_monodir_flux_syspulcap_3Dheart_iterative 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_flow0d_monodir_flux_syspulcap_3Dheart_iterative.py'])

    errs['solid_flow0d_monodir2field_flux_syspulcap_3Dheart_iterative 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'solid_flow0d_monodir2field_flux_syspulcap_3Dheart_iterative.py'])

    errs['solid_flow0d_monodir_syspulcor_2Dheart_ROM 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_flow0d_monodir_syspulcor_2Dheart_ROM.py'])

if category=='solid_constraint' or category=='all':
    errs['solid_constraint_volume_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'solid_constraint_volume_chamber.py'])
    errs['solid_constraint_volume_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'solid_constraint_volume_chamber.py'])

if category=='frsi' or category=='all':
    errs['frsi_artseg_prefile 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'frsi_artseg_prefile.py'])
    errs['frsi_artseg_prefile 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'frsi_artseg_prefile.py'])

    errs['frsi_artseg_partition 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'frsi_artseg_partition.py'])
    errs['frsi_artseg_partition 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'frsi_artseg_partition.py'])
    errs['frsi_artseg_partition 2 restart'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'frsi_artseg_partition.py', str(2)])

    errs['frsi_artseg_prefile_iterative 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'frsi_artseg_prefile_iterative.py'])


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
