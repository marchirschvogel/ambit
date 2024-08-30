#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, subprocess, time
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--category', dest='c', action='store', type=str, default='all') # all, solid, fluid, flow0d, solid_flow0d, fluid_flow0d, fluid_ale_flow0d, solid_constraint, fsi, fsi_flow0d
parser.add_argument('-b', '--branch', dest='b', action='store', type=str, default=' ')

args = parser.parse_args()

category = args.c

branch = args.b

assert(branch==' ' or branch=='mixed')

errs = {}

start = time.time()

# make directory for temporary results output
subprocess.call(['mkdir', '-p', 'tmp'])


if category=='solid' or category=='all':
    errs['test_solid_mat_uniax_hex_2field 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_mat_uniax_hex_2field.py', branch])
    errs['test_solid_mat_uniax_hex_2field 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_mat_uniax_hex_2field.py', branch])

    errs['test_solid_2d_pres 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_2d_pres.py', branch])
    
    errs['test_solid_ost_dbc_ramp 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_ost_dbc_ramp.py', branch])
    errs['test_solid_ost_dbc_ramp 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_ost_dbc_ramp.py', branch])

    errs['test_solid_robin_genalpha 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_robin_genalpha.py', branch])
    errs['test_solid_robin_genalpha 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_robin_genalpha.py', branch])
    errs['test_solid_robin_genalpha 3 restart'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_robin_genalpha.py', str(8), branch])

    errs['test_solid_robin_genalpha_amg 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_robin_genalpha_amg.py', branch])

    errs['test_solid_robin_visco 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_robin_visco.py', branch])

    errs['test_solid_bodyforce_gravity 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_bodyforce_gravity.py', branch])
    errs['test_solid_bodyforce_gravity 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_bodyforce_gravity.py', branch])

    errs['test_solid_robin_static_prestress 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_robin_static_prestress.py', branch])
    errs['test_solid_robin_static_prestress 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_robin_static_prestress.py', branch])

    errs['test_solid_divcont_ptc 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_divcont_ptc.py', branch])

    errs['test_solid_growth_volstressmandel 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_growth_volstressmandel.py', branch])
    errs['test_solid_growth_volstressmandel 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_growth_volstressmandel.py', branch])

    errs['test_solid_growth_volstressmandel_incomp 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_growth_volstressmandel_incomp.py', branch])
    errs['test_solid_growth_volstressmandel_incomp 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_growth_volstressmandel_incomp.py', branch])

    errs['test_solid_growth_prescribed_iso_lv 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_growth_prescribed_iso_lv.py', branch])
    errs['test_solid_growth_prescribed_iso_lv 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_growth_prescribed_iso_lv.py', branch])

    errs['test_solid_growthremodeling_fiberstretch 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_growthremodeling_fiberstretch.py', branch]) # only 1 element, cannot run on multiple cores

    errs['test_solid_2dheart_frankstarling 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_2dheart_frankstarling.py', branch])
    errs['test_solid_2dheart_frankstarling 3 restart'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_2dheart_frankstarling.py', str(2), branch])

    errs['test_solid_membrane 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_membrane.py', branch])
    errs['test_solid_membrane 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_membrane.py', branch])
    
    errs['test_solid_constraint_volume_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_constraint_volume_chamber.py', branch])
    errs['test_solid_constraint_volume_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_constraint_volume_chamber.py', branch])

if category=='fluid' or category=='all':
    errs['test_fluid_taylorhood_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_fluid_taylorhood_cylinder.py', branch])
    errs['test_fluid_taylorhood_cylinder 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_fluid_taylorhood_cylinder.py', branch])

    errs['test_fluid_p1p1_stab_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_fluid_p1p1_stab_cylinder.py', branch])
    errs['test_fluid_p1p1_stab_cylinder 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_fluid_p1p1_stab_cylinder.py', branch])

    errs['test_fluid_p1p1_stab_cylinder_schur2x2 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'test_fluid_p1p1_stab_cylinder_schur2x2.py', branch])
    
    errs['test_fluid_constraint_flux_p1p1_stab_cylinder 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_fluid_constraint_flux_p1p1_stab_cylinder.py', branch])
    
    if branch=='mixed':
        errs['test_fluid_p1p1_stab_cylinder_valve 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_fluid_p1p1_stab_cylinder_valve.py', branch])
        errs['test_fluid_p1p1_stab_cylinder_valve 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_fluid_p1p1_stab_cylinder_valve.py', branch])

if category=='fsi' or category=='all':
    if branch=='mixed':
        errs['test_fsi_taylorhood_artseg 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_fsi_taylorhood_artseg.py', branch])
        errs['test_fsi_taylorhood_artseg 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_fsi_taylorhood_artseg.py', branch])

        errs['test_fsi_p1p1_stab_artseg 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_fsi_p1p1_stab_artseg.py', branch])
        errs['test_fsi_p1p1_stab_artseg 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_fsi_p1p1_stab_artseg.py', branch])
        errs['test_fsi_p1p1_stab_artseg 2 restart'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_fsi_p1p1_stab_artseg.py', str(4), branch])

if category=='fsi_flow0d' or category=='all':
    if branch=='mixed':
        errs['test_fsi_flow0d_p1p1_stab_artseg 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_fsi_flow0d_p1p1_stab_artseg.py', branch])

if category=='fluid_flow0d' or category=='all':
    errs['test_fluid_flow0d_monolagr_taylorhood_cylinder 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_fluid_flow0d_monolagr_taylorhood_cylinder.py', branch])
    errs['test_fluid_flow0d_monolagr_taylorhood_cylinder 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_fluid_flow0d_monolagr_taylorhood_cylinder.py', branch])

    errs['test_fluid_flow0d_monolagr_taylorhood_cylinder_condensed 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_fluid_flow0d_monolagr_taylorhood_cylinder_condensed.py', branch])

if category=='fluid_ale_flow0d' or category=='all':
    if branch=='mixed':
        errs['test_fluid_ale_flow0d_lalv_syspul_prescribed 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'test_fluid_ale_flow0d_lalv_syspul_prescribed.py', branch])

if category=='flow0d' or category=='all':
    errs['test_flow0d_0dvol_2elwindkessel_n3 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dvol_2elwindkessel_n3.py', branch])
    
    errs['test_flow0d_0dvol_4elwindkesselLsZ 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dvol_4elwindkesselLsZ.py', branch])
    errs['test_flow0d_0dvol_4elwindkesselLpZ 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dvol_4elwindkesselLpZ.py', branch])

    errs['test_flow0d_0dheart_syspul 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dheart_syspul.py', branch])
    errs['test_flow0d_0dheart_syspul 1 restart'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dheart_syspul.py', str(450), branch]) # tests restart from step 450

    errs['test_flow0d_0dheart_syspulcor 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dheart_syspulcor.py', branch])
    errs['test_flow0d_0dheart_syspulcap 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dheart_syspulcap.py', branch])

    errs['test_flow0d_0dheart_syspulcaprespir_periodic 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_flow0d_0dheart_syspulcaprespir_periodic.py', branch])
    errs['test_flow0d_0dheart_syspulcaprespir_periodic 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_flow0d_0dheart_syspulcaprespir_periodic.py', branch])

if category=='solid_flow0d' or category=='all':
    errs['test_solid_flow0d_monodir_4elwindkesselLsZ_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_monodir_4elwindkesselLsZ_chamber.py', branch])
    errs['test_solid_flow0d_monodir_4elwindkesselLsZ_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monodir_4elwindkesselLsZ_chamber.py', branch])

    errs['test_solid_flow0d_monodir_4elwindkesselLsZ_chamber_bgs2x2 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monodir_4elwindkesselLsZ_chamber_bgs2x2.py', branch])
    errs['test_solid_flow0d_monodir_4elwindkesselLsZ_chamber_bgs2x2fieldsplit 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monodir_4elwindkesselLsZ_chamber_bgs2x2fieldsplit.py', branch])

    errs['test_solid_flow0d_monodir2field_4elwindkesselLpZ_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_monodir2field_4elwindkesselLpZ_chamber.py', branch])
    errs['test_solid_flow0d_monodir2field_4elwindkesselLpZ_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monodir2field_4elwindkesselLpZ_chamber.py', branch])

    errs['test_solid_flow0d_monolagr2field_2elwindkessel_chamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_monolagr2field_2elwindkessel_chamber.py', branch])
    errs['test_solid_flow0d_monolagr2field_2elwindkessel_chamber 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monolagr2field_2elwindkessel_chamber.py', branch])

    errs['test_solid_flow0d_monolagr_CRLinoutlink_chambers 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monolagr_CRLinoutlink_chambers.py', branch])

    errs['test_solid_flow0d_monodir_syspul_2dheart_prestress 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_monodir_syspul_2dheart_prestress.py', branch])
    errs['test_solid_flow0d_monodir_syspul_2dheart_prestress 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_flow0d_monodir_syspul_2dheart_prestress.py', branch])
    errs['test_solid_flow0d_monodir_syspul_2dheart_prestress 3 restart'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_flow0d_monodir_syspul_2dheart_prestress.py', str(3), branch]) # tests restart from step 3

    errs['test_solid_flow0d_monodir_flux_syspulcap_3Dheart_schur2x2 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_solid_flow0d_monodir_flux_syspulcap_3Dheart_schur2x2.py', branch])

    errs['test_solid_flow0d_monodir2field_flux_syspulcap_3Dheart_schur3x3 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'test_solid_flow0d_monodir2field_flux_syspulcap_3Dheart_schur3x3.py', branch])

    errs['test_solid_flow0d_monodir_syspulcor_2dheart_rom 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_monodir_syspulcor_2dheart_rom.py', branch])
    errs['test_solid_flow0d_monodir_syspulcor_2dheart_rom 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_solid_flow0d_monodir_syspulcor_2dheart_rom.py', branch])

    errs['test_solid_flow0d_periodicref_syspul_lvchamber 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_periodicref_syspul_lvchamber.py', branch])
    errs['test_solid_flow0d_periodicref_syspul_lvchamber 1 restart'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_solid_flow0d_periodicref_syspul_lvchamber.py', str(50), str(0), branch]) # TODO: Fix outer loop restart

if category=='frsi' or category=='fluid_ale' or category=='all':
    errs['test_frsi_artseg_prefile 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_frsi_artseg_prefile.py', branch])
    errs['test_frsi_artseg_prefile 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_frsi_artseg_prefile.py', branch])
    errs['test_frsi_artseg_prestress 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_frsi_artseg_prestress.py', branch])
    errs['test_frsi_artseg_prefile_partitioned_schur3x3 4'] = subprocess.call(['mpiexec', '-n', '4', 'python3', 'test_frsi_artseg_prefile_partitioned_schur3x3.py', branch])

    errs['test_frsi_artseg_modepartitionunity 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_frsi_artseg_modepartitionunity.py', branch])
    errs['test_frsi_artseg_modepartitionunity 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_frsi_artseg_modepartitionunity.py', branch])
    errs['test_frsi_artseg_modepartitionunity 2 restart'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_frsi_artseg_modepartitionunity.py', str(2), branch])

    errs['test_frsi_artseg_prefile_bgsschur4x4 3'] = subprocess.call(['mpiexec', '-n', '3', 'python3', 'test_frsi_artseg_prefile_bgsschur4x4.py', branch])

    errs['test_frsi_blocks_active 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_frsi_blocks_active.py', branch])
    
    errs['test_frsi_artseg_constraint 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_frsi_artseg_constraint.py', branch])

if category=='ale' or category=='all':
    errs['test_ale_linelast 1'] = subprocess.call(['mpiexec', '-n', '1', 'python3', 'test_ale_linelast.py', branch])
    errs['test_ale_linelast 2'] = subprocess.call(['mpiexec', '-n', '2', 'python3', 'test_ale_linelast.py', branch])


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
