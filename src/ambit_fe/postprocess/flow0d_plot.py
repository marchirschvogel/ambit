#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, subprocess, time
from pathlib import Path
import numpy as np
import argparse
import distutils.util
import importlib.util

# postprocessing script for flow0d model results (needs gnuplot to be installed when plot generation is desired)
# can/probably should be run outside of Docker container

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', dest='p', action='store', type=str, default='.') # output path
parser.add_argument('-s', '--simname', dest='s', action='store', type=str, default='') # name of simulation to postprocess
parser.add_argument('-n', '--nstep', dest='n', action='store', type=int, default=500) # number of time steps used in simulation
parser.add_argument('-ted', '--tenddias', dest='ted', action='store', type=float, default=0.2) # end-diastolic time point (relative to cycle time)
parser.add_argument('-tes', '--tendsyst', dest='tes', action='store', type=float, default=0.53) # end-systolic time point (relative to cycle time)
parser.add_argument('-T', '--Tcycl', dest='T', action='store', type=float, default=1.0) # cardiac cycle time
parser.add_argument('-m', '--model', dest='m', action='store', type=str, default='syspul') # type of 0D model
parser.add_argument('-mc', '--modelcoronary', dest='mc', action='store', default=None) # type of coronary sub-model: None, 'std_lr', 'std'
parser.add_argument('-cf', '--calcfunc', dest='cf', action='store', type=lambda x:bool(distutils.util.strtobool(x)), default=True) # whether to calculate funtion parameters (like stroke volume, cardiac output, ...)
parser.add_argument('-ip', '--inducepertafter', dest='ip', action='store', type=int, default=-1) # at which cycle a perturbation has been introduced (e.g. valvular defect/repair)
parser.add_argument('-mgr', '--multgandr', dest='mgr', action='store', type=lambda x:bool(distutils.util.strtobool(x)), default=False) # whether we have results from multiscale G&R analysis
parser.add_argument('-lgr', '--lastgandrcycl', dest='lgr', action='store', type=int, default=-1) # what cycle is last G&R cycle in case of results from multiscale G&R analysis
parser.add_argument('-V0', '--Vinitial', dest='V0', nargs=5, action='store', type=float, default=[113.25e3,150e3,50e3,50e3, 0e3]) # initial chamber vols: order is lv,rv,la,ra,ao
parser.add_argument('-png', '--pngexport', dest='png', action='store', type=lambda x:bool(distutils.util.strtobool(x)), default=True) # whether png files should be created for the plots
parser.add_argument('-plt', '--genplots', dest='plt', action='store', type=lambda x:bool(distutils.util.strtobool(x)), default=True) # whether plots should be generated
parser.add_argument('-ext', '--extplot', dest='ext', action='store', type=lambda x:bool(distutils.util.strtobool(x)), default=False) # whether some external data should be added to some plots (needs to be specified...)

def main():

    args = parser.parse_args()

    postprocess0D(args.p, args.s, args.n, args.T, args.ted, args.tes, args.m, args.mc, args.ip, calc_func_params=args.cf, V0=args.V0, multiscalegandr=args.mgr, lastgandrcycl=args.lgr, export_png=args.png, generate_plots=args.plt, ext_plot=args.ext)


def postprocess0D(path, sname, nstep_cycl, T_cycl, t_ed, t_es, model, coronarymodel, indpertaftercyl=0, calc_func_params=False, V0=[113.25e3,150e3,50e3,50e3, 0e3], multiscalegandr=False, lastgandrcycl=1, export_png=True, generate_plots=True, ext_plot=False):

    fpath = Path(__file__).parent.absolute()

    # return the groups we want to plot
    groups = []

    if model == 'syspul':

        spec = importlib.util.spec_from_file_location('cardiovascular0D_syspul', str(fpath)+'/../flow0d/cardiovascular0D_syspul.py')
        module_name = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_name)
        module_name.postprocess_groups_syspul(groups,coronarymodel,indpertaftercyl,multiscalegandr)
        iscirculation = True
        calculate_function_params = calc_func_params

    elif model == 'syspulcap':

        spec = importlib.util.spec_from_file_location('cardiovascular0D_syspulcap', str(fpath)+'/../flow0d/cardiovascular0D_syspulcap.py')
        module_name = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_name)
        module_name.postprocess_groups_syspulcap(groups,coronarymodel,indpertaftercyl,multiscalegandr)
        iscirculation = True
        calculate_function_params = calc_func_params

    elif model == 'syspulcapcor':

        spec = importlib.util.spec_from_file_location('cardiovascular0D_syspulcap', str(fpath)+'/../flow0d/cardiovascular0D_syspulcap.py')
        module_name = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_name)
        module_name.postprocess_groups_syspulcapcor(groups,coronarymodel,indpertaftercyl,multiscalegandr)
        iscirculation = True
        calculate_function_params = calc_func_params

    elif model == 'syspulcaprespir':

        spec = importlib.util.spec_from_file_location('cardiovascular0D_syspulcaprespir', str(fpath)+'/../flow0d/cardiovascular0D_syspulcaprespir.py')
        module_name = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_name)
        module_name.postprocess_groups_syspulcaprespir(groups,coronarymodel,indpertaftercyl,multiscalegandr)
        iscirculation = True
        calculate_function_params = calc_func_params

    elif model == '4elwindkesselLsZ':

        # TODO: Should we implement this?
        iscirculation = False
        pass

    elif model == '4elwindkesselLpZ':

        # TODO: Should we implement this?
        iscirculation = False
        pass

    elif model == '2elwindkessel':

        import cardiovascular0D_2elwindkessel
        cardiovascular0D_2elwindkessel.postprocess_groups(groups,indpertaftercyl)
        iscirculation = False

    else:

        raise NameError("Unknown 0D model!")

    # make a directory for the plots
    subprocess.call(['mkdir', '-p', path+'/plot0d_'+sname+'/'])

    if iscirculation:

        # get the data and check its length
        tmp = np.loadtxt(path+'/results_'+sname+'_p_ar_pul.txt', usecols=0) # could be another file - all should have the same length!
        numdata = len(tmp)

        # in case our chamber model was "rigid", neither flux nor volume files exist, so write dummies
        ch_dummy = []
        for i, ch in enumerate(['v_l','v_r','at_l','at_r', 'aort_sys']):
            test_Q = os.system('test -e '+path+'/results_'+sname+'_Q_'+ch+'.txt')
            test_V = os.system('test -e '+path+'/results_'+sname+'_V_'+ch+'.txt')
            if test_Q > 0 and test_V > 0:
                print(">>> WARNING: Neither volume nor flux file available for chamber %s! Writing dummy files." % (ch))
                file_dummy_V = open(path+'/results_'+sname+'_V_'+ch+'.txt', 'wt')
                for n in range(numdata):
                    file_dummy_V.write('%.16E %.16E\n' % (tmp[n], 0.0))
                file_dummy_V.close()
                file_dummy_Q = open(path+'/results_'+sname+'_Q_'+ch+'.txt', 'wt')
                for n in range(numdata):
                    file_dummy_Q.write('%.16E %.16E\n' % (tmp[n], 0.0))
                file_dummy_Q.close()
                ch_dummy.append(ch)

        # in case our coupling quantity was not volume, but flux or pressure, we should calculate the volume out of the flux data
        for i, ch in enumerate(['v_l','v_r','at_l','at_r', 'aort_sys']):
            if ch not in ch_dummy:
                # test if volume file exists
                test_V = os.system('test -e '+path+'/results_'+sname+'_V_'+ch+'.txt')
                if test_V > 0:
                    # safety check - flux file should exist in case of missing volume file!
                    test_Q = os.system('test -e '+path+'/results_'+sname+'_Q_'+ch+'.txt')
                    if test_Q == 0:
                        fluxes = np.loadtxt(path+'/results_'+sname+'_Q_'+ch+'.txt', usecols=1)
                        # integrate volume (mid-point rule): Q_{mid} = -(V_{n+1} - V_{n})/dt --> V_{n+1} = -Q_{mid}*dt + V_{n}
                        # --> V_{mid} = 0.5 * V_{n+1} + 0.5 * V_{n}
                        file_vol = open(path+'/results_'+sname+'_V_'+ch+'.txt', 'wt')
                        vol_n = V0[i]
                        file_vol.write('%.16E %.16E\n' % (tmp[0], vol_n))
                        for n in range(len(fluxes)-1):
                            dt = tmp[n+1] - tmp[n]
                            vol_np = -fluxes[n+1]*dt + vol_n
                            vol_mid = 0.5*vol_np + 0.5*vol_n
                            file_vol.write('%.16E %.16E\n' % (tmp[n+1], vol_mid))
                            vol_n = vol_np
                        file_vol.close()

        # in case our coupling quantity was not flux or pressure, but volume, we could calculate the chamber fluxes Q
        for i, ch in enumerate(['v_l','v_r','at_l','at_r', 'aort_sys']):
            if ch not in ch_dummy:
                # test if flux file exists
                test_Q = os.system('test -e '+path+'/results_'+sname+'_Q_'+ch+'.txt')
                if test_Q > 0:
                    # safety check - volume file should exist in case of missing flux file!
                    test_V = os.system('test -e '+path+'/results_'+sname+'_V_'+ch+'.txt')

                    if test_V == 0:

                        if ch=='v_l':
                            flux_i = np.loadtxt(path+'/results_'+sname+'_q_vin_l.txt', usecols=1)
                            flux_o = np.loadtxt(path+'/results_'+sname+'_q_vout_l.txt', usecols=1)
                        elif ch=='v_r':
                            flux_i = np.loadtxt(path+'/results_'+sname+'_q_vin_r.txt', usecols=1)
                            flux_o = np.loadtxt(path+'/results_'+sname+'_q_vout_r.txt', usecols=1)
                        elif ch=='at_l':
                            flux_i = np.loadtxt(path+'/results_'+sname+'_q_ven1_pul.txt', usecols=1)
                            flux_o = np.loadtxt(path+'/results_'+sname+'_q_vin_l.txt', usecols=1)
                        elif ch=='at_r':
                            flux_i = np.loadtxt(path+'/results_'+sname+'_q_ven1_sys.txt', usecols=1)
                            flux_o = np.loadtxt(path+'/results_'+sname+'_q_vin_r.txt', usecols=1)
                        elif ch=='aort_sys':
                            flux_i = np.loadtxt(path+'/results_'+sname+'_q_vout_l.txt', usecols=1)
                            flux_o = np.loadtxt(path+'/results_'+sname+'_q_arp_sys.txt', usecols=1)
                        else:
                            raise NameError("Unknown chamber/compartment!")

                        flux = -flux_i + flux_o # -Q_ch = q_ch_in - q_ch_out
                        file_flx = open(path+'/results_'+sname+'_Q_'+ch+'.txt', 'wt')
                        for n in range(len(flux)):
                            file_flx.write('%.16E %.16E\n' % (tmp[n], flux[n]))
                        file_flx.close()

        # check number of veins
        sysveins, pulveins = 0, 0
        for i in range(10):
            if os.system('test -e '+path+'/results_'+sname+'_q_ven'+str(i+1)+'_sys.txt')==0: sysveins += 1
            if os.system('test -e '+path+'/results_'+sname+'_q_ven'+str(i+1)+'_pul.txt')==0: pulveins += 1

        # in 3D fluid dynamics, we may have "distributed" 0D in-/outflow pressures, so here we check presence of these
        # and then average them for visualization
        # check presence of default chamber pressure variable
        for ch in ['v_l','v_r','at_l','at_r', 'aort_sys']:
            err = os.system('test -e '+path+'/results_'+sname+'_p_'+ch+'.txt')
            if ch=='aort_sys': err = os.system('test -e '+path+'/results_'+sname+'_p_ar_sys.txt') # extra check due to naming conventions...
            if err==0: # nothing to do if present
                pass
            else:
                numpi, numpo = 0, 0
                # now check chamber inflow/outflow distributed pressures
                pall = np.zeros(numdata)
                for i in range(10):
                    if os.system('test -e '+path+'/results_'+sname+'_p_'+ch+'_i'+str(i+1)+'.txt')==0: numpi += 1
                    if os.system('test -e '+path+'/results_'+sname+'_p_'+ch+'_o'+str(i+1)+'.txt')==0: numpo += 1
                for i in range(numpi):
                    pi = np.loadtxt(path+'/results_'+sname+'_p_'+ch+'_i'+str(i+1)+'.txt', usecols=1)
                    for j in range(len(pall)):
                        pall[j] += pi[j]/(numpi+numpo)
                for i in range(numpo):
                    po = np.loadtxt(path+'/results_'+sname+'_p_'+ch+'_o'+str(i+1)+'.txt', usecols=1)
                    for j in range(len(pall)):
                        pall[j] += po[j]/(numpi+numpo)

                # write averaged pressure file
                file_pavg = path+'/results_'+sname+'_p_'+ch+'.txt'
                fpa = open(file_pavg, 'wt')
                for i in range(len(pall)):
                    fpa.write('%.16E %.16E\n' % (tmp[i], pall[i]))
                fpa.close()
                # rename file to ar_sys - due to naming conventions...
                if ch=='aort_sys': os.system('mv '+path+'/results_'+sname+'_p_'+ch+'.txt '+path+'/results_'+sname+'_p_ar_sys.txt')

        # for plotting of pressure-volume loops
        for ch in ['v_l','v_r','at_l','at_r']:
            subprocess.call(['cp', path+'/results_'+sname+'_p_'+ch+'.txt', path+'/results_'+sname+'_p_'+ch+'_tmp.txt'])
            subprocess.call(['cp', path+'/results_'+sname+'_V_'+ch+'.txt', path+'/results_'+sname+'_V_'+ch+'_tmp.txt'])
            # drop first (time) columns
            subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', path+'/results_'+sname+'_p_'+ch+'_tmp.txt'])
            subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', path+'/results_'+sname+'_V_'+ch+'_tmp.txt'])
            # paste files together
            os.system('paste '+path+'/results_'+sname+'_V_'+ch+'_tmp.txt '+path+'/results_'+sname+'_p_'+ch+'_tmp.txt > '+path+'/results_'+sname+'_pV_'+ch+'.txt')
            # isolate last cycle
            os.system('tail -n '+str(nstep_cycl)+' '+path+'/results_'+sname+'_pV_'+ch+'.txt > '+path+'/results_'+sname+'_pV_'+ch+'_last.txt')
            if multiscalegandr and indpertaftercyl > 0:
                subprocess.call(['cp', path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_p_'+ch+'.txt', path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_p_'+ch+'_tmp.txt'])
                subprocess.call(['cp', path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_V_'+ch+'.txt', path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_V_'+ch+'_tmp.txt'])
                # drop first (time) columns
                subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_p_'+ch+'_tmp.txt'])
                subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_V_'+ch+'_tmp.txt'])
                # paste files together
                os.system('paste '+path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_V_'+ch+'_tmp.txt '+path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_p_'+ch+'_tmp.txt > '+path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_pV_'+ch+'.txt')
                # isolate last cycle
                os.system('tail -n '+str(nstep_cycl)+' '+path+'/results_'+sname.replace('small1','small'+str(lastgandrcycl))+'_pV_'+ch+'.txt > '+path+'/results_'+sname+'_pV_'+ch+'_gandr.txt')
            # isolate healthy/baseline cycle
            if indpertaftercyl > 0:
                os.system('sed -n "'+str((indpertaftercyl-1)*nstep_cycl+1)+','+str(indpertaftercyl*nstep_cycl)+'p" '+path+'/results_'+sname+'_pV_'+ch+'.txt > '+path+'/results_'+sname+'_pV_'+ch+'_baseline.txt')
            # clean-up
            subprocess.call(['rm', path+'/results_'+sname+'_p_'+ch+'_tmp.txt'])
            subprocess.call(['rm', path+'/results_'+sname+'_V_'+ch+'_tmp.txt'])

        # for plotting of compartment volumes: gather all volumes and add them in order to check if volume conservation is fulfilled!
        # Be worried if the total sum in V_all.txt changes over time (more than to a certain tolerance)!
        volall = np.zeros(numdata)
        for c in range(len(list(groups[5].values())[0])-1): # compartment volumes should be stored in group index 5
            # load volume data
            vols = np.loadtxt(path+'/results_'+sname+'_'+list(groups[5].values())[0][c]+'.txt', usecols=1)
            # add together
            for i in range(len(volall)):
                volall[i] += vols[i]

        # write time and vol value to file
        file_vollall = path+'/results_'+sname+'_V_all.txt'
        fva = open(file_vollall, 'wt')
        for i in range(len(volall)):
            fva.write('%.16E %.16E\n' % (tmp[i], volall[i]))
        fva.close()

        # compute integral data
        file_integral = path+'/results_'+sname+'_data_integral.txt'
        fi = open(file_integral, 'wt')

        fi.write('T_cycl ' +str(T_cycl) + '\n')
        fi.write('N_step ' +str(nstep_cycl) + '\n')

        # function parameters of left and right ventricle
        if calculate_function_params:

            # number of heart cycles
            n_cycl = int(numdata/nstep_cycl)
            t_off = tmp[0]-T_cycl/nstep_cycl

            sw, sv, co, ef, edv, esv, vmin, vmax, vend, edp, esp, sv_net, co_net, ef_net, v_reg, f_reg = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            for ch in ['v_l','v_r']:

                # stroke work
                pv = np.loadtxt(path+'/results_'+sname+'_pV_'+ch+'_last.txt') # this is already last (periodic) cycle pv data!
                val = 0.0
                for k in range(len(pv)-1):
                    # we need the negative sign since we go counter-clockwise around the loop!
                    val -= 0.5*(pv[k+1,1]+pv[k,1]) * (pv[k+1,0] - pv[k,0])
                sw.append(val)

                # stroke volume, cardiac output, end-diastolic and end-systolic volume, ejection fraction
                vol = np.loadtxt(path+'/results_'+sname+'_V_'+ch+'.txt', skiprows=max(0,numdata-nstep_cycl))
                vmin.append(min(vol[:,1]))
                vmax.append(max(vol[:,1]))
                vend.append(vol[-1,1])
                edv.append(np.interp(t_ed+(n_cycl-1)*T_cycl+t_off, vol[:,0], vol[:,1]))
                esv.append(np.interp(t_es+(n_cycl-1)*T_cycl+t_off, vol[:,0], vol[:,1]))
                sv.append(max(vol[:,1])-min(vol[:,1]))
                co.append((max(vol[:,1])-min(vol[:,1]))/T_cycl)
                ef.append((max(vol[:,1])-min(vol[:,1]))/max(vol[:,1]))

                pres = np.loadtxt(path+'/results_'+sname+'_p_'+ch+'.txt', skiprows=max(0,numdata-nstep_cycl))

                # end-diastolic pressure
                edp.append(np.interp(t_ed+(n_cycl-1)*T_cycl+t_off, pres[:,0], pres[:,1]))

                # end-systolic pressure
                esp.append(np.interp(t_es+(n_cycl-1)*T_cycl+t_off, pres[:,0], pres[:,1]))

                # net values (in case of regurgitation of valves, for example), computed by integrating in- and out-fluxes
                if ch=='v_l':
                    fluxout = np.loadtxt(path+'/results_'+sname+'_q_vout_l.txt', skiprows=max(0,numdata-nstep_cycl))
                    fluxin = np.loadtxt(path+'/results_'+sname+'_q_vin_l.txt', skiprows=max(0,numdata-nstep_cycl))
                if ch=='v_r':
                    fluxout = np.loadtxt(path+'/results_'+sname+'_q_vout_r.txt', skiprows=max(0,numdata-nstep_cycl))
                    fluxin = np.loadtxt(path+'/results_'+sname+'_q_vin_r.txt', skiprows=max(0,numdata-nstep_cycl))

                # true (net) stroke volume
                val = 0.0
                for i in range(len(fluxout)-1):
                    # mid-point rule
                    val += 0.5*(fluxout[i+1,1]+fluxout[i,1]) * (fluxout[i+1,0]-fluxout[i,0])
                sv_net.append(val)
                co_net.append(val/T_cycl)

                # true (net) ejection fraction
                ef_net.append(sv_net[-1]/edv[-1])

                # regurgitant volume
                val = 0.0
                for i in range(len(fluxin)-1):
                    # mid-point rule
                    if fluxin[i+1,1] < 0.:
                        val += 0.5*(fluxin[i+1,1]+fluxin[i,1]) * (fluxin[i+1,0]-fluxin[i,0])
                v_reg.append(abs(val))

                # regurgitant fraction
                f_reg.append(v_reg[-1]/sv[-1])

            # atrial min, max, and stroke volumes
            vmin_at, vmax_at, vend_at, sv_at = [], [], [], []
            for ch in ['at_l','at_r']:
                vol_at = np.loadtxt(path+'/results_'+sname+'_V_'+ch+'.txt', skiprows=max(0,numdata-nstep_cycl))
                vmin_at.append(min(vol_at[:,1]))
                vmax_at.append(max(vol_at[:,1]))
                sv_at.append(max(vol_at[:,1])-min(vol_at[:,1]))
                vend_at.append(vol_at[-1,1])

            # mean arterial pressure
            marp = []
            for pc in ['ar_sys','ar_pul']:
                pr = np.loadtxt(path+'/results_'+sname+'_p_'+pc+'.txt', skiprows=max(0,numdata-nstep_cycl))
                val = 0.0
                for k in range(len(pr)-1):
                    val += 0.5*(pr[k+1,1]+pr[k,1]) * (pr[k+1,0] - pr[k,0])
                val /= (pr[-1,0]-pr[0,0])
                marp.append(val)

            # systolic and diastolic blood pressures
            p_ar_dias, p_ar_syst = [], []
            for pc in ['ar_sys','ar_pul']:
                par = np.loadtxt(path+'/results_'+sname+'_p_'+pc+'.txt', skiprows=max(0,numdata-nstep_cycl))
                p_ar_dias.append(min(par[:,1]))
                p_ar_syst.append(max(par[:,1]))

            # distal systemic arterial blood pressure (should only differ significantly if Z_ar_sys and/or I_ar_sys != 0)
            pard = np.loadtxt(path+'/results_'+sname+'_p_ard_sys.txt', skiprows=max(0,numdata-nstep_cycl))
            p_ard_dias = min(pard[:,1])
            p_ard_syst = max(pard[:,1])

            # mean atrial pressure
            mpat = []
            for pc in ['at_l','at_r']:
                pr = np.loadtxt(path+'/results_'+sname+'_p_'+pc+'.txt', skiprows=max(0,numdata-nstep_cycl))
                val = 0.0
                for k in range(len(pr)-1):
                    val += 0.5*(pr[k+1,1]+pr[k,1]) * (pr[k+1,0] - pr[k,0])
                val /= (pr[-1,0]-pr[0,0])
                mpat.append(val)

            # end-cyclic pressures
            pend = []
            for pc in ['v_l','v_r','at_l','at_r']:
                pr = np.loadtxt(path+'/results_'+sname+'_p_'+pc+'.txt', skiprows=max(0,numdata-nstep_cycl))
                pend.append(pr[-1,1])

            # we assume here that units kg - mm - s are used --> pressures are kPa, forces are mN, volumes are mm^3
            fi.write('sw_lv %.16f\n' % (sw[0]))
            fi.write('sw_rv %.16f\n' % (sw[1]))
            fi.write('sv_lv %.16f\n' % (sv[0]))
            fi.write('sv_rv %.16f\n' % (sv[1]))
            fi.write('co_lv %.16f\n' % (co[0]))
            fi.write('co_rv %.16f\n' % (co[1]))
            fi.write('ef_lv %.16f\n' % (ef[0]))
            fi.write('ef_rv %.16f\n' % (ef[1]))
            fi.write('edv_lv %.16f\n' % (edv[0]))
            fi.write('edv_rv %.16f\n' % (edv[1]))
            fi.write('esv_lv %.16f\n' % (esv[0]))
            fi.write('esv_rv %.16f\n' % (esv[1]))
            fi.write('vmin_lv %.16f\n' % (vmin[0]))
            fi.write('vmin_rv %.16f\n' % (vmin[1]))
            fi.write('vmax_lv %.16f\n' % (vmax[0]))
            fi.write('vmax_rv %.16f\n' % (vmax[1]))
            fi.write('vend_lv %.16f\n' % (vend[0]))
            fi.write('vend_rv %.16f\n' % (vend[1]))
            fi.write('edp_lv %.16f\n' % (edp[0]))
            fi.write('edp_rv %.16f\n' % (edp[1]))
            fi.write('esp_lv %.16f\n' % (esp[0]))
            fi.write('esp_rv %.16f\n' % (esp[1]))
            fi.write('map_sys %.16f\n' % (marp[0]))
            fi.write('map_pul %.16f\n' % (marp[1]))
            fi.write('sv_net_lv %.16f\n' % (sv_net[0]))
            fi.write('sv_net_rv %.16f\n' % (sv_net[1]))
            fi.write('co_net_lv %.16f\n' % (co_net[0]))
            fi.write('co_net_rv %.16f\n' % (co_net[1]))
            fi.write('ef_net_lv %.16f\n' % (ef_net[0]))
            fi.write('ef_net_rv %.16f\n' % (ef_net[1]))
            fi.write('v_reg_lv %.16f\n' % (v_reg[0]))
            fi.write('v_reg_rv %.16f\n' % (v_reg[1]))
            fi.write('f_reg_lv %.16f\n' % (f_reg[0]))
            fi.write('f_reg_rv %.16f\n' % (f_reg[1]))
            fi.write('p_ard_sys_dias %.16f\n' % (p_ard_dias))
            fi.write('p_ard_sys_syst %.16f\n' % (p_ard_syst))
            fi.write('p_ar_sys_dias %.16f\n' % (p_ar_dias[0]))
            fi.write('p_ar_sys_syst %.16f\n' % (p_ar_syst[0]))
            fi.write('p_ar_pul_dias %.16f\n' % (p_ar_dias[1]))
            fi.write('p_ar_pul_syst %.16f\n' % (p_ar_syst[1]))
            fi.write('mpat_l %.16f\n' % (mpat[0]))
            fi.write('mpat_r %.16f\n' % (mpat[1]))
            fi.write('sv_la %.16f\n' % (sv_at[0]))
            fi.write('sv_ra %.16f\n' % (sv_at[1]))
            fi.write('vmin_la %.16f\n' % (vmin_at[0]))
            fi.write('vmin_ra %.16f\n' % (vmin_at[1]))
            fi.write('vmax_la %.16f\n' % (vmax_at[0]))
            fi.write('vmax_ra %.16f\n' % (vmax_at[1]))
            fi.write('vend_la %.16f\n' % (vend_at[0]))
            fi.write('vend_ra %.16f\n' % (vend_at[1]))
            fi.write('pend_lv %.16f\n' % (pend[0]))
            fi.write('pend_rv %.16f\n' % (pend[1]))
            fi.write('pend_la %.16f\n' % (pend[2]))
            fi.write('pend_ra %.16f\n' % (pend[3]))

            fi.close()

    if generate_plots:

        # tmp!!!!
        if ext_plot:
            grind, grname = 4, 'vol_time_l_r'
            groups[grind][grname].pop(-1)
            groups[grind][grname].pop(-1)
            groups[grind]['tex'].pop(-1)
            groups[grind]['tex'].pop(-1)
            groups[grind]['lines'].pop(-1)
            groups[grind]['lines'].pop(-1)
            groups[grind][grname].append('Meas_Vlv')
            groups[grind][grname].append('Meas_Vla')
            groups[grind]['tex'].append('$\\\hat{V}_{\\\mathrm{lv}}$')
            groups[grind]['tex'].append('$\\\hat{V}_{\\\mathrm{la}}$')
            groups[grind]['lines'].append(300)
            groups[grind]['lines'].append(301)

        for g in range(len(groups)):

            numitems = len(list(groups[g].values())[0])

            # safety (and sanity...) check
            if numitems > 18:
                print("More than 18 items to plot in one graph! Adjust plotfile template or consider if this is sane...")
                sys.exit()

            subprocess.call(['cp', str(fpath)+'/flow0d_gnuplot_template.p', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's#__OUTDIR__#'+path+'/plot0d_'+sname+'/#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's#__FILEDIR__#'+path+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])

            subprocess.call(['sed', '-i', 's/__OUTNAME__/'+list(groups[g].keys())[0]+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])

            factor_kPa_mmHg = 7.500615

            if 'pres_time' in list(groups[g].keys())[0]:
                x1value, x2value     = 't', ''
                x1unit, x2unit       = 's', ''
                y1value, y2value     = 'p', 'p'
                y1unit, y2unit       = 'kPa', 'mmHg'
                xscale, yscale       = 1.0, 1.0
                x2rescale, y2rescale = 1.0, factor_kPa_mmHg
                xextend, yextend     = 1.0, 1.1
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
                if (model == 'syspulcap' or model == 'syspulcapcor' or model == 'syspulcaprespir') and 'pres_time_sys_l' in list(groups[g].keys())[0]:
                    xextend, yextend     = 1.0, 1.2
                    maxrows, maxcols, sl, swd = 2, 5, 19, 50
            if 'flux_time' in list(groups[g].keys())[0]:
                x1value, x2value     = 't', ''
                x1unit, x2unit       = 's', ''
                y1value, y2value     = 'q', ''
                y1unit, y2unit       = 'ml/s', ''
                xscale, yscale       = 1.0, 1.0e-3
                x2rescale, y2rescale = 1.0, 1.0
                xextend, yextend     = 1.0, 1.1
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
                if (model == 'syspulcap' or model == 'syspulcapcor' or model == 'syspulcaprespir') and 'flux_time_sys_l' in list(groups[g].keys())[0]:
                    xextend, yextend     = 1.0, 1.3
                    maxrows, maxcols, sl, swd = 3, 5, 20, 50
                if 'flux_time_pul_r' in list(groups[g].keys())[0] and pulveins > 2:
                    maxrows, maxcols, sl, swd = 1, 7, 16, 34
                if 'flux_time_cor' in list(groups[g].keys())[0]:
                    maxrows, maxcols, sl, swd = 1, 6, 13, 41
            if 'flux_time_compart' in list(groups[g].keys())[0]:
                y1value, y2value     = 'Q', ''
            if 'vol_time' in list(groups[g].keys())[0]:
                x1value, x2value     = 't', ''
                x1unit, x2unit       = 's', ''
                y1value, y2value     = 'V', ''
                y1unit, y2unit       = 'ml', ''
                xscale, yscale       = 1.0, 1.0e-3
                x2rescale, y2rescale = 1.0, 1.0
                xextend, yextend     = 1.0, 1.1
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
            if 'vol_time_compart' in list(groups[g].keys())[0]:
                xextend, yextend     = 1.0, 1.2
                maxrows, maxcols, sl, swd = 2, 5, 20, 50
                if (model == 'syspulcap' or model == 'syspulcapcor' or model == 'syspulcaprespir'):
                    xextend, yextend     = 1.0, 1.3
                    maxrows, maxcols, sl, swd = 3, 5, 10, 50
                if coronarymodel == 'ZCRp_CRd_lr' or coronarymodel == 'std_lr':
                    maxrows, maxcols, sl, swd = 2, 5, 20, 40
                if coronarymodel == 'ZCRp_CRd' or coronarymodel == 'std': # TODO: Same settings as for _lr?
                    maxrows, maxcols, sl, swd = 2, 5, 20, 40
            if 'pres_vol_v' in list(groups[g].keys())[0]:
                x1value, x2value     = 'V_{\\\mathrm{v}}', ''
                x1unit, x2unit       = 'ml', ''
                y1value, y2value     = 'p_{\\\mathrm{v}}', 'p_{\\\mathrm{v}}'
                y1unit, y2unit       = 'kPa', 'mmHg'
                xscale, yscale       = 1.0e-3, 1.0
                x2rescale, y2rescale = 1.0, factor_kPa_mmHg
                xextend, yextend     = 1.1, 1.1
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
                if multiscalegandr: sl, swd = 19, 33
            if 'pres_vol_at' in list(groups[g].keys())[0]:
                x1value, x2value     = 'V_{\\\mathrm{at}}', ''
                x1unit, x2unit       = 'ml', ''
                y1value, y2value     = 'p_{\\\mathrm{at}}', 'p_{\\\mathrm{at}}'
                y1unit, y2unit       = 'kPa', 'mmHg'
                xscale, yscale       = 1.0e-3, 1.0
                x2rescale, y2rescale = 1.0, factor_kPa_mmHg
                xextend, yextend     = 1.1, 1.1
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
                if multiscalegandr: sl, swd = 19, 33
            if 'ppO2_time' in list(groups[g].keys())[0]:
                x1value, x2value     = 't', ''
                x1unit, x2unit       = 's', ''
                y1value, y2value     = 'p_{\\\mathrm{O}_2}', 'p_{\\\mathrm{O}_2}'
                y1unit, y2unit       = 'kPa', 'mmHg'
                xscale, yscale       = 1.0, 1.0
                x2rescale, y2rescale = 1.0, factor_kPa_mmHg
                xextend, yextend     = 1.0, 1.2
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
                if 'sys_l' in list(groups[g].keys())[0]:
                    xextend, yextend     = 1.0, 1.3
                    maxrows, maxcols, sl, swd = 3, 5, 10, 50
            if 'ppCO2_time' in list(groups[g].keys())[0]:
                x1value, x2value     = 't', ''
                x1unit, x2unit       = 's', ''
                y1value, y2value     = 'p_{\\\mathrm{CO}_2}', 'p_{\\\mathrm{CO}_2}'
                y1unit, y2unit       = 'kPa', 'mmHg'
                xscale, yscale       = 1.0, 1.0
                x2rescale, y2rescale = 1.0, factor_kPa_mmHg
                xextend, yextend     = 1.0, 1.2
                maxrows, maxcols, sl, swd = 1, 5, 20, 50
                if 'sys_l' in list(groups[g].keys())[0]:
                    xextend, yextend     = 1.0, 1.3
                    maxrows, maxcols, sl, swd = 3, 5, 10, 50

            data = []
            x_s_all, x_e_all = [], []
            y_s_all, y_e_all = [], []

            for q in range(numitems):

                prfx = 'results_'+sname+'_'

                # continue if file does not exist
                if os.system('test -e '+path+'/'+prfx+list(groups[g].values())[0][q]+'.txt') > 0:
                    continue

                # get the data and check its length
                tmp = np.loadtxt(path+'/'+prfx+list(groups[g].values())[0][q]+'.txt') # could be another file - all should have the same length!
                numdata = len(tmp)

                # set quantity, title, and plotting line
                subprocess.call(['sed', '-i', 's/__QTY'+str(q+1)+'__/'+prfx+list(groups[g].values())[0][q]+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
                subprocess.call(['sed', '-i', 's/__TIT'+str(q+1)+'__/'+list(groups[g].values())[1][q]+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
                subprocess.call(['sed', '-i', 's/__LIN'+str(q+1)+'__/'+str(list(groups[g].values())[2][q])+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])

                # adjust the plotting command to include all the files to plot in one graph
                if q!=0: subprocess.call(['sed', '-i', 's/#__'+str(q+1)+'__//g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])

                if 'PERIODIC' in list(groups[g].keys())[0]: skip = max(0,numdata-nstep_cycl)
                else: skip = 0

                # get the x,y range on which to plot
                data.append(np.loadtxt(path+'/'+prfx+list(groups[g].values())[0][q]+'.txt', skiprows=skip))

                # if time is our x-axis
                if 'time' in list(groups[g].keys())[0]:
                    x_s_all.append(min(data[q][:,0]))
                else: # start plots from x=0 even if data is larger than zero
                    if min(data[q][:,0]) > 0.0: x_s_all.append(0.0)
                    else: x_s_all.append(min(data[q][:,0]))

                x_e_all.append(max(data[q][:,0]))

                # start plots from y=0 even if data is larger than zero
                if min(data[q][:,1]) > 0.0: y_s_all.append(0.0)
                else: y_s_all.append(min(data[q][:,1]))

                y_e_all.append(max(data[q][:,1]))

            # get the min and the max of all x's and y's
            x_s, x_e = xscale*min(x_s_all), xscale*max(x_e_all)
            #x_s, x_e = 0.0, xscale*max(x_e_all)
            y_s, y_e = yscale*min(y_s_all), yscale*max(y_e_all)

            # nothing to print if we have a vanishing y-range
            if abs(y_e-y_s) <= 1.0e-16:
                continue

            # if we want to use a x2 or y2 axis
            if x2value != '': subprocess.call(['sed', '-i', 's/#__HAVEX2__//', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            if y2value != '': subprocess.call(['sed', '-i', 's/#__HAVEY2__//', path+'/plot_'+list(groups[g].keys())[0]+'.p'])

            # axis segments - x
            subprocess.call(['sed', '-i', 's/__X1S__/'+str(x_s)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__X1E__/'+str(x_e*xextend)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__X2S__/'+str(x2rescale*x_s)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__X2E__/'+str(x2rescale*x_e*xextend)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            # axis segments - y
            subprocess.call(['sed', '-i', 's/__Y1S__/'+str(y_s)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__Y1E__/'+str(y_e*yextend)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__Y2S__/'+str(y2rescale*y_s)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__Y2E__/'+str(y2rescale*y_e*yextend)+'/', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            # units
            subprocess.call(['sed', '-i', 's#__X1UNIT__#'+x1unit+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's#__Y1UNIT__#'+y1unit+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            if x2unit != '': subprocess.call(['sed', '-i', 's#__X2UNIT__#'+x2unit+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            if y2unit != '': subprocess.call(['sed', '-i', 's#__Y2UNIT__#'+y2unit+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            # values
            subprocess.call(['sed', '-i', 's#__X1VALUE__#'+x1value+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's#__Y1VALUE__#'+y1value+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            if x2value != '': subprocess.call(['sed', '-i', 's#__X2VALUE__#'+x2value+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            if y2value != '': subprocess.call(['sed', '-i', 's#__Y2VALUE__#'+y2value+'#', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            # scales
            subprocess.call(['sed', '-i', 's/__XSCALE__/'+str(xscale)+'/g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__YSCALE__/'+str(yscale)+'/g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            # rows, columns and sample length for legend
            subprocess.call(['sed', '-i', 's/__MAXROWS__/'+str(maxrows)+'/g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__MAXCOLS__/'+str(maxcols)+'/g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__SAMPLEN__/'+str(sl)+'/g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__SAMPWID__/'+str(swd)+'/g', path+'/plot_'+list(groups[g].keys())[0]+'.p'])

            # do the plotting
            subprocess.call(['gnuplot', path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            # convert to PDF
            subprocess.call(['ps2pdf', '-dEPSCrop', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.eps', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.pdf'])
            subprocess.call(['pdflatex', '-interaction=batchmode', '-output-directory='+path+'/plot0d_'+sname+'/', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.tex'])

            if export_png:
                subprocess.call(['pdftoppm', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.pdf', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0], '-png', '-rx', '300', '-ry', '300'])
                subprocess.call(['mv', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-1.png', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.png']) # output has -1, so rename
                # delete PDFs
                subprocess.call(['rm', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.pdf'])

            # clean up
            subprocess.call(['rm', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.aux', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.log'])
            # guess we do not need these files anymore since we have the final PDF...
            subprocess.call(['rm', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.tex'])
            subprocess.call(['rm', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.pdf'])
            subprocess.call(['rm', path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.eps'])
            # delete gnuplot file
            subprocess.call(['rm', path+'/plot_'+list(groups[g].keys())[0]+'.p'])


if __name__ == "__main__":

    main()
