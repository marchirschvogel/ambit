#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys, os, subprocess, time
import math
from pathlib import Path
import numpy as np


def main():
    
    try: # from command line
        path = sys.argv[1]
        sname = sys.argv[2]
        nstep_cycl = int(sys.argv[3])
        T_cycl = float(sys.argv[4])
        t_ed = float(sys.argv[5])
        t_es = float(sys.argv[6])
        model = sys.argv[7]
        indpertaftercyl = int(sys.argv[8])
        theta = float(sys.argv[9])
        calc_func_params = str_to_bool(sys.argv[10])
    except:
        path = '/home/mh/work/sim/heart3D4ch/p0/01/growthremodeling_ecc/0D'#'/home/mh/Downloads/marc_input/tmp/0D'#'/home/mh/work/ambit/testing/tmp/'
        sname = 'multiscale_eccentric_mr1_small1'
        nstep_cycl = 100
        T_cycl = 1.0
        t_ed = 0.2
        t_es = 0.53
        model = 'syspulcap'
        indpertaftercyl = 1
        theta = 0.5
        calc_func_params = True
    
    postprocess0D(path, sname, nstep_cycl, T_cycl, t_ed, t_es, model, indpertaftercyl, theta, calc_func_params)


def postprocess0D(path, sname, nstep_cycl, T_cycl, t_ed, t_es, model, indpertaftercyl=0, theta=0.5, calc_func_params=False):

    fpath = Path(__file__).parent.absolute()
    
    # return the groups we want to plot
    groups = []
    
    if model == 'syspul':
        
        import cardiovascular0D_syspul
        cardiovascular0D_syspul.postprocess_groups_syspul(groups,indpertaftercyl)
        iscirculation = True
        calculate_function_params = calc_func_params

    elif model == 'syspulcap':
        
        import cardiovascular0D_syspulcap
        cardiovascular0D_syspulcap.postprocess_groups_syspulcap(groups,indpertaftercyl)
        iscirculation = True
        calculate_function_params = calc_func_params
        
    elif model == 'syspulcap2':
        
        import cardiovascular0D_syspulcap
        cardiovascular0D_syspulcap.postprocess_groups_syspulcap2(groups,indpertaftercyl)
        iscirculation = True
        calculate_function_params = calc_func_params
        
    elif model == 'syspulcaprespir':
        
        import cardiovascular0D_syspulcaprespir
        cardiovascular0D_syspulcaprespir.postprocess_groups_syspulcaprespir(groups,indpertaftercyl)
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
    subprocess.call(['mkdir', '-p', ''+path+'/plot0d_'+sname+'/'])

    if iscirculation:

        # get the data and check its length
        tmp = np.loadtxt(''+path+'/results_'+sname+'_p_v_l.txt', usecols=0) # could be another file - all should have the same length!
        numdata = len(tmp)
        # number of heart cycles
        n_cycl = int(numdata/nstep_cycl)
        
        # in case our coupling quantity was not volume, but flux or pressure, we should calculate the volume out of the flux data
        for ch in ['v_l','v_r','at_l','at_r']:
            # test if volume file exists
            test_V = os.system('test -e '+path+'/results_'+sname+'_V_'+ch+'.txt')
            if test_V > 0:
                # safety check - flux file should exist in case of missing volume file!
                test_Q = os.system('test -e '+path+'/results_'+sname+'_Q_'+ch+'.txt')
                if test_Q == 0:
                    fluxes = np.loadtxt(''+path+'/results_'+sname+'_Q_'+ch+'.txt', usecols=1)
                    # integrate volume: Q_{n+theta} = -(V_{n+1} - V_{n})/dt --> V_{n+1} = -Q_{n+theta}*dt + V_{n}
                    # --> V_{n+theta} = theta * V_{n+1} + (1-theta) * V_{n}
                    filename_vol = path+'/results_'+sname+'_V_'+ch+'.txt'
                    file_vol = open(filename_vol, 'wt')
                    vol_n = 0.0 # we do not have the initial volume...
                    file_vol.write('%.16E %.16E\n' % (tmp[0], vol_n))
                    for i in range(len(fluxes)-1):
                        dt = tmp[i+1] - tmp[i]
                        vol_np = -fluxes[i+1]*dt + vol_n
                        vol_ntheta = theta*vol_np + (1.-theta)*vol_n
                        file_vol.write('%.16E %.16E\n' % (tmp[i+1], vol_ntheta))
                        vol_n = vol_np
                    file_vol.close()
                else:
                    raise AttributeError("No flux file avaialble for chamber %s!" % (ch))
        
        
        # for plotting of pressure-volume loops
        for ch in ['v_l','v_r','at_l','at_r']:
            subprocess.call(['cp', ''+path+'/results_'+sname+'_p_'+ch+'.txt', ''+path+'/results_'+sname+'_p_'+ch+'_tmp.txt'])
            subprocess.call(['cp', ''+path+'/results_'+sname+'_V_'+ch+'.txt', ''+path+'/results_'+sname+'_V_'+ch+'_tmp.txt'])
            # drop first (time) columns
            subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', ''+path+'/results_'+sname+'_p_'+ch+'_tmp.txt'])
            subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', ''+path+'/results_'+sname+'_V_'+ch+'_tmp.txt'])
            # paste files together
            os.system('paste '+path+'/results_'+sname+'_V_'+ch+'_tmp.txt '+path+'/results_'+sname+'_p_'+ch+'_tmp.txt > '+path+'/results_'+sname+'_pV_'+ch+'.txt')
            # isolate last cycle                
            os.system('tail -n '+str(nstep_cycl)+' '+path+'/results_'+sname+'_pV_'+ch+'.txt > '+path+'/results_'+sname+'_pV_'+ch+'_last.txt')
            # isolate healthy/baseline cycle
            if indpertaftercyl > 0:
                os.system('sed -n "'+str((indpertaftercyl-1)*nstep_cycl+1)+','+str(indpertaftercyl*nstep_cycl)+'p" '+path+'/results_'+sname+'_pV_'+ch+'.txt > '+path+'/results_'+sname+'_pV_'+ch+'_baseline.txt')
            # clean-up
            subprocess.call(['rm', ''+path+'/results_'+sname+'_p_'+ch+'_tmp.txt'])
            subprocess.call(['rm', ''+path+'/results_'+sname+'_V_'+ch+'_tmp.txt'])
            
            
        # for plotting of compartment volumes: gather all volumes and add them in order to check if volume conservation is fulfilled!
        # Be worried if the total sum in V_all.txt changes over time (more than to a certain tolerance)!
        volall = np.zeros(numdata)
        for c in range(len(list(groups[5].values())[0])-1): # compartment volumes should be stored in group index 5
            subprocess.call(['cp', ''+path+'/results_'+sname+'_'+list(groups[5].values())[0][c]+'.txt', ''+path+'/results_'+sname+'_'+list(groups[5].values())[0][c]+'_tmp.txt'])
            # drop first (time) column
            subprocess.call(['sed', '-r', '-i', 's/(\s+)?\S+//1', ''+path+'/results_'+sname+'_'+list(groups[5].values())[0][c]+'_tmp.txt'])
            # load volume data
            vols = np.loadtxt(''+path+'/results_'+sname+'_'+list(groups[5].values())[0][c]+'_tmp.txt')
            # add together
            for i in range(len(volall)):
                volall[i] += vols[i]
            # clean-up
            subprocess.call(['rm', ''+path+'/results_'+sname+'_'+list(groups[5].values())[0][c]+'_tmp.txt'])

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
        fi.write('N_step ' +str(nstep_cycl) + '\n\n')
        
        
        # function parameters of left and right ventricle
        if calculate_function_params:
        
            sw, sv, co, ef, edv, esv, edp, esp = [], [], [], [], [], [], [], []
            for ch in ['v_l','v_r']:
                
                # stroke work
                pv = np.loadtxt(''+path+'/results_'+sname+'_pV_'+ch+'_last.txt') # this is already last (periodic) cycle pv data!
                val = 0.0
                for k in range(len(pv)-1):
                    # we need the negative sign since we go counter-clockwise around the loop!
                    val -= 0.5*(pv[k+1,1]+pv[k,1]) * (pv[k+1,0] - pv[k,0])
                sw.append(val)
                
                # stroke volume, cardiac output, end-diastolic and end-systolic volume, ejection fraction
                vol = np.loadtxt(''+path+'/results_'+sname+'_V_'+ch+'.txt', skiprows=numdata-nstep_cycl)
                sv.append(max(vol[:,1])-min(vol[:,1]))
                co.append((max(vol[:,1])-min(vol[:,1]))/T_cycl)
                edv.append(max(vol[:,1]))
                esv.append(min(vol[:,1]))
                ef.append((max(vol[:,1])-min(vol[:,1]))/max(vol[:,1]))
                
                pres = np.loadtxt(''+path+'/results_'+sname+'_p_'+ch+'.txt', skiprows=numdata-nstep_cycl)

                # end-diastolic pressure
                for k in range(len(pres)):
                    if round(pres[k,0],2) == round(t_ed+(n_cycl-1)*T_cycl,2):
                        edp_index = k
                        break
                edp.append(pres[edp_index,1])

                # end-systolic pressure
                for k in range(len(pres)):
                    if round(pres[k,0],2) == round(t_es+(n_cycl-1)*T_cycl,2):
                        esp_index = k
                        break
                esp.append(pres[esp_index,1])
                
            # mean arterial pressure
            marp = []
            for pc in ['ar_sys','ar_pul']:
                
                pr = np.loadtxt(''+path+'/results_'+sname+'_p_'+pc+'.txt', skiprows=numdata-nstep_cycl)
                
                val = 0.0
                for k in range(len(pr)-1):
                    val += 0.5*(pr[k+1,1]+pr[k,1]) * (pr[k+1,0] - pr[k,0])
                val /= (pr[-1,0]-pr[0,0])
                marp.append(val)
            
            # we assume here that units kg - mm - s are used --> pressures are kPa, forces are mN, volumes are mm^3
            # for convenience, we convert work to mJ, volumes to ml and cardiac output to l/min
            fi.write('sw_l %.4f [mJ]\n' % (sw[0]/1000.))
            fi.write('sw_r %.4f [mJ]\n' % (sw[1]/1000.))
            fi.write('sv_l %.4f [ml]\n' % (sv[0]))
            fi.write('sv_r %.4f [ml]\n' % (sv[1]))
            fi.write('co_l %.4f [l/min]\n' % (co[0]*60./1.0e6))
            fi.write('co_r %.4f [l/min]\n' % (co[1]*60./1.0e6))
            fi.write('ef_l %.4f [%%]\n' % (ef[0]*100.))
            fi.write('ef_r %.4f [%%]\n' % (ef[1]*100.))
            fi.write('edv_l %.4f [ml]\n' % (edv[0]/1000.))
            fi.write('edv_r %.4f [ml]\n' % (edv[1]/1000.))
            fi.write('esv_l %.4f [ml]\n' % (esv[0]/1000.))
            fi.write('esv_r %.4f [ml]\n' % (esv[1]/1000.))
            fi.write('edp_l %.4f [kPa]\n' % (edp[0]))
            fi.write('edp_r %.4f [kPa]\n' % (edp[1]))
            fi.write('esp_l %.4f [kPa]\n' % (esp[0]))
            fi.write('esp_r %.4f [kPa]\n' % (esp[1]))
            fi.write('map_sys %.4f [kPa]\n' % (marp[0]))
            fi.write('map_pul %.4f [kPa]\n' % (marp[1]))
            fi.close()



    for g in range(len(groups)):
        
        numitems = len(list(groups[g].values())[0])
        
        # safety (and sanity...) check
        if numitems > 16:
            print("More than 16 items to plot in one graph! Adjust plotfile template or consider if this is sane...")
            sys.exit()
        
        subprocess.call(['cp', ''+str(fpath)+'/flow0d_gnuplot_template.p', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's#__OUTDIR__#'+path+'/plot0d_'+sname+'/#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's#__FILEDIR__#'+path+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        
        subprocess.call(['sed', '-i', 's/__OUTNAME__/'+list(groups[g].keys())[0]+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        
        factor_kPa_mmHg = 7.500615
        
        if 'pres_time' in list(groups[g].keys())[0]:
            x1value, x2value     = 't', ''
            x1unit, x2unit       = 's', ''
            y1value, y2value     = 'p', 'p'
            y1unit, y2unit       = 'kPa', 'mmHg'
            xscale, yscale       = 1.0, 1.0
            x2rescale, y2rescale = 1.0, factor_kPa_mmHg
            xextend, yextend     = 1.0, 1.1
            maxrows, maxcols, sl = 1, 5, 20
            if (model == 'syspulcap' or model == 'syspulcap2' or model == 'syspulcaprespir') and 'pres_time_sys_l' in list(groups[g].keys())[0]:
                xextend, yextend     = 1.0, 1.2
                maxrows, maxcols, sl = 2, 5, 19
        if 'flux_time' in list(groups[g].keys())[0]:
            x1value, x2value     = 't', ''
            x1unit, x2unit       = 's', ''
            y1value, y2value     = 'q', ''
            y1unit, y2unit       = 'ml/s', ''
            xscale, yscale       = 1.0, 1.0e-3
            x2rescale, y2rescale = 1.0, 1.0
            xextend, yextend     = 1.0, 1.1
            maxrows, maxcols, sl = 1, 5, 20
            if (model == 'syspulcap' or model == 'syspulcap2' or model == 'syspulcaprespir') and 'flux_time_sys_l' in list(groups[g].keys())[0]:
                xextend, yextend     = 1.0, 1.3
                maxrows, maxcols, sl = 3, 5, 20
        if 'vol_time' in list(groups[g].keys())[0]:
            x1value, x2value     = 't', ''
            x1unit, x2unit       = 's', ''
            y1value, y2value     = 'V', ''
            y1unit, y2unit       = 'ml', ''
            xscale, yscale       = 1.0, 1.0e-3
            x2rescale, y2rescale = 1.0, 1.0
            xextend, yextend     = 1.0, 1.1
            maxrows, maxcols, sl = 1, 5, 20
        if 'pres_vol_v' in list(groups[g].keys())[0]:
            x1value, x2value     = 'V_{\\\mathrm{v}}', ''
            x1unit, x2unit       = 'ml', ''
            y1value, y2value     = 'p_{\\\mathrm{v}}', 'p_{\\\mathrm{v}}'
            y1unit, y2unit       = 'kPa', 'mmHg'
            xscale, yscale       = 1.0e-3, 1.0
            x2rescale, y2rescale = 1.0, factor_kPa_mmHg
            xextend, yextend     = 1.1, 1.1
            maxrows, maxcols, sl = 1, 5, 20
        if 'pres_vol_at' in list(groups[g].keys())[0]:
            x1value, x2value     = 'V_{\\\mathrm{at}}', ''
            x1unit, x2unit       = 'ml', ''
            y1value, y2value     = 'p_{\\\mathrm{at}}', 'p_{\\\mathrm{at}}'
            y1unit, y2unit       = 'kPa', 'mmHg'
            xscale, yscale       = 1.0e-3, 1.0
            x2rescale, y2rescale = 1.0, factor_kPa_mmHg
            xextend, yextend     = 1.1, 1.1
            maxrows, maxcols, sl = 1, 5, 20
        if 'vol_time_compart' in list(groups[g].keys())[0]:
            x1value, x2value     = 't', ''
            x1unit, x2unit       = 's', ''
            y1value, y2value     = 'V', ''
            y1unit, y2unit       = 'ml', ''
            xscale, yscale       = 1.0, 1.0e-3
            x2rescale, y2rescale = 1.0, 1.0
            xextend, yextend     = 1.0, 1.2
            maxrows, maxcols, sl = 2, 5, 20
            if (model == 'syspulcap' or model == 'syspulcap2' or model == 'syspulcaprespir'):
                xextend, yextend     = 1.0, 1.3
                maxrows, maxcols, sl = 3, 5, 10
        if 'ppO2_time' in list(groups[g].keys())[0]:
            x1value, x2value     = 't', ''
            x1unit, x2unit       = 's', ''
            y1value, y2value     = 'p_{\\\mathrm{O}_2}', 'p_{\\\mathrm{O}_2}'
            y1unit, y2unit       = 'kPa', 'mmHg'
            xscale, yscale       = 1.0, 1.0
            x2rescale, y2rescale = 1.0, factor_kPa_mmHg
            xextend, yextend     = 1.0, 1.2
            maxrows, maxcols, sl = 1, 5, 20
            if 'sys_l' in list(groups[g].keys())[0]:
                xextend, yextend     = 1.0, 1.3
                maxrows, maxcols, sl = 3, 5, 10
        if 'ppCO2_time' in list(groups[g].keys())[0]:
            x1value, x2value     = 't', ''
            x1unit, x2unit       = 's', ''
            y1value, y2value     = 'p_{\\\mathrm{CO}_2}', 'p_{\\\mathrm{CO}_2}'
            y1unit, y2unit       = 'kPa', 'mmHg'
            xscale, yscale       = 1.0, 1.0
            x2rescale, y2rescale = 1.0, factor_kPa_mmHg
            xextend, yextend     = 1.0, 1.2
            maxrows, maxcols, sl = 1, 5, 20
            if 'sys_l' in list(groups[g].keys())[0]:
                xextend, yextend     = 1.0, 1.3
                maxrows, maxcols, sl = 3, 5, 10
        
        data = []
        x_s_all, x_e_all = [], []
        y_s_all, y_e_all = [], []
        
        for q in range(numitems):
            
            # get the data and check its length
            tmp = np.loadtxt(''+path+'/results_'+sname+'_'+list(groups[g].values())[0][q]+'.txt') # could be another file - all should have the same length!
            numdata = len(tmp)
            
            # set quantity, title, and plotting line
            subprocess.call(['sed', '-i', 's/__QTY'+str(q+1)+'__/results_'+sname+'_'+list(groups[g].values())[0][q]+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__TIT'+str(q+1)+'__/'+list(groups[g].values())[1][q]+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            subprocess.call(['sed', '-i', 's/__LIN'+str(q+1)+'__/'+str(list(groups[g].values())[2][q])+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            
            # adjust the plotting command to include all the files to plot in one graph
            if q!=0: subprocess.call(['sed', '-i', 's/#__'+str(q+1)+'__//g', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
            
            if 'PERIODIC' in list(groups[g].keys())[0]: skip = numdata-nstep_cycl
            else: skip = 0
            
            # get the x,y range on which to plot
            data.append(np.loadtxt(''+path+'/results_'+sname+'_'+list(groups[g].values())[0][q]+'.txt', skiprows=skip))

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

        # if we want to use a x2 or y2 axis
        if x2value != '': subprocess.call(['sed', '-i', 's/#__HAVEX2__//', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        if y2value != '': subprocess.call(['sed', '-i', 's/#__HAVEY2__//', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        
        # axis segments - x
        subprocess.call(['sed', '-i', 's/__X1S__/'+str(x_s)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__X1E__/'+str(x_e*xextend)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__X2S__/'+str(x2rescale*x_s)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__X2E__/'+str(x2rescale*x_e*xextend)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        # axis segments - y
        subprocess.call(['sed', '-i', 's/__Y1S__/'+str(y_s)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__Y1E__/'+str(y_e*yextend)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__Y2S__/'+str(y2rescale*y_s)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__Y2E__/'+str(y2rescale*y_e*yextend)+'/', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        # units
        subprocess.call(['sed', '-i', 's#__X1UNIT__#'+x1unit+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's#__Y1UNIT__#'+y1unit+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        if x2unit != '': subprocess.call(['sed', '-i', 's#__X2UNIT__#'+x2unit+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        if y2unit != '': subprocess.call(['sed', '-i', 's#__Y2UNIT__#'+y2unit+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        # values
        subprocess.call(['sed', '-i', 's#__X1VALUE__#'+x1value+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's#__Y1VALUE__#'+y1value+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        if x2value != '': subprocess.call(['sed', '-i', 's#__X2VALUE__#'+x2value+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        if y2value != '': subprocess.call(['sed', '-i', 's#__Y2VALUE__#'+y2value+'#', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        # scales
        subprocess.call(['sed', '-i', 's/__XSCALE__/'+str(xscale)+'/g', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__YSCALE__/'+str(yscale)+'/g', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        # rows, columns and sample length for legend
        subprocess.call(['sed', '-i', 's/__MAXROWS__/'+str(maxrows)+'/g', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__MAXCOLS__/'+str(maxcols)+'/g', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        subprocess.call(['sed', '-i', 's/__SAMPLEN__/'+str(sl)+'/g', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])

        # do the plotting
        subprocess.call(['gnuplot', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])
        # convert to PDF
        subprocess.call(['ps2pdf', '-dEPSCrop', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.eps', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.pdf'])
        subprocess.call(['pdflatex', '-interaction=batchmode', '-output-directory='+path+'/plot0d_'+sname+'/', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.tex'])
            
        # clean up
        subprocess.call(['rm', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.aux', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.log'])
        # guess we do not need these files anymore since we have the final PDF...
        subprocess.call(['rm', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'.tex'])
        subprocess.call(['rm', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.pdf'])
        subprocess.call(['rm', ''+path+'/plot0d_'+sname+'/'+list(groups[g].keys())[0]+'-inc.eps'])
        # delete gnuplot file
        subprocess.call(['rm', ''+path+'/plot_'+list(groups[g].keys())[0]+'.p'])



def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise RuntimeError("str_to_bool failed!")



if __name__ == "__main__":
    
    main()

