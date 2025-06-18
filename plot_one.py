import numpy as np 
import matplotlib.pyplot as plt 
from utils import load_timeout 
import sys
from utils import get_start_times, get_color, get_temperatures, get_event_time, fold, average
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from utils import apply_correction, get_light
import os 
plt.style.use("wms.mplstyle")
wavelens = [450, 410, 365, 295, 278, 255]

NMERGE = 1
FIT_DARKNOISE = True
TIME_CUT = False 
PCHANGE = True
RAW = False
NEW_DRATE = False
JUST_PLOT_DRATE =  False
TPLOT = False     
GAD = True
LIGHT = True
correction = False

baselines = [
    np.nan, 
    1.18208,
    0.7887176,
    0.3253037117526836,
    0.279768,
    0.2344264
]

if GAD:
    baselines = [np.nan, 1.1820821, 0.7887177, 0.3253037, 0.279768, 0.2285] # reverse but pure


    # with resin, but without Gd 
    baselines = [np.nan, 1.16484, 0.758492, 0.277763, 0.2188047, 0.124414965]


    
if True:
    baselines =[
        np.nan, 1.1502848465930784, 0.7291714208209754, 0.24093727694429143, 
        0.17412167436851508, 0.08823493414080769
    ] # baseline after first Gd loading, before second

    baselines = [
        np.nan, 1.14840897086328,0.7447017558664221,
        0.2556593826546975, 0.1846610426449705, 0.10193979703149392
    ]
    baselines = [
        np.nan, 
        1.1006778167145266,
        0.6939987328288022,
        0.17372054376604,
        0.0995812003763459,
        0.05400655458140932
    ]

    baselines = [
        np.nan,
        1.0403869794773686, 
        0.6590904580273826,
        0.12407494094631229,
        0.07425881959242897,
        0.04817418995625853
    ]

#start = datetime(year=2025, month=4, day=12, hour=10)

#start = datetime(year=2025, month=4, day=3, hour=15)
start = datetime(year=2025, month=4, day=4,hour=2 )
start = datetime(year=2025, month=5, day=3, hour=1)
start = datetime(year=2025, month=4, day=5, hour=11) # for the stagnation
start = datetime(year=2025, month=4, day=25, hour=16)
end = datetime(year=2025, month=6, day=2, hour=0)



files = ["data/picodat_run67_Return Untreated_various_variousadc_mHz.dat",
            "data/picodat_run68_Supply Untreated_various_variousadc_mHz.dat"]

#files = ["data/picodat_run73_Supply Untreated_various_variousadc_mHz.dat"]
files = ["data/picodat_run75_Supply Untreated_various_variousadc_mHz.dat",
         "data/picodat_run77_Supply Untreated_various_variousadc_mHz.dat",
         "data/picodat_run78_Supply Untreated_various_variousadc_mHz.dat",
         "data/picodat_run87_Supply Untreated_various_variousadc_mHz.dat",
         "data/picodat_run84_Supply Untreated_various_variousadc_mHz.dat"]
files = ["data/picodat_run89_Supply Untreated_various_variousadc_mHz.dat"]
files = ["data/picodat_run87_Supply Untreated_various_variousadc_mHz.dat",]
files = ["data/picodat_run95_Supply Untreated_various_variousadc_mHz.dat",]

files = ["data/picodat_run96_Supply Untreated_various_variousadc_mHz.dat",]
#files = ["data/picodat_run66_Return Untreated_various_variousadc_mHz.dat",]

#files = ["data/picodat_run73_Supply Untreated_various_variousadc_mHz.dat"]
#files = ["data/picodat_run62_Supply Untreated_365nm_715adc_mHz.dat"]
#files = ["data/picodat_run61_Supply Untreated_various_variousadc_mHz.dat"]

run_no = ""

shift = 8*3600
plt.clf()
plt.close()

plt.figure(figsize=(10,6))
for ifile, fname in enumerate(files):


    data = np.loadtxt(fname, delimiter=",").T 

    run_no = os.path.split(fname)[1].split("_")[1]

    if TPLOT:
        ttime = average(data[0], NMERGE) 
        temperature = get_temperatures(ttime)

        ttime = ttime 
        ttime = np.array([datetime.fromtimestamp(entry+shift) for entry in ttime])
        mask = np.logical_and( ttime > start , ttime<end)

        if TIME_CUT:
            ttime = ttime[mask]
            temperature = temperature[mask]
        
    if LIGHT:
        ttime = average(data[0], NMERGE)
        light =  get_light(ttime, 0)
        ttime = np.array([datetime.fromtimestamp(entry+shift) for entry in ttime])
        mask = np.logical_and( ttime > start , ttime<end)
        if TIME_CUT:
            ttime = ttime[mask]
            light = light[mask]


    wave = data[7]
    alltime = np.array([datetime.fromtimestamp(entry+shift) for entry in data[0]])

    if NEW_DRATE or JUST_PLOT_DRATE:
        
        mask = wave==-1 
        dark_time = np.array([datetime.fromtimestamp(entry+shift) for entry in data[0]])
        if TIME_CUT:
            mask = np.logical_and( np.logical_and(dark_time> start , dark_time<end ), mask)

        dark_time =average(data[0][mask]+shift, NMERGE)
        
        rec_dark_raw = fold(data[5][mask], NMERGE)
        mon_dark_raw = fold(data[4][mask], NMERGE)
        dark_triggers = fold(data[1][mask], NMERGE)
        
        dark_date = np.array([datetime.fromtimestamp(entry) for entry in dark_time])
        mon_dark_sp = CubicSpline(dark_time, mon_dark_raw)
        rec_dark_sp = CubicSpline(dark_time, rec_dark_raw)


    fill_end_key = "pu1 off signal sent"
    fill_end    = get_event_time(data[0], fill_end_key) +shift
    
    fill_end = np.array([datetime.fromtimestamp(entry) for entry in fill_end])
    fill_end = fill_end[np.logical_and( fill_end>start , fill_end<end)]
    
    for i in range(1, 6):

        mask = wave ==i # np.logical_and( wave==i, data[1]>3.7e6)
        time_mask = np.logical_and( alltime>start , alltime<end)
        if TIME_CUT:
            mask = np.logical_and(mask, time_mask)

        triggers = fold(data[1][mask], NMERGE)
        def find_constant(_mon, _mon_dark, _rec, _rec_dark):
            def metric(params):
                _reflectivity = params[0]
                _reflectivity2 = params[1]
                recmdark = np.log( (triggers - _rec)/(triggers - _rec_dark*_reflectivity) )
                monmdark = np.log( (triggers - _mon)/(triggers - _mon_dark*_reflectivity2))

                ratio = recmdark / monmdark
                metric = (ratio -np.nanmean(ratio))/np.nanmean(ratio)
                return np.sum(np.log10( 1+ metric**2))

                mask = np.abs(recmdark/monmdark)<0.05

                return  np.log( 1 + np.std( recmdark) + np.std( monmdark ))


            bounds = [(0.01,100),
                      (0.01,100)]
            options={
                "eps":1e-2,
                "gtol":1e-20,
                "ftol":1e-20
            }
            res = minimize(metric, x0=[1.0, 1.0,], bounds=bounds, options=options)
            print(res.x)
            return res.x 
        
        monitor = fold(data[2][mask], NMERGE)
        receiver = fold(data[3][mask], NMERGE)
        times= average(data[0][mask]+shift, NMERGE)
        if NEW_DRATE:
            mon_dark = mon_dark_sp(times)
            rec_dark = rec_dark_sp(times)
        else:
            mon_dark = fold(data[4][mask], NMERGE)#*20/367
            rec_dark = fold(data[5][mask], NMERGE)#*20/367
            
        if FIT_DARKNOISE:
                    
            res = find_constant(monitor, mon_dark, receiver, rec_dark)
            #res = [6.5,5]
            #res = [3.8, 3.8]
            #res= [0.75, 0.5]
            res = [0.64, 0.58]
            #print(res)
            rec_refl = res[0]
            mon_refl = res[1]
            recmdark = rec_refl*rec_dark 
            monmdark =  mon_refl*mon_dark 
        else:
            recmdark = rec_dark 
            monmdark = mon_dark

        if correction:
            rcor = apply_correction(receiver/triggers, recmdark/triggers)        
            mcor = apply_correction(monitor/triggers, monmdark/triggers)
        else:
            rcor = 1.0 
            mcor = 1.0

        mwaves = average(wave[mask], NMERGE)        
        tempy = get_temperatures(times)
                
        ratiomdark = np.log( rcor*(triggers - receiver)/(triggers - recmdark) )/np.log( mcor*(triggers - monitor)/(triggers - monmdark))

        alpha = 0.5

        times = np.array([datetime.fromtimestamp(entry) for entry in times])
        if len(ratiomdark)==0:
            continue
        if RAW:
            plt.plot([], [], color=get_color(i+1, 8, 'nipy_spectral_r') ,label="{} nm".format(wavelens[i]), ls='-')
            if PCHANGE:
                plt.plot(times, (monitor - np.nanmean(monitor))/np.nanmean(monitor),  color=get_color(i+1, 8, 'nipy_spectral_r'), marker='d', ls='', alpha=1.0)
                plt.plot(times, (receiver - np.nanmean(receiver))/np.nanmean(receiver),  color=get_color(i+1, 8, 'nipy_spectral_r'), marker='o', ls='', alpha=1.0)
            else:
                plt.plot(times, (monitor/triggers),  color=get_color(i+1, 8, 'nipy_spectral_r'), marker='d', ls='', alpha=1.0)
                plt.plot(times, (receiver/triggers),  color=get_color(i+1, 8, 'nipy_spectral_r'), marker='o', ls='', alpha=1.0)

        else:
            if PCHANGE:
                print("{} nm : ".format(wavelens[i]),np.nanmean(ratiomdark))
                plt.errorbar(times, (ratiomdark -np.nanmean(ratiomdark))/np.nanmean(ratiomdark), yerr= None, color=get_color(i+1, 8, 'nipy_spectral_r'),label="{} nm".format(wavelens[i]), marker='d', ls='-', alpha=alpha)
                #plt.errorbar(times, (ratiomdark -baselines[i])/baselines[i], yerr= None, color=get_color(i+1, 8, 'nipy_spectral_r'),label="{} nm".format(wavelens[i]), marker='d', ls='-', alpha=alpha)
            else:
                plt.errorbar(times,ratiomdark, yerr= None, color=get_color(i+1, 8, 'nipy_spectral_r'),label="{} nm".format(wavelens[i]), marker='d', ls='', alpha=alpha)

    #plt.vlines(fill_end, -0.5, 0.2, 'gray', alpha=0.5, ls='--', label="Pump Off")


if False:
    if PCHANGE:
        plt.vlines(fill_end, -0.1, 0.1, color='gray', ls='--', label="Pump Off")    
    else:
        plt.vlines(fill_end, 0, 2, color='gray', ls='--', label="Pump Off")    
if RAW:
    plt.plot([], [], color='k',marker='d', ls='', label="Monitor")
    plt.plot([], [], color='k',marker='o', ls='', label="Receiver")


#plt.hlines([-0.01, 0.01], start, end, color='red', ls='--')
#plt.hlines([-0.001, 0.001], start, end, color='gray', ls='--')

if PCHANGE:
    plt.ylabel(r"Fractional Diff. [$\mu$]",size=14)
    #plt.ylim([-0.05, 0.05 ])
    
else:
    if not RAW:
        plt.ylim([0.0, 1.4])
    plt.ylabel(r"$\mu$ Ratio",size=14)
plt.xlabel("Time Stamp",size=14)

if (NEW_DRATE or  JUST_PLOT_DRATE) and not TPLOT:
    plt.plot([], [], color='k', marker='1', label="Mon Dark")
    plt.plot([], [], color='k', marker='2', label="Rec Dark")


#plt.yscale('log')
plt.gcf().autofmt_xdate()
#plt.xlim([start, end])
plt.legend(loc='upper left')

if TPLOT:
    plt.plot([], [], color='red', alpha=0.5, label="Temp")
    twax = plt.twinx()
    twax.plot(ttime, temperature, color='red', alpha=0.5)
if LIGHT:
    plt.plot([], [], color='red', alpha=0.5, label="Light")
    twax = plt.twinx()
    mlight =light/3000
    twax.plot(ttime,mlight , color='red', alpha=0.5)

if (NEW_DRATE or  JUST_PLOT_DRATE) and not TPLOT:
    
    plt.plot([], [], color='k', marker='1',  alpha=0.5, label="Monitor")
    plt.plot([], [], color='k', marker='2',  alpha=0.5, label="Receiver")
    #twax = plt.twinx()
    
    plt.plot(dark_date, 0.05*((mon_dark_raw/dark_triggers)-np.mean(mon_dark_raw/dark_triggers))/np.mean(mon_dark_raw/dark_triggers) ,color='k', marker='1',  alpha=0.5)
    plt.plot(dark_date, 0.05*((rec_dark_raw/dark_triggers)-np.mean(rec_dark_raw/dark_triggers))/np.mean(rec_dark_raw/dark_triggers) ,color='k', marker='2', alpha=0.5)

    #twax.set_ylim([-3,3])
    #twax.set_ylabel("Light Leak norm by Mon")

image_name = os.path.join(
    os.path.dirname(__file__),
    "plots",
    run_no + "trends.png"
)
plt.tight_layout()
plt.savefig(image_name, dpi=400)
plt.show()
