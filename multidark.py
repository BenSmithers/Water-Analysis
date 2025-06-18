import numpy as np 
import matplotlib.pyplot as plt 
from utils import load_timeout 
import sys
from utils import get_start_times, get_color, get_temperatures, get_event_time, fold, average
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import os 
plt.style.use("wms.mplstyle")
NMERGE = 1
wavelens = [450, 410, 365, 295, 278, 255, "n/a"]


files = ["data/picodat_run84_Supply Untreated_various_variousadc_mHz.dat",
        "data/picodat_run87_Supply Untreated_various_variousadc_mHz.dat",
            "data/picodat_run88_Supply Untreated_various_variousadc_mHz.dat",
            "data/picodat_run89_Supply Untreated_various_variousadc_mHz.dat",
            "data/picodat_run90_Supply Untreated_various_variousadc_mHz.dat",
            "data/picodat_run91_Supply Untreated_various_variousadc_mHz.dat",
            "data/picodat_run95_Supply Untreated_various_variousadc_mHz.dat",]


for ifile, fname in enumerate(files):
    data = np.loadtxt(fname, delimiter=",").T 

    run_no = os.path.split(fname)[1].split("_")[1]
    wave = data[7]
    mask = wave==-1
    i=-1

    triggers = fold(data[1][mask], NMERGE)

    monitor = fold(data[2][mask], NMERGE)
    mon_dark = fold(data[4][mask], NMERGE)
    receiver = fold(data[3][mask], NMERGE)
    rec_dark = fold(data[5][mask], NMERGE)

    no_light_mon  = 1 - monitor/triggers
    no_light_rec =  1 - receiver/triggers

    adj_mon = 1 - (no_light_mon)/(1 - mon_dark/triggers)
    adj_rec = 1 - (no_light_rec)/(1 - rec_dark/triggers)

    ratio = np.log10(1-adj_rec)/np.log10(1 - adj_mon)
    
    mon_to_plot = mon_dark/triggers
    #mon_to_plot-=np.min(mon_to_plot)
   # mon_to_plot/=np.max(mon_to_plot)
    rec_to_plot =  rec_dark/triggers


    times= average(data[0][mask], NMERGE)
    mwaves = average(wave[mask], NMERGE) 
    times = np.array([datetime.fromtimestamp(entry) for entry in times])


    plt.errorbar(times, mon_to_plot, yerr= None,color='k', marker='d', ls='-', alpha=0.5)
    plt.errorbar(times, rec_to_plot + 0.001, yerr= None,color='k', marker='o', alpha=0.5)

plt.errorbar([],[], yerr= None,label="Mon {} nm".format(wavelens[i]),color='k', marker='d', ls='-', alpha=0.5)
plt.errorbar([], [], yerr= None,label="Rec {} nm".format(wavelens[i]),color='k', marker='o', alpha=0.5)


plt.gcf().autofmt_xdate()
plt.xlabel("Time Stamp",size=14)
plt.ylabel(r"Monitor Dark Rate",size=14)
plt.legend()
#plt.ylim([-0.05, 0.05])
plt.tight_layout()
plt.show()