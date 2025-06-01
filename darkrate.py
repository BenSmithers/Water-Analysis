import numpy as np 
import matplotlib.pyplot as plt 
import sys
from utils import load_bv1, fold, average, get_color
from datetime import datetime


NMERGE = 1
start = datetime(year=2025, month=5, day=22, hour=2)
end = datetime(year=2026, month=5, day=22, hour=7)
TIME_CUT = False 
wavelens = [450, 410, 365, 295, 278, 255, "n/a"]


fname  = "data/picodat_run95_Supply Untreated_various_variousadc_mHz.dat"
#fname = "data/picodat_run93_Supply Untreated_255nm_818adc_mHz.dat"
data = np.loadtxt(fname, delimiter=",").T 


wave = data[7]
alltime = np.array([datetime.fromtimestamp(entry + 9*3600) for entry in data[0]])

for i in [-1,1,2,3,4,5]:
    if i!=5:
        continue
    mask = wave ==i # np.logical_and( wave==i, data[1]>3.7e6)
    time_mask = np.logical_and( alltime>start , alltime<end)
    if TIME_CUT:
        mask = np.logical_and(mask, time_mask)
    #touts = load_timeout(times)
    
    #touts = (touts - times.min())/3600 

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
    #rec_to_plot-=np.min(rec_to_plot)
   # rec_to_plot/=np.max(rec_to_plot)


    times= average(data[0][mask], NMERGE)
    mwaves = average(wave[mask], NMERGE) 
    times = np.array([datetime.fromtimestamp(entry) for entry in times])

    plt.errorbar(times, mon_to_plot, yerr= None,label="Mon {} nm".format(wavelens[i]),color='k', marker='d', ls='-', alpha=0.5)
    plt.errorbar(times, rec_to_plot + 0.001, yerr= None,label="Rec {} nm".format(wavelens[i]),color='k', marker='o', alpha=0.5)
    #plt.errorbar(times, receiver/triggers, yerr= None, color=get_color(i+1, 8, 'nipy_spectral_r'),label="{} nm".format(wavelens[i]), marker='d', ls='-')
    #plt.errorbar(times, rec_dark/triggers, yerr= None, color='k',label="{} nm".format(wavelens[i]), marker='d', ls='-')
    

plt.gcf().autofmt_xdate()
plt.xlabel("Time Stamp",size=14)
plt.ylabel(r"Monitor Dark Rate",size=14)
plt.legend()
#plt.ylim([-0.05, 0.05])
plt.tight_layout()
plt.show()