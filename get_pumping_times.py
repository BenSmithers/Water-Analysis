"""
    Gets the times to fill and drain the open tank. 
    Checking how feasible it is to pump on full...
"""

import numpy as np 
from datetime import datetime

from utils import get_event_time
import matplotlib.pyplot as plt 

drain_start_key = "pu3 on signal sent"
drain_end_key = "pu3 off signal sent"

fill_start_key = "pu1 on signal sent"
fill_end_key = "pu1 off signal sent"

fname = "data/picodat_run82_Supply Untreated_various_variousadc_mHz.dat"
data = np.loadtxt(fname, delimiter=",").T 
alltime = data[0]#  np.array([datetime.fromtimestamp(entry) for entry in data[0]])

drain_start = get_event_time(alltime, drain_start_key)
drain_end   = get_event_time(alltime, drain_end_key)
fill_start  = get_event_time(alltime, fill_start_key)
fill_end    = get_event_time(alltime, fill_end_key)

n_drain = min([len(drain_start), len(drain_end)])
n_fill = min([len(fill_start), len(fill_end)])

drain_times = [ (drain_end[i] - drain_start[i])/60 for i in range(n_drain)]
fill_times = [ (fill_end[i] - fill_start[i])/60 for i in range(n_fill)]

print("Mean drain time", np.mean(drain_times)/60)
print("Mean fill time", np.mean(fill_times)/60)

tbins = np.linspace(0,15, 45)
plt.hist(drain_times, tbins, label="Drain")
plt.hist(fill_times, tbins, label="Fill")
plt.xlabel("Time to Fill [min]")
plt.legend()
plt.show()