import os 
import numpy as np 
import datetime 
import numpy as np
from scipy.interpolate import CubicSpline
logfile = os.path.join(os.path.dirname(__file__), "meta_history", "command.log")



def fold(thisdat, nmerge=2):
    if nmerge==1:
        return thisdat
    # cut off the extra so it's divisible 
    if int(len(thisdat)%nmerge)!=0:
        thisdat = thisdat[:-(len(thisdat)%nmerge)]
    holder =np.nansum(np.reshape(thisdat, (int(len(thisdat)/nmerge), nmerge)), axis=1)
    return holder

def average(thisdat, nmerge=3):
    if nmerge==1:
        return thisdat
    if int((len(thisdat)%nmerge))!=0:
        thisdat = thisdat[:-(len(thisdat)%nmerge)]
    return np.mean(np.reshape(thisdat, (int(len(thisdat)/nmerge), nmerge)), axis=1)
    


def load_log():
    _obj = open(logfile,'rt')
    lines = _obj.readlines()
    _obj.close()

    timestamps = []
    entry = []
    for line in lines:
        brk = line.split(" : ")
        if "TUBEFULL" in brk[1]:

            timestamps.append(datetime.datetime.fromisoformat(brk[0]).timestamp())
            entry.append("TUBEFULL")

    return np.array(timestamps, dtype=float)

def get_rtime(trigs, hits):
    min_time = 0
    max_time = 346

    tdiffs = []

    hit_index = 0
    for i in range(len(trigs)):
        while hits[hit_index]<(trigs[i]+max_time) and hit_index<len(hits)-1:

            # now the hit_index is the hit just after the trig we're on
            dif = hits[hit_index] - trigs[i]
            if dif>min_time and dif<max_time:
                tdiffs.append(dif)
                hit_index +=1
                break

            hit_index+=1

    return tdiffs 

def get_valid(trigs, hits, is_rec):
    if is_rec:
        min_time = 113
        max_time = 140
    else:
        min_time = 14
        max_time = 40

    is_valid = np.zeros_like(hits)
    is_valid = is_valid.astype(bool)
    tdiffs = []

    hit_index = 0
    for i in range(len(trigs)):
        while hits[hit_index]<(trigs[i]+max_time) and hit_index<len(hits)-1:

            # now the hit_index is the hit just after the trig we're on
            dif = hits[hit_index] - trigs[i]
            if dif>min_time and dif<max_time:
                is_valid[hit_index] = True
                tdiffs.append(dif)
                hit_index +=1
                break
            else:
                is_valid[hit_index] = False or is_valid[hit_index]

            hit_index+=1
    

    return is_valid, np.logical_not(is_valid) 

def get_event_time(times, text_key):
    mint = times.min()
    maxt = times.max()
    print(mint)
    print(maxt)

    _obj = open(logfile,'rt')
    lines = _obj.readlines()
    _obj.close()

    timestamps = []
    for line in lines:
        brk = line.split(" : ")
        if text_key in brk[1]:

            timestamps.append(datetime.datetime.fromisoformat(brk[0]).timestamp()-9*3600)
    gstimes = np.array(timestamps)
    mask = np.logical_and(gstimes>mint, gstimes<maxt)
    return gstimes[mask]

def load_timeout(times):
    return get_event_time(times, "GS Status")
def load_bv1(times):
    return get_event_time(times, "pu3 on signal sent")

    
FILL_TIMES = load_log()

THRESH = 0.05
def get_start_times(times):
    mint = times.min()
    maxt = times.max()


    mask = np.logical_and(FILL_TIMES>mint, FILL_TIMES<maxt)
    return FILL_TIMES[mask]

__data = np.loadtxt(os.path.join(
    os.path.dirname(__file__), 
        "meta_history",
        "data_history.csv"
        ),
    delimiter=",").T


__times = __data[0]
__temps = __data[5]
__terpo = CubicSpline(__times, __temps, extrapolate=True)
def get_temperatures(sample_times:np.ndarray):
    return __terpo(sample_times)

def get_pressures(sample_times, pressure_no):
    # pressure shoudl be 1-4
    if pressure_no<1 or pressure_no>4:
        raise IndexError("Invalid pressure number, {}".format(pressure_no))
    else:
        __pressure = __data[pressure_no]
        terpo = CubicSpline(__times, __pressure, extrapolate=True)
        return terpo(sample_times)


def get_flow(sample_times, pressure_no):
    # pressure shoudl be 1-4
    if pressure_no<1 or pressure_no>5:
        raise IndexError("Invalid pressure number, {}".format(pressure_no))
    else:
        __pressure = __data[6+pressure_no]
        terpo = CubicSpline(__times, __pressure, extrapolate=True)
        return terpo(sample_times)


scalers = {1: -0.03772147450524445, 2: -0.011294464142536029, 3: -0.03861504974976303, 4: -0.015417349525356323, 5: -0.03092684049424451}
T0 = 28.16
def apply_rescale(ratios, times, waves):
    """
        Rescales the ratios according to measured temperature
        and measured correlation between temperature and ratio
    """
    tdiff = get_temperatures(times) -T0
    rdiff = np.ones_like(ratios)
    for i in range(1, 6):
        rdiff[waves==i] = tdiff[waves==i]*scalers[i]
    return ratios -rdiff



def _get_start_times(times, amplitudes, waveno):
    """
        Takes a full-time spectrum and extracts out the times when the tank was refilled. 
    """

    # get the times where it crosses 
    crossings = np.diff(np.sign(amplitudes - THRESH))
    crossings[crossings<0]=0
    crossings = np.where(crossings)[0]
    montime = times[crossings]

    starts = []

    for cross in crossings:
        this_waveno = waveno[cross]
        
        x1 = times[cross] 
        y1 = amplitudes[cross]

        cuttime = times[this_waveno==waveno]
        cutamp = amplitudes[this_waveno==waveno]

        x0 = times[cross-1]
        y0 = amplitudes[cross -1] 
     
        slope =  (y1-y0) / (x1-x0)
        y_interp = y0 -  slope * x0  
        
        # 0.1 = slope*x + b
        starts.append((THRESH-y_interp)/slope)
    return montime 


import matplotlib.pyplot as plt
def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)