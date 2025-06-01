import os 
import numpy as np 
import datetime 
import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline
import h5py as h5 


from scipy.interpolate import griddata, RectBivariateSpline
import numpy as np 
import matplotlib.pyplot as plt 
from math import log10,sqrt 

logfile = os.path.join(os.path.dirname(__file__), "meta_history", "command.log")

_cor_data = h5.File(
    os.path.join(os.path.dirname(__file__), "toymc","data","correction_data.h5"),
    "r"
)

class Irregular2DInterpolator:
    """
        This is used to make a 2D interpolator given a set of data that do not lie perfectly on a grid.
        This is done using scipy griddata and scipy RectBivariateSpline 
        interpolation can be `linear` or `cubic` 
        if linear_x/y, then the interpolation is done in linear space. Otherwise, it's done in log space
            setting this to False is helpful if your x/y values span many orders of magnitude 
        if linear_values, then the values are calculated in linear space. Otherwise they'll be evaluated in log space- but returned in linear space 
            setting this to False is helpful if your data values span many orders of magnitude 
        By default, nans are replaced with zeros. 
    """
    def __init__(self, xdata:np.ndarray, 
                 ydata:np.ndarray,
                   values:np.ndarray, linear_x = True, linear_y = True, linear_values=True,
                   replace_nans_with= 0.0, interpolation='linear'):

        self._nomesh_x = xdata
        self._nomesh_y = ydata 
        self._values = values if linear_values else np.log10(values)
        self._linear_values = linear_values
        if linear_x:
            self._xfine = np.linspace(min(self._nomesh_x), 
                                      max(self._nomesh_x), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)
        else:
            self._xfine = np.logspace(log10(min(self._nomesh_x)), 
                                      log10(max(self._nomesh_x)), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)

        
        if linear_y:
            self._yfine = np.linspace(min(self._nomesh_y), 
                                      max(self._nomesh_y), 
                                      int(sqrt(len(self._nomesh_y)))*2, endpoint=True)
        else:
            self._yfine = np.logspace(log10(min(self._nomesh_y)), 
                                      log10(max(self._nomesh_y)), 
                                      int(sqrt(len(self._nomesh_y)))*2, endpoint=True)

        mesh_x, mesh_y = np.meshgrid(self._xfine, self._yfine)

        # usee grideval to evaluate a grid of points 
        self._grid_eval = griddata(
            points=np.transpose([self._nomesh_x, self._nomesh_y]),
            values=self._values, 
            xi=(mesh_x, mesh_y),
            method=interpolation,
            fill_value=1.0
        )
        
        # if there are any nans, scipy 
        if np.any(np.isnan(self._grid_eval)):
            print("Warning! Nans were found in the evaluation of griddata - we're replacing those with zeros")
        self._grid_eval[np.isnan(self._grid_eval)] = replace_nans_with

        # and then prepare an interpolator 
        self._data_int = RectBivariateSpline(
            self._xfine, 
            self._yfine, 
            self._grid_eval.T
        )
    def plot_mesh(self):
        plt.pcolormesh(self._xfine, self._yfine, self._grid_eval.T)  
        plt.colorbar()
        plt.show()

    def __call__(self, xs, ys, grid=False):
        if self._linear_values:
            return self._data_int( xs, ys ,grid=grid)
        else:
            return 10**self._data_int( xs, ys , grid=grid)

true_signal = _cor_data["true_led"]
true_leak = _cor_data["true_leak"]
measured = np.array(_cor_data["measured"])
#for i in range(5):
#    measured[i*6:i*6+6] /= measured[i*6]    

__effective_scaler = Irregular2DInterpolator(
    measured, true_leak, true_signal/measured,
    replace_nans_with=1.0
)

if __name__=="__main__":
    __effective_scaler.plot_mesh()

def apply_correction(measured, dark):
    """
        applies a correction based on the intensity of the hitrate 

        dark is the number of dark PEs per trigger signal
        will return a correction factor to make things right 
    """

    return __effective_scaler(measured, dark)



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
        min_time = 35
        max_time = 50

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

            timestamps.append(datetime.datetime.fromisoformat(brk[0]).timestamp()-0*3600)
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

def get_light(sample_times, pressure_no):
    # pressure shoudl be 1-4
    if pressure_no<0 or pressure_no>1:
        raise IndexError("Invalid light number, {}".format(pressure_no))
    else:
        __pressure = __data[12+pressure_no]
        terpo = CubicSpline(__times, __pressure, extrapolate=True)
        return terpo(sample_times)

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