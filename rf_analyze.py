import numpy as np
import h5py as h5 
import os 
import matplotlib.pyplot as plt 
from scipy import fftpack, fft 
from utils import average
from scipy.optimize import minimize
plt.style.use("wms.mplstyle")

NAVG=1000

baseline = os.path.join(
    os.path.dirname(__file__),
    "meta_history",
    "waveforms_1748481079.7464.hdf5"
)

bad = os.path.join(
    os.path.dirname(__file__),
    "meta_history",
    "waveforms_1748839598.3849.hdf5"
)

baseline_data = h5.File(baseline, 'r')
bad_data = h5.File(bad, 'r')

def process(datafile):
    receiver = np.array(datafile["receiver"])
    times = np.array(range(len(receiver)))*4
    
    sample_spacing = times[1]-times[0] 
    freq_space = fft.fft(receiver)
    frequencies = fftpack.fftfreq(len(receiver), sample_spacing)

    mon_freq = fft.fft(np.array(datafile["monitor"]))
    trig_freq = fft.fft(np.array(datafile["trigger"]))

    return average(frequencies, NAVG)*(1e9), np.log10(average(mon_freq, NAVG)),np.log10(average(freq_space, NAVG)), np.log10(average(trig_freq, NAVG))


base_freq, base_mon_freq, base_rec_freq, base_trig = process(baseline_data)
bad_freq, bad_mon_freq, bad_rec_freq, bad_trig = process(bad_data)

mask = bad_freq*(1e-6)>0


def metric(params):
    return np.sum((bad_amp - params[0]*base_amp)**2)
if False:
    res = minimize(
        metric, [1,], bounds=[[0,np.inf]]
    )

    scale_res = res.x 

if True:
    #plt.plot(bad_freq, bad_amp - scale_res[0]*base_amp)
    plt.plot((1e-6)*bad_freq[mask], 1000*bad_trig[mask]/np.sum(bad_trig[mask]), label="Trigger",alpha=0.5)
    plt.plot((1e-6)*bad_freq[mask], 0.1+1000*bad_mon_freq[mask]/np.sum(bad_mon_freq[mask]), label="Monitor",alpha=0.5)
    plt.plot((1e-6)*base_freq[mask], 0.2+1000*bad_rec_freq[mask]/np.sum(bad_rec_freq[mask]), label="Receiver",alpha=0.5)
    #plt.xlim([1e4, 2e8])
    plt.xscale('log')

    #plt.xlim()
    #plt.xlabel("Wavelength [m]")
    plt.xlabel("Frequency [MHz]")
    plt.grid(which='both', alpha=0.5)
    plt.ylabel("Arb")
    plt.tight_layout()
    plt.legend()
    plt.show()

if False:
    plt.plot(4*np.array(range(len(highpass[:10000]))), np.array(bad_data["trigger"][:10000])/200, label="Trig")
    plt.plot(4*np.array(range(len(highpass[:10000]))), bad_data["receiver"][:10000], label="Raw")
    plt.plot(4*np.array(range(len(highpass[:10000]))), highpass[:10000], label="Cut")
    plt.legend()
    plt.show()