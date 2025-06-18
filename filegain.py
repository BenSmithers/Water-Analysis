import numpy as np
import h5py as h5 
import matplotlib.pyplot as plt 
import os 
from glob import glob 
from toymc.utils import get_rtime, get_cfd_time
from tqdm import tqdm
plt.style.use("wms.mplstyle")

all_files = glob(
    os.path.join(
        os.path.dirname(__file__),
        "meta_history",
        "waveforms*.hdf5"
    )
)
if False:
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

    all_files = [baseline, bad]

bins = np.linspace(0, 201, 129)

allmon = np.zeros(len(bins)-1)
allrec = np.zeros(len(bins)-1)

time_bin = np.arange(0, 380, 4)
rel_m = np.zeros(len(time_bin)-1)
rel_r = np.zeros(len(time_bin)-1)

for file in tqdm(all_files):
    #plt.clf()
    data = h5.File(file, 'r')
    trigger = np.array(data["trigger"])
    times = np.array(range(len(trigger)))*4
    chanb = np.array(data["monitor"])
    chand = np.array(data["receiver"])


    
    raw_crossings = np.diff(np.sign((trigger-0.0) - 1000))
    if True:
        raw_crossings[raw_crossings<0] = 0    
    else:
        raw_crossings[raw_crossings>0] = 0
    
    raw_crossings = np.where(raw_crossings)[0]
    crossings = times[raw_crossings]

    mon_time = get_cfd_time(times, -chanb, 10,use_rise= True)
    rec_time = get_cfd_time(times, -chand, 10, use_rise=True)
    mon_rtime = get_rtime(crossings, mon_time)
    rec_rtime = get_rtime(crossings, rec_time)

    window = int(370 / 4)
    skip = 0 # 42  int(0.6*window)
    
    mon_peaks = []
    rec_peaks = []
    for ic in raw_crossings:
        if len(chanb[ic+skip:ic+window])==0:
            continue
        mon_peaks.append(-1*np.min(chanb[ic+skip:ic+window]))
        rec_peaks.append(-1*np.min(chand[ic+skip:ic+window]))
    
    tallmon = np.histogram(mon_peaks, bins)[0]
    tallrec = np.histogram(rec_peaks, bins)[0]

    if np.sum(tallmon[60:])>500:
        print("Skip good")
        continue
    else:
        print(np.sum(tallmon[60:]))

    allmon += tallmon
    allrec += tallrec

    rel_m += np.histogram(mon_rtime, time_bin)[0]
    rel_r += np.histogram(rec_rtime, time_bin)[0]
    print(np.sum(rel_r))
plt.stairs(allmon, bins, label="Monitor")
plt.stairs(allrec, bins, label="Receiver")
plt.xlabel("Charge [ADC]")
plt.ylabel("Counts")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("./plots/badgain.png", dpi=400)
plt.show()

plt.clf()
plt.stairs(rel_m, time_bin, label="Monitor")
plt.stairs(rel_r, time_bin, label="Receiver")
plt.xlabel("Time Since Trig [ns]")
plt.legend()
plt.grid(which='both', alpha=0.5)
#plt.yscale('log')
plt.tight_layout()
plt.savefig("./plots/badtime.png", dpi=400)

plt.show()