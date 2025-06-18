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

window = int(370/4)
window = 80
window_times = np.arange(window)*4

skip =0 

for file in tqdm(all_files):
    #plt.clf()
    data = h5.File(file, 'r')
    trigger = np.array(data["trigger"])
    times = np.array(range(len(trigger)))*4
    chanb = np.array(data["monitor"])
    chand = np.array(data["receiver"])
    
    crossings = np.diff(np.sign((trigger-0.0) - 1000))
    if True:
        crossings[crossings<0] = 0    
    else:
        crossings[crossings>0] = 0
    
    crossings = np.where(crossings)[0]
    
    receiver = np.zeros(window)
    monitor = np.zeros(window)
    avg_trig = np.zeros(window)

    n_skipped = 0
    n_do = 0
    for ic in crossings:
        #print(np.shape(chand[ic+skip:ic+window]), np.shape(receiver))
        if len(chanb[ic+skip:ic+window])!=80:
            n_skipped+=1
            continue
        else:
            n_do +=1

        receiver += chand[ic+skip:ic+window]
        monitor += chanb[ic+skip:ic+window]
        avg_trig += trigger[ic+skip:ic+window]


    plt.plot(window_times, monitor/n_do, label="Monitor")
    plt.plot(window_times, receiver/n_do, label="Receiver")
    #plt.plot(window_times,avg_trig/(200*n_do), label="Trigger/200")
    plt.xlabel("Tim Since Trig [ns]")
    plt.ylabel("Avg. Voltage [mV]")
    plt.ylim([-3, 3])
    plt.legend()
    plt.tight_layout()
    plt.show()
