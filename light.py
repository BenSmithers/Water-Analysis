import numpy as np 
import matplotlib.pyplot as plt 
from utils import get_pressures
from datetime import datetime
from utils import fold, average, get_color, get_light
N_MERGE = 1

plt.style.use("wms.mplstyle")

fname  = "data/picodat_run95_Supply Untreated_various_variousadc_mHz.dat"
#fname = "data/picodat_run93_Supply Untreated_255nm_818adc_mHz.dat"
labels = ["Monitor", "Receiver"]
data = np.loadtxt(fname, delimiter=",").T 
timestamp = data[0] #+ 3600*9
timestamp = average(timestamp, N_MERGE)
ttime = np.array([datetime.fromtimestamp(entry) for entry in timestamp])
for i in range(0,2):
    pressure = get_light(data[0], i)
    pressure = average(pressure, N_MERGE)
    
    
    plt.plot(ttime, pressure, color=get_color(i+1, 3), label=labels[i])

plt.ylabel("Light")
plt.xlabel("Time Stamp")
plt.gcf().autofmt_xdate()
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.savefig("./plots/light.png", dpi=400)
plt.show()