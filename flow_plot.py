import numpy as np 
import matplotlib.pyplot as plt 
from utils import get_pressures
from datetime import datetime
from utils import fold, average, get_color, get_flow
N_MERGE = 1

plt.style.use("wms.mplstyle")

fname  = "data/picodat_run87_Supply Untreated_various_variousadc_mHz.dat"
data = np.loadtxt(fname, delimiter=",").T 
timestamp = data[0] + 9*3600
timestamp = average(timestamp, N_MERGE)
ttime = np.array([datetime.fromtimestamp(entry) for entry in timestamp])
for i in range(1,6):
    pressure = get_flow(data[0], i)
    pressure = average(pressure, N_MERGE)
    
    
    plt.plot(ttime, pressure, color=get_color(i+1, 7), label="Flow {}".format(i))

plt.ylabel("Flow On/Off")
plt.xlabel("Time Stamp")
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.legend()
plt.savefig("./plots/flow.png", dpi=400)
plt.show()