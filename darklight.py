
import numpy as np 
import matplotlib.pyplot as plt 
from utils import get_pressures
from datetime import datetime
from utils import fold, average, get_color, get_light
N_MERGE = 1
light_bin = np.linspace(0, 1500, 41)
dark_bin = np.logspace(-4,-1, 60)

mondat = np.zeros((59,40))
recdat = np.zeros((59,40))
labels = ["Monitor", "Receiver"]

plt.style.use("wms.mplstyle")
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
#fname  = "data/picodat_run95_Supply Untreated_various_variousadc_mHz.dat"


files = [#"data/picodat_run87_Supply Untreated_various_variousadc_mHz.dat",
         #"data/picodat_run88_Supply Untreated_various_variousadc_mHz.dat",
        #"data/picodat_run89_Supply Untreated_various_variousadc_mHz.dat",
        #"data/picodat_run90_Supply Untreated_various_variousadc_mHz.dat",
        "data/picodat_run92_Supply Untreated_various_variousadc_mHz.dat",
        "data/picodat_run95_Supply Untreated_various_variousadc_mHz.dat",]

for fname in files:

    data = np.loadtxt(fname, delimiter=",").T 

    wave = data[7]
    mask = wave ==-1

    mon_dark = fold(data[4][mask]/data[1][mask], N_MERGE)
    rec_dark = fold(data[5][mask]/data[1][mask], N_MERGE)

    light_mon = average(get_light(data[0], 0)[mask], N_MERGE)
    light_rec = average(get_light(data[0], 1)[mask], N_MERGE)

    mondat+=np.histogram2d(mon_dark, light_mon, bins=(dark_bin, light_bin))[0]
    recdat+=np.histogram2d(rec_dark, light_rec, bins=(dark_bin, light_bin))[0]

axes[0].set_title("Monitor PMT")
axes[1].set_title("Receiver PMT")
axes[0].pcolormesh(dark_bin, light_bin, np.log10(mondat.T))
axes[0].set_ylabel("Ambient Light [lux]")
axes[0].set_xlabel("Dark Rate")
axes[1].set_xlabel("Dark Rate")
axes[1].set_xscale('log')

axes[1].pcolormesh(dark_bin, light_bin, np.log10(recdat.T))
plt.tight_layout()
plt.show()



