from dr_pseudo import gen_block
from utils import get_cfd_time, get_rtime, get_valid
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm
from math import log , sqrt 
def measure(iterate = 50, leak = 0.0, led_intensity=0.3):
    """
        Generates a block `iterate` times and returns the number of triggers, good hits, and "bad" hits
    """
    n_good = 0
    n_bad = 0
    n_trig = 0 
    for i in tqdm(range(iterate)):
        times, trigger, sig = gen_block(1000, leak, led_intensity)

        trigger_time = get_cfd_time(times, trigger, 1000, use_rise = True)
        
        sig_time = get_cfd_time(times, -sig, 10, True )

        n_good += np.sum(get_valid(trigger_time, sig_time, False)[0])
        n_bad += np.sum(get_valid(trigger_time, sig_time, False, invalid=True)[0])
        n_trig += len(trigger_time)
    return n_trig, n_good, n_bad

def tdiff():
    time_bin = np.arange(0, 380, 4)
    binny = np.zeros(len(time_bin)-1 )
    for i in tqdm(range(500)):
        times, trigger, sig = gen_block(1000, 0.3)


        trigger_time = get_cfd_time(times, trigger, 1000, use_rise = True)
        
        sig_time = get_cfd_time(times, -sig, 10, True )    

        rtime = get_rtime(trigger_time, sig_time)

        
        binny += np.histogram(rtime, time_bin)[0]
    plt.stairs(binny, time_bin)
    plt.show()

import h5py as h5 

if __name__=="__main__":
    leak_test = [0, 0.01, 0.05, 0.1,  0.3, 0.5, 1.0]
    led_intensities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.6]
    signal = [] # effective measured ones 
    errors = []

    true_led = []
    all_leak = []

    for led in led_intensities:
        for leak in leak_test:
            trig, good, bad = measure(iterate=500, leak=leak,led_intensity=led)

            if bad==0:
                errors.append(((trig-good)/(trig - bad))*sqrt( (1/sqrt(good)) ) )
            else:
                errors.append(((trig-good)/(trig - bad))*sqrt( (1/sqrt(good)) + 1/sqrt(bad) ))

            signal.append(good / trig)
            true_led.append(led)
            all_leak.append(leak)


    all_data = {
        "true_led":true_led ,
        "true_leak":all_leak,
        "measured":signal
    }
    data = h5.File("./data/correction_data.h5",'w')
    for key in all_data:
        data.create_dataset(key, data=all_data[key])
    data.close()
