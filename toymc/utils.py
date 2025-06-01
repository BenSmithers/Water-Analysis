
import numpy as np 

def get_cfd_time(times, signal, threshold, auto_adjust_ped = False, use_rise=False):

    crossings = np.diff(np.sign((signal-0.0) - threshold))
    if use_rise:
        crossings[crossings<0] = 0    
    else:
        crossings[crossings>0] = 0
    
    crossings = np.where(crossings)
    x0 = times[crossings[0]]
    return x0

def get_valid(trigs, hits, is_rec, invalid=False):
    window = 24
    if invalid:
        shift = 150
    else:
        shift = 0
    if is_rec:
        min_time = 104+shift
        max_time = min_time+window
    else:
        min_time = 36+shift
        max_time = min_time+window

    hit_trig_time = hits - trigs[np.digitize(hits, trigs)-1]
    good = np.logical_and( hit_trig_time>min_time, hit_trig_time<max_time)
    return good, np.logical_not(good)

def get_rtime(trigs, hits):
    hit_trig_time = hits - trigs[np.digitize(hits, trigs)-1]
    return hit_trig_time