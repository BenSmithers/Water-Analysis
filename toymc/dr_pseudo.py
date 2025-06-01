"""
    There will be a non-linearity! 
"""
import numpy as np
import os 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt 
from tqdm import tqdm
from math import sqrt 
_gain_file = os.path.join(os.path.dirname(__file__), "data","gain.dat")
_gain_data = np.loadtxt(_gain_file,delimiter=",").T
_gain_data[1] /=np.sum(_gain_data[1])
_cumsum = np.cumsum(_gain_data[1])

_gain_spline = CubicSpline(_cumsum, _gain_data[0])

NS =1 
_pulse_width = 15*NS # sigma
_sample_period = 4*NS 
_n_samples = int(368*NS/4*NS)
_led_intensity = 0.05 
_offset = -210*NS

def sample_n(size): # sample N over 10
    vals = np.random.random(size)
    return _gain_spline(vals)

"""
Signal is 
    sigma*sqrt(-2log(10/A))+mu=  x
"""

def gen_block(n_triggers, leak_intensity, led_intensity=_led_intensity):
    
    trigger = np.repeat(np.tile([0,1],n_triggers), int(_n_samples/2))*2000 - 500
    times = np.array(range(len(trigger)))*_sample_period
    signal = np.zeros_like(trigger, dtype=float)

    # numbers are large enough so that it's approximately gaussian... 
    n_signal = int(np.random.randn()*np.sqrt(led_intensity*n_triggers) + led_intensity*n_triggers)
    if n_signal<0:
        n_signal=0
    hit_heights = sample_n(n_signal)
    
    hit_times = np.random.randint(0, n_triggers, n_signal)*368*NS +_offset
    hit_times = hit_times.astype(float)
    shift = _pulse_width*np.sqrt(-2*np.log(0.01)) 
    hit_times += shift
    

    _mesh_time, _mesh_height = np.meshgrid(times, hit_heights)
    _mesh_time, _mesh_hit = np.meshgrid(times, hit_times)
    signal += np.sum ( -_mesh_height*np.exp( -0.5*((_mesh_time - _mesh_hit)/ _pulse_width)**2),axis=0)

    
    n_leak = int(np.random.randn()*np.sqrt(leak_intensity*n_triggers) + leak_intensity*n_triggers)
    if n_leak<0:
        n_leak = 0
    hit_times = np.random.random(n_leak)*times.max()
    hit_heights = sample_n(n_leak   )
    _mesh_time, _mesh_height = np.meshgrid(times, hit_heights)
    _mesh_time, _mesh_hit = np.meshgrid(times, hit_times)
    signal += np.sum ( -_mesh_height*np.exp( -0.5*((_mesh_time - _mesh_hit)/ _pulse_width)**2),axis=0)


    return times, trigger, signal

if __name__=="__main__":
    times, trigger, sig = gen_block(1000, 0.0)

    plt.plot(times, trigger, label="Trigger")
    plt.plot(times, sig, label="Signal")
    plt.show()