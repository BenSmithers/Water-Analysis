"""
    There will be a non-linearity! 
"""
import numpy as np
import os 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt 
_gain_file = os.path.join(os.path.dirname(__file__), "data","gain.dat")
_gain_data = np.loadtxt(_gain_file,delimiter=",").T
_gain_data[1] /=np.sum(_gain_data[1])
_cumsum = np.cumsum(_gain_data[1])

_gain_spline = CubicSpline(_cumsum, _gain_data[0])

NS =1 
_pulse_width = 10*NS # sigma
_sample_period = 4*NS 
_n_samples = int(368*NS/4*NS)
_led_intensity = 0.3 
_offset = 21.47*NS

def sample_n(size): # sample N over 10
    vals = np.random.random(size)
    return _gain_spline(vals)

def gen_block(n_triggers, leak_intensity):
    
    trigger = np.repeat(np.tile([0,1],n_triggers), int(_n_samples/2))*100 - 50
    times = np.array(range(len(trigger)))*_sample_period
    signal = np.zeros_like(trigger, dtype=float)

    # numbers are large enough so that it's approximately gaussian... 
    n_signal = int(np.random.randn()*np.sqrt(_led_intensity*n_triggers) + _led_intensity*n_triggers)
    hit_times = np.random.randint(0, n_triggers, n_signal)*368*NS + _offset
    hit_heights = sample_n(n_signal)

    for i in range(len(hit_times)):
        signal -= hit_heights[i]*np.exp( -0.5*((times - hit_times[i])/_pulse_width )**2 )
    
    n_leak = int(np.random.randn()*np.sqrt(leak_intensity*n_triggers) + leak_intensity*n_triggers)
    hit_times = np.random.random(n_leak)*times.max()
    hit_heights = sample_n(n_leak   )
    for i in range(len(hit_times)):
        signal -= hit_heights[i]*np.exp( -0.5*((times - hit_times[i])/_pulse_width )**2 )    

    return times, trigger, signal


times, trigger, sig = gen_block(1000, 0.3)

plt.plot(times, trigger, label="Trigger")
plt.plot(times, sig, label="Signal")
plt.show()