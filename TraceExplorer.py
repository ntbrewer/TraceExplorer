# coding: utf-8


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
from matplotlib.widgets import Slider, Button, RadioButtons
from Pyspectr.pydamm import *

def CFD(times,res,L,G):
    retvec = np.zeros(len(times))
    zidx = times.searchsorted(0)
    for i in range( zidx,len(times) ):
      #print(i)
      #j = i-zidx
      retvec[i]=(res[i-L+1:i]-res[i-2*L-G+1:i-L-G]).sum()
    return(retvec)

e=Experiment('Feb16Run.his') 
plt.close()
plt.ioff()
t0=125
l0=100
g0=200
pulse_histo = e.gy(900,(t0,t0))
weights = e.hisfile.load_histogram(900)[3]
pulse_len = 1750 #! len(pulse) actual pulse may not be the length of the histo.
pulse = np.zeros(2*pulse_len)
pulse[pulse_len:] += pulse_histo.histogram.weights[:pulse_len]
pulse[:pulse_len] += pulse_histo.histogram.weights[2]
t = np.arange(-pulse_len,pulse_len,1)  #!pulse_histo.histogram.x_axis-.5 needs to be integer and padded


#plt.ion()
ax2 = plt.subplot(3,1,2)
ax1 = plt.subplot(3,1,1,sharex=ax2)
fig = plt.figure(1)



norm = 1 #np.sqrt(2*3.14159)*s0
margin = 100

#pulse = fn(t,a0*norm,m0,s0)
ff = CFD(t,pulse/norm,l0,g0)
#input('ok')
l,= ax1.plot(t,pulse,lw=2,color='red')
ll,= ax2.plot(t,ff,lw=2,color='blue')
ax2.set_xlim(0,pulse_len)
ax1.set_ylim(pulse.min()-margin,pulse.max()+margin)
ax2.set_ylim(ff.min()-margin*10,ff.max()+margin*10)
#plt.axis([0,2000,-100,100])
#ax1.autoscale(axis='y')
#ax2.autoscale(axis='y')

#axamp = plt.axes([0.15,0.05, 0.65, 0.03])
#axmean = plt.axes([0.15,0.09, 0.65, 0.03])
#axsigma = plt.axes([0.15,0.13, 0.65, 0.03])
axlen = plt.axes([0.15,0.17, 0.65, 0.03])
axgap = plt.axes([0.15,0.21, 0.65, 0.03])
axid = plt.axes([0.15,0.13, 0.65, 0.03])

#samp = Slider(axamp, 'Amp', 1, 500, valinit=a0)
#smean = Slider(axmean, 'Mean', 1, 1000.0, valinit=m0)
#ssigma = Slider(axsigma, 'Sigma', 1, 1000.0, valinit=s0)
slen = Slider(axlen, 'Length', 1, 1000.0, valinit=l0)
sgap = Slider(axgap, 'Gap', 1, 1000.0, valinit=g0)
sid = Slider(axid, 'Trace id', 0, 255, valinit=t0)

def update(val):
    #mean = smean.val
    #sigma = ssigma.val
    #amp = samp.val
    slen.val = round(slen.val)
    length = slen.val
    sgap.val = round(sgap.val)
    gap = sgap.val
    sid.val = int(sid.val)
    trace_id = sid.val
    pulse[pulse_len:] =weights[:,trace_id][:pulse_len]
    ff = CFD(t,pulse/norm,length,gap)
    l.set_ydata( pulse )
    ll.set_ydata( ff )
    ax1.set_ylim(pulse.min()-margin,pulse.max()+margin)
    ax2.set_ylim(ff.min()-margin,ff.max()+margin)
    fig.canvas.draw_idle()
    
#samp.on_changed(update)
#smean.on_changed(update)
#ssigma.on_changed(update)
slen.on_changed(update)
sgap.on_changed(update)
sid.on_changed(update)

plt.show()



