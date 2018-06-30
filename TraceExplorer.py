# coding: utf-8


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
from matplotlib.widgets import Slider, Button, RadioButtons
from Pyspectr.pydamm import *

def trap_filter(times,res,L,G):
    retvec = np.zeros(len(times))
    zidx = times.searchsorted(0)
    for i in range( zidx,len(times) ):
      #print(i)
      #j = i-zidx
      retvec[i]=(res[i-L+1:i]-res[i-2*L-G+1:i-L-G]).sum()
    return(retvec)

#def tau_adjust(response,tau,L,G):
#    for i in range(int(len(response)/2),len(response)):
#        response[i] = response[i-L-G]*tau
#    return(response)

def tau_adjust(pulse,tau):
    bls = cp(pulse)
    retvec = cp(pulse)
    bls -= pulse[:100].mean()
    for t in range(3,len(pulse)):
        pz = bls[:t-1].sum()
        retvec[t] += bls[t] + pz/tau 
    return(retvec)

def zero_crossing(trap):
    delay = cp(trap)
    td = 20
    cf = .8
    delay[:-td] += -cf*trap[td:] 
    return(delay)

#e=Experiment('Feb16Run.his') 
e=Experiment('HIS/1516Cal.his') 
hisnum = 1030
#hisnum = 900
plt.close()
plt.ioff()
t0=125
l0=100
g0=200
T0 = 20

pulse_histo = e.gy(hisnum,(t0,t0))
weights = e.hisfile.load_histogram(hisnum)[3]
pulse_len = 1000 #! len(pulse) actual pulse may not be the length of the histo.
pulse = np.zeros(2*pulse_len)
pulse[pulse_len:] += pulse_histo.histogram.weights[:pulse_len]
pulse[:pulse_len] += pulse_histo.histogram.weights[2]
t = np.arange(-pulse_len,pulse_len,1)  #!pulse_histo.histogram.x_axis-.5 needs to be integer and padded


#plt.ion()
ax1 = plt.subplot(3,2,1)
ax2 = plt.subplot(3,2,2,sharex=ax1)
ax3 = plt.subplot(3,2,3,sharex=ax1)
ax4 = plt.subplot(3,2,4,sharex=ax1)

fig = plt.figure(1)



norm = 1 #np.sqrt(2*3.14159)*s0
margin = 100

#pulse = fn(t,a0*norm,m0,s0)
pz = tau_adjust(pulse,tau)
ff = trap_filter(t,pz,l0,g0)
zc = zero_crossing(ff)

#input('ok')
#l,= ax1.plot(t,pulse,lw=2,color='red')
#ll,= ax2.plot(t,ff,lw=2,color='blue')
#plt.axis([0,2000,-100,100])
#ax1.autoscale(axis='y')
#ax2.autoscale(axis='y')

l,= ax1.plot(t,pulse,lw=2,color='red')
ax1.legend(['Input Pulse'])
l2,= ax2.plot(t,pz,lw=2,color='k')
ax2.legend(['Pole-zero/Tau Corrected'])
l3,= ax3.plot(t,ff,lw=2,color='blue')
ax3.legend(['Trapezoidal Filter Output'])
l4,= ax4.plot(t,zc,lw=2,color='green')
ax4.legend(['CFD Output'])

ax2.set_xlim(0,pulse_len)
ax1.set_ylim(pulse.min()-margin,pulse.max()+margin)
ax2.set_ylim(pz.min()-margin*10,pz.max()+margin*10)
ax3.set_ylim(ff.min()-margin*10,ff.max()+margin*10)
ax4.set_ylim(zc.min()-margin*10,zc.max()+margin*10)

#axamp = plt.axes([0.15,0.05, 0.65, 0.03])
#axmean = plt.axes([0.15,0.09, 0.65, 0.03])
#axsigma = plt.axes([0.15,0.13, 0.65, 0.03])
axlen = plt.axes([0.15,0.17, 0.65, 0.03])
axgap = plt.axes([0.15,0.21, 0.65, 0.03])
axid = plt.axes([0.15,0.13, 0.65, 0.03])
axtau = plt.axes([0.15,0.09, 0.65, 0.03])
#samp = Slider(axamp, 'Amp', 1, 500, valinit=a0)
#smean = Slider(axmean, 'Mean', 1, 1000.0, valinit=m0)
#ssigma = Slider(axsigma, 'Sigma', 1, 1000.0, valinit=s0)
slen = Slider(axlen, 'Length', 1, 1000.0, valinit=l0)
sgap = Slider(axgap, 'Gap', 1, 1000.0, valinit=g0)
sid = Slider(axid, 'Trace id', 0, 255, valinit=t0)
stau = Slider(axtau, 'Tau', 0, 50, valinit=T0)

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
    tau = stau.val
    pulse[pulse_len:] =weights[:,trace_id][:pulse_len]
    pulse[:pulse_len] =weights[:,trace_id][2]
    pz = tau_adjust(pulse, tau, length,gap)
    ff = CFD(t,pz,length,gap)
    zc = zero_crossing(ff)
    l.set_ydata( pulse )
    l2.set_ydata( pz )
    l3.set_ydata( ff )
    l4.set_ydata( zc )
#    l.set_ydata( pulse )
#    ll.set_ydata( ff )
    ax1.set_ylim(pulse.min()-margin,pulse.max()+margin)
    ax2.set_ylim(pz.min()-margin,pz.max()+margin)
    ax3.set_ylim(ff.min()-margin,ff.max()+margin)
    ax4.set_ylim(zc.min()-margin,zc.max()+margin)
    fig.canvas.draw_idle()
    
#samp.on_changed(update)
#smean.on_changed(update)
#ssigma.on_changed(update)
slen.on_changed(update)
sgap.on_changed(update)
sid.on_changed(update)
stau.on_changed(update)

plt.show()



