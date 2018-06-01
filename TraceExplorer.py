# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
from matplotlib.widgets import Slider, Button, RadioButtons


fig , ax = plt.subplots()

def fn(x,a,m,s):
    return((a/np.sqrt(2*3.14159)/s * np.exp(-0.5 * (x-m) * (x-m) / s**2) )+200+np.random.randn(len(x)))

def CFD(times,res,L,G):
    retvec = np.zeros(len(res))
    zidx = times.searchsorted(0)
    for i in range(zidx,len(times)):
#      print(i,i-L+1,i-2*L-G+1,i-L-G)
#      if i> 2*L - G:
      retvec[times.searchsorted(times[i])]=(res[i-L+1:i]-res[i-2*L-G+1:i-L-G]).sum()
    return(retvec)


t = np.arange(-2000,2000,1)
a0=200
m0=300
s0=150
l0=100
g0=200

pulse = fn(t,a0,m0,s0)
ff = CFD(t,pulse,l0,g0)

l,= plt.plot(t,pulse,lw=2,color='red')
ll,= plt.plot(t,ff,lw=2,color='blue')
plt.axis([0,2000,-1000,1000])

axamp = plt.axes([1/4,1/10, 0.65, 0.03])
axmean = plt.axes([1/4,0.15, 0.65, 0.03])
axsigma = plt.axes([1/4,0.20, 0.65, 0.03])
axlen = plt.axes([1/4,0.25, 0.65, 0.03])
axgap = plt.axes([1/4,0.30, 0.65, 0.03])

samp = Slider(axamp, 'Amp', 1, 500, valinit=a0)
smean = Slider(axmean, 'Mean', 1, 1000.0, valinit=m0)
ssigma = Slider(axsigma, 'Sigma', 1, 1000.0, valinit=s0)
slen = Slider(axlen, 'Length', 1, 1000.0, valinit=l0)
sgap = Slider(axgap, 'Gap', 1, 1000.0, valinit=g0)

def update(val):
    mean = smean.val
    sigma = ssigma.val
    amp = samp.val
    slen.val = round(slen.val)
    length = slen.val
    sgap.val = round(sgap.val)
    gap = sgap.val
#    pulse = fn(t,amp,mean,sigma)
    l.set_ydata( fn(t,amp*np.sqrt(2*3.14159)*sigma,mean,sigma) )
    ll.set_ydata( CFD(t,fn(t,amp,mean,sigma),length,gap) )
    fig.canvas.draw_idle()

samp.on_changed(update)
smean.on_changed(update)
ssigma.on_changed(update)
slen.on_changed(update)
sgap.on_changed(update)

plt.show()

