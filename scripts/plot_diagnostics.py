#!/usr/bin/env python3
"""Noiseless demonstrations of Morse phases and a detector/image null test.

The chirp is an illustrative analytic signal, not a modeled black-hole merger.
The cross difference is an algebraic check, not a significance statistic.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gw_diffraction.diagnostics import morse_factor,image_detector_cross_difference
from gw_diffraction.waveoptics import response,project_modes


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    n=2049;dt=.0002
    t=(np.arange(n)-n//2)*dt
    h=np.exp(-(t/.050)**2)*np.cos(2*np.pi*(55*t+75*t*t))
    h-=h.mean()
    frequency=np.fft.fftfreq(n,dt)
    # Positive-sign forward transform and negative-sign inverse, matching note.
    spectrum=np.fft.ifft(h)*n
    signals=[(np.fft.fft(spectrum*morse_factor(frequency,index))/n).real for index in [0,.5,1]]
    fig,ax=plt.subplots(1,2,figsize=(10.6,3.9),layout='constrained')
    for y,color,label in zip(signals,['#14627a','#cd732e','#755290'],['Minimum: unchanged','Saddle: quadrature / Hilbert','Maximum: sign reversed']):
        ax[0].plot(t*1000,y,color=color,label=label,lw=1.4)
    ax[0].set(xlabel='Time relative to the aligned pulse [ms]',ylabel='Illustrative strain (arbitrary units)',xlim=(-65,65),title='Macro image phase is not always a sign')
    ax[0].legend(fontsize=8,loc='lower left')

    f=np.geomspace(10,100,100)
    w=np.ones_like(f)/len(f)
    lens1=1+.004*project_modes(response(f,.3,0,(.1,1)),f,w)
    lens2=1+.003*project_modes(response(f,.4,.3,(-.1,1)),f,w)
    source=np.exp(.15j*np.log(f/10))
    gain_a=1.03*np.exp(.02j*np.log(f/10));gain_b=.98*np.exp(-.01j*np.log(f/10))
    a1=gain_a*lens1*source;a2=gain_a*lens2*source
    b1=gain_b*lens1*source;b2=gain_b*lens2*source
    stable=image_detector_cross_difference(a1,a2,b1,b2)
    drift=.005*np.exp(-.5*(np.log(f/40)/.3)**2)*(1+.6j)
    one=image_detector_cross_difference(a1*(1+drift),a2,b1,b2)
    common=image_detector_cross_difference(a1*(1+drift),a2,b1*(1+drift),b2)
    scale=np.abs(source**2*gain_a*gain_b)
    ax[1].semilogx(f,100*np.abs(stable)/scale,color='#14627a',label='Stable calibration; physical lensing')
    ax[1].semilogx(f,100*np.abs(one)/scale,color='#cd732e',label='Calibration drift in one detector')
    ax[1].semilogx(f,100*np.abs(common)/scale,color='#755290',ls='--',label='Common drift: a blind spot')
    ax[1].set(xlabel='Observed frequency [Hz]',ylabel='Normalized cross-difference [%]',title='Detector/image null: noiseless example')
    ax[1].legend(fontsize=8)
    (ROOT/'figures').mkdir(exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(ROOT/'figures'/f'morse_and_systematics.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
