#!/usr/bin/env python3
"""Regenerate all figures/tables; run from the repository root.

This is an exploratory same-plane, weak-screen calculation. Input SNR is a
parameter; no ET/CE or speculative fourth-generation noise curve is assumed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gw_diffraction.waveoptics import response,project_modes,projected_covariance,scale_free_gain
from gw_diffraction.inference import covariance_metrics,eigenvalue_metrics,whitened_covariance,effective_snr
from gw_diffraction.physical import Geometry,SubhaloModel,stellar_power,delta2_from_physical_power,point_mass_wave_parameter


def band_grid(low,high,n=64,weighting='log_uniform'):
    f=np.geomspace(low,high,n)
    dl=np.empty(n)
    dl[0]=(np.log(f[1])-np.log(f[0]))/2
    dl[-1]=(np.log(f[-1])-np.log(f[-2]))/2
    dl[1:-1]=(np.log(f[2:])-np.log(f[:-2]))/2
    w=dl if weighting=='log_uniform' else dl*f**(-4/3)
    return f,w/w.sum()


def radial_response(logq,angles,f,w,eigs):
    """R(q) = angle average of the weighted projected |H|^2."""
    q=np.exp(logq)
    result=np.zeros_like(q)
    for a in angles:
        h=project_modes(response(f,q,a,eigs),f,w)
        result+=np.sum(w*np.abs(h)**2,axis=-1)/len(angles)
    return result


def savefig(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close(fig)


def covariance_foreground_metrics(cdm,cstar,w,rho):
    """Optimal Gaussian DM discrimination with fixed stellar covariance.

    Additionally return a local weak-DM Fisher information after marginalizing
    one stellar covariance amplitude with no prior. Neither metric is a p-value.
    """
    d=whitened_covariance(cdm,w,rho)
    s=whitened_covariance(cstar,w,rho)
    ev,vec=np.linalg.eigh(np.eye(len(s))+s)
    invsqrt=(vec/np.sqrt(ev))@vec.T
    dd=invsqrt@d@invsqrt
    ss=invsqrt@s@invsqrt
    metrics=eigenvalue_metrics(np.maximum(np.linalg.eigvalsh(dd),0))
    fdd=.5*np.sum(dd*dd.T)
    fss=.5*np.sum(ss*ss.T)
    fds=.5*np.sum(dd*ss.T)
    metrics['weak_dm_fisher_fixed_star']=float(np.sqrt(max(fdd,0)))
    metrics['weak_dm_fisher_free_star_amplitude']=float(np.sqrt(max(fdd-fds*fds/fss,0))) if fss>0 else float(np.sqrt(fdd))
    metrics['fisher_shape_cosine']=float(fds/np.sqrt(fdd*fss)) if fdd*fss>0 else 0.
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,default=1,help='Multiply radial/angular sample counts for convergence checks')
    args=parser.parse_args()
    r=args.resolution
    if r<1: raise ValueError('resolution must be positive')
    for folder in ['figures','results','docs']:(ROOT/folder).mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'figure.constrained_layout.use':True,
                         'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#14627a','#cd732e','#755290','#498d57']
    logq=np.linspace(-12,7.5,1500*r)
    q=np.exp(logq)
    angles=(np.arange(56*r)+.5)*np.pi/(56*r)
    f,w=band_grid(10,100)
    geometry=Geometry()
    rf=float(geometry.fresnel_kpc(100))
    k=q/rf
    model=SubhaloModel()
    models={f'c_ref_{c}':SubhaloModel(concentration_ref=c) for c in [15.,30.,60.]}
    powers={key:m.power(k,geometry) for key,m in models.items()}
    pstar=np.full_like(k,stellar_power(geometry))
    spectra={key:delta2_from_physical_power(q,p,rf) for key,p in powers.items()}
    spectra['stars']=delta2_from_physical_power(q,pstar,rf)
    mus=np.geomspace(1,100,25)
    rms={key:[] for key in spectra}
    min_rms={key:[] for key in spectra}
    saddle_rms={key:[] for key in spectra}
    cache={}
    for i,mu in enumerate(mus):
        rp=radial_response(logq,angles,f,w,(1/mu,1))
        rm=radial_response(logq,angles,f,w,(-1/mu,1))
        for key,spec in spectra.items():
            vp=float(np.trapezoid(spec*rp,logq)); vm=float(np.trapezoid(spec*rm,logq))
            min_rms[key].append(np.sqrt(vp));saddle_rms[key].append(np.sqrt(vm));rms[key].append(np.sqrt(vp+vm))
        if i%6==0:print(f'magnification grid {i+1}/{len(mus)}',flush=True)
    for d in [rms,min_rms,saddle_rms]:
        for key in d:d[key]=np.array(d[key])

    # Conditional exact power-law results.
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.0))
    mm=np.geomspace(1,100,180)
    for color,n in zip(colors,[-3,-2,-1,0]):
        g=np.array([scale_free_gain(n,(1/mu,1)) for mu in mm])
        ax[0].loglog(mm,g,label=fr'$P_\kappa\propto k^{{{n}}}$',color=color)
        ax[1].loglog(mm,np.sqrt(mm)*g,color=color,label=fr'$n={n}$')
    ax[0].set(xlabel=r'Absolute macro magnification $|\mu|$',ylabel='Fractional RMS enhancement $G$',title='Same screen; projected frequency residual')
    ax[1].set(xlabel=r'Absolute macro magnification $|\mu|$',ylabel='Known-residual SNR enhancement',title='Fractional effect × strain amplification')
    ax[0].legend();ax[1].legend()
    savefig(fig,'scale_free_enhancement')

    # Unit-independent SNR thresholds and exact illustrative flat-Delta^2 result.
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.1))
    for color,rr in zip(colors,[20,50,100,300]):
        required=5/(rr*np.sqrt(3/8*mm*(mm**2+1)))
        ax[0].loglog(mm,required,color=color,label=fr'$\rho_0={rr}$')
    ax[0].set(xlabel=r'Per-image $|\mu|$ in an idealized fold pair',ylabel=r'Required baseline per-image $\epsilon_0$',title='Known-realization residual SNR = 5')
    ax[0].legend()
    observed=np.geomspace(20,3000,160)
    ax[1].loglog(observed,5*np.sqrt(2)/observed,color=colors[0])
    ax[1].axhline(.01,color=colors[1],ls='--',label='1% differential residual')
    ax[1].axhline(.001,color=colors[2],ls='--',label='0.1% differential residual')
    ax[1].set(xlabel='Observed SNR of each image',ylabel=r'Required differential RMS $\epsilon_\Delta$',title='Equal images; any residual spectrum')
    ax[1].legend()
    savefig(fig,'snr_requirements')

    # Explicit local population spectra and projected differential power.
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.2))
    for color,key in zip(colors,models):
        ax[0].loglog(k/1000,spectra[key],color=color,label=key.replace('c_ref_','$c_6=')+'$')
        ax[1].loglog(mus,rms[key],color=color,label=key.replace('c_ref_','$c_6=')+'$')
    ax[0].loglog(k/1000,spectra['stars'],color='#555555',ls='--',label='Poisson stars (weak screen)')
    ax[1].loglog(mus,rms['stars'],color='#555555',ls='--',label='Poisson stars (weak screen)')
    ax[0].set(xlabel='Physical wavenumber $k$ [pc$^{-1}$]',ylabel=r'$\Delta_\kappa^2=k^2 P_\kappa/(2\pi)$',title='Illustrative local mass populations',xlim=(.01,1e4),ylim=(1e-12,1e-1))
    ax[1].set(xlabel=r'Per-image $|\mu|$',ylabel='Projected differential RMS',title='10–100 Hz, equal log-frequency weight')
    ax[0].legend(fontsize=8);ax[1].legend(fontsize=8)
    savefig(fig,'local_population_comparison')

    # Band dependence for the physical benchmark, and contributing masses.
    centers=np.geomspace(.003,100,21)
    band_rms={}
    for mu in [1.,30.]:
        vals=[];starvals=[]
        for center in centers:
            ff,ww=band_grid(center/np.sqrt(10),center*np.sqrt(10))
            rr=radial_response(logq,angles,ff,ww,(1/mu,1))
            vals.append(np.sqrt(np.trapezoid(spectra['c_ref_30.0']*rr,logq)))
            starvals.append(np.sqrt(np.trapezoid(spectra['stars']*rr,logq)))
        band_rms[str(mu)]={'subhalos':vals,'stars':starvals}
    pp,mass,integrand=model.power(k,geometry,True)
    mass_delta=k[:,None]**2/(2*np.pi)*integrand
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.2))
    for color,mu in zip(colors,[1.,30.]):
        ax[0].loglog(centers,band_rms[str(mu)]['subhalos'],color=color,label=fr'Subhalos, $\mu={mu:g}$')
        ax[0].loglog(centers,band_rms[str(mu)]['stars'],color=color,ls='--',label=fr'Stars, $\mu={mu:g}$')
        rr=radial_response(logq,angles,f,w,(1/mu,1))
        perlogmass=np.trapezoid(mass_delta*rr[:,None],logq,axis=0)
        ax[1].semilogx(mass,perlogmass/np.trapezoid(perlogmass,np.log(mass)),color=color,label=fr'$\mu={mu:g}$')
    ax[0].set(xlabel='Center of a one-decade band [Hz]',ylabel='Projected RMS, one minimum image',title='Frequency dependence is model dependent')
    ax[1].set(xlabel=r'Truncated profile mass [$M_\odot$]',ylabel=r'Fractional variance per $d\ln M$',title='Dominant masses: 10–100 Hz benchmark')
    ax[0].legend(fontsize=8);ax[1].legend()
    savefig(fig,'frequency_and_mass_dependence')

    # Full real joint covariance, retaining pseudo-covariance and nuisance projection.
    detailed=[]
    all_cov={}
    for mu in [1.,10.,30.,100.]:
        cov={}
        for key in ['c_ref_30.0','stars']:
            plus=projected_covariance(f,w,(1/mu,1),logq,spectra[key],angles)
            minus=projected_covariance(f,w,(-1/mu,1),logq,spectra[key],angles)
            cov[key]=plus+minus
        all_cov[mu]=cov
        for rho0 in [20.,50.,100.]:
            rho_each=rho0*np.sqrt(mu)
            rho_eff=effective_snr(rho_each,rho_each)
            dm=covariance_metrics(cov['c_ref_30.0'],w,rho_eff)
            st=covariance_metrics(cov['stars'],w,rho_eff)
            foreground=covariance_foreground_metrics(cov['c_ref_30.0'],cov['stars'],w,rho_eff)
            foreground_scan={str(ks):covariance_foreground_metrics(cov['c_ref_30.0'],cov['stars']*(ks/.025),w,rho_eff)['kl_separation'] for ks in [0.,.00025,.0025,.025]}
            detailed.append({'mu':mu,'rho0':rho0,'rho_image':rho_each,'rho_effective':rho_eff,
                             'dm':dm,'stars':st,'dm_given_stars':foreground,
                             'dm_kl_by_kappa_star':foreground_scan,
                             'star_wave_parameter_times_mu_at_fmax':float(mu*point_mass_wave_parameter(f[-1],.5,geometry.lens_redshift))})
        print(f'covariance completed: mu={mu:g}',flush=True)

    # Plot covariance separation instead of mislabeling a known-template ceiling as significance.
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.1))
    for color,rho0 in zip(colors,[20.,50.,100.]):
        rows=[row for row in detailed if row['rho0']==rho0]
        mm4=[row['mu'] for row in rows]
        ax[0].loglog(mm4,[row['dm']['deterministic_rms_snr'] for row in rows],color=color,marker='o',label=fr'$\rho_0={rho0:g}$, known residual')
        ax[0].loglog(mm4,[row['dm']['kl_separation'] for row in rows],color=color,ls='--',marker='s')
    rows=[row for row in detailed if row['rho0']==50.]
    for color,ks in zip(colors,[0.,.00025,.0025,.025]):
        ax[1].loglog([row['mu'] for row in rows],[row['dm_kl_by_kappa_star'][str(ks)] for row in rows],color=color,marker='o',label=fr'$\kappa_\star={ks:g}$')
    ax[0].set(xlabel=r'Per-image $|\mu|$',ylabel='SNR ceiling / Gaussian separation',title=r'Subhalos alone: dashed = $\sqrt{2D_{KL}}$')
    ax[1].set(xlabel=r'Per-image $|\mu|$',ylabel=r'$\sqrt{2D_{KL}}$ for adding subhalos',title=r'Stellar foreground dependence; $\rho_0=50$')
    ax[0].legend(fontsize=8);ax[1].legend(fontsize=8)
    savefig(fig,'covariance_detectability')

    fig,ax=plt.subplots(1,2,figsize=(10.6,3.8))
    cc=all_cov[30.]['c_ref_30.0']
    evals,evecs=np.linalg.eigh(cc)
    rng=np.random.default_rng(20261121)
    for color in colors[:3]:
        sample=evecs@(np.sqrt(np.maximum(evals,0))*rng.standard_normal(len(evals)))
        ax[0].semilogx(f,100*sample[:len(f)],color=color,lw=1.3)
        ax[1].semilogx(f,100*sample[len(f):],color=color,lw=1.3)
    ax[0].set(xlabel='Observed frequency [Hz]',ylabel='Differential amplitude [%]',title=r'Gaussian subhalo realizations; $|\mu|=30$')
    ax[1].set(xlabel='Observed frequency [Hz]',ylabel='Differential phase [centiradians]',title='Amplitude, phase, delay already fitted')
    savefig(fig,'example_residuals')

    # Weight dependence and numerical resolution diagnostics are explicit outputs.
    weight_checks={}
    for lo,hi in [(3,30),(10,100),(30,300)]:
        for kind in ['log_uniform','inspiral_white_noise']:
            ff,ww=band_grid(lo,hi,weighting=kind)
            rp=radial_response(logq,angles,ff,ww,(1/30,1))
            rm=radial_response(logq,angles,ff,ww,(-1/30,1))
            weight_checks[f'{lo}-{hi}_{kind}']=float(np.sqrt(np.trapezoid(spectra['c_ref_30.0']*(rp+rm),logq)))
    data={'status':'conditional exploratory same-plane Born benchmarks, not a detector/population forecast',
          'geometry':asdict(geometry),'subhalo_model':asdict(model),'stars':{'kappa_star':.025,'effective_mass_msun':.5},
          'frequency_hz':f.tolist(),'weights':w.tolist(),'weighting':'uniform per dln f',
          'reference_hz':100.,'rF_at_100Hz_pc':rf*1000,'sigma_critical_msun_pc2':geometry.sigma_critical()/1e6,
          'numerics':{'logq_min':float(logq[0]),'logq_max':float(logq[-1]),'nq':len(logq),'nangle':len(angles),'nf':len(f)},
          'mu_grid':mus.tolist(),'differential_rms':{key:val.tolist() for key,val in rms.items()},
          'band_centers_hz':centers.tolist(),'band_rms':band_rms,'weight_checks_mu30':weight_checks,'cases':detailed}
    outfile=ROOT/'results'/('summary.json' if r==1 else f'summary_resolution_{r}.json')
    outfile.write_text(json.dumps(data,indent=2)+'\n')
    header='mu,rho0,rho_image,rho_effective,epsilon_dm_diff,known_residual_snr,dm_kl,dm_kl_with_stars,epsilon_star_diff\n'
    lines=[header]
    for row in detailed:
        vals=[row['mu'],row['rho0'],row['rho_image'],row['rho_effective'],row['dm']['fractional_rms'],row['dm']['deterministic_rms_snr'],row['dm']['kl_separation'],row['dm_given_stars']['kl_separation'],row['stars']['fractional_rms']]
        lines.append(','.join(f'{v:.8g}' for v in vals)+'\n')
    (ROOT/'results/benchmark_table.csv').write_text(''.join(lines))
    tex=['% Generated by scripts/run_calculations.py; do not edit.\n',r'\begin{tabular}{rrrrrr}'+'\n',r'\toprule'+'\n',r'$|\mu|$ & $\rho_0$ & $\rho_{\rm image}$ & $\epsilon_{\Delta,{\rm DM}}$ & $\rho_{\rm res,rms}$ & $d_{\rm DM|stars}$ \\'+'\n',r'\midrule'+'\n']
    for row in detailed:
        if row['rho0']==50:
            tex.append(f"{row['mu']:.0f} & 50 & {row['rho_image']:.0f} & {row['dm']['fractional_rms']:.2e} & {row['dm']['deterministic_rms_snr']:.3f} & {row['dm_given_stars']['kl_separation']:.3f} "+r'\\'+'\n')
    tex += [r'\bottomrule'+'\n',r'\end{tabular}'+'\n']
    (ROOT/'results/benchmark_table.tex').write_text(''.join(tex))
    print(json.dumps({'rF100_pc':rf*1000,'weight_checks_mu30':weight_checks,'cases_rho0_50':[ {k:v for k,v in row.items() if k not in ['dm','stars','dm_given_stars']} | {'epsilon_dm_diff':row['dm']['fractional_rms'],'rho_res':row['dm']['deterministic_rms_snr'],'dm_kl':row['dm']['kl_separation'],'dm_kl_stars':row['dm_given_stars']['kl_separation'],'epsilon_star_diff':row['stars']['fractional_rms']} for row in detailed if row['rho0']==50]},indent=2))


if __name__=='__main__': main()
