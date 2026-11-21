"""Physical lens-plane units and explicitly parameterized shot-noise spectra.

These are controlled benchmark models, not a calibrated subhalo prediction.
All radii and wavenumbers are physical (kpc, inverse kpc), masses are solar.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.special import sici

C_KMS = 299792.458
KPC_M = 3.085677581491367e19
G_KPC = 4.30091727003628e-6  # kpc (km/s)^2 / Msun


@dataclass(frozen=True)
class Geometry:
    """Flat matter + Lambda cosmology; radiation neglected at the example redshifts."""
    lens_redshift: float = 0.5
    source_redshift: float = 2.0
    hubble_km_s_mpc: float = 70.0
    omega_matter: float = 0.3

    def __post_init__(self):
        if not 0 < self.lens_redshift < self.source_redshift:
            raise ValueError("Require 0 < z_lens < z_source")
        if not self.hubble_km_s_mpc > 0 or not 0 < self.omega_matter < 1:
            raise ValueError("Invalid flat-Lambda cosmology")

    def expansion(self, z):
        return np.sqrt(self.omega_matter * (1 + np.asarray(z))**3 + 1 - self.omega_matter)

    def comoving_kpc(self, z):
        return C_KMS / self.hubble_km_s_mpc * 1000 * quad(
            lambda zz: 1 / self.expansion(zz), 0, z, epsabs=1e-10)[0]

    def distances_kpc(self):
        cl, cs = self.comoving_kpc(self.lens_redshift), self.comoving_kpc(self.source_redshift)
        return cl/(1+self.lens_redshift), cs/(1+self.source_redshift), (cs-cl)/(1+self.source_redshift)

    def sigma_critical(self):
        """Critical surface density in Msun/kpc^2, with physical angular distances."""
        dl, ds, dls = self.distances_kpc()
        return C_KMS**2 / (4*np.pi*G_KPC) * ds/(dl*dls)

    def fresnel_kpc(self, frequency_hz):
        """r_F = sqrt[c D_l D_ls / (2 pi f (1+z_l) D_s)]."""
        frequency_hz = np.asarray(frequency_hz, dtype=float)
        if np.any(frequency_hz <= 0):
            raise ValueError("Frequency must be positive")
        dl, ds, dls = self.distances_kpc()
        c_kpc_s = C_KMS * 1000/KPC_M
        return np.sqrt(c_kpc_s * dl*dls/ds / (2*np.pi*frequency_hz*(1+self.lens_redshift)))

    def rho_critical(self):
        h_kpc = self.hubble_km_s_mpc/1000*self.expansion(self.lens_redshift)
        return 3*h_kpc**2/(8*np.pi*G_KPC)


def nfw_transform(k_rs, concentration):
    """Normalized 3-D NFW transform truncated at r200; equals its 2-D projection FT.

    u(0)=1; k_rs=k_phys*r_s. The high-k endpoint expansion avoids catastrophic
    cancellation of sine integrals. Concentration here is r200/r_s.
    """
    x, c = np.broadcast_arrays(np.asarray(k_rs, float), np.asarray(concentration, float))
    if np.any(x < 0) or np.any(c <= 0):
        raise ValueError("Require k_rs >= 0 and concentration > 0")
    norm = np.log1p(c)-c/(1+c)
    out = np.ones_like(x)
    nonzero = x > 0
    middle = nonzero & (x <= 1e4)
    xm, cm, nm = x[middle], c[middle], norm[middle]
    si1, ci1 = sici(xm)
    si2, ci2 = sici((1+cm)*xm)
    out[middle] = (np.sin(xm)*(si2-si1) + np.cos(xm)*(ci2-ci1)
                   - np.sin(cm*xm)/((1+cm)*xm))/nm
    high = x > 1e4
    xh, ch = x[high], c[high]
    out[high] = (1-np.cos(ch*xh)/(1+ch)**2)/(xh**2*norm[high])
    return out


@dataclass(frozen=True)
class SubhaloModel:
    """User-controlled local surface population, normalized by kappa_sub.

    dN/(dA dM) proportional M^-alpha; M is the finite profile mass.
    c(M)=min(c_max,c_ref*(M/mass_ref)^c_slope), r200 from 200 rho_critical(z_l).
    No tidal evolution, depletion, halo correlations, or line-of-sight integral.
    """
    kappa_sub: float = 0.005
    alpha: float = 1.9
    min_mass: float = 1e-6
    max_mass: float = 1e8
    concentration_ref: float = 30.0
    mass_ref: float = 1e6
    concentration_slope: float = -0.1
    concentration_max: float = 200.0
    mass_bins: int = 420

    def mass_grid(self):
        if self.min_mass <= 0 or self.max_mass <= self.min_mass or self.kappa_sub < 0:
            raise ValueError("Invalid subhalo mass interval or surface density")
        return np.geomspace(self.min_mass, self.max_mass, self.mass_bins)

    def concentration(self, mass):
        return np.minimum(self.concentration_max, self.concentration_ref*(np.asarray(mass)/self.mass_ref)**self.concentration_slope)

    def power(self, physical_wavenumber, geometry, return_mass_integrand=False):
        """Return P_kappa in kpc^2; optional integrand per dln M (Nk,Nm)."""
        k = np.atleast_1d(np.asarray(physical_wavenumber, float))
        if np.any(k < 0):
            raise ValueError("Negative wavenumber")
        mass = self.mass_grid()
        logmass = np.log(mass)
        sc = geometry.sigma_critical()
        shape = (mass/self.mass_ref)**(1-self.alpha)
        number_per_logmass = shape * self.kappa_sub * sc / np.trapezoid(mass*shape,logmass)
        c = self.concentration(mass)
        r200 = (3*mass/(4*np.pi*200*geometry.rho_critical()))**(1/3)
        rs = r200/c
        u = nfw_transform(k[:,None]*rs[None,:],c[None,:])
        integrand = number_per_logmass[None,:]*(mass[None,:]/sc)**2*u**2
        power = np.trapezoid(integrand, logmass, axis=1)
        if return_mass_integrand:
            return power, mass, integrand
        return power


def stellar_power(geometry, kappa_star=0.025, effective_mass=0.5):
    """Poisson point masses: P_kappa = kappa_star <M^2>/<M> / Sigma_crit.

    Weak-scattering benchmark only; not a nonlinear stellar microlensing map.
    """
    if kappa_star < 0 or effective_mass <= 0:
        raise ValueError("Invalid stellar normalization")
    return kappa_star*effective_mass/geometry.sigma_critical()


def delta2_from_physical_power(q, power, fresnel_reference_kpc):
    """Dimensionless power per dln k = k_phys^2 P_phys/(2 pi)."""
    k = np.asarray(q)/fresnel_reference_kpc
    return k*k*np.asarray(power)/(2*np.pi)


def point_mass_wave_parameter(frequency_hz,mass_solar,lens_redshift):
    """w_M = 8 pi G M (1+z_l) f/c^3; this is NOT a diffuse halo mass rule."""
    gm_sun_over_c3_s = 4.9254909476412675e-6
    return 8*np.pi*gm_sun_over_c3_s*np.asarray(frequency_hz)*mass_solar*(1+lens_redshift)
