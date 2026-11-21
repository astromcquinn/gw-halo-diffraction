# Literature audit and interpretation boundaries

Inspected 21 November 2026. This audit distinguishes published calculations from the new, conditional calculations in this repository. None of the earlier chat's visual read-offs of the convergence spectrum, event-rate numbers, or suggested high-redshift signal-to-noise ratios is an input to the numerical model.

## 1. Closest prior work: Oguri & Takahashi (2022)

[Amplitude and phase fluctuations of gravitational waves magnified by strong gravitational lensing](https://arxiv.org/pdf/2204.00814), Physical Review D **106**, 043532, DOI [10.1103/PhysRevD.106.043532](https://doi.org/10.1103/PhysRevD.106.043532).

Full version 3 PDF inspected, particularly Eqs. (20)–(21), (34)–(50), Section III, and Figs. 1–4. This is direct prior art for the local macro-Jacobian response derived in the accompanying note; that response must not be presented as a new discovery.

Their comoving-coordinate response, with directional magnifications \(\mu_a=\lambda_a^{-1}\), is

\[
G(\mathbf k,f)=\frac{2i}{r_F^2k^2}
\left[\exp\left(-\frac{i r_F^2}{2}
(\mu_1k_1^2+\mu_2k_2^2)\right)-1\right].
\]

Their Section III.B concludes that stellar microlensing exceeds their CDM subhalo signal at 10 Hz throughout the magnifications examined. At 0.1 Hz, sufficiently magnified images with small stellar fractions can favor the subhalo contribution. This is a model-specific result, not an absolute exclusion of ground-based detection. The calculation is perturbative.

### Published model inputs

| Quantity | Value or definition in Section III.B |
|---|---|
| Cosmology | Flat, \(\Omega_m=0.3\), \(h=0.7\) |
| Host mass | \(10^{13}h^{-1}M_\odot\), overdensity 200 relative to critical |
| Host concentration | 10 |
| Lens/source redshift | 0.3 / 2 |
| Image radius | \(r_{\rm Ein}=0.03r_{200}\) |
| Total convergence | 0.5 |
| Subhalo mass-function slope | \(\alpha=1.9\) |
| Total subhalo mass fraction | \(f_{\rm sub}=0.1\) |
| Upper mass fraction | \(f_{\rm cut}=0.1\) |
| Stellar mass | \(0.5M_\odot\) |
| Local stellar mass fractions | \(f_\star=0.5\) and 0.005 |

Equations (46)–(50) specify

\[
P_\kappa(k)=\int dm\,\frac{dn_{2D}}{dm}
\left(\frac{m}{\Sigma_{\rm cr}}\right)^2|\widetilde u(k|m)|^2,
\qquad
\frac{dn_{2D}}{dm}=\frac{\Sigma_{\rm DM}(r_{\rm Ein})}{M_{\rm host}}
\frac{dN}{dm},
\]

\[
\frac{dN}{dm}=\frac{f_{\rm sub}(2-\alpha)}{f_{\rm cut}^{2-\alpha}M_{\rm host}}
\left(\frac{m}{M_{\rm host}}\right)^{-\alpha},
\quad m\le f_{\rm cut}M_{\rm host},
\qquad
P_{\kappa,\star}=\frac{f_\star M_\star\kappa}{\Sigma_{\rm cr}}.
\]

An explicit subhalo concentration/truncation prescription is not provided in that section. An independent calculation with chosen subhalo profiles should therefore be labeled as such, not an exact reproduction of their figure.

## 2. Choi, Urrutia & Zumalacarregui (2026)

[Signatures of low-mass dark-matter halos in LISA via stochastic diffraction](https://arxiv.org/abs/2607.11887), arXiv:2607.11887. Full v1 PDF inspected through [the PDF mirror](https://pdfs.assets.alphaxiv.org/2607.11887v1.pdf); direct arXiv retrieval was unavailable during this audit.

For \(x=\xi/r_F(f_{\min})\), \(w=f/f_{\min}\), and dimensionless \(k=k_{\rm phys}r_F(f_{\min})\), their Eqs. (3), (4), and (7) are

\[
H(k,w)=\frac{2iw}{k^2}[e^{-ik^2/(2w)}-1],
\quad
\langle\widehat\kappa(\mathbf k)\widehat\kappa^*(\mathbf k')\rangle
=(2\pi)^2\delta^{(2)}(\mathbf k-\mathbf k')P_\kappa(k),
\]

\[
\langle|\Delta F(w)|^2\rangle
=\frac{8w^2}{\pi}\int_0^\infty\frac{dk}{k^3}P_\kappa(k)
\sin^2\left(\frac{k^2}{4w}\right).
\]

Figure 2 is dimensionless convergence power for a source at redshift 2; lens redshift 1 supplies the illustrative physical/frequency-axis mapping. The line-of-sight integral itself covers all lens redshifts. The ground-band calculation is deferred. A mass cutoff at which power saturates does not identify the mass dominating an observable residual. The full LISA analysis removes binary-parameter degeneracies and treats the imprint as excess covariance. The complex field has nonzero pseudo-covariance; independent, identically distributed real and imaginary parts should not be assumed. No paper-specific public code or numerical spectrum table was located in the inspected document.

## 3. Independent audit of the frequency argument

The substitution \(k=\sqrt{w}\,q\) is mathematically correct. If a *fixed-normalization* power spectrum is a pure power law \(P_\kappa(k)\propto k^n\), it produces a formal factor \(w^{1+n/2}\). Whether the remaining integral exists matters:

* At low wavenumber, \(H=1-ik^2/(4w)+O(k^4)\). The raw variance contains geometric magnification and converges in the infrared only for \(n>-2\).
* At high wavenumber, \(|H|=O(w/k^2)\); ultraviolet convergence requires \(n<2\).
* Thus \(n=-2\), the slope implicitly assumed in the earlier chat's claim of equal LISA/LIGO fluctuations, is the **infrared boundary** for the raw variance. The raw integral cannot generally be approximated as a narrow sample at the Fresnel scale.
* In a frequency difference \(H(k,f_1)-H(k,f_2)\), the constant term cancels. The infrared condition becomes \(n>-6\). A projected residual across a fixed fractional bandwidth can therefore have scale-independent variance at \(n=-2\), but that is a conditional scale-free statement.

Consequently, neither an absolute \(10^{-3}\) ground-band residual nor a LIGO/LISA ratio close to one has been established by the earlier chat. This repository should report the spectrum assumptions, frequency weighting, nuisance projection, and normalizations used for every numerical result.

### Direct check of the plotted curve

The vector path in the original v1 PDF was subsequently extracted, after visual inspection of its axes. For the lowest plotted mass cutoff, its values are approximately:

| Equivalent frequency on Fig. 2's upper axis | Dimensionless \(P_\kappa\), as plotted |
|---|---:|
| \(0.01\) Hz | \(5.4\times10^{-14}\) |
| \(100\) Hz | \(2.0\times10^{-19}\) |

These correct the earlier chat's unsupported read-offs. They are **digitized plotted values**, not the original numerical data. The high-wavenumber slope is approximately \(n=-2.71\). Applied conditionally to a projected residual across the same fractional bandwidth, the scale-free argument gives

\[
\frac{\sigma_{100}}{\sigma_{0.01}}
\simeq\left[\frac{100P_\kappa(100)}{0.01P_\kappa(0.01)}\right]^{1/2}
\simeq0.19.
\]

Thus even the slope argument favors roughly **five times smaller amplitude**, rather than equality. This does not determine an absolute residual or compare actual detector/source weightings. The source redshift is fixed; the upper frequency axis is a mapping from transverse wavenumber using reference lens redshift 1.

Reproduce the extraction with `python scripts/digitize_paper_figure2.py`; it requires optional `pymupdf` and downloads the PDF only into a temporary directory. Alternatively pass `--source-pdf /path/to/2607.11887v1.pdf`. The script checks the exact source SHA256, uses the twelve colored vector paths in the upper panel, and selects the topmost right endpoint (dark purple, matching the lowest cutoff on the colorbar). It calibrates independent logarithmic axes against all eight major ticks on each axis. It verifies \(d\log f/d\log k=2\), and obtains slopes \(-2.708\) and \(-2.712\) from adjacent vector segments; the next cutoff curve differs by less than 1% at the two comparison points. The full calibration, source URL/hash, and limitations are recorded in `results/paper_figure2_provenance.json`; samples are in `results/paper_figure2_digitization.csv`. Sub-millidecade calibration residuals quantify extraction consistency only, not the scientific precision of the figure. No source PDF is redistributed.

## 4. Strongly lensed LISA subhalos: Ando (2026)

Two primary sources were inspected:

* [Paper I, arXiv:2603.04267](https://arxiv.org/html/2603.04267v1): local quadratic macro potential, SASHIMI subhalo populations, and wave-optics minimum images.
* [Paper II, arXiv:2606.21519](https://pdfs.assets.alphaxiv.org/2606.21519v1.pdf): saddle calculation and LISA detectability.

Paper I's percent-level modulations concern the LISA band and subhalos around \(10^4\)–\(10^7M_\odot\). Paper II's reported 62% of realizations above its 5-sigma criterion applies to a fiducial massive BBH of total mass around \(10^6M_\odot\), redshift 1.5, and source position close to the caustic. These are not stellar-BBH ground-detector forecasts. Both invoke an explicit local subhalo population; they do not imply that arbitrary cosmological foreground halos receive the same macro enhancement.

The papers link [GLoW](https://github.com/miguelzuma/GLoW_public) and [SASHIMI](https://github.com/shinichiroando/sashimi-c). Paper II's Sazanami footnote says a repository will be released; no repository URL is given there. Availability and version compatibility of those packages were not tested in this audit.

## 5. Stellar diffraction is part of the observable

Zumalacarregui & Shan, [Effective description of lensed gravitational waves diffracted by stellar fields](https://journals.aps.org/prd/pdf/10.1103/d5fx-xgd2), Physical Review D **114**, 063038 (2026), arXiv:2606.17765, was inspected in full text.

It models frequency-dependent distortions from stars and remnants in macro-lensed, ground-band signals, using a basis compressed from simulations. Their demonstration model samples minimum images up to macro magnification 5; its sampling distributions are coverage choices, not astrophysical rates. This supports treating stellar diffraction as a required foreground when interpreting a ground-band residual as evidence for extended dark halos. The repeated-image comparison cancels a common source waveform, but it does not cancel lens-plane stars: the different rays encounter different stellar fields.

## 6. Physical-unit implementation guidance derived for this project

A convenient independent surface-population parameterization is

\[
\frac{dn_{2D}}{dM}=C_M M^{-\alpha},\qquad
C_M=\frac{(2-\alpha)\kappa_{\rm sub}\Sigma_{\rm cr}}
{M_{\max}^{2-\alpha}-M_{\min}^{2-\alpha}}.
\]

This enforces \(\int M(dn_{2D}/dM)dM=\kappa_{\rm sub}\Sigma_{\rm cr}\).
With a finite-mass profile normalized to \(u(0|M)=1\),

\[
P_{\kappa,\rm sub}(q)=\frac{1}{\Sigma_{\rm cr}^2}
\int dM\,\frac{dn_{2D}}{dM}\,M^2|u(q|M)|^2,
\qquad
P_{\kappa,\star}=\frac{\kappa_\star}{\Sigma_{\rm cr}}
\frac{\langle M^2\rangle}{\langle M\rangle}.
\]

Here positions are physical lens-plane distances, \(q\) is their inverse, and \(P_\kappa\) has units of physical area. These equations are useful for a controlled comparison, but specifying \(\kappa_{\rm sub}\), concentration, truncation, and surviving subhalo population is a substantive astrophysical assumption.

To compare with the 2022 paper, \(r_c=(1+z_L)r_p\), \(q_c=q_p/(1+z_L)\), \(\Sigma_{{\rm cr},c}=\Sigma_{{\rm cr},p}/(1+z_L)^2\), and \(P_c(q_c)=(1+z_L)^2P_p(q_p)\). Therefore \(q^2P(q)\) is invariant. Do not mix a comoving spectrum with a physical Fresnel radius.

### Additional implementation trap

The concentration expression S24 in Choi et al. labels its time as lookback time. The originating [Okoli & Afshordi paper](https://arxiv.org/pdf/1510.03868), Eqs. (56)–(57) and its discussion, uses cosmic/virialization time. Literal use of lookback time would diverge at redshift zero. Any implementation of that relation should consult the original definition and record its choice. Their S15 transform is also untruncated NFW, whereas the finite total-mass identity near S14 explicitly assumes virial truncation; normalization and truncation must be made consistent in a finite-mass Poisson implementation.

## Scope of a defensible first numerical release

The response kernel, two-image covariance, nuisance projection, noise penalty, scale-free dependence, and explicitly normalized finite-mass toy population can be computed reproducibly now. They establish **conditional sensitivity**. A detection rate or a precision CDM forecast additionally needs validated subhalo survival/structure, a stellar foreground likelihood, source and lens populations, a chosen detector network, and selection effects. The term “Gen 4” by itself does not specify a noise curve.

## 7. Consistency relations and their domain

Four additional primary papers were inspected for the question of distinguishing lensing from detector or waveform systematics. Their bibliography entries are in `docs/consistency_references.bib`.

| Source | Verified result and location |
|---|---|
| [Inamori & Suyama (2021), arXiv:2107.02443v2](https://arxiv.org/pdf/2107.02443) | Eq. (2.13): the sum of amplitude and phase variances at frequency \(f\) equals the amplitude variance at \(2f\). Section 2.2 states the weak-lensing and statistical assumptions; these are ensemble statistics. |
| [Nakazono & Suyama (2026), arXiv:2604.02216v2](https://arxiv.org/pdf/2604.02216) | Eqs. (3.13)–(3.15) extend variance and mean identities to a fixed strong macro lens plus weak cosmological perturbations. Physical Review D **113**, 123541; DOI [10.1103/6jtr-fsc3](https://doi.org/10.1103/6jtr-fsc3). |
| [Tanaka & Suyama (2023), arXiv:2303.05650v2](https://arxiv.org/pdf/2303.05650) | Eqs. (3.9)–(3.10): subtracted Kramers–Kronig relations for the full causal amplification factor, with its earliest-arrival time convention and \(F(0)=1\). These relate a single realization's real and imaginary parts over all frequencies. |
| [Suyama & Kapadia (2025), arXiv:2506.02430v1](https://arxiv.org/pdf/2506.02430) | Eq. (2.9): the magnitude–phase relation includes a Blaschke contribution from upper-half-plane zeros. Phase is not generally determined uniquely by magnitude. |

### Exact strong-lens ensemble identities

Nakazono & Suyama write \(F_j/F_{j0}=\exp[K_j+iS_j^{\rm tot}]\) and subtract the high-frequency term linear in frequency from \(S_j^{\rm tot}\) to obtain \(S_j\). Their average \(\langle\cdot\rangle_W\) samples the weak potential while holding the macro lens fixed. Through second order in that potential, with their Limber approximation,

\[
\begin{split}
\langle K_j(f)^2\rangle_W+\langle S_j(f)^2\rangle_W
 &=\langle K_j(2f)^2\rangle_W,\\
\langle S_j(f)\rangle_W-\tfrac12\langle S_j(2f)\rangle_W
 &=-\langle K_j(f)S_j(f)\rangle_W,\\
\langle K_j(f)\rangle_W&=-\langle K_j(f)^2\rangle_W.
\end{split}
\]

At this order the second moments equal variances because the first-order means vanish. Their Eq. (3.28), \(\langle e^{2K_j}\rangle_W=1\), is an all-orders energy-conservation statement within their geometric-macro/Limber treatment. The current repository implements first-order field fluctuations and their covariance. It can check the first identity; testing the nonzero mean identities requires a consistent second-order propagation calculation.

### Implications for this project's measured residual

The following are consequences for our analysis, not additional claims of those papers. A single event need not satisfy the ensemble variance identity point by point. An unprojected difference of independent images inherits the variance identity by addition, while correlated images require their cross terms. A bandwise nuisance projection changes the covariance, so the measured residual should be compared with the projected forward model, not forced to obey the unprojected scalar identity. Selection and noise bias also require explicit modeling before an empirical variance can test it.

Kramers–Kronig consistency is a different, deterministic causality constraint. Finite bandwidth, fitted delays, division by another image's response, and isolation of a local saddle contribution require their own analyticity and normalization treatment; the full-transfer relation cannot simply be imposed on an arbitrary projected image difference. A causal detector response can satisfy a Kramers–Kronig relation too. Passing it is therefore useful consistency evidence, but does not uniquely identify lensing or separate stars from dark halos. The detector-network and repeated-image likelihood must supply that discrimination.
