import unittest
import numpy as np
from scipy.integrate import quad
from gw_diffraction.physical import Geometry, SubhaloModel, nfw_transform, stellar_power


class PhysicalTests(unittest.TestCase):
    def test_fresnel_scaling(self):
        g=Geometry()
        self.assertAlmostEqual(float(g.fresnel_kpc(10)/g.fresnel_kpc(1000)),10)
        self.assertGreater(g.sigma_critical(),1e8)
        self.assertLess(g.sigma_critical(),1e10)

    def test_nfw_transform_against_radial_integral(self):
        for c in [5,30,200]:
            norm=np.log1p(c)-c/(1+c)
            self.assertAlmostEqual(float(nfw_transform(0,c)),1)
            for x in [0.0001,0.1,1.0]:
                direct=quad(lambda r:r/(1+r)**2*np.sinc(x*r/np.pi),0,c,epsabs=1e-10,limit=500)[0]/norm
                self.assertAlmostEqual(float(nfw_transform(x,c)),direct,places=9)

    def test_mass_normalization_and_poisson_limit(self):
        g=Geometry()
        m=SubhaloModel()
        power,mass,integrand=m.power([0.0],g,True)
        shape=(mass/m.mass_ref)**(1-m.alpha)
        expected=m.kappa_sub/g.sigma_critical()*np.trapezoid(mass**2*shape,np.log(mass))/np.trapezoid(mass*shape,np.log(mass))
        self.assertAlmostEqual(power[0]/expected,1,places=12)
        self.assertGreater(stellar_power(g),0)


if __name__=='__main__':
    unittest.main()
