import numpy as np
import numpy.ma as ma
from astropy.io import fits
import matplotlib.pyplot as plt

sp1f = '/Users/fmillour/Documents/ARTICLES/GPAO+/spectrum_planet_nochi2_MACAO.fits'
sp2f = '/Users/fmillour/Documents/ARTICLES/GPAO+/spectrum_planet_nochi2_GPAO.fits'

sp1 = fits.getdata(sp1f)
wlen1    = sp1[0,:]
spec1    = sp1[1,:]
spec1err = sp1[2,:]
mask1    = spec1 / spec1err < 5
spc1     = ma.masked_array(spec1, mask1)
spc1err  = ma.masked_array(spec1err, mask1)

# Conversion Jy to W/m2/micron
Jy = 1e-26 # W/m2/Hz
C  = 3e8 # m/s
wlen1w   = wlen1 * 1e-6 # microns
spc1w    = spc1    * Jy * C * 1e6 / (wlen1**2)
spc1errw = spc1err * Jy * C * 1e6 / (wlen1**2)

sp2 = fits.getdata(sp2f)
wlen2    = sp2[0,:]
spec2    = sp2[1,:]
spec2err = sp2[2,:]
mask2    = spec2 / spec2err < 5
spc2     = ma.masked_array(spec2, mask2) 
spc2err  = ma.masked_array(spec2err, mask2)

wlen2w   = wlen2 * 1e-6 # microns
spc2w    = spc2    * Jy * C * 1e6 / wlen2**2
spc2errw = spc2err * Jy * C * 1e6 / wlen2**2

plt.figure(figsize=(8/1.5, 4/1.5))
#plt.errorbar(wlen1, spc1, yerr=spc1err, fmt='o', label='MACAO', color='pink')
#plt.errorbar(wlen2, spc2, yerr=spc2err, fmt='o', label='GPAO', color='purple')
plt.plot(wlen1, spc1*1000*3/4, label='MACAO', color='pink')
plt.plot(wlen2, spc2*1000, label='GPAO', color='purple')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Flux (Jy)')
plt.title(r'$\beta$ Pic b')
plt.ylim(0, 10)
plt.legend()
plt.tight_layout()
plt.savefig('betaPic_b_MATISSE_Jy.png',dpi=300)

plt.figure(figsize=(8/1.5, 4/1.5))
#plt.errorbar(wlen1, spc1, yerr=spc1err, fmt='o', label='MACAO', color='pink')
#plt.errorbar(wlen2, spc2, yerr=spc2err, fmt='o', label='GPAO', color='purple')
plt.plot(wlen1, spc1w*3/4, label='MACAO', color='pink')
plt.plot(wlen2, spc2w, label='GPAO', color='purple')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.title(r'$\beta$ Pic b')
plt.ylim(0, 3e-15)
plt.legend()
plt.tight_layout()
plt.savefig('betaPic_b_MATISSE.png',dpi=300)
plt.show()

