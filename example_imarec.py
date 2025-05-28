import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from imarec import *



basedir = "/Users/fmillour/Downloads/HR_4049/L_sep/LR/"
filelist = os.listdir(basedir)
filelist = [f for f in filelist if f.endswith('.fits')]
filelist.sort()
print('filelist: ', filelist)
data = read_oifits_sequence(basedir, filelist)
fdata = flatten_data(data)

npix = 32;
img0 = np.zeros((npix, npix))
# Define a uniform disk of diameter xx mas
diam = 10  # example diameter in mas
pix_scale = 1  # mas per pixel, adjust as needed
pix_rad = pix_scale / (3600 * 1000 * 180 / np.pi)
freq_scale = 1 / (pix_rad)  # in cycles/rad

radius_pix = (diam / 2) / pix_scale

y, x = np.indices((npix, npix))
center = (npix // 2, npix // 2)
r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
img0[r <= radius_pix] = 1.0

ft, fx, fy = compute_fft(img0, zpad=1)
frx = freq_scale * fx
fry = freq_scale * fy

frequ = fdata["u1coord"] / fdata["wlen"]
freqv = fdata["v1coord"] / fdata["wlen"]

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
print('ax length:', len(ax))

ax[0][0].imshow(img0, cmap='gray', origin='lower')
ax[0][0].set_title('image')

im0 = ax[1][0].imshow(np.abs(ft), extent=[frx.min(), frx.max(), fry.min(), fry.max()], cmap='gray', origin='lower')
ax[1][0].plot( frequ,  freqv, color='red', linestyle='None', marker='.', markersize=2)
ax[1][0].plot(-frequ, -freqv, color='red', linestyle='None', marker='.', markersize=2)
#fig.colorbar(im0, ax=ax[0])
ax[1][0].set_title('FFT amplitude of image')
ax[1][0].set_xlabel('u (cycles/rad)')
ax[1][0].set_ylabel('v (cycles/rad)')

im0 = ax[1][1].imshow(np.angle(ft), extent=[frx.min(), frx.max(), frx.min(), fry.max()], cmap='gray', origin='lower')
ax[1][1].plot( frequ,  freqv, color='red', linestyle='None', marker='.', markersize=2)
ax[1][1].plot(-frequ, -freqv, color='red', linestyle='None', marker='.', markersize=2)
#fig.colorbar(im0, ax=ax[0])
ax[1][1].set_title('FFT phase of image')
ax[1][1].set_xlabel('u (cycles/rad)')
ax[1][1].set_ylabel('v (cycles/rad)')

plot_model(ft, fdata)

plt.show()