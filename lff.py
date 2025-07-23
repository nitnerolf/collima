#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Low frequency filling for optical/long baseine interferometry
# Author: fmillour
# Date: 08/03/2025
# inspired from LFF module of fitomatic 
# necessary packages: pip install numpy scipy astropy matplotlib
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.optimize import curve_fit
from astropy.io import fits
from datetime import datetime
from os.path import expanduser

rad2as = 206265

articlesdir = expanduser('~/driveFlorentin/ARTICLES/AMOI/')

dir = articlesdir + '/NF_HD62623_3/round3_l_Pup/LFF2/'
filename = ['l_Pup_2020_Lband_MATISSE_IR-LM_LOW_noChop_cal_merged_oifits_0.fits',
            'l_Pup_2020_Mband_MATISSE_IR-LM_LOW_noChop_cal_merged_oifits_0.fits',
            'l_Pup_2020_Nband_MATISSE_IR-N_LOW_noChop_cal_merged_oifits_0.fits',
            'l_Pup_2018_Lband_MATISSE_IR-LM_LOW_noChop_cal_merged_oifits_0.fits',
            'l_Pup_2018_Nband_MATISSE_IR-N_LOW_noChop_cal_merged_oifits_0.fits'
            ]

######################################################
# LFF parameters
v2thresh    = 0.3**2 # Threshold for visibility squared to take into account for the LFF fit
freqthresh  = 0 # Threshold for frequency to take into account for the LFF fit
csym        = 1 # Fit a 1D Gaussian (csym=1) or a 2D Gaussian (csym=2)
num_points  = 40 # Number of (u,v) points generated in the LFF file
fracMinFreq = 0.7 # Fraction of the minimum frequency to generate the (u,v) points
#uvtype      = 'spiral'
#uvtype      = 'rspiral'
#uvtype      = 'random'
uvtype      = 'random2'
rdamp       = 0.1
nturns      = np.sqrt(num_points)

######################################################
# Load data from fits file
def load_fits_data(dir, filename):
    hdu = fits.open(dir + filename)
    binary_table_names = np.array([hdi.name for hdi in hdu if isinstance(hdi, fits.BinTableHDU)])
    return hdu, binary_table_names

######################################################
# Extract OI_VIS2 data from fits file
def extract_vis2_data(hdu, binary_table_names):
    OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid = (np.array([]) for _ in range(9))
    count = 0
    for ibin, bnam in enumerate(binary_table_names):
        if bnam == "OI_VIS2":
            oivis2 = hdu[ibin+1]
            vis2, vis2e = oivis2.data['VIS2DATA'], oivis2.data['VIS2ERR']
            u, v = hdu[ibin+1].data['UCOORD'], hdu[ibin+1].data['VCOORD']
            instru = hdu[ibin+1].header['INSNAME']
            Wavelength, Bandwidth = find_wavelength_table(hdu, binary_table_names, instru)
            dist_sign = np.sign(np.mean(np.diff(Wavelength)))
            b = np.sqrt(u**2 + v**2)
            freq = b[:,None]/Wavelength[None,:] / rad2as
            for i in range(vis2.shape[0]):
                count += 1
                wlen, band = np.append(wlen, Wavelength), np.append(band, Bandwidth)
                OIVIS2, OIVIS2e = np.append(OIVIS2, vis2[i,:]), np.append(OIVIS2e, vis2e[i,:])
                FREQ, U, V, B, Bid = (np.append(arr, np.ones(Wavelength.shape[0])*val) for arr, val in zip([FREQ, U, V, B, Bid], [freq[i,:], u[i], v[i], b[i], count]))
    print('Total number of baselines',count)
    #print('bid', Bid)
    return OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid, count, dist_sign

######################################################
# Find OI_WAVELENGTH table for instrument
def find_wavelength_table(hdu, binary_table_names, instru):
    for wbin, wnam in enumerate(binary_table_names):
        oiwlen = hdu[wbin+1]
        if wnam == "OI_WAVELENGTH" and oiwlen.header['INSNAME'] == instru:
            return hdu[wbin+1].data['EFF_WAVE'], hdu[wbin+1].data['EFF_BAND']
    raise ValueError(f"OI_WAVELENGTH table not found for instrument {instru}")

######################################################

def filter_visibilities(OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid, v2thresh, freqthresh):
    if freqthresh is None:
        freqthresh = max(FREQ)
    flag = (OIVIS2 - OIVIS2e > v2thresh) & (B < 5 * min(B)) & (FREQ < freqthresh)
    # Enforce the full baseline wavelength range to be true
    uniqueId = np.unique(Bid)
    for ib in uniqueId:
        if not np.all((OIVIS2[Bid==ib] - OIVIS2e[Bid==ib] > v2thresh)) or not np.all(B[Bid==ib] < 5 * min(B)) or not np.all(FREQ[Bid==ib] < freqthresh):
            flag[Bid == ib] = False
    return (arr[flag] for arr in [OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band])

######################################################

def gaussian_centered(x, size, type='fwhm', a=1, x0=0):
    if type == 'sigma':
        sigma = size
    elif type == 'fwhm':
        sigma = size / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    else:
        raise ValueError("Type must be 'sigma' or 'fwhm'")
    return a * np.exp(-(x - x0)**2 / (2 * (sigma+(sigma==0))**2))

######################################################

def UD_centered(x, diam, V=1, x0=0):
    # Airy pattern squared (normalized), for a uniform disk:
    # V(x) = 2 * J1(pi * diam * x) / (pi * diam * x)
    # Here, x is spatial frequency (cycles/arcsec), diam in arcsec
    # The squared modulus is the visibility squared
    arg = np.pi * diam * (x - x0)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        airy = 2 * j1(arg) / (arg + (arg == 0))
        airy[arg == 0] = 1.0  # lim x->0 of 2*J1(x)/x = 1
    return V**2 * (airy ** 2)

######################################################

def gaussSizeFromV2(v2, freq):
    sig = np.sqrt(-0.5 * freq**2 / np.log(v2))
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sig
    return fwhm

######################################################

def UD_sizeFromV2(v2, freq):
    """
    Calculate the uniform disk size from visibility squared.
    :param v2: Visibility squared
    :param freq: Frequency
    :return: Uniform disk size
    """
    if np.any(v2 <= 0):
        raise ValueError("Visibility squared must be positive.")
    return 2 * np.pi * np.sqrt(-np.log(v2)) / freq


######################################################

def gaussian_2d(xy, sigma_x, sigma_y, theta):
    x, y = xy
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return np.exp(-((x_rot)**2 / (2 * sigma_x**2) + (y_rot)**2 / (2 * sigma_y**2)))


######################################################

def main():
    for ifile in filename:
        global freqthresh
        hdu, binary_table_names = load_fits_data(dir, ifile)
        OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid, count, dist_sign = extract_vis2_data(hdu, binary_table_names)
        print(f'Data file contains {count} visibilities')
        fig, axs = plt.subplots(2, 2, figsize=(18, 7))
        axs = axs.flatten()

        # Left plot: gaussSizeFromV2
        axs[1].plot(FREQ, gaussSizeFromV2(OIVIS2, FREQ), color="red", marker='*', linestyle='')
        axs[1].set_title('Gaussian Size from V2')
        axs[1].set_xlabel('Frequency')
        axs[1].set_ylabel('FWHM')

        # Right plot: plot_vis2_data
        for ibase in range(OIVIS2.shape[0]):
            axs[0].errorbar(FREQ[ibase], OIVIS2[ibase], yerr=OIVIS2e[ibase], color="black")
        axs[0].set_title('V2 Data')
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('V2')

        # Right plot: plot_vis2_data
        for ibase in range(OIVIS2.shape[0]):
            axs[2].errorbar(FREQ[ibase], OIVIS2[ibase], yerr=OIVIS2e[ibase], color="black")
        axs[2].set_title('V2 Data')
        axs[2].set_xlabel('Frequency')
        axs[2].set_ylabel('V2')

        plt.tight_layout()
        
        if freqthresh == 0:
            freqthresh = max(FREQ)

        Vis2fl, Vis2fle, freqfl, Ufl, Vfl, Bfl, wlenfl, bandfl = filter_visibilities(OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid, v2thresh, freqthresh)
        unique_wavelengths, index = np.unique(wlenfl, return_index=True)
        unique_band = bandfl[index]
        freqfl_sorted, Vis2fl_sorted, Vis2fle_sorted = freqfl, Vis2fl, Vis2fle
        
        print(f'Data file contains {len(unique_wavelengths)} lambda')

        
        axs[0].axhline(y=v2thresh, color="red", linestyle=':')
        axs[0].axvline(x=freqthresh, color="black", linestyle=':')
        axs[2].axhline(y=v2thresh, color="red", linestyle=':')
        axs[2].axvline(x=freqthresh, color="black", linestyle=':')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_wavelengths)))
        for i, wl in enumerate(unique_wavelengths):
            mask = wlenfl == wl
            axs[0].errorbar(freqfl_sorted[mask], Vis2fl_sorted[mask], yerr=Vis2fle_sorted[mask], color=colors[i], label=f'wl={wl:.2e}', linestyle='')
            axs[2].errorbar(freqfl_sorted[mask], Vis2fl_sorted[mask], yerr=Vis2fle_sorted[mask], color=colors[i], label=f'wl={wl:.2e}', linestyle='')
        plt.legend()
        
        plt.xlabel(r'$B / \lambda$ (cycles/arcsec)')
        plt.ylabel(r'$V^2$')
        plt.title(r'$V^2$ vs $B / \lambda$')
        axs[2].set_yscale('log')
        axs[0].set_xlim(0, 3*max(freqfl_sorted))
        axs[0].set_ylim(-0.1, 1.1)
        axs[2].set_xlim(0, 3*max(freqfl_sorted))
        axs[2].set_ylim(v2thresh/10, 1.1)

        if csym == 1:
            size_fits = []
            for wl in unique_wavelengths:
                mask = wlenfl == wl
                freq_wl, vis2_wl, vis2e_wl = freqfl_sorted[mask], Vis2fl_sorted[mask], Vis2fle_sorted[mask]
                popt, _ = curve_fit(gaussian_centered, freq_wl, vis2_wl, sigma=vis2e_wl, p0=[np.std(freq_wl)])
                if popt[0] < 0.1:
                    print(f"Warning: Negative fwhm fit for wavelength {wl:.2e}, setting to previous wavelength value.")
                    popt[0] = size_fits[-1]
                size_fits.append(np.abs(popt[0]))
                x_fit = np.linspace(0, max(freq_wl), 1000)
                y_fit = gaussian_centered(x_fit, *popt)
                axs[0].plot(x_fit, y_fit, linestyle='--', label=f'Fit for wl={wl:.2e}', color=colors[np.where(unique_wavelengths == wl)[0][0]])
                axs[2].plot(x_fit, y_fit, linestyle='--', label=f'Fit for wl={wl:.2e}', color=colors[np.where(unique_wavelengths == wl)[0][0]])
                
            axs[3].plot(unique_wavelengths*1e6, np.array(size_fits) * 2.35, marker='o', linestyle='-', color='blue')

        elif csym == 2:
            # 2D Gaussian fit
            sigma_x_fits = []
            sigma_y_fits = []
            theta_fits   = []
            reduced_chi2s_2d = []

            for wl in unique_wavelengths:
                mask = wlen == wl
                xy = np.vstack((U[mask] / wl, V[mask] / wl)) / rad2as
                vis2_wl  = OIVIS2[mask]
                vis2e_wl = OIVIS2e[mask]

                initial_guess = [np.std(U[mask] / wl / rad2as), np.std(V[mask] / wl / rad2as), 0.1]
                popt2, pcov2  = curve_fit(gaussian_2d, xy, vis2_wl, sigma=vis2e_wl, p0=initial_guess)
                sigma_x_fits.append(popt2[0])
                sigma_y_fits.append(popt2[1])
                theta_fits.append(popt2[2])

                # Calculate reduced chi-squared for 2D fit
                residuals_2d = vis2_wl - gaussian_2d(xy, *popt2)
                chi2_2d = np.sum((residuals_2d / vis2e_wl) ** 2)
                dof_2d = len(vis2_wl) - len(popt2)  # degrees of freedom
                reduced_chi2_2d = chi2_2d / dof_2d
                reduced_chi2s_2d.append(reduced_chi2_2d)

            print(f"Fit parameters for 2D Gaussian (sigma_x, sigma_y, theta) for each wavelength: {list(zip(sigma_x_fits, sigma_y_fits, theta_fits))}")
            print(f"Reduced chi-squared for 2D fit for each wavelength: {reduced_chi2s_2d}")


        # Generate new visibilities from the fit parameters
        max_radius = min(freqfl_sorted) * rad2as * min(unique_wavelengths) * fracMinFreq
        
        if uvtype=='random':
            U_new = []
            V_new = []
            while len(U_new) < num_points:
                r = max_radius * np.random.uniform(0, 1)
                theta = np.random.uniform(0, np.pi)
                u = r * np.cos(theta)
                v = r * np.sin(theta)
                U_new.append(u)
                V_new.append(v)
            U_new = np.array(U_new)
            V_new = np.array(V_new)
        if uvtype=='random2':
            U_new = np.random.uniform(-max_radius, max_radius, num_points)
            V_new = np.random.uniform(-max_radius, max_radius, num_points)
            while np.any(np.sqrt(U_new**2 + V_new**2) > max_radius):
                test = np.sqrt(U_new**2 + V_new**2) > max_radius
                numi = np.sum(test)
                U_new[test] = np.random.uniform(-max_radius, max_radius, numi)
                V_new[test] = np.random.uniform(-max_radius, max_radius, numi)
        if uvtype=='spiral':
            r = np.linspace(0.1, max_radius, num_points)
            theta = np.linspace(0, nturns * 2*np.pi, num_points)
            U_new = r * np.cos(theta)
            V_new = r * np.sin(theta)
        if uvtype=='rspiral':
            r = np.linspace(0.1, max_radius, num_points) + np.random.normal(scale=rdamp * max_radius / num_points * 2 * np.sqrt(2*np.log(2)), size=num_points)
            theta = np.linspace(0, nturns * 2*np.pi, num_points) + np.random.normal(scale=rdamp * nturns * 2*np.pi / num_points * 2 * np.sqrt(2*np.log(2)), size=num_points)
            U_new = r * np.cos(theta)
            V_new = r * np.sin(theta)

        wlen_new = unique_wavelengths[::int(dist_sign)]
        band_new = unique_band[::int(dist_sign)]

        new_visibilities = []
        new_B = []
        for iwl, wl in enumerate(wlen_new):
            B_new = np.sqrt(U_new**2 + V_new**2)
            new_B.append(B_new)
            freq = B_new / wl / rad2as
            if csym == 1:
                new_vis = gaussian_centered(freq, size_fits[iwl])
            else:
                xy = np.vstack((U_new, V_new)) / wl / rad2as
                new_vis = gaussian_2d(xy, sigma_x_fits[iwl], sigma_y_fits[iwl], theta_fits[iwl])
                
            new_visibilities.append(new_vis)
        new_visibilities = np.array(new_visibilities).T
        new_B = np.array(new_B).T

        new_vis2err = np.median(Vis2fle)/np.median(Vis2fl) * new_visibilities * new_B / np.max(new_B)

        for ibase in range(new_visibilities.shape[0]):
            new_freq = new_B[ibase,:] / wlen_new / rad2as
            axs[0].plot(new_freq, new_visibilities[ibase,:], color="black")
            axs[0].errorbar(new_freq, new_visibilities[ibase,:], yerr=new_vis2err[ibase,:], color="black")
            axs[2].plot(new_freq, new_visibilities[ibase,:], color="black")
            axs[2].errorbar(new_freq, new_visibilities[ibase,:], yerr=new_vis2err[ibase,:], color="black")

        nwl = wlen_new.shape[0]
        new_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='UCOORD',    format='D',       array=U_new),
            fits.Column(name='VCOORD',    format='D',       array=V_new),
            fits.Column(name='VIS2DATA',  format=f'{nwl}D', array=new_visibilities),
            fits.Column(name='FLAG',      format=f'{nwl}L', array=np.zeros(new_visibilities.shape, dtype=bool)),
            fits.Column(name='INT_TIME',  format='D',       array=np.zeros(U_new.shape)),
            fits.Column(name='MJD',       format='D',       array=hdu['OI_VIS2'].data['MJD'][0]*np.ones(U_new.shape)),
            fits.Column(name='STA_INDEX', format='2I',      array=np.zeros(U_new.shape, dtype=int)),
            fits.Column(name='TARGET_ID', format='I',       array=np.ones(U_new.shape, dtype=int)),
            fits.Column(name='TIME',      format='D',       array=np.zeros(U_new.shape)),
            fits.Column(name='VIS2ERR',   format=f'{nwl}D', array=new_vis2err)
        ])
        new_hdu.header['EXTNAME']  = 'OI_VIS2'
        new_hdu.header['INSNAME']  = hdu['OI_VIS2'].header['INSNAME']
        new_hdu.header['OI_REVN']  = 1
        new_hdu.header['DATE-OBS'] = datetime.now().strftime('%Y-%m-%d')
        new_hdu.data['STA_INDEX'][:,1] = 1

        new_hdul = fits.HDUList([fits.PrimaryHDU(), new_hdu])

        wavelength_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='EFF_WAVE', format='E', array=wlen_new),
            fits.Column(name='EFF_BAND', format='E', array=band_new)
        ])
        wavelength_hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
        wavelength_hdu.header['INSNAME'] = hdu['OI_WAVELENGTH'].header['INSNAME']
        wavelength_hdu.header['OI_REVN'] = 1

        new_hdul.append(wavelength_hdu)
        
        hdu.append(wavelength_hdu)
        hdu.append(new_hdu)

        for hdi in hdu:
            if 'EXTNAME' in hdi.header and hdi.header['EXTNAME'] == 'OI_ARRAY':
                new_hdul.append(hdi)
                new_hdul[-1].header['OI_REVN'] = 1
        for hdi in hdu:
            if 'EXTNAME' in hdi.header and hdi.header['EXTNAME'] == 'OI_TARGET':
                new_hdul.append(hdi)
                new_hdul[-1].header['OI_REVN'] = 1
                break

        if csym == 1:
            new_filename    = ifile.replace('.fits', '_new_v2_fit1D_LFF.fits')
            new_filename2   = ifile.replace('.fits', '_new_v2_fit1D_withLFF.fits')
            output_filename = ifile.replace('.fits', '_v2_fit1D.png')
        else:
            new_filename    = ifile.replace('.fits', '_new_v2_fit2D_LFF.fits')
            new_filename2   = ifile.replace('.fits', '_new_v2_fit2D_withLFF.fits')
            output_filename = ifile.replace('.fits', '_v2_fit2D.png')
            
        new_hdul.writeto(dir + new_filename, overwrite=True)

        plt.savefig(dir + output_filename)
        plt.show()

if __name__ == "__main__":
    main()
