import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

# python3 -m pip install matplotlib pandas tqdm psutil mplcursors astropy numpy scipy

##############################################
# Display the structure of a FITS file
def op_print_fits_structure(fits_data):
    for hdu in fits_data:
        print(f'-------\nHDU: {hdu.name}, header: {len(hdu.header)}')
        #print(f'Header:\n{hdu.header}')
        if hdu.data is not None:
            if hdu.is_image:
                htype = 'Image'
            else:
                htype = 'Table'
            #print(f'Name: {hdu.hdr['EXTNAME']}')
            if isinstance(hdu.data, np.recarray):
                print(f'{htype} ({hdu.data.shape[0]}): {hdu.data.dtype.names}')
        #print('\n')
    
##############################################
# 
def read_oifits(filename):
    fh = fits.open(filename)
    #op_print_fits_structure(fh)
    #fh.close()
    return fh

##############################################
# 
def read_oifits_sequence(basedir, filelist):
    
    hdus = []
    for ifile, file in enumerate(filelist):
        print('reading file: ', file)
        
        ihdu = read_oifits(basedir + file)
        
        hdus.append(ihdu)
    
    return hdus

##############################################
# enum for data types
# 0: VIS2
# 1: T3PHI
# 2: T3AMP
# 3: VISAMP
# 4: VISPHI
# 5: VISAMPDIFF
# 6: VISPHIDIFF
(VIS2, # Squared visibility
 T3PHI, T3AMP, # Closure phase and amplitude of triple product
 VISAMP, VISPHI, # Amplitude and phase of visibility (absolute)
 VISAMPDIFF, VISPHIDIFF # Amplitude and phase of visibility (differential)
 ) = range(0, 7)

##############################################
# 
# For each key, concatenate the new data to the existing array
def append_to_array(data, key, new_data, firstone):
    if firstone:
        data[key] = new_data.flatten()
    else:
        data[key] = np.ma.concatenate([data[key], new_data.flatten()])
        
##############################################
# 
def flatten_data(hdus, reflag=True):
    """
    Flatten the data from the OIFITS files
    """
    data = {"data": [], "dataerr": [], "dataflag": [],
            "u1coord": [], "v1coord": [],
            "u2coord": [], "v2coord": [],
            "u2coord": [], "v2coord": [],
            "wlen": [], "band": [],
            "time": [], "dtype": []
            }
    
    for ihdu in hdus:
        print('reading file: ', ihdu.filename())
        #op_print_fits_structure(ihdu)
        try:
        # if 1:
            wlen = np.array(ihdu["OI_WAVELENGTH"].data["EFF_WAVE"])
            band = np.array(ihdu["OI_WAVELENGTH"].data["EFF_BAND"])
            nbwlen = len(wlen)
            try:
            # if 1:
                vis2flag = np.array(ihdu["OI_VIS2"].data["FLAG"])
                vis2err  = np.array(ihdu["OI_VIS2"].data["VIS2ERR"])
                
                vis2     = np.ma.array(ihdu["OI_VIS2"].data["VIS2DATA"], mask=vis2flag)
                
                if reflag:
                    # Refine flag, removing useless data
                    vis2flag = ~((vis2err < 0.05) & (vis2 > -0.05) & (vis2 < 1.05))
                    vis2     = np.ma.array(ihdu["OI_VIS2"].data["VIS2DATA"], mask=vis2flag)
                
                ucoord   = np.array(ihdu["OI_VIS2"].data["UCOORD"])
                vcoord   = np.array(ihdu["OI_VIS2"].data["VCOORD"])
                nbbases  = len(ucoord)
                time     = np.array(ihdu["OI_VIS2"].data["TIME"])
                
                tucoord = np.transpose(np.tile(ucoord, (nbwlen,1)))
                tvcoord = np.transpose(np.tile(vcoord, (nbwlen,1)))
                twlen = np.tile(wlen, (nbbases,1))
                tband = np.tile(band, (nbbases,1))
                ttime = np.transpose(np.tile(time, (nbwlen,1)))
                dtype = np.tile(VIS2, (nbbases,nbwlen))

                append_to_array(data, "data",     vis2,     ihdu==hdus[0])
                append_to_array(data, "dataerr",  vis2err,  ihdu==hdus[0])
                append_to_array(data, "dataflag", vis2flag, ihdu==hdus[0])
                append_to_array(data, "u1coord",   tucoord,  ihdu==hdus[0])
                append_to_array(data, "v1coord",   tvcoord,  ihdu==hdus[0])
                append_to_array(data, "u2coord",   np.zeros_like(tucoord),  ihdu==hdus[0])
                append_to_array(data, "v2coord",   np.zeros_like(tvcoord),  ihdu==hdus[0])
                append_to_array(data, "wlen",     twlen,    ihdu==hdus[0])
                append_to_array(data, "band",     tband,    ihdu==hdus[0])
                append_to_array(data, "time",     ttime,    ihdu==hdus[0])
                append_to_array(data, "dtype",    dtype,    ihdu==hdus[0])
            except:
                print('No OI_VIS2 in file: ', ihdu.filename())
                continue
            
            try:
            # if 1:
                t3flag = np.array(ihdu["OI_T3"].data["FLAG"])
                print('t3flag: ', t3flag)
                t3phierr = np.array(ihdu["OI_T3"].data["T3PHIERR"])
                
                if reflag:
                    # Refine flag, removing useless data
                    t3flag = ~(t3phierr < 15)
                    
                print('t3flag new: ', t3flag)
                t3phi = np.ma.array(ihdu["OI_T3"].data["T3PHI"], mask=t3flag)
                append_to_array(data, "dataflag", t3flag, ihdu==hdus[0])
                append_to_array(data, "data",     t3phi,  ihdu==hdus[0])
                append_to_array(data, "dataerr",  t3phierr,  ihdu==hdus[0])
                
                u1coord   = np.array(ihdu["OI_T3"].data["U1COORD"])
                v1coord   = np.array(ihdu["OI_T3"].data["V1COORD"])
                u2coord   = np.array(ihdu["OI_T3"].data["U2COORD"])
                v2coord   = np.array(ihdu["OI_T3"].data["V2COORD"])
                nbbases  = len(u1coord)
                time     = np.array(ihdu["OI_T3"].data["TIME"])
                
                append_to_array(data, "u1coord",   np.transpose(np.tile(u1coord, (nbwlen,1))),  ihdu==hdus[0])
                append_to_array(data, "v1coord",   np.transpose(np.tile(v1coord, (nbwlen,1))),  ihdu==hdus[0])
                append_to_array(data, "u2coord",   np.transpose(np.tile(u2coord, (nbwlen,1))),  ihdu==hdus[0])
                append_to_array(data, "v2coord",   np.transpose(np.tile(v2coord, (nbwlen,1))),  ihdu==hdus[0])
                append_to_array(data, "wlen",     np.tile(wlen, (nbbases,1)),  ihdu==hdus[0])
                append_to_array(data, "band",     np.tile(band, (nbbases,1)),  ihdu==hdus[0])
                append_to_array(data, "time",     np.transpose(np.tile(time, (nbwlen,1))),  ihdu==hdus[0])
                append_to_array(data, "dtype",    np.tile(T3PHI, (nbbases,nbwlen)),  ihdu==hdus[0])

                print("length t3: ", len(t3phi))
                print("length data: ", len(data["data"]))
            except:
                print('No OI_T3 in file: ', ihdu.filename())
                continue
            
            print('wave length',len(data["wlen"]))
            #print(data["wlen"])
        except:
            print('No OI_WAVELENGTH in file: ', ihdu.filename())
            continue
    
    # Convert lists to numpy arrays
    for key in data:
        data[key] = np.ma.array(data[key])
    return data

##############################################
#
def compute_fft(img, zpad=2):
    """
    Compute the FFT of an image with zero padding to the next power of 2, centering the original image.
    """
    def next_power_of_2(x):
        return 1 if x == 0 else 2**(x - 1).bit_length()

    # Determine new shape as next power of 2 of each dimension, multiplied by zpad
    new_shape = (
        next_power_of_2(img.shape[0] * zpad),
        next_power_of_2(img.shape[1] * zpad)
    )
    padded_img = np.zeros(new_shape, dtype=img.dtype)

    # Compute starting indices to center the image
    start_x = (new_shape[0] - img.shape[0]) // 2
    start_y = (new_shape[1] - img.shape[1]) // 2

    # Place the original image in the center
    padded_img[start_x:start_x+img.shape[0], start_y:start_y+img.shape[1]] = img

    fftimg = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(padded_img)))
    freq_x = np.fft.fftshift(np.fft.fftfreq(new_shape[0]))
    freq_y = np.fft.fftshift(np.fft.fftfreq(new_shape[1]))
    return fftimg, freq_x, freq_y

##############################################
#
def interpol_fft(fftimg, freq_x, freq_y, freq_ucoord, freq_vcoord):
    # Prepare the interpolator for the FFT image (complex values)
    interp_real = RegularGridInterpolator((freq_x, freq_y), np.real(fftimg), bounds_error=False, fill_value=0)
    interp_imag = RegularGridInterpolator((freq_x, freq_y), np.imag(fftimg), bounds_error=False, fill_value=0)

    # Prepare coordinates for interpolation
    points = np.column_stack((freq_ucoord.flatten(), freq_vcoord.flatten()))

    # Interpolate real and imaginary parts separately, then combine
    values_real = interp_real(points)
    values_imag = interp_imag(points)
    values = values_real + 1j * values_imag

    # Reshape to match input shapes
    return values.reshape(freq_ucoord.shape)

##############################################
#
def calc_v2(cfdata):
    return np.abs(cfdata)**2

##############################################
#
def calc_phi(cfdata):
    return np.angle(cfdata, deg=True)

##############################################
#
def calc_clos(cfdata):
    u1 = cfdata["u1coord"]
    v1 = cfdata["v1coord"]
    u2 = cfdata["u2coord"]
    v2 = cfdata["v2coord"]
    # Calculate closure phase
    # Closure phase = arg(V12 * V23 * V31)
    
##############################################
#
def plot_model(fftim, data):
    
    ntypes = len(set(data["dtype"]))
    nx = int(np.ceil(np.sqrt(ntypes)*1.5))
    ny = int(np.floor(np.sqrt(ntypes)/1.5))
    while nx * ny < ntypes:
        ny+=1
    print('nx, ny: ', nx, ny)
    fig, ax = plt.subplots(nx, ny, figsize=(12, 8))
    ax = ax.flatten()

    frequ = data["u1coord"] / data["wlen"]
    freqv = data["v1coord"] / data["wlen"]
    radius = np.sqrt(frequ**2 + freqv**2)
    
    iwin = 0;
    # Plot squared visibilities
    wherev2 = np.where(data["dtype"] == VIS2)
    if np.any(wherev2):
        ax[iwin].errorbar(
            radius[wherev2],
            data['data'][wherev2],
            yerr=data['dataerr'][wherev2],
            color='red', linestyle='None', marker='o', markersize=2
        )
        ax[iwin].set_ylim(-0.05, 1.05)
        ax[iwin].set_title('Squared visibilities')
        ax[iwin].set_xlabel('Spatial Frequency')
        ax[iwin].set_ylabel('V2')
        
        iwin +=1;
        ax[iwin].errorbar(
            radius[wherev2],
            data['data'][wherev2],
            yerr=data['dataerr'][wherev2],
            color='red', linestyle='None', marker='o', markersize=2
        )
        ax[iwin].set_ylim(5e-4, 1.2)
        ax[iwin].set_title('Squared visibilities')
        ax[iwin].set_xlabel('Spatial Frequency')
        ax[iwin].set_ylabel('V2 (log scale)')
        ax[iwin].set_yscale('log')
        
        iwin +=1;
    wherecp = np.where(data["dtype"] == T3PHI)
    if np.any(wherecp):
        ax[iwin].errorbar(
            radius[wherecp],
            data['data'][wherecp],
            yerr=data['dataerr'][wherecp],
            color='red', linestyle='None', marker='o', markersize=2
        )
        ax[iwin].set_ylim(-180, 180)
        ax[iwin].set_title('Closure phase')
        ax[iwin].set_xlabel('Spatial Frequency')
        ax[iwin].set_ylabel('Closure phase (degrees)')
        
        iwin +=1;
    wherebsamp = np.where(data["dtype"] == T3AMP)
    if np.any(wherebsamp):
        ax[iwin].errorbar(
            radius[wherebsamp],
            data['data'][wherebsamp],
            yerr=data['dataerr'][wherebsamp],
            color='red', linestyle='None', marker='o', markersize=2
        )
        ax[iwin].set_ylim(-0.2, 1.2)
        ax[iwin].set_title('Closure amplitude')
        ax[iwin].set_xlabel('u (pixels)')
        ax[iwin].set_ylabel('Closure amplitude')
        
        iwin +=1;
    wherevisamp = np.where(data["dtype"] == VISAMP)
    if np.any(wherevisamp):
        ax[iwin].errorbar(
            radius[wherevisamp],
            data['data'][wherevisamp],
            yerr=data['dataerr'][wherevisamp],
            color='red', linestyle='None', marker='o', markersize=2
        )
        ax[iwin].set_ylim(-0.2, 1.2)
        ax[iwin].set_title('Visibility amplitude')
        ax[iwin].set_xlabel('u (pixels)')
        ax[iwin].set_ylabel('Visibility amplitude')
        
        iwin +=1;
    wherevisphi = np.where(data["dtype"] == VISPHI)
    if np.any(wherevisphi):
        ax[iwin].errorbar(
            radius[wherevisphi],
            data['data'][wherevisphi],
            yerr=data['dataerr'][wherevisphi],
            color='blue', linestyle='None', marker='o', markersize=2
        )
        ax[iwin].set_ylim(-0.2, 1.2)
        ax[iwin].set_title('Visibility phase')
        ax[iwin].set_xlabel('u (pixels)')
        ax[iwin].set_ylabel('Visibility phase')
        iwin +=1;
    plt.show()