# Plot images from SPARCO
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve
from scipy.ndimage import shift as nd_shift
import os
from matplotlib.patches import Ellipse

dir0 = '/Users/fmillour/Library/CloudStorage/SynologyDrive-cloud/driveFlorentin/ARTICLES/AMOI/HD62623_3/Re_LFF_l_Pup/'

#dir = '/Users/fmillour/Documents/ARTICLES/HD62623_3/Re_LFF_l_Pup/SPARCO_wlen/' 
#dir = '/Users/fmillour/Documents/ARTICLES/HD62623_3/Re_LFF_l_Pup/MIRA_wlen/'
dir = dir0 + 'MIRA_broadband/'
dir = dir0 + 'SPARCO_broadband/'
dir = dir0 + 'MIRA_broadband_LFF/'

files = os.listdir(dir)
fits_files = [f for f in files if f.endswith('.fits')]
fits_files.sort()

def read_scale(naxis, crpix, crval, cdelt, cunit, outputUnit='mas'):
    """
    Create a scale array based on the FITS header parameters.
    
    Parameters:
    naxis (int): Number of axes in the FITS data.
    crpix (float): Reference pixel.
    crval (float): Reference value at the reference pixel.
    cdelt (float): Increment per pixel.
    
    Returns:
    np.ndarray: Scale array.
    """
    axis = np.arange(naxis) * cdelt + crval - (crpix - 1) * cdelt;
    if cunit == 'deg':
         baseunit = 1/np.degrees(1)  # radians
    if outputUnit == 'mas':
        targetunit = 1/np.degrees(1) / 3600 / 1e3  # Convert to milliarcseconds
    print(f'Converting from {cunit} to {outputUnit}')
    print(f'Base unit: {baseunit}, Target unit: {targetunit}', np.pi/180)
    axis_unit = baseunit / targetunit 
    axis = axis * axis_unit
    return axis
    
limz = (36,45,110)
limz = (50,50,50)
    
# Load the  FITS files
data = []
hdr = []
hdr_img = []
for fits_file in fits_files:
    print(f'Loading {fits_file}...')
    file_path = os.path.join(dir, fits_file)
    with fits.open(file_path) as hdul:
        data.append(hdul[0].data)
        hdr_img.append(hdul['IMAGE-OI INPUT PARAM'].header)
        hdr.append(hdul[0].header)
        
L_fits_files = [f for i, f in enumerate(fits_files) if hdr_img[i]['WAVE_Max']*1e6 < 40]

print(f'Loaded {len(data)} FITS files from {dir}')
print('number of files:', len(data))
print('data shape:', data[0].shape if data else 'No data loaded')
# Convert to a numpy array
#data = np.array(data)
# Create a time array based on the number of FITS files
time = np.arange(len(data))
# Plotting the data

nwin = len(L_fits_files)
wdth = 15
hght = 4
ny = int(np.ceil(np.sqrt(nwin * wdth/hght)))
nx = int(np.floor(nwin / ny))
while nx * ny < nwin:
    nx += 1
#plt.style.use('dark_background')
fig, ax = plt.subplots(nx,ny,figsize=(wdth, hght))
print(len(ax))
if len(ax) == 1:
    axf = [ax]  # Ensure ax is always a 2D array for consistency
else:
    axf = ax.flatten()
count = 0;


# Beam parameters
beama = np.array((4.83, 7.2, 12.8)) / 3600 / 1000/2  # FWHM major axis in mas
beamb = np.array((3.17, 4.75,8.45)) / 3600 / 1000/2 # FWHM minor axis in mas
beamang = -25.23  # Position angle in degrees
#beamang = -55.23  # Position angle in degrees

for i, data in enumerate(data):
    print(f'Plotting {fits_files[i]}...')
    if L_fits_files and fits_files[i] not in L_fits_files:
        print(f'Skipping {fits_files[i]} as it is not in the L_fits_files list.')
        continue
    pixsize = hdr[i]['CDELT1']  # Assuming uniform pixel scale
    print(f'Pixel size: {pixsize * 3600 * 1000} mas')
    
    wlen_min = hdr_img[i]['WAVE_MIN']*1e6
    wlen_max = hdr_img[i]['WAVE_MAX']*1e6
    # Recompute beam size according to input wavelength
    resol = (wlen_min * 1e-6) / 130 # in radians
    print('resolution:', resol, 'radians')
    print('resolution:', np.degrees(resol) * 3600 * 1000, 'mas')
    beama = np.repeat(np.degrees(resol)/2,4)
    beamb = beama * 8.45/12.8
    
    print('beam major axis:', beama[count] * 3600 * 1000, 'mas')
    print('beam minor axis:', beamb[count] * 3600 * 1000, 'mas')
    print('beam position angle:', beamang, 'degrees')
    xaxis = read_scale(hdr[i]['NAXIS1'], hdr[i]['CRPIX1'], hdr[i]['CRVAL1'], hdr[i]['CDELT1'], hdr[i]['CUNIT1'], outputUnit='mas')
    yaxis = read_scale(hdr[i]['NAXIS2'], hdr[i]['CRPIX2'], hdr[i]['CRVAL2'], hdr[i]['CDELT2'], hdr[i]['CUNIT1'], outputUnit='mas')
    

    # Convert FWHM to sigma (pixels)
    sigma_a = beama[count] / pixsize / 2.3548
    sigma_b = beamb[count] / pixsize / 2.3548
    print(f'FWHM major axis: {beama[count] * 3600 * 1000} mas, sigma_a: {sigma_a:.4f} pixels')
    print(f'FWHM minor axis: {beamb[count] * 3600 * 1000} mas, sigma_b: {sigma_b:.4f} pixels')

    # Create a 2D elongated Gaussian kernel
    size = int(np.ceil(8 * max(sigma_a, sigma_b)))
    xg = np.arange(-size, size+1)
    yg = np.arange(-size, size+1)
    xg, yg = np.meshgrid(xg, yg)
    theta = np.degrees(beamang)
    print(f'Beam position angle: {np.deg2rad(theta)} degrees')
    x_rot = xg * np.cos(theta) + yg * np.sin(theta)
    y_rot = -xg * np.sin(theta) + yg * np.cos(theta)
    gauss_kernel = np.exp(-0.5 * ((x_rot / sigma_a) ** 2 + (y_rot / sigma_b) ** 2))
    gauss_kernel /= np.sum(gauss_kernel)
    
    # Only center the image if 'SPARCO' is not in the filename (i.e. MIRA images)
    if 'SPARCO' not in fits_files[i]:
        # Find the coordinates of the maximum in the image
        max_pos = np.unravel_index(np.argmax(data), data.shape)
        center = np.array(data.shape) // 2
        shift = center - np.array(max_pos)
        # Shift the image so that the maximum is at the center
        data_centered = nd_shift(data, shift, order=3, mode='nearest')
    else:
        data_centered = data

    # Convolve the centered image with the kernel
    data_conv = fftconvolve(data_centered, gauss_kernel, mode='same')

    im = axf[count].imshow(data_conv/np.max(data_conv), aspect='equal', cmap='hot', origin='lower', extent=(xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]))
    axf[count].set_aspect('equal')
    

    axf[count].text(0.05, 0.95, f'{wlen_min:.2f}-{wlen_max:.2f} micron', transform=axf[count].transAxes, fontsize=8, color='yellow', verticalalignment='top')
    
    axf[count].set_xlim(-limz[count], limz[count]-1)
    axf[count].set_ylim(-limz[count], limz[count]-1)
    if 'SPARCO' in fits_files[i]:
        axf[count].plot(0,0,marker='*', color='lightblue', markeredgecolor='yellow', markersize=7, label='Star Position', markeredgewidth=.5)
    axf[count].set_facecolor("black")
    
    # Calculate ellipse parameters in plot units (mas)
    beam_width = beama[count] * 3600 * 1000  # major axis in mas
    beam_height = beamb[count] * 3600 * 1000  # minor axis in mas
    beam_angle = beamang   # in degrees

    # Place the ellipse in the lower left corner (10% from left, 10% from bottom)
    xlim = axf[count].get_xlim()
    ylim = axf[count].get_ylim()
    ellipse_x = xlim[0] + 0.1 * (xlim[1] - xlim[0]) + beam_width / 2
    ellipse_y = ylim[0] + 0.1 * (ylim[1] - ylim[0]) + beam_height / 2

    ellipse = Ellipse(
        (ellipse_x, ellipse_y),
        width=beam_width,
        height=beam_height,
        angle=beam_angle,
        edgecolor='white',
        facecolor='none',
        linestyle='dashed',
        linewidth=1.5,
        zorder=10
    )
    axf[count].add_patch(ellipse)
    #axf[count].tick_params(labelcolor='none', which='both', top=False, bottom=True, left=True, right=False)
    count+=1
    
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axf:
    ax.label_outer()

    
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.02,hspace=0.02)
for i in range(count, len(axf)):
    axf[i].set_visible(False)
    
# Add a colorbar on the side, matching the height of the subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

#plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for colorbar on the right
cax = fig.add_axes([axf[2].get_position().x1+0.01,axf[2].get_position().y0,0.02,axf[0].get_position().y1-axf[2].get_position().y0])
fig.colorbar(im, cax=cax, location='right')

input_file = os.path.basename(fits_files[0]).replace('.fits', '')
output_dir = dir0
plt.savefig(os.path.join(output_dir, f"{input_file}.pdf"), dpi=300, bbox_inches='tight')
plt.show()
