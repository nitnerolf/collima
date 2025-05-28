#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Self-calibration for optical/long baseine interferometry
# Author: fmillour
# Date: 08/03/2025
# inspired from self-cal module of fitomatic
#
################################################################################

from os import error
from astropy.io import fits
from scipy import *
import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt
from op_instruments import *
from scipy import interpolate
import math
from scipy.optimize import minimize

##############################################

def self_cal(inputFiles, inputImageCube, outputDir, overwrite=False, wlenIdx=, calibrateAmp=False, useVisAmp=False, gain=0.8, plot=True):
    """
    Self-calibration for optical/long baseine interferometry
    Parameters      ----------
    inputFiles:     List of input files (FITS) containing the visibilities
    inputImageCube: Input image (FITS) to be self-calibrated
    outputDir:      Output directory
    overwrite:      Overwrite the output files
    wlenIdx:        Wavelength index
    calibrateAmp:   Calibrate the amplitude
    useVisAmp:      Use the OI_VIS amplitude instead of OI_VIS2 squared visibility
    gain:           Gain factor for the self-calibration
    plot:           Plot the results
    """
    # Read the input files
    vis = []
    for i in range(len(inputFiles)):
        hdul = fits.open(inputFiles[i])
        vis.append(hdul[0].data)
        hdul.close()
        
    # Read the input image
    hdul  = fits.open(inputImage)
    image = hdul[0].data
    hdul.close()
    
    # Read the header
    header = hdul[0].header
    # Get the number of baselines
    nbl = vis[0].shape[0]
    # Get the number of channels
    nchan = vis[0].shape[1]
    # Get the number of polarizations
    npol = vis[0].shape[2