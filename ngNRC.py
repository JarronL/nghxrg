"""
ngNRC - NIRCam Detector Noise Simulator

Modification History:

15 Feb 2016, J.M. Leisenring, UA/Steward
	- First Release
21 July 2016, J.M. Leisenring, UA/Steward
	- Updated many things and more for nghxrg (v3.0)
11 Aug 2016, J.M. Leisenring, UA/Steward
	- Modified how the detector and multiaccum info is handled
	- Copied detector and multiaccum classes from pyNRC
	- In the future, we will want to integrate this directly
	  so that any changes made in the pyNRC classes are accounted.
"""
# Necessary for Python 2.6 and later
from __future__ import division, print_function

import numpy as np
from astropy.io import fits
import datetime
import os

# HxRG Noise Generator
import nghxrg as ng

#import nrc_utils

# Set log output levels
# webbpsf and poppy have too many unnecessary warnings
import logging
logging.getLogger('nghxrg').setLevel(logging.ERROR)

_log = logging.getLogger('ngNRC')
#_log.setLevel(logging.DEBUG)
#_log.setLevel(logging.INFO)
_log.setLevel(logging.WARNING)
#_log.setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING,format='%(name)-10s: %(levelname)-8s %(message)s')

def SCAnoise(scaid, params=None, file_out=None, caldir=None, 
	dark=True, bias=True, out_ADU=False, verbose=False, use_fftw=False, ncores=None):
	"""
	Create a data cube consisting of realistic NIRCam detector noise.

	This is essentially a wrapper for nghxrg.py that selects appropriate values
	for a specified SCA in order to reproduce realistic noise properties similiar
	to those measured during ISIM CV3.

	Parameters
	----------
	scaid : NIRCam SCA number (481, 482, ..., 490)
	params : A set of MULTIACCUM parameters such as:
		params = {'ngroup': 2, 'wind_mode': 'FULL', 
				'xpix': 2048, 'ypix': 2048, 'x0':0, 'y0':0}
		wind_mode can be FULL, STRIPE, or WINDOW
	file_out : Folder name and destination to place optional FITS output. 
		A timestamp will be appended to the end of the file name (and before .fits').
	caldir : Directory location housing the super bias and super darks for each SCA.
	dark : Use super dark? If True, then reads in super dark slope image.
	bias : Use super bias? If True, then reads in super bias image.
	out_ADU : Noise values are calculated in terms of equivalent electrons. This
		gives the option of converting to ADU (True) or keeping in term of e- (False).
		ADU values are converted to 16-bit UINT. Keep in e- if applying to a ramp
		observation then convert combined data to ADU later.
	
	Returns 
	----------
	Primary HDU with noise ramp in hud.data and header info in hdu.header.

	Examples 
	----------
	import ngNRC
	params = {'ngroup': 108, 'wind_mode': 'FULL', 
			'xpix': 2048, 'ypix': 2048, 'x0':0, 'y0':0}
			
	# Output to a file
	scaid = 481
	caldir = '/data/darks_sim/nghxrg/sca_images/'
	file_out = '/data/darks_sim/dark_sim_481.fits'
	hdu = ngNRC.SCAnoise(scaid, params, file_out=file_out, caldir=caldir, \
		dark=True, bias=True, out_ADU=True, use_fftw=False, ncores=None, verbose=False)

	# Don't save file, but keep hdu in e- for adding to simulated observation ramp
	scaid = 481
	caldir = '/data/darks_sim/nghxrg/sca_images/'
	hdu = ngNRC.SCAnoise(scaid, params, file_out=None, caldir=caldir, \
		dark=True, bias=True, out_ADU=False, use_fftw=False, ncores=None, verbose=False)

	"""
	
	# Extensive testing on both Python 2 & 3 shows that 4 cores is optimal for FFTW
	# Beyond four cores, the speed improvement is small. Those other processors are
	# are better used elsewhere.
	if use_fftw and (ncores is None): ncores = 4

	wind_mode = params.pop('wind_mode', 'FULL')
	xpix = params.pop('xpix', 2048)
	ypix = params.pop('ypix', 2048)
	x0 = params.pop('x0', 0)
	y0 = params.pop('y0', 0)
	det = detector_ops(scaid, wind_mode, xpix, ypix, x0, y0, params)

	# Line and frame overheads
	nroh     = det._line_overhead
	nfoh     = det._extra_lines[0]
	nfoh_pix = det._frame_overhead_pix

	# How many total frames (incl. dropped and all) per ramp?
	# Exclude last set of nd2 and nd3 (drops that add nothing)
	ma = det.multiaccum
	naxis3 = ma.nd1 + ma.ngroup*ma.nf + (ma.ngroup-1)*ma.nd2

	# Set bias and dark files
	sca_str = np.str(scaid)
	if caldir is None:
		base_dir  = '/Volumes/NIRData/sca_images/'
	else:
		base_dir = caldir
	bias_file = base_dir + 'SUPER_BIAS_'+sca_str+'.FITS' if bias else None
	dark_file = base_dir + 'SUPER_DARK_'+sca_str+'.FITS' if dark else None

	# Instantiate a noise generator object
	ng_h2rg = ng.HXRGNoise(naxis1=det.xpix, naxis2=det.ypix, naxis3=naxis3, 
				 n_out=det.nout, nroh=nroh, nfoh=nfoh, nfoh_pix=nfoh_pix,
				 dark_file=dark_file, bias_file=bias_file,
				 wind_mode=det.wind_mode, x0=det.x0, y0=det.y0,
				 use_fftw=use_fftw, ncores=ncores, verbose=verbose)
		 
	
	# Lists of each SCA and their corresponding noise info
	sca_arr = range(481,491)

	# These come from measured dark ramps acquired during ISIM CV3 at GSFC
	# Gain values (e/ADU). Everything else will be in measured ADU units
	gn_arr =  [2.07, 2.01, 2.16, 2.01, 1.83, 
			   2.00, 2.42, 1.93, 2.30, 1.85]

	# Noise Values (ADU)
	ktc_arr = [18.5, 15.9, 15.2, 16.9, 20.0, 
			   19.2, 16.1, 19.1, 19.0, 20.0]
	ron_arr  = [[4.8,4.9,5.0,5.3], [4.4,4.4,4.4,4.2], [4.8,4.0,4.1,4.0], [4.5,4.3,4.4,4.4],
				[4.2,4.0,4.5,5.4],
				[5.1,5.1,5.0,5.1], [4.6,4.3,4.5,4.2], [5.1,5.6,4.6,4.9], [4.4,4.5,4.3,4.0],
				[4.5,4.3,4.6,4.8]]
	# Pink Noise Values (ADU)
	cp_arr  = [ 2.0, 2.5, 1.9, 2.5, 2.1,
				2.5, 2.5, 3.2, 3.0, 2.5]
	up_arr  = [[0.9,0.9,0.9,0.9], [0.9,1.0,0.9,1.0], [0.8,0.9,0.8,0.8], [0.8,0.9,0.9,0.8],
			   [1.0,1.3,1.0,1.1],
			   [1.0,0.9,1.0,1.0], [0.9,0.9,1.1,1.0], [1.0,1.0,1.0,0.9], [1.1,1.1,0.8,0.9],
			   [1.1,1.1,1.0,1.0]]
		   
		   
	# Offset Values (ADU)
	bias_avg_arr = [5900, 5400, 6400, 6150, 11650, 
					7300, 7500, 6700, 7500, 11500]
	bias_sig_arr = [20.0, 20.0, 30.0, 11.0, 50.0, 
					20.0, 20.0, 20.0, 20.0, 20.0]
	ch_off_arr   = [[1700, 530, -375, -2370], [-150, 570, -500, 350], [-530, 315, 460, -200],
					[480, 775, 1040, -2280],  [560, 100, -440, -330],
					[105, -29, 550, -735],    [315, 425, -110, -590],   [918, -270, 400, -1240],
					[-100, 500, 300, -950],   [-35, -160, 125, -175]]
	f2f_corr_arr = [14.0, 13.8, 27.0, 14.0, 26.0,
					14.7, 11.5, 18.4, 14.9, 14.8]
	f2f_ucorr_arr= [[18.4,11.1,10.8,9.5], [7.0,7.3,7.3,7.1], [6.9,7.3,7.3,7.5],
					[6.9,7.3,6.5,6.7], [16.6,14.8,13.5,14.2],
					[7.2,7.5,6.9,7.0], [7.2,7.6,7.5,7.4], [7.9,6.8,6.9,7.0],
					[7.6,8.6,7.5,7.4], [13.3,14.3,14.1,15.1]]
	aco_a_arr    = [[770, 440, 890, 140], [800, 410, 840, 800], [210,680,730,885],
					[595, 642, 634, 745], [-95,660,575,410],
					[220, 600, 680, 665], [930,1112, 613, 150], [395, 340, 820, 304],
					[112, 958, 690, 907], [495, 313, 392, 855]]
	ref_inst_arr = [1.0, 1.5, 1.0, 1.3, 1.0, 
					1.0, 1.0, 1.0, 2.2, 1.0]


	# SCA Index
	ind = sca_arr.index(scaid)

	# Convert everything to e-
	gn = gn_arr[ind]
	# Noise Values
	ktc_noise= gn * ktc_arr[ind] * 1.15            # kTC noise in electrons
	rd_noise = gn * np.array(ron_arr[ind]) * 0.93  # White read noise per integration
	# Pink Noise
	c_pink   = gn * cp_arr[ind] * 1.6              # Correlated pink noise
	u_pink   = gn * np.array(up_arr[ind]) * 1.4    # Uncorrelated
	ref_rat  = 0.9  # Ratio of reference pixel noise to that of reg pixels

	# Offset Values
	bias_off_avg = gn * bias_avg_arr[ind] + 110  # On average, integrations start here in electrons
	bias_off_sig = gn * bias_sig_arr[ind]  # bias_off_avg has some variation. This is its std dev.
	bias_amp     = gn * 1.0     # A multiplicative factor to multiply bias_image. 1.0 for NIRCam.

	# Offset of each channel relative to bias_off_avg.
	ch_off = gn * np.array(ch_off_arr[ind]) + 110
	# Random frame-to-frame reference offsets due to PA reset
	ref_f2f_corr  = gn * f2f_corr_arr[ind] * 0.95
	ref_f2f_ucorr = gn * np.array(f2f_ucorr_arr[ind]) * 1.15 # per-amp
	# Relative offsets of altnernating columns
	aco_a = gn * np.array(aco_a_arr[ind])
	aco_b = -1 * aco_a
	#Reference Instability
	ref_inst = gn * ref_inst_arr[ind]

	# If only one output (window mode) then select first elements of each array
	if det.nout == 1:
		rd_noise = rd_noise[0]
		u_pink = u_pink[0]
		ch_off = ch_off[0]
		ref_f2f_ucorr = ref_f2f_ucorr[0]
		aco_a = aco_a[0]; aco_b = aco_b[0]
	
	# Run noise generator
	hdu = ng_h2rg.mknoise(None, gain=gn, rd_noise=rd_noise, c_pink=c_pink, u_pink=u_pink, 
			reference_pixel_noise_ratio=ref_rat, ktc_noise=ktc_noise,
			bias_off_avg=bias_off_avg, bias_off_sig=bias_off_sig, bias_amp=bias_amp,
			ch_off=ch_off, ref_f2f_corr=ref_f2f_corr, ref_f2f_ucorr=ref_f2f_ucorr, 
			aco_a=aco_a, aco_b=aco_b, ref_inst=ref_inst, out_ADU=out_ADU)

	hdu.header = nrc_header(det, header=hdu.header)
	hdu.header['UNITS'] = 'ADU' if out_ADU else 'e-'

	# Write the result to a FITS file
	if file_out is not None:
		now = datetime.datetime.now().isoformat()[:-7]
		hdu.header['DATE'] = datetime.datetime.now().isoformat()[:-7]
		if file_out.lower()[-5:] == '.fits':
			file_out = file_out[:-5]
		if file_out[-1:] == '_':
			file_out = file_out[:-1]
		
# 		file_now = now
# 		file_now = file_now.replace(':', 'h', 1)
# 		file_now = file_now.replace(':', 'm', 1)
# 		file_out = file_out + '_' + file_now + '.fits'
		file_out = file_out + '.fits'

		hdu.header['FILENAME'] = os.path.split(file_out)[1]
		hdu.writeto(file_out, clobber='True')
	
	return hdu

class detector_ops(object):
    """ 
    Class to hold detector operations information. Includes SCA attributes such as
    detector names and IDs as well as a subclass multiaccum for ramp settings.
    """
    
    def __init__(self, detector=481, wind_mode='FULL', xpix=2048, ypix=2048, x0=0, y0=0,
                 multi_args={}):

        # Typical values for SW/LW detectors that get saved based on SCA ID
        # After setting the SCA ID, these various parameters can be updated,
        # however they will be reset whenever the SCA ID is modified.
        #   - Pixel Scales in arcsec/pix (SIAF PRDDEVSOC-D-012, 2016 April)
        #   - Typical dark current values in e-/sec (ISIM CV3)
        #   - Read Noise in e-
        #   - IPC and PPC in %
        #   - p_excess: Parameters that describe the excess variance observed in
        #     effective noise plots.
        self._properties_SW = {'pixel_scale':0.0311, 'dark_current':0.002, 'read_noise':11.5, 
                               'IPC':0.54, 'PPC':0.09, 'p_excess':(1.0,5.0), 'ktc':37.6}
        self._properties_LW = {'pixel_scale':0.0630, 'dark_current':0.034, 'read_noise':9.5, 
                               'IPC':0.60, 'PPC':0.19, 'p_excess':(1.5,10.0), 'ktc':36.8}
        self.auto_pixel = True          # Automatically set the pixel scale based on detector selection
        
        self._scaids = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:'A5',
                        486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:'B5'}
        try: self.scaid = detector
        except ValueError: 
            try: self.detid = detector
            except ValueError: 
                raise ValueError("Invalid detector: {0} \n\tValid names are: {1},\n\t{2}" \
                      .format(detector, ', '.join(self.detid_list), ', '.join(str(e) for e in self.scaid_list)))
        
        self._detector_pixels = 2048
        self.wind_mode = wind_mode
        self.xpix = xpix; self.x0 = x0
        self.ypix = ypix; self.y0 = y0
        self._validate_pixel_settings()
        
        # Pixel Rate in Hz
        self._pixel_rate = 1e5
        # Number of extra clock ticks per line
        self._line_overhead = 12
               
        self.multiaccum = multiaccum(**multi_args)

    @property
    def scaid(self):
        """Selected SCA ID from detectors in the `scaid_list` attribute. 481, 482, etc."""
        return self._scaid

    @property
    def detid(self):
        """Selected Detector ID from detectors in the `detid_list` attribute. A1, A2, etc."""
        return self._detid

    @property
    def detname(self):
        """Selected Detector ID from detectors in the `scaid_list` attribute. NRCA1, NRCA2, etc."""
        return self._detname

    # Used for setting the SCA ID then updating all the other detector properties
    @scaid.setter
    def scaid(self, value):
        """Set SCA ID (481, 482, ..., 489, 490). Automatically updates other relevant attributes."""
        if value not in self.scaid_list:
            raise ValueError("Invalid SCA: {} \n\tValid SCA names are: {}"
                             .format(value, ', '.join(str(e) for e in self.scaid_list)))
            
        self._scaid = value
        self._detid = self._scaids.get(self._scaid)

        # Detector Name (as stored in FITS headers): NRCA1, NRCALONG, etc.
        if self.channel=='LW': self._detname = 'NRC' + self.module + 'LONG'
        else:  self._detname = 'NRC' + self._detid
        
        # Select various detector properties (pixel scale, dark current, read noise, etc)
        # depending on LW or SW detector
        dtemp = self._properties_LW if self.channel=='LW' else self._properties_SW
        if self.auto_pixel: self.pixelscale = dtemp['pixel_scale']
        self.ktc          = dtemp['ktc']
        self.dark_current = dtemp['dark_current']
        self.read_noise   = dtemp['read_noise']
        self.IPC          = dtemp['IPC']
        self.PPC          = dtemp['PPC']
        self.p_excess     = dtemp['p_excess']

    # Similar to scaid.setter, except if detector ID is specified.
    @detid.setter
    def detid(self, value):
        """Set detector ID (A1, A2, ..., B4, B5). Automatically updates other relevant attributes."""
        if value not in self.detid_list:
            raise ValueError("Invalid detector ID: {} \n\tValid names are: {}"
                             .format(value, ', '.join(self.detid_list)))
            
        # Switch dictioary keys and values, grab the corresponding SCA ID,
        # and then call scaid.setter
        newdict = {y:x for x,y in self._scaids.items()}
        self.scaid = newdict.get(value)
            
    @property
    def scaid_list(self):
        """Allowed SCA IDs"""
        return sorted(list(self._scaids.keys()))

    @property
    def detid_list(self):
        """Allowed Detector IDs"""
        return sorted(list(self._scaids.values()))
    
    @property
    def module(self):
        """NIRCam modules A or B (inferred from detector ID)"""
        return self._detid[0]
        
    @property
    def channel(self):
        """Detector channel 'SW' or 'LW' (inferred from detector ID)"""
        return 'LW' if self.detid.endswith('5') else 'SW'
    
    @property
    def nout(self):
        """Number of simultaenous detector output channels stripes"""
        return 1 if self.wind_mode == 'WINDOW' else 4
        
    @property
    def chsize(self):
        """"""
        return self.xpix / self.nout
    
    @property
    def ref_info(self):
        """
        Array of reference pixel borders being read out: [lower, upper, left, right].
        """
        det_size = self._detector_pixels
        x1 = self.x0; x2 = x1 + self.xpix
        y1 = self.y0; y2 = y1 + self.ypix
        
        w = 4 # Width of ref pixel border
        lower = w-y1; upper = w-(det_size-y2)
        left  = w-x1; right = w-(det_size-x2)
        ref_all = np.array([lower,upper,left,right])
        ref_all[ref_all<0] = 0
        return ref_all


    def _validate_pixel_settings(self):
        """ 
        Validation to make sure the defined pixel sizes are consistent with
        detector readout mode (FULL, STRIPE, or WINDOW)
        """
        
        wind_mode = self.wind_mode

        modes = ['FULL', 'STRIPE', 'WINDOW']
        if wind_mode not in modes:
            _log.warning('%s not a valid window readout mode! Returning...' % wind_mode)
            return
        
        detpix = self._detector_pixels
        xpix = self.xpix; x0 = self.x0
        ypix = self.ypix; y0 = self.y0

        # Check some consistencies with frame sizes
        if wind_mode == 'FULL':
            if ypix != detpix:
                _log.warning('In {0} mode, but ypix not {1}. Setting ypix={1}.'.format(wind_mode,detpix))
                ypix = detpix
            if y0 != 0:
                _log.warning('In {0} mode, but x0 not 0. Setting y0=0.'.format(wind_mode))
                y0 = 0

        if (wind_mode == 'STRIPE') or (wind_mode == 'FULL'):
            if xpix != detpix:
                _log.warning('In {0} mode, but xpix not {1}. Setting xpix={1}.'.format(wind_mode,detpix))
                xpix = detpix
            if x0 != 0:
                _log.warning('In {0} mode, but x0 not 0. Setting x0=0.'.format(wind_mode))
                x0 = 0
                
        if (x0+xpix) > detpix:
            raise ValueError("x0+xpix ({}+{}) is larger than detector size ({})!".format(x0,xpix,detpix))
        if (y0+ypix) > detpix:
            raise ValueError("y0+ypix ({}+{}) is larger than detector size ({})!".format(y0,ypix,detpix))
            
        # Update values if no errors were thrown
        self.xpix = xpix; self.x0 = x0
        self.ypix = ypix; self.y0 = y0

    @property
    def _extra_lines(self):
        """Determine how many extra lines/rows are added to a to a given frame"""
        if self.nout == 1:
            sub = 2**(np.arange(8)+4.)         # Subarray size            
            l1 = np.array([3,2.5,2,2,2,2,2,2]) # Extra lines for early frames
            l2 = np.array([6,5.0,4,3,2,2,2,2]) # Extra lines for last frame
            ladd1 = (np.interp([self.xpix],sub,l1))[0]
            ladd2 = (np.interp([self.xpix],sub,l1))[0]
        else:
            ladd1 = 1
            ladd2 = 2
        return (ladd1,ladd2)
    
    @property
    def _exp_delay(self):
        """
        Additional overhead time at the end of an exposure.
        (Does this occur per integration or per exposure???)
        """
        # Note: I don't think this adds any more photons to a pixel
        #   It simply slightly delays the subsequent reset frame AFTER the last pixel read

        # Clock ticks per line
        xticks = self.chsize + self._line_overhead  
        l1,l2 = self._extra_lines
        return xticks * (l2 - l1) / self._pixel_rate
    
    @property
    def _frame_overhead_pix(self):
        pix_offset = 0 if self.nout==1 else 1
        return pix_offset
        
    @property
    def time_frame(self):
        """Determine frame times based on xpix, ypix, and wind_mode."""
    
        chsize = self.chsize                        # Number of x-pixels within a channel
        xticks = self.chsize + self._line_overhead  # Clock ticks per line
        flines = self.ypix + self._extra_lines[0] # Lines per frame (early frames)
        
        # Add a single pix offset for full frame and stripe.
        pix_offset = self._frame_overhead_pix
        end_delay = 0 # Used for syncing each frame w/ FPE bg activity. Not currently used.
        
        # Total number of clock ticks per frame (reads and drops)
        fticks = xticks*flines + pix_offset + end_delay
        
        # Return frame time
        return fticks / self._pixel_rate
    
    @property
    def time_group(self):
        """Time per group based on time_frame, nf, and nd2."""
        return self.time_frame * (self.multiaccum.nf + self.multiaccum.nd2)
    
    @property
    def time_ramp(self):
        """Integration time for a single ramp."""

        # How many total frames (incl. dropped and all) per ramp?
        # Exclude last set of nd2 and nd3 (drops that add nothing)
        nf = self.multiaccum.nf
        nd1 = self.multiaccum.nd1
        nd2 = self.multiaccum.nd2
        ngroup = self.multiaccum.ngroup
        
        tint = (nd1 + ngroup*nf + (ngroup-1)*nd2) * self.time_frame
        if tint > 1200:
            _log.warning('Ramp time of %.2f is long. Is this intentional?' % tint)
            
        return tint
    
    @property
    def time_int(self):
        """Same as time_ramp, except that time_int follows the JWST nomenclature"""
        return self.time_ramp
    
    @property
    def time_exp(self):
        """Total exposure time for all ramps."""
        return self.multiaccum.nint * self.time_ramp
    
    def times_to_dict(self, verbose=False):
        """Export ramp times as dictionary with option to print output to terminal."""
        
        times = [('t_frame',self.time_frame), ('t_group',self.time_group), \
                 ('t_int',self.time_int), ('t_exp',self.time_exp)]
        
        d={}
        for k, v in times:
            d[k] = v
            if verbose: print("{:<8} {:10.4f}".format(k, v))
                
        return d
    
    def pixel_noise(self, fsrc=0.0, fzodi=0.0, fbg=0.0):
        """
        Return theoretical noise calculation for the specified MULTIACCUM ramp in terms of e-/sec.
        Uses the pre-defined detector-specific noise properties. Can specify flux of a source
        as well as background and zodiacal light (in e-/sec/pix).

        Parameters
        ===========
        fsrc (float)  : Flux of source in e-/sec/pix
        fzodi (float) : Flux of the zodiacal background in e-/sec/pix
        fbg (float)   : Flux of telescope background in e-/sec/pix
        
        These three components are functionally the same as they are immediately summed.
        They can also be single values or multiple elements (list, array, tuple, etc.).
        If multiple inputs are arrays, make sure their array sizes match.
        """
        
        ma = self.multiaccum
        return pix_noise(n=ma.ngroup, m=ma.nf, s=ma.nd2, tf=self.time_frame, \
                         rn=self.read_noise, ktc=self.ktc, idark=self.dark_current, \
                         p_excess=self.p_excess, fsrc=fsrc, fzodi=fzodi, fbg=fbg) \
                        / np.sqrt(ma.nint)


class multiaccum(object):
    """
    A class for defining MULTIACCUM ramp settings.
    
    read_mode (string) : NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', 'BRIGHT2', etc.
    nint, ngroup, nf, nd1, nd2, nd3 (int) : Ramp parameters, some of which may be
        overwritten by read_mode. See JWST MULTIACCUM documentation for more details.
        
    NIRCam-specific readout modes:
        patterns = ['RAPID', 'BRIGHT1', 'BRIGHT2', 'SHALLOW2', 'SHALLOW4', 
                'MEDIUM2', 'MEDIUM8', 'DEEP2', 'DEEP8']
        nf_arr   = [1, 1, 2, 2, 4, 2, 8,  2,  8] # Averaged frames per group
        nd2_arr  = [0, 1, 0, 3, 1, 8, 2, 18, 12] # Dropped frames per group (group gap)


    """
    
    def __init__(self, read_mode='RAPID', nint=1, ngroup=1, nf=1, nd1=0, nd2=0, nd3=0):
        
        
        # Pre-defined patterns
        patterns = ['RAPID', 'BRIGHT1', 'BRIGHT2', 'SHALLOW2', 'SHALLOW4', 'MEDIUM2', 'MEDIUM8', 'DEEP2', 'DEEP8']
        nf_arr   = [1,1,2,2,4,2,8, 2, 8]
        nd2_arr  = [0,1,0,3,1,8,2,18,12]
        self._pattern_settings = dict(zip(patterns, zip(nf_arr, nd2_arr)))
        
        self.nint = nint
        self._ngroup_max = 10000
        self.ngroup = ngroup
        self._nf = nf
        self._nd1 = nd1
        self._nd2 = nd2
        self._nd3 = nd3
        self.read_mode = read_mode

    @property
    def nint(self):
        """Number of ramps (integrations) in an exposure."""
        return self._nint
    @nint.setter
    def nint(self, value):
        self._nint = self._check_int(value)

    @property
    def ngroup(self):
        """Number of groups in a ramp (integration)."""
        return self._ngroup
    @ngroup.setter
    def ngroup(self, value):
        value = self._check_int(value)
        if value > self._ngroup_max:
            _log.warning('Specified ngroup (%s) greater than allowed value (%s)' \
                         % (value, self._ngroup_max))
            _log.warning('Setting ngroup = %s' % self._ngroup_max)
            value = self._ngroup_max
        self._ngroup = value

    @property
    def nf(self):
        """Number of frames per group."""
        return self._nf
    @nf.setter
    def nf(self, value):
        value = self._check_int(value)
        self._nf = self._check_custom(value, self._nf)

    @property
    def nd1(self):
        """Number of drop frame after reset (before first group read)."""
        return self._nd1
    @nd1.setter
    def nd1(self, value):
        value = self._check_int(value, minval=0)
        self._nd1 = self._check_custom(value, self._nd1)

    @property
    def nd2(self):
        """Number of drop frames within a group (aka, groupgap)."""
        return self._nd2
    @nd2.setter
    def nd2(self, value):
        value = self._check_int(value, minval=0)
        self._nd2 = self._check_custom(value, self._nd2)

    @property
    def nd3(self):
        """Number of drop frames after final read frame in ramp."""
        return self._nd3
    @nd3.setter
    def nd3(self, value):
        value = self._check_int(value, minval=0)
        self._nd3 = self._check_custom(value, self._nd3)
        
    @property
    def read_mode(self):
        """Selected Read Mode in the `patterns_list` attribute."""
        return self._read_mode
    @read_mode.setter
    def read_mode(self, value):
        """Set MULTIACCUM Readout. Automatically updates other relevant attributes."""
        value = value.upper()
        if value not in (self.patterns_list+['CUSTOM']):
            raise ValueError("Invalid Read Mode: {} \n\tValid SCA names are: {}"
                             .format(value, ', '.join(self.patterns_list)))
            
        self._read_mode = value
        self._validate_readout()
        
    @property
    def patterns_list(self):
        """Allowed NIRCam MULTIACCUM patterns"""
        return sorted(list(self._pattern_settings.keys()))

    #def to_dict(self):
    #    """Export ramp settings to a dictionary."""
    #    return {'read_mode':self.read_mode, 'nint':self.nint, 'ngroup':self.ngroup, 
    #            'nf':self.nf, 'nd1':self.nd1, 'nd2':self.nd2, 'nd3':self.nd3}

    def to_dict(self, verbose=False):
        """Export ramp settings to a dictionary."""
        
        p = [('read_mode',self.read_mode), ('nint',self.nint), ('ngroup',self.ngroup), \
             ('nf',self.nf), ('nd1',self.nd1), ('nd2',self.nd2), ('nd3',self.nd3)]
        
        d={}
        for k, v in p:
            d[k] = v
            if verbose: print("{:<10} {}".format(k, v))
                
        return d

    
    def _validate_readout(self):
        """ 
        Validation to make sure the defined ngroups, nf, etc. are consistent with
        the selected MULTIACCUM readout pattern.
        """

        if self.read_mode not in self.patterns_list:
            _log.warning('Readout %s not a valid NIRCam readout mode. Setting to CUSTOM.' % self.read_mode)
            self._read_mode = 'CUSTOM'
            _log.warning('Using explicit settings: ngroup=%s, nf=%s, nd1=%s, nd2=%s, nd3=%s' \
                         % (self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        elif self.read_mode == 'CUSTOM':
            _log.info('%s readout mode selected.' % self.read_mode)
            _log.info('Using explicit settings: ngroup=%s, nf=%s, nd1=%s, nd2=%s, nd3=%s' \
                         % (self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        else:
            _log.info('%s readout mode selected.' % self.read_mode)
            nf, nd2 = self._pattern_settings.get(self.read_mode)
            self._nf  = nf
            self._nd1 = 0
            self._nd2 = nd2
            self._nd3 = 0
            _log.info('Setting nf=%s, nd1=%s, nd2=%s, nd3=%s.' % (self.nf, self.nd1, self.nd2, self.nd3))
            
    
    def _check_custom(self, val_new, val_orig):
        """Check if read_mode='CUSTOM' before changing variable."""
        if self.read_mode == 'CUSTOM': 
            return val_new
        else: 
            print("Can only modify parameter if read_mode='CUSTOM'.")
            return val_orig
            
    def _check_int(self, val, minval=1):
        """Check if a value is a positive integer, otherwise throw exception."""
        val = float(val)
        if (val.is_integer()) and (val>=minval): 
            return int(val)
        else:
            #_log.error("Value {} should be a positive integer.".format(val))
            raise ValueError("Value {} must be a positive integer.".format(val))
 
 
def pix_noise(n=2, m=1, s=0, tf=10.737, rn=16.3, ktc=29.0, p_excess=(0,0),
    fsrc=0.0, idark=0.007, fzodi=0, fbg=0):
    """
    Theoretical noise calculation of a generalized MULTIACCUM ramp in terms of e-/sec.

    Parameters
    ===========
    n (int) : Number of groups in integration ramp
    m (int) : Number of frames in each group
    s (int) : Number of dropped frames in each group
    tf (float) : Frame time
    rn (float) : Read Noise per pixel
    fsrc (float) : Flux of source in e-/sec/pix
    idark (float) : Dark current in e-/sec/pix
    ktc (float) : kTC noise only valid for single frame (n=1)
    p_excess: An array or list of two elements that holding the
        parameters that describe the excess variance observed in
        effective noise plots. By default these are both 0.
        Recommended values are [1.0,5.0] or SW and [1.5,10.0] for LW.


    Various parameters can either be single values or numpy arrays.
    If multiple inputs are arrays, make sure their array sizes match.
    Variables that need to have the same array sizes (or a single value):
        - n, m, s, & tf
        - rn, idark, ktc, fsrc, fzodi, & fbg

    Array broadcasting also works:
        For Example
        n = np.arange(50)+1 # An array of groups to test out

        # Create 2D Gaussian PSF with FWHM = 3 pix
        npix = 20 # Number of pixels in x and y direction
        x = np.arange(0, npix, 1, dtype=float)
        y = x[:,np.newaxis]
        x0 = y0 = npix // 2 # Center position
        fwhm = 3.0
        fsrc = np.exp(-4*np.log(2.) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        fsrc /= fsrc.max()
        fsrc *= 10 # Total source counts/sec
        fsrc = fsrc.reshape(npix,npix,1) # Necessary for broadcasting

        # Represents pixel array w/ different RN/pix
        rn = np.ones([npix,npix,1])*15. 
        # Results is a (20x20)x50 showing the noise in e-/sec/pix at each group
        noise = pix_noise(n=n, rn=rn, fsrc=fsrc) 
    """

    n = np.array(n)
    m = np.array(m)
    s = np.array(s)
    tf = np.array(tf)
    max_size = np.max([n.size,m.size,s.size,tf.size])
    if n.size  != max_size: n  = n.repeat(max_size)
    if m.size  != max_size: m  = m.repeat(max_size)
    if s.size  != max_size: s  = s.repeat(max_size)
    if tf.size != max_size: tf = tf.repeat(max_size)

    # Total flux (e-/sec/pix)
    ftot = fsrc + idark + fzodi + fbg

    # Special case if n=1
    if (n==1).any():
        # Variance after averaging m frames
        var = ktc**2 + (rn**2 + ftot*tf) / m
        noise = np.sqrt(var) 
        noise /= tf # In terms of e-/sec

        if (n==1).all(): return noise
        noise_n1 = noise

    ind_n1 = (n==1)
    temp = np.array(rn+ktc+ftot)
    temp_bool = np.zeros(temp.shape, dtype=bool)
    ind_n1_all = (temp_bool | ind_n1)
    

    # Group time
    tg = tf * (m + s)

    # Read noise, group time, and frame time variances
    var_rn = rn**2      * 12.               * (n - 1.) / (m * n * (n + 1.))
    var_gp = ftot  * tg * 6. * (n**2. + 1.) * (n - 1.) / (5 * n * (n + 1.))
    var_fm = ftot  * tf * 2. * (m**2. - 1.) * (n - 1.) / (m * n * (n + 1.))

    # Functional form for excess variance above theoretical
    var_ex = 12. * (n - 1.)/(n + 1.) * p_excess[0]**2 - p_excess[1] / m**0.5

    # Variance of total signal
    var = var_rn + var_gp - var_fm + var_ex
    sig = np.sqrt(var)

    # Noise in e-/sec
    noise = sig / (tg * (n - 1))
    #print(ind_n1_all.shape,noise.shape,noise_n1.shape)
    if (n==1).any():
        noise[ind_n1_all] = noise_n1[ind_n1_all]

    return noise

def nrc_header(det_class, filter=None, pupil=None, 
               obs_time=None, header=None):

	"""
	Create a generic NIRCam FITS header from a detector_ops class.

	obs_time specifies when the observation was considered to be executed.
	If not specified, then it will choose the current time. This must be a 
	datetime object:
			datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)
		
	Also has the option of passing an existing header that will be updated.
	"""

	filter = 'UNKNOWN' if filter is None else filter
	pupil  = 'UNKNOWN' if pupil  is None else pupil

	d = det_class
	# MULTIACCUM ramp information
	ma = d.multiaccum

	# How many axes?
	naxis = 2 if ma.ngroup == 1 else 3
	if naxis == 3:
		naxis3 = ma.ngroup
	naxis1 = d.xpix
	naxis2 = d.ypix

	# Select Detector ID based on SCA ID
	detector = d.detname

	# Are we in subarray?
	sub_bool = True if d.wind_mode != 'FULL' else False

	# Window indices (0-indexed)
	x1 = d.x0; x2 = x1 + d.xpix
	y1 = d.y0; y2 = y1 + d.ypix

	# Ref pixel info
	ref_all = d.ref_info

	# Dates and times
	obs_time = datetime.datetime.now() if obs_time is None else obs_time
	# Total time to complete obs = (ramp_time+reset_time)*nramps
	# ramp_time does not include reset frames!!
	tdel = ma.nint * (d.time_int + d.time_frame) + d._exp_delay
	dtstart = obs_time.isoformat()
	dtend = (obs_time + datetime.timedelta(seconds=tdel)).isoformat()
	dstart = dtstart[:10]; dend = dtend[:10]
	tstart = dtstart[11:-3]; tend = dtend[11:-3]
	tsample = 1e6/d._pixel_rate

	################################################################
	# Create blank header
	hdr_update = False  if header is None else True
	hdr = fits.Header() if header is None else header

	# Add in basic header info
	if hdr_update is False:
		hdr['SIMPLE']  = (True,   'conforms to FITS standard')
		hdr['BITPIX']  = (16,     'array data type')
		hdr['NAXIS']   = (naxis,  'number of array dimensions')
		hdr['NAXIS1']  = naxis1
		hdr['NAXIS2']  = naxis2
		temp = hdr.pop('NAXIS3', None)
		if naxis == 3: hdr['NAXIS3']  = (naxis3, 'length of third data axis')
		hdr['EXTEND']  = True

	hdr['DATE']    = ('',   'date file created (yyyy-mm-ddThh:mm:ss,UTC)')
	hdr['BSCALE']  = (1,     'scale factor for array value to physical value')
	hdr['BZERO']   = (32768, 'physical value for an array value of zero')
	hdr['UNITS']   = ('',  'Units for the data type (ADU, e-, etc.)')
	hdr['ORIGIN']  = ('UAz',  'institution responsible for creating FITS file')
	hdr['FILENAME']= ('',   'name of file')
	hdr['FILETYPE']= ('raw', 'type of data found in data file')

	# Observation Description
	hdr['TELESCOP']= ('JWST',    'telescope used to acquire data')
	hdr['INSTRUME']= ('NIRCAM',  'instrument identifier used to acquire data')
	hdr['OBSERVER']= ('UNKNOWN', 'person responsible for acquiring data')
	hdr['DATE-OBS']= (dstart, 'UT date of observation (yyyy-mm-dd)')
	hdr['TIME-OBS']= (tstart, 'Approximate UT time of start of observation (hh:mm:ss.sss)')
	hdr['DATE-END']= (dend,   'UT date of end of observation(yyyy-mm-dd)')
	hdr['TIME-END']= (tend,   'UT time of end of observation (hh:mm:ss.sss)')
	hdr['SCA_ID']  = (d.scaid,   'Unique SCA identification in ISIM')
	hdr['DETECTOR']= (d.detname, 'ASCII Mnemonic corresponding to the SCA_ID')
	hdr['TARGNAME']= ('UNKNOWN', 'Target name')
	hdr['OBSMODE'] = ('UNKNOWN', 'Observation mode')

	# Positions of optical elements
	hdr['FILTER']  = (filter, 'Module ' + d.module + ' ' + d.channel + ' FW element')
	hdr['PUPIL']   = (pupil, 'Module ' + d.module + ' ' + d.channel + ' PW element')
	hdr['PILSTATE']= ('RETRACTED', 'Module ' + d.module + ' PIL deploy state')

	# Readout Mode
	hdr['NSAMPLE'] = (1,            'A/D samples per read of a pixel')
	hdr['NFRAME']  = (ma.nf,         'Number of frames in group')
	hdr['NGROUP']  = (ma.ngroup,     'Number groups in an integration')
	hdr['NINT']    = (ma.nint,     'Number of integrations in an exposure')

	hdr['TSAMPLE'] = (tsample,           'Delta time between samples in microsec')
	hdr['TFRAME']  = (d.time_frame,   'Time in seconds between frames')
	hdr['TGROUP']  = (d.time_group,     'Delta time between groups')
	hdr['DRPFRMS1']= (ma.nd1, 'Number of frame skipped prior to first integration')
	hdr['GROUPGAP']= (ma.nd2, 'Number of frames skipped')
	hdr['DRPFRMS3']= (ma.nd3, 'Number of frames skipped between integrations')
	hdr['FRMDIVSR']= (ma.nf,  'Divisor applied to each group image')
	hdr['INTAVG']  = (1,   'Number of integrations averaged in one image')
	hdr['NRESETS1']= (1,   'Number of reset frames prior to first integration')
	hdr['NRESETS2']= (1,   'Number of reset frames between each integration')
	hdr['INTTIME'] = (d.time_int,   'Total integration time for one MULTIACCUM')
	hdr['EXPTIME'] = (d.time_exp,    'Exposure duration (seconds) calculated')
	hdr['SUBARRAY']= (sub_bool,    'T if subarray used, F if not')
	hdr['READOUT'] = (ma.read_mode, 'Readout pattern name')
	hdr['ZROFRAME']= (False,       'T if zeroth frame present, F if not')

	#Reference Data
	hdr['TREFROW'] = (ref_all[1], 'top reference pixel rows')
	hdr['BREFROW'] = (ref_all[0], 'bottom reference pixel rows')
	hdr['LREFCOL'] = (ref_all[2], 'left col reference pixels')
	hdr['RREFCOL'] = (ref_all[3], 'right col reference pixels')
	hdr['NREFIMG'] = (0, 'number of reference rows added to end')
	hdr['NXREFIMG']= (0, 'reference image columns')
	hdr['NYREFIMG']= (0, 'reference image rows')
	hdr['COLCORNR']= (x1+1, 'The Starting Column for ' + detector)
	hdr['ROWCORNR']= (y1+1, 'The Starting Row for ' + detector)

	hdr.insert('EXTEND', '', after=True)
	hdr.insert('EXTEND', '', after=True)
	hdr.insert('EXTEND', '', after=True)

	hdr.insert('FILETYPE', '', after=True)
	hdr.insert('FILETYPE', ('','Observation Description'), after=True)
	hdr.insert('FILETYPE', '', after=True)

	hdr.insert('OBSMODE', '', after=True)
	hdr.insert('OBSMODE', ('','Optical Mechanisms'), after=True)
	hdr.insert('OBSMODE', '', after=True)

	hdr.insert('PILSTATE', '', after=True)
	hdr.insert('PILSTATE', ('','Readout Mode'), after=True)
	hdr.insert('PILSTATE', '', after=True)

	hdr.insert('ZROFRAME', '', after=True)
	hdr.insert('ZROFRAME', ('','Reference Data'), after=True)
	hdr.insert('ZROFRAME', '', after=True)

	hdr.insert('ROWCORNR', '', after=True)
	hdr.insert('ROWCORNR', '', after=True)

	return hdr


