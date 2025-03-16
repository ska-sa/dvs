#!/usr/bin/python
"""
    Formalization of earlier http://kat-imager.kat.ac.za:8888/notebooks/SysEng/Tipping%20curve%20for%20SKA%20prototype%20report-UHF%20Shroud%20tests.ipynb
    Typical use 1:
        python tipcurve.py /data/132598363.h5 m008 900,1200,1700
        
    Typical use 2:
        import util
        ds = util.open_dataset("/data/132598363.h5", "m008")
        tip_results, aperture_efficiency = tipcurve.process(ds, "m008", 900,1700, freq_mask='', PLANE="antenna",spec_sky=True)
        tipcurve.report(tip_results, aperture_efficiency, select_freq=[900,1200,1700], plot_limits=True)
        
    @author aph@sarao.ac.za
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
import scikits.fitting as fit
import healpy as hp
import ephem
from astropy import units as u
from astropy.coordinates import SkyCoord
import time
import scape
import multiprocessing
from . import util, modelsroot
from analysis import katselib as ksl
from analysis import katsemat as mat
from analysis import katsemodels as models

# Noise Diode Models that are not yet "deployed in the telescope"
nd_models_folder = modelsroot + '/noise-diode-models'

Tcmb = models.Tcmb # [K]
T0degC = models.T0degC # [K]


class Sky_temp:
    """
       T_sky = T_cont + T_cmb  from the global sky model
       Read in  file, and provide a method of passing back the Tsky temp a at a position
    """
    def __init__(self,nu=1828.0,hpbw=0.01,smooth=0): # APH 032017 changed smooth to numeric scale factor
        """ Load The Tsky data from an inputfile in FITS format and scale to frequency
            @param nu: frequency for the sky map, default 1828 [MHz]
            @param hpbw: suggested resolution for the map, default 0.01 [radian]
        """
        self.hpbw = hpbw
        self.null = 1.29*self.hpbw # First null of ^2 beam (i.e. feed has parabolic taper)
        self._freq_map_, self.resolution = models.get_gsm(nu, smooth*self.hpbw)
        self.nu = nu
        
    def freq_map(self): # APH added 092018 instead of object attribute
        return self._freq_map_ + Tcmb  # CMB temperature not included in the GSM maps - do not modify the cached map, and don't cache this modified version!

    def Tsky(self,ra,dec,integrate=True,bm_floor=0): # APH 042017 added integrate=True to replace earlier
        """ Given RA/Dec in Degrees  return the value of the spot
            assuming the healpix map is in Galatic coords
            @param integrate: False to return the temperature at the map resolution, True to return iint_beamnull{GSM}/SA(4pi) (default True)
        """
        freq_map = self.freq_map()
        def get_value(ra,dec):
            if integrate:
                return models.mapvalue(freq_map, ra, dec, radius=self.null,beam="^2",hpbw=self.hpbw, bm_floor=bm_floor)
            else:
                return models.mapvalue(freq_map, ra, dec)
        ras = ra if (len(np.shape([ra]))==2) else [ra]
        decs = dec if (len(np.shape([dec]))==2) else [dec]
        result = [get_value(ra,dec) for ra,dec in zip(ras,decs)]
        return (result[0] if len(result)==1 else np.asarray(result))
        
    def plot_sky(self,ra=None,dec=None,norm='log',unit='Kelvin',  pointstyle='ro',pointlabels=None, lat='-30.713',lon='21.444',elev=1050,date=None, cartesian=True, fig=None): # APH ADDED from pointstyle
        """ Plots the sky temperature and overlays pointing centers as red dots
            The sky temperature is the data that was loaded when the class was initiated.
            plot_sky takes in 3 optional parameters:
                ra,dec  are list/1D-array like values of right ascension and declination
                pointlabels: list of *(galactic lon, galactic lat, string label)* to plot with pointstyle
                fig: figure to re-use, default None
            returns matplotlib figure object that the plot is associated with.
        """
        if not fig: # APH added this switch to re-use fig
            fig = plt.figure(figsize=(16,9))
            if (date is None):
                hp.cartview(self.freq_map(),coord='G', norm=norm,unit=unit,fig=fig.number)
            else:
                ov = models.FastGSMObserver()
                ov.lat, ov.lon, ov.elev = lat, lon, elev
                ov.date = date
                ov.generate(self.nu)
                if cartesian:
                    hp.cartview(ov.observed_gsm+Tcmb,coord='G', norm=norm,unit=unit,fig=fig.number)
                else:
                    hp.orthview(ov.observed_sky+Tcmb,coord='G', half_sky=True, norm=norm,unit=unit,fig=fig.number)
                    # Default flip='astro'i.e. east is left
                    ax = plt.gca(); ax.annotate("E", [-1,0], horizontalalignment='right'); ax.annotate("W", [1,0])
            plt.title("Continuum Brightness including CMB, at %.fMHz"%self.nu)
            #hp.graticule()
        
        if cartesian or (date is None): # Plot pointing coordinates in Cartesian projection
            c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            l = np.degrees(mat.wrap(-c.galactic.l.radian))
            b = np.degrees(c.galactic.b.radian)
            plt.plot(l,b,pointstyle) # APH changed 'ro' to pointstyle
            if pointlabels: # APH added this to annotate lon.lat points
                for _l,_b,_t in zip(l,b,pointlabels):
                    plt.annotate(_t,[_l,_b])
        else: # TODO: Plot pointing coordinates in Orthographic projection
            pass
        return fig

    # APH added these 052017
    def Tavgsky(self, lat='-30.713',lon='21.444',elev=1050,date=None,ra=None,dec=None,radius=-1,blank_mb=True,bm_floor=0,debug=False):
        """
            @param lat,lon,elev: location of observer as per pyephem, default location is at M000.
            @param date: One or more date tuple(s) - assumed in UTC!
            @param ra,dec: RA,DEC where observer's beam is pointed, in decimal degrees
            @param radius: if > 0 then it's the radius to integrate the far field to as a multiple of HPBW, else it's the whole sphere (default -1)
            @param blank_mb: True to exclude the main beam from the integral or False to include it (default True)
            @return beam-averaged visible sky temperature for each date or simply all-sky average if ra,dec is None. Includes CMB.
        """
        ov = models.FastGSMObserver()
        ov.lat, ov.lon, ov.elev = lat, lon, elev
        radius = min([np.pi, radius*self.hpbw]) if (radius > 0) else np.pi 
        radius_blank = self.null if blank_mb else 0
        
        # Match dimensions for date, ra, dec
        dates = date if isinstance(date[0], tuple) else [date]
        ras = ra if (len(np.shape([ra]))==2) else [ra]*len(dates)
        decs = dec if (len(np.shape([dec]))==2) else [dec]*len(dates)
        Tavg = []
        for date,ra,dec in zip(dates,ras,decs):
            # Apply 10 minute resolution to avoid costly fine updates to the horizon mask
            if (abs(ov.date-ephem.Date(date)) > 10/60./24.):
                ov.date = date
                ov.generate(self.nu)
                observed_gsm = ov.observed_gsm + Tcmb # Add CMB since it's not included in the GSM maps
            if (ra is None or dec is None): # Just uniform average of all sky without weighting with a beam... NOT USEFUL
                Tavg.append(np.ma.mean(observed_gsm))
            else: # Average of visible sky weighted by a beam pointed at RA,DEC & truncated at radius
                Tavg.append(models.mapvalue(observed_gsm, ra,dec, radius_blank=radius_blank, radius=radius,beam="^2",hpbw=self.hpbw, bm_floor=bm_floor)) # Floor as set by non-ideal reflector
        
        self.eta_mb = self._beam_factor_(self.null/self.hpbw,False,bm_floor) / self._beam_factor_(-1,False,bm_floor) # The ratio of solid angle of beam (to first null) relative to 4pi Sr beam solid angle for beam+floor
        
        if debug:
            print("%.f MHz: Tavg_bg=%s" % (self.nu, Tavg))
            ov.view()
            ov.view_observed_gsm()
        return (Tavg[0] if len(Tavg)==1 else np.asarray(Tavg))
    
    def _beam_factor_(self, radius=-1,blank_mb=True,bm_floor=0):
        """ @return: iint_{(0|null1)..radius}{B(th,ph) dOmega}"""
        radius = radius*self.hpbw if (radius > 0) else np.pi 
        radius_blank = self.null if blank_mb else 0
        return models.mapvalue(1, ra=None,dec=None, radius_blank=radius_blank, radius=radius,beam="^2",hpbw=self.hpbw, bm_floor=bm_floor)


class Tip_Results:
    """ This represents where & when tipping curve measurements were recorded, and the resulting Tsys & Tant vs. frequency. """
    def __init__(self, ds, filename):
        self.ant = ds.antenna.name
        self.filename = filename
        self.freqs = ds.freqs[:] # MHz
        self.observer_wgs84 = dict(zip(['lat','lon','elev'], map(float, ds.antenna.position_wgs84))) # rad,rad,m. float because the native ephem.Angle isn't pickle-able.
        self.height = self.observer_wgs84['elev']
        self.pressure = np.mean(ds.enviro['pressure']['value'])
        self.air_relative_humidity = np.mean(ds.enviro['humidity']['value'])/100. # Percent -> fraction
        self.surface_temperature = np.mean(ds.enviro['temperature']['value'])
        timestamps = np.array([np.average(scan_time) for scan_time in scape.extract_scan_data(ds.scans,'abs_time').data])
        elevation = np.array([np.average(scan_el) for scan_el in scape.extract_scan_data(ds.scans,'el').data])
        ra = np.array([np.average(scan_ra) for scan_ra in scape.extract_scan_data(ds.scans,'ra').data])
        dec = np.array([np.average(scan_dec) for scan_dec in scape.extract_scan_data(ds.scans,'dec').data])
        sort_ind = elevation.argsort()
        # Sort data in the order of ascending elevation
        self.timestamps, self.elevation, self.ra, self.dec = timestamps[sort_ind], elevation[sort_ind], ra[sort_ind], dec[sort_ind]
        # Cache this - used in process_*()
        self.sort_ind = sort_ind
        
        # Initialise the attributes which are to be computed
        self.PLANE = None # The reference plane at which the temperatures are represented
        self.Tsys = [] # [TsysH, TsysV, 0, sigmaTsysH, sigmaTsysV] with 2nd axis: freqs
        self.Tant = [] # Tsys_meas - Tsys_expected, as [TantH, TantV] with 2nd axis: freqs
    
    def sky_fig(self, freq=None, annotate_every=10, cartesian=True):
        freq = np.nanmean(self.freqs) if (freq is None) else freq
        T_skytemp = Sky_temp(freq)
        fig = T_skytemp.plot_sky(self.ra,self.dec,date=time.gmtime(self.timestamps[0])[:6], cartesian=cartesian, **self.observer_wgs84) # APH added the next line to annotate plot with elevation angles
        return T_skytemp.plot_sky(self.ra[::annotate_every],self.dec[::annotate_every], pointstyle='mo',
                                  pointlabels=['El %.f'%e for e in self.elevation[::annotate_every]], fig=fig, cartesian=cartesian, **self.observer_wgs84)

    def save(self, filename):
        """ Save results to the specified MATLAB file """
        raise NotImplementedError()
    
    @classmethod
    def load(cls, filename):
        """ Load results from the specified MATLAB file """
        raise NotImplementedError()


class System_Temp:
    """Extracted tipping curve data points and surface temperature, at a single frequency."""
    def __init__(self, ds, tip_meta, freq_index):
        """ First extract total power in each scan (both mean and standard deviation)
            @param ds: the measurement dataset.
            @param tip_meta: metadata extracted from the dataset that's the same for all frequencies.
        """
        self.tip_meta = tip_meta
        self.Tsys = {}
        self.sigma_Tsys = {}
        self.Tsys_sky = {}
        self.freq = tip_meta.freqs[freq_index]  #MHz Centre frequency of observation
        ### APH begin 032017 added the default opacity & T_atm calculation here.
        T_atm = 1.12 * (T0degC + tip_meta.surface_temperature) - 50.0 # APH 02/2017 changed comment: Ippolito 1989 eq 6.8-6, quoting Wulfsberg 1964
        tau = models.calc_atmospheric_opacity(tip_meta.surface_temperature,tip_meta.air_relative_humidity,tip_meta.pressure,tip_meta.height/1000.,self.freq/1000.)
        self.opacity = tau/np.sin(np.radians(tip_meta.elevation))
        # APH 03/2017 added the following two corrections, but only significant above 10 GHz and below 10deg elevation, so leave out.
        #a0 = 1/np.sin(np.radians(tip_ms.elevation))
        #self.opacity = tau*(a0 - 2.0/6370.95*a0*(a0**2-1)) # (19) Correction of airmass function for spherical atmosphere (not planar) as per Han & Westwater 2000
        #T_atm = 266.5 + 0.72*tip_ms.surface_temperature # Ballpark from Tabel VII of Han & Westwater 2000, but how does that extrapolate down from 20-30 GHz?
        self.T_atm = T_atm * (1 - np.exp(-self.opacity))
        ### APH end
        for pol in ['HH','VV']:
            power_stats = np.array([scape.stats.mu_sigma(s.pol(pol)[:,freq_index]) for s in ds.scans])[tip_meta.sort_ind]
            tipping_mu, tipping_sigma = np.transpose(power_stats)
            self.Tsys[pol] = tipping_mu
            self.sigma_Tsys[pol] = tipping_sigma
            self.Tsys_sky[pol] = [] # APH 032017 Just initialise, calculated in process_() to correctly transfer to feed input plane.
#            for ra,dec in zip(tip_info.ra,tip_info.dec):
#                self.Tsys_sky[pol].append(tipping_mu-T_sky(ra,dec))# APH 032017 In process(), use apeff to correctly transfer to feed input plane.

def calc_Sky_Temp(tip_meta, freq, D, bm_floor_dBi=-6, radius=30):
    """ Computes the model sky temperature at the specified frequency, at all pointings in `tip_meta`.
        The sky model is integrated with a realistic beam pattern for circular aperture with diameter `D`,
        clipped to `bm_floor_dBi`. The "average sky temperature" is computed from the first null of the pattern,
        out to the specified radius.
        
        @param freq: frequency to use for the sky model [MHz]
        @param D: diameter of the aperture [m]
        @param floor_dBi: the level to clip the beam amplitude to relative to the peak [dB]
        @param radius: [multiple of HPBW]
        @return dict(T_sky, sigma_Tsky, T_avgsky, eta_mb) """
    wavelength = models._c_/1e6/freq
    iso = (D*np.pi/wavelength)**-2 # Isotropic antenna gain relative to bore sight gain of the reflector antenna
    bm_floor = iso * 10**(bm_floor_dBi/10.)
    ra, dec = tip_meta.ra, tip_meta.dec
    T_skytemp = Sky_temp(nu=freq, hpbw=1.27*wavelength/D)
    T_sky = T_skytemp.Tsky(ra, dec, bm_floor=bm_floor)
    sigma_Tsky = 0.1*np.abs(np.asarray([0]+np.diff(T_sky,1).tolist()))
    timestamps, observer_wgs84 = tip_meta.timestamps, tip_meta.observer_wgs84
    T_avgsky = T_skytemp.Tavgsky(date=[time.gmtime(t)[:6] for t in timestamps], ra=ra,dec=dec, radius=radius, blank_mb=True ,bm_floor=bm_floor, **observer_wgs84) # Blank to first null of the parabolic pattern
    eta_mb = T_skytemp.eta_mb # The ratio of solid angle of beam (to first null) relative to 4pi Sr beam solid angle for beam as used to compute T_avgsky
    return dict(T_sky=T_sky, sigma_Tsky=sigma_Tsky, T_avgsky=T_avgsky, eta_mb=eta_mb)


def load_cal(filename_or_file, baseline, nd_models=None, freq_channel=None,channel_range=None,band_input=None, freq_mask='',remove_spikes=False, debug=False):
    """ Load the dataset into memory
        @param filename_or_file: either the filename for, or an already open katdal dataset 
        @param nd_models : None, or a string to override the noise diode models in the dataset using the ones in this directory.
                Model filenames are like 'rx.%(band).%(serialno).%(pol).csv' (e.g. 'rx.l.4.h.csv').
        @param freq_channel: a list containing groups of channel numbers, which defines the 'chunks' of data to process as 'output channels'
        @param channel_range: a string "start,stop" to provide a very coarse mask on top of 'freq_mask'.
        @param freq_mask: either a pickle file of boolean flags for the freq_channels,
                              or a text file with each line frequency start,stop to indicate omitted range.
        @return: scape.Dataset (freqs is in MHz!)
    """
    print('Loading noise diode models')
    
    # TODO: check that scape applies the katdal flags? 
    try:
        d = scape.DataSet(filename_or_file, baseline=baseline, nd_models=nd_models,band=band_input)
    except IOError:
        nd = scape.gaincal.NoiseDiodeModel(freq=[200,20000],temp=[20,20])
        print('Warning: Failed to load/find Noise Diode Models, setting models to 20K')
        d = scape.DataSet(filename_or_file, baseline=baseline,  nd_h_model = nd, nd_v_model=nd ,band=band_input)
    n_chan = len(d.freqs)
    
    if not channel_range is None :
        start_freq_channel = int(channel_range.split(',')[0])
        end_freq_channel = int(channel_range.split(',')[1])
        static_flags = np.tile(True, n_chan)
        static_flags[slice(start_freq_channel, end_freq_channel)] = False
    else :
        static_flags = np.tile(False, n_chan)
    
    if len(freq_mask) > 0:
        rfi_static_flags = ksl.load_frequency_mask(freq_mask, d.freqs)
        if len(rfi_static_flags) == n_chan:
            static_flags = np.logical_or(static_flags, rfi_static_flags)
        else:
            print("Warning: Frequency mask doesn't match the dataset! No static RFI channels flagged.")

    freq_channel_flagged = []
    for band in freq_channel:
        tmp_band = []
        for channel in band : 
            if not static_flags[channel] : # if not flagged
                tmp_band.append(channel)
        freq_channel_flagged.append(tmp_band)   
                
    if remove_spikes: # Automatic rfi flagging
        print("Flagging RFI spikes")
        for i in range(len(d.scans)):
            d.scans[i].data = scape.stats.remove_spikes(d.scans[i].data,axis=1,spike_width=7,outlier_sigma=5)

    print("Converting to Temperature")
    if debug:
        plt.figure(figsize=(14,5))
        plt.plot(d.nd_h_model.freq, d.nd_h_model.temp)
        plt.plot(d.nd_v_model.freq, d.nd_v_model.temp)
        plt.xlabel("f [MHz]"); plt.ylabel("T_ND [K]"); plt.ylim(0,10); plt.grid(True)
    d = d.convert_power_to_temperature(freq_width=0.0) # freq_width=0 == no averaging is done along frequency axis
    if not d is None:
        d = d.select(flagkeep='~nd_on')
        d = d.select(labelkeep='track', copy=False)
        d.average(channels_per_band=freq_channel_flagged)
        
    return d


def plot_data_el(elevation,frequency,Tsys,Tant,title='',line=42,aperture_efficiency=None,PLANE="antenna"):
    fig = plt.figure(figsize=(16,9))
    line1,=plt.plot(elevation, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(elevation, Tsys[:,0], Tsys[:,2], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(elevation, Tant[:,0], color='b'  )
    line3,=plt.plot(elevation, Tsys[:,1], marker='^', color='r', linewidth=0)
    plt.errorbar(elevation, Tsys[:,1],  Tsys[:,3], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(elevation, Tant[:,1], color='r')
    plane = "/\eta_{ap}" if ("antenna" in PLANE) else ""
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}%s$ HH'%plane,'$T_{ant}$ HH', '$T_{sys}%s$ VV'%plane,'$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s at %.1f MHz' % (title,frequency))
    plt.xlabel('Elevation (degrees)')
    lim_min = mat.apply(np.nanmin, [np.percentile(Tsys[:,0:2],10),np.percentile(Tant,10),-5.])
    data2plane = lambda data, POL: data/aperture_efficiency.eff[POL](frequency) if ("antenna" in PLANE) else data
    recLim_apEffH = data2plane(receptor_band_limit(frequency,elevation), "HH") # APH 082018
    recLim_apEffV = data2plane(receptor_band_limit(frequency,elevation), "VV") # APH 082018
    if (line > 0): # APH: Simplified line & linev compared to what's in github 082018, but did I do that? It's not essential.
        plt.hlines(line, elevation.min(), elevation.max(), colors='k')
        plt.plot(elevation, recLim_apEffH, lw=1.1,c='g',linestyle='-')
        plt.plot(elevation, recLim_apEffV, lw=1.1,c='g',linestyle='-')
        for error_margin in [0.9,1.1]:
            plt.plot(elevation, recLim_apEffH*error_margin, lw=1.1,c='g',linestyle='--')
            plt.plot(elevation, recLim_apEffV*error_margin, lw=1.1,c='g',linestyle='--')
        plt.hlines(1, elevation.min(), elevation.max(), colors='k',linestyle='--')
    lim_max = mat.apply(np.nanmax, [np.percentile(Tsys[:,0:2],90),np.percentile(Tant,90)*1.1,np.max(recLim_apEffH)*1.2,line*1.1])
    plt.ylim(lim_min,mat.roundup(lim_max,closest=5)) # APH added roundup
    plt.xlim(10,90) # APH added
    plt.grid()
    plt.ylabel('Noise equivalent temperature  (K)') # APH changed from '$T_{sys}/\eta_{ap}$
    return fig


def plot_data_freq(elevation,frequency,Tsys,Tant,title='',aperture_efficiency=None,PLANE="antenna",plot_limits=True):
    fig = plt.figure(figsize=(16,9))
    line1,=plt.plot(frequency, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(frequency, Tsys[:,0], Tsys[:,2], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(frequency, Tant[:,0], color='b'  )
    line3,=plt.plot(frequency, Tsys[:,1], marker='^', color='r',  linewidth=0)
    plt.errorbar(frequency, Tsys[:,1],  Tsys[:,3], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(frequency, Tant[:,1], color='r')
    plane = "/\eta_{ap}" if ("antenna" in PLANE) else ""
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}%s$ HH'%plane,'$T_{ant}$ HH', '$T_{sys}%s$ VV'%plane,'$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s at %.1f$^\circ$ elevation' % (title, elevation))
    plt.xlabel('Frequency (MHz)')
    if plot_limits:
        data2plane = lambda data, POL: data/aperture_efficiency.eff[POL](frequency) if ("antenna" in PLANE) else data
        recLim_apEffH = data2plane(receptor_band_limit(frequency,elevation), "HH") # APH 082018
        recLim_apEffV = data2plane(receptor_band_limit(frequency,elevation), "VV") # APH 082018
        plt.plot(frequency,recLim_apEffH,lw=1.1,color='limegreen',linestyle='-')
        plt.plot(frequency,recLim_apEffV,lw=1.1,color='limegreen',linestyle='-')
        for error_margin in [0.9,1.1]:
            plt.plot(frequency,recLim_apEffH*error_margin, lw=1.1,color='g',linestyle='--')
            plt.plot(frequency,recLim_apEffV*error_margin, lw=1.1,color='g',linestyle='--')
        plt.hlines(1, frequency.min(), frequency.max(), colors='k',linestyle='--')

    low_lim = np.nanmin( [Tsys[:,0:2], Tant] )
    low_lim = -5. # np.max((low_lim , -5.))
    high_lim = np.nanmax( [np.nanpercentile(Tsys[:,0:2],80), np.nanpercentile(Tant,80)] )
    high_lim = np.nanmax((high_lim , 46*1.1)) # APH changed to nanmin to guard against NaNs, & *1.1 like plot_el, rather than *1.3
    plt.ylim(low_lim,mat.roundup(high_lim,closest=5)) # APH added roundup
    if plot_limits and (frequency.min() < 1500): # Only for UHF & L-band
        fspec = (580,1015) if (frequency.min() < 800) else (900,1670) # APH added this to accommodate UHF band
        plt.vlines(fspec[0],low_lim,high_lim,lw=1.1,color='darkviolet',linestyle='--')
        plt.vlines(fspec[1],low_lim,high_lim,lw=1.1,color='darkviolet',linestyle='--')
        if np.min(frequency) <= 1420 :
            plt.hlines(42, np.min((frequency.min(),1420)), np.min((frequency.max(),1420)), colors='k') # APH limited upper frequency to accommodate UHF band
        if np.max(frequency) >=1420 :
            plt.hlines(46, np.max((1420,frequency.min())), np.max((frequency.max(),1420)), colors='k') # APH limited upper frequency to accommodate UHF band
    plt.grid()
    plt.ylabel('Noise equivalent temperature  (K)') # APH changed from '$T_{sys}/\eta_{ap}$
    return fig


def receptor_UHFLband_CDR_limit(freqsMHz): # MHz, at 15deg elevation, at aperture plane.
    # 275-410 m^2/K at L-band Receivers CDR, extended as 275 down through UHF-band
    Ag = np.pi*(13.5/2)**2 # Specified minimum antenna geometric collecting area
    Tsys_eta = (64*Ag)/np.interp(freqsMHz,[580,900,1670],[275.0,275.0,410.0])
    return Tsys_eta

def receptor_Lband_limit(freqsMHz): # Tsys at zenith, at feed plane
    """275-410 m^2/K at Receivers CDR"""
    Tsys = np.zeros_like(freqsMHz,dtype=float) # APH changed division below to "/float()"
    Tsys[np.array(freqsMHz < 1280)] = np.array(12 + 6+(5.5-6)/float(1280-900)*(freqsMHz-900))[np.array(freqsMHz < 1280)]
    Tsys[np.array(~(freqsMHz < 1280))] = np.array(12 + 5.5+(4-5.5)/float(1670-1280)*(freqsMHz-1280))[np.array(~(freqsMHz < 1280))]
    return Tsys

def receptor_UHFband_limit(freqsMHz): # Tsys at zenith, at feed plane
    Tsys = np.zeros_like(freqsMHz,dtype=float) # APH changed division below to "/float()"
    Tsys[np.array(freqsMHz < 900)] = np.array(8 + (12-8)/float(1015-580)*(freqsMHz-580) + 8+(7-8)/float(900-580)*(freqsMHz-580))[np.array(freqsMHz < 900)]
    Tsys[np.array(~(freqsMHz < 900))] = np.array (8 + (12-8)/float(1015-580)*(freqsMHz-580) + 7+(4-7)/float(1015-900)*(freqsMHz-900))[np.array(~(freqsMHz < 900))]
    Tsys = (8 + (12-8)/float(1015-580)*(freqsMHz-580)) + (10+(4-10)/float(1015-580)*(freqsMHz-580)) # Acceptance RS + Predicted spill+atm above 40degEl
    return Tsys


receptor_band_limits = [#Add limit lines as a tuple of functions (is_MHz_in_band, Tsys_at_zenith(MHz))
                        (lambda freqsMHz: 530<np.min(freqsMHz)<856, receptor_UHFband_limit),
                        (lambda freqsMHz: 530<np.max(freqsMHz)<=1712, receptor_UHFband_limit),
                        (lambda freqsMHz: False, receptor_UHFLband_CDR_limit), # APH 2018: Short-circuited because it needs special care to use correctly.
                       ]
def receptor_band_limit(freqsMHz, elev_deg):
    """ Generate limit lines of Tsys, for the plots. """
    Tsys_r = np.nan*freqsMHz
    for test, limit in receptor_band_limits:
        if test(freqsMHz):
            Tsys_r = limit(freqsMHz)
            break
    
    # Adjust for atmosphere vs. elevation
    if (np.any(np.isfinite(Tsys_r))):
        Tatm = lambda f_MHz, el: 275*(1-np.exp(-(0.005+0.075*(f_MHz/22230.)**4)/np.sin(el*np.pi/180))) # Approximate relation appropriate for spec limit
        Tsys_r = Tsys_r - Tatm(freqsMHz,90) + Tatm(freqsMHz,elev_deg)
    
    return Tsys_r


#### APH 02/2017 factored the code into process_b(*process_a()) which is IDENTICAL to the original process(), just re-factored slightly
def process_a(h5, ant, freq_min=0, freq_max=20e9, channel_bw=10e6, sky_radius=30, freq_mask='', debug=False):
    """ @param freq_min, freq_max: range of frequencies to process [Hz] (default 0, 20e9)
        @param channel_bw: bandwidth of each individual 'chunk' that's processed [Hz] (default 10e6)
        @param sky_radius: radius to truncate the all sky integral for T_sky model [multiple of HPBW] (default 30)
        @param freq_mask: either a pickle file of boolean flags for the dataset's frequency channels,
                              or a text file with each line frequency start,stop to indicate omitted range.
        @return: (tip_results,T_SysTemps,receiver,SpillOver,aperture_efficiency) """
    global nd_models_folder
    h5.select(reset="TB", flags="data_lost,ingest_rfi")
    h5.select(scans='track')
    ant = ant if isinstance(ant, str) else ant.name
    h5.select(ants=ant)
    filename = h5.name.split()[0].split(" ")[0] # "rdb files are different
    
    # Load the data file
    rec = h5.receivers[ant]
    num_channels = int(channel_bw/h5.channel_width) #number of channels per band
    chunks=[h5.channels[x:x+num_channels] for x in range(0, len(h5.channels), num_channels) # APH 032017 added following range selection
                                                if (h5.channel_freqs[x]>=freq_min and h5.channel_freqs[x]<=freq_max)]
    
    freq_list = np.zeros((len(chunks))) # MHz
    for j,chunk in enumerate(chunks):
        freq_list[j] = h5.channel_freqs[chunk].mean()/1e6
    print("Selecting channel data to form %f MHz Channels spanning %.f - %.f MHz"%(channel_bw/1e6, freq_list.min(),freq_list.max()) )
    band_input = rec.split('.')[0].lower() # APH different from script - if a problem implement manual override when file is loaded
    d = load_cal(h5, "%s" % (ant,), nd_models_folder, chunks,band_input=band_input, freq_mask=freq_mask, debug=debug)
    tip_rs = Tip_Results(d, filename)
    
    band = models.band(tip_rs.freqs, ant)
    aperture_efficiency = models.Ap_Eff(band=band)
    bm_floor_dBi = -6 # From as-built predicted patterns, -6dBi is ball park for MeerKAT UHF & L-band, also SKA SPFB1 & B2. TODO: Can this be estimated from aperture_efficiency?
    SpillOver = models.Spill_Temp(band=band, default=0) # Set to 0K if not known
    receiver = models.Rec_Temp(RxID=rec)
    
    T_SysTemps = [System_Temp(d, tip_rs, i) for i in range(len(d.freqs))]
    print("Integrating model sky maps using all available CPU cores (it's extremely slow on < 8 cores!)")
    with multiprocessing.Pool() as p:
        # Not using shared memory at present so no in-place updates possible.
        results = p.starmap(calc_Sky_Temp, [(tip_rs, t_st.freq, aperture_efficiency.D, bm_floor_dBi, sky_radius) for t_st in T_SysTemps])
        for t_st,res in zip(T_SysTemps, results):
            for k,v in res.items():
                t_st.__setattr__(k, v)
    
    return tip_rs,T_SysTemps,receiver,SpillOver,aperture_efficiency

def Tsys_model(tip_meta,T_SysTemps,receiver,SpillOver,aperture_efficiency,mb_crestfactor, full=True,Trxref=None):
    """ Computes the expected Tsys for the given conditions, excluding antenna reflector noise &
        any back-end noise that's not reflected in 'receiver' noise.
        NB: if 'full' is selected, the receiver noise model is added (H & V must labelled correctly!).
        
        @param tip_meta,T_SysTemps,...,aperture_efficiency: as returned by process_a
        @param mb_crestfactor: eta_mb/eta_ap = iint_mb{U}/U(0), which is robustly constant for a family of patterns.
        @param full: False to exclude the elevation-invariant contributions (default True)
        @param Trxref: thermodynamic reference temperature [K] at which the receiver noise was characterised - used in linear scaling to ambient (default None)
        @return: Tsys_model arranged as (el,freq,pol) """
    Tmodel = np.zeros((len(tip_meta.elevation),len(tip_meta.freqs),2)) # (el, freq, pol)
    for i,T_SysTemp in enumerate(T_SysTemps): # Per frequency slice
        el_freq = np.array([tip_meta.elevation, [T_SysTemp.freq]*len(tip_meta.elevation)])
        Tspill_H = SpillOver.spill["HH"](el_freq) # For beam beyond first null, this includes CMB, continuum, atmosphere & ground
        Tspill_V = SpillOver.spill["VV"](el_freq)
        
        atm_atten = np.exp(-T_SysTemp.opacity) # In the proximity of bore sight
        
        # eta_mb == the ratio of beam solid angle(theta=0..null1) relative to beam solid angle over 4pi Sr.
        #  eta_mb / eta_ap = iint_mb{U}/iint_4piSr{U}  / U(0)/iint_4piSr{U}
        #                  = iint_mb{U} / U(0)
        # So as long as the floor is more than 20dB below peak, the ratio above should be essentially unaffected.
        # For a uniformly illuminated aperture (un-tapered Airy pattern), eta_mb ~ 0.84*eta_ap
        # For a cos^2 illuminated aperture eta_mb ~ 1.3*eta_ap; essentially unchanged even if there's a "floor"
        eta_mb = mb_crestfactor * np.asarray([aperture_efficiency.eff["HH"](T_SysTemp.freq), aperture_efficiency.eff["VV"](T_SysTemp.freq)])
        
        # Adjust SpillOver for the difference between sky brightness used in EM analysis vs. present for this observation
        BrightT_sim = models.BrightT_SKA(T_SysTemp.freq/1e3, 90-tip_meta.elevation) # CMB + continuum + atmosphere along boresight direction
        BrightT_meas = lambda eta_mb: atm_atten*T_SysTemp.T_avgsky*1/eta_mb + T_SysTemp.T_atm # CMB + continuum + atmosphere
        # T_SysTemp.T_avgsky is integrated over theta=null1 .. >>10*HPBW on the [CMB+GSM]*[beam pattern] map
        Tspill_H += (1-eta_mb[0])*(BrightT_meas(eta_mb[0]) - BrightT_sim)
        Tspill_V += (1-eta_mb[1])*(BrightT_meas(eta_mb[1]) - BrightT_sim)
        
        # T_SysTemp.T_sky is integrated over theta=0 .. null1 on the [GSM]*[beam pattern] map (excludes CMB)
        Tmodel[:,i,0] = (atm_atten*T_SysTemp.T_sky + eta_mb[0]*T_SysTemp.T_atm) + Tspill_H
        Tmodel[:,i,1] = (atm_atten*T_SysTemp.T_sky + eta_mb[1]*T_SysTemp.T_atm) + Tspill_V
        
        if full: # Add elevation-independent contributions, except for Antenna Ohmic + electronics after LNA.
            dTdT = 1 if (Trxref is None) else (T0degC+tip_meta.surface_temperature)/Trxref
            Tmodel[:,i,0] += Tcmb + receiver.rec["HH"](T_SysTemp.freq)*dTdT 
            Tmodel[:,i,1] += Tcmb + receiver.rec["VV"](T_SysTemp.freq)*dTdT
        
    return Tmodel

def fit_tipping(T_sys,SpillOver,tip_meta,pol,freqs,T_rx,spill_scale=1):
    """ The 'tipping curve' is modeled using the expression below, with the free parameter just $T_{ant}$ - the Antenna noise temperature.
            $T_{sys}(el) = T_{cmb}(ra,dec) + T_{gal}(ra,dec) + T_{atm}*(1-\exp(\frac{-\ta   u_{0}}{\sin(el)})) + T_spill(el) + T_{ant} + T_{rx}$
        T_cmb + T_gal + T_atm is obtained from the T_sys.T_sky values, T_spill & T_rx must have been determined elsewhere.
        
        @return: Tant (i.e. the un-modelled residual)
    """
    Tsky = fit.PiecewisePolynomial1DFit()
    Tsky.fit(tip_meta.elevation, T_sys.T_sky)
    Trx = T_rx.rec[pol](freqs)
    def func(el):
        Tspill = SpillOver.spill[pol](np.array([[el,],[freqs]]))*spill_scale # APH (18) added *spill_scale
        return Trx + Tspill # Tsky(el) + Tatm(el)    # Add these only if using T_sys.Tsys below
    
    resid = []
    for el,t_sys in zip(tip_meta.elevation, T_sys.Tsys_sky[pol]):  # APH 032017 changed from T_sys.Tsys[pol]
        resid.append(t_sys - func(el))
    return resid


def process_b(tip_rs,T_SysTemps,receiver,SpillOver,aperture_efficiency,apscale=1.,PLANE="antenna",spec_sky=True,MeerKAT=True,mb_crestfactor=1.3):
    """ Update 'tip_rs' with derived Tsys & Tant.
        
        @param tip_rs,..,aperture_efficiency: as returned by process_a()
        @param apscale: the ratio ApArea_model/ApArea_report, which scales ap_eff(ApArea_model) to ap_eff(ApArea_report) (default 1 so for MeerKAT ApArea_report=ApArea_model which corresponds to D=13.5m)
        @param PLANE: Either "antenna" (to translate Tsys_at_waveguide to Tsys_at_waveguide/ant_eff) | "feed" (to keep raw measured Tsys_at_waveguide as is)
        @param mb_crestfactor: The ratio $eta_mb / eta_ap = iint_mb{U}/iint_4piSr{U} / U(0)/iint_4piSr{U} = iint_mb{U}/U(0)$, which is well defined and constant within ~5% for a given family of patterns (default 1.3).
        @return: (tip_results,aperture_efficiency) """
    tsys = np.zeros((len(tip_rs.elevation),len(tip_rs.freqs),4))
    tant = np.zeros((len(tip_rs.elevation),len(tip_rs.freqs),2))
    
    # APH 072018 incorporate apscale in aperture_efficiency
    if ("_HH" not in aperture_efficiency.eff.keys()): # Make shadow copies of original, un-scaled values
        aperture_efficiency.eff['_HH'] = aperture_efficiency.eff['HH']
        aperture_efficiency.eff['_VV'] = aperture_efficiency.eff['VV']
        aperture_efficiency._D = aperture_efficiency.D
    aperture_efficiency.eff['HH'] = lambda *args: apscale*aperture_efficiency.eff['_HH'](*args)
    aperture_efficiency.eff['VV'] = lambda *args: apscale*aperture_efficiency.eff['_VV'](*args)
    aperture_efficiency.D = aperture_efficiency._D/(apscale**.5)
    mb_crestfactor /= apscale
    
    # TODO: 11/2020 rewrite below to use Tsys_model(T_SysTemps,freq_list,elevation,ra,dec,surface_temperature,air_relative_humidity,receiver,SpillOver,aperture_efficiency,mb_crestfactor=1.3, full=True)
    ### APH 032017 override sky noise subtraction
    for i,T_SysTemp in enumerate(T_SysTemps):
        ### APH (1) subtract it AT THE FEED input plane -- results & un-modelled residuals look beautiful
        atm_atten = np.exp(-T_SysTemp.opacity)
        eta_mb = [mb_crestfactor*aperture_efficiency.eff['HH'](T_SysTemp.freq), mb_crestfactor*aperture_efficiency.eff['VV'](T_SysTemp.freq)] # APH reinstated this 10/2020 and deactivated all below
        # 08/2018: at the waveguide port, Tsys = eta_rad*TA + Treflectors + Tfeedandreceiver
        #   TA = iint{4pi}{((e^-tau*(Tcmb+Tgsm)+Tatm+Tgnd(theta))*sin(theta)}{dphi dtheta}
        # where GSM excludes CMB! We have a prediction for
        #   Tspillover = iint{>MB}{(e^-tau*(0*Tcmb+Tgal)+Tatm+Tgnd(theta))*sin(theta)}{dphi dtheta} # Lehmensiek confirmed 0*Tcmb on 05/08/2017
        # so that
        #   TA = iint{MB}{((e^-tau*(Tcmb+Tgsm)+Tatm)*sin(theta)}{dphi dtheta} + Tspillover + iint{>MB}{e^-tau*((Tcmb+Tgsm)-Tgal)*sin(theta)}{dphi dtheta}
        eta_rad = 0.99 # Should be >= 0.99 for MeerKAT over 500 - 20000 MHz
#         eta_mb = [T_SysTemp.eta_mb]*2 # eta_mainbeam as used in integrating Tcelest into Tspillover
        Tcelest = 1.7*(T_SysTemp.freq/1e3)**-2.75 + (0 if MeerKAT else Tcmb) # As used to generate T_spill
        atm_atten_allsky = 0.99 # Approx. iint_{2pi SR}{e^-tau(theta)}
        # 10/2018 T_avgsky is now computed with main beam blanked, so don't scale that by (1-eta_mb)
        TA_spill_HH = eta_mb[0]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky) + atm_atten_allsky*atm_atten*(np.asarray(T_SysTemp.T_avgsky)-(1-eta_mb[0])*Tcelest)
        TA_spill_VV = eta_mb[1]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky) + atm_atten_allsky*atm_atten*(np.asarray(T_SysTemp.T_avgsky)-(1-eta_mb[1])*Tcelest)
        T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-eta_rad*TA_spill_HH
        T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-eta_rad*TA_spill_VV
    
    ## Standard code resumes
    for i,T_SysTemp in enumerate(T_SysTemps):
        if spec_sky:
            Tsky_spec = Tcmb + 1.7*(T_SysTemp.freq/1e3)**-2.75 # T_SysTemp.Tsys_sky  is Tsys-Tsky (Tsky includes Tcmb & Tatm). We then add the spec sky aproxx (T_gal+Tcmb)
            Tsky_spec = Tsky_spec*atm_atten + T_SysTemp.T_atm # APH032017 added this line. T_SysTemp.Tsys_sky  is Tsys-Tsky (Tsky includes Tcmb & Tatm) @ feed input. We then add the spec sky aproxx (T_gal+Tcmb+T_atm)
            tsys[:,i,0] = np.array(T_SysTemp.Tsys_sky['HH'])+eta_rad*Tsky_spec
            tsys[:,i,1] = np.array(T_SysTemp.Tsys_sky['VV'])+eta_rad*Tsky_spec
        else:
            tsys[:,i,0] = np.array(T_SysTemp.Tsys['HH'])
            tsys[:,i,1] = np.array(T_SysTemp.Tsys['VV'])
        tsys[:,i,2] = np.array(T_SysTemp.sigma_Tsys['HH'])
        tsys[:,i,3] = np.array(T_SysTemp.sigma_Tsys['VV'])
        if spec_sky: # Increase the residual for the "uncertainty" of the sky map
            tsys[:,i,2] += T_SysTemp.sigma_Tsky
            tsys[:,i,3] += T_SysTemp.sigma_Tsky
        if ("antenna" in PLANE): # APH 072018 Raw measurement (scaled by T_ND) is Tsys_at_waveguide, /= ant_eff translates to Tsys_at_antenna
            tsys[:,i,0] /= aperture_efficiency.eff['HH'](T_SysTemp.freq)
            tsys[:,i,1] /= aperture_efficiency.eff['VV'](T_SysTemp.freq)
            tsys[:,i,2] /= aperture_efficiency.eff['HH'](T_SysTemp.freq)
            tsys[:,i,3] /= aperture_efficiency.eff['VV'](T_SysTemp.freq)
        fit_H = fit_tipping(T_SysTemp,SpillOver,tip_rs,'HH',T_SysTemp.freq,receiver)
        fit_V = fit_tipping(T_SysTemp,SpillOver,tip_rs,'VV',T_SysTemp.freq,receiver)
        tant[:,i,0] = np.array(fit_H)[:,0]
        tant[:,i,1] = np.array(fit_V)[:,0]
        #print("Debug: T_sys = %f   App_eff = %f  value = %f"%( np.array(fit_H)[22,0],aperture_efficiency.eff['HH'](d.freqs[i]),np.array(fit_H)[22,0]/aperture_efficiency.eff['HH'](d.freqs[i])))
    
    if spec_sky:
        PLANE += ",spec_sky"
    if (np.max(SpillOver.spill["HH"]([[45]*len(T_SysTemps),[T.freq for T in T_SysTemps]])) == 0): # It should be enough to test one polarisation at 45degEl
        PLANE += ",nospill"
    tip_rs.PLANE = PLANE
    tip_rs.Tsys = tsys
    tip_rs.Tant = tant
    return tip_rs, aperture_efficiency


def report_a(nice_title, tip_rs, aperture_efficiency):
    reportfilename = "%s_%s_tipping_curve.pdf"%(tip_rs.filename.split("/")[-1].split(".")[0],tip_rs.ant)
    pp = PdfPages(reportfilename)
    print('http://.../'+reportfilename)
    return pp

def report_b(nice_title, tip_rs, aperture_efficiency, pp, cartesian=True):
    fig = tip_rs.sky_fig(annotate_every=10, cartesian=cartesian) # Plot every 10th point with elevation label
    fig.savefig(pp,format='pdf')

def report_c(nice_title, tip_rs, aperture_efficiency, pp, select_freq=None, plot_limits=True):
    length = len(tip_rs.elevation)
    select_freq = select_freq if select_freq else tip_rs.freqs
    iselect_freq = [(np.abs(tip_rs.freqs-freq)).argmin() for freq in select_freq] # Indices into freq_list
    
    tsys_ = np.take(tip_rs.Tsys[0:length,:,0:2], iselect_freq, axis=1) # sigmaH,V are in axis=2@[2:]
    tant_ = np.take(tip_rs.Tant[0:length], iselect_freq, axis=1)
    ylim = (mat.roundup(np.nanpercentile(tant_,2), closest=5)-5, mat.roundup(np.nanpercentile(tsys_,98), closest=5)) # 5 .. 95 for robustness to RFI
    
    for i,freq in zip(iselect_freq,select_freq):
        lineval = 46 if freq > 1420 else 42 #UHF & L-band spec limit
        lineval = lineval if plot_limits else -1
        fig = plot_data_el(tip_rs.elevation,tip_rs.freqs[i],tip_rs.Tsys[0:length,i,:],tip_rs.Tant[0:length,i,:],
                           title=nice_title,line=lineval,aperture_efficiency=aperture_efficiency,PLANE=tip_rs.PLANE)
        plt.ylim(*ylim)
        fig.savefig(pp,format='pdf')

def report_d(nice_title, tip_rs, aperture_efficiency, pp, select_freq=None, select_el=[15,45,90], plot_limits=True):
    ## APH added this to exclude unspecified frequencies at either edge
    freqlim = select_freq if select_freq else tip_rs.freqs
    freqlim = [np.min(freqlim)-20, np.max(freqlim)+20]
    f_low, f_high = abs(tip_rs.freqs-freqlim[0]).argmin(),abs(tip_rs.freqs-freqlim[-1]).argmin()
    _fspec = range(min(f_low,f_high),max(f_low,f_high)+1)
    iselect_el = [(np.abs(tip_rs.elevation-el)).argmin() for el in select_el] # Indices into elevation
    
    tsys_ = np.take(tip_rs.Tsys[:,_fspec,0:2], iselect_el, axis=0) # sigmaH,V are in axis=2@[2:]
    tant_ = np.take(tip_rs.Tant[:,_fspec], iselect_el, axis=0)
    ylim = (mat.roundup(np.nanpercentile(tant_,2), closest=5)-5, mat.roundup(np.nanpercentile(tsys_,98), closest=5))
    
    for i in iselect_el :
        fig = plot_data_freq(tip_rs.elevation[i],tip_rs.freqs[_fspec],tip_rs.Tsys[i,_fspec,:],tip_rs.Tant[i,_fspec,:],
                             title=nice_title,aperture_efficiency=aperture_efficiency,PLANE=tip_rs.PLANE,plot_limits=plot_limits)
        plt.ylim(*ylim)
        fig.savefig(pp,format='pdf')
    return f_low, f_high, _fspec

def report_cd(nice_title, tip_rs, aperture_efficiency, pp): # APH 072018 add elevation vs frequency contour plots to combine c & d
    T_min = 10 # TODO: lowest conceivable system noise used to mask wild low values
    for i,p in enumerate("HV"):
        fig = plt.figure(figsize=(16,9))
        _t_ = tip_rs.Tsys[:,:,i]
        T_max = mat.roundup(np.nanpercentile(_t_,95), closest=5) + 5 # +5 softens saturation in case there's actually little RFI
        plt.contourf(tip_rs.freqs, tip_rs.elevation, _t_, levels=np.linspace(T_min,T_max,10)); plt.colorbar(fraction=0.05)
        plt.xlabel("f [MHz]"); plt.ylabel("Elevation angle [$^\circ$]")
        plt.title("%s, $T_{sys}%s$ %s"%(nice_title,("" if ("feed" in tip_rs.PLANE) else "/\eta_{ap}"),p*2))
        fig.savefig(pp,format='pdf')

def report_e(nice_title, tip_rs, aperture_efficiency, pp):
    fig = plt.figure(None,figsize = (8.27, 11.69))
    text =r"""
    The 'tipping curve' is $T_{\mathrm{sys}}(\mathrm{el})$, derived from measurements that are scaled relative to the noise diode
    calibration tables, which are referred to the receiver's input port:
    $T_{\mathrm{sys}}(\mathrm{el}) = P_{\mathrm{cal OFF}}(\mathrm{el}) \times \frac{T_{\mathrm{cal}}}{P_{\mathrm{cal ON}}(\mathrm{el})-P_{\mathrm{cal OFF}}(\mathrm{el})}$
    
    System noise at the receiver input port is modelled as follows:
    $\mathcal{T}_{\mathrm{sys}}(\mathrm{el}) = \eta_{\mathrm{rad}}\frac{\iint_{4\pi\mathrm{Sr}}{\left(T_{\mathrm{celest}}e^{-\tau(\mathrm{el})}+T_{\mathrm{atm}}(1-e^{-\tau(\mathrm{el})})+T_{\mathrm{ground}}\right)P(\Omega-(\mathrm{az,el}))}{d\Omega}}{\iint_{4\pi\mathrm{Sr}}{P(\Omega)}{d\Omega}} + T_{\mathrm{ant}} + T_{\mathrm{rx}}$
    
    Here $\eta_{\mathrm{rad}}$ is the antenna's radiation efficiency, $T_{\mathrm{celest}}$ is the brightness temperature distributed across
    the celestial sphere, $\tau(\mathrm{el})$ is the atmospheric opacity, $T_{\mathrm{atm}}$ is the atmosphere's mean radiating temperature,
    $T_{\mathrm{ground}}$ is the ground's radiating temperature and $P(\Omega-(\mathrm{az,el}))$ is the antenna's total (co & cross-polar)
    radiation pattern as oriented along the direction of the antenna bore sight $(\mathrm{az,el})$.
    $T_{\mathrm{ant}}$ is the effective contribution from the antenna's reflecting surfaces and $T_{\mathrm{rx}}$ is the cascaded
    receiver equivalent noise temperature. All variables are also functions of frequency.
    
    The model may be simplified by separating the integral into main beam and sidelobes and introducing
    $T_{\mathrm{spill}}(\mathrm{el}) = \frac{\iint_{\mathrm{Sidelobes}}{\left(T_{\mathrm{celest}}e^{-\tau(\mathrm{el})}+T_{\mathrm{atm}}(1-e^{-\tau(\mathrm{el})})+T_{\mathrm{ground}}\right)P(\Omega-(\mathrm{az,el}))}{d\Omega}}{\iint_{4\pi\mathrm{Sr}}{P(\Omega)}{d\Omega}}$
    Further assuming the main beam is never pointed towards the ground, the model relation becomes
    $\mathcal{T}_{\mathrm{sys}}(\mathrm{el}) = \eta_{\mathrm{rad}}\left(\frac{\iint_{\mathrm{MainBeam}}{\left(T_{\mathrm{celest}}e^{-\tau(\mathrm{el})}+T_{\mathrm{atm}}(1-e^{-\tau(\mathrm{el})})\right)P(\Omega-(\mathrm{az,el}))}{d\Omega}}{\iint_{4\pi\mathrm{Sr}}{P(\Omega)}{d\Omega}} + T_{\mathrm{spill}}(\mathrm{el})\right) + T_{\mathrm{ant}} + T_{\mathrm{rx}}$
    
    
    The results contained in this report are derived from the above as follows:
      1. The residual remaining after subtracting known & estimated values for the receiver, spillover
      and sky noise contributions, is computed and labelled as $T_{\mathrm{ant}}$. This provides an upper bound for the
      noise contributed by the antenna's reflector system.
      """
    if ("spec_sky" in tip_rs.PLANE):
        text += r"""2. The measured $T_{\mathrm{sys}}(\mathrm{el})$ is adjusted for the difference between the actual sky brightness and the sky model
        prescribed in the MeerKAT specifications, by adding
        $\Delta T_{celest} = e^{-\tau(el)} ( T_{gal} - T_{gsm} )$
      """
    
    if ("antenna" in tip_rs.PLANE):
        text += r"""The resulting $\frac{T_{\mathrm{sys}}(\mathrm{el})}{\eta_{_{\mathrm{ap}}}}$ may be compared directly to the specification at the ideal antenna aperture plane."""
    else:
        text += r"""The resulting $T_{\mathrm{sys}}(\mathrm{el})$ may be compared directly to the specification at the receiver's input plane."""
      
    text += r"""
    
    The following known and estimated values are employed:
      * $\eta_{\mathrm{rad}}$ is taken to be 0.99
      * $P(\Omega)$ over the main beam is the far field pattern of a circular aperture illuminated with a parabolic taper
    """
    text += r"""  * $\eta_{\mathrm{ap}}$ has been estimated using full wave EM analysis of the as-built optics & scaled to represent $D=%.2f$m.
    """%(aperture_efficiency.D)
    text += r"""  * $\tau_{0}$, the zenith opacity, is calculated according to ITU-R P.676-9."""
    text += r"""
      * $T_{\mathrm{atm}}$ is calculated as per Ippolito 1989 (eq 6.8-6)
    """
    if ("nospill" in tip_rs.PLANE):
        text += r"""  * $T_{\mathrm{spill}}$ is not explicitly modelled and therefore remains lumped in $T_\mathrm{ant}$"""
    else:
        text += r"""  * $T_{\mathrm{spill}}$ has been estimated from full wave EM analysis of the as-built optics, with a nominal sky noise
      model which excludes $T_{\mathrm{cmb}}$!"""
    text += r"""
      * $T_{\mathrm{celest}}=T_{\mathrm{cmb}}+T_{\mathrm{gal}}$ with $T_{cmb} = 2.73$ and $T_{gal} = 20\left(\frac{f}{408MHz}\right)^{-2.75}$.
      * $T_{\mathrm{gsm}}$ is obtained from de Oliveira-Costa's 2008 Global Sky model convolved with $P(\Omega)$.
      * $T_{\mathrm{rx}}$ is a measured result delivered with the receivers' ATP
    
    The green solid lines in the figures reflect the design limits at time of CDR, viz. 275 - 410 $m^2/K$ (array)
     across UHF - L band, with the broken green lines indicating a $\pm10\%$ margin for reference."""

    params = {'font.size': 10}
    plt.rcParams.update(params)
    ax = fig.add_subplot(111)
    anchored_text = AnchoredText(text, loc=2, frameon=False)
    ax.add_artist(anchored_text)
    ax.set_axis_off()
    plt.subplots_adjust(top=0.99,bottom=0,right=0.975,left=0.01)
    fig.savefig(pp,format='pdf')
    pp.close()
    plt.close(fig)

    
def process(dataset, ant, freq_min=0, freq_max=20e9, freq_mask='', apscale=1.,PLANE="antenna",spec_sky=True,MeerKAT=True):
    """ Load and process the tipping curve data file, to generate the information from which a report can be compiled.
        Note that the noise diode models are stored either in the dataset itself, or in the global::nd_models_folder,
        where model filenames are like 'rx.%(band).%(serialno).%(pol).csv' (e.g. 'rx.l.4.h.csv').

        @param dataset: an open katdal.Dataset
        @param freq_min, freq_max: range of frequencies to process [Hz] (default 0, 20e9)
        @param channel_bw: bandwidth of each individual 'chunk' that's processed [Hz] (default 10e6)
        @param sky_radius: radius to truncate the all sky integral for T_sky model [multiple of HPBW] (default 30)
        @param freq_mask: either a pickle file of boolean flags for the dataset's frequency channels,
                              or a text file with each line frequency start,stop to indicate omitted range.
        @param PLANE: Either "antenna" (to translate Tsys_at_waveguide to Tsys_at_waveguide/ant_eff) | "feed" (to keep raw measured Tsys_at_waveguide as is)
        @param spec_sky: True to subtract the model sky and add the sy noise from the model prescribed in the MeerKAT requirements.
        @param MeerKAT: True if the predicted spillover includes CMB, otherwise False (default True - for MeerKAT 2015)
        @return: (tip_results, aperture_efficiency) """
    return process_b(*process_a(dataset, ant, freq_min, freq_max, freq_mask=freq_mask), apscale=apscale,PLANE=PLANE,spec_sky=spec_sky,MeerKAT=MeerKAT)


def report(tip_rs, aperture_efficiency, select_freq=None, select_el=[15,45,90], cartesian=True, plot_limits=False):
    """ Generate the standard tipping curve report.
        @param tip_rs, aperture_efficiency: as generated by `process()`
        @param select_freq: a list of subset of frequencies [MHz] to explicitly report on, or None for all (default None).
        @param select_el: a list of subset of elevation angles [deg] to explicitly report on (default [15,45,90]).
        @param cartesian: True to show the sky tracks against Galactic map in Cartesian projection, False for Orthographic.
        @param plot_limits: True to plot the MeerKAT specification lines (default False) """
    nice_title = " %s  Ant=%s"%(tip_rs.filename.split('/')[-1], tip_rs.ant)
    pp = report_a(nice_title, tip_rs, aperture_efficiency)
    report_b(nice_title, tip_rs, aperture_efficiency, pp, cartesian)
    report_c(nice_title, tip_rs, aperture_efficiency, pp, select_freq, plot_limits=plot_limits)
    f_low, f_high, _fspec = report_d(nice_title, tip_rs, aperture_efficiency, pp, select_freq, select_el=select_el, plot_limits=plot_limits)
    report_cd(nice_title, tip_rs, aperture_efficiency, pp)
    report_e(nice_title, tip_rs, aperture_efficiency, pp)


if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser(usage="%prog [opts] <dataset filename>",
                                   description="Process a 'tip curve' dataset and generate a standard report.")
    parser.add_option("-a", "--ants", dest="ants", help="The name(s) of the specific antenna(s) in the dataset to process, or 'all'")
    parser.add_option("--select-freq", dest="select_freq", help="Comma-separated list of frequencies, including limits in MHz, e.g. '900,1000,1400,1600'")
    parser.add_option("--freq-mask", dest="freq_mask", default="", help="Filename for frequency mask, either text or pickle format.")
    parser.add_option("--hackL", action='store_true',
                      help="Open the data file as if it was recorded with an L-band digitiser sampling in the 1st Nyquist zone.")
    opts, args = parser.parse_args()
    
    filename = args[0]
    ds = util.open_dataset(filename, hackedL=opts.hackL)
    if opts.hackedL: # Minimize unused horizontal space in figures
        ds.select(freqrange=(330e6,856.5e6))
    if (opts.ant=="all"):
        ants = [a.name for a in ds.ants]
    else:
        ants = opts.ants.split(',')
    select_freq = map(float, opts.select_freq.split(","))
    for ant in ants:
        ds.select(ants=ant)
        ru = process(filename, ant, min(select_freq)*1e6, max(select_freq)*1e6, freq_mask=opts.freq_mask, PLANE="antenna",spec_sky=True)
        report(*ru, select_freq=select_freq, plot_limits=True)
