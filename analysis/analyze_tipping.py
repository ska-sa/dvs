#!/usr/bin/python
"""
    Formalization of earlier http://kat-imager.kat.ac.za:8888/notebooks/SysEng/Tipping%20curve%20for%20SKA%20prototype%20report-UHF%20Shroud%20tests.ipynb
    Typical use 1:
        python analyze_tipping.py /data/132598363.h5 m008 900,1200,1700
        
    Typical use 2:
        import analyze_tipping as tip
        h5 = tip.open_dataset("/data/132598363.h5")
        ru = tip.process(h5, "m008", 900,1700, channel_mask='', PLANE="antenna",spec_sky=True)
        tip.report("m008", h5.name.split()[0].split(" ")[0], *ru, select_freq=[900,1200,1700])
        
@author aph@ska.ac.za
"""
import numpy as np
import matplotlib.pyplot as plt

import pickle
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import katdal
import scape
import scikits.fitting as fit
import healpy as hp
import ephem
from astropy import units as u
from astropy.coordinates import SkyCoord
import time
git_info = lambda: "analyze_tipping.py 11/03/2020"
from matplotlib.offsetbox import AnchoredText
import katselib

import models
from models import _c_, Tcmb


def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into the interval -*period* / 2 ... *period* / 2."""
    return (angle + 0.5 * period) % period - 0.5 * period


class Sky_temp:
    #import gsm # APH removed these 03/2017
    #import healpy as hp
    #from astropy import units as u
    #from astropy.coordinates import SkyCoord
    """
       T_sky = T_cont + T_cmb  from the global sky model
       Read in  file, and provide a method of passing back the Tsky temp a at a position
    """
    def __init__(self,nu=1828.0,path="/var/kat/archive/data/models/gsm",diameter=13.965,smooth=0): # APH 032017 changed smooth to numeric scale factor
        """ Load The Tsky data from an inputfile in FITS format and scale to frequency
        This takes in 1 parameter:
        nu (MHz) center frequency
        
        """
        # APH 032017 refactored this function significantly
        #hpbw = 1.23*(3e8/(nu*1e6))/diameter # APH 032017 based on EMSS full wave UHF-band; 1.22 for L-band, which was measured on TauA (L. Schwardt, 11/2014)
        self.hpbw = 1.27*(_c_/(nu*1e6))/diameter # APH 042017 based on parabolic tapered beam used below
        
#         # APH added these for (21)
#         # Tapered Beam pattern: m=1 for uniform, m=2 for parabolic taper [VLA memo 327], closely matches CAD results to 2nd null. 
#         E = lambda theta,a,wl,m: 2*np.pi*a**2/wl*(1+np.cos(theta))*sp.special.jn(m,2*np.pi*a/wl*np.sin(theta))/(2*np.pi*a/wl*np.sin(theta))**m
#         A = lambda theta,D,wl,m: E(theta+1e-8,D/2.,wl,m)**2 / E(1e-8,D/2.,wl,m)**2
#         self.B = lambda theta: A(theta,diameter,300./nu,2) # Beam function now used by Tsky
#         # Pre-computed by numerical methods. eta_beam = SA(-null..null)/SA(4pi)
#         self.SA = 1.075 * np.pi/4./np.log(2)*hpbw**2 # Just a factorisation of SA(4pi), the solid angle of B over 4pi Sr.
#         # TODO: Finalise choosing between 1st & 2nd null - results don't seem sensitive
#         self.null, self.eta_beam = 1.29*hpbw, 0.989 # 1st null. Was 1.367, 0.914 < 072018
#         #self.null, self.eta_beam = 2.11*hpbw, 0.999 # 2nd null. Was 2.252, 0.927 < 072018
# #        self.null = 2.252*hpbw; self.SA = 0.927 * np.pi/4./np.log(2)*hpbw**2; Sky_temp.eta_beam = 0.927  #### Back to analyze_tipping (4).py
        self.null = 1.29*self.hpbw # First null of ^2 beam
        
        #self.freq_map = gsm.get_freq(nu,path)
        #self.freq_map += Tcmb  # CMB tempreture not included in de Oliveira-Costa's GSM
        #if smooth > 0: # APH changed 03/2017 to implement correct smoothing & permit arbitrary smoothing!
        #    native_fwhm = 56*np.pi/60./180. # 'gsm' has 56arcmin resolution
        #    if (smooth*hpbw > native_fwhm): # APH 032017 added this check, can't sharpen the map.
        #        self.freq_map = hp.sphtfunc.smoothing(self.freq_map,fwhm=((smooth*hpbw)**2-native_fwhm**2)**.5)
        self._freq_map_, self.resolution = models.get_gsm(nu, smooth*self.hpbw)

        self.nu = nu
        
    def freq_map(self): # APH added 092018 instead of object attribute
        return self._freq_map_ + Tcmb  # CMB temperature not included in the GSM maps - do not modify the cached map, and don't cache this modified version!

    def Tsky(self,ra,dec,integrate=True,bm_floor=0): # APH 042017 added integrate=True to replace earlier
        """given RA/Dec in Degrees  return the value of the spot
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
        """ plot_sky plots the sky temperature and overlays pointing centers as red dots
        The sky tempreture is the data that was loaded when the class was initiated.
        plot_sky takes in 3 optional parameters:
                ra,dec  are list/1D-array like values of right ascension and declination
                pointlabels: list of *(galactic lon, galactic lat, string label)* to plot with pointstyle
                fig: figure to re-use, default None
        returns matplotlib figure object that the plot is assosated with.
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
            plt.title("Continuum Brightness including CMB, at %.fMHz"%self.nu)
            #hp.graticule()
        
        if cartesian or (date is None): # Plot pointing coordinates in Cartesian projection
            c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            l = np.degrees(angle_wrap(-c.galactic.l.radian % (np.pi*2)) )
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


class System_Temp:
    """Extract tipping curve data points and surface temperature."""
    # APH removed freqs=1822 - just use d.freqs[freq_index]; & added sort_ind=None below - previously it was grabbed out of global namespace.
    def __init__(self,d,freq_index=0,sort_ind=None,elevation=None,ra=None,dec=None,timestamps=None,surface_temperature=23.0,air_relative_humidity=0.23):#d, nu, pol
        """ First extract total power in each scan (both mean and standard deviation) """
        self.units = d.data_unit
        
        self.name = d.antenna.name
        self.filename = d.filename
        self.elevation =  {}
        self.Tsys = {}
        self.sigma_Tsys = {}
        self.Tsys_sky = {}
        self.observer_wgs84 = dict(zip(['lat','lon','elev'], map(float, d.antenna.position_wgs84))) # rad,rad,m. float because the native ephem.Angle isn't pickle-able.
        self.height = self.observer_wgs84['elev']
        self.pressure =  np.mean([line[1] for line in d.enviro['pressure'] ])
        self.air_relative_humidity = air_relative_humidity
        # Sort data in the order of ascending elevation
        valid_el = (elevation >= 10)
        self.elevation =  elevation[valid_el]
        self.ra = ra[valid_el]
        self.dec = dec[valid_el]
        self.timestamps = timestamps[valid_el] # APH added 052017 -> 102018
        self.surface_temperature = surface_temperature# Extract surface temperature from weather data
        self.freq = d.freqs[freq_index]  #MHz Centre frequency of observation APH changed [0] to [freq_index] on 03/2017
        ### APH begin 032017 added the default opacity & T_atm calculation here.
        T_atm = 1.12 * (273.15 + surface_temperature) - 50.0 # APH 02/2017 changed comment: Ippolito 1989 eq 6.8-6, quoting Wulfsberg 1964
        tau = models.calc_atmospheric_opacity(surface_temperature,air_relative_humidity,self.pressure,self.height/1000.,self.freq/1000.)
        self.opacity = tau/np.sin(np.radians(self.elevation))
        # APH 03/2017 added the following two corrections, but only significant above 10 GHz and below 10deg elevation, so leave out.
        #a0 = 1/np.sin(np.radians(self.elevation))
        #self.opacity = tau*(a0 - 2.0/6370.95*a0*(a0**2-1)) # (19) Correction of airmass function for spherical atmosphere (not planar) as per Han & Westwater 2000
        #T_atm = 266.5 + 0.72*surface_temperature # Ballpark from Tabel VII of Han & Westwater 2000, but how does that extrapolate down from 20-30 GHz?
        self.T_atm = T_atm * (1 - np.exp(-self.opacity))
        ### APH end
        for pol in ['HH','VV']:
            power_stats = [scape.stats.mu_sigma(s.pol(pol)[:,freq_index]) for s in d.scans]
            tipping_mu, tipping_sigma = np.array([s[0] for s in power_stats]), np.array([s[1] for s in power_stats])
            tipping_mu, tipping_sigma = tipping_mu[sort_ind], tipping_sigma[sort_ind]
            self.Tsys[pol] = tipping_mu[valid_el]
            self.sigma_Tsys[pol] = tipping_sigma[valid_el]
            self.Tsys_sky[pol] = [] # APH 032017 Just initialise, calculated in process_() to correctly transfer to feed input plane.
#            for ra,dec in zip(self.ra,self.dec):
#                self.Tsys_sky[pol].append(tipping_mu[val_el]-T_sky(ra,dec))# APH 032017 In process(), use apeff to correctly transfer to feed input plane.
    
    def calc_Sky_Temp(self, D, bm_floor_dBi=-6, radius=30): # APH created 10/2018, exposed floor_dBi & radius [multiple of HPBW]
        iso = (D*np.pi/(_c_/1e6/self.freq))**-2 # Isotropic antenna gain relative to bore sight gain of the reflector antenna
        bm_floor = iso * 10**(bm_floor_dBi/10.)
        ## APH moved below ~5 lines from inside '__init__()' and its 'for pol... loop' since sky model is not polarisation dependent
        T_skytemp = Sky_temp(nu=self.freq, diameter=D)
        if not hasattr(self, "T_sky"):
            self.T_sky = T_skytemp.Tsky(self.ra,self.dec,bm_floor=bm_floor)
            for v in self.sigma_Tsys.values(): # APH 032017 added to update sigma_Tsys
                v += 0.1*np.abs(np.asarray([0]+np.diff(self.T_sky,1).tolist()))
        TmpSky = fit.PiecewisePolynomial1DFit() # APH 11/2016 changed from scape.fitting.PiecewisePolynomial1DFit()
        TmpSky.fit(self.elevation, self.T_sky)
        self.Tsky = TmpSky
        self.T_avgsky = T_skytemp.Tavgsky(date=[time.gmtime(t)[:6] for t in self.timestamps], ra=self.ra,dec=self.dec, radius=radius, blank_mb=True ,bm_floor=bm_floor, **self.observer_wgs84) # Blank to first null of the parabolic pattern
        self.eta_mb = T_skytemp.eta_mb # The ratio of solid angle of beam (to first null) relative to 4pi Sr beam solid angle for beam as used to compute T_avgsky
    
    def sky_fig(self,freq=1328,cartesian=True):
        T_skytemp = Sky_temp(freq)
        fig = T_skytemp.plot_sky(self.ra,self.dec,date=time.gmtime(self.timestamps[0])[:6], cartesian=cartesian, **self.observer_wgs84) # APH added the next line to annotate plot with elevation angles
        return T_skytemp.plot_sky(self.ra[::10],self.dec[::10], pointstyle='mo', pointlabels=['El %.f'%e for e in self.elevation[::10]], fig=fig, cartesian=cartesian, **self.observer_wgs84)

    def __iter__(self):
        return self

    def next(self):
        i = -1
        while True:
            i = i + 1
            if not self.ra[i]:raise StopIteration
            yield i,self.ra[i],self.dec[i],self.elevation[i]
            
            

def remove_rfi(d,width=3,sigma=5,axis=1):
    for i in range(len(d.scans)):
        d.scans[i].data = scape.stats.remove_spikes(d.scans[i].data,axis=axis,spike_width=width,outlier_sigma=sigma)
    return d

# APH below changed filename to filename_or_file to clarify one could pass an open katdal dataset 
def load_cal(filename_or_file, baseline, nd_models, freq_channel=None,channel_mask='',channel_range=None,band_input=None, debug=False):
    """ Load the dataset into memory """
    print('Loading noise diode models')
    
    try:
        d = scape.DataSet(filename_or_file, baseline=baseline, nd_models=nd_models,band=band_input)
    except IOError:
        nd = scape.gaincal.NoiseDiodeModel(freq=[200,2000],temp=[20,20]) # APH extended 800 down to 200
        warnings.warn('Warning: Failed to load/find Noise Diode Models, setting models to 20K ')
        print('Warning: Failed to load/find Noise Diode Models, setting models to 20K ')
        d = scape.DataSet(filename_or_file, baseline=baseline,  nd_h_model = nd, nd_v_model=nd ,band=band_input)
    n_chan = len(d.freqs)
    
    if not channel_range is None :
        start_freq_channel = int(channel_range.split(',')[0])
        end_freq_channel = int(channel_range.split(',')[1])
        edge = np.tile(True, n_chan)
        edge[slice(start_freq_channel, end_freq_channel)] = False
    else :
        edge = np.tile(False, n_chan)
    
    if len(channel_mask)>0:
        if isinstance(channel_mask, str): # load static flags if pickle file is given
            pickle_file = open(channel_mask) 
            rfi_static_flags = pickle.load(pickle_file)
            pickle_file.close()
        else:
            rfi_static_flags = channel_mask
    else: # No channels flagged
        rfi_static_flags = np.tile(False, n_chan)

    static_flags = np.logical_or(edge,rfi_static_flags)

    #d = d.select(freqkeep=~static_flags)
    freq_channel_flagged = []
    for band in freq_channel:
        tmp_band = []
        for channel in band : 
            if not static_flags[channel] : # if not flagged
                tmp_band.append(channel)
        #if len(tmp_band) > 0 :
        freq_channel_flagged.append(tmp_band)   
                
    #if not freq_channel is None :
    #    d = d.select(freqkeep=freq_channel)
    #print "Flagging RFI"
    #sd = remove_rfi(d,width=7,sigma=5)  # rfi flaging Needed ?
    print("Converting to Temperature")
#    print "Plotting the number of channels in each band of the list of lists freq_channel_flagged will be usefull "
    if debug:
        plt.figure(figsize=(14,5))
        plt.plot(d.nd_h_model.freq, d.nd_h_model.temp)
        plt.plot(d.nd_v_model.freq, d.nd_v_model.temp)
        plt.xlabel("f [Hz]"); plt.ylabel("T_ND [K]"); plt.ylim(0,10); plt.grid(True)
    d = d.convert_power_to_temperature(freq_width=0.0)
    if not d is None:
        d = d.select(flagkeep='~nd_on')
        d = d.select(labelkeep='track', copy=False)
        d.average(channels_per_band=freq_channel_flagged)
        
    return d



def chisq_pear(fit,Tsys):
    fit = np.array(fit)
    return np.sum((Tsys-fit)**2/fit)


def fit_tipping(T_sys,SpillOver,pol,freqs,T_rx,fixopacity=False,spill_scale=1):
    """The 'tipping curve' is fitted using the expression below, with the free parameters of $T_{ant}$ and $\tau_{0}$
        the Antenna tempreture and the atmospheric opacity. All the varables are also functions of frequency .
        $T_{sys}(el) = T_{cmb}(ra,dec) + T_{gal}(ra,dec) + T_{atm}*(1-\exp(\frac{-\ta   u_{0}}{\sin(el)})) + T_spill(el) + T_{ant} + T_{rx}$
        We will fit the opacity and $T_{ant}$.s
        T_cmb + T_gal is obtained from the T_sys.Tsky() function
        if fixopacity is set to true then $\tau_{0}$ is set to 0.01078 (Van Zee et al.,1997) this means that $T_{ant}$ becomes
        The excess tempreture since the other components are known. When fixopacity is not True then it is fitted and T_ant
        is assumed to be constant with elevation
    """
    ## APH added comment 02/2017: the code below does not in fact FIT anything, it simply computes the model, and returns the residual. And the TODO below is spurious.
#TODO Set this up to take in RA,dec not el to avoid scp problems
    #T_atm = 1.12 * (273.15 + T_sys.surface_temperature) - 50.0 # APH 02/2017 removed, not necessary
    returntext = []
#    if not fixopacity: # APH 032017 removed, not necessary
#        #print T_sys.surface_temperature,T_sys.air_relative_humidity, T_sys.pressure, T_sys.height, freqs
#        tau = calc_atmospheric_opacity(T_sys.surface_temperature,T_sys.air_relative_humidity, T_sys.pressure, T_sys.height/1000., freqs/1000.)
#        # Height in meters above sea level, frequency in GHz.
#    else:   
#        tau = 0.01078
#    print("atmospheric_opacity = %f  at  %f MHz"%(tau,freqs))
    tip = fit.nonlinlstsq.NonLinearLeastSquaresFit(None, [0, 0.00]) # APH changed 11/2016 from scape.fitting.NonLinearLeastSquaresFit(None, [0, 0.00]) # nonsense Vars
    def know_quant(x):
        rx = T_rx.rec[pol](freqs)
        #sky = T_sys.Tsky(x) # APH 032017 unnecessary complication, rather compute residual from Tsys_sky
        spill = SpillOver.spill[pol](np.array([[x,],[freqs]]))*spill_scale # APH (18) added *spill_scale
#        atm = T_atm * (1 - np.exp(-tau / np.sin(np.radians(x)))) # APH 032017 no longer used
        #print "Rec %3.1f + Sky %3.1f + Spill %3.1f + Atm %3.1f = %3.1f" % (rx ,sky , spill , atm,rx+sky+spill+atm)
        return rx + spill # sky + spill + atm    # APH 032017 dropped sky+atm, this is residual Tsys at feed input reference plane
    
    func = know_quant
    fit_func = []
#    returntext.append('Not fitting Opacity assuming a value if %f , $T_{ant}$ is the residual of of model data. ' % (tau,))
    for el,t_sys in zip(T_sys.elevation, T_sys.Tsys_sky[pol]):  # APH 032017 changed from T_sys.Tsys[pol]
        fit_func.append(t_sys - func(el))
        #print "T_sys %3.1f - T_other %3.1f " %(t_sys,func(el))
    chisq =0.0# nonsense Vars
    return {'params': tip.params,'fit':fit_func,'scatter':None,'chisq':chisq,'text':returntext} # APH 032017 'scatter'is unused and meaningless, so changed 'scatter'from (T_sys.Tsys[pol]-fit_func)


def plot_data_el(Tsys,Tant,title='',units='K',line=42,aperture_efficiency=None,PLANE="antenna",frequency=1420):
    fig = plt.figure(figsize=(16,9))
    elevation = Tsys[:,2]
    line1,=plt.plot(elevation, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(elevation, Tsys[:,0], Tsys[:,3], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(elevation, Tant[:,0], color='b'  )
    line3,=plt.plot(elevation, Tsys[:,1], marker='^', color='r', linewidth=0)
    plt.errorbar(elevation, Tsys[:,1],  Tsys[:,4], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(elevation, Tant[:,1], color='r')
    plane = "/\eta_{ap}" if ("antenna" in PLANE) else ""
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}%s$ HH'%plane,'$T_{ant}$ HH', '$T_{sys}%s$ VV'%plane,'$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s' % (title))
    plt.xlabel('Elevation (degrees)')
    lim_min = r_lim([np.percentile(Tsys[:,0:2],10),np.percentile(Tant[:,0:2],10),-5.])
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
    lim_max = r_lim([np.percentile(Tsys[:,0:2],90),np.percentile(Tant[:,0:2],90)*1.1,np.max(recLim_apEffH)*1.2,line*1.1],np.max)
    plt.ylim(lim_min,roundup(lim_max,closest=5)) # APH added roundup
    plt.xlim(10,90) # APH added
    plt.grid()
    plt.ylabel('Noise equivalent temperature  (K)') # APH changed from '$T_{sys}/\eta_{ap}$
    return fig

def roundup(x, closest=1): # Round up to an integer multiple of 'closet' - APH added
    if closest > 0:
        x = int(x/float(closest)+0.5)*float(closest)
    return x

def r_lim(dataf,func=np.nanmin): # APH changed to nanmin to guard against NaNs
    """ Returns the func of the data , not used on nans"""
    dataf = np.array(dataf)
    index = ~np.isnan(dataf)
    return func(dataf[index,...]) if (index.any()) else np.nan # APH added to guard against NaNs 

def receptor_band_limit(frequency,elevation):
    # APH factored out elevation from individual functions, into this function
    Tatm = lambda f_MHz, el: 275*(1-np.exp(-(0.005+0.075*(f_MHz/22230.)**4)/np.sin(el*np.pi/180))) # Approximate relation appropriate for spec limit
    
    if False and (frequency.max() < 1800): # APH added this to short-circuit the previous "bottom-up" limits still coded below
        return_array = receptor_UHFLband_CDR_limit(frequency)
        return_array = return_array - Tatm(frequency,15) + Tatm(frequency,elevation)
        return return_array # CAUTION: this is at aperture plane, the others are at feed plane! Also, more conservative atmospheric adjustment than previously
    
    if (frequency.min() < 800):
        return_array = receptor_UHFband_limit(frequency)
    elif (frequency.max() < 1800):
        return_array = receptor_Lband_limit(frequency)
    else:
        return_array = np.nan*frequency
    # APH added the code below to adjust for atmosphere vs. elevation
    return_array = return_array - Tatm(frequency,90) + Tatm(frequency,elevation)
    return return_array

def receptor_UHFLband_CDR_limit(frequency): # MHz, at 15deg elevation
    # 275-410 m^2/K at L-band Receivers CDR, extended as 275 down through UHF-band
    Ag = np.pi*(13.5/2)**2 # Specified minimum antenna geometric collecting area
    Tsys_eta_zenith = (64*Ag)/np.interp(frequency,[580,900,1670],[275.0,275.0,410.0])
    return Tsys_eta_zenith

def receptor_Lband_limit(frequency): # APH factored out elevation, this is at zenith
    """275-410 m^2/K at Receivers CDR"""
    return_array = np.zeros_like(frequency,dtype=np.float) # APH changed division below to "/float()"
    return_array[np.array(frequency < 1280)] = np.array(12 + 6+(5.5-6)/float(1280-900)*(frequency-900))[np.array(frequency < 1280)]
    return_array[np.array(~(frequency < 1280))] = np.array(12 + 5.5+(4-5.5)/float(1670-1280)*(frequency-1280))[np.array(~(frequency < 1280))]
    return return_array

def receptor_UHFband_limit(frequency): # APH factored out elevation, this is at zenith
    return_array = np.zeros_like(frequency,dtype=np.float) # APH changed division below to "/float()"
    return_array[np.array(frequency < 900)] = np.array(8 + (12-8)/float(1015-580)*(frequency-580) + 8+(7-8)/float(900-580)*(frequency-580))[np.array(frequency < 900)]
    return_array[np.array(~(frequency < 900))] = np.array (8 + (12-8)/float(1015-580)*(frequency-580) + 7+(4-7)/float(1015-900)*(frequency-900))[np.array(~(frequency < 900))]
    return_array = (8 + (12-8)/float(1015-580)*(frequency-580)) + (10+(4-10)/float(1015-580)*(frequency-580)) # Acceptance RS + Predicted spill+atm above 40degEl
    return return_array


def plot_data_freq(frequency,Tsys,Tant,title='',aperture_efficiency=None,PLANE="antenna",plot_limits=True):
    fig = plt.figure(figsize=(16,9))
    line1,=plt.plot(frequency, Tsys[:,0], marker='o', color='b', linewidth=0)
    plt.errorbar(frequency, Tsys[:,0], Tsys[:,3], ecolor='b', color='b', capsize=6, linewidth=0)
    line2,=plt.plot(frequency, Tant[:,0], color='b'  )
    line3,=plt.plot(frequency, Tsys[:,1], marker='^', color='r',  linewidth=0)
    plt.errorbar(frequency, Tsys[:,1],  Tsys[:,4], ecolor='r', color='r', capsize=6, linewidth=0)
    line4,=plt.plot(frequency, Tant[:,1], color='r')
    plane = "/\eta_{ap}" if ("antenna" in PLANE) else ""
    plt.legend((line1, line2, line3,line4 ),  ('$T_{sys}%s$ HH'%plane,'$T_{ant}$ HH', '$T_{sys}%s$ VV'%plane,'$T_{ant}$ VV'), loc='best')
    plt.title('Tipping curve: %s' % (title))
    plt.xlabel('Frequency (MHz)')
    if plot_limits:
        data2plane = lambda data, POL: data/aperture_efficiency.eff[POL](frequency) if ("antenna" in PLANE) else data
        recLim_apEffH = data2plane(receptor_band_limit(frequency,Tsys[:,2]), "HH") # APH 082018
        recLim_apEffV = data2plane(receptor_band_limit(frequency,Tsys[:,2]), "VV") # APH 082018
        plt.plot(frequency,recLim_apEffH,lw=1.1,color='limegreen',linestyle='-')
        plt.plot(frequency,recLim_apEffV,lw=1.1,color='limegreen',linestyle='-')
        for error_margin in [0.9,1.1]:
            plt.plot(frequency,recLim_apEffH*error_margin, lw=1.1,color='g',linestyle='--')
            plt.plot(frequency,recLim_apEffV*error_margin, lw=1.1,color='g',linestyle='--')
        plt.hlines(1, frequency.min(), frequency.max(), colors='k',linestyle='--')

    low_lim = (r_lim(Tsys[:,0:2]),r_lim(Tant[:,0:2]) )
    low_lim = np.nanmin(low_lim) # APH changed to nanmin to guard against NaNs
    low_lim = -5. # np.max((low_lim , -5.))
    def tmp(x):
        return np.percentile(x,80)
    high_lim = (r_lim(Tsys[:,0:2],tmp),r_lim(Tant[:,0:2],tmp))
    high_lim = np.nanmax(high_lim) # APH changed to nanmin to guard against NaNs
    high_lim = np.nanmax((high_lim , 46*1.1)) # APH changed to nanmin to guard against NaNs, & *1.1 like plot_el, rather than *1.3
    plt.ylim(low_lim,roundup(high_lim,closest=5)) # APH added roundup
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


#### APH 02/2017 factored the code into process_b(*process_a()) which is IDENTICAL to the original process(), just re-factored slightly
def process_a(h5, ant, freq_min=0, freq_max=9e9, channel_bw=10., freq_chans=None, sky_radius=30, channel_mask='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', debug=False): # APH added freq_min & freq_max [MHz] to only load subset of interest
    """ @param sky_radius: radius to truncate the all sky integral for T_sky model [multiple of HPBW] (default 30)
        @param channel_mask: full name of pickle file with RFI mask for this band, or boolean array (discard where True) """
    h5.select(reset="TFB")
    h5.select(scans='track')
    ant = ant if isinstance(ant, str) else ant.name
    h5.select(ants=ant)
    filename = h5.name.split()[0].split(" ")[0] # "rdb files are different
    
    # Load the data file
    rec = h5.receivers[ant]
    nice_title = " %s  Ant=%s"%(filename.split('/')[-1], ant)
    
    num_channels = np.int(channel_bw/(h5.channel_width/1e6)) #number of channels per band
    chunks=[h5.channels[x:x+num_channels] for x in range(0, len(h5.channels), num_channels) # APH 032017 added following range selection
                                                if (h5.channel_freqs[x]>=freq_min*1e6 and h5.channel_freqs[x]<=freq_max*1e6)]
    
    freq_list = np.zeros((len(chunks)))
    for j,chunk in enumerate(chunks):freq_list[j] = h5.channel_freqs[chunk].mean()/1e6
    print("Selecting channel data to form %f MHz Channels spanning %.f - %.f MHz"%(channel_bw, freq_list.min(),freq_list.max()) )
    band = rec.split('.')[0].lower() # APH different from script - if a problem implement manual override when file is loaded
    # APH use h5 below instead of filename -- permits overriding of h5 attributes
    d = load_cal(h5, "%s" % (ant,), nd_models, chunks,channel_mask=channel_mask,channel_range=freq_chans,band_input=band, debug=debug)
    # Update after loading has possibly discarded masked-out chunks
    freq_list = np.asarray([d.freqs[j] for j in range(len(d.freqs))]) # APH re-phrased this to fix bug 082018
    
    aperture_efficiency = models.aperture_efficiency_models(band=models.band(freq_list))
    bm_floor_dBi = -6 # From as-built predicted paterns, -6dBi is ball park for MeerKAT UHF & L-band, also SKA SPFB1 & B2. TODO: Can this be estimated from aperture_efficiency?
    SpillOver = models.Spill_Temp(band=models.band(freq_list), default=0) # Set to 0K if not known
    receiver  = models.Rec_Temp(RxID=rec)
    elevation = np.array([np.average(scan_el) for scan_el in scape.extract_scan_data(d.scans,'el').data])
    ra        = np.array([np.average(scan_ra) for scan_ra in scape.extract_scan_data(d.scans,'ra').data])
    dec       = np.array([np.average(scan_dec) for scan_dec in scape.extract_scan_data(d.scans,'dec').data])
    sort_ind  = elevation.argsort()
    elevation,ra,dec = elevation[sort_ind],ra[sort_ind],dec[sort_ind]
    surface_temperature = np.mean(d.enviro['temperature']['value'])
    air_relative_humidity = h5.humidity.mean()/100. # Fractional
    timestamps = np.array([np.average(scan_time) for scan_time in scape.extract_scan_data(d.scans,'abs_time').data]) # APH added 052017
    timestamps = timestamps[sort_ind] # APH added 052017
    
    #length = 0 # APH 032017 removed this - not necessary
    d.filename = [filename]
    #freq loop
    T_SysTemps = []
    for i,freq_val in enumerate(d.freqs):
        print("> Loading %g MHz"%freq_val)
        T_SysTemp = System_Temp(d,freq_index=i,sort_ind=sort_ind,elevation=elevation,ra=ra,dec=dec,timestamps=timestamps,surface_temperature=surface_temperature,air_relative_humidity=air_relative_humidity) # APH added sort_ind
        T_SysTemp.calc_Sky_Temp(D=aperture_efficiency.D, bm_floor_dBi=bm_floor_dBi, radius=sky_radius) # APH re-factored into this 102018
        #print("Load T_sysTemp = %.2f Seconds"%(time.time()-time_start))
        T_SysTemps.append(T_SysTemp)
        
    return nice_title,T_SysTemps,freq_list,elevation,ra,dec,surface_temperature,air_relative_humidity,receiver,SpillOver,aperture_efficiency

def Tsysmodel(T_SysTemps,freq_list,elevation,ra,dec,surface_temperature,air_relative_humidity,receiver,SpillOver,aperture_efficiency,mb_crestfactor, full=True,Trxref=None):
    """ Computes the expected Tsys for the given conditions, excluding antenna reflector noise &
        any back-end noise that's not reflected in 'receiver' noise.
        NB: if 'full' is selected, the receiver noise model is added (H & V must labelled correctly!).
        
        @param T_SysTemps,...,aperture_efficiency: as returned by process_a
        @param mb_crestfactor: eta_mb/eta_ap = iint_mb{U}/U(0), which is robustly constant for a family of patterns.
        @param full: False to exclude the elevation-invariant contributions (default True)
        @param Trxref: thermodynamic reference temperature [K] at which the receiver noise was characterised - used in linear scaling to ambient (default None)
        @return: Tsys_model arranged as (el,freq,pol) """
    Tmodel = np.zeros((len(elevation),len(freq_list),2)) # (el, freq, pol)
    for i,T_SysTemp in enumerate(T_SysTemps): # Per frequency slice
        el_freq = np.array([elevation, [T_SysTemp.freq]*len(elevation)])
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
        BrightT_sim = models.BrightT_SKA(T_SysTemp.freq/1e3, 90-elevation) # CMB + continuum + atmosphere along boresight direction
        BrightT_meas = lambda eta_mb: atm_atten*T_SysTemp.T_avgsky*1/eta_mb + T_SysTemp.T_atm # CMB + continuum + atmosphere
        # T_SysTemp.T_avgsky is integrated over theta=null1 .. >>10*HPBW on the [CMB+GSM]*[beam pattern] map
        Tspill_H += (1-eta_mb[0])*(BrightT_meas(eta_mb[0]) - BrightT_sim)
        Tspill_V += (1-eta_mb[1])*(BrightT_meas(eta_mb[1]) - BrightT_sim)
        
        # T_SysTemp.T_sky is integrated over theta=0 .. null1 on the [GSM]*[beam pattern] map (excludes CMB)
        Tmodel[:,i,0] = (atm_atten*T_SysTemp.T_sky + eta_mb[0]*T_SysTemp.T_atm) + Tspill_H
        Tmodel[:,i,1] = (atm_atten*T_SysTemp.T_sky + eta_mb[1]*T_SysTemp.T_atm) + Tspill_V
        
        if full: # Add elevation-independent contributions, except for Antenna Ohmic + electronics after LNA.
            dTdT = 1 if (Trxref is None) else (273.15+surface_temperature)/Trxref
            Tmodel[:,i,0] += models.Tcmb + receiver.rec["HH"](T_SysTemp.freq)*dTdT 
            Tmodel[:,i,1] += models.Tcmb + receiver.rec["VV"](T_SysTemp.freq)*dTdT
        
    return Tmodel

def process_b(nice_title,T_SysTemps,freq_list,elevation,ra,dec,surface_temperature,air_relative_humidity,receiver,SpillOver,aperture_efficiency,apscale=1.,PLANE="antenna",spec_sky=True,MeerKAT=True,mb_crestfactor=1.3):
    """ @param apscale: the ratio ApArea_model/ApArea_report, which scales ap_eff(ApArea_model) to ap_eff(ApArea_report) (default 1 so for MeerKAT ApArea_report=ApArea_model which corresponds to D=13.5m)
        @param PLANE: Either "antenna" (to translate Tsys_at_waveguide to Tsys_at_waveguide/ant_eff) | "feed" (to keep raw measured Tsys_at_waveguide as is)
        @param mb_crestfactor: The ratio $eta_mb / eta_ap = iint_mb{U}/iint_4piSr{U} / U(0)/iint_4piSr{U} = iint_mb{U}/U(0)$, which is well defined and constant within ~5% for a given family of patterns (default 1.3).
    """
    tsys = np.zeros((len(elevation),len(freq_list),5 ))#*np.NaN
    tant = np.zeros((len(elevation),len(freq_list),5 ))#*np.NaN
    
    # APH 072018 incorporate apscale in aperture_efficiency
    if ("_HH" not in aperture_efficiency.eff.keys()): # Make shadow copies of original, un-scaled values
        aperture_efficiency.eff['_HH'] = aperture_efficiency.eff['HH']
        aperture_efficiency.eff['_VV'] = aperture_efficiency.eff['VV']
        aperture_efficiency._D = aperture_efficiency.D
    aperture_efficiency.eff['HH'] = lambda *args: apscale*aperture_efficiency.eff['_HH'](*args)
    aperture_efficiency.eff['VV'] = lambda *args: apscale*aperture_efficiency.eff['_VV'](*args)
    aperture_efficiency.D = aperture_efficiency._D/(apscale**.5)
    mb_crestfactor /= apscale
    
    # TODO: 11/2020 rewrite below to use Tsysmodel(T_SysTemps,freq_list,elevation,ra,dec,surface_temperature,air_relative_humidity,receiver,SpillOver,aperture_efficiency,mb_crestfactor=1.3, full=True)
    ### APH 032017 override sky noise subtraction
    for i,T_SysTemp in enumerate(T_SysTemps):
        ### APH 032017 this has been the way it's been done till now (still in System_Temp)
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-np.asarray(T_SysTemp.T_sky) 
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-np.asarray(T_SysTemp.T_sky) 
        
        ### APH (1) subtract it AT THE FEED input plane -- results & un-modelled residuals look beautiful
        atm_atten = np.exp(-T_SysTemp.opacity)
        ##T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_SysTemp.T_sky)-Tcmb)*aperture_efficiency.eff['HH'](d.freqs[i]))
        ##T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_SysTemp.T_sky)-Tcmb)*aperture_efficiency.eff['VV'](d.freqs[i]))
        # APH (6) below works quite well, if smooth=1 i.e. beam filling factor = 0.5
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_SysTemp.T_sky)-Tcmb)*0.5)
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_SysTemp.T_sky)-Tcmb)*0.5)
        # APH (7) the above but include "foreground" as point sources
        #T_sky_bg,T_sky_ps = Sky_temp(freq_val,smooth=0,smooth_bg=1).Tsky(T_SysTemp.ra,T_SysTemp.dec,split=True) # BG resolution matches beam i.e. coupling=0.5
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_sky_bg)-Tcmb)*0.5+np.asarray(T_sky_ps)*aperture_efficiency.eff['HH'](d.freqs[i]))
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_sky_bg)-Tcmb)*0.5+np.asarray(T_sky_ps)*aperture_efficiency.eff['VV'](d.freqs[i]))
        # APH (8) the above but include "foreground" as point sources
        #T_sky_bg,T_sky_ps = Sky_temp(freq_val,smooth=1).Tsky(T_SysTemp.ra,T_SysTemp.dec,split=True) # FG resolution matches beam i.e. coupling=0.5
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_sky_bg)-Tcmb)+np.asarray(T_sky_ps)*aperture_efficiency.eff['HH'](d.freqs[i]))
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-T_SysTemp.T_atm-atm_atten*(Tcmb+(np.asarray(T_sky_bg)-Tcmb)+np.asarray(T_sky_ps)*aperture_efficiency.eff['VV'](d.freqs[i]))
        # APH (9) changed source coupling: FG trated as point source and since resolution matches beam i.e. T_ps=2*T_smoothed
        #T_sky_bg,T_sky_ps = Sky_temp(d.freqs[i],smooth=1).Tsky(T_SysTemp.ra,T_SysTemp.dec,split=True) # FG resolution matches beam i.e. T_ps=2*T_smoothed
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-T_SysTemp.T_atm-atm_atten*(np.asarray(T_sky_bg)+np.asarray(T_sky_ps)*2)*aperture_efficiency.eff['HH'](d.freqs[i])
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-T_SysTemp.T_atm-atm_atten*(np.asarray(T_sky_bg)+np.asarray(T_sky_ps)*2)*aperture_efficiency.eff['VV'](d.freqs[i])
        # (12) eta_mb = 1.3*eta_ap and scale everything as if D=13.5m
        eta_rad = 0.99
        eta_mb = [mb_crestfactor*aperture_efficiency.eff['HH'](T_SysTemp.freq), mb_crestfactor*aperture_efficiency.eff['VV'](T_SysTemp.freq)] # APH reinstated this 10/2020 and deactivated all below
#         # (14) eta_mb = 0.97 and use D=13.965m throughout (eta_ap is scaled to match this)
#         eta_mb = [0.97, 0.97]
#         # (16) Use eta_mb for actual pattern, and correct for Tspill being based on "nominal" background rather than actual background
#         eta_mb = np.asarray([[0.9],[0.9]])
        #Delta_Tspill = (1-eta_mb)*atm_atten*(T_sky_bg-(Tcmb + 1.6*(d.freqs[i]/1e3)**-2.75)) # Approximate correction integrated beyond main beam
        # (17) Scale T_*map with 1/(main beam:gaussian beam) factor
        #T_sky_ps *= 1/.86
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-Delta_Tspill[0]-eta_rad*eta_mb[0]*(T_SysTemp.T_atm+atm_atten*(np.asarray(T_sky_bg)+np.asarray(T_sky_ps)))
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-Delta_Tspill[1]-eta_rad*eta_mb[1]*(T_SysTemp.T_atm+atm_atten*(np.asarray(T_sky_bg)+np.asarray(T_sky_ps)))
        # (18) Corrected opacity calculation. Also correct SpillOver for difference in assumed atmosphere used only in fit_tipping
        #spill_scale = (T_SysTemp.T_atm+atm_atten*T_sky_bg) / Tsky_tab_SKA([d.freqs[i]/1e3],(90-T_SysTemp.elevation)*np.pi/180.) # Approximate correction integrated beyond main beam
        ##WIP##spill_scale = (T_SysTemp.T_atm+atm_atten*T_sky_bg+273.15+T_SysTemp.surface_temperature)/2. / BrightT_SKA([d.freqs[i]/1e3],(90-T_SysTemp.elevation)*np.pi/180.)
        #print(d.freqs[i], " spillover scaled by ", spill_scale)
        # (19) Drop spill_scale & Correction of airmass function as per Han & Westwater 2000 -- see System_Temp.__init__()
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-eta_rad*eta_mb[0]*(T_SysTemp.T_atm+atm_atten*(np.asarray(T_sky_bg)+np.asarray(T_sky_ps)))
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-eta_rad*eta_mb[1]*(T_SysTemp.T_atm+atm_atten*(np.asarray(T_sky_bg)+np.asarray(T_sky_ps)))
        spill_scale = 1.
        # (21) Use T_SysTemp.T_sky which is now correctly integrated across the antenna beam out to 2nd null
        # (22) Use sky-based T_ND and correct opacity (also use Han & Westwater's T_atm0)
        # (23) use eta_rad=1 for sky-based T_ND
        eta_rad = 1
        # (29) dropped the one scale by eta_mb
        #T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-eta_rad*eta_mb[0]*(T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky))
        #T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-eta_rad*eta_mb[1]*(T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky))
        T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-eta_rad*(eta_mb[0]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky))
        T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-eta_rad*(eta_mb[1]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky))
        # (27),(28) Correct for the fact that spillover doesn't include Tgal.  For tau < 0.01 it can be shown that avg atmospheric extinction is > 0.99, with <5% error.
        T_SysTemp.Tsys_sky['HH'] -= (1-eta_mb[0])*0.99*np.asarray(T_SysTemp.T_avgsky)
        T_SysTemp.Tsys_sky['VV'] -= (1-eta_mb[1])*0.99*np.asarray(T_SysTemp.T_avgsky)
        
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
#         TA_spill_HH = eta_mb[0]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky) + atm_atten_allsky*(np.asarray(T_SysTemp.T_avgsky)-(1-eta_mb[0])*Tcelest) # 10/2018 T_avgsky is now computed with main beam blanked, so don't scale that by (1-eta_mb)
#         TA_spill_VV = eta_mb[1]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky) + atm_atten_allsky*(np.asarray(T_SysTemp.T_avgsky)-(1-eta_mb[1])*Tcelest)
        TA_spill_HH = eta_mb[0]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky) + atm_atten_allsky*atm_atten*(np.asarray(T_SysTemp.T_avgsky)-(1-eta_mb[0])*Tcelest)
        TA_spill_VV = eta_mb[1]*T_SysTemp.T_atm+atm_atten*np.asarray(T_SysTemp.T_sky) + atm_atten_allsky*atm_atten*(np.asarray(T_SysTemp.T_avgsky)-(1-eta_mb[1])*Tcelest)
        T_SysTemp.Tsys_sky['HH'] = np.asarray(T_SysTemp.Tsys['HH'])-eta_rad*TA_spill_HH
        T_SysTemp.Tsys_sky['VV'] = np.asarray(T_SysTemp.Tsys['VV'])-eta_rad*TA_spill_VV
    
    ## Standard code resumes
    #units = T_SysTemp.units+'' APH 0302017 moved to "report_...()"
    for i,T_SysTemp in enumerate(T_SysTemps):
        fit_H = fit_tipping(T_SysTemp,SpillOver,'HH',T_SysTemp.freq,receiver,fixopacity=fix_opacity,spill_scale=spill_scale)
        #print("Fit tipping H = %.2f Seconds"%(time.time()-time_start))
        fit_V = fit_tipping(T_SysTemp,SpillOver,'VV',T_SysTemp.freq,receiver,fixopacity=fix_opacity,spill_scale=spill_scale)
        #print("Fit tipping V = %.2f Seconds"%(time.time()-time_start))
        #print ('Chi square for HH  at %s MHz is: %6f ' % (np.mean(d.freqs),fit_H['chisq'],))
        #print ('Chi square for VV  at %s MHz is: %6f ' % (np.mean(d.freqs),fit_V['chisq'],))
        #length = len(T_SysTemp.elevation)  # APH 032017 removed -- not necessary. Also changed [0:length,...]  below to [:,...]
        if spec_sky: # APH added this condition 082018
            Tsky_spec = Tcmb + 1.7*(T_SysTemp.freq/1e3)**-2.75 # T_SysTemp.Tsys_sky  is Tsys-Tsky (Tsky includes Tcmb & Tatm). We then add the spec sky aproxx (T_gal+Tcmb)
            Tsky_spec = Tsky_spec*atm_atten + T_SysTemp.T_atm # APH032017 added this line. T_SysTemp.Tsys_sky  is Tsys-Tsky (Tsky includes Tcmb & Tatm) @ feed input. We then add the spec sky aproxx (T_gal+Tcmb+T_atm)
            tsys[:,i,0] = np.array(T_SysTemp.Tsys_sky['HH'])+eta_rad*Tsky_spec
            tsys[:,i,1] = np.array(T_SysTemp.Tsys_sky['VV'])+eta_rad*Tsky_spec
            #tsys[:,i,0] = (np.array(T_SysTemp.Tsys_sky['HH']))+Tsky_spec*aperture_efficiency.eff['HH'](d.freqs[i]) # APH 032017 replaced above
            #tsys[:,i,1] = (np.array(T_SysTemp.Tsys_sky['VV']))+Tsky_spec*aperture_efficiency.eff['VV'](d.freqs[i]) # APH 032017
            PLANE += ",spec_sky"
        else:
            tsys[:,i,0] = np.array(T_SysTemp.Tsys['HH'])
            tsys[:,i,1] = np.array(T_SysTemp.Tsys['VV'])
        tsys[:,i,2] = elevation
        tsys[:,i,3] = np.array(T_SysTemp.sigma_Tsys['HH'])
        tsys[:,i,4] = np.array(T_SysTemp.sigma_Tsys['VV'])
        if ("antenna" in PLANE): # APH 072018 Raw measurement (scaled by T_ND) is Tsys_at_waveguide, /= ant_eff translates to Tsys_at_antenna
            tsys[:,i,0] /= aperture_efficiency.eff['HH'](T_SysTemp.freq)
            tsys[:,i,1] /= aperture_efficiency.eff['VV'](T_SysTemp.freq)
            tsys[:,i,3] /= aperture_efficiency.eff['HH'](T_SysTemp.freq)
            tsys[:,i,4] /= aperture_efficiency.eff['VV'](T_SysTemp.freq)
        tant[:,i,0] = np.array(fit_H['fit'])[:,0]
        tant[:,i,1] = np.array(fit_V['fit'])[:,0]
        tant[:,i,2] = elevation
        #print("Debug: T_sys = %f   App_eff = %f  value = %f"%( np.array(fit_H['fit'])[22,0],aperture_efficiency.eff['HH'](d.freqs[i]),np.array(fit_H['fit'])[22,0]/aperture_efficiency.eff['HH'](d.freqs[i])))
    if (np.max(SpillOver.spill["HH"]([[45]*len(T_SysTemps),[T.freq for T in T_SysTemps]])) == 0): # It should be enough to test one polarisation at 45degEl
        PLANE += ",nospill"
    # The last T_SysTemp is passed on as a convenient container for sky_fig, units, elevation angles
    return nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE

def report_a(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE):
    reportfilename = "%s_%s_tipping_curve.pdf"%(filename.split("/")[-1].split(".")[0],ant)
    pp = PdfPages(reportfilename)
    print('http://kat-imager.kat.ac.za:8888/files/SysEng/'+reportfilename)
    print('http://kat-imager.kat.ac.za:8888/files/SysEng/ska-se/aph/'+reportfilename)
    print('http://com04.science.kat.ac.za:8891/tree/work/sdqm/'+reportfilename)
    return pp

def report_b(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp, cartesian=True):
    fig = T_SysTemp.sky_fig(np.mean(freq_list), cartesian=cartesian)
    # Plot every 10th point with elevation label
    fig.savefig(pp,format='pdf')

def report_c(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp, select_freq=None, channel_bw=10., plot_limits=True):
    length = len(T_SysTemp.elevation)
    select_freq = select_freq if select_freq else freq_list
    iselect_freq = [(np.abs(freq_list-freq)).argmin() for freq in select_freq] # Indices into freq_list
    
    tsys_ = np.take(tsys[0:length,:,0:2], iselect_freq, axis=1) # El, etc. are in axis=2@[2:]
    tant_ = np.take(tant[0:length,:,0:2], iselect_freq, axis=1) # El, etc. are in axis=2@[2:]
    ylim = (roundup(np.nanpercentile(tant_,2), closest=5)-5, roundup(np.nanpercentile(tsys_,98), closest=5)) # 5 .. 95 for robustness to RFI
    
    for i,freq in zip(iselect_freq,select_freq):
        lineval = 46 if freq > 1420 else 42 #UHF & L-band spec limit
        lineval = lineval if plot_limits else -1
        # APH slightly simplified title below and use freq_list instead of d.freqs
        fig = plot_data_el(tsys[0:length,i,:],tant[0:length,i,:],title="%s at %.1f MHz"%(nice_title,freq),units=T_SysTemp.units,line=lineval,aperture_efficiency=aperture_efficiency,PLANE=PLANE,frequency=freq_list[i])
        plt.ylim(*ylim)
        plt.figtext(0.89, 0.04,git_info(), horizontalalignment='right',fontsize=10)
        fig.savefig(pp,format='pdf')

def report_d(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp, select_freq=None, select_el=[15,45,90], plot_limits=True):
    ## APH added this to exclude unspecified frequencies at either edge
    freqlim = select_freq if select_freq else freq_list
    freqlim = [np.min(freqlim)-20, np.max(freqlim)+20]
    f_low, f_high = abs(freq_list-freqlim[0]).argmin(),abs(freq_list-freqlim[-1]).argmin()
    _fspec = range(min(f_low,f_high),max(f_low,f_high)+1)
    iselect_el = [(np.abs(T_SysTemp.elevation-el)).argmin() for el in select_el] # Indices into elevation
    
    tsys_ = np.take(tsys[:,_fspec,0:2], iselect_el, axis=0) # El, etc. are in axis=2@[2:]
    tant_ = np.take(tant[:,_fspec,0:2], iselect_el, axis=0) # El, etc. are in axis=2@[2:]
    ylim = (roundup(np.nanpercentile(tant_,2), closest=5)-5, roundup(np.nanpercentile(tsys_,98), closest=5))
    
    for i in iselect_el :
        fig = plot_data_freq(freq_list[_fspec],tsys[i,_fspec,:],tant[i,_fspec,:],title="%s at %.1f$^\circ$ elevation"%(nice_title,T_SysTemp.elevation[i]),aperture_efficiency=aperture_efficiency,PLANE=PLANE,plot_limits=plot_limits)
        plt.figtext(0.89, 0.04,git_info(), horizontalalignment='right',fontsize=10)
        plt.ylim(*ylim)
        fig.savefig(pp,format='pdf')
    return f_low, f_high, _fspec

def report_cd(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp): # APH 072018 add elevation vs frequency contour plots to combine c & d
    T_min = 10 # TODO: lowest conceivable system noise used to mask wild low values
    for i,p in enumerate("HV"):
        fig = plt.figure(figsize=(16,9))
        _t_ = tsys[:,:,i]
        T_max = roundup(np.nanpercentile(_t_,95), closest=5) + 5 # +5 softens saturation in case there's actually little RFI
        plt.contourf(freq_list, T_SysTemp.elevation, _t_, levels=np.linspace(T_min,T_max,10)); plt.colorbar(fraction=0.05)
        plt.xlabel("f [MHz]"); plt.ylabel("Elevation angle [$^\circ$]")
        plt.title("%s, $T_{sys}%s$ %s"%(nice_title,("" if ("feed" in PLANE) else "/\eta_{ap}"),p*2))
        plt.figtext(0.89, 0.04,git_info(), horizontalalignment='right',fontsize=10)
        fig.savefig(pp,format='pdf')

def report_e(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp):
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
    if ("spec_sky" in PLANE):
        text += r"""2. The measured $T_{\mathrm{sys}}(\mathrm{el})$ is adjusted for the difference between the actual sky brightness and the sky model
        prescribed in the MeerKAT specifications, by adding
        $\Delta T_{celest} = e^{-\tau(el)} ( T_{gal} - T_{gsm} )$
      """
    
    if ("antenna" in PLANE):
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
    if fix_opacity :
        text += r"""  * $\tau_{0}$, the zenith opacity, is set to 0.01078 (Van Zee et al., 1997). """
    else:
        text += r"""  * $\tau_{0}$, the zenith opacity, is calculated according to ITU-R P.676-9."""
    text += r"""
      * $T_{\mathrm{atm}}$ is calculated as per Ippolito 1989 (eq 6.8-6)
    """
    if ("nospill" in PLANE):
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
    #plt.figtext(0.1,0.1,text,fontsizie=10)
    plt.figtext(0.89, 0.04,git_info(), horizontalalignment='right',fontsize=10)
    fig.savefig(pp,format='pdf')
    pp.close()
    plt.close(fig)

    
def process(h5, ant, f_minMHz, f_maxMHz, channel_mask='/var/kat/katsdpscripts/RTS/rfi_mask.pickle', apscale=1.,PLANE="antenna",spec_sky=True,MeerKAT=True):
    return process_b(*process_a(h5, ant, f_minMHz, f_maxMHz, channel_mask=channel_mask), apscale=apscale,PLANE=PLANE,spec_sky=spec_sky,MeerKAT=MeerKAT)

cache = {} # "files":[filenames], ant.name: [ *[filename, process():return]* ], ant.name+"/bb": {filename:intermadiate results}

def process_cached(cache, h5, ant, f_min, f_max, apscale=1.,PLANE="antenna"):
    filename = h5.name.split()[0].split(" ")[0] # "rdb files are different
    try: # Avoid re-processing from scratch
        intermediate_results = cache[ant.name+"/bb"][filename]
    except: # Not yet loaded
        intermediate_results = process_a(h5, ant, f_min, f_max)
        cache[ant.name+"/bb"] = cache.get(ant.name+"/bb", {})
        cache[ant.name+"/bb"][filename]= intermediate_results # Cache this for quick re-processing
    nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant = process_b(*intermediate_results, apscale=apscale,PLANE=PLANE)
    return nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant

def report(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, select_freq=None, select_el=[15,45,90], cartesian=True, plot_limits=True):
    pp = report_a(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE)
    report_b(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp, cartesian)
    report_c(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp, select_freq, plot_limits=plot_limits)
    f_low, f_high, _fspec = report_d(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp, select_freq, select_el=select_el, plot_limits=plot_limits)
    report_cd(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp)
    report_e(ant, filename, nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE, pp)
    return nice_title, freq_list, T_SysTemp, aperture_efficiency, tsys, tant, PLANE


# General band-neutral settings
fix_opacity = False
nd_models = '/var/kat/katconfig/user/noise-diode-models/mkat'
#nd_models = '/home/kat/SysEng/tsys_results/' # Early days - MANUALLY COPY TO HERE

def open_dataset(filename, band=None, hackedL=False, ant_rx_SN={}, verbose=True, **kwargs):
    """ @param band: None to let 'models' guess the band from frequencies, or force to e.g. "B1", "u", "l" (default None)
       @param hackedL: deprecated, use filename=katselib.as_hackedL(katdal.open(fn)) instead
       @param ant_rxSN: {antname:rxband.sn} Early system did not reflect correct receiver ID, so override
       @param kwargs: Early system did not reflect correct centre freq, so pass 'centre_freq='[Hz] to override
       @return: the katdal dataset with data selected & ordered as required. """
    h5 = katdal.open(filename, **kwargs) if isinstance(filename,str) else filename
    if hackedL:
        h5 = katselib.as_hackedL(h5)
        h5.select(freqrange=(300e6,856.5e6))
    
    # Override receiver ID if requested
    for ant in h5.ants:
        h5.receivers[ant.name] = ant_rx_SN.get(ant.name,h5.receivers[ant.name])

    if verbose:
        print(h5)
        print(h5.receivers)
    
    models.BAND = band
    return h5

open_h5 = open_dataset # Deprecated alias

if __name__ == "__main__":
    try:
        import sys
        filename = sys.argv[1]
        antname = sys.argv[2]
        select_freq = eval("[%s]"%sys.argv[3])
    except:
        print("Usage: <program> filename ant select_freq")
        print("    ant: \"all\" or \"m0xx\"")
        print("    select_freq: comma-separated list of MHz e.g. 1000,1300,1600")
    else:
        h5 = open_dataset(filename)
        ants = [a.name for a in h5.ants] if (antname=="all") else [antname]
        for ant in ants:
            ru = process(h5, ant, np.min(select_freq), np.max(select_freq), PLANE="antenna",spec_sky=True)
            report(ant, h5.name.split()[0].split(" ")[0], *ru, select_freq=select_freq)
