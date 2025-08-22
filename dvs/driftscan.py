#!/usr/bin/python
"""
    Analysis of single dish drift scan, to derive SEFD, Ae/Tsys and Noise Diode scale.
    Also compares measurement with expected results, if the necesary engineering data exists.
    
    Typical use 1:
        python driftscan.py /var/data/132598363.h5 0 "Baars 1977"
    which is equivalent to
        from driftscan import analyse
        analyse(sys.argv[1], int(sys.argv[2]), sys.argv[3], saveroot=".", makepdf=True)
        
    Advanced use:
        import driftscan
        ds = driftscan.DriftDataset("/var/data/1417562258.h5", ant="m063", ant_rxSN={"m063":"l.0004"}, debug=True)
        src, hpw_src, profile_src, S_src = driftscan.models.describe_source(ds.target.name, verbose=True)
        
        bore, null_l, null_r, _HPBW, null_w = driftscan.find_nulls(ds, hpw_src=hpw_src, debug_level=1)
        _bore_ = int(np.median(bore))
        
        offbore_deg = driftscan.target_offset(ds.target, ds.timestamps[_bore_], ds.az[_bore_], ds.el[_bore_], np.mean(ds.channel_freqs))
        hpbw0 = np.nanmedian(_HPBW); hpbwf0 = np.median(ds.channel_freqs[np.abs(_HPBW/hpbw0-1)<0.01])
        offbore0 = offbore_deg*np.pi/180 / hpbw0
        wc_scale = driftscan.models.G_bore(offbore0, hpbwf0, np.max(ds.channel_freqs))
        if (wc_scale < 0.99):
            print("CAUTION: source transits far from bore sight, scaling flux by up to %.1f%%"%(100*(1-wc_scale)))
        
        par_angle = ds.parangle[_bore_] * np.pi/180
        Sobs_src = lambda f_GHz,yr: S_src(f_GHz,yr,par_angle) * driftscan.models.G_bore(offbore0, hpbwf0/1e9, np.reshape(f_GHz, (-1,1)))
                
        freqs, counts2Jy, SEFD_meas, SEFD_pred, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg = \
                driftscan.get_SEFD_ND(ds,bore,[null_l[0],null_r[0]],null_w,
                                      Sobs_src,hpw_src/_HPBW,profile_src,
                                      freqmask=[(360e6,380e6),(924e6,960e6),(1084e6,1088e6)]) # Blank out MUOS, GSM & SSR
        
    @author aph@sarao.ac.za
"""
import pylab as plt
import numpy as np
import warnings
import scipy.optimize as sop
import katpoint
from . import driftfit, util
from analysis import katsemodels as models
from analysis.katsemodels import _kB_, _c_
from analysis import katsemat as mat
from analysis.katselib import PDFReport


def _ylim_pct_(data, tail_pct=10, margin_pct=0, snap_to=None):
    """ @param tail_pct: the single sided tail percentage [percent]
        @param margin_pct: min & max values are this much below & above the tails [percent]
        @param snap_to: limits are rounded to integer multiples of this number (default 1).
        @return (min,max) or possibly None """
    lower_pct = tail_pct; upper_pct = 100-tail_pct 
    snap_to = None if (snap_to is None) or (snap_to <= 0) else snap_to
    _data = np.ma.compressed(data) if isinstance(data, np.ma.masked_array) else data[np.isfinite(data)]
    if (len(_data) == 0):
        return None
    else:
        ylim = [(1-margin_pct/100.)*np.nanpercentile(_data,lower_pct), (1+margin_pct/100.)*np.nanpercentile(_data,upper_pct)]
        if np.all(np.isfinite(ylim)) and (snap_to is not None):
            ylim = (int(ylim[0]/snap_to)*snap_to, int(ylim[1]/snap_to+0.5)*snap_to)
        return ylim

def plot_data(x_axis,vis,y_lim=None,x_lim=None,header=None,xtag=None,ytag=None,bars=None, style="-", newfig=True, **plotargs):
    """ @param y_lim: (min,max) limits for y axis, or 'pct,tail,margin', or
                      just 'pct' to base it on 5th & 95th percentiles with 30% margin (default None)
        @param x_lim: (min,max) limits for x axis (default None)
        @param bars: If a sequence of shape 2xN, errorbars are drawn at -row1 and +row2 relative to the data.
        @param header: axis "title".
        @param plotargs: e.g. "errorevery=100" to control placement of error bar ticks
    """
    if newfig:
        plt.figure(figsize=(12,6))
    if bars is None:
        plt.plot(x_axis,vis,style,**plotargs)
    else:
        try:
            plt.errorbar(x_axis, vis, fmt=style, capsize=1, yerr=bars, **plotargs)
        except:
            for p in range(np.shape(vis)[-1]):
                plt.errorbar(x_axis, vis[:,p], fmt=style, capsize=1, yerr=bars[:,p], **plotargs)
    if y_lim and ('pct' in y_lim):
        _ypct = [int(i) for i in (y_lim+",5,30").split(",")[1:3]]
        y_lim = _ylim_pct_(vis, *_ypct)
    if (y_lim is not None) and np.all(np.isfinite(y_lim)):
        plt.ylim(y_lim)
    if x_lim is not None:
        plt.xlim(x_lim)
    if header:
        plt.title(header)
    if xtag:
        plt.xlabel(xtag)
    if ytag:
        plt.ylabel(ytag)
    plt.grid(True)
    if ("label" in plotargs.keys()):
        plt.legend(loc='best', fontsize='small', framealpha=0.8)


def _idx(domain1d, bounds):
    """ @param bounds: a list or tuple of min,max values (same units as domain1d) or None to pass all.
        @return: a boolean index with True where domain1d fall within the bounds, or no bounds given """
    if (bounds is None) or (len(bounds) == 0):
        return np.full(np.shape(domain1d), True)
    else:
        return np.logical_and(domain1d>=np.min(bounds), domain1d<=np.max(bounds))


def mask_where(array_nd, domain1d, domainmask, axis=-1):
    """ Generates a masked array from mask intervals specified for domain1d.
        
        @param array_nd: an array to mask (1-dimensional or higher)
        @param domain1d: the values of the domain for the axis to apply the mask over
        @param domainmask: list of (start,stop) intervals of values in domain1d which must be masked out, inclusive
        @param axis: specify which axis of array2d the domain relates to, in case it is not obvious (default -1)
        @return: masked array representation of array2d
    """
    if (domainmask is not None) and (len(domainmask) > 0):
        array_nd = np.atleast_1d(array_nd)
        if (axis < 0):
            axis = [i for i,n in enumerate(array_nd.shape) if (len(domain1d)==n)]
            assert (len(axis) > 0), "None of the axes match the mask - you must explicitly pick one."
            assert (len(axis) == 1), "Multiple axes can match the mask - you must explicitly pick one."
            axis = axis[0]
        mask1d = np.full(len(domain1d), False)
        for minmax in domainmask:
            mask1d |= _idx(domain1d, minmax)
        mask = np.full(array_nd.shape, False)
        mask = np.apply_along_axis(lambda a:mask1d, axis, mask)
        return np.ma.masked_array(array_nd, mask|np.isnan(array_nd), fill_value=np.nan)
    else:
        return np.ma.masked_array(array_nd, np.isnan(array_nd), fill_value=np.nan)


class DriftDataset(object):
    """ A container for the raw measurement data, to optimise data retrieval behaviour. """
    def __init__(self, ds, ant, pol_labels="H,V", swapped_pol=False, strict=False, flags="data_lost,ingest_rfi", debug=False, **kwargs):
        """
           @param ds: filename string, or an already opened katdal.Dataset
           @param ant: either antenna ID (string) or index (int) in the dataset.ants list
           @param pol_labels: two comma-separated characters e.g. 'H,V' to load the corresponding autocorrelation data in that sequence.
           @param swapped_pol: True to swap the order of the polarisation channels around e.g. if wired incorrectly (default False)
           @param strict: True to only use data while 'track'ing (e.g. tracking the drift target), vs. all data when just not 'slew'ing  (default False)
           @param flags: Select flags used to clean the data (default "data_lost,ingest_rfi") - MUST exclude 'cam'!
           @param kwargs: Passed to utils.open_dataset(). E.g. 'hackedL=True', or 'ant_rx_override={antname:rxband.sn}' to override.
        """
        self._pol = pol_labels.split(",")
        self.ds = ds = util.open_dataset(ds, ref_ant=ant, **kwargs)
        self.name = ds.name.split("/")[-1].split(".")[0] # Without extension
        if (isinstance(ant, int)):
            ant = ds.ants[ant].name
        ds.select(ants=ant)
        self.ant = ds.ants[0]
        self.RxSN = ds.receivers[self.ant.name]
        self.ants = [self.ant] # "Ducktyping" so this DriftDataset looks like a katdal.Dataset to models.fit_bg()
        # Targets are listed in the time sequence of the dataset.
        # Reverse the order to avoid there historic bug where the first scan was associated with the target from the previous observation. 
        self.target = [t for t in reversed(ds.catalogue.targets) if t.body_type=='radec'][0]
        self.target.antenna = ds.ants[0]
        
        self.start_time = ds.start_time
        self.dump_period = ds.dump_period
        
        self._load_vis_(swapped_pol, strict, flags, debug)
        self.__loadargs__ = dict(swapped_pol=swapped_pol, strict=strict, flags=flags) # If needed by future select()
        
        # Ensure dish diameter is as assumed in models (relied upon in this module)
        self.band = models.band(self.channel_freqs/1e6, ant=self.ant.name)
        D_models = models.Ap_Eff(band=self.band).D
        if (D_models > 0): self.ant.diameter = D_models
        
        if debug:
            self.debug()


    def _load_vis_(self, swapped_pol=False, strict=False, flags="data_lost,ingest_rfi", verbose=True):
        """ Loads the dataset and provides it with work-arounds for issues with the current observation procedures.
            Also supplies auxilliary features that are employed in processing steps in this module.
            
            The returned dataset has data filtered and arranged as required for processing. It specifically comprises
            only two products (third index of dataset), arranged as [0=antHH, 1=antVV]. DO NOT use the 'corr_products'
            attribute as that will not reflect the 'swapped_pol' state. Use the hacked '_pol' attribute instead. 
           
           @param swapped_pol: True to swap the order of the polarisation channels around e.g. if wired incorrectly (default False)
           @param strict: True to only use data while 'track'ing (e.g. tracking the drift target), vs. all data when just not 'slew'ing  (default False)
           @param flags: Select flags used to clean the data (default "data_lost,ingest_rfi") - MUST exclude 'cam'!
        """
        h5 = self.ds
        h5.select(flags=flags)
        
        pol_labels = reversed(self._pol) if swapped_pol else self._pol
        h5.select(pol=[_*2 for _ in pol_labels], reset="T")
        if verbose:
            print(h5)
            print("Receivers:", {self.ant.name: h5.receivers[self.ant.name]})
        
        # Current observe script sometimes mislabels the first ND scan as "slew"
        h5.select(compscans="noise diode", scans="~stop")
        ND_scans = h5.scan_indices # Converted from list -> tuple of lists below
        if (len(ND_scans) == 3):
            ND_scans = (ND_scans[:2],ND_scans[-1:]) if (np.diff(ND_scans)[0]==1) else (ND_scans[:1],ND_scans[1:])
        else:
            ND_scans = (ND_scans[:1],ND_scans[-1:])
        def __scans_ND__():
            for scans in ND_scans:
                h5.select(scans=scans)
                yield (scans[0], h5.sensor["Antennas/array/activity"][0], h5.sensor["Observation/target"][0])
        h5.__scans_ND__ = __scans_ND__
        
        # Current observe script generates compscans with blank labels - these are spurious
        # Current observe script generates two (s="track", cs="drift") scans, but the first one is spurious and sometimes is on the
        # other side of the azimuth wrap, which causes issues with tangent plane projection in find_nulls()
        spurious_scans = [] # Scan indices that should always be ignored
        h5.select(reset="T")
        for cs in h5.compscans(): # (index,label,target)
            if (cs[1].strip() == ''):
                spurious_scans.extend([s[0] for s in h5.scans()])
            elif (cs[1] == 'drift'):
                si = [s[0] for s in h5.scans() if (s[1]=='track')]
                if (len(si) > 1): # Multiple s=track,cs=drift 
                    spurious_scans.append(si[0]) # The first of these is spurious
        
        # Basic select rules for extracting SEFD
        def __select_SEFD__(**selectkwargs): # If strict then just 'track', else everything except 'slew'. Filters out spurious scans.
            if strict:
                h5.select(scans="track", **selectkwargs)
                sscan_indices = set(h5.scan_indices)-set(spurious_scans)
                if (len(sscan_indices) < len(h5.scan_indices)):
                    h5.select(reset='', scans=sscan_indices) # Ideally (scans=~spurious_scans)  but that doesn't work. This workaround has draw back that it resets dumps & timerange. 
            else:
                h5.select(scans="~slew", **selectkwargs)
        h5.__select_SEFD__ = __select_SEFD__
        
        # Cache the raw data to avoid costly network access
        self.nd_vis = {}
        for scan in h5.__scans_ND__():
            label = "scan %d: '%s'"%(scan[0],scan[1])
            vis = np.abs(h5.vis[:]); vis[h5.flags[:]] = np.nan; vis[:,0,:] = np.nan
            self.nd_vis[label] = vis
        
        h5.__select_SEFD__()
        vis = np.abs(h5.vis[:]); vis[h5.flags[:]] = np.nan; vis[:,0,:] = np.nan
        self.vis = vis
        
        self.channel_freqs = h5.channel_freqs
        self.channels = h5.channels
        self.az = h5.az
        self.el = h5.el
        self.parangle = h5.parangle
        self.timestamps = h5.timestamps
        self.sensor = h5.sensor
        # These values represent the state as per __select_SEFD__()!
        self.el_deg = np.median(h5.el)
        self.mjd = np.median(h5.mjd)


    def select(self, *args, **kwargs):
        """ Update the dataset selection & re-loads the DriftDataset data. """
        self.ds.select(*args, **kwargs)
        self._load_vis_(verbose=False, **self.__loadargs__)


    def debug(self):
        vis_ = self.vis/np.nanpercentile(self.vis, 1, axis=0) # Normalized as increase above baseline (robust against unlikely zeroes)
        chans = self.channels
        ax = plt.subplots(2,2, sharex=True, gridspec_kw={'height_ratios':[2,1]}, figsize=(16,8))[1]
        for p,P in enumerate(self._pol):
            ax[0][p].imshow(10*np.log10(vis_[:,:,p]), aspect="auto", origin="lower", extent=(chans[0],chans[-1], 0,vis_.shape[0]),
                            vmin=0, vmax=6, cmap=plt.get_cmap("viridis"))
            ax[0][p].set_title("%s\n%s %s"%(self.name,self.ant.name,P)); ax[0][p].set_ylabel("Time [samples]")
            ax[1][p].plot(chans, 10*np.log10(np.nanmax(self.vis[:,:,p], axis=0)))
            ax[1][p].set_ylabel("max [dBcounts]"); ax[1][p].grid(True)
            ax[1][p].set_xlabel("Channel #")
        
        freqs = self.channel_freqs/1e6
        for i in [0,1]:
            alt_x = ax[1][i].secondary_xaxis('top', functions=(lambda c:freqs[0]+(freqs[-1]-freqs[0])/(chans[-1]-chans[0])*(c-chans[0]),
                                                               lambda f:chans[0]+(chans[-1]-chans[0])/(freqs[-1]-freqs[0])*(f-freqs[0])))
            alt_x.set_xlabel("Frequency [MHz]")


def pred_SEFD(freqs, Tcmb, Tgal, Tatm, el_deg, RxID, band):
    """ Computes Tsys & SEFD from predicted values & standard models.
        @param freqs: frequencies [Hz]
        @param Tcmb: either a constant or a vector matching 'freqs' [K]
        @param Tgal: a function of frequency[Hz] [K]
        @param Tatm: a function of (frequency[Hz],elevation[deg]) [K]
        @param el_deg: elevation angle [deg]
        @param RxID: receiver ID / serial number, to load expected receiver noise profiles.
        @param band: the band to assume for the engineering models.
        @return (eff_area [m^2], Trx [K], Tspill [K], Tsys [K], SEFD [Jy]) ordered as (freqs,pol 0=H,1=V)
    """
    f_MHz = freqs/1e6
    
    Tgal = Tgal(freqs)
    Text = np.asarray([Tcmb+Tgal+Tatm(freqs,el_deg)]*2)
    Tspill = models.Spill_Temp(band=band)(el_deg, f_MHz)
    Trec = np.asarray(models.get_lab_Trec(f_MHz, RxID))
    Tsys = Text+Tspill+Trec
    
    ant_eff = models.Ap_Eff(band=band)
    D = ant_eff.D
    Eff_area = ant_eff(f_MHz)/100 * (np.pi*D**2/4)
    SEFD = 2*_kB_*Tsys/Eff_area /1e-26
    
    return (Trec.transpose(), Tspill.transpose(), Text.transpose(), Tsys.transpose(), Eff_area.transpose(), SEFD.transpose()) # pol,freqs -> freqs,pol


def _get_SEFD_(vis, freqs, el_deg, MJD, bore,nulls, S_src, hpw_src=0, profile_src='gaussian', enviro={}):
    """ Returns the frequency spectrum of the deflection from a calibrator source.
        @param vis: the dataset recorded power
        @param freqs: corresponding to the second axis in vis [Hz]
        @param el_deg, MJD: elevation [deg] & modified Julian date of the observation [days].
        @param bore: time indices for source on bore sight, per frequency
        @param nulls: either specific indices or a function with arguments (vis,timestamps,frequencies) identifying the null window.
        @param S_src: 'lambda f_GHz,year' returning flux (HH, VV - corrected for parallactic angle) at the top of the atmosphere [Jy].
        @param hpw_src: half power width of the source as a fraction of HPBW [fraction] (default 0)
        @param profile_src: either 'gaussian' or 'disc' (default 'gaussian')
        @param enviro: a dictionary of "Enviro/air_*" metadata for atmospheric effects - simply use "ds.sensor"
        @return: (freqs, counts2Jy [per pol], SEFD [total power]) the latter ordered as (freqs,pol 0=H,1=V)
    """
    vis_on = np.nanmean(vis[bore,:,:], axis=0) # Mean over time
    if callable(nulls):
        vis_off = nulls(vis,None,freqs)
    else:
        vis_off = vis[nulls,:]
    vis_off = vis_off.mean(axis=0)
    
    # Flux is absorbed as it propagates through the atmosphere: S -> S*g_atm
    opacity_at_el = models.opacity(freqs,enviro)/np.sin(el_deg*np.pi/180.0)
    g_atm = np.exp(-opacity_at_el)
    # Source-to-beam coupling factor e.g. Baars 1973: S -> S*1/K
    if ('gauss' in profile_src.lower()): # Source has a Gaussian intensity distribution, such as Taurus A,Orion A
        K = 1 + hpw_src**2
    elif ('disc' in profile_src.lower()): # Source with a "top-hat" intensity distribution, such as the Moon, or Pictor A along constant declination
        K = hpw_src**2 / (1 - np.exp(-hpw_src**2)) # hpw_src=R/2/ln(2) is the radius
    
    yr = 1858+(365-31-14)/365.242+MJD/365.242 # Sufficiently accurate fractional year from MJD
    print("Scaling source flux for beam coupling, atmosphere at elevation %.f deg above horizon, and year %.2f"% (el_deg, yr))
    Corr = g_atm*1/K
    print("Scale factor between %.4f and %.4f"% (np.nanmin(Corr),np.nanmax(Corr)))
    S_src_at_strat = S_src(freqs/1e9, yr)
    S_src_at_ant = np.stack([Corr,Corr], -1) * S_src_at_strat # freqs,pol
    counts2Jy = S_src_at_ant/(vis_on-vis_off) # Counts 2 Jy per-pol
    SEFD_est = 2 * vis_off * counts2Jy # SEFD scaled back from per-pol to total power!
    
    return (freqs, counts2Jy, SEFD_est)


def _get_ND_(ds, counts2scale=None, y_unit="counts", freqmask=None, rfifilt=None, title=None, y_lim=None):
    """ Isolate Noise cycles to computes the ND spectra, possibly scaled. Generates a figure.
        @param counts2scale: if given (H spectrum, V spectrum) then ND spectra are scaled by this factor (default None)
        @param freqmask: list of frequency [Hz] ranges to omit from ON-OFF detection and from plots e.g. [(924.5e6,960e6)] (default None)
        @param rfifilt: size of smoothing windows in time & freq; time window is limited to < min(ON,OFF)/3 (default None)
        @param title: title for the figure
        @param y_lim: y limit for ND plot, as used by plot_data() (default None)
        @return S_ND (H&V spectra) either as a fraction of the background noise, or scaled by counts2scale. Ordered as (time,freq,pol)
    """
    chans = mask_where(np.full(ds.channel_freqs.shape,True), ds.channel_freqs, freqmask)
    S_ND = []
    for label,vis in ds.nd_vis.items():
        S_ND.append([])
        for pol in [0,1]:
            S = vis[1:-1,:,pol] # Each scan has ON & OFF; discard first & last samples because ND may not be aligned to dump boundaries
            m = np.nanmedian(S[:,chans],axis=1) > np.nanmedian(S[:,chans]) # ON mask where average total power over the cycle is above mean total power
            ON = np.compress(m, S, axis=0)[1:-1,:] # Observe script does not guarantee ND edges to be synchronised to dump boundaries
            OFF = np.compress(~m, S, axis=0)[1:-1,:] # It may be slow in getting on target (OFF is the first phase), but use strict=True for that case
            if (rfifilt is not None):
                if (rfifilt[0] > min(ON.shape[0], OFF.shape[0])/3.): # Limited to < shape[0]/3
                    rfifilt = (int(min(ON.shape[0], OFF.shape[0])/6.)*2-1, rfifilt[-1])
                ON = mat.smooth2d(ON, rfifilt, axes=(0,1))
                OFF = mat.smooth2d(OFF, rfifilt, axes=(0,1))
            ND_delta = np.mean(ON,axis=0) - np.mean(OFF,axis=0) # ON-OFF spectrum for this pol
            if (counts2scale is not None):
                ND_delta = ND_delta*counts2scale[:,pol]
            plot_data(ds.channel_freqs[chans]/1e6, ND_delta, label="%s, %s"%(ds._pol[pol],label), newfig=len(S_ND)+len(S_ND[-1])==1,
                      xtag="Frequency [MHz]", ytag="ND spectral density [%s]"%y_unit, header=title, y_lim=y_lim)
            S_ND[-1].append(ND_delta)
    
    if (len(S_ND) == 0): # No ND data
        S_ND = [[0*ds.channel_freqs, 0*ds.channel_freqs]]

    return np.moveaxis(S_ND, 1, 2) # time,pol,freq -> time,freq,pol


def get_SEFD_ND(ds,bore,nulls,win_len,S_src,hpw_src,profile_src,null_labels=None,rfifilt=None,freqmask=None,Tcmb=2.73,Tatm=None,Tgal=None):
    """ Computes spectra of SEFD and Noise Diode equivalent flux. Also generates expected SEFD given certain estimates.
        Returned values reflect SEFD for a BACKGROUND AVERAGED BETWEEN THE NULLS ENCOUNTERED BEFORE AND AFTER TRANSIT.
        
        @param bore: time indices for source on bore sight, per frequency
        @param nulls: lists of time indices per frequency for each null (source off bore sight)
        @param win_len: the number of time indices to use around bore sight & the nulls (-1/2,+1/2)
        @param S_src: source flux function ('lambda f_GHz,year' returning flux in [HH, VV] - corrected for parallactic angle) [Jy]
        @param hpw_src: half power width of the source as a fraction of HPBW [fraction]
        @param profile_src: either 'gaussian' or 'disc'.
        @param rfifilt: size of smoothing windows in time & freq (default None)
        @param freqmask: list of frequency [Hz] ranges to omit from plots e.g. [(924.5e6,960e6)] (default None)
        @param Tcmb: CMB temperature (not included in Tgal) (default 2.73) [K]
        @param Tatm: Atmospheric noise temperature as func(freq_Hz,el_rad) or None to use standard (default None) [K]
        @param Tgal: Sky background temperature (excluding Tcmb) as func(freq_Hz) [K]. Default None to call models.fit_bg(nulls).
        @return: (freqs [Hz], counts2Jy, SEFD_meas [Jy], SEFD_pred [Jy], Tsys_deduced from predicted Ae [K], Trx_deduced given predicted Tspill [K],
                              Tspill_deduced given LAB Trx, Tsys_pred, Trx_expct, Tspill_expct [K], S_ND [Jy], T_ND [K], elevation [deg]) - indexed by (freqs,polarization)
    """
    null_labels = null_labels if null_labels else [str(i) for i in range(10)]
    
    # Measured SEFD at each null
    print("\nDeriving measured SEFD")
    vis = ds.vis
    el_deg = ds.el_deg

    if rfifilt is not None:
        vis = mat.smooth2d(vis, rfifilt, axes=(0,1))
    
    counts2Jy, SEFD_meas = [], []
    for null in nulls:
        freqs, c2Jy, sefd = _get_SEFD_(vis, ds.channel_freqs, el_deg, ds.mjd,
                                       bore=bore, nulls=lambda v,t,f:getvis_null(v,null,win_len),
                                       S_src=S_src, hpw_src=hpw_src, profile_src=profile_src, enviro=ds.sensor)
        counts2Jy.append(c2Jy)
        SEFD_meas.append(sefd)
    counts2Jy = mask_where(counts2Jy, freqs, freqmask, axis=1) # time,freq,pol
    SEFD_meas = mask_where(SEFD_meas, freqs, freqmask, axis=1)
    
    # Predicted results for assumed background (single polarization)
    ant = ds.ant
    RxSN = ds.RxSN
    fTgal = lambda n: Tgal if Tgal else models.fit_bg(ds, np.nanmedian(nulls[n]), D=ant.diameter, debug=True)[0]
    if (Tatm is None):
        Tatm = lambda freqs, el_deg: 275*(1-np.exp(-models.opacity(freqs, ds.sensor)/np.sin(el_deg*np.pi/180))) # ITU-R P.372-11 suggests 275 as a typical number, NASA's propagation handbook 1108(02) suggests 280 K [p 7-8]
    pText, pTsys, pSEFD, Tsys_deduced = [], [], [], []
    for n in range(len(nulls)):
        print("\nPredicting SEFD at null "+null_labels[n])
        pTrx, pTspill, _pText, _pTsys, _pEffArea, _pSEFD = pred_SEFD(freqs,Tcmb,fTgal(n),Tatm,el_deg,RxSN,ds.band)
        pSEFD.append( _pSEFD )
        pTsys.append( _pTsys )
        pText.append( _pText )
        Tsys_deduced.append( SEFD_meas[n]*_pEffArea/2.0/_kB_*1e-26 )
    # pTrx, pTspill are the same for all nulls
    
    pSEFD = np.asarray(pSEFD)
    pTsys = np.asarray(pTsys)
    pText = np.asarray(pText)
    Tsys_deduced = np.ma.asarray(Tsys_deduced)
    
    # Further deduced quantities
    Trx_deduced = Tsys_deduced - pText - np.asarray([pTspill]*len(nulls))
    Tspill_deduced = Tsys_deduced - pText - np.asarray([pTrx]*len(nulls))
    
    pStyle = dict(marker="o", markevery=256, markersize=6)
    for n in range(len(nulls)):
        _ylim = _ylim_pct_(np.stack([pSEFD, SEFD_meas]), 2.5, 30, snap_to=10) # Range hides at most 5%
        plot_data(freqs/1e6, SEFD_meas[n,:,0], newfig=(n==0), label="H, null "+null_labels[n], color="C%d"%(2*n),
                  xtag="Frequency [MHz]", ytag="SEFD [Jy]", y_lim=_ylim)
        plot_data(freqs/1e6, SEFD_meas[n,:,1], newfig=False, label="V, null "+null_labels[n], color="C%d"%(2*n+1), **pStyle)
        plot_data(freqs/1e6, pSEFD[n,:,0], newfig=False, label="Expected H, null "+null_labels[n], color="C%d"%(2*n), style="--")
        plot_data(freqs/1e6, pSEFD[n,:,1], newfig=False, label="Expected V, null "+null_labels[n], color="C%d"%(2*n+1), style="--", **pStyle)
    
    # Average over nulls
    counts2Jy, SEFD_meas = np.ma.mean(counts2Jy,axis=0), np.ma.mean(SEFD_meas,axis=0)
    Tsys_deduced, Trx_deduced, Tspill_deduced = np.ma.mean(Tsys_deduced,axis=0), np.ma.mean(Trx_deduced,axis=0), np.ma.mean(Tspill_deduced,axis=0)
    pTsys, pSEFD = np.mean(pTsys,axis=0), np.mean(pSEFD,axis=0)

    # Also get the ND spectra, if there are
    S_ND = _get_ND_(ds, counts2scale=counts2Jy, y_unit="Jy", freqmask=freqmask, rfifilt=rfifilt, y_lim='pct')
    S_ND = np.nanmean(S_ND, axis=0) # Average over independent ND measurements -> freq,pol
    S_ND = mask_where(S_ND, freqs, freqmask, axis=0)
    
    T_ND = S_ND * Tsys_deduced/(SEFD_meas/2.) # Remembering that SEFD _per pol_ is scaled by x2 while neither ND nor Tsys is
    
    return (freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_deduced, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg)


def getvis_null(vis, null_indices, win_len=4, debug=False):
    """ @param vis: visibilities over the frequency range of interest
        @param null_indices: vector of time index vs freq (same range as "vis")
        @param win_len: the number of indices of the window (-1/2,+1/2) for data that's returned per frequency
        @return: the vis data selected from the time window corresponding to the computed range of time indices. """
    assert vis.shape[1] == null_indices.shape[0], "vis frequency range not the same as that of null_indices!"
    
    n_a = null_indices - win_len//2
    n_z = null_indices + (win_len-win_len//2)
    _valid_freq_ = np.arange(vis.shape[1])[np.isfinite(null_indices) & (n_a >= 0) & (n_z <= vis.shape[0])]
    if debug: # Show how much the values change over the selected time & frequency points, in this null
        plt.figure(figsize=(12,4))
        plt.title("Stability over selected null window")
        for c in _valid_freq_:
            z = vis[int(n_a[c]):int(n_z[c]),c,:].mean(axis=1)
            plt.plot(range(z.shape[0]), z/z.mean(axis=0), '.-')
        plt.xlabel("Time [sample]"); plt.ylabel("Normalised Power, per frequency bin"); plt.grid(True)
    
    vis_null = np.full((vis.shape[1], win_len, vis.shape[2]), np.nan) # Ordered as [freq,time,prod]
    for c in _valid_freq_:
        vis_null[c] = vis[int(n_a[c]):int(n_z[c]),c,:]
    return np.moveaxis(np.asarray(vis_null), 1, 0) # Re-order to [time,freq,prod]


def find_nulls(ds, cleanchans=None, HPBW=None, N_bore=-1, Nk=[1.292,2.136,2.987,3.861], hpw_src=0, debug_level=0):
    """ Finds time indices when the drift scan source passes through nulls and across bore sight.
        All indices are given relative to the time axis set up by '__select_SEFD__()'.
        
        Nulls are computed from the theoretical relationship between nulls and HPBW of the parabolic illumination function.
        HPBW is derived from the dataset itself, and where no feasible values are found, the specified HPBW (whether numeric or a function)
        is used to fill in the gaps. 
        
        @param cleanchans: used to select the clean channels to use to fit bore sight transit and beam widths on
        @param HPBW: 'lambda f: x' or 1d array [rad] to override fitted widths from the dataset (default None)
        @param N_bore: the minimum time samples to average over the bore sight crossing, or <0 to use average of <HPBW>/16 (default -1).
        @param Nk: beam factors that give the offsets from bore sight of the nulls relative, in multiples of HPBW
                   (default [1.292,2.136,2.987,3.861] as computed from theoretical parabolic illumination pattern)
        @param hpw_src: the equivalent half-power width [rad] of the target (default 0)
        @param debug_level: 0 for no debugging, 1 for some, 2 to add others; 9 is not for regular use (default 0)
        @return: (bore, nulls_before_transit, nulls_after_transit, HPBW_fitted, N_bore).
                 'bore' is the time indices while the target crosses bore sight, against frequency (1D).
                 each null_..transit is the time indices while the target is crossing that null, vs (k'th null, frequency) (2D).
                 'HPBW_fitted' gives the half power beam widths [rad] employed, 1D across frequency.
                 N_bore is the final window length to employ around bore sight & nulls.
    """
    t = np.arange(len(ds.timestamps)) # [samples]
    
    # Find the time of transit ('bore') and the beam widths ('HPBW') at each frequency
    print("INFO: Fitting transit & beam widths from the data itself.")
    # To speed up, fit only to 64 frequency channels
    beamfits = load4hpbw(ds, n_chunks=64, cleanchans=cleanchans, return_all=debug_level>0, debug=3)
    f, bore, sigma = beamfits[:3]
    HPBW_fitted = fit_hpbw(f, bore, sigma, ds.ant.diameter, hpw_src=hpw_src, fitchans=cleanchans, debug=debug_level>0)
    sigma2hpbw = np.sqrt(8*np.log(2)) * (2*np.pi)/(24*60*60.) # rad/sec, as used in fit_hpbw()
    
    if (HPBW is None): # HPBW not forced, so use the fitted positions, filling it in with the fitted function where it is masked
        HPBW = np.nanmean(sigma*sigma2hpbw, axis=1) # Average over products
        mask = np.any(sigma.mask, axis=1) # Mask over frequency (collapse products)
        _HPBW = HPBW_fitted(f)
        HPBW[mask] = _HPBW[mask]
    elif callable(HPBW): # Forced as a function
        HPBW = np.vectorize(HPBW)(f)
    
    # Interpolate 'bore' where it is masked, and convert to time samples relative to current selection i.e. timestamp[0]
    bore = np.apply_along_axis(lambda b:mat.interp(f, f[~b.mask], b[~b.mask], "cubic", bounds_error=False, fill_value=np.nanmedian(b[~b.mask])), 0, bore)
    T0 = ds.timestamps[0] - float(ds.start_time)
    # TODO: mathematical operations are NOT performed on masked values!!! This is different from what I expected, so consider removing all use of masked array!
    bore = np.clip((bore.data-T0)/ds.dump_period, t[0], t[-1]) # [samples]
    
    t_bore = int(np.ma.mean(bore) + 0.5) # Representative sample of bore sight transit
    N_bore = max(N_bore, int(np.nanmedian(HPBW)/(sigma2hpbw*ds.dump_period) / 16.)+1) # The beam changes < 1% within +-HPBW/8 interval
    print("Transit found at relative time sample %d; averaging %d time samples at each datum." % (t_bore, N_bore))
    
    # Find time indices when the source crosses the k-th null at each frequency
    D2R = np.pi/180.
    antaz, antel = ds.az[:,0]*D2R, ds.el[:,0]*D2R # deg->rad, for selected ant=0
    ll, mm = ds.target.sphere_to_plane(antaz,antel,timestamp=ds.timestamps,projection_type="SSN",coord_system="azel") # rad
    angles = 2*np.arcsin((ll**2+mm**2)**0.5 / 2.) # Pointing offset [rad] of target relative to mechanical axis over time. For small angles this is ~ (ll**2+mm**2)**0.5
    if (debug_level > 1):
        # Show pointing & target track - TODO sometimes shows up empty?
        # tgtaz,tgtel = target.azel(ds.timestamps)
        # plot_data(np.unwrap(tgtaz)/D2R,tgtel/D2R, label="Target", xtag="Az [deg]", ytag="El [deg]")
        # plot_data(antaz/D2R,antel/D2R, label="Bore sight", style='x', newfig=False)
        # plt.axes().set_aspect('equal', 'datalim')

        plot_data(t, np.asarray(angles)/D2R, header="Target trajectory with respect to bore sight and nulls",
                  xtag="Time [samples]", ytag="Distance from bore sight [deg]")
        cleanchan = ds.channels[23] if (cleanchans is None) else ds.channels[cleanchans][23] # Arbitrarily choose one
        for n in [0,1]: # target in first two nulls vs. time, for some clean channel
            flags_ch = np.abs(angles-Nk[n]*HPBW[cleanchan])<0.1*D2R
            plot_data(t[flags_ch], angles[flags_ch]/D2R, style='.', label="Null %d @ channel %d"%(n,cleanchan), newfig=False)
    # Use angles from bore sight and Nk relationship to identify nulls
    mn = lambda x: np.nan if len(x)==0 else np.mean(x) # Because np.min fails on empty, np.mean returns empty & issues a warning message
    # From here on we ignore any difference in bore sight direction between the polarisations
    bore = np.asarray(np.nanmean(bore.data, axis=1) + 0.5, int)
    DT = bore - t_bore
    find_null = lambda t,angles,k: np.asarray([mn(t[np.abs(angles-Nk[k]*hpbw)<hpbw/20.])+dt for dt,hpbw in zip(DT,HPBW)]) # Must subset t & angles to avoid this getting both left & right!
    null_l = [find_null(t[t<t_bore],angles[t<t_bore],k) for k in range(len(Nk))] # (k'th null, frequency)
    null_r = [find_null(t[t>t_bore],angles[t>t_bore],k) for k in range(len(Nk))]

    if (debug_level > 0): # Plot the intensity map along with the identified positions of the nulls
        vis = ds.vis
        
        bl, bm = beamfits[3:]
        vis_nb = (vis - bl) / np.ma.max(bm, axis=0) # "Flattened" and normalised to fitted beam height 
        levels = [-0.1,-0.05,0,0.05,0.1] # Linear scale, fraction
        axes = plt.subplots(1, 2, sharex=True, figsize=(12,6))[1]
        axes[0].set_title("Target contribution & postulated nulls, %s (left) & %s (right). Contour spacing 0.05    [peak fraction]"%(*ds._pol,), loc='left')
        for p,ax in enumerate(axes):
            im = ax.imshow(vis_nb[...,p], origin='lower', extent=[f[0]/1e6,f[-1]/1e6,t[0],t[-1]], aspect='auto',
                           vmin=-0.15, vmax=0.15, cmap=plt.get_cmap('viridis'))
            ax.contour(vis_nb[...,p], origin='lower', extent=[f[0]/1e6,f[-1]/1e6,t[0],t[-1]], levels=levels[1:-1], colors='C0', alpha=0.5)
            ax.plot(f/1e6, bore, 'k')
            for null in null_l+null_r:
                ax.plot(f/1e6, null, 'k')
            ax.set_xlabel("Frequency [MHz]")
        axes[0].set_ylabel("Time [indices]")
        plt.colorbar(im, ax=axes[-1])
    
        if (debug_level >= 9): # Not for regular use!
            for k in [0,1]: # Check first two nulls are well defined
                getvis_null(vis, null_l[k], N_bore, debug=True)
                getvis_null(vis, null_r[k], N_bore, debug=True)
                _debug_stats_(vis, ds.channel_freqs, ds.timestamps, bore, (null_l[k], null_r[k]), N_bore, title="%dth Nulls"%k)

    return bore, null_l, null_r, HPBW, N_bore


def _debug_stats_(vis, channel_freqs, timestamps, bore_indices, nulls_indices, win_len, title=None):
    """ Plots sigma/mu spectra for bore sight & off-source & compare to expected value
    
        @param bore_indices: time indices for bore sight data per frequency (1D), relative to selection of '__select_SEFD__()'.
        @param nulls_indices: lists of time indices per frequency for each null (source off bore sight)
    """
    def plotFreq(ax, freqrange, select_dumps=None, title=None):
        """ Produces frequency spectrum plots of the currently selected dumps.
            @param freqrange: frequency start&stop to subselect
            @param select_dumps: None for all time dumps, an integer range or a function(vis,timestamps,frequencies) to allow sub-selection of time interval
            @return: freq [Hz], the average spectra [linear]
        """
        viz = np.array(vis, copy=True)
        if select_dumps is not None:
            if callable(select_dumps):
                viz = select_dumps(viz, timestamps, channel_freqs)
            else:
                viz = viz[select_dumps,:]
        
        chans = _idx(channel_freqs, freqrange)
        x_axis = channel_freqs[chans]
        xlabel = "f [MHz]"
        
        viz = viz[:,chans,:] # Always restrict freq range after select_dumps
        vis_mean = np.nanmean(viz, axis=0)
        
        vis_bars = np.dstack([vis_mean-np.nanmin(viz,axis=0), np.nanmax(viz,axis=0)-vis_mean]).transpose()
        ax.set_title(title)
        for p in range(viz.shape[2]):
            ax.errorbar(x_axis/1.0e6, vis_mean[:,p], yerr=vis_bars[:,p], fmt='-', capsize=1, errorevery=30)
        ylim = _ylim_pct_(vis_mean, 2.5, 30)
        ax.set_xlabel(xlabel); ax.set_ylabel("Radiometer counts"); ax.grid(True); ax.set_ylim(ylim)
        return x_axis, vis_mean

    tau = timestamps[1]-timestamps[0]
    BW0 = abs(channel_freqs[1]-channel_freqs[0])
    freqrange = [np.min(channel_freqs)+BW0,np.max(channel_freqs)-BW0]
    
    fig, axs = plt.subplots(3,1, figsize=(12,8))
    fig.suptitle(title)
    # Plot statistics of bore sight data
    bore_indices = np.array(np.nanmedian(bore_indices) + np.arange(-win_len//2,win_len//2), dtype=int)
    freqs, Son = plotFreq(axs[0], freqrange=freqrange, select_dumps=bore_indices,
                          title="Spectrum with source on bore sight")
    
    # Plot statistics of off-source data
    for null in nulls_indices:
        freqs, Soff = plotFreq(axs[1], freqrange=freqrange,
                               select_dumps=lambda v,t,f:getvis_null(v,null,win_len),
                               title="Spectrum with source in null away from bore sight")
    
    Psrc = (Son-Soff)
    Psys = Soff
    # In the limit Tsrc << Tsys, sigma = Tsys/sqrt(BW*tau), so sigma/mu = 1/sqrt(BW*tau) = const   (per polarization)
    # However, in case Tsrc >> Tsys we must use the complete sigma = sqrt(Tsys^2+2*Tsys*Tsrc+2*Tsrc^2)/sqrt(BW*tau),
    #   so expect bore sight sigma/mu = sigma/(Tsys+Tsrc) > 1/sqrt(BW*tau)
    _K = lambda Tsys,Tsrc: np.sqrt(Tsys**2+2*Tsys*Tsrc+2*Tsrc**2)/(Tsys+Tsrc)
    K = _K(Psys,Psrc).mean(axis=0)
    print("Average bore sight noise factor K=%s" % K)
    axs[2].set_title("std/mu expected from ratios of mean on & off spectra")
    axs[2].plot(np.dstack([freqs/1e6]*K.shape[0]).squeeze(), _K(Psys,Psrc)/np.sqrt(BW0*tau))
    axs[2].set_xlabel("f [MHz]"); axs[2].set_ylabel(r"$\frac{\sigma}{\mu}$"); axs[2].grid(True)
    axs[2].set_ylim([0,3/np.sqrt(BW0*tau)])
    
    print("Expected boresight std/mu = %s at %.f MHz" % (K/np.sqrt(BW0*tau), np.average(freqrange)/1e6)) # May have a slope if src flux slopes
    print("Expected off-source std/mu = %s at %.f MHz" % (1/np.sqrt(BW0*tau), np.average(freqrange)/1e6)) # Must not have a slope


def target_offset(target, timestamp, az, el, freq, label="observation", debug=True):
    """ Determines the offset of the source at the specified time, relative to the pointing direction of the
        specified antenna.
        
        @param target: the katpoint.Target, with the observing antenna set if you have debug enabled.
        @param timestamp: the specified time instant [UTC seconds]
        @param ant: the katpoint.Antenna acting as observer.
        @param az, el: azimuth & elevation look angles [deg] of the antenna.
        @param freq: the frequency [Hz] to use for determining the antenna's beam width.
        @return on_delta: the offset of the source from bore sight [deg]
    """
    t_obs = katpoint.Timestamp(timestamp)
    ant = target.antenna
    on_sky = np.asarray(target.radec(t_obs, antenna=ant))*180/np.pi
    on_exp = np.asarray(target.azel(t_obs))*180/np.pi
    on_rec = np.squeeze([az, el])
    
    on_delta = np.squeeze([np.abs(on_exp[0]-on_rec[0]), np.abs(on_exp[1]-on_rec[1])])
    on_delta[0] = np.angle(np.exp(1j*on_delta[0]*np.pi/180)) * 180/np.pi # Remove 360deg ambiguity from Az angle
    if debug:
        print("UTC for %s of target: %s" % (label, t_obs))
        print("    Antenna pointing to RA %.3f hrs, DEC %.3f deg" % (on_sky[0]*24/360., on_sky[1]))
        print("    Target Az,El: %s deg"%(on_exp))
        print("    Antenna Az,El: %s deg"%(on_rec))
        HPBW = 1.22*(_c_/freq)/ant.diameter *180/np.pi # Standard illumination of circular aperture
        print("    Source within %s deg of beam bore sight (<%.3f HPBW)" % (on_delta,np.max(np.abs(on_delta)/HPBW)))
        # Report on the proximity to some special targets
        for special in ["Sun", "Moon"]:
            tgt = katpoint.Target("%s, special"%special.lower(), antenna=ant)
            print("    Distance from target to the %s %.f deg" % (special, tgt.separation(target, t_obs)*180/np.pi))
    return np.max(on_delta)


def combine(x, y, x_common=None, pctmask=100):
    """ Re-grid all results onto a common range.
    
        @param x: list of N abscissa values for each result set to combine
        @param y: list of N result sets (x[n] & y[n] dimensions must match)
        @param x_common: abscissa of final combined result set or None to construct automatically (default None)
        @param pctmask: mask points which deviate more than this percentage from a smooth curve (default 100)
        @return: x_combined, y_combined
    """
    f, _s = [], []
    for x_ in x:
        f.extend(x_); f.extend([np.nan]) # Nan's help avoid discontinuities in diff(f) below
        _s.append(1 if x_[1]>x_[0] else -1) # keep track of ordering since np.interp can't cope with descending abscissa
    if (x_common is None):
        df = np.nanmin(np.abs(np.diff(f)))
        f = np.arange(np.nanmin(f), np.nanmax(f)+df, df)
    else:
        f = x_common
    
    # Almost certainly the edge channels are bad, so omit it on both ends
    B = 2
    y_combi = [mat.interp(s*f, s*fp[B:-B], yp[B:-B], left=np.nan, right=np.nan) for s,fp,yp in zip(_s,x,y)]
    
    if (pctmask>0 and pctmask<100): # Mask out 2D data which is too far off the expected smooth curve along the second axis.
        diff = np.abs(np.diff(y_combi, axis=1))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in", category=RuntimeWarning)
            mask = [np.r_[np.nan, diff[p,:]] > np.nanpercentile(diff[p,:], pctmask) for p in range(len(y_combi))]
        y_combi = np.ma.masked_array(y_combi, mask|np.isnan(y_combi), fill_value=np.nan)
        
    return f, y_combi


def summarize(results, labels=None, pol=["H","V"], pctmask=100, freqmask=None, plot_singles=True, plot_predicted=False, plot_ratio=False):
    """ Plot results against each other for comparison & combines results onto a common frequency range.
    
        @param results: a list of result sets, each set as generated by get_SEFD_ND(), i.e.
           [freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND ...] 
        @param pctmask: when generating the combined results, mask points which deviate more than this percentage from a smooth curve (default 100)
        @param freqmask: list of frequency [Hz] ranges to omit from plots e.g. [(924.5e6,960e6)] (default None)
        @return: f [Hz], SEFD [Jy], TSYS [K], ND [Jy], TND [K] (all but f is orderd as (freq,pol) with pol=[H,V]) 
    """
    labels = labels if labels else ["%d"%n for n in range(len(results))]
    prods = ["%s pol"%p for p in pol]
    
    def make_figure(tag, header=None):
        nrows = len(prods)
        fig, axes = plt.subplots(nrows,1, figsize=(12,nrows*6))
        fig.suptitle(header)
        for p,ax in zip(prods,np.atleast_1d(axes)):
            ax.set_ylabel(p+" "+tag); ax.grid(True);
        ax.set_xlabel("f [MHz]")
        return fig, axes
    
    def plot_single_and_combi(results, m_index, p_index, tag, labels, header=None, y_lim='pct'):
        # Plots measured & predicted data fom 'results', first individual measurements on one figure, then another figure with average &  predicted
        if plot_singles:
            _ypct =  [int(i) for i in (y_lim+",5,30").split(",")[1:3]] if (y_lim and 'pct' in y_lim) else None
            axes = make_figure(tag, header)[1]
            for p,ax in enumerate(axes):
                for n,x in enumerate(results):
                    ax.plot(x[0]/1e6, x[m_index][:,p], label=labels[n])
                
                ax.set_ylim(_ylim_pct_(x[m_index][:,p],*_ypct) if (_ypct is not None) else y_lim)

        # Re-grid all results onto a common range & combine
        f, m_h = combine([x[0] for x in results], [x[m_index][:,0] for x in results], pctmask=pctmask) # Mask out some interference
        f, m_v = combine([x[0] for x in results], [x[m_index][:,1] for x in results], f, pctmask=pctmask)
        _h, _v = np.ma.mean(m_h,axis=0), np.ma.mean(m_v,axis=0)
        _hv = mask_where(np.ma.dstack([_h, _v]).squeeze(), f, freqmask)

        ### Plot results with error bars to show the range
        _h, _v = _hv[:,0], _hv[:,1]
        bars = {} if (len(m_h) == 1) else dict(bars=[np.ma.max(m_h,axis=0)-_h, _h-np.ma.min(m_h,axis=0)], errorevery=128, capthick=3)
        plot_data(f/1e6, _h, label="Measured "+pol[0], xtag="Frequency [MHz]", ytag=tag, header=header, **bars)
        bars = {} if (len(m_v) == 1) else dict(bars=[np.ma.max(m_v,axis=0)-_v, _v-np.ma.min(m_v,axis=0)], errorevery=128, capthick=3)
        plot_data(f/1e6, _v, label="Measured "+pol[1], newfig=False, y_lim=y_lim, **bars)

        if (p_index == m_index):
            p_hv = _hv
        else: # Combine the predicted values in the same way as above
            f, p_h = combine([x[0] for x in results], [x[p_index][:,0] for x in results], f)
            f, p_v = combine([x[0] for x in results], [x[p_index][:,1] for x in results], f)
            p_h, p_v = np.mean(p_h,axis=0), np.mean(p_v,axis=0)
            p_hv = np.dstack([p_h, p_v]) # Not necessary to mask the predicted values
            if (plot_predicted) and np.any(np.isfinite(p_hv)):
                pStyle={"linestyle":"--", "markevery":256, "markersize":8}
                plot_data(f/1e6, p_h, label="Predicted "+pol[0], newfig=False, y_lim=y_lim, marker="^", **pStyle)
                plot_data(f/1e6, p_v, label="Predicted "+pol[1], newfig=False, marker="v", **pStyle)
        
        return f, _hv, p_hv
    
    
    # counts2Jy
    axes = make_figure("Gain [Jy/#]")[1]
    for p,ax in enumerate(axes):
        for n,x in enumerate(results):
            y = mask_where(x[1][:,p], x[0], freqmask)
            ax.plot(x[0]/1e6, y, label=labels[n])
        ax.legend(); ax.set_ylim(_ylim_pct_(y,10,30))
 
    if plot_ratio: # Gain ratios, relative to the first result (but may fail)
        fig, axes = make_figure("#2Jy Gain ratio")
        try:
            for p,ax in enumerate(axes):
                for n,x in enumerate(results[1:]):
                    y = mask_where(x[1][:,p]/results[0][1][:,p], x[0], freqmask)
                    ax.plot(x[0]/1e6, y, label="%s/%s"%(labels[n+1],labels[0]))
                ax.legend(); ax.set_ylim(_ylim_pct_(y,10,30))
        except: # May fail if frequency ranges don't match - then just ignore this step
            plt.close(fig)
    
    # SEFD index=2, pSEFD index=3
    f, m_SEFD, p_SEFD = plot_single_and_combi(results, 2, 3, "SEFD [Jy]", labels)
    
    # Ae/Tsys
    plot_single_and_combi([[r[0], 2*_kB_/1e-26/r[2], 2*_kB_/1e-26/r[3]] for r in results],
                          1, 2, "$A_e/T_{sys}$ [m$^2$/K]", labels)
    
    # Noise Diode index=10
    f, m_ND, p_ND = plot_single_and_combi(results, 10, 10, "ND [Jy]", labels)
    
    header = "From measurement, with assumed antenna efficiency"
    # TND
    f, m_TND, p_TND = plot_single_and_combi(results, 11, 11, "$T_{ND}$ [K]", labels, header)
    
    # Tsys
    f, m_TSYS, p_TSYS = plot_single_and_combi(results, 4, 7, "$T_{sys}$ [K]", labels, header)
    
    # Tsys residuals (measured - predicted)
    axes = make_figure("$T_{sys}$(meas)-$T_{sys}$(expect) [K]", header)[1]
    for p,ax in enumerate(axes):
        for n,x in enumerate(results):
            ax.plot(x[0]/1e6, mask_where(x[4][:,p]-x[7][:,p],x[0],freqmask), label=labels[n])
        ax.set_ylim(-6,6)
    
    # Trec
    plot_single_and_combi(results, 5, 8, "$T_{rec}$ [K]", labels, header, 'pct,50,100') # i.e. median +-100%
    
    # Tspill + Tremainder (reflector ohmic + leakage, post-receiver contribution)
    plot_single_and_combi(results, 6, 9, "$T_{spill}+T_{rem}$ [K]", labels, header, 'pct,50,100')

    return f, m_SEFD, m_TSYS, m_ND, m_TND


def analyse(f, ant=0, source=None, flux_key=None, cat_file=None, ant_rxSN={}, swapped_pol=False, strict=False,
            HPBW=None, N_bore=-1, Nk=[1.292,2.136,2.987,3.861], nulls=[(0,0)],
            fitfreqrange=None, freqmask=[(0,200e6),(360e6,380e6),(924e6,960e6),(1084e6,1092e6)], rfifilt=[1,7],
            saveroot=None, makepdf=False, debug=False, debug_nulls=1):
    """ Generates measured and predicted SEFD results and collects it all in a PDF report, if required.
        
        @param f: filename string, or an already opened katdal.Dataset or DriftDataset
        @param ant, ant_rxSN, swapped_pol, strict: to be passed to 'DriftDataset()'.
        @param source: a description of the calibrator source (see 'models.describe_source()'), or None to use the defaults defined for the drift scan target.
        @param flux_key: an identifier for the source flux model, passed to 'models.describe_source()' (default None).
        @param cat_file: if given, overrides the default filename used by 'models.describe_source()' (default None).
        @param HPBW: if given, like 'lambda f: 1.2*(3e8/f)/D' [rad], this is used instead of HPBW fitted from the data, fore selection of data in beam nulls (default None).
        @param N_bore: the minimum time samples to average over the bore sight crossing, or <0 to use average of <HPBW>/16 (default -1).
        @param Nk: beam factors that give the offsets from bore sight of the nulls relative, in multiples of HPBW
                   (default [1.292,2.136,2.987,3.861] as computed from theoretical parabolic illumination pattern)
        @param nulls: pairs of indices of nulls to generate results for, zero-based or None and as (prior to, post transit) (default [(0,0)]).
        @param fitfreqrange: frequency range (start,stop)[Hz] to be used to fit the beam model, or None for all (default None).
        @param freqmask: list of 2-vectors for frequency bands to mask out for fit and in results (default covers 0-200MHz, MUOS(370MHz), GSM(930MHz) & SSR (1090MHz)).
        @param rfifilt: median filter lengths for final de-noising over the time & frequency axes (default [1,7]).
        @param saveroot: root folder on filesystem to save files to (default None).
        @param debug_nulls: 1 to plot null traces, 2 to plot advanced statistics (default 1).
        @return: same products as get_SEFD_ND() + [offbore_deg]
    """
    # Select all of the raw data that's relevant
    if (isinstance(f, DriftDataset) and (ant in [0, f.ant.name])): # Avoid re-loading the data - doesn't work if ant is an index. 
        ds = f
    else:
        ds = DriftDataset(f, ant, swapped_pol=swapped_pol, strict=strict, ant_rx_override=ant_rxSN, debug=debug)
    vis = ds.vis
    filename = ds.name
    ant = ds.ant
    source = source if source else ds.target.name
    
    # Combine fitfreqrange and freqmask
    fitchans = _idx(ds.channel_freqs, fitfreqrange)
    fitchans &= ~mask_where(fitchans, ds.channel_freqs, freqmask).mask
    
    pp = PDFReport("%s_%s_driftscan.pdf"%(filename.split(".")[0], ant.name), save=makepdf)
    try:
        pp.capture_stdout(echo=True)
        print("Processing drift scan %s on %s with receiver %s." % (filename, ant.name, ds.RxSN))
        if swapped_pol:
            print("Note: The dataset has been adjusted to correct for a polarisation swap!")
        print("")
        
        src_ID, hpw_src, profile_src, S_src = models.describe_source(source, flux_key, verbose=True, **({'cat_file':cat_file} if (cat_file) else {}))
        pp.header = "Drift scan %s of %s on %s"%(filename, src_ID, ant.name)
        
        # Plot the raw data, integrated over frequency, vs relative time
        F = np.max([0]+plt.get_fignums())
        plt.figure(figsize=(12,6))
        fitMHz = ds.channel_freqs[fitchans]/1e6
        plt.title("Raw drift scan time series, %g - %g MHz" % (np.min(fitMHz), np.max(fitMHz)))
        plt.plot(np.arange(vis.shape[0]), np.nanmean(vis[:,fitchans,:], axis=1)); plt.grid(True)
        plt.ylabel("Radiometer counts"); plt.xlabel("Sample Time Indices (at %g Sec Dump Rate)" % ds.dump_period)
        pp.report_fig(F+1, orientation="landscape")
    
        # Identify the bore sight and null transits
        bore, null_l, null_r, _HPBW, N_bore = find_nulls(ds, cleanchans=fitchans, HPBW=HPBW, N_bore=N_bore, Nk=Nk, hpw_src=hpw_src, debug_level=debug_nulls)
        F = np.max(plt.get_fignums())
        if (debug_nulls>0):
            # 1-> (mu sigma vs time, HPBW fit, map null traces), 2-> (baselines, model residuals, mu sigma vs time, HPBW fit, trajectory, map null traces)
            pp.report_fig(F-(1 if (debug_nulls==1) else 2), orientation="landscape") # Beamwidth fit
            pp.report_fig(F, orientation="landscape") # Map of null traces
    
        # Correct for transit offset relative to bore sight
        _bore_ = int(np.median(bore)) # Calculate offbore_deg only for typical frequency, since offbore_deg gets slow 
        offbore_deg = target_offset(ds.target, ds.timestamps[_bore_], ds.az[_bore_], ds.el[_bore_], np.mean(ds.channel_freqs[fitchans]), "bore sight transit", debug=True)
        hpbw0 = np.nanmedian(_HPBW[fitchans]); hpbwf0 = np.median(ds.channel_freqs[np.abs(_HPBW/hpbw0-1)<0.01])
        offbore0 = offbore_deg*np.pi/180/hpbw0
        C = models.G_bore(offbore0, hpbwf0, ds.channel_freqs)
        print("Scaling source flux for pointing offset, by %.3f - %.3f over frequency range"%(np.max(C), np.min(C)))
        
        par_angle = ds.parangle[_bore_] * np.pi/180
        Sobs_src = lambda f_GHz, yr: S_src(f_GHz, yr, par_angle) * models.G_bore(offbore0, hpbwf0/1e9, np.reshape(f_GHz, (-1,1)))
        
        nulls_l = [N[0] for N in nulls if N[0] is not None]
        nulls_r = [N[1] for N in nulls if N[1] is not None]
        null_groups = [null_l[N] for N in nulls_l] + [null_r[N] for N in nulls_r]
        null_labels = ["k%d"%N for N in nulls_l] + ["K%d"%N for N in nulls_r]
        print("\nNow determining measured and predicted SEFD with target in beam nulls:")
        print("    %s before transit & %s after transit" % (null_labels[:len(nulls_l)], null_labels[len(nulls_l):]))
        freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg = \
                get_SEFD_ND(ds,bore,null_groups,N_bore,Sobs_src,hpw_src/_HPBW,profile_src,null_labels=null_labels,rfifilt=rfifilt,freqmask=freqmask)
        F = np.max(plt.get_fignums())
        pp.report_fig([F-1, F], orientation="landscape") # get_SEFD_ND()-> (SEFD, ND flux)
        
        result = [freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg, offbore_deg]
        summarize([result], pol=ds._pol, freqmask=freqmask, plot_singles=False, plot_predicted=True, plot_ratio=False)
        # (plot_singles=False,plot_ratio=False)-> (counts2Jy, SEFD, Ae/Tsys, ND_Jy, TND, Tsys, Tsys resid, Trec, Tspill)
        pp.report_fig([F+1+i for i in [1,2,4,5,6]], orientation="landscape")
        
        duration = (ds.timestamps[-1]-ds.timestamps[0])/60.
        pp.report_text(r"""
        SEFD is determined from a %.f minute long drift scan of a suitable celestial calibrator.
        
        The deflection measured by the calibrator source is a spectrum formed from
                            $\Delta = <BORE>-<NULL_k>$,
        where $NULL_k$ is the $k^{th}$ null adjacent to $BORE$ which represents the bore sight transit of the source. The
        angle brackets represent averaging over a time window of typically $<HPBW>/16$. The nominal bore sight
        transit time is determined as a single time instant for the entire dataset, while the time instant of each null
        is determined independently for each frequency channel, as follows.
        
        Time is mapped to angular offset of the source from the reported bore sight pointing coordinates. For the
        results in this report, the angular offsets of the nulls have been assumed to appear at
                            $\Theta_{\pm k} = \pm \{%s\}_{[k]} \times HPBW$
        For reference, on a circular aperture illuminated by a parabolic taper, these coefficients are expected to be
        {1.292,2.136,2.987,3.861}.
        
        For each polarisation channel the measured SEFD is determined from the following:
                            $\frac{SEFD_{perpol}}{S_{src@antenna}} = \frac{P_{NULL_k}}{P_{BORE}-P_{NULL_k}} \times 2$
        The source flux per polarisation channel $S_{src}$ is derived from total intensity $I_{src}$, polarization fraction $p$, and
        polarisation angle $\Phi$, at the top of atmosphere, as
                            $S_H = \frac{1}{2}I_{src}(1-p\cos(2\Phi))$
                            $S_V = \frac{1}{2}I_{src}(1+p\cos(2\Phi))$
        
        CAUTION! $SEFD_{perpol}$ above contains a scaling of x2 so that it is related to $A_e/T_{sys}$ by the conventional
                            $SEFD_{perpol} = \frac{2 k_B T_{sys}}{A_e}$
        $T_{sys}$ results in this report are derived from the above using the measured $SEFD_{perpol}$ and modeled $A_e$, from
                            $T_{sys,pol} = \frac{SEFD_{perpol} \times 10^{-26}}{2 k_B} A_e$
        
        Source flux $S_{src}$ is propagated from a reference plane above the atmosphere to the antenna plane by  
                            $S_{src@antenna} = S_{src} e^\frac{-\tau}{\sin(el)} U(\varepsilon_\theta) \frac{\Omega_\Sigma}{\Omega_{src}}$
        The additional terms above are
            * atmospheric extinction with $\tau$ the estimated atmospheric opacity,
            * gain correction $U(\varepsilon_\theta)$ for estimated pointing offset at bore sight transit and
            * source-to-beam coupling correction factor $\frac{\Omega_\Sigma}{\Omega_{src}}$; $\Omega_\Sigma$ is the beam-weighted source solid angle.
        
        At frequencies below 1.7 GHz the atmospheric extinction is typically less than 0.5%%. All of the sources employed
        have half-power widths less than 4 arcmin so that the beam coupling factor is less than 1%% at 1.7 GHz.
        These correction factors as well as the correction applied to compensate for pointing error, are reported in the
        processing logs.
        
        The predicted SEFD is calculated as
                            $\widehat{T}_{sys} = T_{CMB} + T_{GSM} + T_{atm} + T_{spill} + T_{rx}$
                            $\widehat{SEFD}_{perpol} = \frac{2 k_B \widehat{T}_{sys}}{A_e}$
        where $T_{GSM}$ is obtained from Global Sky Model maps and $T_{atm}$ is determined as per ITU-R P.676-9 & Ippolito
        1989 (eq 6.8-6). $T_{spill}$ and $A_e$ have been computed by full-wave EM techniques applied to the as-built model
        for the reflector system. $T_{rx}$ is the input-referred noise that's measured at the factory for each specific receiver.
        
        The "noise diode equivalent flux" is measured while briefly tracking the calibrator and determined as
                            $S_{ND,pol} = (P_{ND_ON}-P_{ND_OFF}) \frac{S_{src@antenna}}{P_{BORE}-P_{NULL_k}}$
        The above may be scaled to an equivalent noise temperature at the waveguide port, by the following relation
                            $T_{ND,pol} = \frac{S_{ND,pol} \times 10^{-26}}{k_B} A_e$
        """%(duration, str(Nk)[1:-1]), fontsize=9)
        
        # The captured stdout logs go on the last page
        pp.report_stdout(fontsize=9)
    finally:
        pp.close()
    
    if saveroot:
        save_results(saveroot, filename, ant.name, src_ID, freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg, offbore_deg)
    return result


def save_results(root,dataset_fname,antname,target, freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg, *ig, **nored):
    """ Saves data products to CSV files named as 'root/dataset-ant-product.csv
        @param root: root folder on filesystem to save files to.
        @param dataset_fname: the filename of the raw dataset.
        @param antname, target: identifies the antenna and target that the dat awas collected from.
        @param freqs, counts2Jy, SEFD_meas, pSEFD, Tsys_meas, Trx_deduced, Tspill_deduced, pTsys, pTrx, pTspill, S_ND, T_ND, el_deg: data products as returned by 'get_SEFD_ND()'.
    """
    fnroot = "%s/%s-%s-"%(root,dataset_fname.split("/")[-1].split(".")[0],antname)
    origin = "Recorded with <%s> at %.fdegEl\nDataset %s" % (target, el_deg, dataset_fname)
    for fn,data,descr,unit in [("counts2Jy",np.c_[freqs, counts2Jy[:,0], counts2Jy[:,1]],"(Source Flux)/(P_src_on_boresight-P_src_in_null)","Jy/#"),
                               ("SEFD",np.c_[freqs, SEFD_meas[:,0], SEFD_meas[:,1]],"SEFD at calibrator background","Jy"),
                               ("S_ND",np.c_[freqs, S_ND[:,0], S_ND[:,1]],"Noise Diode equivalent Flux","Jy")]:
        np.savetxt(fnroot+fn+".csv", data, delimiter=",",
                   header="%s. %s\nfrequency [Hz]\t,H [%s]\t, V [%s]"%(descr,origin,unit,unit))

def load_results(fids, product="SEFD", root="", also_el=False):
    """ Loads the results stored when analyse() completes. Re-grids all data onto a common frequency grid.
        @param fids: list of file ID's (e.g. "{epoch seconds}_sdp_l0-s0000")
        @param product: "SEFD"|"S_ND"|"counts2Jy" (default "SEFD")
        @return: (freq_Hz, H_data, V_data) and el_deg if also_el.
    """
    f_,d_H,d_V,el = [],[],[],[]
    for i in fids:
        f,x_h,x_v = np.loadtxt("%s/%s-%s.csv"%(root,i,product), delimiter=",", unpack=True)
        d_H.append(x_h)
        d_V.append(x_v)
        f_.append(f)
        with open("%s/%s-%s.csv"%(root,i,product)) as file:
            header = file.readline()
        i = header.find("degEl")
        el.append(float(header[header.rfind(" ", 0,i):i]))
    # Re-grid all results onto a common range
    f, d_H = combine(f_, d_H)
    f, d_V = combine(f_, d_V)
    d_H = np.transpose(d_H)
    d_V = np.transpose(d_V)
    if not also_el:
        return f, d_H, d_V
    else: # Parse elevation from first line of header
        return f, d_H, d_V, el


def save_Tnd(freqs, T_ND, rx_band_SN, output_dir, info="", rfi_mask=[], debug=False):
    """ Save noise diode equivalent temperatures to a model file, in the standard MeerKAT format.
        If there are lists of frequencies & T_ND then the data series will automatically be combined.
        
        @param freqs: (lists of) frequencies [Hz]
        @param T_ND: (lists of) Noise diode equivalent temperatures [K], arranged as (freqs,pol)
        @param rx_band_SN: either a tuple (band, serialno) or a string as "band.serialno"; 'band' is a single letter e.g. "u","l".
        @param output_dir: folder where csv files will be stored
        @param info: a short bit of text to include in the header, e.g. 'As measured on m012'.
        @param rfi_mask: list of frequency [Hz] ranges to omit from plots e.g. [(924.5e6,960e6)] (default [])
    """
    T_ND = np.atleast_2d(T_ND)
    if isinstance(rx_band_SN, str):
        rx_band, rx_SN = rx_band_SN.split(".")
    
    if (np.ndim(freqs[0]) == 0):
        TndH, TndV = T_ND[:,0], T_ND[:,1]
    else: # Lists to be combined
        f, _h = combine(freqs, [x[:,0] for x in T_ND])
        freqs, _v = combine(freqs, [x[:,1] for x in T_ND], f)
        TndH, TndV = np.ma.mean(_h,axis=0), np.ma.mean(_v,axis=0)
    
    # Mask & write to file
        
    for pol,Tdiode in enumerate([TndH,TndV]):
        Tdiode = mask_where(Tdiode, freqs, rfi_mask+[(0,np.min(freqs)+1),(np.max(freqs)-1,np.inf)]) # Mask out RFI-contaminated bits & always the "DC" bin
        
        # Scape currently blunders if the file contains nan or --, so only write valid numbers
        notmasked = ~Tdiode.mask & np.isfinite(Tdiode)
        _f, _T = np.compress(notmasked, freqs), np.compress(notmasked, Tdiode)
        if (rx_band.lower() == "u"): # Automatically skip low freq garbage
            _f, _T = _f[_f>540e6], _T[_f>540e6]
        
        if debug:
            plot_data(_f/1e6, _T, style='.', xtag="Frequency [MHz]", ytag="T_ND [K]", newfig=(pol==0))
        
        # Write CSV files
        #np.savetxt('%s/rx.%s.%s.%s.csv' % (output_dir, rx_band.lower(), rx_SN, "hv"[pol]), # TODO: use this rather than below
        #           np.c_[_f,_T], delimiter=",", fmt='%.2f')
        outfile = open('%s/rx.%s.%s.%s.csv' % (output_dir, rx_band.lower(), rx_SN, "hv"[pol]), 'w')
        outfile.write('#Noise Diode table. %s\n'%info)
        outfile.write('# Frequency [Hz], Temperature [K]\n')
        if (_f[-1] < _f[0]): # unflip the flipped first nyquist ordering
            _f, _T = np.flipud(_f), np.flipud(_T)
        outfile.write(''.join(['%s, %s\n' % (entry[0], entry[1]) for entry in zip(_f,_T)]))
        outfile.close()


def load4hpbw(ds, savetofile=None, n_chunks=64, cleanchans=None, jump_zone=0, cached=False, return_all=False, debug=2):
    """ Processes a raw dataset to determine the beam crossing time instant, as well as the half power crossing duration.
        Note that the crossing duration is scaled to represent a target on the ecliptic (declination=0), which simplifies
        further interpretation.
        
        If those results were previously saved to local disk, this function may be used to load it again.
        Note: This function does not modify the active selection filter of the raw dataset.
         
        @param ds: either a raw drift scan dataset or an npz file name
        @param savetofile: npz filename to save the data to (default None)
        @param n_chunks: > 0 to average the frequency range into this many chunks to fit beams, or <=0 to fit band average only (default 64).
        @param cleanchans: used to select the clean channels to use to fit bore sight transit and beam widths on (default None).
        @param jump_zone: controls automatic excision of jumps over time, see 'fit_bm()' (default 0)
        @param cached: True to load from 'savetofile' if it exists (default False)
        @param return_all: True if ds is a raw dataset, to also return fitted (baseline,beam) as extra data (default False)
        @param debug: as for fit_bm()
        @return: (f [Hz], mu@f [seconds since ds.time_start], sigma@f [seconds], extra...). mu & sigma are masked arrays like 'fit_bm()'.
    """
    extra = []
    
    if hasattr(ds, "channel_freqs"): # A raw dataset (might still be cached)
        if cached and not return_all: # We don't cache the extras
            if not savetofile: # Default savetofile name
                savetofile = "%s_%s_load4hpbw_%d.npz" % (ds.name, ds.ant.name, n_chunks)
            try:
                return load4hpbw(ds=savetofile)
            except:
                pass
        # Not cached, so do all the fitting work...
        
        # Only use drift scan section to prevent ND jumps from influencing fits, but don't use select()!
        time_mask = (ds.sensor["Antennas/array/activity"]=="track") & (ds.sensor["Observation/label"]=="drift")
        bl,mdl,sigma,mu = driftfit.fit_bm(ds.vis, n_chunks=n_chunks, freqchans=cleanchans, timemask=time_mask, jump_zone=jump_zone, debug=debug)
        
        # To simplify further processing, the transit duration is scaled to represent a target at declination=0
        dec_tgt = ds.target.apparent_radec(np.mean(ds.timestamps[time_mask]))[1]
        sigma *= ds.dump_period * np.cos(dec_tgt) # [dumps] -> [seconds along the ecliptic]
        
        # Beam crossing time is "at the center of the tangent plane" so does not need to be scaled
        T0 = ds.timestamps[0] - float(ds.start_time)
        mu = T0 + mu*ds.dump_period # [dumps] -> [seconds since start]
        f = ds.channel_freqs
        
        if return_all: 
            extra = [bl, mdl]
        
        if savetofile:
            np.savez(savetofile, mu=mu.data, sigma=sigma.data, mask=mu.mask, f=f)
            data = np.load(savetofile, allow_pickle=True)
            mask, f = data['mask'], data['f']
            mu, sigma = np.ma.masked_array(data['mu'],mask,fill_value=np.nan), np.ma.masked_array(data['sigma'],mask,fill_value=np.nan)
            data.close()
    
    else: # Not a raw dataset so probably the results from a previous run on the raw data
        data = np.load(ds, allow_pickle=True)
        mask, f = data['mask'], data['f']
        mu, sigma = np.ma.masked_array(data['mu'],mask,fill_value=np.nan), np.ma.masked_array(data['sigma'],mask,fill_value=np.nan)
        data.close()
    
    return [f, mu, sigma] + extra


def fit_hpbw(f,mu,sigma, D, hpw_src=0, fitchans=None, debug=True):
    """ Finds the best fit polynomial that describes the half power width over frequency. Includes both the beam and source widths!
        
        @param f,mu,sigma: as returned by 'load4hpbw()', or possibly only the set for a single product.
        @param D: the aperture diameter [m] to scale the fitted coefficients to.
        @param hpw_src: the equivalent half-power width [rad] of the target (default 0)
        @param fitchans: selector to limit the frequency channels over which to fit the model (default None)
        @return: lambda f: hpw [rad]
    """
    hpw_src = 0 if (hpw_src is None) else hpw_src
    fitchans = slice(None) if (fitchans is None) else fitchans
    
    # Basic model and constants
    hpbw = lambda f, p,q: ((p*(_c_/f)**q/D)**2 + hpw_src**2)**.5 # rad
    omega_e = 2*np.pi/(24*60*60.) # rad/sec
    K = np.sqrt(8*np.log(2)) # hpbw = K*sigma
     
    if debug:
        plot_data(f/1e6, K*sigma*omega_e*180/np.pi, style=',', newfig=True, header="Fit to measured half power width",
                  xtag="Frequency [MHz]", ytag="HPBW [deg]", y_lim='pct')
    
    # The data to fit to
    N_freq, N_prod = sigma.shape
    ssigma = np.ma.array(sigma, copy=True)
    ssigma[np.isnan(ssigma) | ssigma.mask] = np.nanmean(ssigma) # Avoid warnings in 'ssigma<_s' below if there are nan's
    # Don't fit to data that is too far off the expected smooth curve
    _s = np.stack([mat.smooth(ssigma[:,p], 3+N_freq//50) for p in range(N_prod)], -1) # Filter out deviations over < N_freq/50
    ff, ssigma, _s = f[fitchans], ssigma[fitchans], _s[fitchans]
    ssigma.mask = ssigma.mask & (np.abs(ssigma-_s)>0.05*_s)
    if debug:
        plot_data(ff/1e6, K*ssigma*omega_e*180/np.pi, style='.', label="To fit", newfig=False)
    print("Fitting HPBW over %.f - %.f MHz assuming D=%.2f m"%(np.min(ff)/1e6, np.max(ff)/1e6, D))
    
    # lambda^1 model
    _p = sop.fmin_bfgs(lambda p: np.nansum((np.dstack([hpbw(ff,p[0],1)]*N_prod)-K*ssigma*omega_e)**2), [1], disp=False)
    _hpbw = hpbw(f,_p[0],1)
    print("Simultaneous fit to %d products: %g lambda^1 / %g"%(N_prod,_p[0],D))
    
    # lambda^n model
    __p = sop.fmin_bfgs(lambda p: np.nansum((np.dstack([hpbw(ff,p[0],p[1])]*N_prod)-K*ssigma*omega_e)**2), [1,1], disp=False)
    __hpbw = hpbw(f,*__p)
    print("Simultaneous fit to %d products: %g lambda^(%g) / %g"%(N_prod,__p[0],__p[1],D))

    if debug:
        plot_data(f/1e6, _hpbw*180/np.pi, label="$%.2f \lambda/%.3f$ [rad]"%(_p[0],D), newfig=False)
        plot_data(f/1e6, __hpbw*180/np.pi, label="$%.2f \lambda^{%.2f}/%.3f$ [rad]"%(__p[0],__p[1],D), newfig=False)
    return lambda f: hpbw(f,*__p)



def compare_FieldInAeTsys(freqs, SEFD_meas, SEFD_pred=None, y_lim='pct'):
    """ Plots the results in terms of Ae/Tsys.
    
        @param freqs, SEFD_meas, SEFD_pred: the leading output terms of get_SEFD_ND()
    """
    plot_data(freqs/1e6, 2*_kB_/1e-26 / SEFD_meas[:,0], label="Measured H", xtag="f [MHz]", ytag="$A_e/T_{sys}$ [m$^2$/K]")
    plot_data(freqs/1e6, 2*_kB_/1e-26 / SEFD_meas[:,1], newfig=False, label="Measured V", y_lim=y_lim)
    if (SEFD_pred is not None):
        plot_data(freqs/1e6, 2*_kB_/1e-26 / SEFD_pred[:,0], newfig=False, label="Predicted H")
        plot_data(freqs/1e6, 2*_kB_/1e-26 / SEFD_pred[:,1], newfig=False, label="Predicted V", y_lim=y_lim)


def compare_FieldToSpec(freqs, SEFD_meas, SEFD_pred, *args):
    """ Compares the measured results to the MeerKAT allocated specifications, in Ae/Tsys.
    
        @param ...: the output of get_SEFD_ND()
    """
    D = 13.5
    Nant = 64.
    
    spec_Ae_Tsys = 0*freqs
    spec_Ae_Tsys[freqs<1420e6] = (Nant * np.pi*D**2/4.) / 42 # [R.T.P095] == 220
    spec_Ae_Tsys[freqs>=1420e6] = (Nant * np.pi*D**2/4.) / 46 # [R.T.P.096] == 200
    
    compare_FieldInAeTsys(freqs, SEFD_meas, SEFD_pred)
    plot_data(freqs/1e6, spec_Ae_Tsys/Nant, newfig=False, label="\"220-200 m$^2$/K\" at Telescope PDR", y_lim='pct')
    plot_data(freqs/1e6, np.linspace(275,410,len(freqs))/Nant, newfig=False, label="\"275-410 m$^2$/K\" at Receivers CDR")
    
    compare_FieldInAeTsys(freqs, SEFD_meas*Nant, SEFD_pred*Nant)
    plot_data(freqs/1e6, spec_Ae_Tsys, newfig=False, label="220-200 m$^2$/K at Telescope PDR", y_lim='pct')
    plot_data(freqs/1e6, np.linspace(275,410,len(freqs)), newfig=False, label="275-410 m$^2$/K at Receivers CDR")


if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser(usage="%prog [options] <data file>",
                                   description="Processes a drift scan dataset, using aperture efficiency model from records"
                                               " (if not available for a band, assumes some nominal default value)."
                                               "Results are summarised in figures and processing log in PDF report.")
    parser.add_option('-a', type='int', default=None,
                      help="Antenna numerical sequence as listed in the dataset - NOT receptor ID. Alternative for --ant.")
    parser.add_option('--ant', type='string', default=None,
                      help="Antenna name as used in the dataset. Alternative for -a.")
    parser.add_option('-x', '--flux-key', type='string', default=None,
                      help="Identify a specific flux model to use from the catalogue (default %default)")
    parser.add_option('--rfi-mask', type='string', default=None,
                      help="RFI frequency mask as a list of tuples in Hz, e.g. [(925e6,965e6)].")
    parser.add_option('--fit-freq', type='string', default="[580e6,720e6]",
                      help="Range of frequencies in Hz over which to fit baselines (default %default)")
    parser.add_option('--strict', action='store_true', default=False,
                      help="Strictly avoid noise diode unless tracking the target (default %default).")
    
    (opts, args) = parser.parse_args()
    opts.hpbw = eval(opts.hpbw)
    freqmask = eval(opts.rfi_mask)
    fitfreqrange = eval(opts.fit_freq)
    ant = opts.ant if (opts.ant is not None) else opts.a
    
    result = analyse(args[0], ant, opts.target, opts.flux_key, fitfreqrange=fitfreqrange, freqmask=freqmask, strict=opts.strict, saveroot=".", makepdf=True)
    