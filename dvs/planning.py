"""
    Various tools that may be used to plan DVS observations.
    
    @author aph@sarao.ac.za
"""
import numpy as np
import pylab as plt
import katpoint
import ephem
import astropy.coordinates as acoords
from analysis import katsemodels
from . import cattools
try:
    import healpy as hp
except:
    print("WARNING: Failed to load healpy. Continuing with limitations.")
    hp = None


def to_ephem_date(datetime):
    if isinstance(datetime, float):
        datetime = katpoint.Timestamp(datetime)
    if isinstance(datetime, katpoint.Timestamp):
        datetime = datetime.to_ephem_date()
    return datetime


def radiosky(date, f_MHz, flux_limit_Jy=None, el_limit_deg=1,
             catfn='catalogues/sources_all.csv', listonly=None,
             llh0=('-30.713','21.444',1050), llh1=None, enu1=None, tabulate=True, figsize=(20,12), fontsize=8, **kwargs):
    """ Make a sky plot for the given date, seen from observer's location. Also plot sources visible from catalogue.
        
        @param date: (y,m,d,H,M,S) in UTC, or UTC seconds since epoch.
        @param listonly: if not None, a list of target names to select from the catalogue (default None).
        @param tabulate: True to print a table of all visible sources (default True)
        @param llh0: lat, lon, height above ellipsoid for nominal observer (default to m000).
        @param llh1: lat, lon, heights above ellipsoid for baseline endpoint, or None (default None)
        @param enu1: fallback ALTERNATIVE to llh1, as East,North,up [m]
        @return: the katpoint.Catalogue as set up & filtered for display.
    """
    try:
        llh_ = list(kwargs['observer'].split(','))
        llh_[2] = float(llh_[2])
        llh0 = llh_
    except:
        pass
    
    date = to_ephem_date(date)
    refant = katpoint.Antenna("A0, %s, %s, %.1f, 13.5" % llh0)
    observer = refant.observer
    
    ov = katsemodels.FastGSMObserver() # Employs pygsm.GlobalSkyModel with defaults, i.e. 2008 locked to the 408 MHz map
    ov.lat, ov.lon, ov.elev = observer.lat, observer.lon, observer.elev
    ov.date = date # PyEphem assumes format as stated above
    ov.generate(f_MHz)
    fig = plt.figure(figsize=figsize)
    plt.subplot(1,2,1); plt.title("%g MHz sky excl. CMB from OBSERVER on %s UTC / %s LST" % (f_MHz,ov.date, ephem.hours(ov.radec_of(0, np.pi / 2)[0])))
    hp.orthview(np.log10(ov.observed_sky), half_sky=True, unit="log10(K)", hold=True) # Default flip='astro'i.e. east is left
    ax = plt.gca(); ax.annotate("E", [-1,0], horizontalalignment='right'); ax.annotate("W", [1,0])
    plt.subplot(1,2,2); hp.cartview(np.log10(ov.observed_gsm), coord="G", unit="log10(K)", hold=True)

    # What sources are up?
    cat = katpoint.Catalogue(open(catfn), add_specials=True, antenna=refant)
    if listonly:
        listonly = set(listonly)
        cat = katpoint.Catalogue([target for target in cat.targets if not (listonly.isdisjoint(set([target.name]).union(set(target.aliases))))],
                         add_specials=False, antenna=cat.antenna, flux_freq_MHz=cat.flux_freq_MHz)
    if (flux_limit_Jy or el_limit_deg):
        cat = cat.filter(flux_freq_MHz=f_MHz,flux_limit_Jy=flux_limit_Jy,antenna=refant,timestamp=ov.date,el_limit_deg=el_limit_deg)
    if tabulate:
        ant1 = None
        if (llh1 is not None):
            ant1 = katpoint.Antenna("A1, %s,%s,%.1f, 13.5" % llh1)
        elif (enu1 is not None): # llh is ignored for delays if enu is given
            ant1 = katpoint.Antenna(("A1, %s,%s,%.1f, 13.5, " % llh0) + ("%.3f %.3f %.3f 0 0 0" % enu1))
        cat.visibility_list(timestamp=ov.date, antenna=refant, antenna2=ant1, flux_freq_MHz=f_MHz)
    
    ax = fig.axes[0] # Plot all filtered sources on first axes == orthographic projection
    for a,e,l in [list(tgt.azel(ov.date))+[tgt.name] for tgt in cat]: # rad,rad,string
        gl, gb = katpoint.sphere_to_plane["SIN"](0,np.pi/2.,a-np.pi,e)
        ax.plot(gl,gb, 'r.')
        ax.annotate(l, [gl,gb], fontsize=fontsize)
    ax = fig.axes[2] # Plot all filtered sources on last axes == galactic coordinates
    for r,d,l in [list(tgt.radec(ov.date))+[tgt.name] for tgt in cat]: # rad,rad,string
        c = acoords.SkyCoord(ra=r, dec=d, unit='rad', frame='icrs')
        gl, gb = -katpoint.wrap_angle(c.galactic.l.degree, 360), c.galactic.b.degree
        ax.plot(gl,gb, 'r.')
        ax.annotate(l, [gl,gb], fontsize=fontsize)

    return cat


def _baseline_endpt_(ant, baseline_pt):
    """
        @param ant: the observing katpoint.Antenna
        @param baseline_pt: the reference end point for the baseline, either katpoint.Antenna or (dEast,dNorth,dUp)[m] relative to 'ant'
        @return: a reference point object to use e.g. for geometric delay calculations. 
    """
    if isinstance(baseline_pt,tuple):
        class Ant(object):
            def __init__(self, enu, ref_position_wgs84):
                self.name = str(enu)
                self.position_enu = enu
                self.ref_position_wgs84 = ref_position_wgs84
        pos = ant.position_enu
        baseline_pt = Ant((pos[0]+baseline_pt[0], pos[1]+baseline_pt[1], pos[2]+baseline_pt[2]), ant.ref_position_wgs84)
    return baseline_pt


def describe_target(target, date, end_date=None, horizon_deg=0, baseline_pt=(1000,0,0), freq=1e9, show_LST=False,
             catfn='catalogues/sources_all.csv',ant="m000, -30.713, 21.444, 1050.0", **figargs):
    """ Notes rise & set times and plots the elevation trajectory if end_date is given.
        Also prints the geometric delay rate for the baseline as specified. 
        
        @param target: a katpoint.Target object as observed by the associated Target.antenna,
                       or a name (or multiple 'a|b|..' or '*' for all) to load from the catalogue.
        @param date: (y,m,d,H,M,S) in UTC, or UTC seconds since epoch.
        @param end_date: if given then will evaluate the target over the period from 'date' up to 'end_date' (default None)
        @param horizon_deg: angle for horizon limit to apply [deg]
        @param baseline_pt: the reference point for a baseline to calculate fringe rate,
                            either katpoint.Antenna or (dEast,dNorth,dUp)[m] (default (1000,0,0)).
        @param freq: frequency for calculating fringe rate (default 1e9) [Hz]
        @param catfn, ant: defaults to use in case target is simply an identifier string.
        @return: target (a list, if multiple matches)- a katpoint.Target
    """
    date = to_ephem_date(date)
    if (end_date is not None):
        end_date = to_ephem_date(end_date)
    ant = figargs.pop("catant", ant) # Backwards compatibility
    
    if isinstance(target,str):
        cat = katpoint.Catalogue(open(catfn), add_specials=True, antenna=katpoint.Antenna(ant))
        if (target != "*"):
            matches = target.split("|")
            targets = [cat[t] for t in matches if (cat[t] is not None)]
            missing = [t for t in matches if (cat[t] is None)]
            if (len(missing) > 0):
                print(f"WARNING: {catfn} does not contain entries for the following targets!")
                print(", ".join(missing))
        else:
            targets = cat.targets
    else:
        targets = [target]
    
    ant = targets[0].antenna
    # Make a copy of the observer to avoid modifying it unexpectedly
    observer = ant.observer
    observer.horizon = "%g" % horizon_deg # strings for degrees
    def LST(date): # Convert UTC to the sidereal time of the observer
        d = observer.date; observer.date=date; st = observer.sidereal_time(); observer.date = d
        return st
    TC = "LST" if show_LST else "UTC"
    
    # Abuse the observer to ensure all dates are parsed if they can be
    observer.date = date; date = observer.date
    if end_date is not None:
        observer.date = end_date; end_date = observer.date
    timestamps = np.arange(katpoint.Timestamp(date).secs, katpoint.Timestamp(end_date).secs, 60)
    
    # Plot sky trajectories
    if (end_date is not None):
        plt.figure(**figargs)
        plt.title("Targets as observed by [%s]" % ant.name)
        dtime = (timestamps-timestamps[0])/3600.
        for target in targets:
            names = target.name + ("|"+"|".join(target.aliases) if target.aliases else "")
            _, el = target.azel(timestamps) # rad
            plt.plot(dtime, el*180/np.pi, '-', label=names)
        plt.hlines(horizon_deg, dtime[0], dtime[-1], 'k'); plt.ylim(0, 90)
        date0 = LST(date) if show_LST else date
        plt.xlabel(f"Time since {str(date0)} {TC} [hours]"); plt.ylabel("El [deg]")
        plt.grid(True); plt.legend(fontsize='small')
    
    
    baseline_pt = _baseline_endpt_(ant, baseline_pt) if (baseline_pt is not None) else None
    
    print(f"{'Name':>14} | {'Rise':>14} {TC} | {'Set':>14} {TC} | Max Rate [deg/sec@{freq/1e9:.0f}GHz] | Flux [Jy@{freq/1e9:.0f}GHz]")
    print("-"*80)
    for target in targets:
        _name = target.name
        _flux = target.flux_density(freq/1e6) if (target.flux_model is not None) else np.nan
        try:
            _rise = observer.previous_rising(target.body) # observer date is still end_date, if given
            observer.date = date
            _set = observer.next_setting(target.body)
            if show_LST:
                _rise, _set = LST(_rise),  LST(_set)
        except (ValueError, AttributeError): # ValueError->ephem.AlwaysUpError, AttributeError->StationaryBody has no attribute 'radius'
            _rise, _set = np.nan, np.nan
        _phrate = np.nan
        if baseline_pt is not None:
            delay_rate = target.geometric_delay(baseline_pt, timestamps, ant)[1]
            imax = np.argmax(np.abs(delay_rate))
            _phrate = delay_rate[imax] * (360*freq) # deg/sec
        print(f"{_name:>14} | {str(_rise):>18} | {str(_set):>18} | {_phrate:>23.1f} | {_flux:.1f}")

    targets = targets[0] if (len(targets) == 1) else targets
    return targets


def plot_Tant(RA,DEC, freq_MHz, extent,D_g=13.965,ellip=0,projection=None, label="", hold=True, draw=True, **figargs):
    """
        Plots the sky temperature of the default Global Sky Model, excluding CMB, centred at the specified coordinate(s).
        Also draws contours at 50% & 90% of the peak value within the max extent diameter.
        An example that may be used to confirm orientation:
            plot_Tant((5,35,17.26), (-5,23,28.8), 1400, 30)
        
        @param RA, DEC: tuples of (h,m,s) & (d,m,s), or decimal degrees
        @param extent: the extent of the square area to plot [deg]
        @param D_g: the equivalent geometric diameter of the nominally circular aperture of the antenna (default 13.965) [m]
        @param projection: if extent is intended in spherical coordinates then the mapping to use to project to cartesian RA,DEC e.g. "SIN" for equal area Orthographic (default None)
        @return: the 2D table of antenna temperatures (0:RA,1:DEC) [K]
    """
    D2R = np.pi/180.
    extent *= D2R # rad
    gsm, res = katsemodels.get_gsm(freq_MHz) # 2d array, rad
    
    if isinstance(RA,tuple): # Convert from (hrs,m,s)(deg,m,s) to float degrees
        RA = (RA[0]+RA[1]/60.+RA[2]/3600.)/24.*360
        DEC = np.sign(DEC[0])*(abs(DEC[0])+DEC[1]/60.+DEC[2]/3600.)
    
    dr,dd = np.linspace(-extent/2.,extent/2.,int(extent/res*2+1)), np.linspace(-extent/2.,extent/2.,int(extent/res*2+1))
    dr,dd = np.meshgrid(dd/D2R, dr/D2R) # RA,DEC offsets in map plane, deg
    
    # Calculate coordinates in Cartesian projection which is required for mapvalue()
    cRA, cDEC = RA+dr, DEC+dd # deg
    if (projection is not None):
        cRA, cDEC = katpoint.plane_to_sphere[projection](RA*D2R,DEC*D2R,dr*D2R,dd*D2R)
        cRA, cDEC = cRA/D2R, cDEC/D2R # deg
    res = max(res, 1.27*(300./freq_MHz)/D_g * 1.29*2) # Approx width between nulls for ^2 pattern (nulls @ HPBW*{1.29,2.14,2.99,...})
    T = katsemodels.mapvalue(gsm,cRA,cDEC,radius=res/2.,beam="^2,null1",ellip=ellip)
    # Orientation of T follows cRA increasing from left to right; cDEC increasing from top to bottom
    
    if draw:
        if not hold:
            plt.figure(**figargs)
        T0 = np.max(T[(dr**2+dd**2)<dr.max()*dd.max()]) # Max value in a centred disk
        plt.imshow(np.flip(T,axis=1), origin='lower', extent=[dr.max(),dr.min(),dd.min(),dd.max()], # min(DEC) in lower corner & min(RA) at the right
                   vmax=T0, interpolation='hanning')
        plt.colorbar()
        plt.contour(dr[0,:],dd[:,0], T, T0*np.asarray([0.5,0.9])) # RA order above is maintained
        projection = "" if (projection is None) else projection+" "
        plt.xlabel(projection+"RA - %.2f$^\circ$"%RA), plt.ylabel(projection+"DEC - %.2f$^\circ$"%DEC)
        plt.title(label + " @ %dMHz, %.1f' resolution [Kelvin]"%(freq_MHz,res/D2R*60.))
    
    return T


def plot_Tant_drift(parked_target,timestamps,freq_MHz=1400,normalize=False,pointing_errors=[0,1],
                        D_g=13.965,Trx=15,ant='m000, -30.713, 21.444, 1050.0',debug=True):
    """ Generates an estimate of the change in antenna temperature over time, if the antenna remains "parked".
    
        @param parked_target: a katpoint.Target object defining where the antenna is parked.
        @param timestamps: timestamps [seconds since epoch]
        @param pointing_errors: pointing errors in "map resolution increments", to consider
        @param Trx: Ball-park for Trx+Tspill+Tatm to complete system noise budget (default 15) [K]
        @param ant: None to park at astrometric coordinate (earth centre), otherwise it's a topocentric/apparent coordinate.
    """
    hrs = (timestamps-timestamps[0])/(60*60.) # sec -> relative hrs
    if ant:
        RA0,DEC = parked_target.apparent_radec(timestamps[0], antenna=katpoint.Antenna(ant)) # rad
    else:
        RA0,DEC = parked_target.astrometric_radec(timestamps[0], antenna=None) # rad with observer at the centre of the earth, no nutation
    HA = [(RA0*12/np.pi+h)%24-12 for h in hrs] # HA (-12..+12) per timestamp
    HA, DEC = np.asarray(HA)*180/12., np.asarray([DEC]*len(timestamps))*180/np.pi # -> deg as required below
    
    # PE = Pointing error in "resolution increments"
    boresight_val = lambda map2d, PE=0: map2d[int(map2d.shape[0]+PE)//2, map2d.shape[1]//2]
    
    extent_deg = 6 * 1.27*(300./freq_MHz)/D_g * 180/np.pi # radius = 3*HPBW, assuming parabolic illumination on circular aperture

    # Trx + GSM + CMB at each HA
    T_0 = [] # No beam integral, even indices have PE=0
    T_b = [] # Beam integral, even indices have PE=0
    T_e = [] # Elliptical beam integral, even indices have PE=0
    ND = len(timestamps)//4 if debug else len(timestamps)+1 # Used to generate 4 debug plots, below
    for i,(RA,DEC) in enumerate(zip(HA,DEC)):
        # To debug, make plots at only at every 6th timestamp
        _T = plot_Tant(RA=RA,DEC=DEC, freq_MHz=freq_MHz,D_g=np.inf, extent=extent_deg,projection="SIN",draw=(i+1)%ND==0,hold=False)
        for PE in pointing_errors:
            T_0.append(Trx+boresight_val(_T, PE))
        _T = plot_Tant(RA=RA,DEC=DEC, freq_MHz=freq_MHz,D_g=D_g, extent=extent_deg,projection="SIN",draw=False)
        for PE in pointing_errors:
            T_b.append(Trx+boresight_val(_T, PE))
        _T = plot_Tant(RA=RA,DEC=DEC, freq_MHz=freq_MHz,D_g=D_g,ellip=0.1, extent=extent_deg,projection="SIN",draw=False)
        for PE in pointing_errors:
            T_e.append(Trx+boresight_val(_T, PE))

    plt.figure(figsize=(16,8))
    norm = lambda X: X/np.mean(X) if normalize else X
    plt.plot(hrs, norm(T_0[::2]), '-', label="No beam")
    plt.plot(hrs, norm(T_0[1::2]), '-', label="... with Pointing error")
    plt.plot(hrs, norm(T_b[::2]), '^-', label="Parabolic beam")
    plt.plot(hrs, norm(T_b[1::2]), '^-', label="... with Pointing error")
    plt.plot(hrs, norm(T_e[::2]), '+-', label="Parabolic squashed beam")
    plt.plot(hrs, norm(T_e[1::2]), '+-', label="... with Pointing error")
    plt.xlabel("time [hrs]"); plt.legend()
    plt.ylabel("Tsys/mean(Tsys) on sky [K/K]" if normalize else "Tsys on sky [K]")


def sim_pointingmeasurements(catfn, Tstart, Hr, S, el_limit_deg=20, separation_deg=1, sunmoon_separation_deg=10,
                             ant='m000, -30.713, 21.444, 1050.0', verbose=False, **fitkwargs):
    """ Simulates pointing measurement sessions and makes plots to allow you to see
        the resulting sky coverage using the specified catalogue.
        @param catfn: the catalogue file.
        @param Hr: the duration of a session in hours
        @param S: the number of consecutive sessions to simulate
        @param separation_deg: eliminate targets closer together than this (default 1) [deg]
        @param sunmoon_separation_deg: omit targets that are closer than this distance from Sun & Moon (default 10) [deg]
        @param fitkwargs: passed to `cattools.sim_pointingfit()`
    """
    cat = katpoint.Catalogue(open(catfn), antenna=katpoint.Antenna(ant))
    print("Using ", catfn)
    axes = np.ravel(plt.subplots(max(1,int(S/2+0.5)),2, subplot_kw=dict(projection='polar'), figsize=(12,12))[1])
    for n in range(S): # Repeat each "plan" so many times
        T0 = Tstart + (Hr*n)*60*60 # Session at intervals of Hr hours (hope the plan fits in that time)
        cat_n = cattools.filter_separation(cat, T0, separation_deg=separation_deg, sunmoon_separation_deg=sunmoon_separation_deg)
        t_observe = 5*(30+2) # 5-point scans
        T = cattools.plan_targets(cat_n, T0, t_observe, 2, el_limit_deg=el_limit_deg, debug=verbose)[1]
        M = int(Hr*60*60/T)+1 # Number of times the plan fits in available hours
        resid = cattools.sim_pointingfit(catfn, el_limit_deg, Hr*60, t_observe/60+1, Tstart=T0, **fitkwargs) # +1min for slews
        cattools.plot_skycat(cat_n, T0+np.arange(M)*T, t_observe, ax=axes[n])
        axes[n].set_title("%s T+%d hrs (RMS<%.f'')"%(catfn, Hr*n, np.max(resid)))


def sim_observations(schedule, catfn, Tstart, interval=60, el_limit_deg=15, ant='m000, -30.713, 21.444, 1050.0', **figkwargs):
    """ Simulate a sequence of observations, to evalaute target elevation angles against the specified limit.
        
        @param schedule: a list of (target [name or description], duration [sec]); targets not recognised will simply pass the time.
        @param catfn: the catalogue file to use to resolve target names.
        @param Tstart: start time [UTC Unix seconds]
        @param interval: time to elapse before starting a schedule entry [sec] (default 60).
    """
    cat = katpoint.Catalogue(open(catfn), add_specials=True, antenna=katpoint.Antenna(ant))
    
    # Plot sky trajectories
    plt.figure(**figkwargs)
    plt.title("Targets as observed by [%s]" % cat.antenna.name)
    all_tgts = [] # Unique list of names - helps keep legend manageable
    start_time = Tstart
    for tgt,duration in schedule:
        target = cat[tgt]
        if (target is not None):
            start_time += interval
            timestamps = start_time + np.linspace(0,duration,10)
            _, el = target.azel(timestamps) # rad
            el = el*180/np.pi
            kwargs = dict()
            if (target.name not in all_tgts):
                all_tgts.append(target.name)
                kwargs['label'] = target.name + ("|"+"|".join(target.aliases) if target.aliases else "")
            plt.plot(np.asarray(timestamps,dtype='datetime64[s]'), el, 'C%d-'%all_tgts.index(target.name), **kwargs)
            # Print out the calendar line for this observation
            if (np.min(el) < el_limit_deg):
                tgt = "CLIPPED-" + tgt + "-CLIPPED"
        print(np.datetime_as_string(np.datetime64(int(start_time),'s'), unit='m', timezone='UTC'), f"{tgt}, {int(duration/60+0.5)}min")
        start_time += duration
    plt.hlines(el_limit_deg, np.datetime64(int(Tstart),'s'), np.datetime64(int(start_time),'s'), 'k'); plt.ylim(0, 90)
    plt.xlabel(f"Time [UTC]"); plt.ylabel("El [deg]")
    plt.grid(True); plt.legend(fontsize='small')
