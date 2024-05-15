"""
    Various tools that may be used to plan DVS observations.
    
    @author aph@sarao.ac.za
"""
import numpy as np
import pylab as plt
import time
import katpoint
import ephem
import astropy.coordinates as acoords
import katsemodels
hp = katsemodels.hp # healpy, if it could be imported by katsemodels


def radiosky(date, f_MHz, flux_limit_Jy=None, el_limit_deg=1,
             catfn='/var/kat/katconfig/user/catalogues/sources_all.csv', listonly=None,
             observer="m000, -30.713, 21.444, 1050.0", tabulate=True, figsize=(20,12), fontsize=8, **kwargs):
    """ Make a sky plot for the given date, seen from observer's location. Also plot sources visible from catalogue.
        
        @param date: (y,m,d,H,M,S) in UTC, or UTC seconds since epoch.
        @param listonly: if not None, a list of target names to select from the catalogue (default None).
        @param tabulate: True to print a table of all visible sources (default True)
        @return: the katpoint.Catalogue as set up & filtered for display.
    """
    catfn = kwargs.get('cataloguefn', catfn) # Support deprecated argument names
    
    date = katpoint.Timestamp(date).to_ephem_date() if isinstance(date,float) else date
    refant = katpoint.Antenna(observer)
    observer = refant.observer
    
    ov = katsemodels.FastGSMObserver() # Employs pygsm.GlobalSkyModel with defaults, i.e. 2008 locked to the 408 MHz map
    ov.lat, ov.lon, ov.elev = observer.lat, observer.lon, observer.elev
    ov.date = date # PyEphem assumes format as stated above
    ov.generate(f_MHz)
    fig = plt.figure(figsize=figsize)
    plt.subplot(1,2,1); hp.orthview(np.log10(ov.observed_sky), half_sky=True, unit="log10(K)", hold=True)
    plt.title("%g MHz sky excl. CMB from OBSERVER on %s UTC / %s LST" % (f_MHz,ov.date, ephem.hours(ov.radec_of(0, np.pi / 2)[0])))
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
        cat.visibility_list(timestamp=ov.date, antenna=refant, antenna2=None, flux_freq_MHz=f_MHz)
    
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


def describe_target(target, date, horizon_deg=0, end_date=None, baseline_pt=None, freq=1e9,
             catfn='/var/kat/katconfig/user/catalogues/sources_all.csv',catant="m000, -30.713, 21.444, 1050.0", **figargs):
    """ Notes rise & set times and plots the elevation trajectory if end_date is given.
        Also prints flux density & fringe rate if additional parameters are given. 
        
        @param target: a katpoint.Target object as observed by the associated Target.antenna, or a name to retrieve it from the catalogue.
        @param date: (y,m,d,H,M,S) in UTC, or UTC seconds since epoch.
        @param horizon_deg: angle for horizon limit to apply [deg]
        @param end_date: if given then will evaluate the target over the period from 'date' up to 'end_date' (default None)
        @param baseline_pt: the reference end point for the baseline to calculate fringe rate, either katpoint.Antenna or (dEast,dNorth,dUp)[m] (default None)
        @param freq: frequency for calculating flux and fringe rate (default 1e9) [Hz]
        @param catfn, catant: defaults to use in case target is simply an alias string.
        @return: target
    """
    if isinstance(target,str):
        target = katpoint.Catalogue(open(catfn), add_specials=True, antenna=katpoint.Antenna(catant))[target]
    
    ant = target.antenna
    # Make a copy of the observer to avoid modifying it unexpectedly
    observer = ephem.Observer()
    observer.lon, observer.lat, observer.elev = ant.observer.lon, ant.observer.lat, ant.observer.elev 
    date = katpoint.Timestamp(date).to_ephem_date() if isinstance(date,float) else date
    observer.date = date
    date = observer.date # ephem.Date
    observer.horizon = "%g" % horizon_deg # strings for degrees
    
    print("Target [%s] as observed by [%s] on %s, at %.f MHz" % (str(target)[:20],ant.name,date,freq/1e6))
    print("-"*80)
    
    if baseline_pt is not None:
        baseline_pt = _baseline_endpt_(ant, baseline_pt)
        delay, delay_rate = target.geometric_delay(baseline_pt, katpoint.Timestamp(date), ant)
        phase_rate = delay_rate * (360*freq) # deg/sec
        fringe_period = 1. / (delay_rate * freq)
        print('Delay rate %g [sec/sec].' % delay_rate)
        print('  At observed frequency: delay rate %.2f [deg/sec], fringe period %.2f [sec]' % (phase_rate, fringe_period))
    
    if (end_date is not None):
        end_date = time.mktime(tuple(list(end_date)+[0,0,0]))-time.timezone if isinstance(end_date,tuple) else end_date
        timestamps = np.arange(katpoint.Timestamp(date).secs,katpoint.Timestamp(end_date).secs, 60)
        az, el = target.azel(timestamps) # rad
        plt.figure(**figargs)
        plt.title("Target [%s] as observed by [%s]"%(str(target)[:20],ant.name))
        plt.plot((timestamps-timestamps[0])/60., el*180/np.pi, 'o')
        plt.xlabel("time since %s [minutes]"%date)
    
    try:
        _rise, _set = observer.previous_rising(target.body), observer.next_setting(target.body)
        def ST(date):
            observer.date=date
            return observer.sidereal_time()
        print("Rises above %g degEl at %s UTC (sidereal %s)" % (horizon_deg, _rise, ST(_rise)))
        print("Sets below %g degEl at %s UTC (sidereal %s)" % (horizon_deg, _set, ST(_set)))
    except ephem.AlwaysUpError:
        print("This target is always above %g degEl." % (horizon_deg))
    
    flux = target.flux_density(freq/1e6)
    if np.isfinite(flux):
        print("Flux %.2f Jy"%flux)
    
    return target


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
