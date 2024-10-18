'''
    Analyse "RF Tracking Stability" measured by using circular_pointing.py
    
    @author: aph@sarao.ac.za
'''
import numpy as np
import scipy.optimize as sop
import pylab as plt
import katpoint
import katdal
import typing


def fit_gaussianoffset(x, y, height, powerbeam=True, constrained=True, constrain_ampl=None, debug=None):
    """ Fit a Gaussian to the magnitude measured along a trajectory (x,y).
        NB: Circular scan provides very little information to fit all of the Gaussian's free variables reliably, so use constrain_ampl.
        
        @param x, y: position coordinates around the circular scan trajectory
        @param height: height above the baseline, measured along the trajectory
        @param powerbeam: True if scanned in total power, False if scanned in complex amplitude
        @param constrain_ampl: > 0 to constrain the fitted peak to within 10% of this
        @return: (xoffset, yoffset, valid) - positions on tangent plane centred on target in same units as `x` & `y` """
    if not powerbeam: # Scan referenced to a tracking antenna scans the voltage beam, convert it to power
        height = np.abs(height)**2
    h_sigma = np.mean([np.std(height[i*10:(i+1)*10]) for i in range(len(height)//10)]) # Estimate of radiometer noise
    
    # Starting estimates
    # These measurements (power & voltage) are typically done by scanning around half _power_ contour
    scanext = 2*np.median(x**2 + y**2)**.5 
    ampl0 = constrain_ampl if (constrain_ampl) else (np.max(height) - np.min(height))
    p0 = [ampl0,0,0,scanext,scanext, np.min(height)]
    
    model = lambda ampl,xoffset,yoffset,wx,wy, h0=0: h0 + ampl*np.exp(-np.log(2)*((x+xoffset)**2/wx**2+(y+yoffset)**2/wy**2))
    
    # Fit model to data
    if not constrained: # Unconstrained fit is VERY BRITTLE, results also very sensitive to method (BFGS seems best, but Powell, Nelder-Mead, LM ...)
        p = sop.minimize(lambda p: np.nansum((height-model(*p))**2), p0, method='BFGS', options=dict(disp=False))
    else: # Constrained fits should be robust, and not very sensitive to bounds
        bounds = [(h_sigma,np.inf)] + ( [(-1*scanext,1*scanext)]*2 ) + ( [(0.5*scanext,2*scanext)]*2 ) + [(0,np.min(height))]
        if (constrain_ampl): # Explicit constraint for e.g. circular scan
            bounds[0] = (0.9*constrain_ampl, 1.1*constrain_ampl)
        p = sop.least_squares(lambda p: (height-model(*p))**2, p0, bounds=np.transpose(bounds), verbose=0)
    # It seems that the fit fails to converge if we fit for h0 too!??
    ampl, xoffset, yoffset, fwhmx, fwhmy, h0 = p.x
    model_height = model(*p.x)
    
    # 'p.success' status isn't quite reliable (e.g. multiple minima)
    # Also check deduced signal-to-noise, and residuals must be "in the noise"
    resid = np.std(height - model_height)
    valid = p.success and (ampl/h_sigma > 6) and (resid < 2*h_sigma)
    
    if debug is not None: # 'debug' is expected to be a plot axis
        if not powerbeam:
            model_height = model_height**.5
        debug.plot(model_height, 'k-', alpha=0.5, label="%.1f + %.1f exp(Q(%.2f, %.2f))"%(h0, ampl, fwhmx/scanext, fwhmy/scanext))
        debug.legend()
    return xoffset, yoffset, valid


def _generate_test_data_(kind, powerbeam=True, hpbw=0.01, ampl=5, SEFD=200, Nsamples_per_dump=1e6*1, scanrad='hpbw', ox=0, oy=0):
    """ @param kind: "circle"|"cartioid"|"epicycles"
        @param ampl, SEFD: powers per polarisation channel [Jy]
        @param Nsamples_per_dump: BW*tau
        @param scanrad: radius for the scan pattern, either "hpbw" or a number in same units as `hpbw`.
        @return: (timestamps, target_x, target_y, magnitudes) """
    scanrad = hpbw if (scanrad == 'hpbw') else scanrad
    
    timestamps = np.arange(133) # Should just be > 20?
    th = np.linspace(0, 2*np.pi, len(timestamps)) # One full cycle (should exclude the last one which overlaps)
    
    if (kind in ["circle","cardioid"]): # "circle" is a trivial case of "cardioid"
        a, b = (1, 0) if (kind == "circle") else (0.4, 1.5)
        a, b = a*scanrad, b*scanrad
        c = b if (b < a) else (b+a**2/4/b+a) # The cardioid is offset in the x direction
        radius = a + b*np.cos(th)
        target_x = radius*np.cos(th) - c/2
        target_y = radius*np.sin(th)
    elif (kind == "epicycles"):
        r2 = scanrad/2 # Sets the ratio of the circles to go between scanrad and center
        n = 4 # 2 looks like a cardioid, 3 is too "sparse"
        target_x = r2*np.cos(th) + r2*np.cos(th*n)
        target_y = r2*np.sin(th) + r2*np.sin(th*n)
    actual_x = target_x + ox
    actual_y = target_y + oy
    
    if powerbeam: # TOTAL power integrated over frequency
        hv = ampl*np.exp(-np.log(2)*(actual_x**2+actual_y**2)/hpbw**2) + SEFD*(1 + 1/Nsamples_per_dump**.5*np.random.randn(len(target_x)))
    else: # Cross-corr voltage amplitude integrated over frequency
        hv = (ampl*np.exp(-np.log(2)*(actual_x**2+actual_y**2)/hpbw**2))**.5 + SEFD*(0 + 1/Nsamples_per_dump**.5*np.random.randn(len(target_x)))
    
    return (timestamps, target_x, target_y, hv)


def _test_fit_gaussianoffset_():
    np.random.seed(1)
    
    hpbw = 0.18 # 11arcmin in degrees
    ampl = 10/2 # Jy per pol
    SEFD = 400/2 # Jy per pol
    Nsamples_per_dump = 1e6*1 # 1MHz * 1sec
    for ox,oy in [(0,0),(hpbw/3,0),(0,hpbw/3)]: # Different pointing offsets 
        for powerbeam in [True,False]: # Power & voltage beams
            timestamps,target_x,target_y,m = _generate_test_data_("epicycles", powerbeam=powerbeam, hpbw=hpbw, ampl=ampl, SEFD=SEFD,
                                                                  Nsamples_per_dump=Nsamples_per_dump, scanrad='hpbw', ox=ox, oy=oy)
            axs = plt.subplots(1, 2)[1]
            axs[0].plot(timestamps, target_x, '.', timestamps, target_y)
            axs[1].plot(m, '.')
            xoff, yoff, valid = fit_gaussianoffset(target_x, target_y, m, powerbeam=powerbeam, constrain_ampl=None, debug=axs[1])
            print(powerbeam, "x: %g -> %g"%(ox, xoff), "y: %g -> %g"%(oy, yoff), "valid: %s"%valid)


def flag_circscancycle(target_x, target_y, discard='1sigma', debug=False):
    """ @param target_x, target_x: should trace out more than 60% of a circular path!
        @param discard: strategy for identifying outliers, either '1sigma' or a percentile (both wrt. deviations from fitted orbit)
        @return: mask - boolean mask shaped like target_x|y """
    data = np.c_[target_x, target_y]
    # We'll down-weight 10% at start
    weights = np.ones((len(target_x),2), float)
    weights[:len(target_x)//10] = 0.1
    
    ### Fitting amplitude, freq & phase of a fragment of a partial sinusoid is non-trivial!
    # Robust amplitude if data traces out > 60% of a circular path - don't need to fit this
    r = np.median(np.sum(data**2, axis=1)**.5)
    # x & y don't cover 100% so the period & phase need to be fitted
    n = np.linspace(0, 2*np.pi, len(target_x)+1)[:-1]
    model = lambda f, theta: r*np.c_[np.cos(f*n+theta), np.sin(f*n+theta)]
    f, theta = sop.fmin(lambda p: np.nansum(weights*(data-model(*p))**2), [0,0], full_output=False, disp=False)
    
    model = model(f, theta)
    dev = data - model
    if (discard == '1sigma'):
        mask = np.all(np.abs(dev) < 1*np.std(dev,axis=0), axis=1) # "AND" over x,y
    else:
        mask = np.all(np.abs(dev) < np.percentile(np.abs(dev),100-discard,axis=0), axis=1) # "AND" over x,y
    
    if debug:
        print("flag_circscancycle: f=%g, theta=%g, resid(std)=%g"%(f, theta, np.std(dev)))
        x,y = model.transpose()
        plt.figure(); plt.subplot(2,1,1); plt.plot(target_x,target_y,'.', x,y,'k,'); plt.plot(target_x[mask], target_y[mask], '.')
        plt.subplot(2,1,2); plt.plot(data[...,0], '.'); plt.plot(data[...,1], '.'); plt.plot(x, ','); plt.plot(y, ',')
    return mask


def reduce_circular_pointing(ds, ant, chanmapping, track_ant=None, strict=True, discard=1, verbose=True, debug=False):
    """ Generates pointing offsets for a dataset created with circular_pointing.py
        @param ds: the dataset (selection is reset!)
        @param ant: the identifier of the scanning antenna in the dataset
        @param chanmapping: channel indices to use or a function like `lambda target_name,fGHz: channel_indices`, or None to use all channels
        @param track_ant: the identifier of the tracking antenna in the dataset, if not single dish mode (default None)
        @param strict: True to set invalid fits to nan (default True)
        @param discard: strategy for identifying outliers, either '1sigma' or a percentile (both wrt. deviations from fitted orbit)
        @return: [(timestamp [sec], target ID, dAz, dEl [deg]), ...(for each cycle)]
    """
    if (not callable(chanmapping)) and (chanmapping is not None):
        _chans_ = chanmapping
        chanmapping = lambda *a: _chans_
    
    R2D = 180/np.pi
    offsets = [] # (timestamp, target, dAz, dEl)
    
    ds.select(scans="track", compscans="~slew")
    ds.select(reset="", compscans="~track") # Also omit the bore sight tracks, which are not in some measurements and dont seem to be necessary
    if (track_ant):
        ds.select(corrprods="cross", pol=["HH","VV"], ants=[ant,track_ant])
    else:
        ds.select(pol=["HH","VV"], ants=[ant])
    fGHz = np.round(ds.spectral_windows[0].centre_freq/1e9, 1)
    
    ant_ix = [a.name for a in ds.ants].index(ant)
    ant = ds.ants[ant_ix]
    for scan in ds.scans():
        # Fit offsets to circular scan total power
        target = ds.catalogue.targets[ds.target_indices[0]]
        if chanmapping:
            ds.select(channels=chanmapping(target.name, fGHz))
        mask = flag_circscancycle(ds.target_x[...,ant_ix], ds.target_y[...,ant_ix], discard=discard)
        hv = np.abs(ds.vis[:])
        height = np.mean(hv/np.mean(hv,axis=0), axis=(1,2)) # Normalise for gains then make TOTAL power integrated over frequency
        t_ref = np.mean(ds.timestamps[mask])
        rAz, rEl = target.azel(t_ref, antenna=ant)
        if debug: # Prepare figure for debugging
            axs = plt.subplots(1,3, figsize=(14,3))[1]
            axs[0].plot(ds.channels, np.mean(np.abs(ds.vis[:]), axis=(0,2)), '.')
            axs[1].plot(ds.timestamps[mask], ds.target_x[mask,...], '.', ds.timestamps[mask], ds.target_y[mask,...]) # Also plots track antenna if present
        xoff, yoff, valid = fit_gaussianoffset(ds.target_x[mask,ant_ix], ds.target_y[mask,ant_ix], height[mask],
                                               powerbeam=(track_ant is None), debug=axs[2] if debug else None)
        try:
            aAz, aEl = target.plane_to_sphere(xoff/R2D, yoff/R2D, t_ref, antenna=ant, coord_system='azel') # [rad]
        except: # Sometimes if fit is way out it may appear to be in the other hemisphere - OutOfRangeError!
            aAz, aEl = np.nan, np.nan
        dAz, dEl = (aAz - rAz)*R2D, (aEl - rEl)*R2D
        if debug: # Report
            plt.suptitle("%s %s [Fit: %s]"%(ds.compscan_indices, target, valid))
        if debug or verbose:
            print("%s Fit: %s\t xy offsets %g, %g, AzEl offsets %g, %g [arcsec]"%(ds.compscan_indices, valid, xoff*3600, yoff*3600, dAz*3600, dEl*3600))
        
        if strict and not valid:
            dAz, dEl = np.nan, np.nan
        offsets.append((t_ref, target.name, dAz, dEl))
        
    if debug or verbose:
        print("Std [arcsec]", np.nanstd([o[2] for o in offsets])*3600, np.nanstd([o[3] for o in offsets])*3600)
    return offsets


def _test_reduce_circular_pointing_():
    np.random.seed(1)
    
    class TestDataset(object):
        def __init__(self, f_c, ant_names, targets_scanned, n_cycles, Dant=15):
            """ @param f_c: center frequency, in Hz
                @param ant_names: just the names - they will get random coordinates in the Karoo.
                @param targets_scanned: list of target names, one entry for each "scan", in order.
                @param n_cycles: the number of cycles at each "scan"
                @param Dant: diameter of the individual dish [meter]
            """
            self.spectral_windows = [typing.NamedTuple('SpectralWindow', centre_freq=float)(f_c)]
            self.channels = np.arange(16) # Not important, this keeps it easy 
            self.ants = [katpoint.Antenna(a, -np.pi/3+0.1*np.random.rand(), np.pi/3.3+0.1*np.random.rand(), 1050) for a in ant_names] # Roughly in the Karoo
            self.catalogue = katpoint.Catalogue(antenna=self.ants[0], add_specials=True)
            for t in set(targets_scanned):
                if (t not in self.catalogue):
                    self.catalogue.add("%s, radec, %.3f, %.3f" % (t, 24*np.random.rand(), -90*np.random.rand()))
            self.__targets_scanned__ = targets_scanned
            self.__n_cycles__ = n_cycles
            self.__hpbw__ = 1.22*(3e9/f_c)/Dant * 180/np.pi # [deg]
            self.__set_testopts__() # Defaults

        def select(self, *a, **k): # For test purposes we may ignore the "select()" in reduce_circular_pointing()
            pass
        
        def __set_testopts__(self, kind="cardioid", scanrad='hpbw', ampl=1, SEFD=300, BW=1e6, ox=0,oy=0):
            """ @param kind: "circle"|"cardioid"|"epicycles"
                @param scanrad: 'hpbw' or radius to scan at, in [degrees]
                @param ampl, SEFD: Jy per pol
                @param ox, oy: "error" offsets in degrees
            """
            self.__testopts__ = [kind, scanrad, ampl, SEFD, BW, ox, oy]
                    
        def scans(self):
            """ Generate the test data for a single cycle at a time (yields) """
            for i,tgt in enumerate(self.__targets_scanned__):
                self.compscan_indices = [i]
                self.target_indices = [k for k,t in enumerate(self.catalogue.targets) if (tgt in t.name)]
                for s in range(self.__n_cycles__):
                    kind, scanrad, ampl, SEFD, BW, ox,oy = self.__testopts__
                    scanrad = self.__hpbw__ if (scanrad == 'hpbw') else scanrad 
                    t, x, y, m = _generate_test_data_(kind, powerbeam=len(self.ants)==1, ampl=ampl/2,SEFD=SEFD/2, Nsamples_per_dump=BW*1,
                                                      hpbw=self.__hpbw__, scanrad=scanrad, ox=ox, oy=oy)
                    m = np.transpose(np.stack([m]*len(self.channels),axis=0)) # time,freq
                    self.target_x, self.target_y, self.vis = np.stack([x],axis=1), np.stack([y],axis=1), np.stack([m/2,m/2],axis=2)
                    self.timestamps = t + (t[-1]-t[0])*(i*self.__n_cycles__ + s)
                    yield

    # Somewhat representative of DVS Ku-band
    ds = TestDataset(13.5e9, ["s0000"], ["Jupiter"], n_cycles=3)
    for ox,oy in [(0,0),(120/3600,0),(0,20/3600)]:
        ds.__set_testopts__(kind="cardioid", scanrad='hpbw', ampl=10/2, SEFD=700/2, BW=10e6, ox=ox,oy=oy)
        reduce_circular_pointing(ds, ds.ants[0].name, None, track_ant=None, strict=True, discard=1, verbose=True, debug=False)
    # Somewhat representative of DVS Ku-band
    ds = TestDataset(11.5e9, ["s0000"], ["GEOS"], n_cycles=5)
    for ox,oy in [(0,0),(120/3600,0),(0,20/3600)]:
        ds.__set_testopts__(kind="cardioid", scanrad='hpbw', ampl=500/2, SEFD=700/2, BW=.3e6, ox=ox,oy=oy)
        reduce_circular_pointing(ds, ds.ants[0].name, None, track_ant=None, strict=True, discard=1, verbose=True, debug=False)


def analyse_circ_scans(fn, ants, chanmapping, debug=True, verbose=False, **kwargs): # TODO: eventually generate "APSS-like" csv file
    """ Generates pointing offsets for a dataset created with circular_pointing.py
        @param fn: the URL for the dataset
        @param ants: the identifiers of the scanning antennas in the dataset
        @param chanmapping: channel indices to use or a function like `lambda target_name,fGHz: channel_indices`, or None to use all channels
        @param kwargs: extra arguments for reduce_circular_pointing()
        @return: [results_ant0, ... (for each ant in ants)] with 'results_ant' a list of (timestamp, target, dAz, dEl) for all cycles
    """
    results = []
    for ant in ants:
        ds = katdal_open(ant, fn)
        offsets = reduce_circular_pointing(ds, ant, chanmapping, debug=debug, verbose=verbose, **kwargs)
        results.append(offsets)
        
    if verbose: # Plot the offsets
        symbols = ['.', '+', '^']
        axs = plt.subplots(2,1, figsize=(14,5))[1]
        for ant,offsets,fmt in zip(ants,results,symbols):
            oss = np.array([[m[0],m[2],m[3]] for m in offsets])
            axs[0].plot(oss[:,0], (oss[:,1]-np.nanmedian(oss[:,1]))*3600, fmt, label=ant)
            axs[1].plot(oss[:,0], (oss[:,2]-np.nanmedian(oss[:,2]))*3600, fmt, label=ant)
        axs[0].legend(); axs[0].set_ylabel("dAz"); axs[1].set_ylabel("dEl");
    
    return results


def katdal_open(sys, fn, *args, **kwargs):
    """ Work around misalignment of scna boundaries and data, due to inconsistent implementation of Receptor/DishProxy COMMAND_TIME_OFFSET.
        1. All datasets have s0000 activities misaligned from data by 10sec but m028 by 1.2sec. The following fixes that for s0000. 
        ``katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = 10 ``
        2. In most cases pointing coordinates are misaligned from visibilities by ~0.1sec (LMC sample & hold @ 0.2sec), the following improves that
        ``katdal.open(timingoffset=0.1)``

        Use katdal_open("m", ...) instead of katdal.open(...)
        
        @param sys: "m*" or "s*", only the first character is considered.
        @param kwargs: e.g. 'X' to force the time_offset
        @return: katdal.Dataset """
    curr_val = katdal.visdatav4.SENSOR_PROPS['*activity'].get('time_offset', 0)
    try:
        if sys.startswith("m"):
            X = kwargs.get("X", 1.2)
            katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = X # Some datasets (where plan_targets() takes too long) require anything [0.1,0.7]
            timingoffset = kwargs.pop("timingoffset", 0)
        elif sys.startswith("s"):
            X = kwargs.get("X", 10)
            katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = X
            timingoffset = kwargs.pop("timingoffset", 0.1)
        else:
            katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = 0
            timingoffset = kwargs.pop("timingoffset", 0)
        return katdal.open(fn, *args, timingoffset=timingoffset, **kwargs)
    finally: # Safe to restore after katdal.open() completed
        katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = curr_val


def make_apss_record(ds, offsets, sigma):
    """ Compiles a record like what's generated by analyse_point_source_scans.py
        @param ds: the dataset with selection applied
        @parma offsets: total power centroid offsets [dxEl, dEl] in deg.
        @parma sigma: width of total power pattern [xEl, El] in deg.
        @return: [dataset, target, timestamp_string, az, el, delta_az, delta_az_std, delta_el, delta_el_std, ..., timestamp_sec]
    """
    timestamp = np.mean(ds.timestamps)
    dataset, target, timestamp_ut = ds.name.split(" ")[-1], np.take(ds.catalogue.targets, ds.target_indices)[0].name, str(katpoint.Timestamp(timestamp))
    azimuth, elevation = np.mean(ds.az), np.mean(ds.el) # deg
    delta_azimuth, delta_elevation = offsets[0]/np.cos(elevation*180/np.pi), offsets[1] # deg 
    delta_azimuth_std, delta_elevation_std = 0, 0
    beam_width_HH, beam_width_VV = sigma[0]/np.cos(elevation*180/np.pi), sigma[1] # deg 
    beam_width_I, beam_width_I_std = (beam_width_HH+beam_width_VV)/2., 0
    
    data_unit, beam_height_I, beam_height_I_std = "counts", 1, 0
    baseline_height_I, baseline_height_I_std, refined_I = 0, 0, 1
    beam_height_HH, baseline_height_HH, refined_HH = 1, 0, 1
    beam_height_VV, baseline_height_VV, refined_VV = 1, 0, 1
    frequency, flux = np.mean(ds.freqs), 0
    
    temperature = np.mean(ds.temperature)
    pressure = np.mean(ds.pressure)
    humidity = np.mean(ds.humidity)
    # Do a 2-D vector average of wind speed + direction
    raw_wind_speed = ds.wind_speed
    raw_wind_direction = ds.wind_direction
    mean_north_wind = np.mean(raw_wind_speed * np.cos(np.radians(raw_wind_direction)))
    mean_east_wind = np.mean(raw_wind_speed * np.sin(np.radians(raw_wind_direction)))
    wind_speed = np.sqrt(mean_north_wind ** 2 + mean_east_wind ** 2)
    wind_direction = np.degrees(np.arctan2(mean_east_wind, mean_north_wind))
    # Sun angle relative to the antenna
    sun = katpoint.Target('Sun, special')
    sun_az, sun_el = katpoint.rad2deg(np.array(sun.azel(timestamp, antenna=ds.ants[0])))
    
    fields = f'{dataset:s}, {target:s}, {timestamp_ut:s}, {azimuth:.7f}, {elevation:.7f}, ' \
             f'{delta_azimuth}.7f, {delta_azimuth_std}.7f, ' \
             f'{delta_elevation}.7f, {delta_elevation_std}.7f, ' \
             f'{data_unit}s, {beam_height_I}.7f, {beam_height_I_std}.7f, ' \
             f'{beam_width_I}.7f, {beam_width_I_std}.7f, ' \
             f'{baseline_height_I}.7f, {baseline_height_I_std}.7f, {refined_I}.7f, ' \
             f'{beam_height_HH}.7f, {beam_width_HH}.7f, {baseline_height_HH}.7f, {refined_HH}.7f, ' \
             f'{beam_height_VV}.7f, {beam_width_VV}.7f, {baseline_height_VV}.7f, {refined_VV}.7f, ' \
             f'{frequency}.7f, {flux}.4f, ' \
             f'{temperature}.2f, {pressure}.2f, {humidity}.2f, {wind_speed}.2f, ' \
             f'{wind_direction}.2f , {sun_az}.7f, {sun_el}.7f, {timestamp}i'
    return fields


if __name__ == "__main__":
    _test_fit_gaussianoffset_()
    _test_reduce_circular_pointing_()
    plt.show()
