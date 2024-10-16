'''
    Analyse "RF Tracking Stability" measured by using circular_pointing.py
    CAUTION: consider this still work in progress!
    
    @author: aph@sarao.ac.za
'''
import numpy as np
import scipy.optimize as sop
import pylab as plt
import katpoint
import katdal


def fit_gaussianoffset(x, y, height, powerbeam=True, check_convergence=True, debug=None):
    """ @param x, y: position coordinates around the circular scan trajectory
        @param heigh: heigh above the baseline, measured along the trajectory
        @param powerbeam: True if scanned in total power, False if scanned in complex amplitude
        @return: xoffset, yoffset, valid """
    # BRITTLENESS WARNING
    # Circular scan provides very little information to fit all of the Gaussian's free variables reliably,
    # seems vital to start with very good guesses, and constrain the fit!
    
    # Circular scan is nominally around the half-power point.
    fwhm = 2*np.median(x**2 + y**2)**.5 # Full width spanning half max, i.e. model(@fwhm/2)~0.5
    xoffset, yoffset = 0, 0 # Don't use mean(x,y) because x & y may be partial orbits
    if not powerbeam: # Scan referenced to a tracking antenna scans the voltage beam, but likely also along HP contour
        height = np.abs(height)**2
    height /= np.std(height)
    # Auto & cross correlations are different: auto can have weak signal riding on large system noise, so need to model offset
    ampl = np.median(height) # Close for "cross-corr" scans, but "high" for typical "total power" targets. Results not very sensitive to this.
    h0 = max([0, np.median(height)-6]) # 0 for "cross-corr", and for "total power" targets are typically weak so 6 sigma if the amplitude "in the noise"

    model = lambda ampl,xoffset,yoffset,wx,wy, h0=0: abs(h0) + ampl*np.exp(-4*np.log(2) * ((x-xoffset)**2/wx**2+(y-yoffset)**2/wy**2))
    
    p0 = [2*ampl,xoffset,yoffset,0.5*fwhm,0.5*fwhm] + ([h0] if powerbeam else []) # Starting narrower seems best
    # Fits easily converge to local minima if ampl AND wx,wy are free variables
    if True: # This constrained fit is quite robust, also if bounds change a bit
        if powerbeam:
            bounds = [(0,np.inf)] +( [(-fwhm/2,fwhm/2)]*2 ) + ( [(0.2*fwhm,1.5*fwhm)]*2 ) + [(0,np.min(height))]
        else:
            bounds = [(2*ampl,np.inf)] +( [(-fwhm/2,fwhm/2)]*2 ) + ( [(0.2*fwhm,0.7*fwhm)]*2 )
        p = sop.least_squares(lambda p: height-model(*p), p0, bounds=np.transpose(bounds), verbose=0)
        if p.success and check_convergence: # Check for sensitivity of offsets wrt. bounds
            if powerbeam:
                bounds = [(0,np.inf)] +( [(-fwhm/2,fwhm/2)]*2 ) + ( [(0.1*fwhm,2*fwhm)]*2 ) + [(0,np.min(height))]
            else:
                bounds = [(2*ampl,np.inf)] +( [(-fwhm/2,fwhm/2)]*2 ) + ( [(0.1*fwhm,1*fwhm)]*2 )
            p_ = sop.least_squares(lambda p: height-model(*p), p0, bounds=np.transpose(bounds), verbose=0)
            converged = p_.success and np.all(np.abs(1-p.x[1:3]/p_.x[1:3]) < 0.05) # Within 5%
            p.success = converged
    else: # TODO: unconstrained fit is VERY BRITTLE, results very sensitive to method (BFGS seems best, but Powell, Nelder-Mead, LM ...)
        p = sop.minimize(lambda p: np.nansum((height-model(*p))**2), p0, method='BFGS', options=dict(disp=False))
    p, valid = p.x, p.success
    ampl, xoffset, yoffset, fwhmx, fwhmy = p[:5]
    h0 = p[5] if powerbeam else 0
    model_height = model(*p)
    snr = (np.median(height) - h0) / np.std((height - model_height)[1:-1])
    
    # 'success' status isn't quite relevant - multiple minima?
    # The following criteria are somewhat arbitrary but empirically OK over many datasets:
    valid = valid and (snr > 4) # Less than 4 sigma is often visually a poor fit
    valid = valid and (np.std(height - model_height) < 0.6) # Some pathological cases where h0 isn't fitted well for SNR to be accurate
#     valid = valid and np.all(np.array([fwhmx,fwhmy])/fwhm < 1.3) # Throw out fits that get stuck at upper bound
#     valid = valid and np.all(np.abs(1 - np.array([fwhmx,fwhmy])/fwhm) < 0.3) # Throw out fits that get stuck at bounds
    
    if debug is not None:
        debug.plot(height, '.', alpha=0.5)
        debug.plot(model_height, '-', label="%.1f exp(Q(%.2f, %.2f)) + %.1f"%(ampl, fwhmx/fwhm, fwhmy/fwhm, h0))
        debug.legend()
    return xoffset, yoffset, valid


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


def reduce_circular_pointing(ds, ant, chanmapping, track_ant=None, strict=True, discard=1, check_convergence=True, verbose=True, debug=False):
    """ Generates pointing offsets for a dataset created with circular_pointing.py
        @param ds: the dataset (selection is reset!)
        @param ant: the identifier of the scanning antenna in the dataset
        @param chanmapping: like `lambda target_name,fGHz: channel_indices`
        @param track_ant: the identifier of the tracking antenna in the dataset, if not single dish mode (default None)
        @param strict: True to set invalid fits to nan (default True)
        @param discard: strategy for identifying outliers, either '1sigma' or a percentile (both wrt. deviations from fitted orbit)
        @return: [(timestamp [sec], target ID, dAz, dEl [deg]), ...(for each cycle)]
    """
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
        ds.select(channels=chanmapping(target.name, fGHz))
        mask = flag_circscancycle(ds.target_x[...,ant_ix], ds.target_y[...,ant_ix], discard=discard)
        h = np.mean(np.abs(ds.vis[:]),axis=(1,2)) # TOTAL power integrated over frequency
        t_ref = np.mean(ds.timestamps[mask])
        rAz, rEl = target.azel(t_ref, antenna=ant)
        if debug: # Prepare figure for debugging
            axs = plt.subplots(1,3, figsize=(14,3))[1]
            axs[0].plot(ds.channels, np.mean(np.abs(ds.vis[:]), axis=(0,2)), '.')
            axs[1].plot(ds.timestamps[mask], ds.target_x[mask,...], '.', ds.timestamps[mask], ds.target_y[mask,...]) # Also plots track antenna if present - diagnostic
#             axs[2].plot(ds.timestamps[mask], h[mask], '.') # Instead of debug below
        xoff, yoff, valid = fit_gaussianoffset(ds.target_x[mask,ant_ix], ds.target_y[mask,ant_ix], h[mask],
                                               powerbeam=(track_ant is None), check_convergence=check_convergence, debug=axs[-1] if debug else None) # Positions on tangent plane centred on target [degrees]
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


def analyse_circ_scans(fn, ants, chans, debug=True, **kwargs): # TODO: eventually generate "APSS-like" csv file
    results = []
    for ant in ants:
        ds = katdal_open(ant, fn)
        offsets = reduce_circular_pointing(ds, ant, chans, debug=debug, **kwargs)
        results.append(offsets)
        
    # Plot the offsets
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