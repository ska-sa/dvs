'''
    Lightweight centroid fitting for "intensity mapped" beam measurements,
    specifically for use with `circular_pointing.py`.
    
    
    @author: aph@sarao.ac.za
'''
import numpy as np
import scipy.optimize as sop
import pylab as plt
import katpoint
import typing
from analysis import katselib, katsepnt
from analysis.katsemat import sliding_window


_c_ = 299792458
R2D = 180/np.pi


def fit_background(x, y, intensity_map, along_edges=True):
    """ Fit a linear background to the intensity map
        @param x, y, intensity_map: 1D vectors
        @param along_edges: True to fit only using data along the edges (default True)
        @return: the background map """
    mask = np.full(len(intensity_map), 1.0)
    if along_edges:
        r = np.squeeze((x**2+y**2)**.5)
        mask[r < np.percentile(r,66)] = np.nan
    
    model = lambda x0,y0, mx,my, nx=0,ny=0: mx*(x-x0)+nx*(x-x0)**2 + my*(y-y0)+ny*(y-y0)**2
    p0 = [0,0, 0,0] # + [0,0] # to make it quadratic
    
    p = sop.minimize(lambda p: np.nansum(mask*(intensity_map-model(*p))**2), p0, method='BFGS', options=dict(disp=False))
    model_bg = model(*p.x)
    model_bg += np.nanmin(intensity_map - model_bg) - 0.1 # Ensure the background is always lower than the map

    return model_bg


def gaussian2D(x, y, x0, y0, wx, wy, theta=0):
    """ @param theta: rotation angle [rad]
        @return: Un-normalized Gaussian that's been rotated around its center. """
    # Rotate the x,y coordinates by theta
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    a = cos_theta*x - sin_theta*y
    b = sin_theta*x + cos_theta*y
    a0 = cos_theta*x0 - sin_theta*y0
    b0 = sin_theta*x0 + cos_theta*y0
    # Evaluate the Gaussian
    return np.exp( -np.log(2) * ( (a-a0)**2/wx**2 + (b-b0)**2/wy**2) )

def fit_gaussianoffset(x, y, height, powerbeam=True, constrained=True, constrain_ampl=None, constrain_hpbw=None, constrain_center=True, debug=None):
    """ Fit a Gaussian to the magnitude measured in tangent plane coordinates.
        NB: this does not fit the background, so only works reliably if the background is "flat".
        
        @param x, y: position coordinates in the tangent plane.
        @param height: height above the baseline, measured along the trajectory defined by the coordinates.
        @param powerbeam: True if scanned in total power, False if scanned in complex amplitude
        @param constrain_ampl, constrain_hpbw: > 0 to constrain the fitted parameters to within 20% of this.
        @param constrain_center: True to suppress signal beyond constrain_hpbw/2 from the center of the plane.
        @return: (xoffset, yoffset, valid, xfwhm, yfwhm, rot, ampl, resid) - positions on tangent plane centred on target in same units as `x` & `y` """
    # Starting estimates
    # These measurements (power & voltage) are typically done by scanning around half _power_ contour
    r = np.squeeze((x**2+y**2)**.5)
    scanext = 2*np.median(r)
    width0 = constrain_hpbw if (constrain_hpbw) else scanext/2
    ampl0 = constrain_ampl if (constrain_ampl) else (np.max(height) - np.min(height))
    p0 = [ampl0,0,0,width0,width0,0, np.min(height)]
    
    mask = np.full(len(height), 1.0)
    if constrain_center: # Suppress things towards the periphery
        mask = 2./(1+r/width0); mask[r < width0] = 1
    
    if powerbeam:
        model = lambda ampl,xoffset,yoffset,wx,wy,rot, h0=0: h0 + ampl*gaussian2D(x,y,xoffset,yoffset,wx,wy,rot)
    else: # Scan referenced to a tracking antenna scans the voltage beam, convert it to power
        SQRT2 = 2**.5
        model = lambda ampl,xoffset,yoffset,wx,wy,rot, h0=0: h0 + ampl/SQRT2*gaussian2D(x,y,xoffset,yoffset,wx*SQRT2,wy*SQRT2,rot)
    
    # Fit model to data
    if not constrained: # Unconstrained fit is VERY BRITTLE, results also very sensitive to method (BFGS seems best, but Powell, Nelder-Mead, LM ...)
        p = sop.minimize(lambda p: np.nansum(mask*(height-model(*p))**2), p0, method='BFGS', options=dict(disp=False))
    else: # Constrained fits should be robust, and not very sensitive to bounds
        bounds = [(0,np.inf)] + ( [(-width0,width0)]*2 ) + ( [(0.1*width0,2*width0)]*2 ) + [(-np.pi/2,np.pi/2)] + [(0,np.min(height))]
        if (constrain_ampl): # Explicit constraint for e.g. circular scan
            bounds[0] = (0.20*ampl0, 1.20*ampl0)
        if (constrain_hpbw):
            bounds[3] = bounds[4] = (0.80*width0,1.20*width0)
        p = sop.least_squares(lambda p: mask*(height-model(*p)), p0, bounds=np.transpose(bounds), verbose=0)
    ampl, xoffset, yoffset, fwhmx, fwhmy, rot, h0 = p.x
    model_height = model(*p.x)
    
    # 'p.success' status isn't quite reliable (e.g. multiple minima)
    # Also check deduced signal-to-noise, and residuals must be "in the noise"
    resid = np.mean((height - model_height)**2)**.5
    valid = p.success and (ampl/resid > 7) # 3sigma is the absolute minimum for reliable fits! 
    valid = valid and (fwhmx/scanext < 0.99) and (fwhmy/scanext < 0.99) # Width must be smaller than measured extent for reliable fits - especially when fit is constrained! 
    valid = valid and (xoffset/scanext < 0.8) and (yoffset/scanext < 0.8) # A fit is not reliable if it extrapolates far outside the measured region!
    
    if debug is not None: # 'debug' is expected to be two plot axes: first one for amplitude series, second one for x,y
        debug[0].plot(model_height, 'k-', alpha=0.5, label="%.1f + %.1f exp(Q(%.2f, %.2f))"%(h0, ampl, fwhmx/scanext, fwhmy/scanext))
        debug[0].legend()
        debug[1].plot([xoffset], [yoffset], 'r+'); debug[1].text(xoffset, yoffset, "%.2f, %.2f"%(xoffset,yoffset), fontsize='x-small')
    return (xoffset, yoffset, valid, fwhmx, fwhmy, rot, ampl, resid)


def _generate_test_data_(kind, powerbeam=True, hpbw=0.01, ampl=5, SEFD=200, Nsamples_per_dump=1e6*1, scanrad='hpbw', ox=0, oy=0):
    """ @param kind: "circle"|"cardioid"|"epicycle"|"raster"
        @param ampl, SEFD: powers per polarisation channel [Jy]
        @param Nsamples_per_dump: BW*tau
        @param scanrad: radius for the scan pattern, either "hpbw" or a number in same units as `hpbw`.
        @return: (timestamps, target_x, target_y, magnitudes) """
    scanrad = hpbw if (scanrad == 'hpbw') else scanrad
    
    timestamps = np.arange(12*12) # Should be a square to make "raster" simple
    
    th = np.linspace(0, 2*np.pi, len(timestamps)) # One full cycle (should exclude the last one which overlaps)
    if (kind in ["circle","cardioid"]): # "circle" is a trivial case of "cardioid"
        a, b = (1, 0) if (kind == "circle") else (0.4, 1.5)
        a, b = a*scanrad, b*scanrad
        c = b if (b < a) else (b+a**2/4/b+a) # The cardioid is offset in the x direction
        radius = a + b*np.cos(th)
        target_x = radius*np.cos(th) - c/2
        target_y = radius*np.sin(th)
    elif (kind == "epicycle"):
        r2 = scanrad/2 # Sets the ratio of the circles to go between scanrad and center
        n = 4 # 2 looks like a cardioid, 3 is too "sparse"
        target_x = r2*np.cos(th) + r2*np.cos(th*n)
        target_y = r2*np.sin(th) + r2*np.sin(th*n)
    elif (kind == "raster"):
        N = int(len(timestamps)**.5) # We ensure timestamp is square!
        target_x, target_y = np.meshgrid(np.linspace(-hpbw,hpbw,N), np.linspace(-hpbw,hpbw,N))
        target_x[::2,:] = target_x[::2,::-1]
        target_x, target_y = target_x.ravel(), target_y.ravel()
    actual_x = target_x - ox
    actual_y = target_y - oy
    
    if powerbeam: # TOTAL power integrated over frequency
        hv = ampl*np.exp(-np.log(2)*(actual_x**2+actual_y**2)/hpbw**2)**2 + SEFD*(1 + 1/Nsamples_per_dump**.5*np.random.randn(len(target_x))**2)
    else: # Cross-corr voltage amplitude integrated over frequency
        hv = ampl**.5*np.exp(-np.log(2)*(actual_x**2+actual_y**2)/hpbw**2) + (SEFD*(0 + 1/Nsamples_per_dump**.5*np.random.randn(len(target_x))**2))**.5
    
    return (timestamps, target_x, target_y, hv)


def _demo_fit_gaussianoffset_(hpbw=11, ampl=1, SEFD=200, cycles=100):
    """ Demonstrate the fits for single dish & interferometric measurements, with different scan patterns.
    
        @param hpbw: half power beam width to use for simulated pattern [angle units e.g. arcmin] - pointing residuals will be in the same units (default 11)
        @param ampl: Jansky per pol for simulated target [power e.g. Jansky] (default 1)
        @param SEFD: Jansky per pol for simulated measurement noise (same power units as `ampl`) (default 200)
        @param cycles: number of measurements to simulate; if 1 then will only plot the raw measurement and fit (default 1).
    """ 
    # Default values are meant to be typical values for SKA-MID Band 5b?
    Nsamples_per_dump = 1e6*1 # 1MHz * 1sec
    
    if (cycles == 1): # Debug plots for a specific offset, all different kinds
        ox,oy = (hpbw/6, hpbw/11)
        for powerbeam in [True, False]: # Power & voltage beams
            axs = plt.subplots(3, 4)[1]
            for i,kind in enumerate(['circle','cardioid','epicycle','raster']):
                # Same pointing offsets for each 'kind'
                np.random.seed(1)
                timestamps,target_x,target_y,m = _generate_test_data_(kind, powerbeam=powerbeam, hpbw=hpbw, ampl=ampl, SEFD=SEFD,
                                                                      Nsamples_per_dump=Nsamples_per_dump, scanrad='hpbw', ox=ox, oy=oy)
                axs[0][i].set_title(kind)
                axs[0][i].plot(target_x, target_y, '.')
                axs[1][i].plot(timestamps, target_x, '.', timestamps, target_y, '.')
                axs[1][i] = axs[1][i].twinx(); axs[1][i].plot(m, 'k.-')
                axs[2][i].scatter(target_x, target_y, m)
                xoff, yoff, valid, hpwx, hpwy, rot, a0, resid = fit_gaussianoffset(target_x, target_y, m, powerbeam=powerbeam, debug=[axs[1][i],axs[2][i]])
                print(kind, "x: %g -> %g"%(ox, xoff), "y: %g -> %g"%(oy, yoff), "valid: %s"%valid, "HPBW %.3f -> %.3f, %.3f"%(hpbw, hpwx, hpwy), "ampl %g -> %g"%(ampl,a0))
    
    else: # Statistical
        axes = plt.subplots(2,3)[1]
        for powerbeam,axs in zip([True, False], axes): # Power & voltage beams
            axs[0].set_ylabel(" SD" if powerbeam else " INTF")
            for kind in ['circle','cardioid','epicycle','raster']:
                # Same pointing offsets for each 'kind'
                np.random.seed(1)
                offsets = hpbw * np.c_[np.random.rand(cycles) - 0.5, np.random.rand(cycles) - 0.5]
                fits = []
                for ox,oy in offsets:
                    timestamps,target_x,target_y,m = _generate_test_data_(kind, powerbeam=powerbeam, hpbw=hpbw, ampl=ampl, SEFD=SEFD,
                                                                          Nsamples_per_dump=Nsamples_per_dump, scanrad='hpbw', ox=ox, oy=oy)
                    xoff, yoff, valid, hpwx, hpwy, rot, a0, resid = fit_gaussianoffset(target_x, target_y, m, powerbeam=powerbeam, debug=None)
                    fits.append([ox-xoff, oy-yoff, hpwx/hpbw, hpwy/hpbw, a0/ampl])
                fits = np.asarray(fits)
                axs[0].hist(fits[:,0], bins=20, range=(-hpbw,hpbw), alpha=0.5, label=kind+" X")
                axs[0].hist(fits[:,1], bins=20, range=(-hpbw,hpbw), alpha=0.5, label=kind+" Y")
                axs[1].hist(fits[:,2], bins=20, range=(0,2), alpha=0.5, label=kind+" X")
                axs[1].hist(fits[:,3], bins=20, range=(0,2), alpha=0.5, label=kind+" Y")
                axs[2].hist(fits[:,4], bins=20, range=(0,2), alpha=0.5, label=kind)
            for ax,unit in zip(axs,["$\Delta$pointing", "fit_hpw/hpbw", "fit_ampl/ampl"]):
                ax.legend(); ax.set_xlabel(unit)


def reduce_pointing_scans(ds, ant, chans=None, freq_MHz=None, track_ant=None, phased_up=False, flags='cam,data_lost', scans="~slew", compscans="~slew",
                          polswap=None, kind=None, min_len=20, clip_radius=None, strict=True, output_filepattern="%s_%s_circular_pointing.csv",
                          verbose=True, debug=False):
    """ Generates pointing offsets for a dataset created with (any) intensity mapping technique (point_source_scan.py, circular_pointing.py etc),
        exactly equivalent to how `analyse_point_source_scans.py` calculates it.
    
        @param ds: the dataset (selection is reset!)
        @param ant: the identifier of the scanning antenna in the dataset
        @param chans: channel indices to use, or a function like `lambda target_name,fGHz: channel_indices`, or None to use the pre-existing selection.
        @param freq_MHz: (f_MHzstart,f_MHzstop) to use instead of chans (default None)
        @param track_ant: the identifier of the tracking antenna in the dataset, if not single dish mode (default None)
        @param phased_up: True if the phase slope between ant & track_ant products is <<180deg (default False)
        @param flags: the katdal flags to apply to the data, or None (default 'cam,data_lost') 
        @param polswap: a comma-separated list of antenna IDs where the polarisation is swapped (default None)
        @param kind: specifically used with 'circle','cardioid','epicycle' from "circular_pointing.py"
        @param min_len: the minimum number of data points required to fit a centroid on (default 20).
        @param clip_radius: maximum radius around target to use for fit, or None for 'median+std' [deg] (default None)
        @param strict: True to set invalid fits to nan (default True)
        @param output_filepattern: filename pattern (with %s for dataset and antenna names) for CSV file to store results to (default '%s_%s_circular_pointing.csv')
        @return: ( [(timestamp [sec], target ID [string], Az, El, dAz, dEl, hpw_x, hpw_y [deg], ampl, resid, bkgnd [power]), ...(for each cycle)]
                   [(temperature, pressure, humidity, wind_std, wind_speed, wind_dir, sun_Az, sun_El, feedindexer_angle), ...(for each cycle)] )
    """
    if freq_MHz is not None:
        ds.select(reset="F")
        freqsMHz = ds.freqs/1e6
        df = abs(freqsMHz[1]-freqsMHz[0])
        f_start, f_stop, *_ = list(np.atleast_1d(freq_MHz))*2 # Allow a single value to be given
        chans = (f_start-.5*df <= freqsMHz) & (freqsMHz <= f_stop+.5*df)
    
    if (not callable(chans)) and (chans is not None): # Convert a mask array to a function
        _chans_ = chans
        chans = lambda *a: _chans_
    
    fitted = [] # (timestamp, target, Az, El, dAz, dEl, hpw_x, hpw_y, ampl, resid)
    enviro = [] # (temperature, pressure, humidity, wind_std, wind_speed, wind_dir, sun_az, sun_el, wind_dynamic, feedindexer_angle)
    
    ds.select(reset="", scans=scans, compscans=compscans)
    ds.select(reset="", compscans="~unwrap") # Definitely don't want these - OK to hard-code this.
    polswap = "" if polswap is None else polswap
    track_ant = "" if track_ant is None else track_ant
    pols = {a:"HV" if (a not in polswap) else "VH" for a in [ant,track_ant]}
    if (track_ant):
        pols_to_use = [pols[ant][i]+pols[track_ant][i] for i in (0,1)]
        ds.select(corrprods="cross", pol=pols_to_use, ants=[ant,track_ant])
    else:
        pols_to_use = [pols[ant][i]+pols[ant][i] for i in (0,1)]
        ds.select(pol=pols_to_use, ants=[ant])
    if flags:
        ds.select(flags=flags)
    fGHz = np.round(ds.spectral_windows[0].centre_freq/1e9, 4)
    
    scan_ant_ix = [a.name for a in ds.ants].index(ant)
    scan_ant = ds.ants[scan_ant_ix]
    avgws_timestamps, avgws = ds.timestamps[:], sliding_window(ds.timestamps, ds.wind_speed, int(1000/(ds.timestamps[1]-ds.timestamps[0])+0.5), np.mean)
    fi_sensor = "%s_ap_indexer_position_raw" if ant.startswith('m') else "%s_dsm_indexerActualPosition" # MeerKAT or MKE Dish
    fi_timestamps, fi_angles = katselib.getsensorvalues(fi_sensor%ant, ds.timestamps)
    
    sun = katpoint.Target('Sun, special')
    rc = katpoint.RefractionCorrection()
    wrap_angle = lambda angle, period=360: (angle + 0.5*period) % period - 0.5*period
    # Fit offsets to an "intensity map"
    for (cs_no, cs_label, target) in ds.compscans():
        if chans:
            ds.select(channels=chans(target.name, fGHz))
        if (ds.channels[0] == 0): # Always omit the 0th channel (FFT bin 0)
            ds.select(reset="", channels=ds.channels[1:])
        
        mask = np.any(~ds.flags[:],axis=(1,2)) if flags else np.full(ds.timestamps.shape, True)
        if (len(ds.timestamps[mask]) == 0): # All points flagged out
            continue
        # Also omit data points that are far from the majority - to avoid stray points from skewing the fit
        scan_r = np.squeeze((ds.target_x[...,scan_ant_ix]**2+ds.target_y[...,scan_ant_ix]**2)**.5)
        mask &= scan_r < (clip_radius if clip_radius else (np.median(scan_r) + 1*np.std(scan_r)))
        scan_r = scan_r[mask]
        if (len(scan_r) == 0) or (np.max(scan_r) > 6): # All points flagged, or large scan offset may mean antenna lagged behind e.g. near zenith or unwrap!
            continue
        
        # Obtain middle timestamp of compound scan, where all pointing calculations are done
        t_ref = np.nanmedian(ds.timestamps[mask])
        
        # Environmental parameters
        sun_azel = katpoint.rad2deg(np.array(sun.azel(t_ref, antenna=scan_ant)))
        temperature, pressure, humidity = np.mean(ds.temperature[mask]), np.mean(ds.pressure[mask]), np.mean(ds.humidity[mask])
        # Do a 2-D vector average of wind speed + direction
        raw_wind_speed = ds.wind_speed[mask]
        raw_wind_direction = ds.wind_direction[mask]/R2D
        mean_north_wind = np.mean(raw_wind_speed * np.cos(raw_wind_direction))
        mean_east_wind = np.mean(raw_wind_speed * np.sin(raw_wind_direction))
        wind_speed = (mean_north_wind**2 + mean_east_wind**2)**.5
        wind_direction = np.degrees(np.arctan2(mean_east_wind, mean_north_wind))
        wind_std = np.std(raw_wind_speed)
        # Extra sensor values
        wind_1000sec = np.mean(avgws[(np.nanmin(ds.timestamps)<=avgws_timestamps) & (avgws_timestamps<=np.nanmax(ds.timestamps))])
        wind_dynamic = np.percentile(raw_wind_speed, 95) - wind_1000sec # SKA Dish definition, 3*std - mean
        fi_angle = np.median(fi_angles[(np.nanmin(ds.timestamps)<=fi_timestamps) & (fi_timestamps<=np.nanmax(ds.timestamps))])
        
        # The requested (az, el) coordinates, as they apply at the middle time for a moving target
        rAz, rEl = target.azel(t_ref, antenna=scan_ant) # [rad]
        # Correct for refraction, which becomes the requested value at input of pointing model
        rEl = rc.apply(rEl, temperature, pressure, humidity)

        try: # Fit the beam
            target_x, target_y = ds.target_x[mask,scan_ant_ix], ds.target_y[mask,scan_ant_ix]
            assert (min_len <= 0) or (len(target_x) > min_len), f"This scan has fewer than the minimum number of data points required to fit: {len(target_x)} < {min_len}." 
            hv = ds.vis[mask]
            hv_mag = np.abs(hv)
            hv_c_angle = np.median(np.unwrap(np.angle(hv), axis=0), axis=0) # Flatten phases relative to median over time
            hv /= np.mean(hv_mag, axis=0)*np.exp(1j*hv_c_angle) # Normalise for H-V gains & bandpass
            if track_ant and phased_up: # Phase coherent average
                height = np.abs(np.sum(hv, axis=(1,2))) # TOTAL complex power integrated  over frequency
            else:
                height = np.sum(np.abs(hv), axis=(1,2)) # TOTAL power integrated over frequency
            
            if (kind in ['circle','cardioid','epicycle']): # These don't have enough sampling to fit background reliably
                bkg = np.array([0])
            else:
                bkg = fit_background(target_x, target_y, height, along_edges=True)
            height -= bkg
            
            if debug: # Prepare figure for debugging
                fig, axs = plt.subplots(1,3, figsize=(14,3))
                mu_sigma = np.mean(hv_mag, axis=0)/np.std(hv_mag, axis=0) # Identify non-gaussian processes e.g. RFI
                axs[0].plot(ds.channels, mu_sigma[:,0], '.', ds.channels, mu_sigma[:,1], '.') # H & V separately
                axs[0].set_xlabel("Frequency [channels]"); axs[0].set_ylabel("$\mu/\sigma$")
                axs[1].plot(target_x, '.', target_y, '.') # Also plots track antenna if present
                axs[1].set_xlabel("Time [#]"); axs[1].set_ylabel("target y [deg]")
                axs[1] = axs[1].twinx(); axs[1].plot(height, 'k.-')
                # 2D plot, with height scaled to [0, 1]
                delta_n = height - np.nanmin(height)
                delta_n /= np.nanpercentile(delta_n, 99.9, axis=0)
                if (kind in ['circle','cardioid','epicycle']):
                    axs[2].scatter(target_x, target_y, s=1+100*delta_n, c=delta_n, alpha=0.5)
                else:
                    axs[2].tricontourf(target_x.squeeze(), target_y.squeeze(), delta_n.squeeze(), 20)
                axs[2].set_xlabel("target x [deg]")
            constr = {}
            if (kind in ['circle', 'raster']): # Extra constraints - not necessary for cardioid & epicycle
                constr = dict(constrain_hpbw=1.22*(_c_/np.mean(ds.freqs))/scan_ant.diameter * R2D)
            # Fitted beam center is in (x, y) coordinates, in projection centered on target
            xoff, yoff, valid, hpwx, hpwy, rot, ampl, resid = fit_gaussianoffset(target_x, target_y, height, powerbeam=(track_ant is None),
                                                                                 debug=axs[1:] if debug else None, **constr)
            if debug:
                fig.suptitle(f"Scan #{cs_no}: {cs_label}, on {target.name} @ {rEl*R2D:.0f}degEl [Fit: {valid}]")
        except Exception as e:
            print(f"INFO: Skipping scan #{cs_no}: {cs_label}, on {target.name} - {e}")
            continue
        
        # Convert this offset back to spherical (az, el) coordinates
        with katpoint.projection.out_of_range_context('nan'):
            aAz, aEl = target.plane_to_sphere(xoff/R2D, yoff/R2D, t_ref, antenna=scan_ant, coord_system='azel') # [rad]
        # Now correct the measured (az, el) for refraction and then apply the old pointing model
        aEl = rc.apply(aEl, temperature, pressure, humidity)
        # Get a "raw" measured (az, el) at the output of the pointing model
        mAz, mEl = scan_ant.pointing_model.apply(aAz, aEl)
        
        # The difference between requested & measured, as a small angle around 0 degrees
        dAz, dEl = (mAz - rAz)*R2D, (mEl - rEl)*R2D
        dAz, dEl = wrap_angle(dAz), wrap_angle(dEl)
        
        if debug or verbose:
            print(f"Scan #{cs_no}: {cs_label}, on {target.name}")
            print("    SNR %.1f: fit=%s\t xy offsets %g, %g, AzEl errors %g, %g [arcsec]"%(ampl/resid, valid, xoff*3600, yoff*3600, dAz*3600, dEl*3600))
        
        if strict and not valid:
            dAz, dEl = np.nan, np.nan
        fitted.append((t_ref, target.name, rAz*R2D, rEl*R2D, dAz, dEl, hpwx/R2D, hpwy/R2D, ampl, resid, np.mean(bkg)))
        
        enviro.append([temperature, pressure, humidity, wind_std, wind_speed, wind_direction] + list(sun_azel) + [wind_dynamic, fi_angle])
    
    if (output_filepattern):
        save_apss_file(output_filepattern, ds, [a for a in ds.ants if (a.name==ant)][0], fitted, enviro)
    
    if debug or verbose:
        print("Std [arcsec]", np.nanstd([o[4] for o in fitted])*3600, np.nanstd([o[5] for o in fitted])*3600)
    
    return (fitted, enviro)


def _demo_reduce_pointing_scans_(freq=11e9, ampl=1, SEFD=200, kind="cardioid", cycles=7, debug=False):
    """ Demonstrate a simulated "single dish" dataset similar to what's expected with DVS Ku-band """
    np.random.seed(1)
    
    class TestDataset(object):
        def __init__(self, f_c, ant_names, targets_scanned, n_cycles, Dant=15):
            """ @param f_c: center frequency, in Hz
                @param ant_names: just the names - they will get random coordinates in the Karoo.
                @param targets_scanned: list of target names, one entry for each "scan", in order.
                @param n_cycles: the number of cycles at each "scan"
                @param Dant: diameter of the individual dish [meter]
            """
            self.name = '1234567890_sdp_l0'
            self.spectral_windows = [typing.NamedTuple('SpectralWindow', centre_freq=float)(f_c)]
            self.channels = np.arange(16) # Not important, this keeps it easy
            self.freqs = f_c + np.linspace(-10e6,10e6,len(self.channels)) 
            self.ants = [katpoint.Antenna(a, -np.pi/6+0.01*np.random.rand(), np.pi/9+0.1*np.random.rand(), 1050, Dant) for a in ant_names] # Roughly in the Karoo
            self.catalogue = katpoint.Catalogue(antenna=self.ants[0], add_specials=True)
            for t in set(targets_scanned):
                if (t not in self.catalogue):
                    self.catalogue.add("%s, radec, %.3f, %.3f" % (t, 24*np.random.rand(), -90*np.random.rand()))
            self.__targets_scanned__ = targets_scanned
            self.__n_cycles__ = n_cycles
            self.__hpbw__ = 1.22*(_c_/f_c)/Dant * R2D # [deg]
            self.__set_testopts__() # Defaults

        def select(self, *a, **k):
            self.timestamps = np.arange(0,1000*self.__n_cycles__) # This must be an outer envelope of timestamps produced in compscans()
        
        def __set_testopts__(self, kind="cardioid", scanrad='hpbw', ampl=1, SEFD=300, BW=1e6, ox=0,oy=0):
            """ @param kind: "circle"|"cardioid"|"epicycle"
                @param scanrad: 'hpbw' or radius to scan at, in [degrees]
                @param ampl, SEFD: Jy per pol
                @param ox, oy: "error" offsets in degrees
            """
            self.__testopts__ = [kind, scanrad, ampl, SEFD, BW, ox, oy]
                    
        def compscans(self):
            """ Generate the test data for a single cycle at a time (yields) """
            for i,tgt in enumerate(self.__targets_scanned__):
                self.compscan_indices = [i]
                target = [t for t in self.catalogue.targets if (tgt in t.name)][0]
                for s in range(self.__n_cycles__):
                    kind, scanrad, ampl, SEFD, BW, ox,oy = self.__testopts__
                    scanrad = self.__hpbw__ if (scanrad == 'hpbw') else scanrad 
                    t, x, y, m = _generate_test_data_(kind, powerbeam=len(self.ants)==1, ampl=ampl/2,SEFD=SEFD/2, Nsamples_per_dump=BW*1,
                                                      hpbw=self.__hpbw__, scanrad=scanrad, ox=ox, oy=oy)
                    m = np.transpose(np.stack([m]*len(self.channels),axis=0)) # time,freq
                    self.target_x, self.target_y, self.vis = np.stack([x],axis=1), np.stack([y],axis=1), np.stack([m/2,m/2],axis=2)
                    self.flags = np.full(self.vis.shape, False)
                    self.timestamps = t + (t[-1]-t[0])*(i*self.__n_cycles__ + s)
                    self.temperature = 15 + np.random.rand(len(self.timestamps))
                    self.pressure = 900 + 5*np.random.rand(len(self.timestamps))
                    self.humidity = 20 + 10*np.random.rand(len(self.timestamps))
                    self.wind_speed = 1 + np.random.rand(len(self.timestamps))
                    self.wind_direction = 123 + 10*np.random.rand(len(self.timestamps))
                    yield (s, "compscan", target)

    try:
        # Test fixtures
        _gsv_ = katselib.getsensorvalues
        katselib.getsensorvalues = lambda sensor, timestamps, *a, **k: (timestamps, np.random.randn(len(timestamps)))
        
        print("Demo using " + kind)
        ds = TestDataset(freq, ["s0000"], ["Jupiter"], n_cycles=cycles)
        hpbw = ds.__hpbw__ # deg
        for ox,oy in [(0,0),(hpbw/3,0),(0,hpbw/3)]:
            print("Simulated xy offsets", ox*3600, oy*3600, "[arcsec]")
            ds.__set_testopts__(kind=kind, scanrad='hpbw', ampl=ampl, SEFD=SEFD, BW=10e6, ox=ox,oy=oy)
            fitted, enviro = reduce_pointing_scans(ds, ds.ants[0].name, track_ant=None, strict=True, verbose=True, debug=debug)
            save_apss_file("./_demo_reduce_pointing_scans_%.f_%.f.csv"%(ox*3600,oy*3600), ds, ds.ants[0], fitted, enviro)
    
    finally: # Undo fixtures
        katselib.getsensorvalues = _gsv_


def save_apss_file(output_filename, ds, ant, fitted, enviro):
    """ Creates a CSV file like what's generated by `analyse_point_source_scans.py`
        
        @param output_filename: explicit filename or a pattern with '%s' - first for dataset name second for antenna ID.
        @param ds: the dataset with selection applied
        @param ant: the katpoint.Antenna (with pointing model in use at the time)
        @param fitted, enviro: as returned by `reduce_pointing_scans()`
    """
    assert output_filename, "output_filename must be specified!"
    ds_id = ds.name.split(" ")[-1]
    try: # Expand the filename if it's a pattern
        output_filename = output_filename % (ds_id, ant.name)
    except TypeError: # No %s's as expected
        pass
    
    fitted = np.asarray(fitted) # Mixed types!
    enviro = np.asarray(enviro)
    
    # Field names as used by 'analyse_point_source_scans.py' in the order out of reduce_pointing_scans()
    fields = ['timestamp', 'target', 'azimuth', 'elevation', 'delta_azimuth', 'delta_elevation',
              'beam_width_HH','beam_width_VV', 'beam_height_I', 'beam_height_I_std', 'baseline_height_I']
    # Note: we map 'resid'-> 'beam_height_I_std', which is not quite the same, but equivalent?
    fields_enviro = ['temperature', 'pressure', 'humidity', 'wind_std', 'wind_speed', 'wind_direction', 'sun_az', 'sun_el', 'wind_dynamic', 'feedindexer_angle']
    string_fields = ['target']
    record = {}
    for c,f in enumerate(fields):
        record[f] = fitted[:,c] if (f in string_fields) else np.asarray(fitted[:,c], float)
    record['dataset'] = [ds_id]*len(fitted)
    record['frequency'] = [np.mean(ds.freqs)]*len(fitted)
    record['timestamp_ut'] = [str(katpoint.Timestamp(_)) for _ in record['timestamp']]
    record['data_unit'] = ['counts']*len(fitted)
    record['beam_height_HH'] = record['beam_height_I']
    record['beam_height_VV'] = record['beam_height_I']
    record['beam_width_I'] = ( record['beam_width_HH']*record['beam_width_VV'] )**.5
    for k in ['baseline_height_I','baseline_height_I_std','baseline_height_HH','baseline_height_VV',
              'refined_I','refined_HH','refined_VV','flux','delta_azimuth_std',
              'delta_elevation_std','beam_width_I_std']:
        record[k] = [0]*len(fitted)
    
    for c,f in enumerate(fields_enviro):
        record[f] = enviro[:,c]
    
    katsepnt.save_apss_data(output_filename, record, ant)


if __name__ == "__main__":
    if True:
        _demo_fit_gaussianoffset_(ampl=5, SEFD=200, cycles=1)
        _demo_fit_gaussianoffset_(ampl=5, SEFD=200, cycles=100)
        _demo_reduce_pointing_scans_(freq=11e9, ampl=5, SEFD=200, kind="cardioid")
        _demo_reduce_pointing_scans_(freq=11e9, ampl=5, SEFD=200, kind="raster", debug=True)
        katsepnt.eval_pointingstability(["./_demo_reduce_pointing_scans_0_0.csv"], blind_pointing=True, update_model=False,
                                        metrics=["timestamp","azimuth","elevation"], meshplot=[], figs=[])
    plt.show()
