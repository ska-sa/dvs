'''
    Analyse "RF Tracking Stability" measured by using circular_pointing.py
    
    @author: aph@sarao.ac.za
'''
import numpy as np
import scipy.optimize as sop
import pylab as plt
import katpoint
import typing
from analysis.katsepnt import save_apss_data


_c_ = 299792458
R2D = 180/np.pi


def fit_gaussianoffset(x, y, height, powerbeam=True, constrained=True, constrain_ampl=None, constrain_width=None, debug=None):
    """ Fit a Gaussian to the magnitude measured along a trajectory (x,y).
        NB: Circular scan provides very little information to fit all of the Gaussian's free variables reliably, so use constrain_ampl.
        
        @param x, y: position coordinates around the circular scan trajectory
        @param height: height above the baseline, measured along the trajectory
        @param powerbeam: True if scanned in total power, False if scanned in complex amplitude
        @param constrain_ampl, constrain_width: > 0 to constrain the fitted parameters to within 10% of this
        @return: (xoffset, yoffset, valid, xfwhm, yfwhm, ampl, resid) - positions on tangent plane centred on target in same units as `x` & `y` """
    h_sigma = np.mean([np.std(height[i*10:(i+1)*10]) for i in range(len(height)//10)]) # Estimate of radiometer noise
    
    # Starting estimates
    # These measurements (power & voltage) are typically done by scanning around half _power_ contour
    scanext = 2*np.median(x**2 + y**2)**.5
    width0 = constrain_width if (constrain_width) else scanext
    ampl0 = constrain_ampl if (constrain_ampl) else (np.max(height) - np.min(height))
    p0 = [ampl0,0,0,width0,width0, np.min(height)]
    
    if powerbeam:
        model = lambda ampl,xoffset,yoffset,wx,wy, h0=0: h0 + ampl*np.exp(-np.log(2)*((x+xoffset)**2/wx**2+(y+yoffset)**2/wy**2))**2
    else: # Scan referenced to a tracking antenna scans the voltage beam, convert it to power
        model = lambda ampl,xoffset,yoffset,wx,wy, h0=0: h0 + ampl**.5*np.exp(-np.log(2)*((x+xoffset)**2/wx**2+(y+yoffset)**2/wy**2))
    
    # Fit model to data
    if not constrained: # Unconstrained fit is VERY BRITTLE, results also very sensitive to method (BFGS seems best, but Powell, Nelder-Mead, LM ...)
        p = sop.minimize(lambda p: np.nansum((height-model(*p))**2), p0, method='BFGS', options=dict(disp=False))
    else: # Constrained fits should be robust, and not very sensitive to bounds
        bounds = [(h_sigma/2,np.inf)] + ( [(-width0,width0)]*2 ) + ( [(0.5*width0,2*width0)]*2 ) + [(0,np.min(height))]
        if (constrain_ampl): # Explicit constraint for e.g. circular scan
            bounds[0] = (0.9*ampl0, 1.1*ampl0)
        if (constrain_width):
            bounds[3] = bounds[4] = (0.9*width0,1.1*width0)
        p = sop.least_squares(lambda p: height-model(*p), p0, bounds=np.transpose(bounds), verbose=0)
    # It seems that the fit fails to converge if we fit for h0 too!??
    ampl, xoffset, yoffset, fwhmx, fwhmy, h0 = p.x
    model_height = model(*p.x)
    
    # 'p.success' status isn't quite reliable (e.g. multiple minima)
    # Also check deduced signal-to-noise, and residuals must be "in the noise"
    resid = np.std(height - model_height)
    valid = p.success and (ampl/h_sigma > 6) and (resid < h_sigma)
    
    if debug is not None: # 'debug' is expected to be a plot axis
        debug.plot(model_height, 'k-', alpha=0.5, label="%.1f + %.1f exp(Q(%.2f, %.2f))"%(h0, ampl, fwhmx/scanext, fwhmy/scanext))
        debug.legend()
    return (xoffset, yoffset, valid, fwhmx, fwhmy, ampl, resid)


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
        ox,oy = (hpbw/6, 0)
        for powerbeam in [True, False]: # Power & voltage beams
            axs = plt.subplots(3, 3)[1]
            for i,kind in enumerate(['circle','cardioid','epicycles']):
                timestamps,target_x,target_y,m = _generate_test_data_(kind, powerbeam=powerbeam, hpbw=hpbw, ampl=ampl, SEFD=SEFD,
                                                                      Nsamples_per_dump=Nsamples_per_dump, scanrad='hpbw', ox=ox, oy=oy)
                axs[0][i].set_title(kind)
                axs[0][i].plot(target_x, target_y, '.')
                axs[1][i].plot(timestamps, target_x, '.', timestamps, target_y)
                axs[2][i].plot(m, '.')
                xoff, yoff, valid, hpwx, hpwy, a0, resid = fit_gaussianoffset(target_x, target_y, m, powerbeam=powerbeam, debug=axs[-1][i])
                # print("x: %g -> %g"%(ox, xoff), "y: %g -> %g"%(oy, yoff), "valid: %s"%valid, "HPBW %.3f -> %.3f, %.3f"%(hpbw, hpwx, hpwy), "ampl %g -> %g"%(ampl,a0))
    
    else: # Statistical
        axes = plt.subplots(2,3)[1]
        for powerbeam,axs in zip([True, False], axes): # Power & voltage beams
            axs[0].set_ylabel(" SD" if powerbeam else " INTF")
            for kind in ['circle','cardioid','epicycles']:
                # Same pointing offsets for each 'kind'
                np.random.seed(1)
                offsets = hpbw * np.c_[np.random.rand(cycles) - 0.5, np.random.rand(cycles) - 0.5]
                fits = []
                for ox,oy in offsets:
                    timestamps,target_x,target_y,m = _generate_test_data_(kind, powerbeam=powerbeam, hpbw=hpbw, ampl=ampl, SEFD=SEFD,
                                                                          Nsamples_per_dump=Nsamples_per_dump, scanrad='hpbw', ox=ox, oy=oy)
                    xoff, yoff, valid, hpwx, hpwy, a0, resid = fit_gaussianoffset(target_x, target_y, m, powerbeam=powerbeam, debug=None)
                    fits.append([ox-xoff, oy-yoff, hpwx/hpbw, hpwy/hpbw, a0/ampl])
                fits = np.asarray(fits)
                axs[0].hist(fits[:,0], bins=20, range=(-hpbw,hpbw), alpha=0.5, label=kind+" X")
                axs[0].hist(fits[:,1], bins=20, range=(-hpbw,hpbw), alpha=0.5, label=kind+" Y")
                axs[1].hist(fits[:,2], bins=20, range=(0,2), alpha=0.5, label=kind+" X")
                axs[1].hist(fits[:,3], bins=20, range=(0,2), alpha=0.5, label=kind+" Y")
                axs[2].hist(fits[:,4], bins=20, range=(0,2), alpha=0.5, label=kind)
            for ax,unit in zip(axs,["$\Delta$pointing", "fit_hpw/hpbw", "fit_ampl/ampl"]):
                ax.legend(); ax.set_xlabel(unit)
            


def reduce_circular_pointing(ds, ant, chanmapping, track_ant=None, strict=True, verbose=True, debug=False, kind=None):
    """ Generates pointing offsets for a dataset created with circular_pointing.py
        @param ds: the dataset (selection is reset!)
        @param ant: the identifier of the scanning antenna in the dataset
        @param chanmapping: channel indices to use or a function like `lambda target_name,fGHz: channel_indices`, or None to use all channels
        @param track_ant: the identifier of the tracking antenna in the dataset, if not single dish mode (default None)
        @param strict: True to set invalid fits to nan (default True)
        @return: ( [(timestamp [sec], target ID [string], Az, El, dAz, dEl, hpw_x, hpw_y [deg], ampl, resid [power]), ...(for each cycle)]
                   [(temperature, pressure, humidity, wind_speed, wind_dir), ...(for each cycle)] )
    """
    if (not callable(chanmapping)) and (chanmapping is not None):
        _chans_ = chanmapping
        chanmapping = lambda *a: _chans_
    
    fitted = [] # (timestamp, target, Az, El, dAz, dEl, hpw_x, hpw_y, ampl, resid)
    enviro = [] # (temperature, pressure, humidity, wind_speed, wind_dir)
    
    ds.select(scans="track", compscans="~slew")
    if (track_ant):
        ds.select(corrprods="cross", pol=["HH","VV"], ants=[ant,track_ant])
    else:
        ds.select(pol=["HH","VV"], ants=[ant])
    fGHz = np.round(ds.spectral_windows[0].centre_freq/1e9, 4)
    
    ant_ix = [a.name for a in ds.ants].index(ant)
    ant = ds.ants[ant_ix]
    for scan in ds.scans():
        # Fit offsets to circular scan total power
        target = ds.catalogue.targets[ds.target_indices[0]]
        if chanmapping:
            ds.select(channels=chanmapping(target.name, fGHz))
        mask = slice(None) # TODO: mask any extra data, e.g. too high acceleration, or data lost?
        hv = np.abs(ds.vis[mask])
        hv /= np.median(hv,axis=0) # Normalise for H-V gains & bandpass
        height = np.sum(hv, axis=(1,2)) # TOTAL power integrated over frequency
        t_ref = np.mean(ds.timestamps[mask])
        rAz, rEl = target.azel(t_ref, antenna=ant)
        if debug: # Prepare figure for debugging
            axs = plt.subplots(1,3, figsize=(14,3))[1]
            axs[0].plot(ds.channels, np.mean(hv[...,0], axis=0), '.', ds.channels, np.mean(hv[...,1], axis=0), '.') # H & V separately
            axs[1].plot(ds.timestamps[mask], ds.target_x[mask,...], '.', ds.timestamps[mask], ds.target_y[mask,...]) # Also plots track antenna if present
            axs[2].plot(height, '.')
        constr = {}
        if (kind == "circle"): # Extra constraints only for circle patterns: either amplitude or hpbw
            constr["constrain_width"] = 1.22*(_c_/np.mean(ds.freqs))/ant.diameter * R2D
        xoff, yoff, valid, hpwx, hpwy, ampl, resid = fit_gaussianoffset(ds.target_x[mask,ant_ix], ds.target_y[mask,ant_ix], height[mask],
                                                                        powerbeam=(track_ant is None), debug=axs[2] if debug else None, **constr)
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
        fitted.append((t_ref, target.name, rAz*R2D, rEl*R2D, dAz, dEl, hpwx/R2D, hpwy/R2D, ampl, resid))
        
        # Do a 2-D vector average of wind speed + direction
        raw_wind_speed = ds.wind_speed
        raw_wind_direction = ds.wind_direction/R2D
        mean_north_wind = np.mean(raw_wind_speed * np.cos(raw_wind_direction))
        mean_east_wind = np.mean(raw_wind_speed * np.sin(raw_wind_direction))
        wind_speed = (mean_north_wind**2 + mean_east_wind**2)**.5
        wind_direction = np.degrees(np.arctan2(mean_east_wind, mean_north_wind))
        enviro.append((np.mean(ds.temperature), np.mean(ds.pressure), np.mean(ds.humidity), wind_speed, wind_direction))
    
    if debug or verbose:
        print("Std [arcsec]", np.nanstd([o[4] for o in fitted])*3600, np.nanstd([o[5] for o in fitted])*3600)
    
    return (fitted, enviro)


def _demo_reduce_circular_pointing_(freq=11e9, ampl=1, SEFD=200, kind="cardioid", cycles=7):
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
            self.spectral_windows = [typing.NamedTuple('SpectralWindow', centre_freq=float)(f_c)]
            self.channels = np.arange(16) # Not important, this keeps it easy
            self.freqs = f_c + np.linspace(-10e6,10e6,len(self.channels)) 
            self.ants = [katpoint.Antenna(a, -np.pi/3+0.1*np.random.rand(), np.pi/3.3+0.1*np.random.rand(), 1050, Dant) for a in ant_names] # Roughly in the Karoo
            self.catalogue = katpoint.Catalogue(antenna=self.ants[0], add_specials=True)
            for t in set(targets_scanned):
                if (t not in self.catalogue):
                    self.catalogue.add("%s, radec, %.3f, %.3f" % (t, 24*np.random.rand(), -90*np.random.rand()))
            self.__targets_scanned__ = targets_scanned
            self.__n_cycles__ = n_cycles
            self.__hpbw__ = 1.22*(_c_/f_c)/Dant * R2D # [deg]
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
                    self.temperature = 15 + np.random.rand(len(self.timestamps))
                    self.pressure = 900 + 5*np.random.rand(len(self.timestamps))
                    self.humidity = 20 + 10*np.random.rand(len(self.timestamps))
                    self.wind_speed = 1 + np.random.rand(len(self.timestamps))
                    self.wind_direction = 123 + 10*np.random.rand(len(self.timestamps))
                    yield

    ds = TestDataset(freq, ["s0000"], ["Jupiter"], n_cycles=cycles)
    hpbw = ds.__hpbw__ # deg
    for ox,oy in [(0,0),(hpbw/3,0),(0,hpbw/3)]:
        print("Simulated xy offsets", ox*3600, oy*3600, "[arcsec]")
        ds.__set_testopts__(kind=kind, scanrad='hpbw', ampl=ampl, SEFD=SEFD, BW=10e6, ox=ox,oy=oy)
        reduce_circular_pointing(ds, ds.ants[0].name, None, track_ant=None, strict=True, verbose=True, debug=False)


def analyse_circ_scans(ds, ants, chanmapping, output_filepattern=None, debug=False, verbose=True, **kwargs):
    """ Generates pointing offsets for a dataset created with circular_pointing.py
    
        @param ds: the katdal dataset
        @param ants: the identifiers of the scanning antennas in the dataset
        @param chanmapping: channel indices to use or a function like `lambda target_name,fGHz: channel_indices`, or None to use all channels
        @param output_filepattern: filename pattern (with %s for antenna ID) for CSV file to store results to (default None)
        @param kwargs: extra arguments for reduce_circular_pointing()
        @return: { antID:[(timestamp [sec], target ID [string], Az, El, dAz, dEl, hpw_x, hpw_y [deg], ampl, resid [power]), ...(for each cycle)] }
    """
    results = {}
    for ant in ants:
        fitted, enviro = reduce_circular_pointing(ds, ant, chanmapping, debug=debug, verbose=verbose, **kwargs)
        results[ant] = fitted
        if (output_filepattern):
            save_apss_file(output_filepattern%ant, ds, [a for a in ds.ants if (a.name==ant)][0], fitted, enviro)
        
    if verbose: # Plot the offsets
        symbols = ['.', '+', '^', 'D', 'S', 'O', 'v', '*']
        axs = plt.subplots(2,1, figsize=(14,5))[1]
        for (ant,fitted),fmt in zip(results.items(),symbols):
            tAE = np.array([[m[0],m[4],m[5]] for m in fitted])
            axs[0].plot(tAE[:,0], (tAE[:,1]-np.nanmedian(tAE[:,1]))*3600, fmt, label=ant)
            axs[1].plot(tAE[:,0], (tAE[:,2]-np.nanmedian(tAE[:,2]))*3600, fmt, label=ant)
        axs[0].legend(); axs[0].set_ylabel("dAz"); axs[1].set_ylabel("dEl");
    
    return results


def save_apss_file(output_filename, ds, ant, fitted, enviro):
    """ Creates a CSV file like what's generated by analyse_point_source_scans.py
        @param ds: the dataset with selection applied
        @param ant: the katpoint.Antenna (with pointing model in use at the time)
        @param fitted, enviro: as returned by `reduce_circular_pointing()`
    """
    fields = ['target', 'timestamp', 'azimuth', 'elevation', 'delta_azimuth', 'delta_elevation',
              'beam_height_I','beam_height_I_std','beam_width_I','beam_width_HH','beam_width_VV']
    record = {k:[] for k in fields}
    for rec in fitted: # (timestamp [sec], target ID [string], Az, El, dAz, dEl, hpw_x, hpw_y [deg], ampl, resid [power]
        record['timestamp'].append(rec[0])
        record['target'].append(rec[1])
        record['azimuth'].append(rec[2])
        record['elevation'].append(rec[3])
        record['delta_azimuth'].append(rec[4])
        record['delta_elevation'].append(rec[5])
        record['beam_width_I'].append((rec[6]*rec[7])**.5)
        record['beam_width_HH'].append(rec[6])
        record['beam_width_VV'].append(rec[7])
        record['beam_height_I'].append(rec[8])
        record['beam_height_I_std'].append(rec[9]) # Not quite the same, but relevant
    record['dataset'] = [ds.name.split(" ")[-1]]*len(fitted)
    record['frequency'] = [np.mean(ds.freqs)]*len(fitted)
    record['timestamp_ut'] = [str(katpoint.Timestamp(_)) for _ in record['timestamp']]
    record['data_unit'] = ['counts']*len(fitted)
    for k in ['baseline_height_I','baseline_height_I_std','baseline_height_HH','baseline_height_VV','refined_I','refined_HH','refined_VV','flux']:
        record[k] = [0]*len(fitted)
    record['beam_height_HH'] = record['beam_height_I']
    record['beam_height_VV'] = record['beam_height_I']
    for k in ['delta_azimuth_std','delta_elevation_std','beam_width_I_std',]:
        record[k] = [0]*len(fitted)
    
    # Sun angle relative to the antenna (not critical which antenna)
    sun = katpoint.Target('Sun, special')
    sun_az, sun_el = katpoint.rad2deg(np.array(sun.azel(record['timestamp'], antenna=ds.ants[0])))
    record['sun_az'] = sun_az
    record['sun_el'] = sun_el
    
    enviro = np.asarray(enviro)
    record['temperature'] = enviro[:,0]
    record['pressure'] = enviro[:,1]
    record['humidity'] = enviro[:,2]
    record['wind_speed'] = enviro[:,3]
    record['wind_direction'] = enviro[:,4]
    
    save_apss_data(output_filename, record, ant)


if __name__ == "__main__":
    _demo_fit_gaussianoffset_(ampl=1, SEFD=200, cycles=1)
    _demo_fit_gaussianoffset_(ampl=1, SEFD=200, cycles=100)
    _demo_reduce_circular_pointing_(freq=11e9, ampl=1, SEFD=200, kind="cardioid")
    plt.show()
