""" Functions associated with the various resources under the ../catalogues folder.
    
    @author aph@sarao.ac.za
"""
import numpy as np
import pylab as plt
import matplotlib
import katpoint

D2R = np.pi/180
R2D = 1/D2R


def remove_overlapping(catalogue, eps=0.1, debug=True):
    """ Creates a new catalogue with duplicates in RA,DEC removed (keeps the first one in the list).
    
        @param catalgoue: original katpoint.Catalogue
        @param eps: radius within which two positions are considered identical [arcsec]
        @return: katpoint.Catalogue without duplicates """
    eps = eps/3600*D2R # arcsec -> rad
    
    # Don't need an antenna if for "radec" targets, but in case there are other types
    ant = catalogue.antenna if (catalogue.antenna is not None) else katpoint.Antenna("Z, 0,0,0, 0")
    
    newcat = katpoint.Catalogue(antenna=ant)
    for tgt in catalogue.targets:
        overlap = [(tgt.separation(t, antenna=ant) < eps) for t in newcat.targets]
        if (np.count_nonzero(overlap) == 0):
            newcat.add(tgt)
        elif debug:
            keep = np.take(newcat.targets, np.flatnonzero(overlap))[0]
            print("Removing overlapping target: ", tgt.description)
            print("  ( keeping ", keep.description, " -- %.1f arcsec off" % (tgt.separation(keep, antenna=ant)/D2R*3600), ")")
    return newcat


class ElevationFormatter(matplotlib.projections.polar.ThetaFormatter):
    """ Generate tick labels on a matplotlib polar plot, with angles increasing to 90deg at the centre.
        This is simply the ThetaFormatter, with angles "inverted". """
    def __call__(self, x, pos=None):
        return super().__call__(np.pi/2-x, pos)


def plot_skycat(catalogue, timestamps, t_observe=120, antenna=None, el_limit_deg=20, flip='astro', ax=None):
    """ Plots distribution of catalogue targets on the sky, at timestamps
        @param flip: 'astro' (E<- -> W) or 'geo' (W<- ->E) (default 'astro') """
    if (ax is None):
        ax = plt.figure().add_subplot(111, projection='polar')
    if (flip == 'astro'): ax.set_theta_direction(-1); ax.set_theta_zero_location('W')
    ax.set_xticks(np.arange(0,360,90)*D2R)
    ax.set_xticklabels(['E','N','W','S',])
    ax.set_ylim(0, np.pi/2)
    ax.set_yticks(np.arange(0,90,15)*D2R)
    ax.yaxis.set_major_formatter(ElevationFormatter())
    
    antenna = catalogue.antenna if (antenna is None) else antenna
    for T in timestamps:
        az, el = [], []
        for t,tgt in enumerate(catalogue.targets):
            _az, _el = tgt.azel(timestamp=T+t*t_observe, antenna=antenna)
            if (_el*180/np.pi > el_limit_deg):
                az.append(_az)
                el.append(_el)
        ax.plot(np.pi/2.-np.asarray(az), np.pi/2.-np.asarray(el), 'o', markersize=7)
    

def filter_separation(catalogue, T_observed, antenna=None, separation_deg=1, sunmoon_separation_deg=10):
    """ Removes targets from the supplied catalogue which are within the specified distance from others or either the Sun or Moon.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_observed: UTC timestamp, seconds since epoch [sec].
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str]
        @param separation_deg: eliminate targets closer together than this (default 1) [deg]
        @param sunmoon_separation_deg: omit targets that are closer than this distance from Sun & Moon (default 10) [deg]
        @return: katpoint.Catalogue (a filtered copy of input catalogue)
    """
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    targets = list(catalogue.targets)
    avoid_sol = [katpoint.Target('%s, special'%n) for n in ['Sun','Moon']] if (sunmoon_separation_deg>0) else []
    
    separation_rad = separation_deg*D2R
    sunmoon_separation_rad = sunmoon_separation_deg*D2R
    
    # Remove targets that are too close together (unfortunately also duplicated pairs)
    overlap = np.zeros(len(targets), float)
    for i in range(len(targets)-1):
        t_i = targets[i]
        sep = [(t_i.separation(targets[j], T_observed, antenna) < separation_rad) for j in range(i+1, len(targets))]
        sep = np.r_[np.any(sep), sep] # Flag t_j too, if overlapped
        overlap[i:] += np.asarray(sep, int)
        # Check for t_i overlapping with solar system bodies
        sep = [(t_i.separation(j, T_observed, antenna) < sunmoon_separation_rad) for j in avoid_sol]
        if np.any(sep):
            print("  # %s appears within %g deg from %s"%(t_i, sunmoon_separation_deg, np.compress(sep,avoid_sol)))
            overlap[i] += 1
    if np.any(overlap > 0):
        print("  # Planning drops the following due to being within %g deg away from other targets:\n%s"%(separation_deg, np.compress(overlap>0,targets)))
        targets = list(np.compress(overlap==0, targets))
    
    filtered = katpoint.Catalogue(targets, antenna=antenna)
    return filtered

   
def plan_targets(catalogue, T_start, t_observe, dAdt, antenna=None, el_limit_deg=20, debug=False):
    """ Generates a "nearest-neighbour" sequence of targets to observe, starting at the specified time.
        This does not consider behaviour around the azimuth wrap zone.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_start: UTC timestamp, seconds since epoch [sec].
        @param t_observe: duration of an observation per target [sec]
        @param dAdt: angular rate when slewing [deg/sec]
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str]
        @param el_limit_deg: observation elevation limit (default 15) [deg]
        @return: [list of Targets], expected duration in seconds """
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    todo = list(catalogue.targets)
    done = []
    T = T_start # Absolute time
    # Start with any old target that's visible
    available = catalogue.filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
    next_tgt = available.targets[0] if (len(available.targets) > 0) else None
    while (next_tgt is not None):
        # Observe
        if debug: print(next_tgt)
        next_tgt.antenna = antenna
        done.append(next_tgt)
        todo.pop(todo.index(next_tgt))
        T += t_observe
        # Find next visible target
        available = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
        next_tgt, dGC = available.closest_to(done[-1], T, antenna)
        # # Above as opposed to standard strategy which is 'next in list'
        # next_tgt = available.targets[0] if (len(available.targets) > 0) else None
        # dGC = 0 if next_tgt is None else next_tgt.separation(done[-1], T, antenna)*180/np.pi
        # Slew
        if next_tgt:
            if debug: print("  SLEWING %.1f deg"%dGC)
            T += dGC * dAdt
    T -= T_start # Now duration
    if debug: print("DURATION: %d out of %d targets in %.f min"%(len(done),len(catalogue.targets),T/60.))
    return done, T


def nominal_pos(ska_pad):
    """ Calculates the E,N,U offsets of an antenna, relative to the MeerKAT array reference coordinate.
        
        @param ska_pad: pad number to identify the antenna as SKA* [integer]
        @return: (delta_East, delta_North, delta_Up) [m] """
    lat0, lon0, h0 = -30.7110555, 21.4438888, 1086.6 # As of 2021 the formal "centre" of MeerKAT [https://github.com/ska-sa/katconfig/blob/karoo/static/arrays/karoo.array_mkat.conf]
    if (ska_pad == 0): # SKA-MPI is not in the catalogue
        name = "SKA-MPI"
        llh = (-30.717956, 21.413028, 1093)
        ph = 10.11 # 316-000000-022 rev 1
    else:
        name = "SKA%03d" % ska_pad
        llh = np.loadtxt(__file__+"/../../catalogues/arrays/ska1-mid.txt", comments="#", delimiter=",")[ska_pad-1]
        ph = 9.82 # 316-000000-022 rev 2, Fig 10
    
    lat, lon, hae = llh
    hae += ph
    E, N, U = katpoint.ecef_to_enu( lat0*D2R,lon0*D2R,h0,
                                    *katpoint.lla_to_ecef(lat*D2R,lon*D2R,hae) )
    print(name, "llh = (%.6f, %.6f, %.1f)" % (lat, lon, hae), "ENU = (%.1f %.1f %.1f)" % (E, N, U), "; all including pedestal height!")


def sim_pointingfit(catfn, el_floor_deg=20, duration_min=120, meas_min=5, enabled_params=[1,3,4,5,6,7,8,11], Tstart=None,
                    rms_error_arcsec=30, env_sigma=0.1, verbose=0, flip='astro', **rnd):
    """ Simulate pointing measurements with the specified catalogue, to determine the expected
        model fit residual.
        
        @param catfn: either a filename, or an already loaded catalogue.
        @param duration_min: the time duration to collect a measurement set (default 120) [minutes]
        @param meas_min: the time duration of a single target measurement, including slew time (default 5) [minutes]
        @param enabled_params: pointing model terms to fit (default [1,3,4,5,6,7,8,11])
        @param Tstart: the start time for the measurement set (defaults to "now") [Unix seconds]
        @param env_sigma: the 1sigma fractional uncertainty in all environmental state variables (default 0.1)
        @param rms_error_arcsec: the accuracy with which a centroid can be determined on a single target measurement (default 30) [arcsec]
        @param verbose: 1 to print a summary, 2 to also make a plot (default 0)
        @param flip: 'astro' (E<- -> W) or 'geo' (W<- ->E) (default 'astro')
        @return: (1sigma residuals of fitted params) [arcsec] """
    randomseed = rnd.get('randomseed', 1) # Force a predictable random sequence, so that differences are due to user input variables!
    N_rnd = rnd.get('N_rnd', 20) # The number of Monte Carlo runs to ensure the results are "typical"
    Tstart = katpoint.Timestamp(Tstart)
    rng = np.random.default_rng(randomseed)
    
    if isinstance(catfn, str):
        # An antenna in the vicinity of the MeerKAT core
        ant = katpoint.Antenna('ant, -30:43:17.3, 21:24:38.5, 1038.0, 12.0')
        cat = katpoint.Catalogue(open(catfn), antenna=ant)
    else:
        cat = catfn
        ant = cat.antenna
    
    # Pick an arbitrary "true" pointing model - which is to be recovered
    true_pm = katpoint.PointingModel('-0:07:53, 0, -0:00:53, -0:05:17, -0:01:36, 0:00:21, -0:02:27, -0:01:1, 0, 0:00:0, 0:01:20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')
    enabled_params = np.asarray(enabled_params)
    true_pm_values = np.take(true_pm.values(), enabled_params-1)
    
    rc = katpoint.RefractionCorrection()
    temperature_C = 20; pressure_hPa = 900; humidity_percent = 0.3 # ~50% of the time in the Karoo
    env_delta = env_sigma*rng.normal(size=3)*[temperature_C, pressure_hPa, humidity_percent] # Error assumed for the environmental terms
    
    sigma = (rms_error_arcsec/3600)*D2R / np.sqrt(2.) # Per "tangent plane" dimension
    
    resid_pm_values = []
    for _ in range(N_rnd): # Repeat the simulation to ensure results are "typical"
        pointing_request_az, pointing_request_el, pointing_offset_az, pointing_offset_el = [],[],[],[]
        end_time = Tstart.secs + duration_min*60
        # Observe each of the targets, one at a time, and repeat as necessary, until the time is up
        t = Tstart.secs
        targets = list(cat.targets)
        while (t < end_time):
            if (len(targets) == 0):
                targets = list(cat.targets)
            tgt = targets.pop(0)
            # Generate the "true" (az, el) positions of the sources, before pointing errors & refraction
            src_true_az, src_true_el = tgt.azel(timestamp=t, antenna=ant)
            if (src_true_el < (el_floor_deg)*D2R):
                continue
            
            # Apply true refraction
            src_true_rc_el = rc.apply(src_true_el, temperature_C, pressure_hPa, humidity_percent)
            # Apply true pointing model
            src_pointm_az, src_pointm_el = true_pm.apply(src_true_az, src_true_rc_el)
            
            # The requested "true" (az, el), used to fit the new model, has systematically imperfect knowledge of the environment
            request_az = src_true_az
            request_el = rc.apply(src_true_el, temperature_C+env_delta[0], pressure_hPa+env_delta[1], humidity_percent+env_delta[2])
            
            # Add random noise representing the residual from a fit to the measured centroid
            src_meas_az = src_pointm_az + sigma*rng.normal()/np.clip(np.cos(src_pointm_el), 1e-6, 1.)
            src_meas_el = src_pointm_el + sigma*rng.normal()
            # The desired offset is now the difference between the requested commanded coordinates at the input of the antenna
            # control system and the ideal (i.e. with a PM applied)
            pointing_request_az.append(request_az)
            pointing_request_el.append(request_el)
            pointing_offset_az.append(katpoint.wrap_angle(src_meas_az - request_az))
            pointing_offset_el.append(katpoint.wrap_angle(src_meas_el - request_el))
            
            t = t + meas_min*60
        
        # Fit a new pointing model to the pointing offsets
        pm = katpoint.PointingModel()
        pm.fit(pointing_request_az, pointing_request_el, pointing_offset_az, pointing_offset_el, enabled_params=enabled_params)
        fit_pm_values = np.take(pm.values(), enabled_params-1)
        resid_pm_values.append((fit_pm_values - true_pm_values)**2) # ~ 1sigma variance
    # Get the "typical" 1sigma values
    resid_pm_values = np.mean(resid_pm_values, axis=0)**.5
    
    if (verbose > 0):
        print(catfn, Tstart.local())
        print("TRUE   model:", true_pm.description)
        print("FITTED model:", pm.description)
    
    if (verbose > 1):
        fig = plt.figure()
        fig.suptitle(f"{catfn[-20:]}, from {Tstart.local()}")
        
        ax = fig.add_subplot(121, projection='polar')
        if (flip == 'astro'): ax.set_theta_direction(-1); ax.set_theta_zero_location('W')
        ax.plot(np.pi/2 - np.array(pointing_request_az), np.pi/2 - np.array(pointing_request_el), 'ob')
        ax.set_xticks(np.arange(0, 360, 90)*D2R)
        ax.set_xticklabels(['E', 'N', 'W', 'S'])
        ax.set_ylim(0, np.pi/2)
        ax.set_yticks(np.arange(0, 90, 15)*D2R)
        ax.yaxis.set_major_formatter(ElevationFormatter())
        
        ax = fig.add_subplot(122)
        ax.errorbar(enabled_params, 0*true_pm_values, yerr=resid_pm_values*R2D*3600, fmt='none', ecolor='r', capsize=5)
        plt.ylabel('$1\sigma$ [arcsec]')
        # ax.bar(enabled_params, true_pm_values*R2D*60, 1, align='center')
        # ax.errorbar(enabled_params, true_pm_values*R2D*60, yerr=resid_pm_values*R2D*60, fmt='none', ecolor='r', capsize=5)
        # plt.ylabel('Value [arcmin]')
        ax.set_xticks(enabled_params)
        plt.xlabel('Pointing model terms')
        plt.grid(True)
    
    return resid_pm_values*R2D*3600


if __name__ == "__main__":
    catroot = __file__ + "/../../catalogues/"
    
    if True: # Check spacing of pointing targets
        remove_overlapping(katpoint.Catalogue(open(catroot+"targets_pnt_L.csv")), eps=2*60*60, debug=True)
        remove_overlapping(katpoint.Catalogue(open(catroot+"targets_pnt_S.csv")), eps=60*60, debug=True)
        remove_overlapping(katpoint.Catalogue(open(catroot+"targets_pnt_Ku.csv")), eps=30*60, debug=True)
    
    elif True: # Planning a pointing measurement session
        catfiles = ["targets_pnt_S.csv"]
        for catfile in catfiles:
            axes = np.ravel(plt.figure().subplots(2,2, subplot_kw=dict(projection='polar')))
            Hr = 2 # Duration of a session in hours
            for N in range(4): # Repeat each "plan" 4 times
                T0 = katpoint.Timestamp().secs + (Hr*N)*60*60 # Session at intervals of Hr hours (hope the plan fits in that time)
                print("\n%s"%catfile)
                cat = filter_separation(katpoint.Catalogue(open(catroot+catfile), antenna=katpoint.Antenna('m000, -30.713, 21.444, 1050.0')), T0)
                T = plan_targets(cat, T0, 8*(30+2), 2, debug=True)[1]
                M = int(Hr*60*60/T)+1 # Number of times the plan fits in 4 hours
                plot_skycat(cat, T0+np.arange(M)*T, 8*(30+2), ax=axes[N])
                axes[N].set_title(catfile + ("+ %d hrs"%(Hr*N)))
        plt.show()

    elif True: # Print coordinates of an antenna
        nominal_pos(ska_pad=0)
    
    elif True:
        sim_pointingfit(catroot+"targets_pnt_L.csv", Tstart=1718604536, verbose=2)
        # Confirm that the default value for N_rnd is reasonable - expect both max & mean to be stable
        N_rnd = np.arange(1,200,10)
        R = [sim_pointingfit(catroot+"targets_pnt_L.csv", Tstart=1718604536, randomseed=1, N_rnd=N) for N in N_rnd]
        plt.figure(); plt.title("Convergence check for sim_pointingfit()")
        plt.plot(N_rnd, np.mean(R, axis=1), '.', N_rnd, np.max(R, axis=1), '^')
        plt.xlabel("N_rnd")
        plt.show()
        