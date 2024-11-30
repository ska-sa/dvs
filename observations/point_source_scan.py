#!/usr/bin/python
# Raster scans on targets from a specified catalogue.
# Typically used for SingleDish pointing model fits and gain curve calculation.

import time
import numpy as np

from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger
import katpoint

def collect_targets(cam, args, opts):
    """ Similar to katcorelib.collect_targets(), but this can take TLE files!
        @return: katpoint.Catalogue
    """
    try:
        cat = katpoint.Catalogue(antenna=cam.sources.antenna)
        try: # Maybe a standard catalogue file
            cat.add(open(opts.catalogue, 'rt'))
        except ValueError: # Possibly a TLE formatted file
            try:
                cat.add_tle(open(opts.catalogue, 'rt'))
            except:
                raise ValueError("%s is not a valid target catalogue file!" % opts.catalogue)
        if (len(args) == 0):
            return cat
        else:
            tgt = cat[args[0]]
            if (tgt is None):
                raise ValueError("No target retrieved from argument list!")
            return katpoint.Catalogue(tgt, antenna=cam.sources.antenna)
    except Exception as e:
        user_logger.warning("Didn't find a specific catalogue and/or target to load, continuing with default catalogue.\n\t[ %s ]" % e)
        return cam.sources


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

    separation_rad = separation_deg*np.pi/180.
    sunmoon_separation_rad = sunmoon_separation_deg*np.pi/180.

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
            user_logger.info("%s appears within %g deg from %s"%(t_i, sunmoon_separation_deg, np.compress(sep,avoid_sol)))
            overlap[i] += 1
    if np.any(overlap > 0):
        user_logger.info("Planning drops the following due to being within %g deg away from other targets:\n%s"%(separation_deg, np.compress(overlap>0,targets)))
        targets = list(np.compress(overlap==0, targets))

    filtered = katpoint.Catalogue(targets, antenna=antenna)
    return filtered


def plan_targets(catalogue, T_start, t_observe, dAdt=1.8, antenna=None, el_limit_deg=20):
    """ Generates a "nearest-neighbour" sequence of targets to observe, starting at the specified time.
        This does not consider behaviour around the azimuth wrap zone.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_start: UTC timestamp, seconds since epoch [sec].
        @param t_observe: duration of an observation per target [sec]
        @param dAdt: angular rate when slewing (default 1.8) [deg/sec]
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str or antenna proxy]
        @param el_limit_deg: observation elevation limit (default 20) [deg]
        @return: [list of Targets], expected duration in seconds
    """
    # If it's an "antenna proxy, use current coordinates as starting point
    try:
        az0, el0 = antenna.sensor.pos_actual_scan_azim.get_value(), antenna.sensor.pos_actual_scan_elev.get_value()
        antenna = antenna.sensor.observer.value
    except: # No "live" coordinates so start from zenith
        az0, el0 = 0, 90
    start_pos = katpoint.construct_azel_target(az0*np.pi/180., el0*np.pi/180.)
    
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    
    todo = list(catalogue.targets)
    done = []
    T = T_start # Absolute time
    available = catalogue.filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
    next_tgt = available.closest_to(start_pos, T, antenna)[0] if (len(available.targets) > 0) else None
    while (next_tgt is not None):
        # Observe
        next_tgt.antenna = antenna
        done.append(next_tgt)
        todo.pop(todo.index(next_tgt))
        T += t_observe
        # Find next visible target
        available = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
        next_tgt, dGC = available.closest_to(done[-1], T, antenna)
        # Slew to next
        if next_tgt:
            T += dGC * dAdt
    return done, (T-T_start)



# Script options
parser = standard_script_options(
    usage="%prog [options] [<'target/catalogue'> ...]",
    description="Perform raster scans across sources loaded from a TLE catalogue.")
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Point source scan', dump_rate=None)

# Observation-specific options
parser.add_option('--catalogue', type='string',
                  help="Name of file containing catalogue of targets to use, instead of default system catalogue.")
parser.add_option('--filter-tags', default='',
                  help="Comma-separated list of tags to use from the catalogue, if any e.g. 'radec,gain' (default %default)")
parser.add_option('--min-separation', type="float", default=1.0,
                  help="Minimum separation angle to enforce between any two targets, in degrees (default=%default)")
parser.add_option('--sunmoon-separation', type="float", default=10,
                  help="Minimum separation angle to enforce between targets and the sun & moon, in degrees (default=%default)")
parser.add_option('-m', '--min-time', type="float", default=-1,
                  help="Minimum duration to run experiment, in seconds (default=one loop through sources)")
parser.add_option('-e', '--scan-in-elevation', action="store_true", default=False,
                  help="Scan in elevation rather than in azimuth (default=%default)")
# Raster scan styles need to cover null-to-null at centre frequency (+-1.3*HPBW), resolution ~HPBW/2
styles = { 
    # Standard for MeerKAT
    'uhf': dict(num_scans=9, scan_duration=60, scan_extent=5.5, scan_spacing=5.5/8),
    'l': dict(num_scans=9, scan_duration=40, scan_extent=3.5, scan_spacing=3.5/8),
    's': dict(num_scans=9, scan_duration=30, scan_extent=2.0, scan_spacing=2.0/8),
    'ku': dict(num_scans=11, scan_duration=20, scan_extent=0.5, scan_spacing=0.5/10),
    'ku-slow': dict(num_scans=11, scan_duration=40, scan_extent=0.5, scan_spacing=0.5/10),
    # Ku-band initial pointing, either MeerKAT or SKA Dish
    'ku-wide': dict(num_scans=17, scan_duration=35, scan_extent=1.0, scan_spacing=0.06), # Az 1.0 x El 1.0deg, 2sec per spacing
    'ku-search': dict(num_scans=17, scan_duration=35, scan_extent=3.0, scan_spacing=0.09), # Az 3.0 x El 1.5deg, 1sec per spacing
    # Standard for SKA Dish
    'skab1': dict(num_scans=9, scan_duration=60, scan_extent=6.6, scan_spacing=6.6/8),
    'skab2': dict(num_scans=9, scan_duration=30, scan_extent=3.0, scan_spacing=3.0/8),
    'skab3': dict(num_scans=9, scan_duration=16, scan_extent=1.8, scan_spacing=1.8/8),
    'skab5a': dict(num_scans=9, scan_duration=16, scan_extent=0.6, scan_spacing=0.6/8),
}
parser.add_option('--style', type='choice', choices=styles.keys(),
                  help="Raster scan style determining number of scans, scan duration, scan extent "
                       "and scan spacing. The available styles are: %s" % (styles,))


# Parse the command line
opts, args = parser.parse_args()

with verify_and_connect(opts) as kat:
    
    # Load pointing calibrator catalogues and command line targets
    pointing_sources = collect_targets(kat, args, opts)
    if opts.filter_tags:
        pointing_sources = pointing_sources.filter(tags=opts.filter_tags)

    # Remove sources that crowd too closely
    pointing_sources = filter_separation(pointing_sources, time.time(), kat.sources.antenna,
                                         separation_deg=opts.min_separation, sunmoon_separation_deg=opts.sunmoon_separation)
    raster_params = styles[opts.style]

    # Quit early if there are no sources to observe
    if (len(pointing_sources) == 0):
        user_logger.warning("Empty point source catalogue or all targets are skipped")
    elif len(pointing_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            import _hacks_; _hacks_.apply(kat)
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one from the nearest neighbour plan
                raster_duration = raster_params["num_scans"] * (raster_params["scan_duration"]+5) # Incl. buffer between scans. TODO: Ignoring ND
                for target in plan_targets(pointing_sources, start_time, t_observe=raster_duration,
                                           antenna=kat.ants[0], el_limit_deg=opts.horizon+5.0)[0]:
                    session.label('raster')
                    az, el = target.azel()
                    user_logger.info("Scanning target %r with current azel (%s, %s)" % (target.description, az, el))
                    if session.raster_scan(target, scan_in_azimuth=not opts.scan_in_elevation,
                                           projection=opts.projection, **raster_params):
                        targets_observed.append(target.name)
                    # The default is to do only one iteration through source list; or if the time is up, stop immediately
                    keep_going = (opts.min_time <= 0) or (time.time() - start_time < opts.min_time)
                    if not keep_going:
                        break
                # Terminate if nothing currently visible
                if keep_going and (len(targets_observed) == targets_before_loop):
                    user_logger.warning("No targets are currently visible - stopping script instead of hanging around")
                    keep_going = False
            
            user_logger.info("Targets observed : %d (%d unique)" % (len(targets_observed), len(set(targets_observed))))
