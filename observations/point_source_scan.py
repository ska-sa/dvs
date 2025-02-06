#!/usr/bin/python
# Raster scans on targets from a specified catalogue.
# Typically used for SingleDish pointing model fits and gain curve calculation.

import time

from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger
from dvs_obslib import plan_targets, filter_separation, collect_targets, standard_script_options, start_hacked_session as start_session # Override previous import


# Script options
parser = standard_script_options(
    usage="%prog [options] [<'target/catalogue'> ...]",
    description="Perform raster scans across sources loaded from a catalogue.")
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
    'uhf': dict(num_scans=9, scan_duration=60, scan_extent=5.5, scan_spacing=5.5/8, scan_in_az=[True]),
    'l': dict(num_scans=9, scan_duration=40, scan_extent=3.5, scan_spacing=3.5/8, scan_in_az=[True]),
    's': dict(num_scans=9, scan_duration=30, scan_extent=2.0, scan_spacing=2.0/8, scan_in_az=[True]),
    'ku': dict(num_scans=11, scan_duration=20, scan_extent=0.5, scan_spacing=0.5/10, scan_in_az=[True]),
    'ku-slow': dict(num_scans=11, scan_duration=40, scan_extent=0.5, scan_spacing=0.5/10, scan_in_az=[True]),
    # Ku-band initial pointing, either MeerKAT or SKA Dish
    'ku-wide': dict(num_scans=17, scan_duration=35, scan_extent=1.0, scan_spacing=0.06, scan_in_az=[True]), # Az 1.0 x El 1.0deg, 0.029deg per sec (~HPBWmin/4)
    'ku-search': dict(num_scans=17, scan_duration=35, scan_extent=3.0, scan_spacing=0.09, scan_in_az=[True]), # Az 3.0 x El 1.5deg, 0.086deg per sec
    # Standard for SKA Dish
    'skab1': dict(num_scans=9, scan_duration=60, scan_extent=6.6, scan_spacing=6.6/8, scan_in_az=[True]),
    'skab2': dict(num_scans=9, scan_duration=30, scan_extent=3.0, scan_spacing=3.0/8, scan_in_az=[True]),
    'skab3': dict(num_scans=9, scan_duration=24, scan_extent=1.8, scan_spacing=1.8/8, scan_in_az=[True]),
    'skab3_AzEl': dict(num_scans=9, scan_duration=12, scan_extent=1.8, scan_spacing=1.8/8, scan_in_az=[True,False]), # Experimental "basket weave"
    'skab4': dict(num_scans=9, scan_duration=24, scan_extent=1.0, scan_spacing=1.0/8, scan_in_az=[True]),
    'skab5a': dict(num_scans=9, scan_duration=24, scan_extent=0.6, scan_spacing=0.6/8, scan_in_az=[True]),
    'skaku': dict(num_scans=13, scan_duration=13, scan_extent=0.5, scan_spacing=0.5/12, scan_in_az=[True]),
    # Proposed for SKA Dish: extent = 2.3*HPBWmax; speed = HPBW/3/1sec - to be done with 1sec dumps!
    '_skab1': dict(num_scans=21, scan_duration=21, scan_extent=9.0, scan_spacing=9.0/20, scan_in_az=[True]),
    '_skab2': dict(num_scans=13, scan_duration=13, scan_extent=3.3, scan_spacing=3.3/12, scan_in_az=[True]),
    '_skab3': dict(num_scans=13, scan_duration=13, scan_extent=1.8, scan_spacing=1.8/12, scan_in_az=[True]),
    '_skab4': dict(num_scans=13, scan_duration=13, scan_extent=1.0, scan_spacing=1.0/12, scan_in_az=[True]),
    '_skab5a': dict(num_scans=13, scan_duration=13, scan_extent=0.63, scan_spacing=0.63/12, scan_in_az=[True]),
}
parser.add_option('--style', type='choice', choices=styles.keys(),
                  help="Raster scan style determining number of scans, scan duration, scan extent "
                       "and scan spacing. The available styles are: %s" % (styles,))
parser.add_option('--alternate-style', type='choice', choices=styles.keys(), default=None,
                  help="If given, will repeat the measurement on each target with this style (default=%default)")


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
    raster_params = [styles[opts.style]]
    if opts.alternate_style:
        raster_params.append(styles[opts.alternate_style])
    
    # Quit early if there are no sources to observe
    if (len(pointing_sources) == 0):
        user_logger.warning("Empty point source catalogue or all targets are skipped")
    elif len(pointing_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        with start_session(kat, **vars(opts)) as session:
            session.standard_setup(**vars(opts))
            session.capture_start()

            start_time = time.time()
            targets_observed = []
            # Keep going until the time is up
            keep_going = True
            while keep_going:
                targets_before_loop = len(targets_observed)
                # Iterate through source list, picking the next one from the nearest neighbour plan
                raster_duration = raster_params[0]["num_scans"] * (raster_params[0]["scan_duration"]+5) # Incl. buffer between scans. TODO: Ignoring ND
                raster_duration *= max([len(_["scan_in_az"]) for _ in raster_params])
                for target in plan_targets(pointing_sources, time.time(), t_observe=raster_duration,
                                           antenna=kat.ants[0], el_limit_deg=opts.horizon+5.0)[0]:
                    for style_params in raster_params:
                        session.label('raster')
                        completed = 0
                        style_params = dict(style_params) # Make a copy since we need to remove 'scan_in_az'
                        for scan_in_az in style_params.pop("scan_in_az"):
                            az, el = target.azel()
                            user_logger.info("Scanning target %r with current azel (%s, %s)" % (target.description, az, el))
                            if session.raster_scan(target, scan_in_azimuth=not scan_in_az if opts.scan_in_elevation else scan_in_az,
                                                   projection=opts.projection, **style_params):
                                completed += 1
                        if (completed > 0):
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
