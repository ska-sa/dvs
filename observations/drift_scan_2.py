#!/usr/bin/python
# Track target(s) for a specified time.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import time
from katcorelib import standard_script_options, verify_and_connect, collect_targets, start_session, user_logger
import katpoint
import math

# Set up standard script options
parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description='Track one or more sources for a specified time. At least one '
                                             'target must be specified. Note also some **required** options below.')
# Add experiment-specific options
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which script will end '
                       'as soon as the current track finishes (no limit by default)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum duration (which must be set for this)')
parser.add_option('--drift-duration', type='float', default=300,
                  help='Total duration of drift scan')                  

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Drift scan',nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

#if len(args) == 0:
#    raise ValueError("Please specify at least one target argument via name ('Cygnus A'), "
#                     "description ('azel, 20, 30') or catalogue file name ('sources.csv')")

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
#    observation_sources = collect_targets(kat, args)
    # Quit early if there are no sources to observe
#    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
#        user_logger.warning("No targets are currently visible - please re-run the script later")
#    else:
        # Start capture session, which creates HDF5 file
    if not kat.dry_run and kat.ants.req.mode('STOP') :
        user_logger.info("Setting Antenna Mode to 'STOP', Powering on Antenna Drives.")
        time.sleep(5)
    else:
        user_logger.error("Unable to set Antenna mode to 'STOP'.")
        
    observation_sources = collect_targets(kat, args)
    if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
        user_logger.warning("No targets are currently visible - please re-run the script later")
    else:
        with start_session(kat, **vars(opts)) as session:

            session.standard_setup(**vars(opts))
            session.capture_start()
            
            target =observation_sources.filter(el_limit_deg=opts.horizon).targets[0]
            
            session.track(target, duration=1)

            time.sleep(4)
            start_time = time.time()
            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)
            
            session.label('track')
            user_logger.info("Tracking %s for 30 seconds" % (target.name))	
            session.track(target, duration=30)
            
            #session.label('raster')
            #user_logger.info("Doing scan of '%s' with current azel (%s,%s) "%(target.description,target.azel()[0],target.azel()[1]))
            #session.raster_scan(target, num_scans=5, scan_duration=60, scan_extent=6.0,
            #                            scan_spacing=1, scan_in_azimuth=True,
            #                            projection=opts.projection)
            
            session.label('drift')
            start_time = time.time()
            az,el = target.azel(start_time + (opts.drift_duration / 2))
            if (az*180/math.pi > 275.0):
                az = az - (360/180 * math.pi)
            new_targ = katpoint.Target('Drift scan_duration of %s, azel, %10.8f, %10.8f' % (target.name, az*180/math.pi ,el*180/math.pi))
            user_logger.info("Initiating drift scan of %s" % (target.name))	
            az,el = target.azel(start_time + (opts.drift_duration / 2))            
            session.track(new_targ, duration=opts.drift_duration)
            
            session.label('track')
            user_logger.info("Tracking %s for 30 seconds" % (target.name))	
            session.track(target, duration=30)
            
            session.label('noise diode')
            session.fire_noise_diode('coupler', on=10, off=10)


