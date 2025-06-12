#!/usr/bin/env python
# Track target(s) for a specified time.

import time, os
import numpy as np

from katcorelib import (standard_script_options, verify_and_connect,
                        start_session, user_logger)
from dvs_obslib import plan_targets, filter_separation, collect_targets, standard_script_options, start_nocapture_session, start_hacked_session as start_session # Override previous import


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Track one or more sources for a specified time. At least one ' \
              'target must be specified. Note also some **required** options below.'
parser = standard_script_options(usage=usage, description=description)
parser.add_option('--catalogue', default='',
                  help="Name of file containing catalogue of targets to use, instead of default system catalogue.")

# Add experiment-specific options
parser.add_option('--no-capture', action="store_true", default=False,
                  help='Start the session without interacting with either CBF or SDP (i.e. subarray doesn\'t need to have them!)')
parser.add_option('-t', '--track-duration', type='float', default=60.0,
                  help='Length of time to track each source, in seconds '
                       '(default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=None,
                  help='Maximum duration of the script in seconds, after which the script will end as soon as the current track finishes '
                       '(no limit by default, i.e. once through all specified sources)')
parser.add_option('--repeat', action="store_true", default=False,
                  help='Repeatedly loop through the targets until maximum '
                       'duration (which must be set for this)')
parser.add_option('--min-separation', type="float", default=1.0,
                  help="Minimum separation angle to enforce between any two targets, in degrees (default=%default)")
parser.add_option('--sunmoon-separation', type="float", default=10,
                  help="Minimum separation angle to enforce between targets and the sun & moon, in degrees (default=%default)")

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Target track', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if (not os.path.isfile(opts.catalogue)) and (len(args) == 0):
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

if opts.no_capture:
    start_session = start_nocapture_session

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:
    targets = collect_targets(kat, args, opts)
    # Remove sources that crowd too closely
    targets = filter_separation(targets, time.time(), kat.sources.antenna,
                                separation_deg=opts.min_separation, sunmoon_separation_deg=opts.sunmoon_separation)
    # Quit early if there are no sources to observe
    if len(targets.filter(el_limit_deg=opts.horizon)) == 0:
        raise NoTargetsUpError("No targets are currently visible - "
                               "please re-run the script later")
    # Start capture session, which creates HDF5 file
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()

        start_time = time.time()
        targets_observed = []
        # Keep going until the time is up
        target_total_duration = [0.0] * len(targets)
        keep_going = True
        while keep_going:
            keep_going = (opts.max_duration is not None) and opts.repeat
            targets_before_loop = len(targets_observed)
            if False: # Iterate through source list, picking the next one that is up
                sequence_of_targets = targets.iterfilter(el_limit_deg=opts.horizon)
            else: # OR in a nearest-neighbour sequence
                sequence_of_targets = plan_targets(targets, time.time(), t_observe=opts.track_duration,
                                                   antenna=kat.ants[0], el_limit_deg=opts.horizon)[0]
            for n, target in enumerate(sequence_of_targets):
                # Cut the track short if time ran out
                duration = opts.track_duration
                if opts.max_duration is not None:
                    time_left = opts.max_duration - (time.time() - start_time)
                    if time_left <= 0.:
                        user_logger.warning("Maximum duration of %g seconds "
                                            "has elapsed - stopping script",
                                            opts.max_duration)
                        keep_going = False
                        break
                    duration = min(duration, time_left)
                session.label('track')
                if session.track(target, duration=duration):
                    targets_observed.append(target.description)
                    target_total_duration[n] += duration
            if keep_going and len(targets_observed) == targets_before_loop:
                user_logger.warning("No targets are currently visible - "
                                    "stopping script instead of hanging around")
                keep_going = False
        session.ants.req.dig_noise_source("now", 0)
        user_logger.info("Targets observed : %d (%d unique)",
                         len(targets_observed), len(set(targets_observed)))
        # print out a sorted list of target durations
        ind = np.argsort(target_total_duration)
        for i in reversed(ind):
            user_logger.info('Source %s observed for %.2f hrs',
                             targets.targets[i].description, target_total_duration[i] / 3600.0)
