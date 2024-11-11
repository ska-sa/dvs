#!/usr/bin/env python
# While recording only sensor data: Turns antenna at a constant rate in azimuth & elevation over a specified range

import time
import katpoint
import numpy as np

from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)

class ApDriveFailure(Exception):
    """AP failed to move due to drive failure."""


class ParametersExceedTravelRange(Exception):
    """Indexer is stuck in an undefined position."""


MKAT = True # This code currently only works with homogenous subarrays - either MeerKAT Receptors or SKA1-MID Dishes!

def rate_slew(ants, azim, elev, azim_speed=0.5, azim_range=360, elev_range=0.0, dry_run=False):
    """ Turn ants from starting azim,elev at the specified speeds for a duration to cover specified azim_range. """

    # Determine expected duration from requested test speed.
    T_duration = abs(azim_range/azim_speed)
    elev_speed = elev_range/T_duration
    # Slew timeout in case some of the antennas need to unwrap, at max azim speed of 2 deg/sec
    T_timeout = (270+360)/2.0 + 100
    # Only testing azim_range, not full travel range (460 degrees)
    expected_azim = azim + np.sign(azim_speed)*azim_range
    expected_elev = elev + elev_range
    
    if not ((15 < elev < 90) and (15 < expected_elev < 90)):
        raise ParametersExceedTravelRange("Cannot perform %g degree elev slew "
                                          "within the AP elev travel range "
                                          "from the given start position %g."%(elev_range,elev))
    if (-185 < azim < 275) and (-185 < expected_azim < 275):
        user_logger.info("Antennas will perform a rate slew to azimuth %s",
                         expected_azim)
    else:
        raise ParametersExceedTravelRange("Cannot perform %g degree azim slew "
                                          "within the AP azim travel range "
                                          "from the given start position %g."%(azim_range,azim))

    # Set this target for the receptor target sensor
    target = katpoint.Target('Name, azel, %s, %s' % (azim, elev))

    if not dry_run:
        ants.req.mode('STOP')
        if MKAT: # Use slew to bypass pointing model
            time.sleep(2)
            ants.req.ap_slew(azim, elev)
            # Since we are not using the proxy request we will have to explicitly
            # wait for the servo brakes to open and on-target sensor to update.
            time.sleep(2)
        else: # For SDQM don't have equivalent of ap_rate(), so must use the scan() mechanism. TODO: Pointing model corrections to be omitted.
            ants.req.target_azel(azim+azim_range/2., elev+elev_range/2.) # SDQM
            ants.req.scan(-azim_range/2., -elev_range/2., azim_range/2., elev_range/2., T_duration, 'plate-carree') # SDQM
            ants.req.mode('POINT') # Go to start of scan
        time.sleep(2)

    user_logger.info("Starting target description: '%s' ", target.description)

    if not dry_run:
        try:
            if MKAT:
                ants.wait('ap.on-target', True, timeout=T_timeout)
            else:
                ants.wait('lock', True, timeout=T_timeout) # SDQM Need this state before SCAN allowed (not dsm.targetLock since that is True during the scan phase).
        except Exception as e:
            user_logger.error("Timed out while waiting for AP to reach starting position. %s"%e)
            raise

    user_logger.info("AP has reached start position.")

    if not dry_run:
        if MKAT:
            ants.req.mode('STOP')
            time.sleep(2)
            ants.req.ap_rate(azim_speed, elev_speed)
        else: # For SDQM we use the scan() mechanism rather than ap_rate()
            ants.req.mode('SCAN') # Start the configured scan
        user_logger.info("Performing rate slew to azimuth %s at %s deg/sec.",
                         expected_azim, azim_speed)
        time.sleep(2)
        
        try: # Wait until we are at the expected end point
            # Ideally want to use ants.wait('ap.on-target', True, timeout) but that doesn't work for SDQM scan() approach???
            sensor_name = "ap.actual-azim" if MKAT else "dsm.azPosition"
            ants.set_sampling_strategy(sensor_name, "period 0.5")
            # Position threshold 2 degrees to catch it at 0.5 second polling
            # period at full speed (2 deg/sec).
            # NOTE (LW) : Increased this threshold in case the stop point is
            #             being missed and causing a timeout.
            threshold = 4
            wait_on_target = lambda t_az,t_el,timeout: ants.wait(sensor_name, lambda c: abs(c.value - t_az) < threshold, timeout=timeout)
        
            try: # Wait for the shortest possible time required
                wait_on_target(expected_azim, expected_elev, timeout=T_duration+100)
            except Exception as e:
                user_logger.info("Taking longer than expected to reach the end point. Allowing more time.")
                wait_on_target(expected_azim, expected_elev, timeout=T_timeout)
            if not MKAT:
                ants.wait('lock', True, timeout=300) # SDQM Need this to wait for scan to complete gracefully
            user_logger.info("AP has reached end position.")
        except Exception as e: # E.g. wind stow is recoverable, so just log and continue
            user_logger.info("Timed out while waiting for AP to reach end point. %s"%e)
        ants.req.mode('STOP')

    else: # Simulate the run time
        #time.sleep(T_duration) # Dry run may "hang" when it encounters long sleep()
        user_logger.info("AP has reached end position.")


# Set up standard script options

parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                 description="Turn the antennas in azimuth at the specified speed.")

parser.add_option('--start-az', type='float',
                  default=-135.0,
                  help='Starting azimuth for sequence (default=%default)')
parser.add_option('--start-el', type='float',
                  default=20.0,
                  help='Starting elevation for sequence (default=%default)')
parser.add_option('--az-range', type='float',
                  default=360.0,
                  help='Azimuth range (positive number) over which to turn, from starting position (default=%default)')
parser.add_option('--azim-speed', type='float',
                  default=0.5,
                  help='Azimuth turn speed (signed number) in deg/sec (default=%default)')
parser.add_option('--el-range', type='float',
                  default=0.0,
                  help='Elevation range (signed number) over which to turn, from starting position (default=%default)')
parser.add_option('--reverse', action='store_true', default=False,
                  help='Do the rate movement in both directions')
parser.add_option('--repeats', type='int', default=1,
                  help='Number of times to repeat a complete sequence, (default=%default)')
parser.add_option('--no-corrections', action='store_true',
                  help='Disable static and tilt corrections during the controlled movement, restore afterwards.')



# Parse the command line
opts, args = parser.parse_args()
opts.az_range = abs(opts.az_range)

# Check options and build KAT configuration, connecting to proxies and devices
with verify_and_connect(opts) as kat:

    ant_types = set([ant.name[0] for ant in kat.ants])
    assert (len(ant_types) == 1), "This procedure only works with subarrays consisting of just one type of receptor!"
    MKAT = "m" in ant_types # Global
    
    # Set sensor strategies
    kat.ants.set_sampling_strategy("ap.on-target", "event") # This is a combination of mount-lock & lock, so set all three
    kat.ants.set_sampling_strategy("mount-lock", "event")
    kat.ants.set_sampling_strategy("lock", "event")

    if not kat.dry_run and kat.ants.req.mode('STOP'):
        user_logger.info("Setting antennas to mode 'STOP'")
        time.sleep(2)
    else:
        if not kat.dry_run:
            user_logger.error("Unable to set antennas to mode 'STOP'!")

    try:
        if opts.no_corrections and not kat.dry_run:
            if MKAT:
                SPEM_state = False # For MeerKAT in operation it is safe to assume this is always False
                TILT_state = [(ant,ant.sensor.ap_point_error_tiltmeter_enabled.get_value()) for ant in kat.ants]
                kat.ants.req.ap_enable_point_error_systematic(False)
                kat.ants.req.ap_enable_point_error_tiltmeter(False)
            else:
                kat.ants.req.dsm_DisablePointingCorrections()

        for n in range(opts.repeats):
            rate_slew(kat.ants, opts.start_az, opts.start_el, opts.azim_speed, opts.az_range, opts.el_range, dry_run=kat.dry_run)
    
            if opts.reverse or (opts.repeats > 1):
                user_logger.info("1/2 sequence completed successfully!")
                user_logger.info("Scanning in reverse direction...")
                rate_slew(kat.ants, (opts.start_az+np.sign(opts.azim_speed)*opts.az_range)%360, (opts.start_el+opts.el_range), -opts.azim_speed, opts.az_range, -opts.el_range,
                          dry_run=kat.dry_run)
    
            user_logger.info("Sequence completed successfully!")
            
    finally:
        if not kat.dry_run:
            if opts.no_corrections:
                user_logger.info("Restoring ACU pointing correction states...")
                if MKAT:
                    kat.ants.req.ap_enable_point_error_systematic(SPEM_state)
                    for ant,state in TILT_state:
                        resp = ant.req.ap_enable_point_error_tiltmeter(state)
                        if not resp.succeeded:
                            user_logger.error("FAILED to restore ACU state for %s: %s"%(ant.name, resp.reply))
                else:
                    kat.ants.req.dsm_EnablePointingCorrections()

            kat.ants.req.mode('STOP')
            user_logger.info("Stopping antennas")
