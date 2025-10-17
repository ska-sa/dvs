#!/usr/bin/env python
# While recording only sensor data: Turns antenna at a constant rate in azimuth & elevation over a specified range

import time
import numpy as np

from katcorelib import (standard_script_options,
                        verify_and_connect,
                        user_logger)
from dvs_obslib import hack_SetPointingCorrections


class ParametersExceedTravelRange(Exception):
    pass

def wrap(angle, period):
    """ @return: angle in the interval -period/2 ... period/2 """
    return (angle + 0.5*period) % period - 0.5*period


def rate_slew(ants, azim, elev, azim_speed=0.5, azim_range=360, elev_range=0.0, dry_run=False):
    """ Turn ants from center azim,elev at the specified speeds for a duration to cover specified azim_range & elev_range. """
    azim = wrap(azim, 360)
    target = "Scan, azel, %f,%f, d_az %f..%f, d_el %f..%f"%(azim,elev, -azim_range/2,azim_range/2, -elev_range/2,elev_range/2)
    start_azim = azim - np.sign(azim_speed)*azim_range/2
    start_elev = elev - elev_range/2
    end_azim = azim + np.sign(azim_speed)*azim_range/2
    end_elev = elev + elev_range/2
    if not ((15 < start_elev < 90) and (15 < end_elev < 90)):
        raise ParametersExceedTravelRange("Cannot perform %g degree elev scan centered on %g."%(elev_range,elev))
    if not ((-185 < start_azim < 275) and (-185 < end_azim < 275)):
        raise ParametersExceedTravelRange("Cannot perform %g degree azim scan centered on %g."%(azim_range,azim))
    
    # Slew timeout in case some of the antennas need to unwrap, at max azim speed of 2 deg/sec
    T_timeout = (270+360)/2.0 + 100
    
    # Determine expected duration from requested test speed.
    T_duration = abs(azim_range/azim_speed)
    
    if not dry_run:
        ants.req.mode('STOP')
        ants.req.target_azel(azim, elev)
        ants.req.scan(-azim_range/2, -elev_range/2, azim_range/2, elev_range/2, T_duration, 'plate-carree')
        ants.req.mode('POINT') # Go to start of scan
    user_logger.info("Going to starting position for: '%s' ", target)
    
    if not dry_run:
        time.sleep(5) # Avoid triggering before the antennas have started moving
        ants.wait('lock', True, timeout=T_timeout) # Need to achieve this state before allowed to trigger SCAN
    
    user_logger.info("Performing scan to azimuth %s at %s deg/sec.", end_azim, azim_speed)
    if not dry_run:
        ants.req.mode('SCAN') # Start the configured scan
        hack_SetPointingCorrections(ants, tilt_enabled=False) # TODO: hack necessary 11/2024 because ACU re-enables it
    
    if not dry_run:
        time.sleep(2) # Avoid triggering before the antennas have started moving
        try: # Wait until we are at the expected end point
            try: # Wait for the shortest possible time required
                ants.wait('lock', True, timeout=T_duration+100)
                user_logger.info("Reached the end position.")
            except Exception as e:
                user_logger.info("Taking longer than expected to reach the end point. Allowing unwrap time.")
                ants.wait('lock', True, timeout=T_timeout)
            ants.wait('lock', True, timeout=100) # Allow scan to complete gracefully
        except Exception as e: # Some conditions like wind stow are recoverable, so just log and try to continue
            user_logger.info("Timed out while waiting to reach end point. %s"%e)
        
        ants.req.mode('STOP')
    else:
        user_logger.info("Reached the end position.")
    


# Set up standard script options

parser = standard_script_options(usage="%prog [options]",
                                 description="Turn the antennas in azimuth at the specified speed - without capturing sky data.")

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

    # The interfaces for these two are different in some respects
    mkat_ants = []
    mke_ants = []
    ska_ants = []
    TILT_state = {} # ant name:tilt_corr_enabled boolean
    
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

    mean_az = opts.start_az + opts.az_range/2
    mean_el = opts.start_el + opts.el_range/2
    try:
        if opts.no_corrections and not kat.dry_run: # Disable tilt correction
            for ant in kat.ants:
                if hasattr(ant.sensor, "ap_point_error_tiltmeter_enabled"):
                    mkat_ants.append(ant)
                    TILT_state[ant.name] = ant.sensor.ap_point_error_tiltmeter_enabled.get_value()
                    ant.req.ap_enable_point_error_tiltmeter(False)
                elif hasattr(ant.sensor, "dsm_tiltPointCorrEnabled"):
                    mke_ants.append(ant)
                    TILT_state[ant.name] = ant.sensor.dsm_tiltPointCorrEnabled.get_value()
                else:
                    ska_ants.append(ant)
                    TILT_state[ant.name] = ant.sensor.dsm_tiltOnInput.get_value()
            hack_SetPointingCorrections(mke_ants+ska_ants, tilt_enabled=False)

        for n in range(opts.repeats):
            rate_slew(kat.ants, mean_az, mean_el, opts.azim_speed, opts.az_range, opts.el_range, dry_run=kat.dry_run)

            if opts.reverse or (opts.repeats > 1):
                user_logger.info("1/2 sequence completed successfully!")
                user_logger.info("Scanning in reverse direction...")
                rate_slew(kat.ants, mean_az, mean_el, opts.azim_speed, -opts.az_range, -opts.el_range, dry_run=kat.dry_run)
    
            user_logger.info("Sequence completed successfully!")
            
    finally:
        if not kat.dry_run:
            if opts.no_corrections:
                user_logger.info("Restoring ACU pointing correction states...")
                for ant in mkat_ants:
                    resp = ant.req.ap_enable_point_error_tiltmeter(TILT_state[ant.name])
                    if not resp.succeeded:
                        user_logger.error("FAILED to restore ACU state for %s: %s"%(ant.name, resp.reply))
                for ant in mke_ants:
                    if (TILT_state[ant.name] == True):
                        # ant.req.dsh_EnableTiltCorrections() # TODO: not exposed 11/2024
                        hack_SetPointingCorrections([ant], tilt_enabled=True, force=True)
                for ant in ska_ants:
                    if (TILT_state[ant.name] == True):
                        user_logger.error("FAILED to restore ACU state for %s: NOT IMPLEMENTED!"%(ant.name))

            kat.ants.req.mode('STOP')
            user_logger.info("Stopping antennas")
