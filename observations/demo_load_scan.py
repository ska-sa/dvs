#!/usr/bin/python
""" A basic sequence using load_scan mechanism to steer the antennas - without any data capturing.
    @author: aph@sarao.ac.za
"""

import time
import numpy as np
import katpoint
from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger
from dvs_obslib import start_hacked_session as start_session # Override previous import, allows this to be used in a subarray without CBF & SDP


def load_load_scan_data(filename):
    """ Load a very simple scan pattern file.
        @return: [ [seconds from start, deg azimuth, deg elevation], ... ] """
    return np.loadtxt(filename, comments="#", delimiter=",", skiprows=1, unpack=False) # [ [seconds from start, deg azimuth, deg elevation], ... ]


def gen_scan(starttime, scan_pattern, clip_safety_margin=1.0, min_elevation=15., max_elevation=90.):
    """ @param starttime: Unix timestamp to assign to the first point.
        @param scan_pattern: [ [time0, az0, el0], ... [timeN, azN, elN] ] with time in floating point seconds, az & el in degrees
        @return: (scan_data, duration) with scan_data as expected by session.load_scan(), duration in seconds  """
    scan_data = np.array(scan_pattern, copy=True)
    scan_time, scan_az, scan_el = np.transpose(scan_data)
    
    timedata = starttime + (scan_time-scan_time[0])
    #clipping prevents antenna from hitting hard limit and getting stuck until technician reset it, 
    #but not ideal to reach this because then actual azel could be equal to requested azel even though not right and may falsely cause one to believe everything is ok
    azdata = np.unwrap(scan_az, 180)
    eldata = scan_el
    scan_data[:,0] = timedata
    scan_data[:,1] = np.clip(np.nan_to_num(azdata),-180.0+clip_safety_margin,270.0-clip_safety_margin)
    scan_data[:,2] = np.clip(np.nan_to_num(eldata),min_elevation+clip_safety_margin,max_elevation-clip_safety_margin)
    clipping_occurred = (azdata!=scan_data[:,1]) | (eldata!=scan_data[:,2])
    if np.any(clipping_occurred):
        user_logger.warn("Clipping occurs in scan pattern - continuing with it clipped.")
    valid_time = timedata > time.time()
    if not np.all(valid_time):
        user_logger.warn("Part of the scan pattern is in the past - continuing with it truncated.")
        scan_data = scan_data[valid_time,:]
    
    duration = scan_data[-1,0] - scan_data[0,0]
    return scan_data, duration


# Set up standard script options
parser = standard_script_options(usage="%prog [options] load_scan_filename",
                                 description='This script performs a simple load_scan raster, without capturing data.'
                                             'Note the **required** options below.')
# Add experiment-specific options
parser.add_option('--T0', type='string', default='',
                  help='Exact *future* start time in UTC katpoint.Timestamp format (default=%default) or else ASAP.')
parser.add_option('--num-cycles', type='int', default=1,
                  help='Number of cycles to complete (default=%default) use -1 for indefinite')
parser.add_option('--max-duration', type='float', default=1800,
                  help='Maximum total measurement time, in seconds (default=%default)')
parser.add_option('--prepopulatetime', type='float', default=10.0,
                  help='time in seconds to prepopulate buffer in advance (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Demonstration for load_scan', quorum=1.0, nd_params='off')


# Parse the command line
opts, args = parser.parse_args()
try:
    T0 = katpoint.Timestamp(opts.T0).secs
except Exception as e:
    if opts.T0:
        user_logger.warn("Specified start time could not be parsed - starting ASAP. Error is: %s" % e)
    T0 = -1

# Load the pattern and confirm that it is suitable
scan_pattern = load_load_scan_data(args[0])
scan_data, cycle_expected_duration = gen_scan(time.time()+1, scan_pattern, min_elevation=opts.horizon)


# Check basic command-line options and obtain a kat object connected to the appropriate system
with verify_and_connect(opts) as kat:

    # Initialise the session
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        
        # Prepare for the first cycle
        next_start = (time.time() + opts.prepopulatetime) if T0<0 else T0
        assert (next_start + max(1,opts.num_cycles)*cycle_expected_duration > time.time()), "All track timestamps are in the past - aborting!"
        
        user_logger.info("Initiating load_scan cycles (%s %g-second cycles)", 
                         ('unlimited' if opts.num_cycles<0 else '%d'%opts.num_cycles), cycle_expected_duration)
        
        start_time = time.time()
        cycle = 0
        while (cycle < opts.num_cycles) or (opts.num_cycles < 0): # Override exit conditions are coded in the next ten lines
            if (opts.max_duration is not None) and (time.time() - start_time >= opts.max_duration):
                user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script", opts.max_duration)
                break
            user_logger.info("Performing scan cycle %d of %s", cycle+1, ("unlimited" if opts.num_cycles<0 else '%d'%opts.num_cycles))
            user_logger.info("Using all antennas: %s", ' '.join([ant.name for ant in session.ants]))
            
            # Scan pattern for this cycle. It's a demo, so simply use the exact same az, el for all antennas
            scan_data, cycle_expected_duration = gen_scan(next_start, scan_pattern, min_elevation=opts.horizon)
            next_start = scan_data[0,0]
            user_logger.info("Current scan estimated to complete at %s UTC (duration %.1f minutes)", time.ctime(next_start+cycle_expected_duration+time.timezone), cycle_expected_duration/60.)
            
            # Slew / unwrap to get to start position
            _, t_az, t_el = scan_data[0]
            user_logger.info("Performing azimuth unwrap to start position: %s,%s"%(t_az, t_el))
            azeltarget = katpoint.Target('azimuthunwrap,azel,%s,%s'%(t_az, t_el))
            session.track(azeltarget, duration=0, announce=False)
            
            user_logger.info("Starting scan at %s UTC, in %g seconds time" % (time.ctime(next_start+time.timezone), next_start-time.time()))
            time.sleep(max(0, next_start-time.time() - opts.prepopulatetime))
            
            # Load in segments to avoid potential limitations and allow feedback
            for i in range(0, len(scan_data)//1000+1, 1):
                segment = slice(i*1000, min((i+1)*1000,len(scan_data))+1) # End with next segment's starting point ensures the ACU never sees a discontinuity
                t_t, t_az, t_el = scan_data[segment,0], scan_data[segment,1], scan_data[segment,2]
                user_logger.info("Loading segment %d: timestamps %.2f - %.2f", i, t_t[0], t_t[-1])
                if not kat.dry_run:
                    session.load_scan(t_t, t_az, t_el)
                time.sleep(max(0, t_t[-1]-time.time() - opts.prepopulatetime))
            # Prepare to seamlessly continue with the next cycle
            next_start = t_t[-1] + (t_t[1]-t_t[0])
            
            cycle+=1
            if (cycle > 1) and kat.dry_run:
                user_logger.info("Testing only two cycles for dry-run")
                break
        
        # Ensure last cycle is completed before terminating the script
        time.sleep(max(0, next_start-time.time()))
