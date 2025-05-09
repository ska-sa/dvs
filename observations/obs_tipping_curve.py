#!/usr/bin/python
# Perform tipping curve scan  and find a specified azimith if one is not give.

# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement
import time
from katcorelib import standard_script_options, verify_and_connect, start_session, user_logger
from dvs_obslib import standard_script_options, start_hacked_session as start_session # Override previous import
import numpy as np
import katpoint
# Set up standard script options
parser = standard_script_options(usage="%prog [options]",
                                 description="Perform tipping curve scan for a specified azimuth position. \
                                              Or Select a Satilite clear Azimith,\
                                              Some options are **required**.")
# Add experiment-specific options
parser.add_option('--el-limits', type="string", default="15,90",
                  help='"Minimum, maximum" elevation angles for the tipping curve, in degrees (default="%default")')
parser.add_option('-z', '--az', type="float", default=None,
                  help='Azimuth angle along which to do tipping curve, in degrees (default="%default")')
parser.add_option('--spacing', type="float", default=1.0,
                  help='The Spacing along the elevation axis of the tipping curve that measuremnts are taken, in degrees (default="%default")')
parser.add_option('-t', '--dwell-time', type="float", default=30,
                  help='Total dwell time per elevation step, with ND on for 1/3rd of the time (default="%default")')
parser.add_option( '--tip-both-directions', action="store_true" , default=False,
                  help='Do tipping curve from low to high elevation and then from high to low elevation')

# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Tipping Curve', nd_params='off', horizon=15)

# Parse the command line
opts, args = parser.parse_args()

el_lim = [float(e) for e in opts.el_limits.split(",")]
on_time = opts.dwell_time/3.
nd_period = on_time*2

with verify_and_connect(opts) as kat:
    # Iterate through elevation angles
    spacings = list(np.arange(el_lim[0],el_lim[1]+0.1,opts.spacing))
    if opts.tip_both_directions :
        spacings += list(np.arange(el_lim[1],el_lim[0]-0.1,-opts.spacing))
    # Ensure that azimuth is in valid physical range of -185 to 275 degrees
    if opts.az is None:
        user_logger.info("No Azimuth selected , selecting clear Azimuth")
        if not kat.dry_run:
            timestamp = [katpoint.Timestamp(time.time()+i) for i in (np.arange(len(spacings))*(on_time+nd_period+1.0))]
            #timestamps calculated as  number of pointings times the length of each pointing added to time.now()
            #load the standard KAT sources ... similar to the SkyPlot of the katgui
            observation_sources = kat.sources.filter(tags=['~Pulsars'])
            source_az = []
            for source in observation_sources.targets:
                az, el = np.degrees(source.azel(timestamp=timestamp))   # was rad2deg
                az[az > 180] = az[az > 180] - 360
                source_az += list(set(az[el > el_lim[0]]))
            source_az.sort()
            gap = np.diff(source_az).argmax()+1
            opts.az = (source_az[gap] + source_az[gap+1]) /2.0
            user_logger.info("Selecting Satillite clear Azimuth=%f"%(opts.az,))
        else:
            opts.az = 0
            user_logger.info("Selecting dummey Satillite clear Azimuth=%f"%(opts.az,))
    else:
        if (opts.az < -185.) or (opts.az > 275.):
            opts.az = (opts.az + 180.) % 360. - 180.
        user_logger.info("Tipping Curve at Azimuth=%f"%(opts.az,))
    user_logger.info("Tipping Curve at Azimuth=%f"%(opts.az,))

    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.capture_start()
        for el in spacings:
            session.label('track')
            session.track('azel, %f, %f' % (opts.az, el), duration=on_time)
            session.fire_noise_diode('coupler', on=nd_period/2., off=nd_period/2.)
