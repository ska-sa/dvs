#!/usr/bin/env python
# Track the Moon and offset points from that, for a specified time.

import time

from katcorelib import (standard_script_options, verify_and_connect,
                        start_session, user_logger)
import katpoint
from katpoint import wrap_angle
import numpy as np


# Set up standard script options
description = 'Perform a measurement of system temperature using hot and cold ' \
              'on sky loads.'
parser = standard_script_options(usage="%prog [options]", description=description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=30.0,
                  help='Length of time for each pointing, in seconds (default=%default)')
parser.add_option('-m', '--max-duration', type='float', default=-1,
                  help='Maximum duration of script to repeat the measurement, in seconds ' \
                  '(the default is to repeat the measurement just once)')
parser.add_option('--offset', type='float', default=10.0,
                  help='Offset from the Moon for "cold load" measurement, in degrees (default=%default)')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Hotload and Coldload observation', dump_rate=1.0 / 0.512)
# Parse the command line
opts, args = parser.parse_args()


nd_off = {'diode': 'coupler', 'on': 0., 'off': 0., 'period': -1.}
nd_coupler = {'diode': 'coupler', 'on': opts.track_duration, 'off': 0., 'period': 0.}


with verify_and_connect(opts) as kat:
    moon = kat.sources.lookup['moon'][0]
    with start_session(kat, **vars(opts)) as session:
        session.standard_setup(**vars(opts))
        session.nd_params = nd_off
        session.capture_start()
        once = True
        start_time = time.time()
        while once or time.time() < start_time + opts.max_duration:
            once = False
            # TODO: avoid making an azel target: use radec target, with session.track_offset() instead 
            off1_azel = katpoint.construct_azel_target(wrap_angle(moon.azel()[0] + np.radians(opts.offset)),
                                                       moon.azel()[1])
            off1_azel.antenna = moon.antenna
            off1 = katpoint.construct_radec_target(off1_azel.radec()[0], off1_azel.radec()[1])
            off1.antenna = moon.antenna
            off1.name = 'off1'

            off2_azel = katpoint.construct_azel_target(wrap_angle(moon.azel()[0] - np.radians(opts.offset)),
                                                       moon.azel()[1])
            off2_azel.antenna = moon.antenna
            off2 = katpoint.construct_radec_target(off2_azel.radec()[0], off2_azel.radec()[1])
            off2.antenna = moon.antenna
            off2.name = 'off2'
            sources = [moon, off2, off1]
            txtlist = ', '.join(["'%s'" % (target.name,) for target in sources])
            user_logger.info("Calibration targets are [%s]", txtlist)
            for target in sources:
                session.nd_params = nd_off
                for nd in [nd_coupler]:
                    session.nd_params = nd_off
                    session.track(target, duration=0)  # get onto the source
                    user_logger.info("Now capturing data - diode %s on", nd['diode'])
                    session.label('%s' % (nd['diode'],))
                    if not session.fire_noise_diode(announce=True, **nd):
                        user_logger.error("Noise diode %s did not fire", nd['diode'])
                session.nd_params = nd_off
                user_logger.info("Now capturing data - noise diode off")
                session.label('track')
                session.track(target, duration=opts.track_duration)
        if opts.max_duration and time.time() > start_time + opts.max_duration:
            user_logger.info('Maximum script duration (%d s) exceeded, stopping script',
                             opts.max_duration)