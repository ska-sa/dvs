""" Temporary hacks and "hack frameworks", for use in the OBS environment.
    @author: aph@sarao.ac.za
"""
try:
    from katcorelib import user_logger
except:
    print("INFO: not running in the OBS framework, some hacks may break!")
import katpoint, time, numpy


class start_session(object):
    """ Hacked "no capture session" to ignore cbf & sdp """
    def __enter__(self):
        return self
    def __exit(self, *a):
        pass
    def __init__(self, kat, *a, **k):
        self.obs_params = {}
        self.nd_params = {}
        self.ants = kat.ants
        kat.ants.set_sampling_strategy("lock", "event")
        self.dry_run = kat.dry_run
        self.telstate = self
    def add(self, *a, **k): # For telstate
        pass
    def capture_start(self): # Ignored
        pass
    def standard_setup(self, *a, **k): # Ignored
        pass
    def set_target(self, target): # copied from https://github.com/ska-sa/katcorelib/blob/master/katcorelib/rts_session.py
        if self.ants is None:
            raise ValueError('No antennas specified for session - '
                             'please run session.standard_setup first')
        # Create references to allow easy copy-and-pasting from this function
        ants = self.ants
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)
        
        if self.dry_run: return True
        for ant in ants:
            # Don't set the target unnecessarily as this upsets the antenna
            # proxy, causing momentary loss of lock
            current_target = ant.sensor.target.get_value()
            if target != current_target:
                # Set the antenna target (antennas will already move there if in mode 'POINT')
                ant.req.target(target.description)
    def on_target(self, target): # copied from https://github.com/ska-sa/katcorelib/blob/master/katcorelib/rts_session.py
        if self.ants is None:
            return False
        # Turn target object into description string (or use string as is)
        target = getattr(target, 'description', target)
        for ant in self.ants:
            # Ignore disconnected antennas or ones with missing sensors
            if (not ant.is_connected() or
                    any([s not in ant.sensor for s in ('target', 'mode', 'lock')])):
                continue
            if (ant.sensor.target.get_value() != target or
                    ant.sensor.mode.get_value() != 'POINT' or
                    not ant.sensor.lock.get_value()):
                return False
        return True
    def track(self, target, duration=20.0, announce=True): # copied from https://github.com/ska-sa/katcorelib/blob/master/katcorelib/rts_session.py
        if self.ants is None:
            raise ValueError('No antennas specified for session - '
                             'please run session.standard_setup first')
        # Create references to allow easy copy-and-pasting from this function
        session, ants = self, self.ants
        # Convert description string to target object, or keep object as is
        target = target if isinstance(target, katpoint.Target) else katpoint.Target(target)

        if announce:
            user_logger.info("Initiating %g-second track on target '%s'",
                             duration, target.name)
        if self.dry_run: return True
        
        # This already sets antennas in motion if in mode POINT
        session.set_target(target)

        # Avoid slewing if we are already on target
        if not session.on_target(target):
            user_logger.info('slewing to target')
            ants.req.mode('POINT')
            # Wait until they are all in position (with 5 minute timeout)
            ants.wait('lock', True, timeout=300)
            user_logger.info('target reached')

        user_logger.info('tracking target')
        # Do nothing else for the duration of the track
        time.sleep(duration)
        user_logger.info('target tracked for %g seconds', duration)
        return True
    def load_scan(self, timestamp, az, el): # copied from https://github.com/ska-sa/katcorelib/blob/master/katcorelib/rts_session.py
        samples = numpy.atleast_2d(numpy.c_[timestamp, az, el])
        csv = '\n'.join([('%.3f,%.4f,%.4f' % (t, a, e)) for t, a, e in samples])
        if not self.dry_run:
            return self.ants.req.load_scan(csv)
        else:
            return True                    


def apply(kat):
    """ Apply mandatory hacks - if in a configured OBS framework """
    if not kat.dry_run:
        # HACK 1: disable tilt corrections because it's not calibrated / implemented correctly as of 14/11/2024
        # NB: this must happen after dish proxy STOP - which happens in session.standard_setup()
        try:
            import tango
            dsm = tango.DeviceProxy('10.96.66.100:10000/mid_dsh_0119/lmc/ds_manager')
            dsm.tiltPointCorrEnabled = False
            dsm.DisablePointingCorrections()
            user_logger.warning("APPLIED HACK: Dish#119 Tilt & Pointing Corrections Disabled")
        except Exception as e:
            user_logger.warning("Failed to disable Corrections on Dish#119: %s" % e)
