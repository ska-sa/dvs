""" Hacks for use in the DVS observation framework.
    Intended to be used in the observe scripts as per following examples:
    
        from dvs_obslib import start_hacked_session as start_session
        
        with verify_and_connect(...) as kat:
            ...
            with start_session(...) as...:
                session.standard_setup(...)
                ...
    
    Or:
    
        from dvs_obslib import start_nocapture_session as start_session
        ...
    
    @author: aph@sarao.ac.za
"""
try:
    from katcorelib import user_logger, start_session as _kcl_start_session_
except:
    kcl_start_session = None
    import logging
    user_logger = logging.getLogger("user")
    user_logger.info("Not running in the OBS framework, some hacks may break!")
import time, numpy, os, telnetlib
import katpoint


class start_nocapture_session(object):
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


def match_ku_siggen_freq(cam, override=False):
    """ Permanent "hack" for DVS: the frequency of the Ku-band reference LO signal generator must be changed manually,
        the subarray's "x band" center frequency only updates sensors and metadata.
        See https://skaafrica.atlassian.net/browse/MKT-50 
    
        A change is only made if all active "x band" subarrays have the same center frequency.
    """
    try:
        active_subs = [sa for sa in [cam.subarray_1,cam.subarray_2,cam.subarray_3,cam.subarray_4] if (sa.sensors.state.get_value()=='active')]
    except: # subarray_X is not configured when this is run from an observation session
        active_subs = [cam.sub] # TODO: see other active subarrays?
    xband_subs = [sa for sa in active_subs if (sa.sensors.band.get_value()=='x')]
    xband_fc = [sa.sensors.requested_rx_centre_frequency.get_value() for sa in xband_subs]
    if (len(set(xband_fc)) == 1) or override:
        fc = xband_fc[0]
        fLO = (fc -  1712e6*3/4.) / 100. # Hz
        current_fLO = cam.anc.sensors.siggen_ku_frequency.get_value()
        if (abs(current_fLO-fLO) > 0.01):
            # cam.anc.req.siggen_ku_frequency(fLO) # Not exposed in observation session, even if "anc" added to the subarray controlled resources!
            kumc_conf = cam.anc.sensors.siggen_address.get_value()
            kumc = telnetlib.Telnet(*kumc_conf, timeout=2)
            kumc.write("?ku-frequency %.2f\n" % fLO)
            resp = kumc.read_until("ok", timeout=2).decode("UTF-8")
            kumc.close()
            if ("!ku-frequency ok" in resp):
                user_logger.info("UPDATED Ku reference LO from %g to %g" % (current_fLO, fLO))
            else:
                user_logger.error("ERROR! Failed to update Ku reference. Response:")
                user_logger.error(resp)
    elif (len(set(xband_fc)) > 1):
        user_logger.error("MANUAL OVERRIDE REQUIRED. Multiple x band subarrays are active at present with different center frequencies.") 


def temp_hack_DisableAllPointingCorrections(cam):
    """ Temporary hack to disable the MKE ACU static and dynamic pointing corrections - which gets automatically re-enabled
        by the ACU when there's a transition out of STANDBY (ESTOP/STOW/STOP) to OPERATE.
        
        MPI/OHB have been requested to make the "disabling" a latching configuration setting in the ACU.
        
        RE static corrections: this should always be disabled when CAM DishProxy is used
        RE tilt corrections: disable tilt corrections because it's not calibrated / implemented correctly as of 14/11/2024
    """
    # Find the antennas that expose this functionality:
    resp = cam.ants.req.dsm_DisablePointingCorrections()
    d_ants = [a for a,r in resp.items() if (r is not None)]
    
    # Now ensure those antennas have transitioned to the "Operating" mode, by POINT
    for ant in cam.ants:
        if (ant.name in d_ants):
            az, el = (ant.sensor.dsh_achievedPointing_1.get_value(), ant.sensor.dsh_achievedPointing_2.get_value())
            ant.req.target_azel(az+0.01, el+0.01) # Must be different or else the proxy doesn't propagate this to the ACU?
            ant.req.mode("POINT")
    time.sleep(5)
    resp = cam.ants.req.dsm_DisablePointingCorrections()
    time.sleep(1)
    
    user_logger.info("APPLIED HACK: Static Corrections Disabled on %s" % d_ants)
    user_logger.info("APPLIED HACK: Tilt Corrections Disabled on %s" % d_ants)


def start_hacked_session(cam, **kwargs):
    """ Start a capture session and apply standard hacks as required for proper operation of the DVS system.
        1. `standard_setup()` checks & updates Ku-band siggen frequency
        2. `standard_setup()` disables pointing corrections for MKE Dish ACU's
        
        @return: the session object.
    """
    # Start the session in the standard way
    session = _kcl_start_session_(cam, **kwargs)
    
    # Hack the "standard setup" function
    session._standard_setup_ = session.standard_setup
    session._cam_ = cam
    def hacked_setup(*a, **k):
        result = session._standard_setup_(*a, **k)
        if (not session._cam_.dry_run):
            match_ku_siggen_freq(session._cam_)
            # NB: the following must be called after `standard_setup()` because for MKE Dishes that causes a "major state transition"
            # during which the ACU resets some things which we are trying to hack around.
            temp_hack_DisableAllPointingCorrections(session._cam_)
            
            # TODO: for holog-scans set attenuation from a table, for satellites
        return result
    session.standard_setup = hacked_setup
    
    return session


def collect_targets(cam, args, opts=None):
    """ Similar to katcorelib.collect_targets(), but this can take TLE files!
        @param args: either empty list, or the first string should contain a target name or comma-separated list of names
        @param opts: parsed options containing at least `.catalogue`, the filename of the catalogue to load.
        @return: katpoint.Catalogue
    """
    try: # Optionally specify --catalogue
        catfn = opts.catalogue
    except: # Possibly first argument is the catalogue
        if (len(args) > 0) and os.path.isfile(args[0]):
            catfn = args[0]
            args = args[1:]
    if os.path.isfile(catfn):
        cat = katpoint.Catalogue(antenna=cam.sources.antenna)
        try: # Maybe a standard catalogue file
            cat.add(open(catfn, 'rt'))
        except ValueError: # Possibly a TLE formatted file
            try:
                cat.add_tle(open(catfn, 'rt'))
            except:
                raise ValueError("%s is not a valid target catalogue file!" % catfn)
    else: # No catalogue file specified, use the standard one
        cat = cam.sources
    if (len(args) > 0):
        try: # See if it is target definitions
            for arg in args:
                cat.add(arg)
        except: # Perhaps just target names
            tgts = [cat[tgt] for tgt in args[0].split(",")]
            tgts = [tgt for tgt in tgts if (tgt is not None)]
            assert (len(tgts) > 0), "No target retrieved from argument list!"
            return katpoint.Catalogue(tgts, antenna=cam.sources.antenna)
    return cat
