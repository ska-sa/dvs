""" Common functions for use in the DVS observation framework.
    Intended to be used in the observe scripts as per following examples:
    
    Session "HACKS" to work around CAM/DVS gaps:
    -------------------------------------------
    Either this way:
    
        from dvs_obslib import standard_script_options, start_hacked_session as start_session
        
        with verify_and_connect(...) as kat:
            ...
            with start_session(...) as...:
                session.standard_setup(...)
                ...
    
    or this way:
    
        from dvs_obslib import start_nocapture_session as start_session
        ...
    
    
    Observation functions:
    ----------------------
    More versatile than the standard one:
        from dvs_obslib import collect_targets
    
    
    More efficient planning and filtering for target collections:
        from dvs_obslib import plan_targets, filter_separation
        
        target = filter_separation(pointing_sources, time.time(), kat.sources.antenna,
                                   separation_deg=opts.min_separation, sunmoon_separation_deg=5)
        
        target, expected_duration = plan_targets(targets, time.time(), t_observe=duration,
                                                 antenna=kat.ants[0], el_limit_deg=opts.horizon+5.0)
    
    
    @author: aph@sarao.ac.za
"""
try:
    from katcorelib import user_logger, standard_script_options as _kcl_std_opts_, start_session as _kcl_start_session_
except:
    import logging
    user_logger = logging.getLogger("user")
    user_logger.info("Not running in the OBS framework, some hacks may break!")
import time, os, telnetlib
import numpy as np
import katpoint


def standard_script_options(usage, description):
    """ Add additional options that are used by `start_hacked_session()`. """
    parser = _kcl_std_opts_(usage, description)
    parser.add_option('--reset-gain', type='int', default=None,
                      help='Value for the reset of the correlator F-engine gain '
                           '(default=%default)')
    ## HACK this legacy option:
    # parser.add_option('-n', '--nd-params', default='coupler,10,10,180',
    #                   help="Noise diode parameters as "
    #                        "'<diode>,<on>,<off>,<period>', in seconds or 'off' "
    #                        "for no noise diode firing (default='%default')")
    nd_params = parser.get_option('--nd-params')
    nd_params.help += ". Use 'switching,3,7,-1' to activate digitiser-level switching, in integer multiples of SDP visibilities dump intervals."
    return parser


class start_nocapture_session(object):
    """ Hacked "no capture session" to ignore cbf & sdp """
    def __enter__(self):
        return self
    
    def __exit__(self, *a):
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
        samples = np.atleast_2d(np.c_[timestamp, az, el])
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


def start_nd_switching(sub, n_on, n_off, T0='now', ND_LEAD_TIME=5):
    """ Start Digitiser-level synchronous Noise Diode cycling. S-band not supported of course.
        @param n_on, n_off: describe integer number of SDP dump intervals
        @param T0: time when the digitisers should trigger the start of the switching cycles.
    """
    nd_switching = [n_on, n_off]
    on_fraction = float(nd_switching[0])/np.sum(nd_switching)
    T0 = time.time() if (T0 == 'now') else T0
    T0 = int(max(T0, time.time()+ND_LEAD_TIME) + 0.5) # On a PPS boundary
    
    try: # 'sub' is an active session
        cbf, sdp, ants = sub.cbf, sub.sdp, sub.ants
    except: # 'sub' is a cam control environment. TODO: we assume subarray 1!
        cbf, sdp, ants = sub.cbf_1, sub.sdp_1, sub.ants
    
    cbf_dt = cbf.sensors.wide_baseline_correlation_products_int_time.get_value()
    sdp_dt = cbf_dt * np.round(sdp.sensors.dump_rate.get_value() * cbf_dt) # SDP dump rate is not accurate
    ants.req.dig_noise_source(T0, on_fraction, sdp_dt*np.sum(nd_switching)) # TODO: noise_source() vs noise_diode()?
    
    user_logger.info("Started digitiser-level noise diode switching.")


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
        
        # Set the gain to a single non complex number if requested
        eq_gain = kwargs.get("reset_gain", None)
        if eq_gain:
            session.set_fengine_gains(eq_gain)
        
        if (not session._cam_.dry_run):
            # Ensure the Ku-band signal generator matches the center frequency of the subarray
            match_ku_siggen_freq(session._cam_)
            # NB: the following must be called after `standard_setup()` because for MKE Dishes that causes a "major state transition"
            # during which the ACU resets some things which we are trying to hack around.
            temp_hack_DisableAllPointingCorrections(session._cam_)
        
        return result
    session.standard_setup = hacked_setup
    
    # Hack the "standard capture start" function
    session._capture_start_ = session.capture_start
    def hacked_capture_start(*a, **k):
        result = session._capture_start_(*a, **k)
        if (not session._cam_.dry_run):
            # Start noise diode switching, if requested
            nd_params = kwargs.get("nd_params", dict(diode=None))
            if (nd_params['diode'] == "switching"):
                n_on, n_off = int(nd_params['on']), int(nd_params['off'])
                start_nd_switching(session, n_on, n_off, T0=session.capture_block_ids[0]) # TODO: confirm that X-engine accumulation always starts on a PPS edge
        return result
    session.capture_start = hacked_capture_start
    
    # Hack the "standard end" function
    session._end_ = session.end
    def hacked_end(*a, **k):
        if (not session._cam_.dry_run):
            # Stop noise diode switching, if requested
            nd_params = kwargs.get("nd_params", dict(diode=None))
            if (nd_params['diode'] == "switching"):
                session.ants.req.dig_noise_source('now', 'off')
                user_logger.info("Stopped digitiser-level noise diode switching.")
        return session._end_(*a, **k)
    session.end = hacked_end
    
    return session


def collect_targets(cam, args, opts=None):
    """ Alternative to katcorelib.collect_targets():
            a) this can take a catalogue in either radec or TLE text formats
            b) if a catalogue is specified plus target names then only that SUBSET of targets are used (katcorelib.collect_targets() uses the catalogue PLUS targets)
        If a catalogue is specified it must either be the first argument, or opts/catalogue.
        
        @param args: either empty list, or the first string should contain a target name or comma-separated list of names
        @param opts: parsed options may contain `.catalogue`, the filename of the catalogue to load.
        @return: katpoint.Catalogue
    """
    # Get the catalogue to draw from
    cat, catfn = None, None
    # Possibly first argument is the catalogue
    if (len(args) > 0) and os.path.isfile(args[0]):
        catfn = args[0]
        args = args[1:]
    else: # Optionally specify --catalogue
        try: 
            catfn = opts.catalogue
        except:
            pass  
    if catfn and os.path.isfile(catfn):
        cat = katpoint.Catalogue(antenna=cam.sources.antenna)
        try: # Maybe a standard catalogue file
            cat.add(open(catfn, 'rt'))
        except ValueError: # Possibly a TLE formatted file
            try:
                cat.add_tle(open(catfn, 'rt'))
            except:
                raise ValueError("%s is not a valid target catalogue file!" % catfn)
    
    # Just a catalogue specified - so return that as the targets to use
    if (len(args) == 0) and (cat is not None):
        return cat
    elif (cat is None): # No catalogue file specified, use the standard one
        cat = cam.sources
    
    # Subset / specified targets from the catalogue
    selected_tgts = []
    for arg in args:
        tgts = [cat[tgt] for tgt in arg.split(",")] 
        named_tgts = [tgt for tgt in tgts if (tgt is not None)]
        if (len(named_tgts) > 0): # Targets by name
            selected_tgts.extend(named_tgts)
        else: # Else may only be a full target description
            selected_tgts.append(katpoint.Target(arg))
    assert (len(selected_tgts) > 0), "No target retrieved from argument list!"
    cat = katpoint.Catalogue(selected_tgts, antenna=cam.sources.antenna)
    return cat


def filter_separation(catalogue, T_observed, antenna=None, separation_deg=1, sunmoon_separation_deg=10):
    """ Removes targets from the supplied catalogue which are within the specified distance from others or either the Sun or Moon.

        @param catalogue: [katpoint.Catalogue]
        @param T_observed: UTC timestamp, seconds since epoch [sec].
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str]
        @param separation_deg: eliminate targets closer together than this (default 1) [deg]
        @param sunmoon_separation_deg: omit targets that are closer than this distance from Sun & Moon (default 10) [deg]
        @return: katpoint.Catalogue (a filtered copy of input catalogue)
    """
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    targets = list(catalogue.targets)
    avoid_sol = [katpoint.Target('%s, special'%n) for n in ['Sun','Moon']] if (sunmoon_separation_deg>0) else []

    separation_rad = separation_deg*np.pi/180.
    sunmoon_separation_rad = sunmoon_separation_deg*np.pi/180.

    # Remove targets that are too close together (unfortunately also duplicated pairs)
    overlap = np.zeros(len(targets), float)
    for i in range(len(targets)-1):
        t_i = targets[i]
        sep = [(t_i.separation(targets[j], T_observed, antenna) < separation_rad) for j in range(i+1, len(targets))]
        sep = np.r_[np.any(sep), sep] # Flag t_j too, if overlapped
        overlap[i:] += np.asarray(sep, int)
        # Check for t_i overlapping with solar system bodies
        sep = [(t_i.separation(j, T_observed, antenna) < sunmoon_separation_rad) for j in avoid_sol]
        if np.any(sep):
            user_logger.info("%s appears within %g deg from %s"%(t_i, sunmoon_separation_deg, np.compress(sep,avoid_sol)))
            overlap[i] += 1
    if np.any(overlap > 0):
        user_logger.info("Planning drops the following due to being within %g deg away from other targets:\n%s"%(separation_deg, np.compress(overlap>0,targets)))
        targets = list(np.compress(overlap==0, targets))

    filtered = katpoint.Catalogue(targets, antenna=antenna)
    return filtered


def plan_targets(catalogue, T_start, t_observe, dAdt=1.8, antenna=None, el_limit_deg=20):
    """ Generates a "nearest-neighbour" sequence of targets to observe, starting at the specified time.
        This does not consider behaviour around the azimuth wrap zone.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_start: UTC timestamp, seconds since epoch [sec].
        @param t_observe: duration of an observation per target [sec]
        @param dAdt: angular rate when slewing (default 1.8) [deg/sec]
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str or antenna proxy]
        @param el_limit_deg: observation elevation limit (default 20) [deg]
        @return: [list of Targets], expected duration in seconds
    """
    # If it's an "antenna proxy, use current coordinates as starting point
    try:
        az0, el0 = antenna.sensor.pos_actual_scan_azim.get_value(), antenna.sensor.pos_actual_scan_elev.get_value()
        antenna = antenna.sensor.observer.value
    except: # No "live" coordinates so start from zenith
        az0, el0 = 0, 90
    start_pos = katpoint.construct_azel_target(az0*np.pi/180., el0*np.pi/180.)
    
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    
    todo = list(catalogue.targets)
    done = []
    T = T_start # Absolute time
    available = catalogue.filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
    next_tgt = available.closest_to(start_pos, T, antenna)[0] if (len(available.targets) > 0) else None
    while (next_tgt is not None):
        # Observe
        next_tgt.antenna = antenna
        done.append(next_tgt)
        todo.pop(todo.index(next_tgt))
        T += t_observe
        # Find next visible target
        available = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
        next_tgt, dGC = available.closest_to(done[-1], T, antenna)
        # Slew to next
        if next_tgt:
            T += dGC * dAdt
    return done, (T-T_start)
