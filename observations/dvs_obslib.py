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
import tango


def standard_script_options(usage, description):
    """ Add additional options that are used by `start_hacked_session()`. """
    parser = _kcl_std_opts_(usage, description)
    
    # The default was changed from 15 to 20 in Dec. 2018, but was not intended to be changed for RTS (https://skaafrica.atlassian.net/browse/MKAIV-1378)
    parser.set_default('horizon', 15)
    
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
    
    parser.add_option('--no-tiltcorrections', action='store_true',
                      help='Explicitly disable tilt corrections during controlled movement.')

    return parser


class start_nocapture_session(object):
    """ Like katsdpscripts.rts_session.CaptureSession, but this one avoids interacting with cbf & sdp. """
    def __enter__(self):
        return self
    
    def __exit__(self, *a):
        pass
    
    def __init__(self, kat, *a, **k):
        self.obs_params = {}
        self.nd_params = {}
        self.ants = kat.ants
        kat.ants.set_sampling_strategy("lock", "event")
        self._cam_ = kat
        self.dry_run = kat.dry_run
        self.telstate = self
    
    def add(self, *a, **k): # For telstate
        pass
    
    def capture_start(self): # Start capturing data to HDF5 file. Ignored!
        pass
    
    def label(self, *a, **k): # Add timestamped label to HDF5 file. Ignored!
        pass
    
    def standard_setup(self, *a, **kwargs): # Like start_hacked_session#hacked_setup(), bust the basics that apply to dishes 
        if (not self.dry_run):
            # Ensure the Ku-band signal generator matches the center frequency of the subarray
            match_ku_siggen_freq(self._cam_)
            # NB: the following must be called after `standard_setup()` because for MKE Dishes that causes a "major state transition"
            # during which the ACU resets some things which we are trying to hack around.
            temp_hack_SetupPointingCorrections(self._cam_, allow_tiltcorrections=not kwargs.get("no_tiltcorrections", False))
    
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


__tilt_corr_allowed__ = True

def temp_hack_SetupPointingCorrections(cam, allow_tiltcorrections=True):
    """ Temporary hack to disable the MKE ACU static pointing corrections and set tilt corrections according
        to the default rules.
        NB: at present the state gets automatically re-enabled by the ACU when there's a transition out
        of STANDBY (ESTOP/STOW/STOP) to OPERATE - so additional hacks must be added to some observation scripts.
        
        MPI/OHB have been requested to make the "disabling" a latching configuration setting in the ACU.
        
        RE static corrections: this should always be disabled when CAM DishProxy is used
        RE tilt corrections: default rules applied as per `hack_SetPointingCorrections()`
        
        @param allow_tiltcorrections: if False, disables tilt corrections and ensures that all attempts to enable
            tilt corrections (without 'force') for the duration of the session will be ignored (default True).
    """
    global __tilt_corr_allowed__
    # Change the default behaviour of hack_SetPointingCorrections
    __tilt_corr_allowed__ = allow_tiltcorrections
    
    # Find the antennas that expose this functionality:
    d_ants = hack_SetPointingCorrections(cam.ants, spem_enabled=False)
    # Now ensure those antennas have transitioned to the "Operating" mode, by POINT
    for ant in cam.ants:
        if (ant.name in d_ants):
            az, el = (ant.sensor.dsh_achievedPointing_1.get_value(), ant.sensor.dsh_achievedPointing_2.get_value())
            ant.req.target_azel(az+0.01, el+0.01) # Must be different or else the proxy doesn't propagate this to the ACU?
            ant.req.mode("POINT")
    time.sleep(5)
    # Now disable ACU SPEM corrections (for as long as it only uses POINT mode for controlled motion)
    hack_SetPointingCorrections(cam.ants, spem_enabled=False)


def hack_SetPointingCorrections(ants, spem_enabled=False, tilt_enabled=True, temp_enabled=False, force=False):
    """ Enable/Disable SPEM and or Tilt corrections on the specified dishes.
        This command is currently not exposed in enough detail by the CAM proxy, so use the tango interface directly.
        
        Default rules for tilt corrections: disable tilt corrections on all (except MKE121) because they are not
         calibrated / implemented correctly as of 16/05/2025.
        
        @param ants: list of instances of CAM Receptor/Dish Proxy.
        @param spem_enabled: ACU's internal Static Pointing Error Model corrections (default False)
        @param tilt_enabled: ACU's internal inclinometer-based corrections (default True)
        @param temp_enabled: ACU's internal ambient temperature-based corrections (default False)
        @param force: if True then force tilt to the specified value, else will disable tilt for all except s0121 (default False)
        @return: list of names of ants where the SPEM corrections were changed.
    """
    global __tilt_corr_allowed__
    mod_spem, mod_tilt, force_tilt = [], [], []
    # Mapping to match https://github.com/ska-sa/katcamconfig/pull/955/files
    d_numbers = {"s0000":64, "s0121":65, "s0119":66, "s0118":67, "s0107":68, "s0060":69, "s0105":70, "s0110":71, # MKE
                 "s0115":72, "s0117":73, "s0116":74, "s0017":75, "s0018":76, "s0020":77, "s0023":78, # MKE
                 "s0063":90, "s0001":91, "s0100":92, "s0036":93} # SKA - TBC
    d_tilt_OK = ["s0121"] # These tilt installations believed to be OK
    for a in ants:
        if (a.name in d_numbers.keys()):
            lmc_root = "10.96.%d.100:10000/mid_dsh_%s"%(d_numbers[a.name], a.name[1:])
            dsm = tango.DeviceProxy(lmc_root+'/lmc/ds_manager')
            dsm.tempPointCorrEnabled = temp_enabled
            dsm.staticPointCorrEnabled = spem_enabled
            mod_spem.append(a.name)
            if (not __tilt_corr_allowed__) or ((a.name not in d_tilt_OK) and not force): # Default rule for disabling
                dsm.tiltPointCorrEnabled = False
                force_tilt.append(a.name)
            else:
                dsm.tiltPointCorrEnabled = tilt_enabled
                mod_tilt.append(a.name)
    if (len(mod_spem) > 0):
        user_logger.info("APPLIED HACK: Temperature Corrections %s on %s" % ("Enabled" if temp_enabled else "Disabled", ",".join(mod_spem)))
        user_logger.info("APPLIED HACK: SPEM Corrections %s on %s" % ("Enabled" if spem_enabled else "Disabled", ",".join(mod_spem)))
    if (len(force_tilt) > 0):
        user_logger.info("APPLIED HACK: Tilt Corrections Disabled on %s" % (",".join(force_tilt)))
    if (len(mod_tilt) > 0):
        user_logger.info("APPLIED HACK: Tilt Corrections %s on %s" % ("Enabled" if tilt_enabled else "Disabled", ",".join(mod_tilt)))
    return mod_spem


def cycle_feedindexer(cam, cycle, switch_indexer_every_nth_cycle):
    """ Switch the indexer out & back, if requested. This implementation currently only works for MKE/SKA Dish!
        
        @param cycle: the number of the current cycle, assumed to start from 0.
        @param switch_indexer_every_nth_cycle: indexer will be moved on cycle==0 and thereafter at intervals as per this argument. 
    """
    # Set up parameters to use to switch the indexer between rasters, if requested
    index0, indexer_sequence = None, [] # Arguments to pass to dsh_SetIndexerPosition()
    if (switch_indexer_every_nth_cycle > 0):
        # TODO: This mapping is for MKE - TBC for SKA Dishes
        indexer_positions, indices = ["B1","B5c","B2"], [1,7,2] # Arranged in angle sequence, only the positions relevant to DVS listed
        for ant in cam.ants:
            try:
                index0 = indexer_positions.index(ant.sensor.dsm_indexerPosition.get_value())
                break
            except:
                pass
        assert (index0 is not None), "Unable to query indexer status, cannot perform indexer switching as required!"
        indexer_sequence = [indices[min(max(0,index0+i),len(indices)-1)] for i in [-1,1]]
        index0 = indices[index0]
        try: # Remove "switch to self" end case
            indexer_sequence.remove(index0)
        except:
            pass
    
    # Execute the switch for the current cycle
    if (len(indexer_sequence) > 0) and (cycle%switch_indexer_every_nth_cycle == 0): # Switching alternates between indexer_sequence[0] and [1]
        wrapped_cycle = cycle % (2*switch_indexer_every_nth_cycle)
        i_cycle = -int(wrapped_cycle/switch_indexer_every_nth_cycle) # Alternates between 0 & -1
        try:
            index = indexer_sequence[i_cycle]
            user_logger.info("Switching Feed Indexer to index %s"%index)
            if not cam.dry_run:
                cam.ants.req.dsh_SetIndexerPosition(index)
            time.sleep(30)
        finally: # Switch back to the nominal position. This also ensures that we "clean up"
            user_logger.info("Switching Feed Indexer back to index %s"%index0)
            if not cam.dry_run:
                cam.ants.req.dsh_SetIndexerPosition(index0)
            time.sleep(30) # TODO: rather "wait on dsm_indexerAxisState==PARKED" to avoid possible errors if indexer is slow


def start_nd_switching(sub, n_on, n_off, T_start='now'):
    """ Start Digitiser-level synchronous Noise Diode cycling.
        
        @param n_on, n_off: describe integer number of SDP dump intervals
        @param T_start: time when the digitisers should trigger the start of the switching cycles.
        @return: (T_start[sec], on_fraction, period[sec])
    """
    on_fraction = float(n_on)/(n_on+n_off)
    T_start = time.time() if (T_start == 'now') else T_start
    
    try: # 'sub' is an active session
        cbf, sdp, ants = sub.cbf, sub.sdp, sub.ants
    except: # 'sub' is a cam control environment. TODO: we assume subarray 1!
        cbf, sdp, ants = sub.cbf_1, sub.sdp_1, sub.ants
    
    cbf_dt = cbf.sensors.wide_baseline_correlation_products_int_time.get_value()
    sdp_dt = cbf_dt * np.round(1/sdp.sensors.dump_rate.get_value() / cbf_dt + 0.5) # SDP dump rate is not accurate
    time.sleep(2*sdp_dt) # Guard against getting old timestamp immediately after capture_start
    T0 = sdp.sensors.spmc_array_1_wide_0_ingest_sdp_l0_1_last_dump_timestamp.get_value()
    ND_LEAD_TIME = np.round(10/sdp_dt+0.5) * sdp_dt # At least 10 seconds - needed because of sensor update rates etc
    T_start = max(T_start, T0 + ND_LEAD_TIME) # Specified time or on the next(+1) dump boundary
    ants.req.dig_noise_source(T_start, on_fraction, sdp_dt*(n_on+n_off))
    return T_start, on_fraction, sdp_dt*(n_on+n_off)


def start_hacked_session(cam, **kwargs):
    """ Start a capture session and apply standard hacks as required for proper operation of the DVS system.
        1. `standard_setup()` checks & updates Ku-band siggen frequency
        2. `standard_setup()` disables pointing corrections for MKE Dish ACU's
        3. `capture_start()` handles hardware-level noise diode switching
        4. `capture_start()`, `label()` & `add()` are skipped if no_capture option is True
        
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
            temp_hack_SetupPointingCorrections(session._cam_, allow_tiltcorrections=not kwargs.get("no_tiltcorrections", False))
        
        return result
    session.standard_setup = hacked_setup
    
    # Hack the "standard capture start" function
    session._capture_start_ = session.capture_start
    def hacked_capture_start(*a, **k):
        result = session._capture_start_(*a, **k)
        # Start noise diode switching, if requested
        nd_params = kwargs.get("nd_params", dict(diode=None))
        if (nd_params['diode'] == "switching"):
            n_on, n_off = int(nd_params['on']), int(nd_params['off'])
            if (not session._cam_.dry_run):
                params = start_nd_switching(session, n_on, n_off)
            else:
                params = ('now', n_on, n_off)
            user_logger.info("Started digitiser-level noise diode switching: " + str(params))
        return result
    session.capture_start = hacked_capture_start
    
    if kwargs.get('no_capture', False): # Hack the functions that don't properly handle no_capture
        session.add = lambda *a, **k: None # For telstate
        session.capture_start = lambda *a, **k: None # Start capturing data to HDF5 file. Ignored!
        session.label = lambda *a, **k: None # Add timestamped label to HDF5 file. Ignored!
    
    # Hack the "standard end" function
    session._end_ = session.end
    def hacked_end(*a, **k):
        # Stop noise diode switching, if requested
        nd_params = kwargs.get("nd_params", dict(diode=None))
        if (nd_params['diode'] == "switching"):
            if (not session._cam_.dry_run):
                session.ants.req.dig_noise_source('now', 'off')
            user_logger.info("Stopped digitiser-level noise diode switching.")
        # Shut down capturing stream, except for no-capture session
        if session._cam_.dry_run or (not kwargs.get('no_capture', False)):
            session._end_(*a, **k)
        else:
            user_logger.info('DONE')
            user_logger.info('Ended data capturing session with experiment '
                             'ID %s', session.experiment_id)
            user_logger.info('==========================')
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


def plan_targets(catalogue, T_start, t_observe, dAdt=1.8, cluster_radius=0, antenna=None, el_limit_deg=20):
    """ Generates a "nearest-neighbour" sequence of targets to observe, starting at the specified time.
        This does not consider behaviour around the azimuth wrap zone.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_start: UTC timestamp, seconds since epoch [sec].
        @param t_observe: duration of an observation per target [sec]
        @param dAdt: angular rate when slewing (default 1.8) [deg/sec]
        @param cluster_radius: returned targets are clustered with this great circle dimension (default 0) [degrees]
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
    next_tgt, dGC = available.closest_to(start_pos, T, antenna) if (len(available.targets) > 0) else (None, 0)
    while (next_tgt is not None):
        # Slew to next
        T += dGC * dAdt
        # Observe this current target
        next_tgt.antenna = antenna
        T += t_observe
        done.append(next_tgt)
        todo.pop(todo.index(next_tgt))
        
        if cluster_radius and (cluster_radius > 0): # Find targets in the current cluster and visit them in turn
            tgt0 = next_tgt
            cluster = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna,
                                                      dist_limit_deg=(0,cluster_radius), proximity_targets=tgt0)
            # Find next nearest in the cluster
            next_tgt, dGC = cluster.closest_to(done[-1], T, antenna)
            while next_tgt:
                # Slew to next
                T += dGC * dAdt
                # Observe
                next_tgt.antenna = antenna
                T += t_observe
                done.append(next_tgt)
                todo.pop(todo.index(next_tgt))
                # Find next nearest in the cluster
                cluster.remove(next_tgt.name)
                next_tgt, dGC = cluster.closest_to(done[-1], T, antenna)
        
        # Done with cluster (if any), now continue to nearest remaining & visible neighbour
        available = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
        next_tgt, dGC = available.closest_to(done[-1], T, antenna)
    
    return done, (T-T_start)
