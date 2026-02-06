"""
    Some general-purpose functions.
    
    @author: aph@sarao.ac.za
"""
import katdal
import logging; logging.disable(logging.DEBUG) # Otherwise katdal is unbearable
import os, subprocess, shutil, pickle
import numpy as np
from analysis import katselib as ksl
from analysis import katsemat as ksm
from analysis import __res__


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall

load_rfi_static_mask = ksl.load_frequency_mask

# HACK 02/2025 Usually kat-flap is contacted first for sensor data; it is offline (12/2024 - ...), so to prevent timeouts:
for k in ksl.SENSOR_PORTALS.keys():
    if ("kat-flap" in k):
        ksl.SENSOR_PORTALS[k] = np.inf


def open_dataset(dataset, ref_ant='', hackedL=False, ant_rx_override=None, cache_root=None, **kwargs):
    """ Use this to open a dataset recorded with DVS, instead of katdal.open(), for the following reasons:
        1) easily accommodate the "hacked L-band digitiser"
        2) override the antennas' "receiver" serial numbers, which are some times set incorrectly with DVS "slip-ups"
        3) work-around for the CAM activity time_offset issue that affects SKA-type Dishes
        4) supports local caching of the dataset. 
        
        Use this either in "function call" form, or as a "context manager". The context manager automatically
        deletes the local cache, if that is used.
        
        Use as "function call"
        
            ds = open_dataset(cbid, ..., cache_root="./l1_data")
            ...
            ds.del_cache() # Clean up explicitly, in case you used 'cache_root'
        
        Use as "context manager"
        
            with  open_dataset(cbid, ...).cache_manager as ds:
                ...
        
        @param dataset: the URL of the katdal dataset to open (or an already opened dataset to modify in-situ).
                  If this is an integer (or string representation of an integer) it is converted using `cbid2url`.
        @param ref_ant: the name of reference antenna, used to partition data set into scans (essential if you
                  are interpreting the data for SKA-type Dishes, because their activities have a time offset from MeerKAT).
        @param hackedL: True if the dataset was generated with the hacked L-band digitiser i.e. sampled in 1st Nyquist zone.
        @param ant_rx_override: {ant:rx_serial} to override (default None)
        @param cache_root: None, or the folder to download the dataset to, until the cache is deleted (default None).
                           Note: will be ignored if 'dataset' is a URL. 
        @param kwargs: passed to katdal.open()
        @return: the opened dataset. """
    __del_cache__ = lambda: None # Default function, overridden below
    if (cache_root): # Try to download
        try:
            cbid = int(str(dataset))
            cache_fn = f"{cache_root}/{cbid}/{cbid}_sdp_l0.full.rdb"
            if not os.path.exists(cache_fn):
                # err = os.system(f"python {os.path.dirname(__file__)}/../bin/mvf_download.py {cbid2url(cbid)} {cache_root}")
                # The above one-liner is (currently) much too verbose, so use the following:
                proc = subprocess.Popen(["python", os.path.dirname(__file__)+"/../bin/mvf_download.py", cbid2url(cbid), cache_root],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Convert the verbose output to a more efficient form
                while proc.poll() is None:
                    out = proc.stdout.read(30).decode() # Arb. 30 chars - responsiveness vs excessiveness
                    perc = [p for p in out.split(",") if ('%' in p) and (3 <= len(p) <= 5)] # Extracted percentages
                    if perc:
                        print("".join(perc), end="")
                print()
                err = proc.returncode
                assert (err == 0), "mvf_download.py failed with error code %s"%err
            dataset = cache_fn
            # A function to remove the cached files
            __del_cache__ = lambda: [shutil.rmtree(f"{cache_root}/{cbid}", ignore_errors=True),
                                     shutil.rmtree(f"{cache_root}/{cbid}-sdp-l0", ignore_errors=True)]
        except ValueError: # Not a "CaptureBlockId", so don't cache
            pass
        except Exception as e:
            print("WARNING: unable to cache the dataset locally: ", e)
    
    # Convert dataset from an integer to a URL, if necessary
    try:
        if (int(str(dataset)) > 0): # Raises an exception if not an integer
            dataset = cbid2url(dataset)
    except ValueError:
        pass
    dsname = dataset if isinstance(dataset, str) else dataset.url
    cbid = int(dsname.split("/")[-2]) # ASSUMES "*cbid/cbid_l0_.ext"
    
    # Take care of activity boundary time mismatches
    try:
        _time_offset = katdal.visdatav4.SENSOR_PROPS['*activity'].get('time_offset', 0)
        if (1738674000 < cbid): # From 4/02/2025 ~13h00 UTC, the Receptor & Dish proxies have the same lead time offset
            t_o = 5 # https://github.com/ska-sa/katmisc/blob/release/karoocamv30/katmisc/app/dish_proxy/katproxy/proxy/mke_dsh_model.py#L39
            if (1760000000 < cbid) and ref_ant.startswith('s'): # TODO: SkaoDishProxy seems to require this - get it back to 5sec!
                t_o = 6.2
            katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = t_o
        elif ref_ant:
            if ref_ant.startswith("s"):
                # https://github.com/ska-sa/katproxy/blob/master/katproxy/proxy/ska_mpi_dsh_model.py#L34
                if ("/159" in dsname): t_o = 18 # 06/2020 - 09/2020
                elif ("/162" in dsname) or ("/163" in dsname): t_o = 10 # 06/2021 - 12/2021
                else: t_o = 5 # https://github.com/ska-sa/katproxy/pull/702/files
                katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = t_o
            # elif ref_ant.startswith("m"): # Taken care of by default?
            #     katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = 1.2 # https://github.com/ska-sa/katproxy/blob/master/katproxy/proxy/base_receptor_model.py#L46
        
        dataset = katdal.open(dataset, **kwargs) if isinstance(dataset,str) else dataset
    finally: # It is "baked in" when katdal.open() completes
        katdal.visdatav4.SENSOR_PROPS['*activity']['time_offset'] = _time_offset
    
    if hackedL: # Change centre freq and flip channel/frequency mapping
        for spw in dataset.spectral_windows:
            spw.__init__(856e6/2., spw.channel_width, spw.num_chans, spw.product, sideband=-1)
        dataset.select(reset="F")
    
    if (ant_rx_override is not None): # Change receiver serial numbers
        for ant in dataset.ants:
            dataset.receivers[ant.name] = ant_rx_override.get(ant.name, dataset.receivers[ant.name])
    

    # An explicit function to clean-up in case it's been cached locally
    dataset.del_cache = __del_cache__
    # A context manager to automate the cache clean up
    class ctx_wrapper(object):
        def __init__(self, dataset):
            self.dataset = dataset
        def __enter__(self):
            return self.dataset
        def __exit__(self, except_type, except_val, except_tb):
            self.dataset.del_cache()
    dataset.cache_manager = ctx_wrapper(dataset)
    
    return dataset


def get_fft_shift_and_gains(dataset, channel=123, verbose=False):
    """ Determines the RF attenuation, F-engine "fft_shift" and "equalisation gains" that were applied during
        the observation that generated the dataset.
        
        @param dataset: a katdal.Dataset object
        @param verbose: True to print out the results (default False).
        @return: (fft_shift, [eq_gains_scan0, ... eq_gains_scanN], atten) - eq_gains and atten as dictionaries indexed by antenna name & polarisation.
    """
    # in v4, fft_shift sensor values are stored per timestamp, but these never change
    try: # v4 after 2019?
        fft_shift = dataset.sensor['wide_antenna_channelised_voltage_fft_shift'][0]
    except:
        try: # v4 up to 2019?
            fft_shift = dataset.sensor['i0_antenna_channelised_voltage_fft_shift'][0]
        except: # v3 -- but these are always just the defaults?
            try:
                fft_shift = dataset.file['TelescopeState'].attrs['cbf_fft_shift']
            except: # < v3
                fft_shift = "UNKNOWN"
    
    # Load requant gains from metadata. for timestamp[0] of each scan, assuming it never changes during a scan
    eq_gains = []
    for scan in dataset.scans():
        if (len(dataset.timestamps) < 2): continue # Some buggy observations have such tracks -- applied in troubleshoot() too
        eq_gains.append(dict(zip(["%sh"%a.name for a in dataset.ants]+["%sv"%a.name for a in dataset.ants],
                                 [-1 for a in dataset.ants]+[-1 for a in dataset.ants])))
        for port in eq_gains[-1].keys():
            try: # v4 after 2019?
                eq_gains[-1][port] = dataset.sensor['wide_antenna_channelised_voltage_%s_eq'%port][0][channel]
            except:
                try: # v4 up to 2019?
                    eq_gains[-1][port] = dataset.sensor['i0_antenna_channelised_voltage_%s_eq'%port][0][channel]
                except: # v3 -- but these are always just the defaults?
                    ports = [k for k in dataset.sensor.keys() if "cbf_eq_coef" in k]
                    eq_gains[-1][port] = str(pickle.loads(dataset.file[ports[0]][0][1]))
    
    band = "UNKNOWN"
    atten = {} # Attenuation is not stored in the dataset, need to get it from the sensor database
    # Find the sensor portal, for sensors that are not in the dataset
    ant = dataset.ants[0]
    for store in ['portal.mkat-rts.karoo.kat.ac.za', 'portal.mkat.karoo.kat.ac.za']:
        try:
            dataset.sensor.store = store
            dataset.sensor.get(ant.name+"_state")[:]    
        except:
            dataset.sensor.store = None
        else:
            break
    if dataset.sensor.store:
        subarray = "subarray_%d" % (dataset.sensor["Observation/subarray_index"][0] + 1)
        band = dataset.sensor.get(subarray+"_band")[0]
        try:
            atten_sensor = {"u":"dig_u_band_rfcu_%spol_attenuation",
                            "l":"dig_l_band_rfcu_%spol_attenuation",
                            "s":"rsc_rxs_signalprocessors_sp%s_attenuation",
                            "x":"dig_x_band_rfcu_%spol_attenuation"}[band]
            atten_hv = ["01","02"] if (band=="s") else ["h","v"]
            for ant in dataset.ants:
                for pol in atten_hv:
                    atten[ant.name+pol] = dataset.sensor.get(ant.name+"_"+atten_sensor%pol)[0]
        except Exception as e:
            print("WARNING: Encountered an error while retrieving attenuation values - continuing.", type(e), e)
        
    if verbose:
        print("Band: %s" % band)
        print("CBF FFT shift:%s %s" % (fft_shift, "" if isinstance(fft_shift,str) else bin(fft_shift)))
        print("CBF requantization (equalization) gains:\n%s" % eq_gains)
        print("RF attenuation:\n%s" % atten)
    
    return fft_shift, eq_gains, atten


def remove_RFI(freq, x0, x1, rfi_mask, flag_thresh=0.2, smoothing=0, axis=0):
    """ Remove RFI and smooth over frequency bins. The RFI mask is created by first applying the given static mask
        to create smooth interpolated reference curves; the RFI mask is then determined as all values which deviate
        by more than the specified threshold (fraction) from the reference curves.
    
        @param freq: list of frequency bins
        @param x0,x1: 2-D arrays of values, to be transformed
        @param rfi_mask: identifier for the static mask, passed to `load_rfi_static_mask()`. 
        @param flag_thresh: the threshold to use to create the RFI mask that is applied, as a fraction of the smooth
                 interpolated reference curves (default 0.2 i.e. values exceeding +/-20% of the reference curves are removed).
        @param smoothing: window length over freq axis to apply to the data after flagging - only effective if > 1 (default 0)
        @param axis: identifies which axis of x0 & x1 corresponds to the `freq` axis (default 0)
        @return (filt_smooth_x0, filt_smooth_x1) """
    if isinstance(rfi_mask, str):
        rfi_mask = ksl.load_frequency_mask(rfi_mask, freq)
    sm_x0, sm_x1 = [], []
    for msd,sms in ([x0 if axis==0 else np.transpose(x0),sm_x0],[x1 if axis==0 else np.transpose(x1),sm_x1]):
        for m in msd:
            _m = ksm.interp(freq, freq[~rfi_mask], m[~rfi_mask], 'linear')
            sm = ksm.smooth(_m, N=smoothing, padlen=len(freq)//2, padtype='even')
            flags = np.argwhere(np.abs(m/sm-1) > flag_thresh)
            m[flags] = np.nan
            sms.append(m if (smoothing <= 1) else ksm.smooth(m, N=smoothing))
    return np.array(sm_x0 if axis==0 else np.transpose(sm_x0)), np.array(sm_x1 if axis==0 else np.transpose(sm_x1))


def load_dsc_dataset(fn, delimiter=";", header_len=2):
    """ Load a datset that was recorded using OHB's datalogging recording facility.
    
        @param fn: the filename to the CSV file.
        @return: {column_name:column_values} """
    d = np.genfromtxt(fn, delimiter=delimiter, names=True,deletechars='', dtype=None, skip_header=header_len,
                      converters={'Date/Time':lambda s:np.datetime64(s[:-1].replace("T"," "), 's')})
    return d


def calc_FIangle_adjustment(delta_Yf=None, delta_P4=None):
    """ Calculate adjustments to SKA Dish pointing model and FI angle, given a Yf offset from hologreport.

        Exactly one of delta_* must be given!
        @param delta_Yf: Y_f translation in SKA Dish coordinate system [mm]
        @param delta_P4: P4 in katpoint model [deg]
        @return (P4_adjust_angle, FI_adjust_angle) [deg] to be added to the current FI angle """
    BDF=0.894; R_FI=1400; F_eq=8507 # [], mm, mm for SKA Dish

    if (delta_P4 is not None): # Change in P4 pointing term
        # Correction is in opposite sense of the model coefficient
        dP4 = - delta_P4
        dFI_angle = dP4 / (1 - BDF*R_FI/F_eq)
        _delta_Yf = np.tan(dFI_angle*np.pi/180) * R_FI
    else: # Change in Feed effective in-plane translation
        # If feed is pointed right looking at SR (Yf>0), correction should decrease FI angle (ICD)
        dFI_angle = np.atan2(- delta_Yf, R_FI) * 180/np.pi
        dP4 = dFI_angle * (1 - BDF*R_FI/F_eq) # Translation AND rotation if FI angle is adjusted
        _delta_P4 = BDF * np.arctan2(delta_Yf, F_eq) * 180/np.pi # Just in-plane translation without rotation
    return (dP4, dFI_angle)


def add_datalog_entry(ant, dataset, description, center_freq, notes, env_conditions, test_procedures=[],
                      replace_all=False):
    """ Update the log messages against a dataset as used for a specific antenna.
        
        @param ant: the ID of the antenna that the log message is relevant for, e.g. "SKA119".
        @param dataset: a specific CaptureBlockId
        @param description..env_conditions: column entries as text or numbers.
        @param test_procedures: list of names of test procedures where this dataset has been used.
        @param replace_all: True to remove all existing log messages for this dataset & antenna before adding the new entry (default False). """
    assert (isinstance(dataset, int) or isinstance(dataset, str)), "Dataset may not be unspecified!"
    
    gs = ksl.GSheet("1RKre2WGCRxG_DzcmKACbtXrC195Js4YRGcqq6IKPcfw", __res__.dvs_log_auth_token)

    if replace_all:
        # Find the rows to clear
        values = gs[f"{ant}!A2:Z"] # Skip the headings and expect no more than 26 columns
        dataset = str(dataset)
        rows = [2+i for i,r in enumerate(values) if (str(r[0]).startswith(dataset))]
        # Clear just those rows
        gs.clear([f"{ant}!A{r}:Z" for r in rows])
    
    entry = [dataset, description, center_freq, notes, env_conditions] + test_procedures
    gs.append(f"{ant}!A2:Z", [entry])
    

def get_datalog_entries(ant, dataset="*"):
    """ Retrieve all log messages against a dataset as used for a specific antenna.
        
        @param ant: the ID of the antenna that the log message is relevant for, e.g. "SKA119".
        @param dataset: a specific CaptureBlockId or "*" for all
        @return: (headings, values) of entries have been logged against the dataset """
    gs = ksl.GSheet("1RKre2WGCRxG_DzcmKACbtXrC195Js4YRGcqq6IKPcfw", __res__.dvs_log_auth_token)
    
    values = gs[f"{ant}!A1:Z"] # Expect no more than 26 columns
    headings, values = values[0], values[1:]
    
    if (dataset != "*"): # Select the rows to return
        dataset = str(dataset)
        selected = [r for r in values if (str(r[0]).startswith(dataset))]
        values = selected
    
    return headings, values


def ls_archive(query, min_duration=0, min_night_duration=0, vis_status=False,
               fields=['CaptureBlockId','StartTime','CenterFrequency','InstructionSet'], field_len=60):
    """ Query the archive and ensure an entry is added to the DVS data registry.
    """ + ksl.ls_archive.__doc__
    
    recs = ksl.ls_archive(query, min_duration, min_night_duration, vis_status=vis_status, fields=fields, field_len=field_len)
    
    # TODO: check & add entries to the data registry spreadsheet
    
    return recs



if __name__ == "__main__":
    # add_datalog_entry("SKA119", 123456, "description", "center_freq", "notes", "env_conditions")
    h, v = get_datalog_entries("SKA119", dataset="*")
    print(h)
    for r in v:
        print(r)
    # add_datalog_entry("SKA119", 1234567890, "description2", "center_freq", "notes", "env_conditions", ["TP 1"])
    # h, v = get_datalog_entries("SKA119", dataset="123456")
    # print(h)
    # for r in v:
    #     print(r)
    #
    # add_datalog_entry("SKA119", 1234567890, "description3", -99, "notes", "supper", ["TP 1", "TP 2"],
    #                   replace_all=True)
    