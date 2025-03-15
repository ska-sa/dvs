"""
    Some general-purpose functions.
    
    @author: aph@sarao.ac.za
"""
import katdal, os, subprocess, shutil, pickle
import logging; logging.disable(logging.DEBUG) # Otherwise katdal is unbearable
import numpy as np
from analysis import katselib as ksl
from analysis import katsemat as ksm
from analysis import __res__


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall

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
    dsname = dataset if isinstance(dataset, str) else dataset.name
    cbid = int(dsname.split("/")[-2]) # ASSUMES "*cbid/cbid_l0_.ext"
    
    # Take care of activity boundary time mismatches
    try:
        _time_offset = katdal.visdatav4.SENSOR_PROPS['*activity'].get('time_offset', 0)
        if (1738674000 < cbid): # From 4/02/2025 ~13h00 UTC, the Receptor & Dish proxies have the same lead time offset
            t_o = 5 # github link TBD
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


def load_rfi_static_mask(filename, freqs, debug_chunks=0):
    """ Construct a mask either from a pickle file, or a text file with frequency ranges.
        
        @param filename: pickle or text file.
        @param freqs: list of frequencies [Hz] for which a mask must be created.
        @param debug_chunks: for debugging purposes, print out how much bandwidth is removed by the mask if 'freqs' is split into this many chunks.
        @return: boolean mask to match 'freqs' """
    nchans = len(freqs)
    channel_width = abs(freqs[1]-freqs[0])
    try:
        with open(filename, "rb") as pickle_file:
            channel_flags = pickle.load(pickle_file)
        nflags = len(channel_flags)
        if (nchans != nflags):
            print("Warning channel mask (%d) is stretched to fit dataset (%d)!"%(nflags,nchans))
            N = nchans/float(nflags)
            channel_flags = np.repeat(channel_flags, int(N+0.5)) if (N > 1) else channel_flags[::int(1/N)]
        channel_flags = channel_flags[:nchans] # Clip, just in case
    except pickle.UnpicklingError: # Not a pickle file, perhaps a plain text file with frequency ranges in MHz?
        mask_ranges = np.loadtxt(filename, comments='#', delimiter=',')
        channel_flags = np.full((nchans,), False)
        low = freqs - 0.5 * channel_width
        high = freqs + 0.5 * channel_width
        for r in mask_ranges:
            in_range = (low <= r[1]*1e6) & (r[0]*1e6 <= high)
            idx = np.where(in_range)[0]
            channel_flags[idx] = True
    if debug_chunks > 0:
        for chunk in range(debug_chunks):
            freq = slice(chunk*(nchans//debug_chunks),(chunk+1)*(nchans//debug_chunks))
            masked_f = freqs[freq][channel_flags[freq]]
            if (len(masked_f) > 0):
                mBW = len(masked_f)*(freqs[1]-freqs[0])
                print("\tFreq. chunk %d: mask omits %.1fMHz between (%.1f - %.1f)MHz"%(chunk,mBW/1e6,np.min(masked_f)/1e6,np.max(masked_f)/1e6))
            else:
                print("\tFreq. chunk %d: mask omits nothing"%chunk)
    return channel_flags


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
        rfi_mask = load_rfi_static_mask(rfi_mask, freq)
    sm_x0, sm_x1 = [], []
    for msd,sms in ([x0 if axis==0 else np.transpose(x0),sm_x0],[x1 if axis==0 else np.transpose(x1),sm_x1]):
        for m in msd:
            _m = np.array(m, copy=True); _m[rfi_mask] = np.nan
            sm = ksm.smooth(m, N=smoothing, padlen=len(freq)//2, padtype='even')
            flags = np.argwhere(np.abs(m/sm-1) > flag_thresh)
            m[flags] = np.nan
            sms.append(m if (smoothing <= 1) else ksm.smooth(m, N=smoothing))
    return np.array(sm_x0 if axis==0 else np.transpose(sm_x0)), np.array(sm_x1 if axis==0 else np.transpose(sm_x1))


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


    