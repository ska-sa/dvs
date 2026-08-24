"""
    Some general-purpose functions.
    
    @author: aph@sarao.ac.za
"""
import katdal
import logging; logging.disable(logging.DEBUG) # Otherwise katdal is unbearable
import os, subprocess, shutil, pickle
import numpy as np
from analysis import katselib as ksl


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall

load_rfi_static_mask = ksl.load_frequency_mask

# HACK 08/2026 Try to avoid unintended acess of "mkat" portals for DVS work
for k in ksl.SENSOR_PORTALS.keys():
    if (".mkat." in k):
        ksl.SENSOR_PORTALS[k] = np.inf
        print("CAUTION: disabled access to %s - use katselib._reset_SENSOR_PORTALS_() if necessary!"%k)


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
    for _ in dataset.scans():
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
    for store in [dataset.sensor.store, 'portal.mkat-rts.karoo.kat.ac.za', 'portal.mkat.karoo.kat.ac.za']:
        try:
            dataset.sensor.store = store
            dataset.sensor.get(ant.name+"_state")[:]    
        except:
            dataset.sensor.store = None
        else:
            break
    if dataset.sensor.store:
        band = dataset.sensor["Observation/spw"][0].band.lower()
        try:
            atten_sensor = {"u":"dig_u_band_rfcu_%spol_attenuation",
                            "l":"dig_l_band_rfcu_%spol_attenuation",
                            "s":"rsc_rxs_signalprocessors_sp%s_attenuation",
                            "x":"dig_x_band_rfcu_%spol_attenuation"}[band]
            atten_hv = ["01","02"] if (band=="s") else ["h","v"]
            for ant in dataset.ants:
                for pol in atten_hv:
                    atten[ant.name+pol] = float(dataset.sensor.get(ant.name+"_"+atten_sensor%pol)[0])
        except Exception as e:
            print("WARNING: Encountered an error while retrieving attenuation values - continuing.", type(e), e)
        
    if verbose:
        print("Band: %s" % band)
        print("CBF FFT shift:%s %s" % (fft_shift, "" if isinstance(fft_shift,str) else bin(fft_shift)))
        print("CBF requantization (equalization) gains:\n%s" % eq_gains)
        print("RF attenuation:\n%s" % atten)
    
    return fft_shift, eq_gains, atten


def load_dsc_dataset(fn, delimiter=";", header_len=2):
    """ Load a datset that was recorded using OHB's datalogging recording facility.
    
        @param fn: the filename to the CSV file.
        @return: {column_name:column_values} """
    d = np.genfromtxt(fn, delimiter=delimiter, names=True,deletechars='', dtype=None, skip_header=header_len,
                      converters={'Date/Time':lambda s:np.datetime64(s[:-1].replace("T"," "), 's')})
    return d


def save_filterbankfile(outfile, freqs, data_timefreq, data_time=None, time_keys=None, fmt='%2.8f', delimiter=',', headline="", metadata={}):
    """ Saves a CSV file of 2D time,frequency numerical data. Each row represents a consecutive timestamp; time series data may also be
        given, which is then located in the first set of columns.
        
        @param data_timefreq: numerical data, ordered as (time, freq)
        @param data_time: one or more lists of numerical data, ordered as (time, time, ...)
        @param time_keys: the keys that explain the data_time columns, printed in second last line of the header.
        @param headline: the very first line in the header
        @param metadata: key:value pairs to print in the header
    """
    header = headline + "\n"
    header += "\n".join([str(k)+": "+str(v) for k,v in metadata.items()])
    header += "\n"
    
    ff = list(freqs)
    data = data_timefreq
    if (time_keys is not None) and (data_time is not None):
        time_keys = list(np.atleast_1d(time_keys).reshape(-1))
        ff = [np.concat([[np.nan]*len(time_keys), ff], axis=0)]
        header += "Initial %d column(s) represent: %s\n"%(len(time_keys), delimiter.join(time_keys))
        header += "Each of the remaining columns give the power in the matching frequency channel."
        data_time = data_time if (np.shape(data_time)[1]==len(time_keys)) else np.transpose(data_time)
        data = np.concat([data_time, data], axis=1)
    else:
        ff = [ff]
        header += "Each of the columns give the power in the matching frequency channel."
    header += "\nThe first row gives the channel frequencies [Hz], rows following that gives the linear detected power at consecutive sample times @dt."
    
    packed = np.concat([ff, data], axis=0)
    np.savetxt(outfile, packed, fmt=fmt, delimiter=delimiter, header=header)


def calc_FIangle_adjustment(delta_Yf=None, delta_P4=None):
    """ Calculate adjustments to SKA Dish pointing model and FI angle, given a Yf offset from hologreport.

        If both 'delta_XX' are given, only delta_P4 will be used!
        @param delta_Yf: Y_f from hologreport (not ray-traced!) as per SKA Dish coordinate system [mm]
        @param delta_P4: P4 in katpoint model [deg]
        @return (P4_adjust_angle, FI_adjust_angle) [deg] to be added to the current P4 and FI angle """
    BDF=0.894; R_FI=1400; F_eq=8507 # [], mm, mm for SKA-MID
    shape_factor = 0.73 # For SKA-MID

    if (delta_P4 is not None): # Convert P4 pointing term to equivalent translation
        delta_Yf = np.tan(delta_P4*np.pi/180 / BDF) * F_eq

    delta_Yf = delta_Yf*shape_factor
    # Change in Feed effective in-plane translation
    # If feed is pointed right of SR (Yf>0), correction should decrease FI angle (ICD)
    delta_Yf *= -1
    dFI_angle = np.atan2(delta_Yf, R_FI) * 180/np.pi
    dP4 = BDF * np.arctan2(delta_Yf, F_eq) * 180/np.pi # In-plane translation, no tilt
    return (dP4, dFI_angle)
