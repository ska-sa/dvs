"""
    Some general-purpose functions.
    
    @author: aph@sarao.ac.za
"""
import katdal
import logging; logging.disable(logging.DEBUG) # Otherwise katdal is unbearable


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall


def open_dataset(dataset, ref_ant='', hackedL=False, ant_rx_override=None, verbose=False, **kwargs):
    """ Use this to open a dataset recorded with DVS, instead of katdal.open(), for the following reasons:
        1) easily accommodate the "hacked L-band digitiser"
        2) override the antennas' "receiver" serial numbers, which are some times set incorrectly with DVS "slip-ups"
        3) work-around for the CAM activity time_offset issue that affects SKA-type Dishes.
        
        Use like 'ds = open_dataset(URL, ...)'
        
        @param dataset: the URL of the katdal dataset to open (or an already opened dataset to modify in-situ).
                  If this is an integer (or string representation of an integer) it is converted using `cbid2url`.
        @param ref_ant: the name of reference antenna, used to partition data set into scans (essential if you
                  are interpreting the data for SKA-type Dishes, because their activities have a time offset from MeerKAT).
        @param hackedL: True if the dataset was generated with the hacked L-band digitiser i.e. sampled in 1st Nyquist zone.
        @param ant_rx_override: {ant:rx_serial} to override (default None)
        @param kwargs: passed to katdal.open()
        @return: the opened dataset. """
    # Convert dataset from an integer to a URL, if necessary
    try:
        if (int(str(dataset)) > 0): # Raises an exception if not an integer
            dataset = cbid2url(dataset)
    except ValueError:
        pass
    
    # Take care of activity boundary time mismatches
    try:
        _time_offset = katdal.visdatav4.SENSOR_PROPS['*activity'].get('time_offset', 0)
        if ref_ant:
            if ref_ant.startswith("s"):
                # https://github.com/ska-sa/katproxy/blob/master/katproxy/proxy/ska_mpi_dsh_model.py#L34
                dsname = dataset if isinstance(dataset, str) else dataset.name
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
    
    if verbose:
        print(dataset)
        print(dataset.receivers)
    
    return dataset
