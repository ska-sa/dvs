"""
    Some general-purpose functions.
    
    @author: aph@sarao.ac.za
"""
import katdal


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall


def hack_dataset(dataset, hackedL=False, ant_rx_override=None, verbose=False, **kwargs):
    """ Typical modifications to datasets that were generated with the DVS.
        Use like 'ds = hack_dataset(katdal.open(URL), ...)'
        
        @param dataset: the katdal dataset to modify in-situ, or a URL to open as a katdal dataset.
        @param hackedL: True if the dataset was generated with the hacked L-band digitiser i.e. sampled in 1st Nyquist zone.
        @param ant_rx_override: {ant:rx_serial} to override (default None)
        @return: the opened dataset. """
    dataset = katdal.open(dataset, **kwargs) if isinstance(dataset,str) else dataset
    
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


def as_hackedL(dataset, **kwargs):
    """ Hacked L-band digitiser is sampling in 1st Nyquist zone, so change centre freq and flip channel/frequency mapping.
        This is a shortcut for hack_dataset(dataset, hackedL=True, ...) """
    return hack_dataset(dataset, hackedL=True, **kwargs)
