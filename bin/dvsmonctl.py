'''
    Automate standard tasks when a new interactive monctl session is started for the DVS system
    1. Import the CAM console framework
    2. Tool to reset the link to the LMC e.g. after a crash.
    
    NB: This file is to be deployed on the monctl.mkat-rts server.
    
    Use as follows:
    
        import dvsmonctl; configure_cam('all')
        
        ...
        
        dvsmonctl.reset_LMC(cam.s0002)
        
        ...
        
        cat = geo_cat(cam) # Download to /home/kat/usersnfs/aph/geo.txt by default

    @author: aph@sarao.ac.za
'''
import katuilib, time
# configure_cam('all') # TODO: must run this manually in interactive console


def reset_ACU(cam_ant):
    """ Let LMC take authority of ACU & clear old tasks etc. Necessary e.g. after ESTOP, manual control or OHB GUI /SCU work.
        This should be a temporary hack - if not sorted out by 01/03/2025 follow up with LMC team!
    """
    import tango
    dsm_addr = cam_ant.sensors.dsm_tango_address.get_value()
    dsm = tango.DeviceProxy(dsm_addr)
    dsm.RequestAuthority(); time.sleep(1)
    dsm.AckInterlock(); time.sleep(5)
    dsm.ClearLatchedErrors(); time.sleep(1)
    dsm.ClearOldTasks(); time.sleep(1)
    dsm.SetStandbyFPMode(); time.sleep(5)
    dsm.ResetTrackTableBuffer(); time.sleep(1)
    dsm.ResetTrackTable(); time.sleep(1)
    dsm.ResetDishMode() # This is necessary to get rid of (some) occurrences of FORBIDDEN


def match_ku_siggen_freq(cam, override=False):
    """ The frequency of the Ku-band reference LO signal generator must be changed manually,
        the subarray's "x band" center frequency only updates sensors and metadata.
        See https://skaafrica.atlassian.net/browse/MKT-50 
    
        A change is only made if all active "x band" subarrays have the same center frequency.
    """
    active_subs = [sa for sa in [cam.subarray_1,cam.subarray_2,cam.subarray_3,cam.subarray_4] if (sa.sensors.state.get_value()=='active')]
    xband_subs = [sa for sa in active_subs if (sa.sensors.band.get_value()=='x')]
    xband_fc = [sa.sensors.requested_rx_centre_frequency.get_value() for sa in xband_subs]
    if (len(set(xband_fc)) == 1) or override:
        fc = xband_fc[0]
        fLO = (fc -  1712e6*3/4.) / 100. # Hz
        current_fLO = cam.anc.sensors.siggen_ku_frequency.get_value()
        if (abs(current_fLO-fLO) > 0.01):
            cam.anc.req.siggen_ku_frequency(fLO)
            print("INFO: UPDATED Ku reference LO from %g to %g" % (current_fLO, fLO))
    elif (len(set(xband_fc)) > 1):
        print("WARNING: MANUAL OVERRIDE REQUIRED. Multiple x band subarrays are active at present with different center frequencies.") 


def geo_cat(cam, savefn="/home/kat/usersnfs/aph/geo.txt", download="geo,intelsat"):
    """ Download the current CelesTrak TLEs and combine it into one catalogue file.
        Then instantiates a katpoint catalogue from this file, using the current session's reference antenna.
        
        @param savefn: the name of the file that will contain the downloaded TLEs (all concatenated in sequence)
        @param download: the TLE groups to download as comma-separated list (degault "geo,intelsat").
        @return: the katpoint.Catalogue 
    """
    import katpoint
    import urllib.request
    
    groups = download.split(",")
    urllib.request.urlretrieve("https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle"%groups[0], savefn)
    for group in groups[1:]:
        urllib.request.urlretrieve("https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle"%group, "/tmp/tle.txt")
        with open(savefn,"at") as geos:
            with open("/tmp/tle.txt", "rt") as temp:
                geos.writelines([line for line in temp])
    
    cat = katpoint.Catalogue(add_specials=False, antenna=cam.sources.antenna)
    cat.add_tle(open(savefn))
    return cat
