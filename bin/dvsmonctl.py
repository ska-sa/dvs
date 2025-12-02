'''
    Automate standard tasks when a new interactive monctl session is started for the DVS system
    1. Import the CAM console framework
    2. Tool to reset the link to the LMC e.g. after a crash.
    
    NB: This file is to be deployed on the monctl.mkat-rts server.
    
    Use as follows:
        configure_cam('all')
        import dvsmonctl
        
        ...
        
        dvsmonctl.reset_LMC(cam.s0002)
        
        ...
        
        cat = tle_cat(cam) # Downloads to /home/kat/usersnfs/aph/tle.txt by default

    @author: aph@sarao.ac.za
'''
import katuilib, katpoint, time
import numpy as np
import urllib.request


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
    dsm.ResetTrackTableBuffer(); time.sleep(1)
    dsm.ResetTrackTable(); time.sleep(1)
    dsm.SetStandbyFPMode()


def toggle_LNAs(*cam_ants, band=2, also_tempctl=False):
    """ Toggle the Band's LNA's ON/OFF.
        @param band: (default 2)
        @param also_tempctl: if True then also turn the temperature control system ON/OFF (default False) """
    assert (band == 2), "TODO: this function is currently only for B2"
    import tango
    for ant in cam_ants:
        spfc_addr = ant.sensors.spfc_tango_address.get_value()
        spfc = tango.DeviceProxy(spfc_addr)
        if (spfc.b2OperatingState != 3): # "OPERATE"
            print("ERROR: Band%d is not in OPERATE - turn it on manually, just to be safe!"%band)
            return
        
        current_on = spfc.b2LnaHPowerState | spfc.b2LnaVPowerState
        if (current_on):
            print("INFO: Band%d amplifiers are currently ON, now turning them OFF"%band)
            spfc.b2LnaHPowerState = False
            spfc.b2LnaVPowerState = False
            if also_tempctl:
                spfc.b2LnaPidPowerState = False
        else: # cam.s0002.req.dsh_SetSPFLnaPowerOn(0) # 0 = the "active band" 
            print("INFO: Band%d amplifiers are currently OFF, now turning them ON"%band)
            spfc.b2LnaHPowerState = True
            spfc.b2LnaVPowerState = True
            if also_tempctl:
                spfc.b2LnaPidPowerState = True


def _ska_tango_(ant, sub, cmd_args, attr_value):
    """ Either perform a command, or set an attribute value, on the SKA-MID tango device.
        @param ant: control object for dish proxy (e.g. cam.s0001)
        @param sub: either 'dsh', 'dsm' or 'spfc'
        @param cmd_args: name of command, or (name, args...) for the command to execute, or None.
        @param attr_value: (name, value) for the attribute, or None
    """
    import subprocess
    addr = "\"%s\"" % eval("ant.sensor.%s_tango_address.get_value()"%sub)
    tango_dp_instr = lambda dp_addr,instr: subprocess.check_output(["ssh","kat@10.97.8.2","python","-c","'import tango; dp=tango.DeviceProxy(%s); %s'"%(dp_addr,instr)], shell=False)
    if (cmd_args is not None):
        cmd_args = np.atleast_1d(cmd_args)
        cmd, args = cmd_args[0], cmd_args[1:]
        cmd_args = cmd + ("()" if (len(args) == 0) else "(%s)"%",".join([str(_) for _ in args]))
        if (cmd_args.lower() == "restartserver()"):
            addr = "tango.DeviceProxy(%s).adm_name()"%addr
        print(tango_dp_instr(addr, "dp."+cmd_args))
    if (attr_value is not None):
        attr_value = np.atleast_1d(attr_value)
        set_value = "" if (len(attr_value) == 1) else "=%s"%attr_value[1]
        print(tango_dp_instr(addr, "dp.%s%s; print(dp.%s)"%(attr_value[0],set_value,attr_value[0])))
def x_dsh(ant, cmd_args=None, attr_value=None):
    """ Either perform a command, or set an attribute value, on the SKA-MID dish-manager.
        @param ant: control object for dish proxy (e.g. cam.s0001)
        @param cmd_args: name of command, or (name, args...) for the command to execute (default None).
        @param attr_value: (name, value) for the attribute (default None)
    """
    _ska_tango_(ant, 'dsh', cmd_args, attr_value)
def x_dsm(ant, cmd_args=None, attr_value=None):
    """ Either perform a command, or set an attribute value, on the SKA-MID ds-manager.
        @param ant: control object for dish proxy (e.g. cam.s0001)
        @param cmd_args: name of command, or (name, args...) for the command to execute (default None).
        @param attr_value: (name, value) for the attribute (default None)
    """
    _ska_tango_(ant, 'dsm', cmd_args, attr_value)


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


def trk(pointables, tgt):
    """ Track the specified target in a "manual control session"
        @param pointables: list of control objects e.g. [cam.ant1, cam.ant2, cam.cbf_1]
        @param tgt: target object or description e.g. cam.sources['3C 279']
    """
    tgt = tgt if isinstance(tgt, str) else tgt.description
    for proxy in pointables:
        proxy.req.target(tgt)
    for proxy in pointables:
        if hasattr(proxy.req, 'mode'):
            proxy.req.mode("POINT")
    # HACK for MKE Dishes as of 08/2025
    for proxy in pointables:
        if hasattr(proxy.req, 'dsm_DisablePointingCorrections'):
            proxy.req.dsm_DisablePointingCorrections()
     

def tle_cat(cam, tags="geo,intelsat", savefn="/home/kat/usersnfs/aph/tle.txt"):
    """ Download the current CelesTrak TLEs and combine it into one catalogue file.
        Then instantiates a katpoint catalogue from this file, using the current session's reference antenna.
        
        @param tags: the TLE groups to download as comma-separated list (default "geo,intelsat").
        @param savefn: the name of the file that will contain the downloaded TLEs (all concatenated in sequence)
        @return: the katpoint.Catalogue 
    """
    groups = tags.split(",")
    urllib.request.urlretrieve("https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle"%groups[0], savefn)
    for group in groups[1:]:
        urllib.request.urlretrieve("https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle"%group, "/tmp/tle.txt")
        with open(savefn,"at") as geos:
            with open("/tmp/tle.txt", "rt") as temp:
                geos.writelines([line for line in temp])
    
    cat = katpoint.Catalogue(add_specials=False, antenna=cam.sources.antenna)
    cat.add_tle(open(savefn))
    return cat

geo_cat = tle_cat # Alias
