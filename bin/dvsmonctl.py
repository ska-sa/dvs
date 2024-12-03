'''
    Automate standard tasks when a new interactive monctl session is started for the DVS system
    1. Import the CAM console framework
    2. Tool to reset the link to the LMC e.g. after a crash.
    
    NB: This file is to be deployed on the monctl.mkat-rts server.
    
    Use as follows:
    
        import dvsmonctl; configure_cam('all')
        
        ...
        
        reset_LMC(cam.s0002)
        

    @author: aph@sarao.ac.za
'''
import katuilib
# configure_cam('all') # TODO: must run this manually in interactive console



def reset_LMC(cam_ant):
    """ Take authority & Clear old tasks - e.g. like after OHB GUI work.
        This should be a temporary hack - if not sorted out by 01/03/2025 follow up with CAM team!
    """
    import time, tango
    dsm_addr = cam_ant.sensors.dsm_tango_address.get_value()
    dsm = tango.DeviceProxy(dsm_addr)
    dsm.RequestAuthority(); time.sleep(1)
    dsm.AckInterlock(); time.sleep(5)
    dsm.SetStandbyFPMode(); time.sleep(5)
    dsh_addr = cam_ant.sensors.dsh_tango_address.get_value()
    dsh = tango.DeviceProxy(dsh_addr)
    dsh.ClearOldTasks(); time.sleep(1)
    dsh.ClearTaskHistory(); time.sleep(1)
    dsh.ResetDishTasks()
