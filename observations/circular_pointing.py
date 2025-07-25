#!/usr/bin/env python
""" Modified from holography_scan.py to specifically implement continuous scan patterns for determining pointing offsets while tracking.
        circle: scans along a fixed radial offset from the target.
        cardioid: makes a "loopy heart"-shaped scan centred on the target.
        epicycle: cycles around a circle centred on the target.
        
    @author: aph@sarao.ac.za
"""
import time
import numpy as np
import katpoint
try:
    from katcorelib import (standard_script_options, verify_and_connect, start_session, user_logger, ant_array)
    from dvs_obslib import plan_targets, filter_separation, collect_targets, standard_script_options, start_hacked_session as start_session # Override previous import
    from dvs_obslib import cycle_feedindexer, hack_SetPointingCorrections
    testmode=False
except:
    testmode=True
    import optparse
    standard_script_options = optparse.OptionParser
    cycle_feedindexer = lambda *a, **k: None
    import matplotlib.pyplot as plt
    


def plane_to_sphere_holography(targetaz,targetel,ll,mm):
    scanaz=targetaz-np.arcsin(np.clip(ll/np.cos(targetel),-1.0,1.0))
    scanel=np.arcsin(np.clip((np.sqrt(1.0-ll**2-mm**2)*np.sin(targetel)+np.sqrt(np.cos(targetel)**2-ll**2)*mm)/(1.0-ll**2),-1.0,1.0))
    return scanaz,scanel

def generatepattern(totextent=10,tottime=1800,sampletime=1,scanspeed=0.15,slewspeed=-1,kind='circle'):
    """ Generates the basic scan pattern in target coordinates.
        
        All quantities in seconds except:
        @param kind: only 'circle'|'cardioid'|'epicycle' supported
        @param totextent: degrees
        @param scanspeed,slewspeed: degrees/second
        @param sampletime: seconds per sample
        @return: (x,y,slew)
    """
    if (kind == '_circle_') or (kind == 'circle'): # Scan along a constant radial offset from target centre
        a, b = 1, 0 # Constant radius
    elif (kind == 'cardioid'): # "Loopy hearth-shaped" scan around the target centre
        a, b = 0.4, 1.5 # Large inner loop; effective radius ~ a+b/2
    elif (kind == 'epicycle'):
        pass
    else:
        raise ValueError("Patterns of kind %s not supported!" % kind)
    
    if slewspeed<0.:#then factor of scanspeed
        slewspeed *= -scanspeed
    radextent = totextent/2.
    
    orbittime = 2*np.pi*radextent/scanspeed
    norbits = max(1, int(tottime/orbittime + 0.5)) # At least 1
    dt = np.linspace(0,1,int(orbittime/sampletime))[:-1] # First & last points must not be duplicates, to ensure rate continuity
    th = 2.0*np.pi*dt + np.pi # Trajectories start near 0,0
    if (kind in ['cardioid', 'circle', '_circle_']): # These are all generically the same pattern
        a *= radextent; b *= radextent
        radius = a + b*np.cos(th)
        c = b if (b < a) else (b+a**2/4/b+a) # The cardioid is offset in the x direction
        armx = radius*np.cos(th) - c/2
        army = radius*np.sin(th)

    elif (kind == "epicycle"):
        r2 = radextent/2 # Sets the ratio of the circles to go between scanrad and center
        n = 4 # 2 looks like a cardioid, 3 is too "sparse"
        armx = r2*np.cos(th) + r2*np.cos(th*n)
        army = r2*np.sin(th) + r2*np.sin(th*n)
    
    compositex = []
    compositey = []
    compositeslew = []
    for orbit in range(norbits):
        compositex.append(armx)
        compositey.append(army)
        compositeslew.append(np.zeros(len(army)))
    
    return compositex,compositey,compositeslew #these coordinates are such that the upper part of pattern is sampled first; reverse order to sample bottom part first


def gen_scan(lasttime,target,az_arm,el_arm,timeperstep,high_elevation_slowdown_factor=1.0,clip_safety_margin=1.0,min_elevation=15.,max_elevation=90.):
    """
        high_elevation_slowdown_factor: normal speed up to 60degrees elevation slowed down linearly by said factor at 90 degrees elevation
        note due to the azimuth branch cut moved to -135 degrees, it gives 45 degrees (center to extreme) azimuth range before hitting limits either side
        for a given 10 degree (5 degree center to extreme) scan, this limits maximum elevation of a target to np.arccos(5./45)*180./np.pi=83.62 degrees before experiencing unachievable azimuth values within a scan arm in the worst case scenario
        note scan arms are unwrapped based on current target azimuth position, so may choose new branch cut for next scan arm, possibly corrupting current cycle.
    """
    num_points = np.shape(az_arm)[0]
    az_arm = az_arm*np.pi/180.0
    el_arm = el_arm*np.pi/180.0
    scan_data = np.zeros((num_points,3))
    attime = lasttime+np.arange(1,num_points+1)*timeperstep
    # arm scan
    targetaz_rad,targetel_rad=target.azel(attime)#gives targetaz in range 0 to 2*pi
    targetaz_rad=((targetaz_rad+135*np.pi/180.)%(2.*np.pi)-135.*np.pi/180.)#valid steerable az is from -180 to 270 degrees so move branch cut to -135 or 225 degrees
    scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,az_arm ,el_arm)
    if high_elevation_slowdown_factor>1.0:
        meanscanarmel=np.mean(scanel)*180./np.pi
        if meanscanarmel>60.:#recompute slower scan arm based on average elevation at if measured at normal speed
            slowdown_factor=(meanscanarmel-60.)/(90.-60.)*(high_elevation_slowdown_factor-1.)+1.0#scales linearly from 1 at 60 deg el, to high_elevation_slowdown_factor at 90 deg el
            attime = lasttime+np.arange(1,num_points+1)*timeperstep*slowdown_factor
            targetaz_rad,targetel_rad=target.azel(attime)#gives targetaz in range 0 to 2*pi
            targetaz_rad=((targetaz_rad+135*np.pi/180.)%(2.*np.pi)-135.*np.pi/180.)#valid steerable az is from -180 to 270 degrees so move branch cut to -135 or 225 degrees
            scanaz,scanel=plane_to_sphere_holography(targetaz_rad,targetel_rad,az_arm ,el_arm)
    #clipping prevents antenna from hitting hard limit and getting stuck until technician reset it, 
    #but not ideal to reach this because then actual azel could be equal to requested azel even though not right and may falsely cause one to believe everything is ok
    azdata=np.unwrap(scanaz)*180.0/np.pi
    eldata=scanel*180.0/np.pi
    scan_data[:,0] = attime
    scan_data[:,1] = np.clip(np.nan_to_num(azdata),-180.0+clip_safety_margin,270.0-clip_safety_margin)
    scan_data[:,2] = np.clip(np.nan_to_num(eldata),min_elevation+clip_safety_margin,max_elevation-clip_safety_margin)
    clipping_occurred=(np.sum(azdata==scan_data[:,1])+np.sum(eldata==scan_data[:,2])!=len(eldata)*2)
    return scan_data,clipping_occurred

def gen_track(attime,target):
    track_data = np.zeros((len(attime),3))
    targetaz_rad,targetel_rad=target.azel(attime)#gives targetaz in range 0 to 2*pi
    targetaz_rad=((targetaz_rad+135*np.pi/180.)%(2.*np.pi)-135.*np.pi/180.)#valid steerable az is from -180 to 270 degrees so move branch cut to -135 or 225 degrees
    track_data[:,0] = attime
    track_data[:,1] = np.unwrap(targetaz_rad)*180.0/np.pi
    track_data[:,2] = targetel_rad*180.0/np.pi
    return track_data


def test_target_azel_limits(target,clip_safety_margin,min_elevation,max_elevation,cycle_tracktime,sampletime,high_elevation_slowdown_factor):
    now=time.time()
    targetazel=gen_track([now],target)[0][1:]
    slewtotargettime=np.max([0.5*np.abs(currentaz-targetazel[0]),1.*np.abs(currentel-targetazel[1])])+1.0#antenna can slew at 2 degrees per sec in azimuth and 1 degree per sec in elev
    starttime=now+slewtotargettime+cycle_tracktime
    targetel=np.array(target.azel([starttime,starttime+1.])[1])*180.0/np.pi
    rising=targetel[1]>targetel[0]
    if rising:#target is rising - scan top half of pattern first
        cx=compositex
        cy=compositey
    else:  #target is setting - scan bottom half of pattern first
        cx=[com[::-1] for com in compositex[::-1]]
        cy=[com[::-1] for com in compositey[::-1]]
    meanelev=np.zeros(len(cx))
    for iarm in range(len(cx)):# arm index
        scan_data,clipping_occurred = gen_scan(starttime,target,cx[iarm],cy[iarm],timeperstep=sampletime,high_elevation_slowdown_factor=high_elevation_slowdown_factor,clip_safety_margin=clip_safety_margin,min_elevation=min_elevation,max_elevation=max_elevation)
        meanelev[iarm]=np.mean(scan_data[:,2])
        starttime=scan_data[-1,0]
        if clipping_occurred:
            return False, rising, starttime-now, meanelev[iarm]
    return True, rising, starttime-now, np.mean(meanelev)


if __name__=="__main__":
    # Set up standard script options
    parser = standard_script_options(usage="%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]",
                                     description='This script performs smooth pointing scans on the specified targets. '
                                                 'All the antennas initially track the target, whereafter a subset '
                                                 'of the antennas (the "scan antennas" specified by the --scan-ants '
                                                 'option) perform a scan on the target. Note also some '
                                                 '**required** options below. Targets ordered by preference.')
    # Add experiment-specific options
    parser.add_option('-b', '--scan-ants', default='*',
                      help='Subset of all antennas that will do raster scan, or "*" | "all" (default=%default). The rest will "track" only.')
    parser.add_option('--max-duration', type='float', default=43200,
                      help='Maximum total measurement time, in seconds (default=%default)')
    parser.add_option('--num-cycles', type='int', default=-1,
                      help='Number of beam measurement cycles to complete (default=%default) use -1 for indefinite')
    parser.add_option('--cycle-duration', type='float', default=1800,
                      help='Time to spend on the pattern per cycle, in seconds (default=%default)')
    parser.add_option('-l', '--scan-extent', type='float', default=10,
                      help='Diameter of pattern to measure, in degrees (default=%default)')
    parser.add_option('--kind', type='string', default='circle',
                      help='Select the kind of pattern: may only be "circle"|"cardioid"|"epicycle" (default=%default)')
    parser.add_option('--cycle-tracktime', type='float', default=30,
                      help='time in seconds to track a new target before starting the pattern (default=%default)')
    parser.add_option('--sampletime', type='float', default=0.25,
                      help='time in seconds to spend on each sample point generated (default=%default)')
    parser.add_option('--scanspeed', type='float', default=0.1,
                      help='scan speed in degrees per second (default=%default)')
    parser.add_option('--slewspeed', type='float', default=-1,
                      help='speed at which to slew in degrees per second, or if negative number then this multiplied by scanspeed (default=%default)')
    parser.add_option('--high-elevation-slowdown-factor', type='float', default=2.0,
                      help='factor by which to slow down nominal scanning speed at 90 degree elevation, linearly scaled from factor of 1 at 60 degrees elevation (default=%default)')
    parser.add_option('--prepopulatetime', type='float', default=10.0,
                      help='time in seconds to prepopulate buffer in advance (default=%default)')
                  
    parser.add_option('--min-separation', type="float", default=1.0,
                      help="Minimum separation angle to enforce between any two targets, in degrees (default=%default)")
    parser.add_option('--sunmoon-separation', type="float", default=10,
                      help="Minimum separation angle to enforce between targets and the sun & moon, in degrees (default=%default)")
    
    parser.add_option('--cluster-radius', type="float", default=0,
                      help="Group targets into clusters with this great circle dimension, in degrees (default=%default)")
    
    parser.add_option('--switch-indexer-every', type="int", default=-1,
                      help="Switch the feed indexer out & back again after every few scans, alternating directions if possible (default=never)")
    
    # Set default value for any option (both standard and experiment-specific options)
    parser.set_defaults(description='Circular pointing scan', quorum=1.0, nd_params='off')
    # Parse the command line
    opts, args = parser.parse_args()

    compositex,compositey,compositeslew=generatepattern(totextent=opts.scan_extent,tottime=opts.cycle_duration,sampletime=opts.sampletime,scanspeed=opts.scanspeed,slewspeed=opts.slewspeed,kind=opts.kind)
    if testmode:
        plt.figure()
        x=[]
        y=[]
        sl=[]
        for iarm in range(len(compositex)):
            cx, cy, cs = compositex[iarm], compositey[iarm], compositeslew[iarm]
            if  False: # (iarm == 0) and (opts.cycle_tracktime > 0): # Add a trajectory from cycle_track on bore sight to start of arm
                nt = int((cx[0]**2 + cy[0]**2)**.5 / (opts.scanspeed*opts.sampletime) + 0.5) # Number of points for this trajectory
                cx = list(np.linspace(0,cx[0],nt)[:-1]) + list(cx)
                cy = list(np.linspace(0,cy[0],nt)[:-1]) + list(cy)
                cs = list(np.zeros((nt,))[:-1]) + list(cs)
            plt.plot(cx,cy,'.')
            x.extend(cx)
            y.extend(cy)
            sl.extend(cs)
        x=np.array(x)
        y=np.array(y)
        sl=np.array(sl)
        for iarm in range(len(compositex)):
            slewindex=np.nonzero(compositeslew[iarm])[0]
            plt.plot(compositex[iarm][slewindex],compositey[iarm][slewindex],'.k',ms=1)
        plt.ylim([-opts.scan_extent/2,opts.scan_extent/2])
        plt.axis('equal')
        plt.title('%s scans: %d total time: %.1fs slew: %.1fs'%(opts.kind,len(compositex),len(sl)*opts.sampletime,np.sum(sl)*opts.sampletime))
        slewindex=np.nonzero(sl)[0]
        t=np.arange(len(x))*float(opts.sampletime)
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t[slewindex],x[slewindex],'r.')
        plt.plot(t[slewindex],y[slewindex],'r.')
        plt.plot(t,x,'-')
        plt.plot(t,y,'--')
        plt.ylabel('[degrees]')
        plt.legend(['x','y'])
        plt.title('Position profile')
        plt.subplot(3,1,2)
        plt.plot(t[slewindex],(np.diff(x)/opts.sampletime)[slewindex],'r.')
        plt.plot(t[slewindex],(np.diff(y)/opts.sampletime)[slewindex],'r.')
        plt.plot(t[:-1],np.diff(x)/opts.sampletime,'-')
        plt.plot(t[:-1],np.diff(y)/opts.sampletime,'--')
        plt.ylabel('[degrees/s]')
        plt.legend(['dx','dy'])
        plt.title('Speed profile')
        plt.subplot(3,1,3)
        plt.plot(t[slewindex-1],(np.diff(np.diff(x))/opts.sampletime/opts.sampletime)[slewindex-1],'r.')
        plt.plot(t[slewindex-1],(np.diff(np.diff(y))/opts.sampletime/opts.sampletime)[slewindex-1],'r.')
        plt.plot(t[:-2],np.diff(np.diff(x))/opts.sampletime/opts.sampletime,'-')
        plt.plot(t[:-2],np.diff(np.diff(y))/opts.sampletime/opts.sampletime,'--')
        plt.ylabel('[degrees/s^2]')
        plt.legend(['ddx','ddy'])
        plt.title('Acceleration profile')
        plt.show()
    else:
        # Check basic command-line options and obtain a kat object connected to the appropriate system
        with verify_and_connect(opts) as kat:
            catalogue = collect_targets(kat, args)
            if len(catalogue) == 0:
                raise ValueError("Please specify a target argument via name ('Ori A'), "
                                 "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
            # Remove sources that crowd too closely
            catalogue = filter_separation(catalogue, time.time(), kat.sources.antenna,
                                          separation_deg=opts.min_separation, sunmoon_separation_deg=opts.sunmoon_separation)
            targets = plan_targets(catalogue, time.time(), t_observe=opts.cycle_duration+opts.cycle_tracktime,
                                   cluster_radius=opts.cluster_radius, antenna=kat.ants[0], el_limit_deg=opts.horizon)[0]
            # Initialise a capturing session
            with start_session(kat, **vars(opts)) as session:
                # Use the command-line options to set up the system
                session.standard_setup(**vars(opts))
                all_ants = session.ants
                session.obs_params['num_scans'] = len(compositex)
                
                always_scan_ants_names=[] # Not currently used
                #determine scan antennas
                if opts.scan_ants in ['*', 'all']:
                    scan_ants = all_ants
                else:
                    # Form scanning antenna subarray (or pick the first antenna as the default scanning antenna)
                    scan_ants = ant_array(kat, opts.scan_ants if opts.scan_ants else session.ants[0], 'scan_ants')

                # Assign rest of antennas to tracking antenna subarray (or use given antennas)
                track_ants = [ant for ant in all_ants if ant not in scan_ants]
                track_ants = ant_array(kat, track_ants, 'track_ants')
                track_ants_array = [ant_array(kat, [track_ant], 'track_ant') for track_ant in track_ants]
                scan_ants_array = [ant_array(kat, [scan_ant], 'scan_ant') for scan_ant in scan_ants]

                # Add metadata
                #note obs_params is immutable and can only be changed before capture_start is called
                session.obs_params['scan_ants_always']=','.join(np.sort(always_scan_ants_names))
                session.obs_params['scan_ants']=','.join(np.sort([ant.name for ant in scan_ants]))
                session.obs_params['track_ants']=','.join(np.sort([ant.name for ant in track_ants]))
                # Get observers
                scan_observers = [katpoint.Antenna(scan_ant.sensor.observer.get_value()) for scan_ant in scan_ants]
                track_observers = [katpoint.Antenna(track_ant.sensor.observer.get_value()) for track_ant in track_ants]
                
                # Disable noise diode by default (to prevent it firing on scan antennas only during scans)
                nd_params = session.nd_params
                session.nd_params = {'diode': 'coupler', 'off': 0, 'on': 0, 'period': -1}
                
                # This also does capture_init, which adds capture_block_id view to telstate and saves obs_params
                session.capture_start()
                session.label("cycle.group.scan") # Compscan label - same as holography_scan.py, for possible future use

                user_logger.info("Initiating %s scan cycles (%s %g-second "
                                 "cycles extending %g degrees) using targets %s",
                                 opts.kind,('unlimited' if opts.num_cycles<0 else '%d'%opts.num_cycles), opts.cycle_duration,
                                 opts.scan_extent, ','.join(["'%s'"%(t.name) for t in targets]))

                start_time = time.time()
                cycle = 0
                targets = []
                prev_target = None # To keep track of changes in targets
                while cycle<opts.num_cycles or opts.num_cycles<0: # Override exit conditions are coded in the next ten lines
                    fresh_plan = False
                    if (len(targets) == 0): # Re-initialise the list in case there's more time & cycles left
                        targets = plan_targets(catalogue, time.time(), t_observe=opts.cycle_duration+opts.cycle_tracktime,
                                               cluster_radius=opts.cluster_radius, antenna=track_ants[0], el_limit_deg=opts.horizon)[0]
                        fresh_plan = True
                    if (len(targets) == 0):
                        user_logger.warning("No targets defined!")
                        break
                    if (opts.max_duration is not None) and (time.time() - start_time >= opts.max_duration):
                        user_logger.warning("Maximum duration of %g seconds has elapsed - stopping script", opts.max_duration)
                        break
                    if opts.num_cycles<0:
                        user_logger.info("Performing scan cycle %d of unlimited", cycle + 1)
                    else:
                        user_logger.info("Performing scan cycle %d of %d", cycle + 1, opts.num_cycles)
                    user_logger.info("Using all antennas: %s",' '.join([ant.name for ant in session.ants]))

                    #determine current azel for all antennas
                    currentaz=np.zeros(len(all_ants))
                    currentel=np.zeros(len(all_ants))
                    for iant,ant in enumerate(all_ants):
                        currentaz[iant]=ant.sensor.pos_actual_scan_azim.get_value()
                        currentel[iant]=ant.sensor.pos_actual_scan_elev.get_value()
                    #choose target
                    target=None
                    target_rising=False
                    target_elevation_cost=1e10
                    target_expected_duration=0
                    target_meanelev=0
                    target_histindex=0
                    targetinfotext=[]
                    for testtarget in targets:
                        suitable, rising, expected_duration, meanelev = test_target_azel_limits(testtarget,clip_safety_margin=2.0,min_elevation=opts.horizon,max_elevation=90.,
                                                                                                cycle_tracktime=opts.cycle_tracktime,sampletime=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor)
                        targetinfotext.append('%s (elev %.1f%s)'%(testtarget.name,meanelev,'' if suitable else ', unsuitable'))
                        if suitable:
                            target=testtarget
                            target_rising=rising
                            target_expected_duration=expected_duration
                            target_meanelev=meanelev
                            break
                    user_logger.info("Targets considered: %s"%(', '.join(targetinfotext)))
                    if target is None:
                        if fresh_plan:
                            user_logger.info("Quitting because none of the preferred targets are up")
                            break
                        else: # Make a fresh plan
                            del targets[:]
                            continue
                    else:
                        targets.remove(target)
                        user_logger.info("Using target '%s' (mean elevation %.1f degrees)",target.name,target_meanelev)
                        user_logger.info("Current scan estimated to complete at UT %s (in %.1f minutes)",time.ctime(time.time()+target_expected_duration+time.timezone),target_expected_duration/60.)
                    session.set_target(target)
                    
                    # Perform the cycle_track if requested 
                    if (target != prev_target) or (opts.cycle_tracktime > 0):
                        if not kat.dry_run: hack_SetPointingCorrections(all_ants) # HACK: change to & from load_scan causes OHB's ACU to re-enable ACU corrections
                        user_logger.info("Performing initial track")
                        session.label("track") # Compscan label
                        session.track(target, duration=opts.cycle_tracktime, announce=False)#radec target
                    
                    if (target_rising):#target is rising - scan top half of pattern first
                        cx=compositex
                        cy=compositey
                        cs=compositeslew
                    else:  #target is setting - scan bottom half of pattern first
                        cx=[com[::-1] for com in compositex[::-1]]
                        cy=[com[::-1] for com in compositey[::-1]]
                        cs=[com[::-1] for com in compositeslew[::-1]]
                    
                    user_logger.info("Using Track antennas: %s",' '.join([ant.name for ant in track_ants if ant.name not in always_scan_ants_names]))
                    lasttime = time.time()
                    session.activity("track") # Scan labels for all below - 'track' is counter-intuitive but is also used by holography_scan.py
                    for iarm in range(len(cx)):# arm index
                        user_logger.info("Performing scan arm %d of %d.", iarm + 1, len(cx))
                        user_logger.info("Using Scan antennas: %s %s",
                                         ' '.join(always_scan_ants_names),' '.join([ant.name for ant in scan_ants if ant.name not in always_scan_ants_names]))
                        armx, army, arms = cx[iarm], cy[iarm], cs[iarm]
                        if False: # (iarm == 0) and ((target != prev_target) or (opts.cycle_tracktime > 0)): # WIP: Add a trajectory from bore sight to start of arm
                            nt = int((armx[0]**2 + army[0]**2)**.5 / (opts.scanspeed*session.dump_period) + 0.5) # Number of points for this trajectory
                            if (nt > 0):
                                armx = np.r_[np.linspace(0,armx[0],nt)[:-1], armx]
                                army = np.r_[np.linspace(0,army[0],nt)[:-1], army]
                                arms = np.r_[np.zeros((nt,))[:-1], arms]
                        
                        for iant,scan_ant in enumerate(scan_ants):
                            session.ants = scan_ants_array[iant]
                            target.antenna = scan_observers[iant]
                            scan_data, clipping_occurred = gen_scan(lasttime,target,armx,army,timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                            if not kat.dry_run:
                                if clipping_occurred:
                                    user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                        if not kat.dry_run: hack_SetPointingCorrections(all_ants) # HACK: change to & from load_scan causes OHB's ACU to re-enable ACU corrections
                        for iant,track_ant in enumerate(track_ants):#also include always_scan_ants in track_ant list
                            if track_ant.name not in always_scan_ants_names:
                                continue
                            session.ants = track_ants_array[iant]
                            target.antenna = track_observers[iant]
                            scan_data, clipping_occurred = gen_scan(lasttime,target,armx,army,timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                            if not kat.dry_run:
                                if clipping_occurred:
                                    user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                        # Retrospectively add scan labels
                        lastisslew=None#so that first sample's state is also recorded
                        for it in range(len(armx)):
                            if arms[it]!=lastisslew:
                                lastisslew=arms[it]
                                session.telstate.add('obs_label',"slew" if lastisslew else "%d.0.%d"%(cycle,iarm),ts=scan_data[it,0]) # Compscan label
                        
                        time.sleep(scan_data[-1,0]-time.time()-opts.prepopulatetime)
                        lasttime = scan_data[-1,0]

                    session.telstate.add('obs_label',"slew",ts=lasttime) # Compscan label
                    time.sleep(lasttime-time.time()) #wait until last coordinate's time value elapsed
                    prev_target = target
                    
                    #set session antennas to all so that stow-when-done option will stow all used antennas and not just the scanning antennas
                    session.ants = all_ants
                    user_logger.info("Safe to interrupt script now if necessary")
                    cycle+=1

                    # Switch the indexer out & back, if requested
                    cycle_feedindexer(kat, cycle, opts.switch_indexer_every)
