#!/usr/bin/env python
""" Modified from holography_scan.py to specifically implement continuous scan patterns for determining pointing offsets while tracking.
        circle: scans along a fixed radial offset from the target.
    
    Potential future changes:
        "circle" data suffers from ambiguity: need to constrain either HPBW or peak amplitude.
        One way around this is to also scan across bore sight - something like a cardioid pattern?
    
    @author: aph@sarao.ac.za
"""
import time
import numpy as np
import katpoint
try:
    from katcorelib import (standard_script_options, verify_and_connect,collect_targets, start_session, user_logger, ant_array)
    testmode=False
except:
    testmode=True
    import optparse
    standard_script_options=optparse.OptionParser
    import matplotlib.pyplot as plt
    


def plane_to_sphere_holography(targetaz,targetel,ll,mm):
    scanaz=targetaz-np.arcsin(np.clip(ll/np.cos(targetel),-1.0,1.0))
    scanel=np.arcsin(np.clip((np.sqrt(1.0-ll**2-mm**2)*np.sin(targetel)+np.sqrt(np.cos(targetel)**2-ll**2)*mm)/(1.0-ll**2),-1.0,1.0))
    return scanaz,scanel

def generatepattern(totextent=10,tottime=1800,sampletime=1,scanspeed=0.15,slewspeed=-1,kind='circle'):
    """ Generates the basic scan pattern in target coordinates.
        
        All quantities in seconds except:
        @param kind: only 'circle' supported
        @param totextent: degrees
        @param scanspeed,slewspeed: degrees/second
        @param sampletime: seconds per sample
        @return: (x,y,slew)
    """
    if slewspeed<0.:#then factor of scanspeed
        slewspeed*=-scanspeed
    radextent=totextent/2.
    if (kind=='_circle_') or (kind == 'circle'):# Scan along a constant radial offset from target centre
        orbittime=2*np.pi*radextent/scanspeed
        norbits=int(tottime/orbittime + 0.5)
        dt=np.linspace(0,1,int(orbittime/sampletime))[:-1] # First & last points must not be duplicates, to ensure rate continuity
        armx=radextent*np.cos(2.0*np.pi*dt)
        army=radextent*np.sin(2.0*np.pi*dt)

        compositex=[]
        compositey=[]
        compositeslew=[]
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
    parser.add_option('--max-duration', type='float', default=None,
                      help='Maximum total measurement time, in seconds (will repeat the entire sequence, if suitable)  (default=%default)')
    parser.add_option('--num-cycles', type='int', default=-1,
                      help='Number of beam measurement cycles to complete (default=%default) use -1 for indefinite')
    parser.add_option('--cycle-duration', type='float', default=1800,
                      help='Time to spend on the pattern per cycle, in seconds (default=%default)')
    parser.add_option('-l', '--scan-extent', type='float', default=10,
                      help='Diameter of pattern to measure, in degrees (default=%default)')
    parser.add_option('--kind', type='string', default='circle',
                      help='Select the kind of pattern: may only be "circle" (default=%default)')
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
            plt.plot(compositex[iarm],compositey[iarm],'.')
            x.extend(compositex[iarm])
            y.extend(compositey[iarm])
            sl.extend(compositeslew[iarm])
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
            targets=catalogue.targets
            if len(targets) == 0:
                raise ValueError("Please specify a target argument via name ('Ori A'), "
                                 "description ('azel, 20, 30') or catalogue file name ('sources.csv')")
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
                session.telstate.add('obs_label','cycle.group.scan')

                user_logger.info("Initiating %s scan cycles (%s %g-second "
                                 "cycles extending %g degrees) using targets %s",
                                 opts.kind,('unlimited' if opts.num_cycles<0 else '%d'%opts.num_cycles), opts.cycle_duration,
                                 opts.scan_extent, ','.join(["'%s'"%(t.name) for t in targets]))

                start_time = time.time()
                cycle = 0
                targets = []
                while cycle<opts.num_cycles or opts.num_cycles<0: # Override exit conditions are coded in the next ten lines
                    if (len(targets) == 0): # Re-initialise the list in case there's more time & cycles left
                        targets = list(catalogue.targets)
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
                        user_logger.info("Quitting because none of the preferred targets are up")
                        break
                    else:
                        targets.remove(target)
                        user_logger.info("Using target '%s' (mean elevation %.1f degrees)",target.name,target_meanelev)
                        user_logger.info("Current scan estimated to complete at UT %s (in %.1f minutes)",time.ctime(time.time()+target_expected_duration+time.timezone),target_expected_duration/60.)
                    session.set_target(target)
                    user_logger.info("Performing azimuth unwrap")#ensures wrap of session.track is same as being used in load_scan
                    targetazel=gen_track([time.time()],target)[0][1:]
                    azeltarget=katpoint.Target('azimuthunwrap,azel,%s,%s'%(targetazel[0], targetazel[1]))
                    session.track(azeltarget, duration=0, announce=False)#azel target

                    user_logger.info("Performing initial track")
                    session.telstate.add('obs_label','track')
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
                    for iarm in range(len(cx)):# arm index
                        user_logger.info("Performing scan arm %d of %d.", iarm + 1, len(cx))
                        user_logger.info("Using Scan antennas: %s %s",
                                         ' '.join(always_scan_ants_names),' '.join([ant.name for ant in scan_ants if ant.name not in always_scan_ants_names]))
                        for iant,scan_ant in enumerate(scan_ants):
                            session.ants = scan_ants_array[iant]
                            target.antenna = scan_observers[iant]
                            scan_data, clipping_occurred = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                            if not kat.dry_run:
                                if clipping_occurred:
                                    user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                        for iant,track_ant in enumerate(track_ants):#also include always_scan_ants in track_ant list
                            if track_ant.name not in always_scan_ants_names:
                                continue
                            session.ants = track_ants_array[iant]
                            target.antenna = track_observers[iant]
                            scan_data, clipping_occurred = gen_scan(lasttime,target,cx[iarm],cy[iarm],timeperstep=opts.sampletime,high_elevation_slowdown_factor=opts.high_elevation_slowdown_factor,clip_safety_margin=1.0,min_elevation=opts.horizon)
                            if not kat.dry_run:
                                if clipping_occurred:
                                    user_logger.info("Warning unexpected clipping occurred in scan pattern")
                                session.load_scan(scan_data[:,0],scan_data[:,1],scan_data[:,2])
                        
                        lastisslew=None#so that first sample's state is also recorded
                        for it in range(len(cx[iarm])):
                            if cs[iarm][it]!=lastisslew:
                                lastisslew=cs[iarm][it]
                                session.telstate.add('obs_label','slew' if lastisslew else '%d.0.%d'%(cycle,iarm),ts=scan_data[it,0])
                        
                        time.sleep(scan_data[-1,0]-time.time()-opts.prepopulatetime)
                        lasttime = scan_data[-1,0]

                    session.telstate.add('obs_label','slew',ts=lasttime)
                    time.sleep(lasttime-time.time())#wait until last coordinate's time value elapsed
                    
                    #set session antennas to all so that stow-when-done option will stow all used antennas and not just the scanning antennas
                    session.ants = all_ants
                    user_logger.info("Safe to interrupt script now if necessary")
                    if kat.dry_run:#only test one cycle - dryrun takes too long and causes CAM to bomb out
                        user_logger.info("Testing only one cycle for dry-run")
                        break
                    cycle+=1
