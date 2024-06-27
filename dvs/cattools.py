""" Functions associated with the various resources under the ../catalogues folder.
    
    @author aph@sarao.ac.za
"""
import numpy as np
import time
import pylab as plt
import matplotlib.projections
import katpoint

D2R = np.pi/180


def remove_overlapping(catalogue, eps=0.1, debug=True):
    """ Creates a new catalogue with duplicates in RA,DEC removed (keeps the first one in the list).
    
        @param catalgoue: original katpoint.Catalogue
        @param eps: radius within which two positions are considered identical [arcsec]
        @return: katpoint.Catalogue without duplicates """
    eps = eps/3600*D2R # arcsec -> rad
    
    # Don't need an antenna if for "radec" targets, but in case there are other types
    ant = catalogue.antenna if (catalogue.antenna is not None) else katpoint.Antenna("Z, 0,0,0, 0")
    
    newcat = katpoint.Catalogue(antenna=ant)
    for tgt in catalogue.targets:
        overlap = [(tgt.separation(t, antenna=ant) < eps) for t in newcat.targets]
        if (np.count_nonzero(overlap) == 0):
            newcat.add(tgt)
        elif debug:
            keep = np.take(newcat.targets, np.flatnonzero(overlap))[0]
            print("Removing overlapping target: ", tgt.description)
            print("  ( keeping ", keep.description, " -- %.1f arcsec off" % (tgt.separation(keep, antenna=ant)/D2R*3600), ")")
    return newcat


def plot_skycat(catalogue, timestamps, t_observe=120, antenna=None, el_limit_deg=20, ax=None):
    """ Plots distribution of catalogue targets on the sky, at timestamps """
    if (ax is None):
        ax = plt.figure().add_subplot(111, projection='polar')
    ax.set_xticks(np.arange(0,360,90)*D2R)
    ax.set_xticklabels(['E','N','W','S',])
    angle_formatter = lambda x,pos=None:matplotlib.projections.polar.ThetaFormatter(katpoint.wrap_angle(np.pi/2-x), pos)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(angle_formatter))
    ax.set_ylim(0, np.pi/2)
    ax.set_yticks(np.arange(0,90,10)*D2R)
    ax.set_yticklabels([])
    
    antenna = catalogue.antenna if (antenna is None) else antenna
    for T in timestamps:
        az, el = [], []
        for t,tgt in enumerate(catalogue.targets):
            _az, _el = tgt.azel(timestamp=T+t*t_observe, antenna=antenna)
            if (_el*180/np.pi > el_limit_deg):
                az.append(_az)
                el.append(_el)
        ax.plot(np.pi/2.-np.asarray(az), np.pi/2.-np.asarray(el), 'o', markersize=7)
    

def filter_separation(catalogue, T_observed, antenna=None, separation_deg=1, sunmoon_separation_deg=10):
    """ Removes targets from the supplied catalogue which are within the specified distance from others or either the Sun or Moon.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_observed: UTC timestamp, seconds since epoch [sec].
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str]
        @param separation_deg: eliminate targets closer together than this (default 1) [deg]
        @param sunmoon_separation_deg: omit targets that are closer than this distance from Sun & Moon (default 10) [deg]
        @return: katpoint.Catalogue (a filtered copy of input catalogue)
    """
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    targets = list(catalogue.targets)
    avoid_sol = [katpoint.Target('%s, special'%n) for n in ['Sun','Moon']] if (sunmoon_separation_deg>0) else []
    
    separation_rad = separation_deg*D2R
    sunmoon_separation_rad = sunmoon_separation_deg*D2R
    
    # Remove targets that are too close together (unfortunately also duplicated pairs)
    overlap = np.zeros(len(targets), float)
    for i in range(len(targets)-1):
        t_i = targets[i]
        sep = [(t_i.separation(targets[j], T_observed, antenna) < separation_rad) for j in range(i+1, len(targets))]
        sep = np.r_[np.any(sep), sep] # Flag t_j too, if overlapped
        overlap[i:] += np.asarray(sep, int)
        # Check for t_i overlapping with solar system bodies
        sep = [(t_i.separation(j, T_observed, antenna) < sunmoon_separation_rad) for j in avoid_sol]
        if np.any(sep):
            print("  # %s appears within %g deg from %s"%(t_i, sunmoon_separation_deg, np.compress(sep,avoid_sol)))
            overlap[i] += 1
    if np.any(overlap > 0):
        print("  # Planning drops the following due to being within %g deg away from other targets:\n%s"%(separation_deg, np.compress(overlap>0,targets)))
        targets = list(np.compress(overlap==0, targets))
    
    filtered = katpoint.Catalogue(targets, antenna=antenna)
    return filtered

   
def plan_targets(catalogue, T_start, t_observe, dAdt, antenna=None, el_limit_deg=20, debug=False):
    """ Generates a "nearest-neighbour" sequence of targets to observe, starting at the specified time.
        This does not consider behaviour around the azimuth wrap zone.
         
        @param catalogue: [katpoint.Catalogue]
        @param T_start: UTC timestamp, seconds since epoch [sec].
        @param t_observe: duration of an observation per target [sec]
        @param dAdt: angular rate when slewing [deg/sec]
        @param antenna: None to use the catalogue's antenna (default None) [katpoint.Antenna or str]
        @param el_limit_deg: observation elevation limit (default 15) [deg]
        @return: [list of Targets], expected duration in seconds
    """
    antenna = katpoint.Antenna(antenna) if isinstance(antenna, str) else antenna
    antenna = catalogue.antenna if (antenna is None) else antenna
    todo = list(catalogue.targets)
    done = []
    T = T_start # Absolute time
    # Start with any old target that's visible
    available = catalogue.filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
    next_tgt = available.targets[0] if (len(available.targets) > 0) else None
    while (next_tgt is not None):
        # Observe
        if debug: print(next_tgt)
        next_tgt.antenna = antenna
        done.append(next_tgt)
        todo.pop(todo.index(next_tgt))
        T += t_observe
        # Find next visible target
        available = katpoint.Catalogue(todo).filter(el_limit_deg=el_limit_deg, timestamp=T, antenna=antenna)
        next_tgt, dGC = available.closest_to(done[-1], T, antenna)
        # # Above as opposed to standard strategy which is 'next in list'
        # next_tgt = available.targets[0] if (len(available.targets) > 0) else None
        # dGC = 0 if next_tgt is None else next_tgt.separation(done[-1], T, antenna)*180/np.pi
        # Slew
        if next_tgt:
            if debug: print("  SLEWING %.1f deg"%dGC)
            T += dGC * dAdt
    T -= T_start # Now duration
    if debug: print("DURATION: %d out of %d targets in %.f min"%(len(done),len(catalogue.targets),T/60.))
    return done, T


def nominal_pos(ska_pad):
    """ Calculates the E,N,U offsets of an antenna, relative to the MeerKAT array reference coordinate.
        
        @param ska_pad: pad number to identify the antenna as SKA* [integer]
        @return: (delta_East, delta_North, delta_Up) [m] """
    lat0, lon0, h0 = -30.7110555, 21.4438888, 1086.6 # As of 2021 the formal "centre" of MeerKAT [https://github.com/ska-sa/katconfig/blob/karoo/static/arrays/karoo.array_mkat.conf]
    if (ska_pad == 0): # SKA-MPI is not in the catalogue
        name = "SKA-MPI"
        llh = (-30.717956, 21.413028, 1093)
        ph = 10.11 # 316-000000-022 rev 1
    else:
        name = "SKA%03d" % ska_pad
        llh = np.loadtxt(__file__+"/../../catalogues/arrays/ska1-mid.txt", comments="#", delimiter=",")[ska_pad-1]
        ph = 9.82 # 316-000000-022 rev 2, Fig 10
    
    lat, lon, hae = llh
    hae += ph
    E, N, U = katpoint.ecef_to_enu( lat0*D2R,lon0*D2R,h0,
                                    *katpoint.lla_to_ecef(lat*D2R,lon*D2R,hae) )
    print(name, "llh = (%.6f, %.6f, %.1f)" % (lat, lon, hae), "ENU = (%.1f %.1f %.1f)" % (E, N, U), "; all including pedestal height!")



if __name__ == "__main__":
    catroot = __file__ + "/../../catalogues/"
    
    if True: # Check spacing of pointing targets
        remove_overlapping(katpoint.Catalogue(open(catroot+"targets_pnt_L.csv")), eps=2*60*60, debug=True)
        remove_overlapping(katpoint.Catalogue(open(catroot+"targets_pnt_S.csv")), eps=60*60, debug=True)
        remove_overlapping(katpoint.Catalogue(open(catroot+"targets_pnt_Ku.csv")), eps=30*60, debug=True)
    
    elif True: # Planning a pointing measurement session
        catfiles = ["targets_pnt_S.csv"]
        for catfile in catfiles:
            axes = np.ravel(plt.figure().subplots(2,2, subplot_kw=dict(projection='polar')))
            Hr = 2 # Duration of a session in hours
            for N in range(4): # Repeat each "plan" 4 times
                T0 = time.time() + (Hr*N)*60*60 # Session at intervals of Hr hours (hope the plan fits in that time)
                print("\n%s"%catfile)
                cat = filter_separation(katpoint.Catalogue(open(catroot+catfile), antenna=katpoint.Antenna('m000, -30.713, 21.444, 1050.0')), T0)
                T = plan_targets(cat, T0, 8*(30+2), 2, debug=True)[1]
                M = int(Hr*60*60/T)+1 # Number of times the plan fits in 4 hours
                plot_skycat(cat, T0+np.arange(M)*T, 8*(30+2), ax=axes[N])
                axes[N].set_title(catfile + ("+ %d hrs"%(Hr*N)))
        plt.show()

    elif True: # Print coordinates of an antenna
        nominal_pos(ska_pad=0)
        