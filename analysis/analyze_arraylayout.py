"""
    Some analysis of the characteristics of interferometric telescope arrays.
"""
import numpy as np
from katlab.geoconvert import llhlist2enu
import pylab as plt


def plot_EN(*enu_lists, labels=None):
    """ Generates a plot of E & N positions - should of course have same centres to make sense.
        @param enu_lists: one or more lists of (E,N,U) positions
    """
    plt.figure(figsize=plt.figaspect(1))
    markers = "ox^>"
    for enu_a,m in zip(enu_lists,markers):
        plt.plot(enu_a[:,0], enu_a[:,1], m)
    plt.xlabel("E [m]"); plt.ylabel("N [m]")
    if labels:
        plt.legend(labels)
    plt.grid(True)


def baseline_density(enu_list, display=False, **histkwargs):
    """ Computes baseline vectors and lengths, including zero baselines, and plots the density if requested.
        @param enu_list: a list of (E,N,U) positions
        @return: baseline list of (delta E, delta N delta U), baseline length -- sorted in ascending order of length
    """
    baselines = []
    lengths = []
    for p in enu_list:
        for q in enu_list:
            bl = (p[0]-q[0], p[1]-q[1], p[2]-q[2])
            baselines.append(bl)
            lengths.append(np.sum([d**2 for d in bl])**.5)
    
    if display:
        plt.hist(lengths, bins=100, **histkwargs)
        plt.xlabel("Baseline length [m]")
    
    # Sort baselines & lengths
    i = np.argsort(lengths) # Indices to sort lengths ascending
    baselines = np.take(baselines, i)
    lengths = np.take(lengths, i)
    return baselines, lengths


def distances(enu_ref, enu_list):
    """ Returns distances from positions relative to a reference position
        @param enu_ref
        @param enu_list: a list of (E,N,U) coordinates
        @return: list of distances in the same order as the positions """
    lengths = []
    for p in enu_list:
        bl = (p[0]-enu_ref[0], p[1]-enu_ref[1], p[2]-enu_ref[2])
        lengths.append(np.sum([d**2 for d in bl])**.5)
    return lengths
    

def analyse_meerkatext(what="plan,dist,baseline"):
    lon_0, lat_0, h_0 = 21.44389, -30.71111, 1086.6 # as of 2021 the formal "centre" of MeerKAT - see also https://github.com/ska-sa/katconfig/blob/karoo/static/arrays/karoo.array_mkat.conf
    mkt_llh = np.loadtxt("../catalogues/arrays/meerkat.txt", delimiter=",")
    mkt_enu = np.array(llhlist2enu(mkt_llh, lon_0, lat_0, h_0))
    skamid_llh = np.loadtxt("../catalogues/arrays/ska1-mid.txt", delimiter=",")
    skamid_enu = np.array(llhlist2enu(skamid_llh, lon_0, lat_0, h_0))
    
    mke_skaids = [17,18,20,23,105,107,115,117,119,110,118,60,121,26,116] # 15 preferred SKA* positions for MKE as of 04/2023
    mke_enu = np.take(skamid_enu, np.array(mke_skaids)-1, axis=0) 
    aa05_enu = np.take(skamid_enu, np.array([1,36,63,100])-1, axis=0) # From https://confluence.skatelescope.org/display/TDT/AA0.5+MID
    
    if ("plan" in what): # Plan view of receptor positions
        plot_EN(mkt_enu, mke_enu, aa05_enu, labels=["MeerKAT","MeerKAT+","AA0.5"])
    
    if ("dist" in what): # Nearest distance to existing MeerKAT antenna
        print("Nearest distance to new MeerKAT+ dish")
        for i,p in enumerate(mkt_enu):
            d = distances(p, mke_enu)
            print("m%03d"%i, min(d))
        print("Nearest distance to existing MeerKAT antennas")
        for i,p in enumerate(mke_enu):
            d = distances(p, mkt_enu)
            print("SKA%03d"%mke_skaids[i], min(d))
        
    if ("baseline" in what): # Baseline density plots
        # For entire arrays
        plt.figure()
        baseline_density(mkt_enu, display=True, density=False, log=True)
        baseline_density(np.r_[mkt_enu, mke_enu], display=True, density=False, log=True, alpha=0.5)
        plt.legend(["MeerKAT","MeerKAT+"])
        
        # For the MeerKAT array and each new dish in turn
        for i,p in enumerate(mke_enu):
            s = i%4
            if (s == 0):
                plt.figure()
            plt.subplot(2, 2, s+1)
            baseline_density(np.r_[mkt_enu, mke_enu], display=True, density=False, log=True)
            baseline_density(np.r_[mkt_enu, [q for q in mke_enu if distances(p,[q])[0]>0]], display=True, density=False, log=True)
            plt.legend(["MeerKAT+","MeerKAT+ w/o SKA%03d"%mke_skaids[i]])
    

if __name__ == "__main__":
    analyse_meerkatext()
    plt.show()
