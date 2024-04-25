import os, sys
# If not deployed as a site package, expand the python path
__pkgroot__ = os.path.abspath(__file__+"/../..")
if ('site-packages' not in __pkgroot__):
    sys.path.append(__pkgroot__+"/libraries")
    for n in os.listdir(__pkgroot__+"/libraries"):
        if os.path.isdir(n):
            sys.path.append(__pkgroot__+"/libraries/"+n)
    sys.path.append(__pkgroot__+"/libraries/systems-analysis/analysis") # TODO: improve deployment of systems-analysis


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall
