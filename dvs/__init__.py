import os, sys
# If not deployed as a site package, expand the python path
__pkgroot__ = os.path.realpath(__file__+"/../..")
if ('site-packages' not in __pkgroot__):
    libroot = __pkgroot__+"/libraries"
    sys.path.append(libroot)
    for n in os.listdir(libroot):
        lib = __pkgroot__+"/libraries/"+n
        if os.path.isdir(lib):
            sys.path.append(lib)


cbid2url = lambda cbid: "http://archive-gw-1.kat.ac.za/%s/%s_sdp_l0.full.rdb"%(cbid,cbid) # Only works from inside the SARAO firewall
