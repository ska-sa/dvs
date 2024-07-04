import os, sys
# If not deployed as a site package, expand the python path
__pkgroot__ = os.path.realpath(__file__+"/../..")
if ('site-packages' not in __pkgroot__):
    libroot = __pkgroot__+"/libraries"
    sys.path.append(libroot)


from .util import cbid2url


modelsroot = __pkgroot__ + "/models"


from analysis import katsemodels as models
# Update these global paths (the module loader ensures there's only one instance, so all are affected!)
models.aperture_efficiency_dir = modelsroot+'/aperture-efficiency'
models.spill_over_dir = modelsroot+'/spill-over'
models.lab_Trec_dir = modelsroot+'/receiver-models'
