"""
L-band beam shape report, code for <http://kat-imager.kat.ac.za:8888/notebooks/RTS_reduction_results/3.1-Aperture_Phase_Efficiency/L-band%20beam%20shape%20report-APH.ipynb>

Work In Progress to upgrade from http://m.kat.ac.za:8888/notebooks/RTS%20holography/L-band%20beam%20shape%20report.ipynb and also http://kat-imager.kat.ac.za:8888/notebooks/RTS_reduction_results/3.1-Aperture_Phase_Efficiency/L-band%20beam%20shape%20report.ipynb

Note: rather refer to git/katsdpscripts/RTS/3.6-Beam_Shape__Sidelobe_Level/lband_beam_shape_report.py

Note: Gx,Dx,Gy,Dy is to be interpreted per the Jones matrix formulation:
      J = |Gx0   0| |Gx Dx|
          |0   Gy0| |Dy Gy|
     Gx0, Gy0 are the bore sight gains, which are not currently constrained by these measurements (& require a target with known, non-zero polarisation).
@author aph@ska.ac.za
"""
from katholog.hologreport import *
import numpy as np
import scipy.optimize as sop
import scipy.io
from pylab import *


_cM_ = 299.792458 # one millionth the speed of light

#https://docs.google.com/a/ska.ac.za/spreadsheets/d/1JI-RPBAyoEOsKYCqZPNuS5DXD8GPjBJH0A9WCxToBPI/edit?usp=sharing
wks = Spreadsheet('RTS lband beam shape register') # New since 11/2017

USE_BESTBEAMS = False

def print_URL(relative_path): # TODO: figure out how to not hard-code this!
    print('http://kat-imager.kat.ac.za:8888/files/RTS_reduction_results/3.1-Aperture_Phase_Efficiency/'+relative_path)
    print('http://kat-imager.kat.ac.za:8888/files/SysEng/ska-se/aph/'+relative_path)


## A complete re-factoring of beamshape_load() of above, along with an improvement in model beam applypointing!
## Now also exposes fitdBlevel for fitpoly (original default -14dB & degree=6 may be too poorly constrained for 900 MHz beam?)

model_cache = {}

def registry_load(ifile, usecycle=None): 
    """ @param ifile: row number in beam shape data registry
        @param usecycle: override the registry (default None).
        @return: filename, comment, scanantenna, usecycle, ignoreantennas (all strings) """
    filename,comment,ignoreantennastring,_usecycle,_usearr,_tgt,scanant=wks.get_row(None,ifile)[:7]
    usecycle = usecycle if usecycle else _usecycle
    return filename,comment,scanant,usecycle,ignoreantennastring

def beamshape_load(filename,comment,scanant,usecycle,ignoreantennastring='',applypointing='Gx',
        clipextent=6,bandwidth=4,f_centre=1712e6*3/4.,fitdBlevel=-14,freqs=[900], modelfolder='./',**kwargs): # APH added kwargs 092018 specifically to support measuredbeams_load(dishdiameter)
    ignoreantennas=ignoreantennastring.split(',') if (len(ignoreantennastring)>0) else []
    scanantennaname = scanant if scanant.strip() else None
    group = None # TODO?

    dataset, beams, apmaps = measuredbeams_load(filename,scanantennaname,ignoreantennas,clipextent,bandwidth,f_centre,usecycle,group,applypointing,fitdBlevel,freqs,**kwargs)
    if modelfolder:
        key = str([clipextent,freqs,applypointing])
        try:
            beamemss = model_cache[key][0]
        except:
            model_load = modelfolder if callable(modelfolder) else modelbeams_load
            model_cache[key] = model_load(clipextent,freqs, applypointing, fitdBlevel, apmaps=False, modelfolder=modelfolder)
            beamemss = model_cache[key][0]
    else:
        beamemss = beams
    return dataset, beamemss, beams, apmaps, freqs, bandwidth

def measuredbeams_load(filename,scanantennaname=None,ignoreantennas=[],clipextent=6,bandwidth=4,f_centre=1712e6*3/4.,usecycle=None,group=None,applypointing='Gx',fitdBlevel=-14,freqs=[900],dishdiameter=13.5,time_offset=0): # First code cell, added clipextent & bandwidth as marked APH below, added f_centre for Dataset default, original aplypointing='Gx' but 'I' or 'perfeed' is correct for "error beam" assessment
    """
        @param scanantennaname: Default None for the usual way, to detect it automatically
        @param group: Only used for multi-antenna scan sets
        @param dishdiameter: dish diameter override for ApertureMap - setting this to 13.965 (matches geomtric aperture) does impact aperture efficiency, TODO finalize this! (default 13.5 as used by katholog 07/2018)
    """
    extent=clipextent
    timingoffset = 0 # On top of the time offset for 2017-2018 datasets, which is now integratedinto katdal
    # APH: note that m.kat version of the original notebook has 'model' instead of 'direct' below but that causes e.g. M018 maps to crash on load
    dataset=katholog.Dataset(filename,'meerkat',method='direct',dobandpass=True,onaxissampling=0.1,scanantname=scanantennaname,ignoreantennas=ignoreantennas,katdal_centre_freq=f_centre,timingoffset=time_offset+timingoffset)
    dataset.scanantname = scanantennaname if scanantennaname else dataset.radialscan_allantenna[dataset.scanantennas[0]]
    if (usecycle=='best'): # Cherry pick the best individual scans from all cycles in the dataset - not sure if this should be allowed?
        flags_hrs=dataset.findworstscanflags(freqMHz=freqs,dMHz=bandwidth,scanantennaname=scanantennaname,trackantennaname=dataset.radialscan_allantenna[dataset.trackantennas[0]],doplot=True)
        dataset.flagdata(flags_hrs=flags_hrs,clipextent=clipextent,ignoreantennas=ignoreantennas) # APH 092018 added clipextent
    elif (usecycle=='' or usecycle=='all'):
        print('Using all cycles')
    elif (group is not None):
        dataset.flagdata(clipextent=clipextent,cycle=usecycle,group=group,ignoreantennas=ignoreantennas) # APH 092018 added clipextent
    else:
        cyclestart,cyclestop,nscanspercycle=dataset.findcycles(cycleoffset=0,doplot=False)
        cycleoffset=int((float(usecycle)-floor(float(usecycle)))*nscanspercycle)
        cyclestart,cyclestop,nscanspercycle=dataset.findcycles(cycleoffset=cycleoffset,doplot=True)
        cycles=list(zip(cyclestart,cyclestop))
        print('Using cycle %d of %d with cycleoffset %d of %d'%(int(float(usecycle)),len(cycles),cycleoffset,nscanspercycle))
        cycle=cycles[int(float(usecycle))]
        dataset.flagdata(timestart_hrs=cycle[0],clipextent=clipextent,timeduration_hrs=cycle[1]-cycle[0],ignoreantennas=ignoreantennas) # APH 092018 added clipextent

    #H and V pol was swapped (incorrectly) due to a DIG firmware bug from 21/11/2017 to 23/11/2017
    swappedfeed=(dataset.rawtime[0]>time.mktime((2017, 11, 21, 1,0,0,0,0,0)) and dataset.rawtime[0]<time.mktime((2017, 11, 24, 1,0,0,0,0,0)))
    swappedfeed=swappedfeed or ('1516231806' in dataset.filename) # Also on 2018/01/17
    
    beams=[]
    for thefreq in freqs:
        beam = katholog.BeamCube(dataset,freqMHz=thefreq,dMHz=bandwidth,scanantennaname=dataset.scanantname,interpmethod='scipy',
                                 applypointing=applypointing,extent=extent) # APH changed applypointing
        if swappedfeed:
            beam.Gx,beam.Dx,beam.Dy,beam.Gy = beam.Gy,beam.Dy,beam.Dx,beam.Gx
            beam.beamoffsetI,beam.beamoffsetGx,beam.beamoffsetGy=np.fliplr(beam.beamoffsetI),np.fliplr(beam.beamoffsetGy),np.fliplr(beam.beamoffsetGx)
            beam.beamwidthI,beam.beamwidthGx,beam.beamwidthGy=np.fliplr(beam.beamwidthI),np.fliplr(beam.beamwidthGy),np.fliplr(beam.beamwidthGx)
            beam.Gxgainlist,beam.Gygainlist,beam.Dxgainlist,beam.Dygainlist = beam.Gygainlist,beam.Gxgainlist,beam.Dygainlist,beam.Dxgainlist
        
        try:
            beam.fitpoly(fitdBlevel=fitdBlevel)
        except:
            beam.mGx=beam.Gx[:]
            beam.mDx=beam.Dx[:]
            beam.mDy=beam.Dy[:]
            beam.mGy=beam.Gy[:]
        
        beams.append(beam)
        
    apmaps = [] # [ApertureMaps (h,v)]
    xmag = -1.35340292717 # hologreport.py
    xyzoffsets = [0.0,-(-5.200+13.5/2.),-1.000-((13.5/2.)**2/(4*5.48617)-0.600)] # hologreport.py
    for thefreq in freqs:
        dfth=katholog.ApertureMap(dataset,scanantennaname=dataset.scanantname,freqMHz=thefreq,dMHz=bandwidth,feed='H',gridsize=128,xmag=xmag,xyzoffsets=xyzoffsets,dishdiameter=dishdiameter) # APH added dishdiameter
        dftv=katholog.ApertureMap(dataset,scanantennaname=dataset.scanantname,freqMHz=thefreq,dMHz=bandwidth,feed='V',gridsize=128,xmag=xmag,xyzoffsets=xyzoffsets,dishdiameter=dishdiameter) # APH added dishdiameter
        if swappedfeed:
            dfth,dftv = dftv,dfth
        apmaps.append((dfth,dftv))
    
    return dataset, beams, apmaps

def fit_beam(JHH,JVV, ll, mm): # Similar to what's implemented by katholog.Dataset.processoffaxisindependentwithgaindrift()
    """ @param JHH, JVV: linear scale complex voltage patterns.
        @param ll, mm: direction cosines [rad].
        @return: beamoffsets, beamwidths (all as [[I.h,v], [H], [V]] in rad)"""  
    sigma2fwhm = 2*np.sqrt(2.0*np.log(2.0)) # Multiply sigma of gaussian by this to get fwhm
    model = lambda p,x,y: np.exp(-0.5*(((x-p[0])/p[2])**2+((y-p[1])/p[3])**2)) * p[4]
    HH, VV = np.abs(JHH)**2, np.abs(JVV)**2
    beamwidths, beamoffsets = [], [] # [[I.h,v], [H], [V]] in deg
    for B in [(HH+VV)/2., HH, VV]:
        valid = B>=0.5 # Only use data within -3dB contour
        B, l, m = B[valid].reshape(-1), ll[valid].reshape(-1), mm[valid].reshape(-1)
        P = sop.fmin(lambda p:np.sum((model(p,l,m)-B)**2), [0,0,0.01,0.01,np.nanmax(B)], disp=False)
        beamoffsets.extend(P[0:2])
        beamwidths.extend(P[2:4]*sigma2fwhm)
    return np.asarray(beamoffsets), np.asarray(beamwidths)

def modelbeams_load(clipextent=6,freqs=[900], applypointing=None, fitdBlevel=-14, apmaps=False, modelfolder='./'):
    if USE_BESTBEAMS: # Use best measured rather than theoretical predicted
        return bestbeams_load(clipextent,freqs,applypointing,fitdBlevel,apmaps,modelfolder)
        
    extent=clipextent
    xmag = -1.35340292717 # hologreport.py
    xyzoffsets=[0,-13.5/2.0,0] # For predicted L-band beams
    beams, aperturemaps = [], []
    for thefreq in freqs:
        datasetemss=katholog.Dataset(modelfolder+'MK_GDSatcom_%d.mat'%thefreq,'meerkat',freq_MHz=thefreq,method='raw',clipextent=clipextent)
        # Predicted patterns are generated in transmit configuration, we need the reciprocal to compare with patterns measured in receive configuration
        # See [Potton, R. J. "Reciprocity in Optics." Reports on Progress in Physics, 2004]
        datasetemss.visibilities=[np.conj(v) for v in datasetemss.visibilities]
        datasetemss.ll=-datasetemss.ll # Conjugation & sign flips because of transmit - receive inversion
        datasetemss.mm=-datasetemss.mm

        beamemss=katholog.BeamCube(datasetemss,interpmethod='scipy',xyzoffsets=xyzoffsets,extent=extent)
        # Currently katholog.Dataset.getvisslice() zeroes beamoffsets & beamwidths for .mat files
        # katholog arranges visibiliteis as [HH, VV, HV, VH]
        beamoffsets, beamwidths = fit_beam(datasetemss.visibilities[0],datasetemss.visibilities[1], datasetemss.ll, datasetemss.mm)
        if (applypointing in ["I","Gx","Gy","perfeed"]): # Only effect is to shift Gx, Gy, Dx, Dy patterns
            if (applypointing=="I"): offsets = beamoffsets[0:2]
            elif (applypointing=="Gx"): offsets = beamoffsets[2:4]
            elif (applypointing=="Gy"): offsets = beamoffsets[4:6]
            elif (applypointing=="perfeed"): offsets = np.asarray([beamoffsets[i] for i in [2,5]])
            beamemss=katholog.BeamCube(datasetemss,interpmethod='scipy',xyzoffsets=xyzoffsets,extent=extent,applypointing=-offsets*180/np.pi)
        beamemss.beamoffsetI = np.r_[[beamoffsets[0:2]]] # Extra dimension is for frequency channel
        beamemss.beamoffsetGx = np.r_[[beamoffsets[2:4]]]
        beamemss.beamoffsetGy = np.r_[[beamoffsets[4:6]]]
        beamemss.beamwidthI = np.r_[[beamwidths[0:2]]]
        beamemss.beamwidthGx = np.r_[[beamwidths[2:4]]]
        beamemss.beamwidthGy = np.r_[[beamwidths[4:6]]]
        
        beamemss.fitpoly(fitdBlevel=fitdBlevel)
        beams.append(beamemss)
        
        if apmaps:
            dfth=katholog.ApertureMap(datasetemss,xmag=xmag,xyzoffsets=xyzoffsets,feed='H')
            dftv=katholog.ApertureMap(datasetemss,xmag=xmag,xyzoffsets=xyzoffsets,feed='V')
            aperturemaps.append((dfth,dftv))
    
    return beams, aperturemaps, freqs

def bestbeams_load(clipextent,freqs, applypointing, fitdBlevel, apmaps=False, modelfolder=None):
    """
        A replacement for modelbeams_load() which returns a "lucky best measurement" (m021 @ 57deg El on 2017/11/24) instead.
        Since many factors distort the pattern we need a large sample free of bad ones to use mean as best fit.
        Therefore we don't average, rather lucky image it!
    """
    bandwidth=4; f_centre=1712e6*3/4.; group=None
    filename = '/var/kat/archive3/data/MeerKATAR1/telescope_products/2017/11/24/1511490350.h5'
    scanantennaname='m021'
    ignoreantennas=[]
    usecycle=4 # 3 & 4 are very similar & good, but 3 has a small scan pattern anomaly which adds to error beam results
    dataset, beams, apmaps = measuredbeams_load(filename,scanantennaname,ignoreantennas,clipextent,bandwidth,f_centre,usecycle,group,applypointing,fitdBlevel,freqs)
    return beams, apmaps, freqs


### This code is a minor adaptation of <http://m.kat.ac.za:8888/notebooks/RTS%20holography/L-band%20beam%20shape%20report.ipynb> on 16/05/2017
### -- see also http://kat-imager.kat.ac.za:8888/notebooks/RTS_reduction_results/3.1-Aperture_Phase_Efficiency/L-band%20beam%20shape%20report.ipynb
# All changes are flagged "APH". Note: fitpoly() is always limited to -14dB

# Work in progress of original at katholog.hologreport.geterrorbeam
# APH added thispeak & modelpeak to avoid distorting especially noisy measured pattern
def _geterrorbeam(thisbeam,modelbeam,thispeak=None,modelpeak=None,contourdB=-12): # Assumes all are centred on same grid
    """@param thispeak,modelpeak: values to normalize the patterns to max=1, None to normalize to the center of the map (default None). """
    gridsize = modelbeam.shape[0]
    modelbeam = modelbeam / ( beam_peak(modelbeam) if modelpeak is None else float(modelpeak) ) # APH 12/2017 changed from "= norm_centre(modelbeam)"
    thisbeam = thisbeam / ( beam_peak(thisbeam) if thispeak is None else float(thispeak) ) # APH 12/2017 changed from "= norm_centre(thisbeam)"
    powbeam=20.0*np.log10(np.abs(modelbeam)).reshape(-1)
    dbeam=(np.abs(thisbeam)**2-np.abs(modelbeam)**2).reshape(-1)
    valid12dB=np.nonzero(powbeam>=contourdB)[0]
    dbeam[np.nonzero(np.isnan(dbeam))[0]]=0.0 # Error beams are 0 outside of area of interest
    errorbeam=dbeam.reshape([gridsize,gridsize])
    maxbeam=np.max(np.abs(errorbeam).reshape(-1)[valid12dB])
    stdbeam=np.std(errorbeam.reshape(-1)[valid12dB])
    return errorbeam,maxbeam,stdbeam

# Re-factoring of original at katholog.hologreport.norm_centre # APH added!
def beam_peak(map2d, pct=90): # Central (peak) value of ABS(map2d)
    m,n = map2d.shape
    map2d = np.abs(map2d) # We are concerned with power beams after all
    
    _m, _n = np.meshgrid(np.arange(m), np.arange(n))
    centre = (_m-m/2.)**2+(_n-n/2.)**2 <= 9 # APH refactored, unchanged 02/2017. [3^2->29 pixels, 2^2->13 pixels]
    
    ## Used prior to Sept 2016, but too simplistic in the presence of gridding & noise
    #C = np.max(map2d[centre])
    
    ## Introduced by APH 26/09/2016, the following is more robust against centering errors and noise
    if (pct > 0):
        x = np.percentile(map2d[centre],pct) # Consider only the 100-pct % largest values around the centre (e.g. 100-90=10% -> 4 out of 29)
        C = np.mean(map2d[map2d>=x]) # Mean of the few highest values around the centre
    else: # APH added else (pct <=0) 01/2018
        C = np.nanmax(map2d[centre])
        
    return C


def beamshape_report(dataset, beamemss, beams, apmaps, freqs, bandwidth,contourdB=-60,savefigs=list(range(99)),band="l",fntag=""):
    # APH added this nesting 12/2017 to normalize measured & fitted maps so that max(fit PP) = 1
    # which was necessary to correct an issue with error beam and also beam cut figures in the report.
    # However as this destroys the Gx0, Gy0 scale information required to know polarisation fully we
    # ONLY do it for this report and ensure the scale is restored on exit!
    try:
        for beam in beams:
            beam.Gx0 = beam_peak(beam.mGx[0],-1) # Keep track of scale
            beam.Gy0 = beam_peak(beam.mGy[0],-1)
            for bm,bf,b0 in zip([beam.Gx,beam.Dx,beam.Dy,beam.Gy], [beam.mGx,beam.mDx,beam.mDy,beam.mGy],
                                [beam.Gx0]*2+[beam.Gy0]*2):
                bm /= b0; bf /= b0
        if (beams != beamemss):
            for beam in beamemss:
                beam.Gx0 = np.nanmax(np.abs(beam.mGx)) # Keep track of scale
                beam.Gy0 = np.nanmax(np.abs(beam.mGy))
                for bm,bf,b0 in zip([beam.Gx,beam.Dx,beam.Dy,beam.Gy], [beam.mGx,beam.mDx,beam.mDy,beam.mGy],
                                    [beam.Gx0]*2+[beam.Gy0]*2):
                    bm /= b0; bf /= b0
        
        return _beamshape_report_(dataset, beamemss, beams, apmaps, freqs, bandwidth,contourdB,savefigs,band,fntag)
    
    finally: # This always runs before return completes
        for beamset in ([beams] if (beams==beamemss) else [beams, beamemss]):
            for beam in beamset:
                for bm,bf,b0 in zip([beam.Gx,beam.Dx,beam.Dy,beam.Gy], [beam.mGx,beam.mDx,beam.mDy,beam.mGy],
                                    [beam.Gx0]*2+[beam.Gy0]*2):
                    bm *= b0; bf *= b0
                del beam.Gx0, beam.Gy0
def _beamshape_report_(dataset, beamemss, beams, apmaps, freqs, bandwidth,contourdB=-60,savefigs=list(range(99)),band="l",fntag=""): # Second code cell: improved first line ->"dataset.filename" and added three lines after that
    """ @return: reportfilename, [(Gx error beam x freq),(Gy error beam x freq)], [(Gx xyz offsets x freq),(Gy xyz offsets x freq)] """
    reportfilename='_%sband_beam_report_%s_%s_%s%s.pdf'%(band,dataset.scanantname,dataset.radialscan_allantenna[dataset.trackantennas[0]],os.path.splitext(os.path.basename(dataset.filename))[0],fntag) # APH
    print_URL(reportfilename)
    extent = dataset.findsampling(dataset.ll,dataset.mm,dataset.flagmask)[1]*180/np.pi # APH
    geterrorbeam = lambda *args,**kwargs: _geterrorbeam(*args,contourdB=contourdB,**kwargs)[0:2] # APH
    gridsize=beamemss[0].gridsize
    margin=beamemss[0].margin
    emsslabel = "m021@57degEl" if USE_BESTBEAMS else "EMSS model" # TODO: APH hard-coded to match bestbeams_load
    plt.close('all') # APH added. Avoid trouble when re-using numbered figures.
    with PdfPages(reportfilename) as pdf:
        figure(1,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*4) # APH changed *len(freqs) to 4, also everywhere below
            beams[ifreq].plot('Gx','pow',clim=[-60,0],doclf=False)
            ylabel('%.f MHz\ndegrees'%thefreq) # APH added
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*4)
            beams[ifreq].plot('Dx','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*4)
            beams[ifreq].plot('Dy','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*4)
            beams[ifreq].plot('Gy','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('Katholog version: %s Processed: %s\n%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\n'%(katholog.__version__,time.ctime(),dataset.filename,dataset.scanantname,dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        if (1 in savefigs): pdf.savefig() # APH

        figure(2,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*4)
            beams[ifreq].plot('mGx','pow',clim=[-60,0],doclf=False)
            ylabel('%.f MHz\ndegrees'%thefreq) # APH added
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*4)
            beams[ifreq].plot('mDx','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*4)
            beams[ifreq].plot('mDy','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*4)
            beams[ifreq].plot('mGy','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\n'%('Polynomial fit',dataset.scanantname,dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        if (2 in savefigs): pdf.savefig() # APH

        figure(3,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*4)
            beamemss[ifreq].plot('Gx','pow',clim=[-60,0],doclf=False)
            ylabel('%.f MHz\ndegrees'%thefreq) # APH added
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*4)
            beamemss[ifreq].plot('Dx','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*4)
            beamemss[ifreq].plot('Dy','pow',clim=[-60,-20],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*4)
            beamemss[ifreq].plot('Gy','pow',clim=[-60,0],doclf=False)
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('%s\n%dMHz %dMHz %dMHz %dMHz\n'%(emsslabel,freqs[0],freqs[1],freqs[2],freqs[3]))
        if (3 in savefigs): pdf.savefig() # APH

        figure(4,figsize=(14,12))
        clf()
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].Gx[0,:,:]))
            bc=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,gridsize/2,:]))
            bc[np.nonzero(np.isnan(bc))[0]]=0.01
            plot(margin,bm[gridsize/2,:],'r',alpha=0.5) # APH added alpha here and below
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',alpha=0.5)
            plot(margin,bm[:,gridsize/2],'b',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',alpha=0.5)
            plt.axhline(-23,color='k',linestyle='--')
            xlim([-extent/2,extent/2])
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel("%.f MHz\ndB"%thefreq) # APH
            if (ifreq==0):
                legend(['0','45','90','135'])
            title('Gx')
            subplot(len(freqs),4,2+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].Dx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',alpha=0.5)
            plot(margin,bm[:,gridsize/2],'b',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',alpha=0.5)
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            xlim([-extent/2,extent/2])
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].Dy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',alpha=0.5)
            plot(margin,bm[:,gridsize/2],'b',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',alpha=0.5)
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            xlim([-extent/2,extent/2])
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].Gy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',alpha=0.5)
            plot(margin,bm[:,gridsize/2],'b',alpha=0.5)
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',alpha=0.5)
            plt.axhline(-23,color='k',linestyle='--')
            xlim([-extent/2,extent/2])
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        suptitle('%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\nCo-pol pattern second sidelobe upper limit is -23dB (dashed line)\nCross pol patterns limit of -26dB within the -1dB region (dashed rectangle)'%(dataset.filename,dataset.scanantname,dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        #pdf.savefig() # APH

        #figure(5,figsize=(14,12)) # APH
        #clf() # APH
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].mGx[0,:,:]))
            bc=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,gridsize/2,:]))
            bc[np.nonzero(np.isnan(bc))[0]]=0.01
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel("%.f MHz\ndB"%thefreq) # APH
            if (ifreq==0) and False: # APH forced False 
                legend(['0','45','90','135'])
            title('Gx')
            subplot(len(freqs),4,2+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].mDx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].mDy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*4)
            bm=20.*np.log10(np.abs(beams[ifreq].mGy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g')
            plot(margin,bm[:,gridsize/2],'b')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c')
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            xlim([-extent/2,extent/2])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        #suptitle('%s: %s (%s) %s\n%dMHz %dMHz %dMHz %dMHz bandwidth: %dMHz\nCo-pol pattern second sidelobe upper limit is -23dB (dashed line)\nCross pol patterns limit of -26dB within the -1dB region (dashed rectangle)'%('Polynomial fit',dataset.scanantname,dataset.radialscan_allantenna[dataset.trackantennas[0]],dataset.target.name,freqs[0],freqs[1],freqs[2],freqs[3],bandwidth))#+'Katholog version: %s'%(git_info('/var/kat/katholog')))
        #pdf.savefig() # APH

        #figure(6,figsize=(14,12)) # APH
        #clf() # APH
        for ifreq,thefreq in enumerate(freqs):
            subplot(len(freqs),4,1+ifreq*4)
            bm=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,:,:]))
            bc=20.*np.log10(np.abs(beamemss[ifreq].Gx[0,gridsize/2,:]))
            bc[np.nonzero(np.isnan(bc))[0]]=0.01
            plot(margin,bm[gridsize/2,:],'r',linestyle=':') # APH added linestyle here and below
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',linestyle=':')
            plot(margin,bm[:,gridsize/2],'b',linestyle=':')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',linestyle=':')
            ylabel("%.f MHz\ndB"%thefreq) # APH
            if (ifreq==0) and False: # APH forced False 
                legend(['0','45','90','135'])
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx')
            subplot(len(freqs),4,2+ifreq*4)
            bm=20.*np.log10(np.abs(beamemss[ifreq].Dx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r',linestyle=':')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',linestyle=':')
            plot(margin,bm[:,gridsize/2],'b',linestyle=':')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',linestyle=':')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dx')
            subplot(len(freqs),4,3+ifreq*4)
            bm=20.*np.log10(np.abs(beamemss[ifreq].Dx[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r',linestyle=':')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',linestyle=':')
            plot(margin,bm[:,gridsize/2],'b',linestyle=':')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',linestyle=':')
            plot(np.linspace(beamemss[ifreq].margin[0],beamemss[ifreq].margin[-1],beamemss[ifreq].gridsize),-20-6.0*(bc>-1.0),'k--')
            ylim([-50,-20])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Dy')
            subplot(len(freqs),4,4+ifreq*4)
            bm=20.*np.log10(np.abs(beamemss[ifreq].Gy[0,:,:]))
            plot(margin,bm[gridsize/2,:],'r',linestyle=':')
            plot(margin*np.sqrt(2),[bm[i,i] for i in range(len(margin))],'g',linestyle=':')
            plot(margin,bm[:,gridsize/2],'b',linestyle=':')
            plot(margin*np.sqrt(2),[bm[-1-i,i] for i in range(len(margin))],'c',linestyle=':')
            plt.axhline(-23,color='k',linestyle='--')
            ylim([-50,0])
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy')
        #suptitle('EMSS model\n%dMHz %dMHz %dMHz %dMHz\nCo-pol pattern second sidelobe upper limit is -23dB (dashed line)\nCross pol patterns limit of -26dB within the -1dB region (dashed rectangle)'%(freqs[0],freqs[1],freqs[2],freqs[3]))
        if (4 in savefigs): pdf.savefig() # APH

        figure(7,figsize=(14,12))
        clf()
        ext=margin[0]
        dGxmax=[0 for f in freqs]
        dGymax=[0 for f in freqs]
        mdGxmax=[0 for f in freqs]
        mdGymax=[0 for f in freqs]
        dGxstdev=[0 for f in freqs]
        dGystdev=[0 for f in freqs]
        for ifreq,thefreq in enumerate(freqs):
            dGx,dGxmax[ifreq]=geterrorbeam(beams[ifreq].Gx[0,:,:],beamemss[ifreq].Gx[0,:,:])
            dGy,dGymax[ifreq]=geterrorbeam(beams[ifreq].Gy[0,:,:],beamemss[ifreq].Gy[0,:,:])
            mdGx,mdGxmax[ifreq]=geterrorbeam(beams[ifreq].mGx[0,:,:],beamemss[ifreq].mGx[0,:,:])
            mdGy,mdGymax[ifreq]=geterrorbeam(beams[ifreq].mGy[0,:,:],beamemss[ifreq].mGy[0,:,:])
            
            mdGx=mdGx.reshape(-1)
            mdGy=mdGy.reshape(-1)
            dGx=dGx.reshape(-1)
            dGy=dGy.reshape(-1)
            dGxstdev[ifreq]=np.nanstd(mdGx-dGx)
            dGystdev[ifreq]=np.nanstd(mdGy-dGy)
            mdGx=np.abs(mdGx)
            mdGy=np.abs(mdGy)
            dGx=np.abs(dGx)
            dGy=np.abs(dGy)
            idx=np.nonzero(20.0*np.log10(np.abs(beamemss[ifreq].Gx[0,:,:].reshape(-1)))<contourdB)[0] # APH changed -12 -> contourdB
            idy=np.nonzero(20.0*np.log10(np.abs(beamemss[ifreq].Gy[0,:,:].reshape(-1)))<contourdB)[0] # APH as above
            mdGx[idx]=np.nan
            dGx[idx]=np.nan
            mdGy[idy]=np.nan
            dGy[idy]=np.nan
            mdGx=mdGx.reshape([gridsize,gridsize])
            mdGy=mdGy.reshape([gridsize,gridsize])
            dGx=dGx.reshape([gridsize,gridsize])
            dGy=dGy.reshape([gridsize,gridsize])

            subplot(len(freqs),4,1+ifreq*4)
            imshow(mdGx,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            ylabel('%.f MHz\ndegrees'%thefreq) # APH added
            if (ifreq<len(freqs)-1):
                xlabel('')
            title('Gx (max %.2f%%)'%(mdGxmax[ifreq]*100.0))
            subplot(len(freqs),4,2+ifreq*4)
            imshow(dGx,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gx (stddev %.2f%%)'%(dGxstdev[ifreq]*100.0))
            subplot(len(freqs),4,3+ifreq*4)
            imshow(mdGy,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy (max %.2f%%)'%(mdGymax[ifreq]*100.0))
            subplot(len(freqs),4,4+ifreq*4)
            imshow(dGy,extent=[-ext,ext,-ext,ext],clim=[0,0.05])
            xlim([-1.5,1.5])
            ylim([-1.5,1.5])
            colorbar()
            if (ifreq<len(freqs)-1):
                xlabel('')
            ylabel('')
            title('Gy (stddev %.2f%%)'%(dGystdev[ifreq]*100.0))
        suptitle('Error beam and polynomial fit within -12dB region\n%dMHz %dMHz %dMHz %dMHz\nMax error: %.2f%%'%(freqs[0],freqs[1],freqs[2],freqs[3],100.0*np.max([np.max(mdGxmax),np.max(mdGymax)])))
        if (7 in savefigs): pdf.savefig() # APH

        # APH
        h_feedoffsetx=[apmap[0].feedoffset[0] for apmap in apmaps]
        h_feedoffsety=[apmap[0].feedoffset[1] for apmap in apmaps]
        h_feedoffsetz=[apmap[0].feedoffset[2] for apmap in apmaps]
        h_feedoffsets=np.asarray([h_feedoffsetx,h_feedoffsety,h_feedoffsetz])
        v_feedoffsetx=[apmap[1].feedoffset[0] for apmap in apmaps]
        v_feedoffsety=[apmap[1].feedoffset[1] for apmap in apmaps]
        v_feedoffsetz=[apmap[1].feedoffset[2] for apmap in apmaps]
        v_feedoffsets=np.asarray([v_feedoffsetx,v_feedoffsety,v_feedoffsetz])
        if (99 in savefigs): # APH added this block
            figure(99,figsize=(8.27, 11.69), dpi=100)
            clf()
            gca().set_axis_off()
            printline('From %s until %s'%(dataset.env_time[0],dataset.env_time[-1]),setprinty=1.0)
            printline('')
            printline(['Ambient temperture: "mean (min to max)"','%.1f (%.1f to %.1f)'%(dataset.env_temp[0],dataset.env_temp[1],dataset.env_temp[2]),'[$^o$C]'],setprintcolwidths=[0,23,38])
            printline(['Wind speed:','%.1f (%.1f to %.1f)'%(dataset.env_wind[0],dataset.env_wind[1],dataset.env_wind[2]),'[mps]'])
            printline(['Elevation:','%.1f (%.1f to %.1f)'%(dataset.env_el[0],dataset.env_el[1],dataset.env_el[2]),'[degrees]'])
            printline(['Sun angle:','%.1f (%.1f to %.1f)'%(dataset.env_sun[0],dataset.env_sun[1],dataset.env_sun[2]),'[degrees]'])
            printline('')
            printline('')
            printline(['','H measurement','V Measurement',''],setprintcolwidths=[0,12,23,38])
            nstr = lambda array: ",".join("%.1f"%a for a in array)
            printline(['X Feed offset: avg (x freq)','%.1f (%s)'%(h_feedoffsets[0].mean(),nstr(h_feedoffsetx)),
                                        '%.1f (%s)'%(v_feedoffsets[0].mean(),nstr(v_feedoffsetx)),'[mm]'])
            printline(['Y Feed offset: avg (x freq)','%.1f (%s)'%(h_feedoffsets[1].mean(),nstr(h_feedoffsety)),
                                        '%.1f (%s)'%(v_feedoffsets[1].mean(),nstr(v_feedoffsety)),'[mm]'])
            printline(['Z Feed offset: fmax (x freq)','%.1f (%s)'%(h_feedoffsets[2,-1],nstr(h_feedoffsetz)),
                                        '%.1f (%s)'%(v_feedoffsets[2,-1],nstr(v_feedoffsetz)),'[mm]'])
            
            pdf.savefig()
        
        d = pdf.infodict()
        d['Title'] = 'RTS L band beam shape report'
        d['Author'] = socket.gethostname()
        d['Subject'] = 'L band beam shape report'
        d['Keywords'] = 'rts holography'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()
    
    return reportfilename, freqs, [mdGxmax,mdGymax], [h_feedoffsets,v_feedoffsets]

        
def __beamshape_minupdate(ifile,dataset): # Code cell 3
    cell_list = ['']*3
    cell_list[0]=dataset.target.name#targetname
    cell_list[1]=dataset.scanantname#antenna
    cell_list[2]=dataset.radialscan_allantenna[dataset.trackantennas[0]]#antenna
    wks.update_row(None,cell_list,ifile,5) # cols F-H


def beamshape_update(ifile,dataset,reportfilename,freqs,GxGyerrorbeams,hv_xyzfeedoffsets, applypointing='Gx'): # APH added
    """
        @param dataset: first argument returned by beamshape_load()
        @param GxGyerrorbeams: [Gxerrorbeams x freqs, Gyerrorbeams x freqs]
        @param hv_xyzfeedoffsets: [h_xyzfeedoffsets x freqs, v_xyzfeedoffsets x freqs]
        @param applypointing: 'Gx__','Gy__' registers only H or V, 'I' registers both against I columns, anything else registers both H & V
    """ 
    __beamshape_minupdate(ifile,dataset)
    
    if not USE_BESTBEAMS: # Hyperlink
        cell_list = wks.get_row(None,ifile)[0:1] # col A
        cell_list[0] = hyperlink(reportURL(reportfilename),hyperlinktext(cell_list[0]))
        wks.update_row(None,cell_list,ifile,0)
    # Error beam
    c0 = 29 if USE_BESTBEAMS else 12 # cols AD-AK for 'best' or cols M-T for 'emss'
    cell_list = ['']*4
    if (applypointing == "I"): # Error beam with pointing from I beam
        cell_list[0] = np.max(GxGyerrorbeams[0])*100
        cell_list[1] = ", ".join([str(f) for f,eb in zip(freqs,GxGyerrorbeams[0]) if np.max(eb)>0.04])
        cell_list[2] = np.max(GxGyerrorbeams[1])*100
        cell_list[3] = ", ".join([str(f) for f,eb in zip(freqs,GxGyerrorbeams[1]) if np.max(eb)>0.04])
        wks.update_row(None,cell_list,ifile,c0+4)
    else:
        if (applypointing in ["Gx__","perfeed"]): # H Error beam with pointing from Gx beam
            cell_list[0] = np.max(GxGyerrorbeams[0])*100
            cell_list[1] = ", ".join([str(f) for f,eb in zip(freqs,GxGyerrorbeams[0]) if np.max(eb)>0.04])
        if (applypointing in ["Gy__","perfeed"]): # V Error beam with pointing from Gy beam
            cell_list[2] = np.max(GxGyerrorbeams[1])*100
            cell_list[3] = ", ".join([str(f) for f,eb in zip(freqs,GxGyerrorbeams[1]) if np.max(eb)>0.04])
        wks.update_row(None,cell_list,ifile,c0)
    # Feed offsets
    hv_xyz = np.asarray(hv_xyzfeedoffsets) # 0: HV, 1: xyz, 2: freq
    xyz_converge = np.mean(hv_xyz, axis=2) # xyz x HV averaged over freq
    xyz_converge[:,-1] = hv_xyz[:,-1,-1] # HV z: z converges at fmax
    I_xyzfeedoffsets = np.mean(xyz_converge, axis=0) # Average of H & V pol
    D_xyzfeedoffsets = np.abs(np.rollaxis(hv_xyz,2) - I_xyzfeedoffsets).max(axis=(0,1)) # Max difference between any pol & Average
    D_xyzfeedoffsets[-1] = np.max(xyz_converge[:,-1] - I_xyzfeedoffsets[-1]) # z converges at fmax
    #I_xyzfeedoffsets = np.asarray(hv_xyzfeedoffsets).mean(axis=0) # Average of H & V pol
    #D_xyzfeedoffsets = np.abs(np.asarray(hv_xyzfeedoffsets[0]) - I_xyzfeedoffsets) # Difference between any pol & Average
    cell_list = ['']*6 #  X,DX,Y,DY,Z,DZ
    cell_list[0] = I_xyzfeedoffsets[0] # X
    cell_list[1] = D_xyzfeedoffsets[0] # DX
    cell_list[2] = I_xyzfeedoffsets[1] # Y
    cell_list[3] = D_xyzfeedoffsets[1] # DY
    cell_list[4] = I_xyzfeedoffsets[2] # Z
    cell_list[5] = D_xyzfeedoffsets[2] # DZ
    wks.update_row(None,cell_list,ifile,20) # cols U-Z:
    # Other contextual info
    cell_list = ['']*2 #  El, Sun
    cell_list[0] = dataset.env_el[0]
    cell_list[1] = dataset.env_sun[0]
    wks.update_row(None,cell_list,ifile,26) # cols AA-AB

def hyperlink(URL, text):
    return '=HYPERLINK("%s","%s")'%(URL,text)    
def hyperlinktext(hyperlink): # Separate out the text from a google sheets hyperlink, if it is
    try:
        return hyperlink.split('"')[-2] # Google sheets requires use of "; the last group is the closing brakcets.
    except:
        return hyperlink

def recalc_eff(apmaps, freqs_MHz, D=None, scaleto_Ag=None, save=False, band="", ret_Ag_avg=True):
    """ Re-computes efficiencies for the physical geometric aperture. For MeerKAT this is as masked for ApertureMap.devmap,
        rather than the initial ApertureMap.dishdiameter which for MeerKAT defaults to 13.5 m.
        TODO: confirm why it doesn't need to be scaled to match the predictions scaled to 13.5m convention of the models module? - is that why derived Tsys is typically 5% lower than expected?
        @param D: give this to force the use of a circular aperture with diameter D [m] (default None)
        @param band: a string to mark the saved filename with.
        @return: illumination_efficiency, antenna_efficiency, Ag (each [freq_MHz,eff_H,eff_V] all efficiencies in percentage, Ag *or scaleto_Ag* in m^2)
    """
    # Update the maskmap for the integrals
    for apmap_xy in apmaps: # Sets of ApertureMaps
        for apmap in apmap_xy: # H & V
            if D: # Apply a circular aperture boundary when integrals are re-evaluated
                diam = apmap.dishdiameter
                apmap.dishdiameter = D; apmap.gain(blockdiameter=0) # blockdiameter=0 avoids using cached *default* mask
                apmap.dishdiameter = diam
            else: # Update the mask to the same as applied to devmap - which follows the main reflector outline
                _maskmap = apmap.maskmap
                apmap.maskmap = np.zeros(apmap.maskmap.shape, np.float); apmap.maskmap[np.isnan(apmap.devmap)] = 1
                apmap.gain(blockdiameter=None) # blockdiameter=None to use *updated* mask
                apmap.maskmap = _maskmap
        
    Ag = np.asarray([[apmap[0].gainuniform*(_cM_/f)**2/(4*np.pi), # Geometric aperture area
                      apmap[1].gainuniform*(_cM_/f)**2/(4*np.pi)] for f,apmap in zip(freqs_MHz,apmaps)])
    Ag = np.nanmean(Ag)
    scale = Ag/scaleto_Ag if scaleto_Ag else 1.
    _Ag = scaleto_Ag if scaleto_Ag else Ag
    
    illeff = np.asarray([[f, scale*apmap[0].eff0_illumination*100, # This is IEEE Std 145 ILLUMINATION EFFICIENCY, includes taper & phase, excludes spillover
                             scale*apmap[1].eff0_illumination*100] for f,apmap in zip(freqs_MHz,apmaps)])
    anteff = np.asarray([[f, scale*apmap[0].gainmeasured/apmap[0].gainuniform*100, # This is IEEE ANTENNA APERTURE ILLUMINATION EFFICIENCY, slightly lower than eff0_illumination - includes all effects that impact on directivity pattern (i.e. only excluding Ohmic?) 
                             scale*apmap[1].gainmeasured/apmap[1].gainuniform*100] for f,apmap in zip(freqs_MHz,apmaps)])

    if save:
        fn = apmap[0].dataset.filename
        np.savetxt("eff0_ap_%s.csv"%({"u":"UHF","l":"L"}.get(band,band)), anteff, fmt='%g', delimiter="\t",
                   header="ant_eff=gainmeasured/gainuniform as determined from ApertureMap of %s\nScaled to Ag=%.1f m^2\n"%(fn[-47:],_Ag)+
                          "f [MHz]\tH [percent]\tV [percent]")
    return illeff, anteff, (_Ag if ret_Ag_avg else Ag)


def get_hpbw(beamcubes, refit=False):
    """ New best fit of beam widths for the BeamCubes
        @param beamcubes: a list of BeamCube objects
        @return: HPBW_X_xy, HPBW_Y_xy [rad] each a vector matching the number of beamcubes.
    """
    bw_x, bw_y = [], []
    for bmcube in beamcubes:
        if refit: # Use katholog's default fitted results
            bw_x.append(bmcube.beamwidthGx.squeeze()) # These are consistently out compared to EMSS predictions as well as fits to drift scans - WHY?
            bw_y.append(bmcube.beamwidthGy.squeeze())
        else: # New fit as defined in the current module -- fit_beam()
            ll,mm,vis = bmcube.dataset.getvisslice(frequencyMHz=bmcube.freqgrid[0],dMHz=4,scanantennaname=bmcube.dataset.scanantname,
                                                   trackantennanames=bmcube.dataset.radialscan_allantenna[bmcube.dataset.trackantennas[0]],
                                                   ich=0)[:3] # ll,mm,vis,beam params
            beamoffsets, beamwidths = fit_beam(vis[0],vis[1], ll, mm) # [0]=HH,[1]=VV
            bw_x.append(beamwidths[2:4]) # Gx
            bw_y.append(beamwidths[4:6]) # Gy
    return bw_x, bw_y


def savemat_beamcubes(dataset, beamcubes, freqs_MHz, fnroot="."): # Save beam patterns in a .mat file.
    Gx = [d.Gx[0] for d in beamcubes] # Data from only freqgrid[0], i.e. each cube represnts a single frequency
    Dx = [d.Dx[0] for d in beamcubes]
    Dy = [d.Dy[0] for d in beamcubes]
    Gy = [d.Gy[0] for d in beamcubes]
    fn = "%s/BeamPatterns-%s.mat"%(fnroot,dataset.filename[-13:-3])
    scipy.io.savemat(fn, {"Gx":Gx,"Dx":Dx,"Dy":Dy,"Gy":Gy,"f_MHz":freqs_MHz,"extent_deg":d.extent}) # All same extent
    return fn

def loadmat_beamcubes(fn, make_plots=False): # Load beam patterns from a .mat file
    d = scipy.io.loadmat(fn, squeeze_me=True)
    if make_plots:
        extent = [-d["extent_deg"]/2.,d["extent_deg"]/2.]*2
        N = len(d["f_MHz"])
        figure(figsize=(16,4*N)); suptitle("Far field magnitude in beam tangent plane (El vs cross-El)")
        for n in range(N):
            subplot(N,4,n*4+1); imshow(log10(abs(d["Gx"][n])), origin='lower', extent=extent); ylabel("%.1f MHz"%(d["f_MHz"][n]))
            subplot(N,4,n*4+2); imshow(log10(abs(d["Dx"][n])), origin='lower', extent=extent)
            subplot(N,4,n*4+3); imshow(log10(abs(d["Dy"][n])), origin='lower', extent=extent)
            subplot(N,4,n*4+4); imshow(log10(abs(d["Gy"][n])), origin='lower', extent=extent)
    return d


def calc_stats(dataset, beams_pred, beams_meas, apmaps, freqs, BW, debug=True):
    """
        @return: beamwidths [I:h,v; Gx:h,v, Gy:h,v rad], beamoffsets [I:h,v; Gx:h,v; Gy:h,v rad], feedoffsets [Gx:x,y,z; Gy:x,y,z mm]) all vs. freq along axis=0
    """
    beamwidths = []
    for beam, f in zip(beams_meas,freqs):
        beamwidths.append(np.r_[beam.beamwidthI, beam.beamwidthGx, beam.beamwidthGy])
        if debug: print(f, 1.22*(_cM_/f/13.965)*180/np.pi, beamwidths[-1]*180/np.pi)

    beamoffsets = []
    for beam, f in zip(beams_meas,freqs):
        beamoffsets.append(np.r_[beam.beamoffsetI, beam.beamoffsetGx, beam.beamoffsetGy])
        if debug: print(f, beamoffsets[-1]*180/np.pi*60)

    feedoffsets = []
    for apmap, f in zip(apmaps,freqs):
        feedoffsets.append(np.r_[[apmap[0].feedoffset], [apmap[1].feedoffset]])
        if debug: print(f, feedoffsets[-1])

    return np.asarray(beamwidths), np.asarray(beamoffsets), np.asarray(feedoffsets)


def distill_stats(dataset, beams_pred, beams_meas, apmaps, freqs, BW, tag=""):
    scanants = [dataset.scanantname]
    beamwidths, beamoffsets, feedoffsets = [], [], []
    beamw, beamo, feedo = calc_stats(dataset, beams_pred, beams_meas, apmaps, freqs, BW, debug=False)
    beamwidths.append(np.c_[freqs, np.reshape(beamw, (len(freqs),-1))])
    beamoffsets.append(np.c_[freqs, np.reshape(beamo, (len(freqs),-1))])
    feedoffsets.append(np.c_[freqs, np.reshape(feedo, (len(freqs),-1))])

    filename = dataset.filename
    np.savetxt(filename[-13:-3]+tag+"_"+scanants[0]+"_beamwidths.csv", np.reshape(beamwidths,(len(scanants)*len(freqs),-1)), header="Freq [MHz] Beam widths [I:h,v; Gx:h,v, Gy:h,v rad] from %s, for %s"%(filename, ",".join(scanants)))
    np.savetxt(filename[-13:-3]+tag+"_"+scanants[0]+"_beamoffsets.csv", np.reshape(beamoffsets,(len(scanants)*len(freqs),-1)), header="Freq [MHz] Beam offsets [I:h,v; Gx:h,v, Gy:h,v rad] from %s, for %s"%(filename, ",".join(scanants)))
    np.savetxt(filename[-13:-3]+tag+"_"+scanants[0]+"_feedoffsets.csv", np.reshape(feedoffsets,(len(scanants)*len(freqs),-1)), header="Freq [MHz] Feed offsets [Gx:x,y,z; Gy:x,y,z mm] from %s, for %s"%(filename, ",".join(scanants)))

    
def hijack_update(wks, switch, logfn=None, append=False): # switch=True makes update_cells() or update_row() rather print to stdout
    if (switch == 'ON') and not ("__hijacked_update_row__" in list(wks.__dict__.keys())):
        print("Hijacking 'update' to "+(logfn if logfn else "console"))
        if logfn:
            wks.__hijack_log__ = open(logfn,'a' if append else 'w')
            def print_cells(x,cell_list,r,c0): wks.__hijack_log__.write(("\t".join(['']*c0+[str(c) for c in cell_list]))+"\n"); wks.__hijack_log__.flush()
        else:
            def print_cells(x,cell_list,r,c0): print("\t".join(['']*c0+[str(c) for c in cell_list]))
        wks.__hijacked_update_row__ = wks.update_row
        wks.update_row = print_cells
        
        wks.__hijacked_get_row__ = wks.get_row
        wks.get_row = lambda *args,**kwargs: ['']*99
    
    elif (switch != 'ON') and ("__hijacked_update_row__" in list(wks.__dict__.keys())):
        print("Restoring 'update' function")
        wks.update_row = wks.__hijacked_update_row__
        wks.get_row = wks.__hijacked_get_row__
        if ("__hijack_log__" in list(wks.__dict__.keys())):
            wks.__hijack_log__.close()
            del wks.__hijack_log__
        del wks.__hijacked_update_row__, wks.__hijacked_get_row__
        del wks.update_row, wks.get_row


def process_AR1(filename, scanants, cycles, pointing, ignoreantennas=None, bestbeams=False, band="l"):
    """ Process new multiscan holography-style datasets. Sample usage (all are strings):
        @param filename: '/var/kat/archive3/data/MeerKATAR1/telescope_products/2017/11/17/1510886426.h5' # 3C 273, 4 cycles
        @param scanants: 'm048,m043,m033,m054,m041,m062,m053,m051,m052,m039'
        @param cyles: cycles to process '[0,1,3]'
        @param pointing:' I,perfeed'
        @param ignoreants: 'm058,m063' (default None)
        @param bestbeams: True if the reference patterns used for error beam were 'best' as oppesed to 'emss' (default False)
        @param band: "l" or "u" (default "l")
    """
    hijack_update(wks, 'ON', 'multiscan_'+filename[-13:-3]+('_bb' if bestbeams else '')+'.log',append=True)

    try:
        for pntng in pointing.split(','):
            for cycle in np.atleast_1d(eval(cycles)):
                for scanant in scanants.split(','):
                    bmresults, fn,freqs,GxGyEB,hv_xyz = process(filename, scanant, cycle, pntng, ignoreantennas, bestbeams, band, tag=".%d"%cycle)
                    beamshape_update(0,bmresults[0],fn,freqs,GxGyEB,hv_xyz,pntng)
    finally:
        hijack_update(wks, 'OFF')


def process_RTSregistry(rows, cycle, pointing, bestbeams=False, band="l"):
    """ Process the RTS registry-based holography datasets. Sample usage (all are strings):
        @param rows: row(s) in the registry e.g. '3' or '3,4,5' or 'range(3,7)'
        @param cyle: '0', 'best' to override what's in the registry (default None).
        @param pointing: 'I,Gx,Gy,perfeed' or either 'I' or 'perfeed'.
        @param bestbeams: True if the reference patterns used for error beam were 'best' as oppesed to 'emss' (default False)
        @param band: "l" or "u" (default "l")
    """
    for ifile in np.atleast_1d(eval(rows)):
        for pntng in pointing.split(','):
            try:
                filename, comment, scanantenna, usecycle, ignoreantennas = registry_load(ifile,usecycle=cycle)
                bmresults, fn,freqs,GxGyEB,hv_xyz = process(filename, scanantenna, usecycle, pntng, ignoreantennas, bestbeams, band)
                beamshape_update(ifile,bmresults[0],fn,freqs,GxGyEB,hv_xyz,pntng)
            except Exception as e: # Some datasets are not usable
                print("!!!!!!!!!!!! Skipping row %d due to %s"%(ifile,e))


def process(filename, scanantenna, cycle, pointing, ignoreantennas=None, bestbeams=False, band="l", tag=""):
    """ Process the RTS holography datasets. Sample usage (all are strings):
        @param filename: '/var/kat/archive3/data/MeerKATAR1/telescope_products/2017/11/17/1510886426.h5' # 3C 273, 4 cycles
        @param target: e.g. '3C273'
        @param cyle: '0', 'best' to override what's in the registry (default None).
        @param pointing: either 'I', 'perfeed', 'Gx' or 'Gy'.
        @param bestbeams: True if the reference patterns used for error beam were 'best' as oppesed to 'emss' (default False)
        @param band: "l" or "u" (default "l")
    """
    global USE_BESTBEAMS
    USE_BESTBEAMS = bestbeams
    
    if (band == "l"):
        params = {"clipextent":6,"bandwidth":4,"f_centre":1712e6*3/4.,"freqs":[900,1100,1350,1500,1670], "modelfolder":'./Predictions/BeamPatterns/L-band/'}
    elif (band == "u"):
        params = {"clipextent":9,"bandwidth":4,"f_centre":1088e6*3/4.,"freqs":[615,715,815,915,1015],#[580,700,840,900,1015],
                  "modelfolder":'./Predictions/BeamPatterns/UHF-band/'}
    
    bmresults = beamshape_load(filename, "ignore comment", scanantenna, cycle, ignoreantennas, pointing, fitdBlevel=-18, **params)
    fn,freqs,GxGyEB,hv_xyz = beamshape_report(*bmresults, band=band, savefigs=[1,3,4,7,99],
                                              fntag="_%s%s"%(pointing,'_bb' if bestbeams else ''))
    distill_stats(*bmresults, tag=tag)
    return bmresults, fn,freqs,GxGyEB,hv_xyz


if __name__=="__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] filename (optional)")
    parser.add_option("-r","--rows",type="string",default="",help="RTS registry row numbers, examples '3', '[3,4,5]', 'range(3,7)'")
    parser.add_option("--rts",action="store_true",help="Process filename as an RTS file (details not from registry)")
    parser.add_option("--band",type="string",default="l",help="example 'l' or 'u'")
    parser.add_option("-s","--scanants",type="string",default="",help="example 'm001,m002'")
    parser.add_option("-c","--cycles",type="string",default="",help="examples '0', '[1,2]', 'range(3)'")
    parser.add_option("-p","--pointing",type="string",default="I,perfeed",help="example 'I' or 'I,perfeed' or 'Gx'")
    parser.add_option("-i","--ignoreants",type="string",default="",help="example 'm058,m063'")
    parser.add_option("-b","--bestbeams",default=False,action="store_true",help="Use this to use 'best measured' instead of 'theoretical predicted'")
    opts,args = parser.parse_args()
    
    if (opts.rows):
        process_RTSregistry(opts.rows, opts.cycles, opts.pointing, opts.bestbeams, opts.band)
    elif (opts.rts):
        process(args[0], opts.scanants, opts.ignoreants, opts.cycles, opts.pointing, opts.bestbeams, opts.band)
    else:
        process_AR1(args[0], opts.scanants, opts.cycles, opts.pointing, opts.ignoreants, opts.bestbeams, opts.band)
